#!/usr/bin/env -S uv run --quiet python
"""Probe a UVC thermal camera for a reachable raw/vendor (XU) mode.

Read-only exploratory tool. It never issues CMD_SAVE (0x80FF) and never writes
to persistent calibration. Every ioctl is individually guarded: a control that
does not answer produces a printed errno, not a traceback.

Steps
-----
1. Enumerate /dev/videoN nodes and their formats (ioctl ENUM_FMT for fourcc,
   v4l2-ctl only to list discrete frame sizes).
2. Walk extension units 1..64 x selectors 1..64 and report UVC_GET_LEN and
   UVC_GET_INFO, so we learn which (unit, selector) pairs exist at all.
3. For every responding pair, read UVC_GET_CUR (first 16 bytes) plus GET_MIN
   and GET_MAX. uvcvideo caches XU controls lazily, so a control that returns
   EIO on a 1-byte probe is retried with the cached 2-byte GET_LEN value; the
   length that actually answered is the one reported.
4. Attempt CMD_RAW_MODE (0x8004) on every responding pair, then on the legacy
   default (unit 2, selector 1), printing exactly which pair took the write.
5. If any write was accepted, re-enumerate formats, force 256x196 YUYV and
   dequeue 3 buffers; report bytesused, the first 64 bytes and the min/max/mean
   of the samples decoded as little-endian uint16.
6. Print the media-controller topology that references the device.

Usage::

    uv run scripts/probe_raw_mode.py --device /dev/video2

Kernel note
-----------
uvcioc_ctl_query() looks controls up in ``uvc_ctrl_get`` which only finds
controls already registered. ``__uvc_find_control`` parses the XU descriptor
on the fly when the control is not cached, so a unit that exists answers
UVC_GET_INFO at any size. An *existing* control whose cached length differs
from the requested size fails with ENOBUFS (>= 2 bytes) or EIO (1 byte).
Both errno values are therefore reported distinctly rather than collapsed
into "no such control".
"""

from __future__ import annotations

import argparse
import array
import ctypes
import errno
import fcntl
import glob
import os
import statistics
import struct
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from hikmicro_lsl.radiometry import (
    CMD_RAW_MODE,
    FRAME_HEIGHT,
    HEIGHT,
    WIDTH,
)
from hikmicro_lsl.uvc_source import (
    DEFAULT_XU_SELECTOR,
    DEFAULT_XU_UNIT,
    UvcXuControlQuery,
    probe_device,
    uvc_xu_query,
)

# --- UVC request codes (linux/usb/video.h, A.8) -----------------------------
UVC_GET_CUR = 0x81
UVC_GET_MIN = 0x82
UVC_GET_MAX = 0x83
UVC_GET_LEN = 0x85
UVC_GET_INFO = 0x86
UVC_SET_CUR = 0x01

# uvc_xu_control_flags advertised by UVC_GET_INFO
XU_FLAG_DISABLED = 0x01
XU_FLAG_AUTOUPDATE = 0x04

# --- V4L2 --------------------------------------------------------------------
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_META_CAPTURE = 0x00800000

_IOC_READ, _IOC_WRITE = 2, 1


def _ioc(direction: int, type_: str, nr: int, size: int) -> int:
    # Matches asm-generic/ioctl.h _IOC(): nr:8 | type:8 | size:14 | dir:2.
    nrbits, typebits, sizebits, dirbits = 8, 8, 14, 2
    return (
        (dirbits << (nrbits + typebits + sizebits))
        | (size << (nrbits + typebits))
        | ((ord(type_) if isinstance(type_, str) else int(type_)) << nrbits)
        | nr
    )


def _IOWR(type_: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, type_, nr, size)


def _ioc_as_uvc_source(direction: int, type_: str, nr: int, size: int) -> int:
    """Replicate uvc_source._ioc bit-for-bit, including its layout bug.

    ``uvc_source._ioc`` places the size field at bit 8 instead of bit 14, so
    ``UVCIOC_CTRL_QUERY`` is 0xc0107521 where the kernel expects 0x80107521
    for ``sizeof(struct uvc_xu_control_query) == 16``. The kernel then fails
    the request with ENOTTY before uvcvideo is ever called. The script keeps
    this number around only to demonstrate the mismatch; every real ioctl in
    this script uses the correct ``UVCIOC_CTRL_QUERY``.
    """
    nrshift, typebits, sizebits = 0, 8, 14
    typeshift = nrshift + 8
    sizeshift = typeshift + typebits
    dirshift = sizeshift + sizebits
    return (
        (direction << dirshift) | (ord(type_) << typeshift) | (size << sizeshift) | (nr << nrshift)
    )


class V4L2FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


class V4L2Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]


VIDIOC_QUERYCAP = _IOWR("V", 0x00, ctypes.sizeof(V4L2Capability))
VIDIOC_ENUM_FMT = _IOWR("V", 0x02, ctypes.sizeof(V4L2FmtDesc))
UVCIOC_CTRL_QUERY = _IOWR("u", 0x21, ctypes.sizeof(UvcXuControlQuery))
UVCIOC_CTRL_QUERY_LIB = _ioc_as_uvc_source(
    _IOC_READ | _IOC_WRITE, "u", 0x21, ctypes.sizeof(UvcXuControlQuery)
)

# errno values that mean "this (unit, selector) does not exist at all"
ABSENT = {errno.ENOENT, errno.ENOTTY, errno.EINVAL}


def section(title: str) -> None:
    print(f"\n== {title} ==")


def errname(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, f"errno={exc.errno}")


def fourcc_str(code: int) -> str:
    raw = struct.pack("<I", code)
    return "".join(chr(b) if 32 <= b < 127 else "." for b in raw)


# ---------------------------------------------------------------------------
# raw UVCIOC_CTRL_QUERY: needed for GET_LEN, which must return the kernel's
# own notion of the size and may therefore answer with any errno.
# ---------------------------------------------------------------------------
def raw_query(fd: int, unit: int, sel: int, query: int, size: int) -> tuple[bytes | None, str]:
    buf = (ctypes.c_uint8 * max(size, 1))()
    req = UvcXuControlQuery(unit=unit, selector=sel, query=query, size=size, data=buf)
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, req)
    except OSError as exc:
        return None, errname(exc)
    return bytes(buf), "ok"


def query_sized(fd: int, unit: int, sel: int, query: int, size: int) -> tuple[bytes | None, str]:
    """uvc_xu_query with a size fallback, reporting the first errno seen."""
    first = "?"
    for attempt in (size, 2, 1, 4, 8):
        try:
            return uvc_xu_query(fd, unit, sel, query, attempt), "ok"
        except NotImplementedError as exc:
            cause = exc.__cause__
            code = errname(cause) if isinstance(cause, OSError) else "?"
            if first == "?":
                first = code
    return None, first


def query_get_len(fd: int, unit: int, sel: int) -> tuple[int | None, str]:
    for size in (1, 2):
        data, why = raw_query(fd, unit, sel, UVC_GET_LEN, size)
        if data is not None:
            return (data[0] if size == 1 else int.from_bytes(data[:2], "little")), "ok"
        if why in ABSENT:
            return None, why
    return None, why


# ---------------------------------------------------------------------------
# 1. nodes + formats
# ---------------------------------------------------------------------------
def video_nodes() -> list[str]:
    def key(path: str) -> int:
        try:
            return int(path[len("/dev/video") :])
        except ValueError:
            return 1 << 30

    return sorted(glob.glob("/dev/video*"), key=key)


def querycap(device: str) -> tuple[int, str, str] | None:
    """VIDIOC_QUERYCAP via the library's ctypes capability struct."""
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        cap = V4L2Capability()
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap)
    except OSError:
        return None
    finally:
        os.close(fd)
    return (
        int(cap.device_caps or cap.capabilities),
        cap.card.decode(errors="replace").rstrip("\x00"),
        cap.driver.decode(errors="replace").rstrip("\x00"),
    )


def enum_fourccs(device: str) -> list[str]:
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return []
    out: list[str] = []
    try:
        index = 0
        while True:
            desc = V4L2FmtDesc()
            desc.index = index
            desc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            try:
                fcntl.ioctl(fd, VIDIOC_ENUM_FMT, desc)
            except OSError:
                break
            out.append(fourcc_str(desc.pixelformat))
            index += 1
    finally:
        os.close(fd)
    return out


def list_formats_ext(device: str) -> str:
    try:
        return subprocess.run(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def print_nodes() -> None:
    section("video nodes and formats")
    for node in video_nodes():
        cap = querycap(node)
        if cap is None:
            print(f"{node}: QUERYCAP failed (not a V4L2 node?)")
            continue
        caps, card, driver = cap
        tags = []
        if caps & V4L2_CAP_VIDEO_CAPTURE:
            tags.append("capture")
        if caps & V4L2_CAP_META_CAPTURE:
            tags.append("metadata")
        print(f"{node}: card={card!r} driver={driver!r} caps=0x{caps:08x} [{'/'.join(tags)}]")
        fourccs = enum_fourccs(node) if caps & V4L2_CAP_VIDEO_CAPTURE else []
        if not fourccs:
            print("    no capture pixel formats (metadata-only node)")
            continue
        for line in list_formats_ext(node).splitlines():
            if "'" in line or line.strip().startswith("Size:"):
                print(f"    {line.strip()}")


# ---------------------------------------------------------------------------
# 2/3. XU descriptor walk
# ---------------------------------------------------------------------------
def xu_walk(device: str) -> list[tuple[int, int]]:
    section("XU descriptor walk: unit 1..64 x selector 1..64")
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        print(f"open failed: {errname(exc)}")
        return []
    found: list[tuple[int, int]] = []
    try:
        absent_units = 0
        for unit in range(1, 65):
            hits = 0
            absent_sels = 0
            absent_why = ""
            for sel in range(1, 65):
                length, why = query_get_len(fd, unit, sel)
                if length is None:
                    if why in ABSENT:
                        absent_sels += 1
                        absent_why = why
                    else:
                        print(f"  unit={unit:2d} sel={sel:2d} GET_LEN failed: {why}")
                    continue
                hits += 1
                info, info_why = query_sized(fd, unit, sel, UVC_GET_INFO, 1)
                if info is None:
                    flags = f"err({info_why})"
                else:
                    flags = f"0x{info[0]:02x}"
                    notes = []
                    if info[0] & XU_FLAG_DISABLED:
                        notes.append("disabled")
                    if info[0] & XU_FLAG_AUTOUPDATE:
                        notes.append("autoupdate")
                    if notes:
                        flags += f" ({', '.join(notes)})"
                print(f"  unit={unit:2d} sel={sel:2d} GET_LEN={length:3d} GET_INFO={flags}")
                found.append((unit, sel))
                print_ctrl(fd, unit, sel, length)
            if hits == 0 and absent_sels == 64:
                absent_units += 1
                if absent_units == 1:
                    print(f"  unit={unit:2d}: all 64 selectors absent ({absent_why})")
                if absent_units >= 16:
                    print(f"  ... units {unit + 1}..64 also absent, stopping walk")
                    break
            else:
                absent_units = 0
                if hits == 0:
                    print(f"  unit={unit:2d}: present but no selector answered GET_LEN")
    finally:
        os.close(fd)
    return found


def print_ctrl(fd: int, unit: int, sel: int, length: int) -> None:
    size = min(max(length, 1), 64)
    for label, req in (("CUR", UVC_GET_CUR), ("MIN", UVC_GET_MIN), ("MAX", UVC_GET_MAX)):
        data, why = query_sized(fd, unit, sel, req, size)
        if data is None:
            print(f"      GET_{label}: failed {why}")
            continue
        line = f"      GET_{label}: {data[:16].hex(' ')}"
        if len(data) >= 2:
            line += f"  (u16={int.from_bytes(data[:2], 'little'):#06x})"
        print(line)


# ---------------------------------------------------------------------------
# 4. raw-mode attempt
# ---------------------------------------------------------------------------
def set_cur(fd: int, unit: int, sel: int, payload: bytes) -> str | None:
    """SET_CUR with the correct ioctl number. Returns None on success."""
    buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    req = UvcXuControlQuery(unit=unit, selector=sel, query=UVC_SET_CUR, size=len(payload), data=buf)
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY, req)
    except OSError as exc:
        return errname(exc)
    return None


def set_cur_lib(fd: int, unit: int, sel: int, payload: bytes) -> str | None:
    """SET_CUR using uvc_source's (miscomputed) ioctl number, for contrast."""
    buf = (ctypes.c_uint8 * len(payload)).from_buffer_copy(payload)
    req = UvcXuControlQuery(unit=unit, selector=sel, query=UVC_SET_CUR, size=len(payload), data=buf)
    try:
        fcntl.ioctl(fd, UVCIOC_CTRL_QUERY_LIB, req)
    except OSError as exc:
        return errname(exc)
    return None


def attempt_raw_mode(device: str, pairs: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    section(f"CMD_RAW_MODE (0x{CMD_RAW_MODE:04x}) SET_CUR attempts")
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        print(f"open failed: {errname(exc)}")
        return []
    accepted: list[tuple[int, int]] = []
    payload = struct.pack("<H", CMD_RAW_MODE)
    try:
        for unit, sel in pairs:
            lib_why = set_cur_lib(fd, unit, sel, payload)
            why = set_cur(fd, unit, sel, payload)
            if why is not None:
                print(
                    f"  unit={unit} sel={sel} SET_CUR 0x8004: failed {why}"
                    f" (uvc_source ioctl 0x{UVCIOC_CTRL_QUERY_LIB:x}: {lib_why})"
                )
                continue
            print(f"  unit={unit} sel={sel} SET_CUR 0x8004: ACCEPTED")
            back, rwhy = query_sized(fd, unit, sel, UVC_GET_CUR, 2)
            if back is not None and len(back) >= 2:
                print(f"    readback GET_CUR u16={int.from_bytes(back[:2], 'little'):#06x}")
            else:
                print(f"    readback GET_CUR failed {rwhy}")
            accepted.append((unit, sel))
    finally:
        os.close(fd)
    return accepted


# ---------------------------------------------------------------------------
# 5. post-raw capture
# ---------------------------------------------------------------------------
def set_format(device: str, width: int, height: int) -> bool:
    section(f"VIDIOC_S_FMT {width}x{height} YUYV")
    try:
        res = subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                device,
                f"--set-fmt-video=width={width},height={height},pixelformat=YUYV",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  v4l2-ctl unavailable: {exc}")
        return False
    for line in res.stdout.splitlines():
        if line.strip():
            print(f"  {line.strip()}")
    if res.returncode != 0:
        print(f"  S_FMT rejected ({res.returncode}): {res.stderr.strip()}")
        return False
    return True


def stream_buffers(device: str, count: int) -> None:
    section(f"capture {count} buffers")
    try:
        res = subprocess.run(
            ["v4l2-ctl", "-d", device, f"--stream-mmap={count}", f"--stream-count={count}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  stream failed: {exc}")
        return
    for line in (res.stdout + res.stderr).splitlines():
        if any(k in line.lower() for k in ("bytesused", "buffer", "fps", "error")):
            print(f"  {line.strip()}")


def dump_one_frame(device: str, width: int, height: int) -> None:
    section("frame content")
    out = Path("/tmp/opencode/probe_raw_frame.bin")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run(
            [
                "v4l2-ctl",
                "-d",
                device,
                "--stream-mmap=3",
                "--stream-count=1",
                f"--stream-to={out}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  capture failed: {exc}")
        return
    if res.returncode != 0 or not out.exists():
        print(f"  capture failed rc={res.returncode}: {res.stderr.strip()[:200]}")
        return
    blob = out.read_bytes()
    print(f"  bytes={len(blob)}")
    print(f"  first64={blob[:64].hex(' ')}")
    if len(blob) >= 3 and blob[:3] == b"\xff\xd8\xff":
        print("  payload is JPEG (SOI marker) despite the YUYV negotiation")
        return
    words = array.array("H", blob[: (len(blob) // 2) * 2])
    if sys.byteorder != "little":
        words.byteswap()
    if not words:
        return
    print(f"  u16 samples={len(words)} (expected {width * height // 2} for {width}x{height} YUYV)")
    print(f"  u16 min={min(words)} max={max(words)} mean={statistics.fmean(words):.1f}")
    per_row = max(width // 2, 1)
    rows = len(words) // per_row
    row_max = [max(words[r * per_row : (r + 1) * per_row]) for r in range(rows)]
    print(f"  rows={rows} row-max head={row_max[:6]} tail={row_max[-4:]}")


# ---------------------------------------------------------------------------
# 6. media controller
# ---------------------------------------------------------------------------
def media_report(device: str) -> None:
    section("media-controller topology")
    sysdir = Path(f"/sys/class/video4linux/{Path(device).name}/device")
    try:
        target = os.path.basename(os.path.realpath(sysdir))
    except OSError:
        target = ""
    media = sorted(glob.glob("/dev/media*"))
    if not media:
        print("  no /dev/mediaN nodes")
        return
    for node in media:
        try:
            res = subprocess.run(
                ["media-ctl", "-d", node, "-p"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"  media-ctl unavailable: {exc}")
            return
        if target and target not in res.stdout:
            continue
        print(f"  --- {node} ---")
        for line in res.stdout.splitlines():
            print(f"  {line}")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Probe a UVC thermal camera for raw mode.")
    ap.add_argument("--device", default="/dev/video2", help="V4L2 video node")
    ap.add_argument("--width", type=int, default=WIDTH)
    ap.add_argument("--height", type=int, default=FRAME_HEIGHT)
    args = ap.parse_args()

    if not os.path.exists(args.device):
        print(f"{args.device} does not exist")
        return 2

    print(f"device={args.device} target grid={WIDTH}x{HEIGHT} frame={FRAME_HEIGHT}")
    print(
        f"UVCIOC_CTRL_QUERY: correct=0x{UVCIOC_CTRL_QUERY:x} uvc_source=0x{UVCIOC_CTRL_QUERY_LIB:x}"
    )
    info = probe_device(args.device)
    if info is not None:
        print(f"probe_device: card={info.card!r} driver={info.driver!r} bus={info.bus_info!r}")

    print_nodes()

    found = xu_walk(args.device)
    section("XU walk summary")
    if found:
        for unit, sel in found:
            print(f"  control present: unit={unit} sel={sel}")
    else:
        print("  no (unit, selector) answered UVC_GET_LEN")

    pairs: list[tuple[int, int]] = list(found)
    default_pair = (DEFAULT_XU_UNIT, DEFAULT_XU_SELECTOR)
    if default_pair not in pairs:
        pairs.append(default_pair)

    accepted = attempt_raw_mode(args.device, pairs)

    if accepted:
        if set_format(args.device, args.width, args.height):
            stream_buffers(args.device, 3)
            dump_one_frame(args.device, args.width, args.height)
    else:
        print("  raw mode not reached; skipping S_FMT/capture experiment")

    media_report(args.device)

    section("verdict")
    if accepted:
        for unit, sel in accepted:
            print(f"  CMD_RAW_MODE accepted on unit={unit} selector={sel}")
    else:
        print("  CMD_RAW_MODE not accepted on any probed (unit, selector)")
        print("  raw mode unreachable over the vendor XU as probed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
