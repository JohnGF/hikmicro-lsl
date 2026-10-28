#!/usr/bin/env python3
"""HIKMICRO Pocket 2 — HCUSBSDK command-channel trial harness (Linux).

Reconstructed from HCUSBSDK.dll (see scratchpad/HCUSBSDK_findings.md):
  * command channel = UVC extension unit 0x11, selector 1, 512-byte buffer
  * SET_CUR a 512-byte command packet, then GET_CUR the 512-byte response
  * wire packet = ~11-byte header (command id + direction + length) + payload
  * command ids: USB_SYSTEM_DEVICE_INFO=0x7DB (safe GET), and for starting the
    radiometric stream, USB_CTRL_THERMAL_STREAM_PARAM 0x83E/0x83F (32-byte struct)

The exact header byte layout is the one piece RE left fuzzy, so this sweeps a few
candidate layouts against the SAFE device-info GET and shows which yields a
plausible (non-zero / ASCII-bearing) response. Nothing is written to the camera
except the command SET_CURs, which are reversible by power-cycling.

Run from the repo so `hikmicro_lsl` imports:
    uv run python scratch_hik_cmd.py --device /dev/video2
Add --hold-stream to keep the MJPG preview streaming during commands (the SDK
opens the stream before sending commands; this may be required).
Add --start-stream to attempt the Y16 stream-start (SET) after the GET probe.
"""
from __future__ import annotations

import argparse
import ctypes
import mmap
import os
import select
import struct
import time

from hikmicro_lsl.uvc_source import (
    UvcXuControlQuery,
    UVCIOC_CTRL_QUERY,
    V4L2Buffer,
    V4L2Format,
    V4L2RequestBuffers,
    V4L2_BUF_TYPE_VIDEO_CAPTURE,
    V4L2_MEMORY_MMAP,
    VIDIOC_QBUF,
    VIDIOC_QUERYBUF,
    VIDIOC_REQBUFS,
    VIDIOC_S_FMT,
    VIDIOC_STREAMON,
    VIDIOC_STREAMOFF,
    _fourcc,
    _ioctl,
)

UNIT = 0x11
SELECTOR = 1
XLEN = 512

# UVC request codes
SET_CUR, GET_CUR, GET_LEN, GET_INFO = 0x01, 0x81, 0x85, 0x86

# command ids (get=even, set=odd)
CMD_DEVICE_INFO = 0x7DB          # safe read
CMD_CTRL_THERMAL_STREAM_GET = 0x83E
CMD_CTRL_THERMAL_STREAM_SET = 0x83F
CMD_THERMAL_STREAM_YUV_RES_GET = 0x853


# ---- raw XU access -------------------------------------------------------
def xu(fd, query, data_or_len):
    if isinstance(data_or_len, int):
        size = data_or_len
        buf = (ctypes.c_uint8 * max(size, 1))()
    else:
        size = len(data_or_len)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(data_or_len)
    req = UvcXuControlQuery(unit=UNIT, selector=SELECTOR, query=query,
                            size=size, data=buf)
    _ioctl(fd, UVCIOC_CTRL_QUERY, req)
    return bytes(buf)


def set_cur(fd, data):
    data = (bytes(data) + b"\x00" * XLEN)[:XLEN]
    return xu(fd, SET_CUR, data)


def get_cur(fd):
    return xu(fd, GET_CUR, XLEN)


# ---- candidate header layouts (the fuzzy bit) ----------------------------
def build_packets(cmd, direction, payload=b""):
    """Yield (name, 512-byte packet) candidates for a command.

    direction: 0 = GET/read, 1 = SET/write (byte-9 flag per RE).
    """
    plen = len(payload)
    out = []
    # A: cmd u16 LE at [0], then payload at [2]
    out.append(("A:u16@0", struct.pack("<H", cmd) + payload))
    # B: cmd u32 LE at [0], payload at [4]
    out.append(("B:u32@0", struct.pack("<I", cmd) + payload))
    # C: 11-byte header = [type u32=0][cmd u32][dir][len u16]; payload@11  (mirrors cmd-obj: type@8,cmd@0xc)
    out.append(("C:type0+cmd+dir+len",
                struct.pack("<IIBH", 0, cmd, direction, plen) + payload))
    # D: [cmd u32][dir u8][len u16][pad] ; payload@11
    out.append(("D:cmd+dir+len+3pad",
                struct.pack("<IBH", cmd, direction, plen) + b"\x00" * 4 + payload))
    # E: cmd big-endian u16 at [0]
    out.append(("E:be16@0", struct.pack(">H", cmd) + payload))
    # F: [cmd u16][dir u8][len u16][6 pad] -> 11-byte header, payload@11
    out.append(("F:u16+dir+len+6pad",
                struct.pack("<HBH", cmd, direction, plen) + b"\x00" * 6 + payload))
    return out


def dump(tag, b):
    nz = b.rstrip(b"\x00")
    printable = "".join(chr(c) if 32 <= c < 127 else "." for c in b[:64])
    print(f"    [{tag}] nonzero={len(nz):3}B  first64hex={b[:32].hex(' ')}")
    print(f"             ascii={printable!r}")


# ---- optional: hold a working video stream open --------------------------
class PreviewHold:
    def __init__(self, device):
        self.device = device
        self.fd = None
        self.maps = []

    def __enter__(self):
        fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        fmt = V4L2Format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        fmt.set_pix(640, 480, _fourcc("MJPG"))
        _ioctl(fd, VIDIOC_S_FMT, fmt)
        req = V4L2RequestBuffers()
        req.count, req.type, req.memory = 4, V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP
        _ioctl(fd, VIDIOC_REQBUFS, req)
        for i in range(req.count):
            qb = V4L2Buffer()
            qb.type, qb.memory, qb.index = V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, i
            _ioctl(fd, VIDIOC_QUERYBUF, qb)
            self.maps.append(mmap.mmap(fd, qb.length, mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE))
            qb2 = V4L2Buffer()
            qb2.type, qb2.memory, qb2.index = V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP, i
            _ioctl(fd, VIDIOC_QBUF, qb2)
        _ioctl(fd, VIDIOC_STREAMON, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
        # let a few frames flow so the device is "previewing"
        for _ in range(10):
            select.select([fd], [], [], 0.3)
        self.fd = fd
        print(f"[preview] MJPG 640x480 streaming on {self.device}")
        return self

    def __exit__(self, *a):
        if self.fd is not None:
            try:
                _ioctl(self.fd, VIDIOC_STREAMOFF, ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE))
            except OSError:
                pass
            for m in self.maps:
                m.close()
            os.close(self.fd)


def set_len_then(fd, pkt):
    """RE-derived two-step SET: first a 0x10-byte 'set length' packet, then data."""
    length_pkt = struct.pack("<I", len(pkt.rstrip(b"\x00")) or len(pkt)) + b"\x00" * 12
    set_cur(fd, length_pkt)
    time.sleep(0.02)
    set_cur(fd, pkt)


def probe_device_info(fd, two_step=False, stop_on_timeout=True):
    """Sweep header layouts for a SAFE device-info GET; show responses.

    Stops after the first SET_CUR timeout to avoid wedging the device.
    """
    print(f"\n== device-info GET sweep (cmd 0x{CMD_DEVICE_INFO:03x}), "
          f"two_step={two_step} ==")
    try:
        dump("baseline GET_CUR", get_cur(fd))
    except OSError as e:
        print(f"    baseline GET_CUR err: {e}")
    for name, pkt in build_packets(CMD_DEVICE_INFO, direction=0):
        try:
            if two_step:
                set_len_then(fd, pkt)
            else:
                set_cur(fd, pkt)
        except OSError as e:
            print(f"    [{name}] SET_CUR err: {e}")
            if stop_on_timeout and getattr(e, "errno", None) in (110,):
                print("    -> stopping sweep (timeout); device may need a replug.")
                return
            continue
        time.sleep(0.05)
        try:
            resp = get_cur(fd)
        except OSError as e:
            print(f"    [{name}] GET_CUR err: {e}")
            continue
        dump(name, resp)
        if resp.rstrip(b"\x00"):
            print(f"    -> [{name}] returned data! likely correct framing.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="/dev/video2")
    ap.add_argument("--hold-stream", action="store_true",
                    help="keep MJPG preview streaming during commands")
    ap.add_argument("--start-stream", action="store_true",
                    help="(after probe) attempt Y16 stream-start SET — writes to camera")
    ap.add_argument("--two-step", action="store_true",
                    help="use the RE-derived 2-step SET (set-length packet, then data)")
    args = ap.parse_args()

    # Open the preview stream FIRST (SDK opens video before sending commands),
    # then do all XU ops while it streams.
    hold = PreviewHold(args.device) if args.hold_stream else None
    if hold:
        hold.__enter__()
    try:
        fd = os.open(args.device, os.O_RDWR)
        try:
            try:
                info = xu(fd, GET_INFO, 1)[0]
                ln = int.from_bytes(xu(fd, GET_LEN, 2), "little")
                print(f"XU unit 0x{UNIT:02x} sel {SELECTOR}: info=0x{info:02x} len={ln}")
            except OSError as e:
                print(f"XU GET_INFO err on {args.device}: {e} "
                      "(device may be wedged from prior timeouts — try a replug)")
            probe_device_info(fd, two_step=args.two_step)
            if args.start_stream:
                print("\n== (experimental) stream-start not yet parameterized — "
                      "fill CTRL_THERMAL_STREAM_PARAM once the header layout is known ==")
        finally:
            os.close(fd)
    finally:
        if hold:
            hold.__exit__()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
