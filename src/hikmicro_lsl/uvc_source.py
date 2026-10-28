"""Live frame capture from the HIKMICRO Pocket 2 / XTherm-family UVC device.

The camera enumerates as a plain UVC video node (``/dev/videoN``) and streams
256x196 little-endian uint16. Raw mode (vendor unit command 0x8004) must be
enabled on the device before frames carry radiometric data; without it the
frames are ordinary video and ``parse_meta`` returns ``None``.

The V4L2 path is implemented directly with ``ctypes`` (mmap + ``VIDIOC_*``)
rather than the ``v4l2py`` package: v4l2py is unmaintained, emits a deprecation
warning on import, and exposes no ``UVCIOC_CTRL_QUERY`` hook, which the vendor
commands require. No root privileges are needed: the video node is normally
group-accessible (``video`` group).

Linux only. Import is safe on other platforms; ``open()`` raises.
"""

from __future__ import annotations

import ctypes
import fcntl
import mmap
import os
import struct
from dataclasses import dataclass

import numpy as np

from .radiometry import FRAME_HEIGHT, WIDTH

# --- V4L2 numeric constants (linux/videodev2.h) -----------------------------
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_FIELD_NONE = 1
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_BUF_FLAG_ERROR = 0x00000040

# --- USB video class request codes (linux/usb/video.h A.8) ------------------
UVC_GET_INFO = 0x86
UVC_SET_CUR = 0x01

# --- Vendor extension unit (HT203U-Thermal/Xtherm.kt lines 37-40) -----------
CMD_CALIBRATE = 0x8000  # run shutter calibration
CMD_RAW_MODE = 0x8004  # switch the stream to raw 16-bit
CMD_SAVE = 0x80FF  # persist settings / save image
CMD_RANGE_NORMAL = 0x8020  # -20..120 C
CMD_RANGE_HIGH = 0x8021  # -20..450 C

# Vendor commands are two-byte little-endian values written to selector 0x01
# of extension unit 0x02 (the layout used by the HT203U Kotlin app). Both are
# overridable from the CLI for units that enumerate differently.
DEFAULT_XU_UNIT = 0x02
DEFAULT_XU_SELECTOR = 0x01


def _fourcc(code: str) -> int:
    if len(code) != 4:
        raise ValueError(f"fourcc must be 4 characters, got {code!r}")
    return struct.unpack("<I", code.encode("ascii"))[0]


def _ioc(direction: int, type_: str, nr: int, size: int) -> int:
    """Compute an ``_IOC`` ioctl number for this architecture."""
    nrbits, typebits, sizebits = 8, 8, 14
    nrshift = 0
    typeshift = nrshift + nrbits
    sizeshift = typeshift + typebits
    dirshift = sizeshift + sizebits
    return (
        (direction << dirshift)
        | (ord(type_) << typeshift)
        | (size << sizeshift)
        | (nr << nrshift)
    )


_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _IOWR(type_: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ | _IOC_WRITE, type_, nr, size)


def _IOW(type_: str, nr: int, size: int) -> int:
    return _ioc(_IOC_WRITE, type_, nr, size)


def _IOR(type_: str, nr: int, size: int) -> int:
    return _ioc(_IOC_READ, type_, nr, size)


# --- ctypes structs ---------------------------------------------------------
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


class V4L2Format(ctypes.Structure):
    """``struct v4l2_format``: 8 bytes of type, then a 200-byte union (the
    ``pix`` member, ``struct v4l2_pix_format``, is the largest we need).
    """

    _fields_ = [
        ("type", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("_raw", ctypes.c_uint8 * 200),
    ]

    # byte offsets inside the union for struct v4l2_pix_format
    _PIX_WIDTH, _PIX_HEIGHT, _PIX_FORMAT, _PIX_FIELD = 0, 4, 8, 12
    _PIX_BPL, _PIX_SIZEIMAGE = 16, 20

    def _u32(self, off: int) -> int:
        return ctypes.c_uint32.from_buffer(self._raw, off).value

    def _set_u32(self, off: int, value: int) -> None:
        ctypes.c_uint32.from_buffer(self._raw, off).value = value

    @property
    def pix_width(self) -> int:
        return int(self._u32(self._PIX_WIDTH))

    @property
    def pix_height(self) -> int:
        return int(self._u32(self._PIX_HEIGHT))

    @property
    def pix_format(self) -> int:
        return int(self._u32(self._PIX_FORMAT))

    @property
    def pix_sizeimage(self) -> int:
        return int(self._u32(self._PIX_SIZEIMAGE))

    def set_pix(self, width: int, height: int, fourcc: int) -> None:
        self._set_u32(self._PIX_WIDTH, width)
        self._set_u32(self._PIX_HEIGHT, height)
        self._set_u32(self._PIX_FORMAT, fourcc)
        self._set_u32(self._PIX_FIELD, V4L2_FIELD_NONE)
        self._set_u32(self._PIX_BPL, 0)


class V4L2RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class V4L2Buffer(ctypes.Structure):
    """``struct v4l2_buffer``; ``timestamp`` is read as two u32 (sec, usec).

    On 64-bit ``struct timeval`` is still ``{long, long}`` so the offsets are
    layout-correct, and only sec/usec are consumed. The remaining fields keep
    the total size at 88 bytes so ``VIDIOC_QUERYBUF``/``DQBUF`` are safe.
    """

    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("ts_sec", ctypes.c_long),
        ("ts_usec", ctypes.c_long),
        ("_timecode", ctypes.c_uint8 * 16),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m_offset", ctypes.c_uint32),
        ("_m_pad", ctypes.c_uint32),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
        ("_tail_pad", ctypes.c_uint32),
    ]


class UvcXuControlQuery(ctypes.Structure):
    """``struct uvc_xu_control_query`` (linux/uvcvideo.h).

    sizeof() == 16 on 64-bit: three __u8, __u16 size (padded), __u8 *data.
    """

    _fields_ = [
        ("unit", ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query", ctypes.c_uint8),
        ("size", ctypes.c_uint16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


VIDIOC_QUERYCAP = _IOR("V", 0x00, ctypes.sizeof(V4L2Capability))
VIDIOC_S_FMT = _IOWR("V", 0x05, ctypes.sizeof(V4L2Format))
VIDIOC_REQBUFS = _IOWR("V", 0x08, ctypes.sizeof(V4L2RequestBuffers))
VIDIOC_QUERYBUF = _IOWR("V", 0x09, ctypes.sizeof(V4L2Buffer))
VIDIOC_QBUF = _IOWR("V", 0x0F, ctypes.sizeof(V4L2Buffer))
VIDIOC_DQBUF = _IOWR("V", 0x11, ctypes.sizeof(V4L2Buffer))
VIDIOC_STREAMON = _IOW("V", 0x12, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _IOW("V", 0x13, ctypes.sizeof(ctypes.c_int))

# UVCIOC_CTRL_QUERY is _IOWR('u', 0x21, struct uvc_xu_control_query); the size
# depends on sizeof(__u8 *) so it must be computed on the host, not hard-coded.
def _uvcioc_ctrl_query() -> int:
    return _IOWR("u", 0x21, ctypes.sizeof(UvcXuControlQuery))


UVCIOC_CTRL_QUERY = _uvcioc_ctrl_query()
UVCIOC_CTRL_ADD = _IOW("u", 0x02, 4)


def _ioctl(fd: int, request: int, arg) -> int:
    return fcntl.ioctl(fd, request, arg)


@dataclass
class DeviceInfo:
    path: str
    driver: str
    card: str
    bus_info: str


def probe_device(device: str) -> DeviceInfo | None:
    """Return capability info for ``device``, or ``None`` if it is not a capture node."""
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        cap = V4L2Capability()
        _ioctl(fd, VIDIOC_QUERYCAP, cap)
        caps = cap.device_caps or cap.capabilities
        if not caps & V4L2_CAP_VIDEO_CAPTURE:
            return None
        return DeviceInfo(
            path=device,
            driver=cap.driver.decode(errors="replace").rstrip("\x00"),
            card=cap.card.decode(errors="replace").rstrip("\x00"),
            bus_info=cap.bus_info.decode(errors="replace").rstrip("\x00"),
        )
    except OSError:
        return None
    finally:
        os.close(fd)


def list_video_devices() -> list[DeviceInfo]:
    """Enumerate ``/dev/video*`` capture nodes that the kernel will open."""
    out: list[DeviceInfo] = []
    root = "/dev"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for name in names:
        if not name.startswith("video"):
            continue
        info = probe_device(os.path.join(root, name))
        if info is not None:
            out.append(info)
    return out


def uvc_xu_query(
    fd: int,
    unit: int,
    selector: int,
    query: int,
    size: int,
) -> bytes:
    """Issue ``UVCIOC_CTRL_QUERY`` and return ``size`` bytes of response data."""
    buf = (ctypes.c_uint8 * max(size, 1))()
    req = UvcXuControlQuery(
        unit=unit, selector=selector, query=query, size=size, data=buf
    )
    try:
        _ioctl(fd, UVCIOC_CTRL_QUERY, req)
    except OSError as exc:
        raise NotImplementedError(
            f"UVCIOC_CTRL_QUERY failed on this host (unit=0x{unit:02x}, "
            f"selector=0x{selector:02x}, query=0x{query:02x}): {exc}. "
            "The uvcvideo extension-unit ioctls are either disabled at kernel "
            "build time (CONFIG_USB_GSPCA / UVC_CTRL not enabled), the device "
            "does not expose that extension unit, or the kernel refuses the "
            "query. Vendor commands (0x8000/0x8004/0x8020/0x8021) cannot be "
            "issued on this host."
        ) from exc
    return bytes(buf)


def _xu_set_cur(
    fd: int, unit: int, selector: int, value: int, query: int = UVC_SET_CUR
) -> None:
    """Write a 2-byte little-endian payload to a vendor XU control."""
    data = struct.pack("<H", int(value) & 0xFFFF)
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    req = UvcXuControlQuery(
        unit=unit, selector=selector, query=query, size=len(data), data=buf
    )
    _ioctl(fd, UVCIOC_CTRL_QUERY, req)


class UvcThermalSource:
    """V4L2 mmap capture source returning 196x256 uint16 frames.

    ``fourcc`` defaults to ``YUYV`` because that is the packed format the core
    normally advertises at 256x196/25fps. Each 2-byte YUYV word carries one
    uint16 sample from the sensor, so the payload is decoded as little-endian
    uint16 regardless of whether the device is in raw mode.
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        fourcc: str = "YUYV",
        width: int = WIDTH,
        height: int = FRAME_HEIGHT,
        buffers: int = 4,
        raw_mode: bool = True,
        raw_command: tuple[int, int] | None = (DEFAULT_XU_UNIT, DEFAULT_XU_SELECTOR),
    ) -> None:
        if width != WIDTH or height != FRAME_HEIGHT:
            raise ValueError(
                f"XTherm frames are fixed at {WIDTH}x{FRAME_HEIGHT}; got {width}x{height}"
            )
        self.device = device
        self.fourcc = fourcc
        self.width = width
        self.height = height
        self.n_buffers = buffers
        self.raw_mode = raw_mode
        # (unit, selector) used for the 0x8004 raw-mode command on open().
        self.raw_command = raw_command
        self._fd: int | None = None
        self._maps: list[mmap.mmap] = []
        self._streaming = False
        self._dropped = 0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> UvcThermalSource:
        if self._fd is not None:
            return self
        fd = os.open(self.device, os.O_RDWR | os.O_NONBLOCK)
        try:
            self._negotiate(fd)
            if self.raw_mode and self.raw_command is not None:
                self._request_raw_mode(fd)
            self._setup_buffers(fd)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def _negotiate(self, fd: int) -> None:
        cap = V4L2Capability()
        try:
            _ioctl(fd, VIDIOC_QUERYCAP, cap)
        except OSError as exc:
            raise OSError(
                f"{self.device}: VIDIOC_QUERYCAP failed ({exc}); this node is not a "
                "V4L2 capture device or is held by another process"
            ) from exc
        caps = cap.device_caps or cap.capabilities
        if not caps & V4L2_CAP_VIDEO_CAPTURE:
            raise OSError(f"{self.device} is not a video capture device")

        fmt = V4L2Format()
        fmt.set_pix(self.width, self.height, _fourcc(self.fourcc))
        try:
            _ioctl(fd, VIDIOC_S_FMT, fmt)
        except OSError as exc:
            raise OSError(
                f"{self.device}: VIDIOC_S_FMT({self.width}x{self.height} "
                f"{self.fourcc}) failed ({exc}). This device does not offer that "
                "format/size, so it is probably not an XTherm-family camera. "
                "Check `v4l2-ctl -d <node> --list-formats-ext`."
            ) from exc
        if (fmt.pix_width, fmt.pix_height) != (self.width, self.height):
            raise OSError(
                f"driver returned {fmt.pix_width}x{fmt.pix_height}, "
                f"requested {self.width}x{self.height}"
            )
        got_cc = struct.pack("<I", fmt.pix_format).decode("ascii", errors="replace")
        if got_cc != self.fourcc:
            raise OSError(f"driver returned fourcc {got_cc!r}, requested {self.fourcc!r}")

    def _request_raw_mode(self, fd: int) -> None:
        """Send 0x8004 before streaming; raise loudly if unsupported.

        Raw mode must be requested while the device is open but before
        ``STREAMON``. A host without extension-unit ioctls cannot supply
        radiometric frames, so this is an error rather than a silent fallback;
        pass ``raw_mode=False`` to stream plain video deliberately.
        """
        unit, selector = self.raw_command
        try:
            self._xu_set_cur(fd, unit, selector, CMD_RAW_MODE)
        except OSError as exc:
            raise NotImplementedError(
                f"could not enable raw mode (vendor command 0x{CMD_RAW_MODE:04x} "
                f"via UVCIOC_CTRL_QUERY): {exc}. Without raw mode the camera "
                "streams plain video and no temperatures can be computed. "
                "Use raw_mode=False to stream video anyway."
            ) from exc

    def _setup_buffers(self, fd: int) -> None:
        req = V4L2RequestBuffers()
        req.count = self.n_buffers
        req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        req.memory = V4L2_MEMORY_MMAP
        _ioctl(fd, VIDIOC_REQBUFS, req)
        if req.count < 1:
            raise OSError("driver granted zero capture buffers")

        for index in range(req.count):
            buf = V4L2Buffer()
            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            buf.memory = V4L2_MEMORY_MMAP
            buf.index = index
            _ioctl(fd, VIDIOC_QUERYBUF, buf)
            mm = mmap.mmap(
                fd, buf.length, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
            )
            self._maps.append(mm)
            self._queue(fd, index)

        self._start_stream(fd)

    def _queue(self, fd: int, index: int) -> None:
        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        buf.index = index
        _ioctl(fd, VIDIOC_QBUF, buf)

    def _start_stream(self, fd: int) -> None:
        if not self._streaming:
            buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            _ioctl(fd, VIDIOC_STREAMON, buf_type)
            self._streaming = True

    def close(self) -> None:
        fd = self._fd
        if fd is None:
            return
        if self._streaming:
            try:
                buf_type = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                _ioctl(fd, VIDIOC_STREAMOFF, buf_type)
            except OSError:
                pass
            self._streaming = False
        for mm in self._maps:
            mm.close()
        self._maps.clear()
        os.close(fd)
        self._fd = None

    def __enter__(self) -> UvcThermalSource:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- capture -----------------------------------------------------------
    def read_frame(self) -> tuple[np.ndarray, float] | None:
        """Return ``(frame, timestamp)`` or ``None`` if this read timed out.

        ``frame`` is a 196x256 uint16 array. ``timestamp`` is the driver
        monotonic timestamp in seconds when available, else ``time.monotonic()``.
        """
        import select
        import time

        fd = self._fd
        if fd is None:
            raise RuntimeError("UvcThermalSource is not open")
        readable, _, _ = select.select([fd], [], [], 0.5)
        if not readable:
            return None

        buf = V4L2Buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        try:
            _ioctl(fd, VIDIOC_DQBUF, buf)
        except OSError:
            # EAGAIN from a non-blocking deque: report as timeout.
            return None

        want = self.width * self.height  # bytes per decoded frame
        frame: np.ndarray | None = None
        if buf.flags & V4L2_BUF_FLAG_ERROR:
            self._dropped += 1
        elif buf.bytesused >= want:
            payload = self._maps[buf.index][:want]
            frame = np.frombuffer(payload, dtype="<u2").reshape(self.height, self.width)
        else:
            self._dropped += 1

        self._queue(fd, buf.index)

        ts = float(buf.ts_sec) + float(buf.ts_usec) / 1e6
        if ts <= 0.0:
            ts = time.monotonic()
        if frame is None:
            return None
        return frame, ts

    # -- vendor hooks ------------------------------------------------------
    def send_vendor_command(
        self,
        unit: int,
        selector_id: int,
        value: int = 1,
        query: int = UVC_SET_CUR,
    ) -> bytes:
        """Write a vendor extension-unit control (Xtherm.kt lines 37-40).

        ``unit``/``selector_id`` address the extension unit and the vendor
        command selector; ``value`` is the two-byte little-endian payload sent
        with ``UVC_SET_CUR`` (1 = enable for 0x8004/0x8020/0x8021 toggles).
        For the Pocket 2 the vendor commands are the selector ids themselves
        (unit 0x02, selector 0x01, payload = command), so a ready-made helper
        ``set_mode(command)`` is provided below.

        Raises ``NotImplementedError`` carrying the ioctl errno when the host
        kernel or device does not support ``UVCIOC_CTRL_QUERY``. It never
        silently no-ops.
        """
        fd = self._fd
        if fd is None:
            raise RuntimeError("UvcThermalSource is not open")
        try:
            self._xu_set_cur(fd, unit, selector_id, value, query=query)
        except OSError as exc:
            raise NotImplementedError(
                f"UVCIOC_CTRL_QUERY failed (unit=0x{unit:02x}, "
                f"selector=0x{selector_id:02x}, value={value}): {exc}. "
                "This kernel or device does not support the uvcvideo "
                "extension-unit query ioctl, so raw mode, shutter calibration "
                "and range switching cannot be issued from userspace on this "
                "host."
            ) from exc
        return b""

    def set_mode(self, command: int) -> None:
        """Send one of the vendor commands (0x8000/0x8004/0x80FF/0x8020/0x8021)."""
        known = (CMD_CALIBRATE, CMD_RAW_MODE, CMD_SAVE, CMD_RANGE_NORMAL, CMD_RANGE_HIGH)
        if command not in known:
            raise ValueError(f"unknown vendor command 0x{command:04x}")
        self.send_vendor_command(DEFAULT_XU_UNIT, DEFAULT_XU_SELECTOR, command & 0xFFFF)
        if command == CMD_RAW_MODE:
            self.raw_mode = True



