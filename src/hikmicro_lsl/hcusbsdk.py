"""Python ctypes wrapper for Hikvision / HIKMICRO HCUSBSDK Linux library.

Loads libHCUSBSDK.so from vendor/hcusbsdk (or system path) and wraps:
- USB_Init / USB_Cleanup
- USB_GetDeviceCount / USB_EnumDevices
- USB_Login / USB_Logout
- USB_StartStreamCallback / USB_StopChannel
- USB_GetDeviceConfig / USB_SetDeviceConfig
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import time
from pathlib import Path
from typing import Callable, Optional

# Constants
MAX_PATH = 260
DEVICE_INFO_SIZE = 648
LOGIN_PARAM_SIZE = 372
STREAM_PARAM_SIZE = 152

# Stream types
STREAM_TYPE_YUYV_PREVIEW = 0x65
STREAM_TYPE_THERMAL_Y16 = 0x66
STREAM_TYPE_MIXED = 0x67
STREAM_TYPE_RAW_RADIOMETRIC = 0x68
STREAM_TYPE_P2P = 0x69


@contextlib.contextmanager
def _chdir_context(target_dir: Path):
    """Hikvision HCUSBSDK looks for companion .so files in getcwd()."""
    old_cwd = os.getcwd()
    try:
        os.chdir(target_dir)
        yield
    finally:
        os.chdir(old_cwd)


class USBDeviceInfo(ctypes.Structure):
    """Corresponds to tagUSB_DEVICE_INFO (648 bytes)."""
    _fields_ = [
        ("dwReserved", ctypes.c_uint32),         # 0x00
        ("dwIndex", ctypes.c_uint32),            # 0x04
        ("dwVID", ctypes.c_uint32),              # 0x08
        ("dwPID", ctypes.c_uint32),              # 0x0C
        ("szManufacturer", ctypes.c_char * 32),  # 0x10
        ("szDeviceName", ctypes.c_char * 32),    # 0x30
        ("szSerialNumber", ctypes.c_char * 48),  # 0x50
        ("byPad", ctypes.c_uint8 * 2),           # 0x80
        ("szDevicePath", ctypes.c_char * MAX_PATH), # 0x82
        ("byReserved", ctypes.c_uint8 * (DEVICE_INFO_SIZE - 0x82 - MAX_PATH)),
    ]

    def vid_hex(self) -> str:
        return f"0x{self.dwVID:04x}"

    def pid_hex(self) -> str:
        return f"0x{self.dwPID:04x}"

    def name(self) -> str:
        return self.szDeviceName.decode("utf-8", errors="ignore").strip("\x00")

    def serial(self) -> str:
        return self.szSerialNumber.decode("utf-8", errors="ignore").strip("\x00")

    def path(self) -> str:
        return self.szDevicePath.decode("utf-8", errors="ignore").strip("\x00")


class USBLoginParam(ctypes.Structure):
    """Login parameter struct (0x174 = 372 bytes)."""
    _fields_ = [
        ("dwSize", ctypes.c_uint32),            # 0x00: 0x174
        ("dwTimeout", ctypes.c_uint32),         # 0x04: 5000 (ms)
        ("byReserved0", ctypes.c_uint32),       # 0x08
        ("dwVID", ctypes.c_uint32),             # 0x0C
        ("dwPID", ctypes.c_uint32),             # 0x10
        ("szDeviceName", ctypes.c_char * 32),   # 0x14
        ("byPad", ctypes.c_uint8 * 16),         # 0x34
        ("szSerialNumber", ctypes.c_char * 48), # 0x44
        ("byReserved", ctypes.c_uint8 * (LOGIN_PARAM_SIZE - 0x44 - 48)),
    ]


class USBLoginDeviceInfo(ctypes.Structure):
    """Device info filled by USB_Login (0x164 = 356 bytes)."""
    _fields_ = [
        ("dwSize", ctypes.c_uint32),            # 0x00: 0x164
        ("byReserved", ctypes.c_uint8 * (0x164 - 4)),
    ]


class USBFrameInfo(ctypes.Structure):
    """Corresponds to tagUSB_FRAME_INFO. Verified by disassembly and live probe."""
    _fields_ = [
        ("dwTimestamp", ctypes.c_uint32),         # 0x00: timestamp ms
        ("dwFrameType", ctypes.c_uint32),         # 0x04: stream type (0x65, 0x67, etc.)
        ("dwWidth", ctypes.c_uint32),             # 0x08: width
        ("dwHeight", ctypes.c_uint32),            # 0x0c: height
        ("dwFramerate", ctypes.c_uint32),         # 0x10: fps
        ("dwReserved0", ctypes.c_uint32),         # 0x14
        ("dwReserved1", ctypes.c_uint32),         # 0x18
        ("dwFrameNum", ctypes.c_uint32),          # 0x1c: frame sequence counter
        ("pBuf", ctypes.POINTER(ctypes.c_uint8)), # 0x20: pointer to buffer
        ("dwBufSize", ctypes.c_uint32),           # 0x28: buffer size in bytes
        ("byReserved", ctypes.c_uint8 * 84),      # 0x2c: padding
    ]


# Callback prototype: void(int lRealHandle, tagUSB_FRAME_INFO *pFrameInfo, void *pUser)
StreamCallbackType = ctypes.CFUNCTYPE(
    None,
    ctypes.c_int,
    ctypes.POINTER(USBFrameInfo),
    ctypes.c_void_p,
)


class USBStreamParam(ctypes.Structure):
    """Stream request struct (0x98 = 152 bytes)."""
    _fields_ = [
        ("dwSize", ctypes.c_uint32),            # 0x00: 0x98
        ("dwStreamType", ctypes.c_uint32),      # 0x04: stream type
        ("fnCallback", StreamCallbackType),     # 0x08: callback pointer
        ("pUserData", ctypes.c_void_p),         # 0x10: user data pointer
        ("byReserved", ctypes.c_uint8 * (STREAM_PARAM_SIZE - 24)),
    ]


class SDKPathCfg(ctypes.Structure):
    """Configuration struct for USB_SetSDKLocalCfg."""
    _fields_ = [
        ("dwType", ctypes.c_uint32),
        ("szPath", ctypes.c_char * 260),
    ]


class InBufferDesc(ctypes.Structure):
    """Input buffer descriptor for USB_Get/SetDeviceConfig (0x50 bytes)."""
    _fields_ = [
        ("dwChannel", ctypes.c_uint32),         # 0x00
        ("dwReserved0", ctypes.c_uint32 * 3),   # 0x04
        ("pBuf", ctypes.c_void_p),              # 0x10
        ("dwBufSize", ctypes.c_uint32),         # 0x18
        ("byReserved", ctypes.c_uint8 * (0x50 - 28)),
    ]


class OutBufferDesc(ctypes.Structure):
    """Output buffer descriptor for USB_Get/SetDeviceConfig (0x48 bytes)."""
    _fields_ = [
        ("pBuf", ctypes.c_void_p),              # 0x00
        ("dwBufSize", ctypes.c_uint32),         # 0x08
        ("byReserved", ctypes.c_uint8 * (0x48 - 12)),
    ]


class HcUsbSdk:
    """Thin wrapper around libHCUSBSDK.so."""

    def __init__(self, lib_dir: Optional[Path | str] = None):
        if lib_dir is None:
            lib_dir = Path(__file__).resolve().parent.parent.parent / "vendor" / "hcusbsdk"
        self.lib_dir = Path(lib_dir)
        self._stream_refs: dict = {}
        self._snap_cap: int = 0
        self._jpeg_buf = None
        self._app_buf = None

        # Preload companion libraries in correct dependency order
        for helper in ("libusb-1.0.so", "libuvc.so", "libhpr.so"):
            p = self.lib_dir / helper
            if p.exists():
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)

        main_so = self.lib_dir / "libHCUSBSDK.so"
        if not main_so.exists():
            raise FileNotFoundError(f"libHCUSBSDK.so not found at {main_so}")

        self._lib = ctypes.CDLL(str(main_so), mode=ctypes.RTLD_GLOBAL)
        self._bind_functions()

        # Tell libHCUSBSDK where to find its companion shared objects
        if hasattr(self._lib, "USB_SetSDKLocalCfg"):
            lib_dir_bytes = str(self.lib_dir).encode("utf-8")
            cfg_usb = SDKPathCfg(dwType=1, szPath=lib_dir_bytes)
            self._lib.USB_SetSDKLocalCfg(0, ctypes.byref(cfg_usb))
            cfg_uvc = SDKPathCfg(dwType=2, szPath=lib_dir_bytes)
            self._lib.USB_SetSDKLocalCfg(0, ctypes.byref(cfg_uvc))

    def _bind_functions(self):
        # int USB_Init()
        self._lib.USB_Init.restype = ctypes.c_int
        self._lib.USB_Init.argtypes = []

        # int USB_Cleanup()
        self._lib.USB_Cleanup.restype = ctypes.c_int
        self._lib.USB_Cleanup.argtypes = []

        # int USB_GetDeviceCount()
        self._lib.USB_GetDeviceCount.restype = ctypes.c_int
        self._lib.USB_GetDeviceCount.argtypes = []

        # int USB_EnumDevices(int count, USBDeviceInfo *devices)
        self._lib.USB_EnumDevices.restype = ctypes.c_int
        self._lib.USB_EnumDevices.argtypes = [ctypes.c_int, ctypes.POINTER(USBDeviceInfo)]

        # int USB_Login(USBLoginParam *param, USBLoginDeviceInfo *info)
        self._lib.USB_Login.restype = ctypes.c_int
        self._lib.USB_Login.argtypes = [ctypes.POINTER(USBLoginParam), ctypes.POINTER(USBLoginDeviceInfo)]

        # int USB_Logout(int lUserID)
        self._lib.USB_Logout.restype = ctypes.c_int
        self._lib.USB_Logout.argtypes = [ctypes.c_int]

        # int USB_StartStreamCallback(int lUserID, USBStreamParam *param)
        self._lib.USB_StartStreamCallback.restype = ctypes.c_int
        self._lib.USB_StartStreamCallback.argtypes = [ctypes.c_int, ctypes.POINTER(USBStreamParam)]

        # int USB_StopChannel(int lUserID)
        self._lib.USB_StopChannel.restype = ctypes.c_int
        self._lib.USB_StopChannel.argtypes = [ctypes.c_int]

        # int USB_GetDeviceConfig(int lUserID, int dwCommand, void *lpInBuffer, void *lpOutBuffer)
        if hasattr(self._lib, "USB_GetDeviceConfig"):
            self._lib.USB_GetDeviceConfig.restype = ctypes.c_int
            self._lib.USB_GetDeviceConfig.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

        # int USB_SetDeviceConfig(int lUserID, int dwCommand, void *lpInBuffer, void *lpOutBuffer)
        if hasattr(self._lib, "USB_SetDeviceConfig"):
            self._lib.USB_SetDeviceConfig.restype = ctypes.c_int
            self._lib.USB_SetDeviceConfig.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]

        # int USB_GetLastError()
        self._lib.USB_GetLastError.restype = ctypes.c_uint32
        self._lib.USB_GetLastError.argtypes = []

        # int USB_SetLogToFile(int nLogLevel, const char *szLogDir, int bAutoDel)
        if hasattr(self._lib, "USB_SetLogToFile"):
            self._lib.USB_SetLogToFile.restype = ctypes.c_int
            self._lib.USB_SetLogToFile.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

        # int USB_SetSDKLocalCfg(int enumType, void *pCfg)
        if hasattr(self._lib, "USB_SetSDKLocalCfg"):
            self._lib.USB_SetSDKLocalCfg.restype = ctypes.c_int
            self._lib.USB_SetSDKLocalCfg.argtypes = [ctypes.c_int, ctypes.c_void_p]

    def init(self) -> bool:
        with _chdir_context(self.lib_dir):
            return self._lib.USB_Init() == 1

    def cleanup(self) -> bool:
        with _chdir_context(self.lib_dir):
            return self._lib.USB_Cleanup() == 1

    def set_log(self, level: int = 3, log_dir: str = "/tmp/hcusbsdk_log", auto_del: bool = True):
        os.makedirs(log_dir, exist_ok=True)
        if hasattr(self._lib, "USB_SetLogToFile"):
            self._lib.USB_SetLogToFile(level, log_dir.encode("utf-8"), 1 if auto_del else 0)

    def get_last_error(self) -> int:
        return self._lib.USB_GetLastError()

    def get_device_count(self) -> int:
        with _chdir_context(self.lib_dir):
            return self._lib.USB_GetDeviceCount()

    def enum_devices(self) -> list[USBDeviceInfo]:
        with _chdir_context(self.lib_dir):
            count = self.get_device_count()
            if count <= 0:
                return []
            arr_type = USBDeviceInfo * count
            arr = arr_type()
            res = self._lib.USB_EnumDevices(count, arr)
            if res != 0:
                return list(arr)
            return []

    def login(self, device_index: int = 0) -> int:
        with _chdir_context(self.lib_dir):
            devices = self.enum_devices()
            if not devices or device_index >= len(devices):
                raise IndexError(f"Device index {device_index} not available (found {len(devices)})")

            dev = devices[device_index]
            param = USBLoginParam()
            param.dwSize = LOGIN_PARAM_SIZE
            param.dwTimeout = 5000
            param.dwVID = dev.dwVID
            param.dwPID = dev.dwPID
            param.szDeviceName = dev.szDeviceName
            param.szSerialNumber = dev.szSerialNumber

            dev_info = USBLoginDeviceInfo()
            dev_info.dwSize = 0x164

            login_id = self._lib.USB_Login(ctypes.byref(param), ctypes.byref(dev_info))
            if login_id < 0:
                err = self.get_last_error()
                raise RuntimeError(f"USB_Login failed with return {login_id}, error code {err}")
            return login_id

    def get_video_param(self, login_id: int) -> bytearray:
        """Query current video param struct via USB_GET_VIDEO_PARAM (0xbbb)."""
        with _chdir_context(self.lib_dir):
            buf = bytearray(156)
            buf_c = (ctypes.c_uint8 * 156).from_buffer(buf)

            in_desc = InBufferDesc()
            out_desc = OutBufferDesc()
            out_desc.pBuf = ctypes.cast(ctypes.byref(buf_c), ctypes.c_void_p)
            out_desc.dwBufSize = 156

            res = self._lib.USB_GetDeviceConfig(login_id, 0xbbb, ctypes.byref(in_desc), ctypes.byref(out_desc))
            if res == 0:
                err = self.get_last_error()
                raise RuntimeError(f"USB_GetDeviceConfig(0xbbb) failed, error {err}")
            return buf

    def set_video_param(
        self,
        login_id: int,
        stream_type: int = STREAM_TYPE_MIXED,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
    ) -> bool:
        """Configure stream parameters via USB_SET_VIDEO_PARAM (0xbbc)."""
        with _chdir_context(self.lib_dir):
            # Select resolution matching format if not explicitly supplied
            if width is None or height is None:
                if stream_type in (STREAM_TYPE_MIXED, STREAM_TYPE_RAW_RADIOMETRIC):
                    width, height = 640, 256
                elif stream_type == STREAM_TYPE_THERMAL_Y16:
                    width, height = 256, 192
                else:  # STREAM_TYPE_YUYV_PREVIEW (0x65)
                    width, height = 640, 480

            # Query current param block
            try:
                buf = self.get_video_param(login_id)
            except Exception:
                buf = bytearray(156)

            # Update format, dimensions, framerate
            import struct
            struct.pack_into("<I", buf, 0, stream_type)
            struct.pack_into("<I", buf, 4, width)
            struct.pack_into("<I", buf, 8, height)
            struct.pack_into("<I", buf, 12, fps)

            buf_c = (ctypes.c_uint8 * len(buf)).from_buffer(buf)

            in_desc = InBufferDesc()
            in_desc.pBuf = ctypes.cast(ctypes.byref(buf_c), ctypes.c_void_p)
            in_desc.dwBufSize = len(buf)

            out_desc = OutBufferDesc()

            res = self._lib.USB_SetDeviceConfig(login_id, 0xbbc, ctypes.byref(in_desc), ctypes.byref(out_desc))
            if res == 0:
                err = self.get_last_error()
                raise RuntimeError(f"USB_SetDeviceConfig(0xbbc) failed, error {err}")
            return True

    def start_stream(
        self,
        login_id: int,
        callback: Callable,
        stream_type: int = STREAM_TYPE_MIXED,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: int = 30,
        video_format: Optional[int] = None,
    ) -> USBStreamParam:
        """Sets video params and starts the stream callback."""
        with _chdir_context(self.lib_dir):
            # Configure video param first to prevent error 82 (call disorder)
            # The vendor recorder configures video as 0x65 (640x480) and then
            # starts stream type 0x69 (P2P); video_format decouples the two.
            fmt = stream_type if video_format is None else video_format
            self.set_video_param(login_id, stream_type=fmt, width=width, height=height, fps=fps)
            time.sleep(0.5)

            cb_func = StreamCallbackType(callback)
            param = USBStreamParam()
            param.dwSize = STREAM_PARAM_SIZE
            param.dwStreamType = stream_type
            param.fnCallback = cb_func
            param.pUserData = None

            res = self._lib.USB_StartStreamCallback(login_id, ctypes.byref(param))
            if res < 0:
                err = self.get_last_error()
                raise RuntimeError(f"USB_StartStreamCallback failed with {res}, error {err}")

            # Keep references alive for the lifetime of the stream: the SDK
            # calls cb_func from its own thread until USB_StopChannel.
            param._cb_ref = cb_func
            self._stream_refs[login_id] = (param, cb_func)
            return param

    def stop_stream(self, login_id: int):
        with _chdir_context(self.lib_dir):
            self._lib.USB_StopChannel(login_id)
        self._stream_refs.pop(login_id, None)

    def logout(self, login_id: int):
        with _chdir_context(self.lib_dir):
            self._lib.USB_Logout(login_id)

    def get_jpeg_with_appenddata(self, login_id: int, capacity: int = 0x100000) -> tuple[bytes, bytes, int]:
        """Snapshot via USB_GET_JPEGPIC_WITH_APPENDDATA (cmd 2046 / 0x7fe).

        Layout recovered from ``UsbDemo::GetJpegpicWithAppenddata`` and
        ``CUsbProtocolUVCBase::ParseRecvData<USB_JPEGPIC_WITH_APPENDDATA>``:
        the output struct (0x50 bytes) carries:
        - +0x00: dwSize (uint32)
        - +0x04: dwJpegPicLen (uint32, in: capacity, out: actual length)
        - +0x08: dwJpegPicWidth (uint32)
        - +0x0c: dwJpegPicHeight (uint32)
        - +0x10: dwP2pDataLen (uint32, in: capacity, out: actual byte length, e.g. 196608)
        - +0x14: byP2pDataType (uint8, 2=int16/uint16, 4=float32 degC)
        - +0x15: byScaledType (uint8)
        - +0x18: dwP2pDataWidth (uint32, e.g. 256)
        - +0x1c: dwP2pDataHeight (uint32, e.g. 192)
        - +0x20: pJpegPic (uint64 pointer)
        - +0x28: pP2pData (uint64 pointer)

        Returns ``(jpeg, append_data, first_u32)``.
        """
        import struct

        if self._jpeg_buf is None or self._app_buf is None or self._snap_cap < capacity:
            self._jpeg_buf = (ctypes.c_uint8 * capacity)()
            self._app_buf = (ctypes.c_uint8 * capacity)()
            self._snap_cap = capacity

        jpeg_buf = self._jpeg_buf
        app_buf = self._app_buf
        out_struct = (ctypes.c_uint8 * 0x50)()
        struct.pack_into("<I", out_struct, 4, capacity)
        struct.pack_into("<I", out_struct, 0x10, capacity)
        struct.pack_into("<Q", out_struct, 0x20, ctypes.addressof(jpeg_buf))
        struct.pack_into("<Q", out_struct, 0x28, ctypes.addressof(app_buf))

        cond = (ctypes.c_uint8 * 12)()
        cond[4] = 1

        in_desc = (ctypes.c_uint8 * 0x50)()
        struct.pack_into("<Q", in_desc, 0, ctypes.addressof(cond))
        struct.pack_into("<I", in_desc, 8, 12)
        out_desc = (ctypes.c_uint8 * 0x48)()
        struct.pack_into("<Q", out_desc, 0, ctypes.addressof(out_struct))
        struct.pack_into("<I", out_desc, 8, 0x50)

        with _chdir_context(self.lib_dir):
            res = self._lib.USB_GetDeviceConfig(
                login_id, 0x7FE, ctypes.addressof(in_desc), ctypes.addressof(out_desc)
            )
        if res == 0:
            raise RuntimeError(f"USB_GetDeviceConfig(0x7fe) failed, error {self.get_last_error()}")
        first = struct.unpack_from("<I", out_struct, 0)[0]
        jpeg_len = min(struct.unpack_from("<I", out_struct, 4)[0], capacity)
        app_len = min(struct.unpack_from("<I", out_struct, 0x10)[0], capacity)
        jpeg_bytes = ctypes.string_at(ctypes.addressof(jpeg_buf), jpeg_len) if jpeg_len else b""
        app_bytes = ctypes.string_at(ctypes.addressof(app_buf), app_len) if app_len else b""
        return jpeg_bytes, app_bytes, first
