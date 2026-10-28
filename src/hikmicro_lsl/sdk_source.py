"""Frame source backed by the vendor ``libHCUSBSDK.so`` (``--backend hcusbsdk``).

When a HIKMICRO Pocket 2 is connected in "USB Cast Screen" mode (``2bdf:017f``),
its UVC video interface only streams a plain MJPEG screen mirror without APP4
radiometric segments. Instead, HIKMICRO Analyzer and ``libHCUSBSDK.so`` fetch
calibrated temperature matrices over the UVC Extension Unit via command ``0x7fe``
(``USB_GET_JPEGPIC_WITH_APPENDDATA``), which returns a ``192 x 256`` little-endian
``float32`` degree-Celsius array (``196,608`` bytes) even if the camera screen is
showing a menu.

``SdkThermalSource.read_frame()`` polls ``get_jpeg_with_appenddata`` by default
and returns ``(temps_c, timestamp)`` where ``temps_c`` is a ``(192, 256)``
``float32`` array in degrees Celsius (or a ``uint16`` raw array if the device
returns raw/APP4 data).
"""

from __future__ import annotations

import ctypes
import queue
import time

import numpy as np

from .hcusbsdk import STREAM_TYPE_MIXED, HcUsbSdk
from .offline import UVC_FRAME_BYTES, _collect_radiometric
from .radiometry import FRAME_HEIGHT, HEIGHT, PIXELS, WIDTH, ZEROC

P2P_FLOAT32_BYTES = PIXELS * 4
P2P_UINT16_BYTES = PIXELS * 2


def decode_appenddata(append_data: bytes) -> np.ndarray | None:
    """Decode the P2P append-data buffer returned by ``0x7fe`` into a thermal frame.

    Returns:
    - ``(192, 256)`` ``float32`` array in degrees Celsius when ``append_data`` is
      ``196,608`` bytes (``byP2pDataType == 4`` float32) or ``98,304`` bytes
      (``byP2pDataType == 2`` centikelvin uint16).
    - ``(196, 256)`` ``uint16`` array when ``append_data`` is a ``100,352``-byte
      raw XTherm frame with metadata rows.
    - ``None`` if ``append_data`` does not match a known radiometric payload size.
    """
    n = len(append_data)
    if n == P2P_FLOAT32_BYTES:
        return np.frombuffer(append_data, dtype="<f4").reshape(HEIGHT, WIDTH).copy()
    if n == UVC_FRAME_BYTES:
        return (
            np.frombuffer(append_data, dtype="<u2")
            .reshape(FRAME_HEIGHT, WIDTH)
            .copy()
        )
    if n == P2P_UINT16_BYTES:
        u16 = np.frombuffer(append_data, dtype="<u2").reshape(HEIGHT, WIDTH)
        return (u16.astype(np.float32) * 0.01 - float(ZEROC)).astype(np.float32)
    return None


class SdkThermalSource:
    """Pull-style thermal frame source using ``libHCUSBSDK.so``."""

    def __init__(
        self,
        device_index: int = 0,
        stream_type: int = STREAM_TYPE_MIXED,
        queue_depth: int = 8,
        use_p2p: bool = True,
        sdk: HcUsbSdk | None = None,
    ) -> None:
        self.device_index = device_index
        self.stream_type = stream_type
        self.use_p2p = use_p2p
        self.height = FRAME_HEIGHT
        self.width = WIDTH
        self._sdk: HcUsbSdk | None = sdk
        self._owns_sdk = sdk is None
        self._login: int | None = None
        self._streaming = False
        self._queue: queue.Queue[tuple[bytes, float]] = queue.Queue(maxsize=queue_depth)
        self.frames_seen = 0
        self.frames_radiometric = 0
        self.frames_dropped = 0
        self.last_preview: bytes | None = None

    # -- SDK callback (runs on an SDK thread when use_p2p=False) -------------
    def _on_frame(self, handle, info_ptr, user) -> None:
        if not info_ptr:
            return
        info = info_ptr.contents
        if not info.pBuf or info.dwBufSize == 0:
            return
        payload = ctypes.string_at(info.pBuf, info.dwBufSize)
        ts = time.monotonic()
        try:
            self._queue.put_nowait((payload, ts))
        except queue.Full:
            self.frames_dropped += 1

    # -- lifecycle -----------------------------------------------------------
    def open(self) -> SdkThermalSource:
        sdk = self._sdk if self._sdk is not None else HcUsbSdk()
        if not sdk.init():
            raise RuntimeError("USB_Init failed")
        if sdk.get_device_count() <= 0:
            sdk.cleanup()
            raise RuntimeError(
                "no HIKMICRO device found via HCUSBSDK; set the camera to "
                "'USB Cast Screen' (2bdf:017f) and install udev/99-hikmicro.rules"
            )
        self._sdk = sdk
        self._login = sdk.login(self.device_index)
        if not self.use_p2p:
            sdk.start_stream(
                self._login, self._on_frame, stream_type=self.stream_type, fps=30
            )
            self._streaming = True
        return self

    def close(self) -> None:
        sdk, login = self._sdk, self._login
        self._sdk = self._login = None
        if sdk is None:
            return
        try:
            if login is not None:
                if self._streaming:
                    sdk.stop_stream(login)
                    self._streaming = False
                sdk.logout(login)
        finally:
            if self._owns_sdk:
                sdk.cleanup()

    def __enter__(self) -> SdkThermalSource:
        if self._login is None:
            self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- frame access --------------------------------------------------------
    def read_frame(self) -> tuple[np.ndarray, float] | None:
        if self.use_p2p and self._sdk is not None and self._login is not None:
            try:
                jpeg, append_data, _first = self._sdk.get_jpeg_with_appenddata(
                    self._login
                )
            except RuntimeError:
                return None
            ts = time.monotonic()
            self.frames_seen += 1
            if jpeg:
                self.last_preview = jpeg
            decoded = decode_appenddata(append_data)
            if decoded is not None:
                self.frames_radiometric += 1
                return decoded, ts
            if jpeg:
                blob = _collect_radiometric(jpeg)
                if blob is not None and len(blob) >= UVC_FRAME_BYTES:
                    self.frames_radiometric += 1
                    frame = np.frombuffer(blob[:UVC_FRAME_BYTES], dtype="<u2").reshape(
                        self.height, self.width
                    )
                    return frame.copy(), ts
            return None

        try:
            payload, ts = self._queue.get(timeout=0.5)
        except queue.Empty:
            return None
        self.frames_seen += 1
        self.last_preview = payload
        blob = _collect_radiometric(payload)
        if blob is None or len(blob) < UVC_FRAME_BYTES:
            return None
        self.frames_radiometric += 1
        frame = np.frombuffer(blob[:UVC_FRAME_BYTES], dtype="<u2").reshape(
            self.height, self.width
        )
        return frame.copy(), ts

    # -- compatibility with UvcThermalSource ---------------------------------
    def set_mode(self, command: int) -> None:
        raise NotImplementedError("range selection is not exposed by the SDK backend")
