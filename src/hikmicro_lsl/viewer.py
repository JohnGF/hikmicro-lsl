"""Thermal viewer for the HIKMICRO Pocket 2.

Provides a PySide6 + pyqtgraph GUI that reads the UVC stream, decodes MJPG
frames, parses the overlay (temperatures, emissivity, timestamp, palette) and
visualizes everything with an interactive thermal image, control panel,
gradient legend and live statistics.

Headless mode is guarded by deferring Qt imports until ``run_viewer`` is
called. ``import hikmicro_lsl.viewer`` works without a display so it does not
break CLI ``--help`` or headless tests.

Run with an ephemeral Qt/PyQtGraph (no install needed):
    uv run --with pyside6 --with pyqtgraph hikmicro-lsl view --device /dev/video2
Or add the extra:
    uv pip install -e ".[ui]"
"""

from __future__ import annotations

import io
import math
import queue
import sys
import threading
import time
import typing as t
from pathlib import Path

from .overlay import OverlayReader, inverse_palette, palette_lut
from .uvc_source import UvcThermalSource

FrameData = t.Dict[str, t.Any]


def _decode_image(jpeg_bytes: bytes) -> t.Any:
    import numpy as np

    try:
        import cv2

        arr = np.frombuffer(jpeg_bytes, np.uint8)
        return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    try:
        from PIL import Image as _PIL

        img = _PIL.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        return np.array(img)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Headless decode helpers (used by CaptureThread and tests, no Qt dependency)
# ---------------------------------------------------------------------------


def decode_jpeg_to_rgb(jpeg_bytes: bytes) -> t.Optional[t.Any]:
    """Decode MJPG JPEG bytes to an (H, W, 3) uint8 numpy array."""
    if not jpeg_bytes:
        return None
    return _decode_image(jpeg_bytes)


def parse_frame_data(
    jpeg_bytes: bytes, reader: OverlayReader, use_palette: bool = True
) -> t.Optional[FrameData]:
    """Decode JPEG and extract overlay metrics into a dict.

    Runs on the capture thread (or a second worker). Returns ``None`` if
    decoding fails. ``reader`` is an instance of ``OverlayReader``.
    """
    rgb = decode_jpeg_to_rgb(jpeg_bytes)
    if rgb is None:
        return None

    overlay = reader.parse(rgb)

    depth_map: t.Optional[t.Any] = None
    if use_palette:
        try:
            depth_map = inverse_palette(
                rgb,
                palette_column=None,
                top_c=overlay.palette_top_c,
                bottom_c=overlay.palette_bottom_c,
            )
        except Exception:
            depth_map = None

    return {
        "rgb_frame": rgb,
        "center_c": overlay.center_c,
        "max_c": overlay.max_c,
        "min_c": overlay.min_c,
        "emissivity": overlay.emissivity,
        "timestamp": str(overlay.timestamp) if overlay.timestamp else None,
        "palette_top_c": overlay.palette_top_c,
        "palette_bottom_c": overlay.palette_bottom_c,
        "depth_map": depth_map,
        "overlay": overlay,
        "took_ms": 0.0,
        "stale": False,
    }


# ---------------------------------------------------------------------------
# Qt imports – deferred so the module is importable headless
# ---------------------------------------------------------------------------

QtCore: t.Any = None
QtGui: t.Any = None
QtWidgets: t.Any = None
pg: t.Any = None
ImageQtMod: t.Any = None


def _ensure_qt() -> None:
    """Import Qt bindings on demand. Raises if unavailable."""
    global QtCore, QtGui, QtWidgets, pg, ImageQtMod
    if QtCore is None:
        from PySide6 import QtCore as _c  # type: ignore
        from PySide6 import QtGui as _g  # type: ignore
        from PySide6 import QtWidgets as _w  # type: ignore

        QtCore, QtGui, QtWidgets = _c, _g, _w
    if pg is None:
        import pyqtgraph as _pg  # type: ignore

        pg = _pg
    if ImageQtMod is None:
        try:
            from PySide6 import QtGui as _qtgui_mod

            if hasattr(_qtgui_mod, "QtImage"):
                ImageQtMod = _qtgui_mod
            else:
                ImageQtMod = _qtgui_mod
        except Exception:
            ImageQtMod = None


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------


class CaptureThread(QtCore.QThread if QtCore else object):  # type: ignore[misc]
    """Background thread that reads MJPG JPEG frames from the UVC device."""

    error = None  # type: ignore[misc]
    frame_ready = None
    frame_dropped = None

    def __init__(self, device: str, parent: t.Optional[t.Any] = None) -> None:
        super().__init__(parent)  # type: ignore[misc]
        self._device = device
        self._stop_event = threading.Event()
        self._dropped = 0
        self._cap = None
        # define signals only when Qt loaded
        if QtCore is not None:
            from PySide6.QtCore import Signal

            self.frame_ready = Signal(bytes)  # type: ignore[method-assign]
            self.frame_dropped = Signal()  # type: ignore[method-assign]
            self.error = Signal(str)  # type: ignore[method-assign]

    def run(self) -> None:  # noqa: N802
        try:
            import cv2

            self._cap = cv2.VideoCapture(self._device)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # Request MJPG explicitly if available
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        except Exception as exc:
            if self.error is not None:
                self.error.emit(str(exc))
            return
        last_fps_t = time.monotonic()
        count = 0
        while not self._stop_event.is_set():
            ok, buf = self._cap.read()
            if not ok or buf is None:
                continue
            # encode to JPEG bytes
            ok2, jpeg = cv2.imencode(".jpg", buf)
            if not ok2:
                continue
            jpeg_bytes = jpeg.tobytes()
            if self.frame_ready is not None:
                self.frame_ready.emit(jpeg_bytes)  # type: ignore[attr-defined]
            count += 1
            now = time.monotonic()
            if now - last_fps_t >= 1.0:
                last_fps_t = now
        if self._cap is not None:
            self._cap.release()

    def stop(self) -> None:
        self._stop_event.set()
        self.wait(3000)


# ---------------------------------------------------------------------------
# Worker that parses overlay on a separate thread (keeps GUI responsive)
# ---------------------------------------------------------------------------


class ParseWorker(QtCore.QThread if QtCore else object):  # type: ignore[misc]
    parsed = None  # set in _ensure_qt

    def __init__(self, reader: OverlayReader, use_palette: bool = True) -> None:
        super().__init__()  # type: ignore[misc]
        self._reader = reader
        self._use_palette = use_palette
        self._q_in: "queue.Queue[t.Tuple[bytes, float]]" = queue.Queue(maxsize=4)
        self._stop_event = threading.Event()
        self._last_good: t.Optional[FrameData] = None

    def enqueue(self, jpeg: bytes, ts: float) -> bool:
        try:
            self._q_in.put_nowait((jpeg, ts))
            return True
        except queue.Full:
            return False

    def run(self) -> None:  # noqa: N802
        self._ensure_signal()
        while not self._stop_event.is_set():
            try:
                jpeg, ts = self._q_in.get(timeout=0.05)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            res = parse_frame_data(jpeg, self._reader, self._use_palette)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if res is None:
                continue
            res["took_ms"] = dt_ms
            self._last_good = res
            if dt_ms > 15.0:
                res["stale"] = True
                self.parsed.emit(res)  # type: ignore[attr-defined]
                # drop next few frames
                for _ in range(2):
                    try:
                        self._q_in.get_nowait()
                    except queue.Empty:
                        break
            else:
                self.parsed.emit(res)  # type: ignore[attr-defined]

    def _ensure_signal(self) -> None:
        if ParseWorker.parsed is None:
            from PySide6.QtCore import Signal

            ParseWorker.parsed = Signal(t.Dict)  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._stop_event.set()
        self.wait(2000)


# ---------------------------------------------------------------------------
# Gradient legend widget
# ---------------------------------------------------------------------------


class GradientLegend(QtWidgets.QWidget if QtWidgets else object):  # type: ignore[misc]
    """Custom horizontal gradient legend showing the palette bar with temp labels."""

    def __init__(self, parent: t.Optional[t.Any] = None) -> None:
        super().__init__(parent)  # type: ignore[misc]
        self._gradient: t.Optional[t.List[RGB]] = None
        self._min_c: float = 0.0
        self._max_c: float = 0.0
        self.setFixedHeight(32)

    def set_gradient(self, lut: t.List[RGB], min_c: float, max_c: float) -> None:
        self._gradient = lut
        self._min_c = min_c
        self._max_c = max_c
        self.update()

    def paintEvent(self, ev: t.Any) -> None:  # noqa: N802
        if QtWidgets is None:
            return
        p = QtGui.QPainter(self)
        w = self.width()
        h = self.height()
        if not self._gradient:
            p.setPen(QtGui.QColor(128, 128, 128))
            p.drawText(self.rect(), QtCore.Qt.AlignCenter, "no palette")
            return
        grad = self._gradient
        for i in range(w):
            idx = int(i / max(1, w - 1) * (len(grad) - 1))
            c = grad[idx]
            p.setPen(QtGui.QColor(int(c[0]), int(c[1]), int(c[2])))
            p.drawLine(i, 0, i, h - 8)
        p.setPen(QtGui.QColor(255, 255, 255))
        p.setFont(QtGui.QFont("monospace", 8))
        p.drawText(0, h - 4, f"{self._max_c:.1f}")
        p.drawText(w - 30, h - 4, f"{self._min_c:.1f}")


# ---------------------------------------------------------------------------
# Main thermal viewer window
# ---------------------------------------------------------------------------


class ThermalViewer(QtWidgets.QMainWindow if QtWidgets else object):  # type: ignore[misc]
    """Main viewer window with thermal image, readouts, legend, sparkline."""

    def __init__(
        self,
        device: str,
        use_palette: bool = True,
        fullscreen: bool = False,
    ) -> None:
        super().__init__()  # type: ignore[misc]
        _ensure_qt()
        self._device = device
        self._use_palette = use_palette
        self._current_rgb: t.Optional[t.Any] = None
        self._current_data: t.Optional[FrameData] = None
        self._last_valid_data: t.Optional[FrameData] = None
        self._stale: bool = False

        self.reader = OverlayReader(backend="template")
        self._build_ui()
        if fullscreen:
            self.showFullScreen()
        else:
            self.resize(1000, 600)

        # Capture + parse threads
        self._thread = CaptureThread(device)
        self._worker = ParseWorker(self.reader, use_palette)

        self._thread.frame_ready.connect(self._on_frame)  # type: ignore[attr-defined]
        self._thread.error.connect(self._on_error)  # type: ignore[attr-defined]
        self._worker.parsed.connect(self._on_parsed)  # type: ignore[attr-defined]

        self._worker.start()
        self._thread.start()

        self._center_times: t.List[float] = []
        self._center_vals: t.List[float] = []

    def _build_ui(self) -> None:
        _ensure_qt()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        splitter = QtWidgets.QSplitter()
        layout.addWidget(splitter)

        # Left: thermal image
        self._img_view = pg.ImageView(view=pg.PlotItem())
        self._img_view.ui.histogram.hide()
        self._img_view.ui.roiBtn.hide()
        self._img_view.ui.menuBtn.hide()
        self._img_view.view.setAspectLocked(True)
        self._img_view.view.setResizeAnchor(pg.PlotItem().getViewBox().getResizeAnchor())
        splitter.addWidget(self._img_view)

        # Right panel
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        splitter.addWidget(right)

        # Readouts
        self._lbl_center = self._make_readout()
        self._lbl_max = self._make_readout()
        self._lbl_min = self._make_readout()
        rv.addWidget(QtWidgets.QLabel("Center"))
        rv.addWidget(self._lbl_center)
        rv.addWidget(QtWidgets.QLabel("Max"))
        rv.addWidget(self._lbl_max)
        rv.addWidget(QtWidgets.QLabel("Min"))
        rv.addWidget(self._lbl_min)

        # Gradient legend
        self._legend = GradientLegend()
        rv.addWidget(self._legend)

        # Emissivity + timestamp
        self._lbl_emiss = QtWidgets.QLabel("Emissivity: --")
        self._lbl_ts = QtWidgets.QLabel("Timestamp: --")
        self._lbl_emiss.setStyleSheet("font-family: monospace;")
        self._lbl_ts.setStyleSheet("font-family: monospace;")
        rv.addWidget(self._lbl_emiss)
        rv.addWidget(self._lbl_ts)

        # Sparkline
        self._spark = pg.PlotWidget()
        self._spark.setLabel("left", "Center C")
        self._spark.setLabel("bottom", "time (s)")
        self._spark.showGrid(x=False, y=False)
        self._spark_curve = self._spark.plot(pen=pg.mkPen(color=(255, 200, 50), width=2))
        rv.addWidget(self._spark)

        # Checkboxes
        cb_box = QtWidgets.QHBoxLayout()
        self._cb_overlay = QtWidgets.QCheckBox("Show overlay mask")
        self._cb_inverse = QtWidgets.QCheckBox("Dense palette inverse")
        self._cb_overlay.setChecked(True)
        self._cb_overlay.toggled.connect(self._on_toggle_overlay)
        self._cb_inverse.toggled.connect(self._on_toggle_inverse)
        cb_box.addWidget(self._cb_overlay)
        cb_box.addWidget(self._cb_inverse)
        rv.addLayout(cb_box)

        # Status bar
        self._status = QtWidgets.QStatusBar()
        self._warn_banner = QtWidgets.QLabel("")
        self._warn_banner.setStyleSheet("color: amber; font-weight: bold;")
        self._status.addPermanentWidget(self._warn_banner)
        self._status_label = QtWidgets.QLabel("connecting...")
        self._status.addPermanentWidget(self._status_label)
        self.setStatusBar(self._status)

        self.setWindowTitle(f"HIKMICRO viewer — {self._device}")

    def _make_readout(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel("--")
        f = QtGui.QFont("monospace", 24)
        f.setBold(True)
        lbl.setFont(f)
        lbl.setStyleSheet("color: #e0e0e0;")
        return lbl

    def _on_frame(self, jpeg: bytes) -> None:
        if self._worker is not None and not self._worker.enqueue(jpeg, time.monotonic()):
            # queue full -> frame dropped
            self._status_label.setText(f"{self._device} | fps -- | drops: ? | (parse backlog)")

    def _on_parsed(self, data: FrameData) -> None:
        self._current_data = data
        self._last_valid_data = data
        self._stale = data.get("stale", False)

        rgb = data["rgb_frame"]
        if self._cb_inverse.isChecked() and data.get("depth_map") is not None:
            # render inverse palette temperature map
            try:
                lut = palette_lut(self.reader)
                inv = inverse_palette(rgb, self.reader, data["overlay"])
                # normalize depth to 0..255 and apply palette
                import numpy as np

                d = inv
                mn = float(np.nanmin(d)) if d.size else 0.0
                mx = float(np.nanmax(d)) if d.size else 1.0
                if mx > mn:
                    norm = ((d - mn) / (mx - mn) * 255).astype("uint8")
                else:
                    norm = np.zeros_like(d, dtype="uint8")
                rgba = pg.applyPalette(norm, lut, alpha=255)  # type: ignore
                self._img_view.setImage(rgba.transpose((1, 0, 2)), autoLevels=True)  # type: ignore
            except Exception:
                self._img_view.setImage(rgb, autoLevels=True)  # type: ignore
        else:
            self._img_view.setImage(rgb, autoLevels=True)  # type: ignore

        self._update_readouts(data)
        self._update_status(data)
        self._update_sparkline(data)
        self._update_legend(data)

    def _on_toggle_overlay(self, on: bool) -> None:
        pass  # overlay mask drawn via pyqtgraph ROI items (stub)

    def _on_toggle_inverse(self, on: bool) -> None:
        if self._last_valid_data is not None:
            self._on_parsed(self._last_valid_data)

    def _on_error(self, msg: str) -> None:
        self._warn_banner.setText(f"device error: {msg}")
        self._thread.stop()

    def _update_readouts(self, data: FrameData) -> None:
        def fmt(v: t.Optional[float]) -> str:
            return f"{v:.1f}" if v is not None else "--"

        self._lbl_center.setText(fmt(data["center_c"]))
        self._lbl_max.setText(fmt(data["max_c"]))
        self._lbl_min.setText(fmt(data["min_c"]))

    def _update_status(self, data: FrameData) -> None:
        fps = getattr(self._thread, "_current_fps", "--")
        drops = getattr(self._thread, "_dropped", 0)
        warn = ""
        if data["center_c"] is None:
            warn = " temperatures unavailable - install the [ocr] extra"
        self._warn_banner.setText(warn)
        stale_str = " (stale)" if self._stale else ""
        self._status_label.setText(f"{self._device} | fps: {fps} | drops: {drops}{stale_str}")

    def _update_sparkline(self, data: FrameData) -> None:
        if data["center_c"] is None:
            return
        now = time.time()
        self._center_times.append(now)
        self._center_vals.append(data["center_c"])
        # keep last 60s
        cutoff = now - 60.0
        while self._center_times and self._center_times[0] < cutoff:
            self._center_times.pop(0)
            self._center_vals.pop(0)
        if self._center_times:
            t_arr = [t - self._center_times[0] for t in self._center_times]
            self._spark_curve.setData(t_arr, self._center_vals)

    def _update_legend(self, data: FrameData) -> None:
        try:
            lut = palette_lut(self.reader)
            t_min = data["palette_bottom_c"]
            t_max = data["palette_top_c"]
            if t_min is None:
                t_min = data["min_c"] or 0.0
            if t_max is None:
                t_max = data["max_c"] or 0.0
            self._legend.set_gradient(lut, t_min, t_max)
        except Exception:
            pass

    def closeEvent(self, ev: t.Any) -> None:  # noqa: N802
        self._thread.stop()
        self._worker.stop()
        super().closeEvent(ev)  # type: ignore

    def keyPressEvent(self, ev: t.Any) -> None:  # noqa: N802
        k = ev.key()
        if k == QtCore.Qt.Key_Q or k == QtCore.Qt.Key_Escape:
            self.close()
        elif k == QtCore.Qt.Key_S:
            self._save_frame()

    def _save_frame(self) -> None:
        if self._last_valid_data is None:
            return
        rgb = self._last_valid_data["rgb_frame"]
        rec_dir = Path.cwd() / "recordings"
        rec_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = rec_dir / f"frame_{ts}.png"
        try:
            img = Image.fromarray(rgb)
            img.save(path)
            self._status_label.setText(f"saved: {path}")
        except Exception as exc:
            self._status_label.setText(f"save failed: {exc}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_viewer(device: str, use_palette: bool = True, fullscreen: bool = False) -> int:
    """Construct QApplication, the window, threads, wire signals, start, run."""
    _ensure_qt()
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    win = ThermalViewer(device, use_palette=use_palette, fullscreen=fullscreen)
    win.show()
    return int(app.exec())
