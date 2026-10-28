"""Browser-based MJPEG thermal viewer for the HIKMICRO Pocket 2.

The capture thread opens :class:`UvcThermalSource` in raw mode, turns each
196x256 uint16 grid into Celsius with :func:`radiometry.frame_to_celsius`,
colour-maps and JPEG-encodes it once, then publishes it through a single-slot
holder. Any number of HTTP clients render the same latest frame; a slow client
can never grow memory because older frames are simply overwritten.

When the node is not an XTherm-family sensor (no 256x196 YUYV raw mode) the
capture thread falls back once, at startup, to plain MJPEG video through
OpenCV and publishes those BGR frames unchanged; temperature readouts then
stay empty while the stream keeps working.

Only the standard library, numpy and opencv are used here. Nothing in this
module imports Qt.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from . import radiometry

BOUNDARY = "frame"
HISTORY_LEN = 200
JPEG_QUALITY = 85
DEFAULT_SCALE = (20.0, 40.0)
RANGE_MODES = ("normal", "high")
VIDEO_FOURCC = "MJPG"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_READ_RETRIES = 5
S_FMT_MARKER = "VIDIOC_S_FMT("

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HIKMICRO thermal viewer</title>
<style>
:root {
  --bg: #0d0f12;
  --panel: #15181d;
  --fg: #d8dee9;
  --dim: #7b8494;
  --accent: #ff9d3c;
  --border: #262b33;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 16px;
  background: var(--bg);
  color: var(--fg);
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
h1 { font-size: 14px; font-weight: 600; letter-spacing: 0.08em; color: var(--dim);
     text-transform: uppercase; margin: 0 0 12px 0; }
.layout { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; }
@media (max-width: 900px) { .layout { grid-template-columns: minmax(0, 1fr); } }
.view { background: #000; border: 1px solid var(--border); border-radius: 6px;
        padding: 4px; display: flex; }
.view img { width: 100%; height: auto; display: block; image-rendering: auto; }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
         padding: 12px; display: flex; flex-direction: column; gap: 14px; }
.readout { font-size: 26px; color: var(--accent); }
.readout small { font-size: 12px; color: var(--dim); }
.row { display: flex; justify-content: space-between; gap: 8px; font-size: 13px; }
.row span:last-child { color: var(--accent); }
label.ctl { display: block; font-size: 12px; color: var(--dim); margin-bottom: 4px; }
input[type=range] { width: 100%; accent-color: var(--accent); }
.chk { display: flex; align-items: center; gap: 8px; font-size: 13px; }
canvas { width: 100%; height: 60px; background: #0a0c0f;
         border: 1px solid var(--border); border-radius: 4px; }
.hint { font-size: 11px; color: var(--dim); line-height: 1.5; }
</style>
</head>
<body>
<h1>HIKMICRO Pocket 2 - live thermal</h1>
<div class="layout">
  <div class="view"><img id="stream" src="/stream" alt="thermal stream"></div>
  <div class="panel">
    <div>
      <div class="readout"><span id="center">--</span><small> C center</small></div>
      <div class="row"><span>mode</span><span id="mode">thermal</span></div>
      <div class="row"><span>min</span><span id="min">--</span></div>
      <div class="row"><span>max</span><span id="max">--</span></div>
      <div class="row"><span>palette</span><span id="pal">-- / --</span></div>
      <div class="row"><span>fps</span><span id="fps">--</span></div>
      <div class="row"><span>frames</span><span id="count">0</span></div>
    </div>
    <canvas id="spark" width="300" height="60"></canvas>
    <div>
      <label class="ctl" for="smin">scale min <span id="sminv">--</span> C</label>
      <input type="range" id="smin" min="-20" max="150" step="0.5">
    </div>
    <div>
      <label class="ctl" for="smax">scale max <span id="smaxv">--</span> C</label>
      <input type="range" id="smax" min="-20" max="150" step="0.5">
    </div>
    <div class="chk"><input type="checkbox" id="high"><label for="high">high range</label></div>
    <div class="chk"><input type="checkbox" id="palette"><label for="palette">palette</label></div>
    <div class="hint">Sliders adjust the colour scale. Range mode re-commands the camera.</div>
  </div>
</div>
<script>
const el = (id) => document.getElementById(id);
const smin = el("smin"), smax = el("smax");
let timer = null;

function fmt(v) { return Number.isFinite(v) ? Number(v).toFixed(1) : "--"; }

function schedule() {
  if (timer !== null) { clearTimeout(timer); }
  timer = setTimeout(send, 100);
}

function send() {
  timer = null;
  let lo = parseFloat(smin.value), hi = parseFloat(smax.value);
  if (lo > hi) { const t = lo; lo = hi; hi = t; }
  const body = { scale_min: lo, scale_max: hi };
  if (el("high").checked !== lastHigh) { body.range_mode = el("high").checked ? "high" : "normal"; }
  if (el("palette").checked !== lastPalette) { body.palette = el("palette").checked; }
  fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body)
  }).then((r) => r.json()).then(apply).catch(() => {});
}

let lastHigh = false, lastPalette = true, dragging = false;
smin.addEventListener("input", () => { dragging = true; el("sminv").textContent = smin.value;
  schedule(); });
smax.addEventListener("input", () => { dragging = true; el("smaxv").textContent = smax.value;
  schedule(); });
el("high").addEventListener("change", schedule);
el("palette").addEventListener("change", schedule);

function drawSpark(hist) {
  const c = el("spark"), ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  if (!hist || hist.length < 2) { return; }
  let lo = Math.min(...hist), hi = Math.max(...hist);
  if (hi - lo < 1) { hi = lo + 1; }
  ctx.beginPath();
  hist.forEach((v, i) => {
    const x = (i / (hist.length - 1)) * (w - 2) + 1;
    const y = h - 1 - ((v - lo) / (hi - lo)) * (h - 2);
    if (i === 0) { ctx.moveTo(x, y); } else { ctx.lineTo(x, y); }
  });
  ctx.strokeStyle = "#ff9d3c";
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function apply(s) {
  if (!s) { return; }
  el("mode").textContent = s.mode || "thermal";
  el("center").textContent = fmt(s.center_c);
  el("min").textContent = fmt(s.min_c) + " C";
  el("max").textContent = fmt(s.max_c) + " C";
  el("pal").textContent = fmt(s.palette_bottom_c) + " / " + fmt(s.palette_top_c) + " C";
  el("fps").textContent = (s.fps || 0).toFixed(1);
  el("count").textContent = s.frame_count || 0;
  el("sminv").textContent = fmt(s.color_scale_min);
  el("smaxv").textContent = fmt(s.color_scale_max);
  if (!dragging) {
    smin.value = s.color_scale_min;
    smax.value = s.color_scale_max;
  }
  el("high").checked = s.range_mode === "high";
  el("palette").checked = !!s.palette;
  lastHigh = el("high").checked;
  lastPalette = el("palette").checked;
  drawSpark(s.history);
}

function poll() {
  fetch("/state").then((r) => r.json()).then((s) => { dragging = false; apply(s); })
    .catch(() => {});
}
poll();
setInterval(poll, 500);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Thread-safe frame publication
# ---------------------------------------------------------------------------


class FrameHolder:
    """Single-slot latest-frame holder (condition variable, no backlog)."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frame: bytes | None = None
        self._seq = 0

    def publish(self, frame: bytes) -> None:
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def current(self) -> bytes | None:
        with self._cond:
            return self._frame

    def sequence(self) -> int:
        with self._cond:
            return self._seq

    def wait(self, last_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        """Return the latest frame once the sequence moves past ``last_seq``.

        Returns ``(None, last_seq)`` on timeout so callers can send keepalives.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._seq > last_seq or self._frame is None, timeout)
            return self._frame, self._seq


class ControlState:
    """Thread-safe view settings shared by the HTTP handlers and the capture loop."""

    def __init__(
        self,
        scale_min: float = DEFAULT_SCALE[0],
        scale_max: float = DEFAULT_SCALE[1],
        range_mode: str = "normal",
        palette: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._scale_min = float(scale_min)
        self._scale_max = float(scale_max)
        self._range_mode = range_mode
        self._palette = bool(palette)
        self._range_changed = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scale_min": self._scale_min,
                "scale_max": self._scale_max,
                "range_mode": self._range_mode,
                "palette": self._palette,
            }

    def take_range_change(self) -> str | None:
        """Return the pending range command once, then clear it."""
        with self._lock:
            if not self._range_changed:
                return None
            self._range_changed = False
            return self._range_mode

    def apply(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Merge and validate a control payload; raise ``ValueError`` if invalid."""
        if not isinstance(payload, dict):
            raise ValueError("control body must be a JSON object")
        with self._lock:
            scale_min = self._scale_min
            scale_max = self._scale_max
            range_mode = self._range_mode
            palette = self._palette

            if "scale_min" in payload:
                scale_min = _coerce_float(payload["scale_min"], "scale_min")
            if "scale_max" in payload:
                scale_max = _coerce_float(payload["scale_max"], "scale_max")
            if scale_min > scale_max:
                scale_min, scale_max = scale_max, scale_min
            if "range_mode" in payload:
                value = payload["range_mode"]
                if not isinstance(value, str) or value not in RANGE_MODES:
                    raise ValueError(f"range_mode must be one of {list(RANGE_MODES)}")
                if value != range_mode:
                    self._range_changed = True
                range_mode = value
            if "palette" in payload:
                value = payload["palette"]
                if not isinstance(value, bool):
                    raise ValueError("palette must be a boolean")
                palette = value

            self._scale_min = scale_min
            self._scale_max = scale_max
            self._range_mode = range_mode
            self._palette = palette
            return {
                "scale_min": scale_min,
                "scale_max": scale_max,
                "range_mode": range_mode,
                "palette": palette,
            }


def _coerce_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------


class WebCapture(threading.Thread):
    """Reads raw grids, converts to Celsius and publishes colour-mapped JPEG."""

    def __init__(
        self,
        device: str,
        holder: FrameHolder,
        control: ControlState,
        use_palette: bool = True,
        source_factory: Any = None,
    ) -> None:
        super().__init__(name="hikmicro-mjpeg", daemon=True)
        self.device = device
        self.holder = holder
        self.control = control
        self.use_palette = use_palette
        self._source_factory = source_factory or _default_source_factory
        self._halt = threading.Event()
        self._state_lock = threading.Lock()
        self._state: dict[str, Any] = {
            "mode": "thermal",
            "center_c": None,
            "min_c": None,
            "max_c": None,
            "palette_top_c": None,
            "palette_bottom_c": None,
            "fps": 0.0,
            "color_scale_min": DEFAULT_SCALE[0],
            "color_scale_max": DEFAULT_SCALE[1],
            "range_mode": control.snapshot()["range_mode"],
            "palette": use_palette,
            "frame_count": 0,
            "history": [],
        }
        self._history: deque[float] = deque(maxlen=HISTORY_LEN)
        self._fps_window: deque[float] = deque(maxlen=30)
        self._error_msg: str | None = None

    # -- public ------------------------------------------------------------
    @property
    def error(self) -> str | None:
        return self._error_msg

    def stop(self) -> None:
        self._halt.set()

    def state(self) -> dict[str, Any]:
        with self._state_lock:
            out = dict(self._state)
            out["history"] = list(self._history)
            return out

    @property
    def mode(self) -> str:
        with self._state_lock:
            return str(self._state["mode"])

    def _set_mode(self, mode: str) -> None:
        with self._state_lock:
            self._state["mode"] = mode

    def run(self) -> None:
        try:
            source = self._source_factory(self.device)
        except Exception as exc:  # noqa: BLE001 - surfaced in state
            self._error_msg = f"cannot open {self.device}: {exc}"
            return
        try:
            source.open()
        except (RuntimeError, OSError, NotImplementedError) as exc:
            if not _raw_mode_unavailable(exc):
                self._error_msg = f"capture failed: {exc}"
                self._halt.set()
                return
            self._run_video()
            return
        except Exception as exc:  # noqa: BLE001 - surfaced in state
            self._error_msg = f"capture failed: {exc}"
            self._halt.set()
            return
        try:
            try:
                with_meta = _apply_range(source, self.control.snapshot()["range_mode"])
                self._set_mode("thermal")
                self._loop(source, with_meta)
            finally:
                source.close()
        except Exception as exc:  # noqa: BLE001 - surfaced in state
            self._error_msg = f"capture failed: {exc}"
        finally:
            self._halt.set()

    def _run_video(self) -> None:
        """Stream plain MJPEG video when the node exposes no XTherm raw mode."""
        import cv2

        self._set_mode("video")
        cap = None
        misses = 0
        try:
            cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
            if not cap.isOpened():
                self._error_msg = f"cannot open {self.device} as video"
                return
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*VIDEO_FOURCC))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            while not self._halt.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    misses += 1
                    if misses >= VIDEO_READ_RETRIES:
                        self._error_msg = f"video capture failed on {self.device}"
                        return
                    time.sleep(0.01)
                    continue
                misses = 0
                self._process_video(np.asarray(frame))
        except Exception as exc:  # noqa: BLE001 - surfaced in state
            self._error_msg = f"video capture failed: {exc}"
        finally:
            if cap is not None:
                cap.release()
            self._halt.set()

    def _process_video(self, frame: np.ndarray) -> None:
        import cv2

        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        elif frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        ok, buf = _encode_jpeg(frame)
        if not ok:
            return
        now = time.monotonic()
        self._fps_window.append(now)
        with self._state_lock:
            self._state.update(
                {
                    "fps": _fps(self._fps_window),
                    "frame_count": self._state["frame_count"] + 1,
                }
            )
        self.holder.publish(bytes(buf))

    # -- internals ---------------------------------------------------------
    def _loop(self, source: Any, with_meta: bool) -> None:
        while not self._halt.is_set():
            pending = self.control.take_range_change()
            if pending is not None:
                try:
                    _apply_range(source, pending)
                except Exception as exc:  # noqa: BLE001 - keep streaming
                    self._error_msg = f"range switch failed: {exc}"
            item = source.read_frame()
            if item is None:
                time.sleep(0.005)
                continue
            self._process(np.asarray(item[0]), with_meta)

    def _process(self, frame: np.ndarray, with_meta: bool) -> None:
        settings = self.control.snapshot()
        temps = _to_celsius(frame, with_meta, settings["range_mode"] == "high")
        if temps is None or temps.size == 0:
            return
        lo = settings["scale_min"]
        hi = settings["scale_max"]
        rgb = colorize(temps, lo, hi, palette=settings["palette"] and self.use_palette)
        ok, buf = _encode_jpeg(rgb)
        if not ok:
            return
        center = float(temps[temps.shape[0] // 2, temps.shape[1] // 2])
        now = time.monotonic()
        self._fps_window.append(now)
        self._history.append(center)
        palette_lo, palette_hi = _palette_span(temps, lo, hi)
        with self._state_lock:
            self._state.update(
                {
                    "center_c": round(center, 2),
                    "min_c": round(float(np.nanmin(temps)), 2),
                    "max_c": round(float(np.nanmax(temps)), 2),
                    "palette_top_c": palette_hi,
                    "palette_bottom_c": palette_lo,
                    "fps": _fps(self._fps_window),
                    "color_scale_min": lo,
                    "color_scale_max": hi,
                    "range_mode": settings["range_mode"],
                    "palette": settings["palette"],
                    "frame_count": self._state["frame_count"] + 1,
                }
            )
        self.holder.publish(bytes(buf))


def _palette_span(temps: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Return the Celsius values the colour ramp actually spans for this frame."""
    valid = temps[np.isfinite(temps)]
    if valid.size == 0:
        return round(lo, 1), round(hi, 1)
    if hi - lo <= 0.0:
        lo = float(np.nanmin(valid))
        hi = float(np.nanmax(valid))
    return round(lo, 1), round(hi, 1)


def _fps(window: deque[float]) -> float:
    if len(window) < 2:
        return 0.0
    span = window[-1] - window[0]
    if span <= 0:
        return 0.0
    return (len(window) - 1) / span


def _to_celsius(frame: np.ndarray, with_meta: bool, high_range: bool) -> np.ndarray | None:
    """Convert a raw grid with :func:`radiometry.frame_to_celsius`.

    Returns ``None`` when the frame carries no plausible metadata (for example a
    YUYV video frame emitted because the camera ignored raw mode) so garbage is
    dropped instead of displayed.
    """
    arr = np.asarray(frame)
    if np.issubdtype(arr.dtype, np.floating):
        flat = arr.reshape(-1)
        if flat.size < radiometry.PIXELS:
            return None
        return flat[: radiometry.PIXELS].astype(np.float32).reshape(
            radiometry.HEIGHT, radiometry.WIDTH
        )
    grid = arr.astype(np.uint16, copy=False)
    if with_meta:
        meta = radiometry.parse_meta(grid, high_range=high_range)
    else:
        meta = None
    if meta is None:
        meta = _synthetic_meta(grid, high_range=high_range)
    if meta is None:
        return None
    try:
        return radiometry.frame_to_celsius(grid, meta, high_range=high_range)
    except (ValueError, IndexError, TypeError):
        return None


def _synthetic_meta(grid: np.ndarray, high_range: bool = False) -> Any | None:
    """Build the ``Meta`` the vendor block would produce for a metadata-free grid."""
    flat = np.asarray(grid).reshape(-1)
    if flat.size < radiometry.PIXELS:
        return None
    values = flat.astype(np.float64)
    cal01 = 1.0
    return radiometry.Meta(
        fpa_average=int(values.mean()),
        temp_fpa=25.0,
        temp_shutter=25.0,
        temp_core=25.0,
        max_x=int(values.argmax()),
        max_y=0,
        max_raw=int(values.max()),
        min_x=int(values.argmin()),
        min_y=0,
        min_raw=int(values.min()),
        avg_raw=int(values.mean()),
        center_raw=int(values[values.size // 2]),
        correction=0.0,
        temp_reflected=25.0,
        temp_air=25.0,
        humidity=50.0,
        emissivity=0.95,
        distance=1,
        cal00=0.0,
        cal01=cal01,
        cal02=0.0,
        cal03=0.0,
        cal04=0.0,
        cal05=0.0,
        high_range=high_range,
    )


def colorize(
    temps: np.ndarray,
    scale_min: float,
    scale_max: float,
    palette: bool = True,
) -> np.ndarray:
    """Map a Celsius grid to an RGB uint8 image using the requested colour scale."""
    import cv2

    span = float(scale_max) - float(scale_min)
    if span <= 0.0:
        span = 1.0
    scaled = (np.nan_to_num(temps, nan=scale_min) - float(scale_min)) * (255.0 / span)
    norm = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    if palette:
        rgb = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    else:
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return rgb


def _encode_jpeg(rgb: np.ndarray) -> tuple[bool, bytes]:
    import cv2

    ok, buf = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return False, b""
    return True, buf.tobytes()


def _default_source_factory(device: str) -> Any:
    if device == "hcusbsdk":
        from .sdk_source import SdkThermalSource

        return SdkThermalSource()

    from .uvc_source import UvcThermalSource

    return UvcThermalSource(device=device, raw_mode=True, fourcc="YUYV")


def _raw_mode_unavailable(exc: BaseException) -> bool:
    """True when a thermal source failed to open because it is not XTherm-family."""
    if isinstance(exc, NotImplementedError):
        return True
    return isinstance(exc, (RuntimeError, OSError)) and S_FMT_MARKER in str(exc)


def _apply_range(source: Any, mode: str, meta: Any = None) -> bool:
    """Command the camera range; return True when radiometry should be applied."""
    from .uvc_source import CMD_RANGE_HIGH, CMD_RANGE_NORMAL

    setter = getattr(source, "set_mode", None)
    honours = True
    if callable(setter):
        try:
            setter(CMD_RANGE_HIGH if mode == "high" else CMD_RANGE_NORMAL)
        except (OSError, NotImplementedError, AttributeError):
            honours = False
    else:
        honours = False
    if meta is not None and hasattr(meta, "high_range"):
        meta.high_range = mode == "high" and honours
    return True


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class ViewerServer(ThreadingHTTPServer):
    """HTTP server carrying the holder, control state and capture thread."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        holder: FrameHolder,
        control: ControlState,
        capture: Any = None,
    ) -> None:
        super().__init__(address, ViewerHandler)
        self.holder = holder
        self.control = control
        self.capture = capture


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return None

    # -- routing -----------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path == "/":
            body = _PAGE.encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)
        elif path == "/state":
            self._send_json(200, self._state())
        elif path == "/frame.jpg":
            frame = self.server.holder.current()
            if frame is None:
                self._send_json(503, {"error": "no frame yet"})
                return
            self._send(200, "image/jpeg", frame, cache=False)
        elif path == "/stream":
            self._stream()
        else:
            self._send_json(404, {"error": f"unknown path {path}"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        if path != "/control":
            self._send_json(404, {"error": f"unknown path {path}"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send_json(400, {"error": "invalid Content-Length"})
            return
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "body must be JSON"})
            return
        try:
            applied = self.server.control.apply(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return
        state = self._state()
        state["color_scale_min"] = applied["scale_min"]
        state["color_scale_max"] = applied["scale_max"]
        state["range_mode"] = applied["range_mode"]
        state["palette"] = applied["palette"]
        self._send_json(200, state)

    # -- helpers -----------------------------------------------------------
    def _state(self) -> dict[str, Any]:
        capture = self.server.capture
        if capture is None:
            state = {
                "mode": "thermal",
                "center_c": None,
                "min_c": None,
                "max_c": None,
                "palette_top_c": None,
                "palette_bottom_c": None,
                "fps": 0.0,
                "color_scale_min": DEFAULT_SCALE[0],
                "color_scale_max": DEFAULT_SCALE[1],
                "range_mode": self.server.control.snapshot()["range_mode"],
                "palette": self.server.control.snapshot()["palette"],
                "frame_count": 0,
                "history": [],
            }
        else:
            state = capture.state()
        state.update(self.server.control.snapshot())
        state["error"] = getattr(capture, "error", None)
        return state

    def _send(self, code: int, ctype: str, body: bytes, cache: bool = True) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, code: int, payload: dict[str, Any]) -> None:
        self._send(code, "application/json", json.dumps(payload).encode("utf-8"), cache=False)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.end_headers()
        seq = -1
        idle = 0.0
        try:
            while True:
                frame, seq = self.server.holder.wait(seq, timeout=1.0)
                if frame is None:
                    idle += 1.0
                    if idle >= 10.0:
                        return
                    continue
                idle = 0.0
                header = (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(header)
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return


def make_server(
    host: str,
    port: int,
    holder: FrameHolder,
    control: ControlState,
    capture: Any = None,
) -> ViewerServer:
    """Bind a ``ViewerServer``; ``port=0`` picks a free port."""
    return ViewerServer((host, port), holder, control, capture)


def close_server(server: ViewerServer, capture: Any = None) -> None:
    """Stop the capture thread and shut the HTTP server down."""
    if capture is not None:
        try:
            capture.stop()
        except AttributeError:
            pass
        join = getattr(capture, "join", None)
        if callable(join):
            join(timeout=3.0)
    server.shutdown()
    server.server_close()


def run_web_viewer(
    device: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    use_palette: bool = True,
    open_browser: bool = True,
    range_mode: str = "normal",
) -> int:
    """Serve the live thermal view in a browser until Ctrl+C."""
    if range_mode not in RANGE_MODES:
        raise ValueError(f"range_mode must be one of {list(RANGE_MODES)}")

    holder = FrameHolder()
    control = ControlState(range_mode=range_mode, palette=use_palette)
    capture = WebCapture(device=device, holder=holder, control=control, use_palette=use_palette)
    server = make_server(host, port, holder, control, capture)
    url = f"http://{host}:{server.server_address[1]}/"
    print(f"thermal viewer listening on {url} (device {device}, {range_mode} range)")
    if capture.error:
        print(f"warning: {capture.error}")

    capture.start()
    thread = threading.Thread(target=server.serve_forever, name="hikmicro-http", daemon=True)
    thread.start()
    if open_browser:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - a missing browser is not fatal
            pass
    try:
        while thread.is_alive():
            time.sleep(0.2)
        if capture.error:
            print(f"error: {capture.error}")
            return 1
        return 0
    except KeyboardInterrupt:
        print("shutting down")
        return 0
    finally:
        close_server(server, capture)


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - manual entry
    parser = argparse.ArgumentParser(description="browser MJPEG thermal viewer")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--range", dest="range_mode", choices=list(RANGE_MODES), default="normal")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    return run_web_viewer(
        device=args.device,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
        range_mode=args.range_mode,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
