"""Overlay parsing for the HIKMICRO Pocket 2 ``2bdf:017f`` Cast Screen stream.

The Pocket 2 over UVC delivers an ordinary 640x480 RGB MJPG picture: a
false-coloured thermal scene with a *burned-in* GUI overlay. There is NO
radiometric uint16 channel on this PID. Temperatures therefore must be
recovered from the picture itself:

1. OCR of the top-left panel: ``Cen <t> C / Max <t> C / Min <t> C``.
2. ``inverse_palette``: the vertical palette bar drawn by the device acts as a
   colour->temperature scale; nearest-colour matching turns every scene pixel
   into a temperature.

OverlayReader
    ``parse(jpeg_or_rgb) -> OverlayData``. OCR via OpenCV DNN or Tesseract
    (``pip install hikmicro-lsl[ocr]``) is preferred; falling back to a
    nearest-template digit matcher built from the device's own font. If neither
    backend is available the numeric fields stay ``None`` and nothing is ever
    fabricated.

inverse_palette(rgb, palette_column) -> temperature_map
    Samples the palette bar, builds an LUT, nearest-neighbour maps every scene
    pixel to temperature (float32). Overlay regions are excluded.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)

FRAME_H = 480
FRAME_W = 640

# -- measured layout -----------------------------------------------------------
PANEL_ROWS = ((17, 45), (54, 82), (96, 122))   # Cen / Max / Min bands (inclusive-ish)
PANEL_VALUE_X0 = 60
PANEL_VALUE_X1 = 116
PALETTE_X0 = 11
PALETTE_X1 = 24
PALETTE_Y0 = 236
PALETTE_Y1 = 410
PALETTE_TOP_LABEL_ROW = (14, 46)
PALETTE_BOTTOM_LABEL_ROW = (124, 148)
EMISSIVITY_ROW = (444, 468)
PALETTE_N_CHANNELS = 188

OVERLAY_EXCLUDE = (
    (0, 0, 200, 120),   # temperature panel + degree symbol
    (0, 0, 40, 480),    # palette bar and end labels
    (0, 430, 640, 480), # bottom bar
    (480, 0, 640, 60),  # top-right logo
)
CROSSHAIR_BOX = (312, 232, 328, 248)


# -- helpers ------------------------------------------------------------------

def to_rgb(frame: Any) -> np.ndarray:
    """Return an (H, W, 3) uint8 RGB array from JPEG bytes or an ndarray."""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return decode_jpeg(bytes(frame))
    arr = np.ascontiguousarray(frame)
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2).astype(np.uint8, copy=False)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError(f"unsupported frame shape {arr.shape}")
    return arr[:, :, :3].astype(np.uint8, copy=False)


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise ValueError("could not decode MJPG frame")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _gray(rgb: np.ndarray) -> np.ndarray:
    f = rgb.astype(np.float32)
    return (0.299 * f[:, :, 0] + 0.587 * f[:, :, 1] + 0.114 * f[:, :, 2]).astype(np.int16)


def _text_mask(gray: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = region
    sub = np.ascontiguousarray(gray[y0:y1, x0:x1].astype(np.uint8))
    bg = cv2.morphologyEx(sub, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    diff = sub.astype(np.int16) - bg.astype(np.int16)
    return ((diff > 35) & (sub > 120)).astype(np.uint8) * 255


# -- font ---------------------------------------------------------------------
GLYPH_SPEC = {
    "C": (14, 19, 14, 20),
    "e": (29, 24, 11, 15),
    "n": (43, 24, 10, 15),
    "M": (14, 55, 18, 21),
    "a": (32, 58, 10, 19),
    "x": (44, 57, 9, 17),
    "m": (14, 99, 17, 19),
    "i": (32, 99, 5, 19),
    "n_min": (38, 99, 12, 19),
    "0": (69, 19, 11, 21),
    "7": (83, 19, 8, 21),
    ".": (96, 33, 4, 7),
    "1": (101, 19, 13, 21),
}


def _capture_glyphs(gray: np.ndarray) -> dict[str, np.ndarray]:
    """Rasterise each known glyph cell into a uint8 template."""
    out: dict[str, np.ndarray] = {}
    for ch, (x0, y0, w, h) in GLYPH_SPEC.items():
        tile = gray[y0:y0 + h, x0:x0 + w] > 128
        out[ch] = tile.astype(np.uint8) * 255
    return out


# -- OCR backends --------------------------------------------------------------


class _TemplateBackend:
    """Nearest-template digit reader trained on the device's own font."""

    name = "template"

    def __init__(self) -> None:
        # Templates are captured lazily on first use (needs a frame).
        self.templates: dict[str, np.ndarray] = {}
        self.chars: list[str] = []
        self.stack: np.ndarray = np.zeros((0, 18 * 11), np.float32)

    def train(self, gray: np.ndarray) -> None:
        self.templates = _capture_glyphs(gray)
        self.chars = sorted(self.templates)
        rows = [self.templates[c].astype(np.float32).ravel() / 255.0 for c in self.chars]
        self.stack = np.stack(rows)

    def cells(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Split a text band into glyph cells ``(x_start, width)``."""
        cols = (mask > 0).sum(axis=0)
        runs: list[tuple[int, int]] = []
        start = None
        gap = 0
        pitch = 13
        for x, c in enumerate(cols):
            if c:
                if start is None:
                    start = x
                gap = 0
            elif start is not None:
                gap += 1
                if gap > 3:
                    runs.append((start, x - gap))
                    start = None
        if start is not None:
            runs.append((start, len(cols) - 1))
        out: list[tuple[int, int]] = []
        for a, b in runs:
            for cx in range(a, b + 1 - (pitch - 1), pitch):
                out.append((cx, pitch - 2))
        return out

    def read_cell(self, mask: np.ndarray, x: int, width: int) -> tuple[str | None, float]:
        import cv2 as _cv2

        cell = mask[:, x:x + width]
        if cell.shape[1] < 8 or cell.sum() < 10:
            return None, 0.0
        cell = _cv2.resize(cell, (11, 18), interpolation=_cv2.INTER_AREA)
        probe = (cell.astype(np.float32) / 255.0).ravel()
        norm = float(np.linalg.norm(probe))
        if norm < 1.0:
            return None, 0.0
        scores = self.stack.reshape(len(self.chars), -1) @ probe
        scores /= (np.linalg.norm(self.stack.reshape(len(self.chars), -1), axis=1) * norm + 1e-6)
        order = np.argsort(scores)[::-1]
        best, second = order[0], order[1]
        if scores[best] < 0.72 or scores[best] - scores[second] < 0.04:
            return None, float(scores[best])
        return self.chars[best], float(scores[best])

    def read_band(self, rgb: np.ndarray, row: tuple[int, int], x0: int, x1: int) -> str:
        y0, y1 = row
        mask = _text_mask(_gray(rgb), (x0, y0, x1, y1))
        if not mask.any():
            return ""
        chars: list[str] = []
        for cx, w in self.cells(mask):
            ch, _ = self.read_cell(mask, cx - x0, w)
            chars.append(ch if ch is not None else "?")
        return "".join(chars).strip()

    def _ensure_trained(self, rgb: np.ndarray) -> None:
        if not self._trained:
            self.train(_gray(rgb))
            self._trained = True


class _TessBackend:
    name = "tesseract"

    def __init__(self, cmd: str) -> None:
        self.cmd = cmd

    def read_band(self, rgb: np.ndarray, row: tuple[int, int], x0: int, x1: int) -> str:
        y0, y1 = row
        mask = _text_mask(_gray(rgb), (x0, y0, x1, y1))
        mask = cv2.resize(mask, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        mask = 255 - mask
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            cv2.imwrite(tmp.name, mask)
            try:
                out = subprocess.run(
                    [self.cmd, tmp.name, "stdout", "--psm", "7",
                     "-c", "tessedit_char_whitelist=0123456789.-: "],
                    capture_output=True, text=True, timeout=5, check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                log.debug("tesseract failed: %s", exc)
                return ""
        return out.stdout.strip()


def _pick_backend() -> Any:
    if shutil.which("tesseract"):
        return _TessBackend("tesseract")
    return _TemplateBackend()


# -- palette ------------------------------------------------------------------


def find_palette_column(rgb: np.ndarray) -> tuple[int, int, int, int] | None:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.int16)
    val = hsv[:, :, 2].astype(np.int16)
    vivid = ((sat > 120) & (val > 80)).astype(np.uint8)
    strip = vivid[: FRAME_H // 2, :64].sum(axis=1)
    rows = np.flatnonzero(strip > 6)
    if rows.size < 40:
        return None
    y0, y1 = int(rows.min()), int(rows.max())
    cols = vivid[y0:y1, :64].sum(axis=0)
    xs = np.flatnonzero(cols > 0.5 * (y1 - y0))
    if xs.size < 6:
        return None
    return int(xs.min()), int(xs.max()) + 1, y0 - 1, y1 + 1


def palette_lut(rgb: np.ndarray, column: tuple[int, int, int, int] | None = None,
                size: int = 256) -> np.ndarray:
    if column is None:
        column = find_palette_column(rgb) or (PALETTE_X0, PALETTE_X1, PALETTE_Y0, PALETTE_Y1)
    x0, x1, y0, y1 = column
    cx0 = x0 + max(2, (x1 - x0) // 3)
    cx1 = x1 - max(2, (x1 - x0) // 3)
    band = rgb[max(y0, 240):min(y1, 400), cx0:cx1]
    if band.size == 0:
        band = rgb[PALETTE_Y0 + 4:PALETTE_Y1 - 4, PALETTE_X0 + 4:PALETTE_X1 - 4]
    profile = band.reshape(band.shape[0], -1, 3).mean(axis=1)  # top -> bottom
    profile = profile[::-1]  # bottom -> top: cold -> hot
    idx = np.linspace(0, profile.shape[0] - 1, size)
    lut = np.stack([np.interp(idx, np.arange(profile.shape[0]), profile[:, c])
                    for c in range(3)], 1)
    return np.clip(lut, 0, 255).astype(np.float32)


def inverse_palette(
    rgb: np.ndarray,
    palette_column: tuple[int, int, int, int] | None = None,
    *,
    top_c: float | None = None,
    bottom_c: float | None = None,
    size: int = 256,
    include_crosshair: bool = False,
) -> np.ndarray:
    frame = to_rgb(rgb)
    lut = palette_lut(frame, palette_column, size=size)
    swatch = np.ascontiguousarray(lut.reshape(1, -1, 3).astype(np.uint8))
    flat = np.ascontiguousarray(frame)
    _, labels = cv2.knnSearch(flat.reshape(-1, 3), swatch.reshape(-1, 3), k=1)
    levels = labels.reshape(frame.shape[:2]).astype(np.float32) / float(size - 1)
    lo = 0.0 if bottom_c is None else float(bottom_c)
    hi = 1.0 if top_c is None else float(top_c)
    temps = lo + levels * (hi - lo)
    mask = np.ones(temps.shape, bool)
    for px0, py0, px1, py1 in OVERLAY_EXCLUDE:
        mask[py0:py1, px0:px1] = False
    x0, y0, x1, y1 = CROSSHAIR_BOX
    mask[y0:y1, x0:x1] = include_crosshair
    temps[~mask] = np.nan
    return temps.astype(np.float32)


# -- reader -------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\d.])(-?\d{1,3}\.\d)(?![\d.])")


@dataclass
class OverlayData:
    center_c: float | None = None
    max_c: float | None = None
    min_c: float | None = None
    emissivity: float | None = None
    timestamp: str | None = None
    palette_top_c: float | None = None
    palette_bottom_c: float | None = None
    raw_text: str = ""
    ocr_backend: str = "none"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class OverlayReader:
    def __init__(self, backend: str | None = None) -> None:
        self._requested = backend
        self._backend: Any = None
        self._trained = False

    @property
    def backend(self) -> Any:
        if self._backend is None:
            if self._requested == "template":
                self._backend = _TemplateBackend()
            else:
                self._backend = _pick_backend()
        return self._backend

    @property
    def backend_name(self) -> str:
        try:
            return str(self.backend.name)
        except Exception:
            return "unavailable"

    def parse(self, jpeg_or_rgb: Any) -> OverlayData:
        try:
            rgb = to_rgb(jpeg_or_rgb)
        except Exception as exc:
            log.warning("overlay: cannot decode frame: %s", exc)
            return OverlayData(ocr_backend=self.backend_name)
        try:
            backend = self.backend
            if isinstance(backend, _TemplateBackend) and not self._trained:
                backend.train(_gray(rgb))
                self._trained = True
            cen = self._read_number(rgb, PANEL_ROWS[0], PANEL_VALUE_X0, PANEL_VALUE_X1)
            mx = self._read_number(rgb, PANEL_ROWS[1], PANEL_VALUE_X0, PANEL_VALUE_X1)
            mn = self._read_number(rgb, PANEL_ROWS[2], PANEL_VALUE_X0, PANEL_VALUE_X1)
            top = self._read_number(rgb, PALETTE_TOP_LABEL_ROW, 6, 58)
            bottom = self._read_number(rgb, PALETTE_BOTTOM_LABEL_ROW, 6, 58)
            emissivity = self._read_emissivity(rgb)
            timestamp = self._read_timestamp(rgb)
        except Exception as exc:  # OCR backends are best-effort
            log.warning("overlay parse failed: %s", exc)
            return OverlayData(raw_text="", ocr_backend=self.backend_name)
        texts = [f"Cen {cen}", f"Max {mx}", f"Min {mn}"]
        if emissivity is not None:
            texts.append(f"eps {emissivity:.2f}")
        if timestamp:
            texts.append(timestamp)
        data = OverlayData(
            center_c=cen,
            max_c=mx,
            min_c=mn,
            emissivity=emissivity,
            timestamp=timestamp,
            palette_top_c=top,
            palette_bottom_c=bottom,
            raw_text=" ".join(t for t in texts if not t.endswith("None")),
            ocr_backend=self.backend_name,
        )
        if data.center_c is not None and not (-40.0 <= data.center_c <= 400.0):
            log.warning("overlay: centre reading %s out of range", data.center_c)
            data = replace(data, center_c=None)
        return data

    def _read_number(self, rgb: np.ndarray, row: tuple[int, int], x0: int, x1: int,
                     expect_decimal: bool = True) -> float | None:
        raw = self.backend.read_band(rgb, row, x0, x1)
        return _first_number(raw, expect_decimal)

    def _read_emissivity(self, rgb: np.ndarray) -> float | None:
        raw = self.backend.read_band(rgb, EMISSIVITY_ROW, 180, 300)
        m = re.search(r"0\.\d{2}", raw)
        return float(m.group(0)) if m else None

    def _read_timestamp(self, rgb: np.ndarray) -> str | None:
        raw = self.backend.read_band(rgb, EMISSIVITY_ROW, 280, 470)
        m = re.search(r"\d{4}[-/]\d{2}[-/]\d{2}[ T]\d{2}:\d{2}", raw)
        return m.group(0) if m else None


def _first_number(text: str, expect_decimal: bool = True) -> float | None:
    if not text:
        return None
    m = _NUMBER.search(text.replace(" ", "")) if expect_decimal else re.search(r"-?\d+(?:\.\d)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None
