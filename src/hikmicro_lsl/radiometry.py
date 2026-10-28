"""Radiometry for Xtherm/InfiRay-family 256x192 thermal cores.

Port of ``vendor/HT203U-Thermal/app/src/main/java/io/github/cbird/ht203u/Xtherm.kt``:

* ``parse_meta``  <- ``parseMeta()``  (lines 74-136)
* ``temp_table``  <- ``tempTable()``  (lines 140-180), with ``wvc``/``atmt``
* ``iron_palette`` <- ``ironPalette()`` (lines 183-198)

Vendor constants (lines 15-40) are reproduced verbatim. Every function here is
hardware free and performs no network access at import time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

# --- Vendor constants (Xtherm.kt lines 15-40) -------------------------------
WIDTH = 256
HEIGHT = 192
META_ROWS = 4
FRAME_HEIGHT = HEIGHT + META_ROWS
PIXELS = WIDTH * HEIGHT  # u16 offset of the metadata block
FRAME_U16 = WIDTH * FRAME_HEIGHT
TABLE_SIZE = 16384

ZEROC = 273.15
FPA_OFF = 8617.0
FPA_DIV = 37.682
AMOUNT_PIXELS = WIDTH
CAL_00_OFFSET = 170.0
CAL_00_FPAMUL = 0.0

CMD_CALIBRATE = 0x8000
CMD_RAW_MODE = 0x8004
CMD_SAVE = 0x80FF
CMD_RANGE_NORMAL = 0x8020
CMD_RANGE_HIGH = 0x8021

UVC_GET_CUR = 0x81
UVC_GET_LEN = 0x85


@dataclass
class Meta:
    """Mirror of the Kotlin ``Xtherm.Meta`` data class (lines 41-60).

    ``high_range`` is not part of the vendor struct; it records which range
    command (0x8020/0x8021) was active when the frame was captured so recorded
    files can be decoded with the right correction.
    """

    fpa_average: int
    temp_fpa: float
    temp_shutter: float
    temp_core: float
    max_x: int
    max_y: int
    max_raw: int
    min_x: int
    min_y: int
    min_raw: int
    avg_raw: int
    center_raw: int
    correction: float
    temp_reflected: float
    temp_air: float
    humidity: float
    emissivity: float
    distance: int
    cal00: float
    cal01: float
    cal02: float
    cal03: float
    cal04: float
    cal05: float
    high_range: bool = field(default=False, compare=False)


def _u16(flat: np.ndarray, off: int) -> int:
    """Unsigned 16-bit read at u16 index ``off`` (Kotlin ``u16``)."""
    return int(flat[off]) & 0xFFFF


def _f32(flat: np.ndarray, off: int) -> float:
    """Reinterpret two u16 as a little-endian IEEE-754 float32 (Kotlin ``f32``)."""
    lo = _u16(flat, off)
    hi = _u16(flat, off + 1)
    bits = (lo | (hi << 16)) & 0xFFFFFFFF
    return float(np.frombuffer(np.uint32(bits).tobytes(), dtype="<f4")[0])


def _or_default(value: float, default: float, lo: float, hi: float) -> float:
    """Kotlin ``Double.orDefault``: finite and in [lo, hi], else ``default``."""
    if math.isfinite(value) and lo <= value <= hi:
        return value
    return default


def _as_flat(frame_u16: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(frame_u16).reshape(-1))


def parse_meta(
    frame_u16: np.ndarray,
    meta_offset: int = PIXELS,
    high_range: bool = False,
) -> Meta | None:
    """Parse the metadata block; return ``None`` for junk or non-raw frames.

    All plausibility checks from ``parseMeta()`` are preserved so that plain
    YUYV video frames (which are emitted by the camera when raw mode is off)
    are rejected rather than silently decoded as temperatures.
    """
    flat = _as_flat(frame_u16)
    if flat.size < meta_offset + META_ROWS * WIDTH:
        return None
    m = meta_offset

    max_x = _u16(flat, m + 2)
    max_y = _u16(flat, m + 3)
    max_raw = _u16(flat, m + 4)
    min_x = _u16(flat, m + 5)
    min_y = _u16(flat, m + 6)
    min_raw = _u16(flat, m + 7)
    center_raw = _u16(flat, m + 12)

    if max_x >= WIDTH or max_y >= HEIGHT or min_x >= WIDTH or min_y >= HEIGHT:
        return None
    if min_raw > max_raw or max_raw >= TABLE_SIZE or max_raw == 0:
        return None
    if not (min_raw <= center_raw <= max_raw):
        return None

    temp_fpa = 20.0 - (_u16(flat, m + 1) - FPA_OFF) / FPA_DIV
    temp_shutter = _u16(flat, m + AMOUNT_PIXELS + 1) / 10.0 - ZEROC
    temp_core = _u16(flat, m + AMOUNT_PIXELS + 2) / 10.0 - ZEROC

    if max_raw < 100 or max_raw - min_raw < 16:
        return None
    if not (-40.0 <= temp_shutter <= 150.0) or not (-40.0 <= temp_fpa <= 150.0):
        return None

    user_area = m + AMOUNT_PIXELS + 127
    return Meta(
        fpa_average=_u16(flat, m),
        temp_fpa=temp_fpa,
        temp_shutter=temp_shutter,
        temp_core=temp_core,
        max_x=max_x,
        max_y=max_y,
        max_raw=max_raw,
        min_x=min_x,
        min_y=min_y,
        min_raw=min_raw,
        avg_raw=_u16(flat, m + 8),
        center_raw=center_raw,
        correction=_or_default(_f32(flat, user_area), 0.0, -50.0, 50.0),
        temp_reflected=_or_default(_f32(flat, user_area + 2), 25.0, -50.0, 300.0),
        temp_air=_or_default(_f32(flat, user_area + 4), 25.0, -50.0, 100.0),
        humidity=_or_default(_f32(flat, user_area + 6), 0.45, 0.0, 1.0),
        emissivity=_or_default(_f32(flat, user_area + 8), 0.95, 0.01, 1.0),
        distance=_u16(flat, user_area + 10),
        cal00=float(_u16(flat, m + AMOUNT_PIXELS)),
        cal01=_f32(flat, m + AMOUNT_PIXELS + 3),
        cal02=_f32(flat, m + AMOUNT_PIXELS + 5),
        cal03=_f32(flat, m + AMOUNT_PIXELS + 7),
        cal04=_f32(flat, m + AMOUNT_PIXELS + 9),
        cal05=_f32(flat, m + AMOUNT_PIXELS + 11),
        high_range=high_range,
    )


def wvc(humidity: float, temp_air: float) -> float:
    """Water vapour coefficient (Xtherm.kt lines 140-143)."""
    h1, h2, h3, h4 = 1.5587, 0.06939, -2.7816e-4, 6.8455e-7
    return humidity * math.exp(
        h1 + h2 * temp_air + h3 * temp_air**2 + h4 * temp_air**3
    )


def atmt(humidity: float, temp_air: float, distance: float) -> float:
    """Atmospheric transmittance (Xtherm.kt lines 146-152)."""
    k_atm = 1.9
    nsqd = -math.sqrt(distance)
    sqw = math.sqrt(wvc(humidity, temp_air))
    a1, a2 = 0.006569, 0.01262
    b1, b2 = -0.002276, -0.00667
    return k_atm * math.exp(nsqd * (a1 + b1 * sqw)) + (1.0 - k_atm) * math.exp(
        nsqd * (a2 + b2 * sqw)
    )


def temp_table(meta: Meta, high_range: bool = False) -> np.ndarray:
    """Raw 14-bit value -> degC lookup table, length 16384, float32.

    Vectorised form of ``Xtherm.tempTable()`` (lines 158-180). The Kotlin loop's
    ``sqrt(abs(...))``, ``pow(4)``, ``pow(0.25)`` sequence is reproduced exactly
    in float64 before the final float32 cast.
    """
    if meta.cal01 == 0.0:
        raise ValueError("cal01 is zero; cannot build temperature table")

    dist_adj = (20.0 if meta.distance >= 20 else float(meta.distance)) * 1.0
    atm = atmt(meta.humidity, meta.temp_air, dist_adj)
    numerator_sub = (1.0 - meta.emissivity) * atm * (
        meta.temp_reflected + ZEROC
    ) ** 4 + (1.0 - atm) * (meta.temp_air + ZEROC) ** 4
    denominator = meta.emissivity * atm
    if denominator == 0.0:
        raise ValueError("emissivity * transmittance is zero")

    ts = meta.temp_shutter
    tfpa = meta.temp_fpa
    cal_a = meta.cal02 / (meta.cal01 + meta.cal01)
    cal_b = meta.cal02 * meta.cal02 / (meta.cal01 * meta.cal01 * 4.0)
    cal_c = meta.cal01 * ts**2 + ts * meta.cal02
    cal_d = meta.cal03 * tfpa**2 + meta.cal04 * tfpa + meta.cal05

    cal00_corr = int(CAL_00_OFFSET - tfpa * CAL_00_FPAMUL)
    table_offset = meta.cal00 - (cal00_corr if cal00_corr > 0 else 0)

    corr_m = 1.17 if high_range else 1.0
    corr_b = -40.9 if high_range else 0.0

    i = np.arange(TABLE_SIZE, dtype=np.float64)
    inner = ((i - table_offset) * cal_d + cal_c) / meta.cal01 + cal_b
    arg = np.abs(inner)
    n = np.sqrt(arg)
    n = np.where(np.isnan(n), 0.0, n)

    wtot = (n - cal_a + ZEROC) ** 4
    ttot_base = (wtot - numerator_sub) / denominator
    # Kotlin pow(0.25) of a negative float is NaN; keep NaN rather than masking.
    with np.errstate(invalid="ignore"):
        ttot = np.power(ttot_base, 0.25) - ZEROC
    t = ttot + (dist_adj * 0.85 - 1.125) * (ttot - meta.temp_air) / 100.0 + meta.correction
    table = corr_m * t + corr_b
    return table.astype(np.float32)


def frame_to_celsius(
    frame_u16: np.ndarray,
    meta: Meta,
    high_range: bool | None = None,
) -> np.ndarray:
    """Map a 196x256 raw frame to a 192x256 float32 degC image.

    ``high_range`` defaults to ``meta.high_range`` so recorded frames replay with
    the correction that was active at capture time.
    """
    if high_range is None:
        high_range = meta.high_range
    flat = _as_flat(frame_u16)
    if flat.size < PIXELS:
        raise ValueError(f"frame has {flat.size} u16, need at least {PIXELS}")
    lut = temp_table(meta, high_range=high_range)
    idx = np.minimum(flat[:PIXELS], TABLE_SIZE - 1).astype(np.intp)
    return lut[idx].reshape(HEIGHT, WIDTH)


# Ironbow stops exactly as in Xtherm.ironPalette() (lines 183-187).
_IRON_STOPS = (
    0x000000,
    0x1E0A3C,
    0x6A0A69,
    0xA01E70,
    0xC83C50,
    0xE86828,
    0xF0A830,
    0xF8DC50,
    0xFFFFC8,
    0xFFFFFF,
)


def iron_palette() -> np.ndarray:
    """Return the 256x3 uint8 ironbow palette (Kotlin ``ironPalette``).

    The Kotlin version returns ARGB ints with alpha 0xFF; the alpha channel is
    dropped here since every entry is opaque.
    """
    stops = np.array(_IRON_STOPS, dtype=np.int64)
    red = (stops >> 16) & 0xFF
    green = (stops >> 8) & 0xFF
    blue = stops & 0xFF
    seg = 255.0 / (len(_IRON_STOPS) - 1)
    f = np.arange(256, dtype=np.float64) / seg
    idx = np.minimum(f.astype(np.int64), len(_IRON_STOPS) - 2)
    t = f - idx
    out = np.empty((256, 3), dtype=np.uint8)
    for ch, src in enumerate((red, green, blue)):
        mixed = src[idx] * (1 - t) + src[idx + 1] * t
        out[:, ch] = mixed.astype(np.uint8)  # Kotlin .toInt() truncates
    return out


def meta_to_dict(meta: Meta | None) -> dict:
    """JSON-ready dict for the LSL metadata stream and .jsonl sidecar."""
    if meta is None:
        return {"valid": False}
    payload = {"valid": True}
    payload.update(meta.__dict__)
    return payload


def meta_to_json(meta: Meta | None) -> str:
    return json.dumps(meta_to_dict(meta), separators=(",", ":"), sort_keys=True)


def meta_from_dict(payload: dict, high_range: bool | None = None) -> Meta | None:
    """Rebuild a ``Meta`` from a JSON dict (inverse of ``meta_to_dict``)."""
    if not payload or not payload.get("valid", False):
        return None
    data = {k: v for k, v in payload.items() if k != "valid"}
    if high_range is not None:
        data["high_range"] = high_range
    return Meta(**data)
