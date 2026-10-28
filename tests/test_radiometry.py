"""Hardware-free tests for the radiometry port (no camera, no LSL)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hikmicro_lsl import radiometry as r

# --- Frame synthesis -------------------------------------------------------
# Metadata field offsets follow Xtherm.parseMeta() (m = PIXELS = 49152):
#   m+0 fpaAverage, m+1 fpaCounter, m+2..m+7 max/min x,y,raw, m+8 avgRaw,
#   m+12 centerRaw, m+256 cal00 (u16), m+257 tempShutter, m+258 tempCore,
#   m+259.. float32 cal01..cal05, m+383 user area (correction/emissivity...).

# Mirrors XthermTest.kt syntheticFrame() (vendor test lines 20-46) so the
# golden temperature values from that test can be asserted directly.
FPA_COUNTER = 8165  # -> temp_fpa ~= 31.995 C
EMISSIVITY = 0.95


def _put_f32(flat: np.ndarray, off: int, value: float) -> None:
    bits = int(np.frombuffer(np.float32(value).tobytes(), dtype="<u4")[0])
    flat[off] = bits & 0xFFFF
    flat[off + 1] = bits >> 16


def make_frame(
    *,
    cal00: int = 8000,
    cal01: float = 4.0,
    cal02: float = 100.0,
    cal03: float = 0.01,
    cal04: float = 0.5,
    cal05: float = 800.0,
    fpa_counter: int = FPA_COUNTER,
    shutter_dk: int = 3031,  # 29.95 C
    core_dk: int = 3100,
    max_raw: int = 9000,
    min_raw: int = 7000,
    center_raw: int = 8000,
    emissivity: float = EMISSIVITY,
    distance: int = 1,
    fill: str = "ramp",
) -> np.ndarray:
    """Build a 196x256 uint16 frame with a self-consistent metadata block."""
    m = r.PIXELS
    frame = np.zeros(r.FRAME_U16, dtype=np.uint16)
    if fill == "ramp":
        row = (np.arange(r.WIDTH, dtype=np.uint32) * 3 + 7000) % 16384
        frame[:m] = np.tile(row, r.HEIGHT).astype(np.uint16)
    frame[m + 0] = 15000  # fpa average
    frame[m + 1] = fpa_counter
    frame[m + 2] = 100  # max_x
    frame[m + 3] = 50  # max_y
    frame[m + 4] = max_raw
    frame[m + 5] = 10  # min_x
    frame[m + 6] = 20  # min_y
    frame[m + 7] = min_raw
    frame[m + 8] = (max_raw + min_raw) // 2  # avg_raw
    frame[m + 12] = center_raw
    frame[m + 256] = cal00
    frame[m + 257] = shutter_dk
    frame[m + 258] = core_dk
    _put_f32(frame, m + 259, cal01)
    _put_f32(frame, m + 261, cal02)
    _put_f32(frame, m + 263, cal03)
    _put_f32(frame, m + 265, cal04)
    _put_f32(frame, m + 267, cal05)
    user = m + 256 + 127
    _put_f32(frame, user + 0, 0.0)  # correction
    _put_f32(frame, user + 2, 25.0)  # reflected
    _put_f32(frame, user + 4, 25.0)  # air
    _put_f32(frame, user + 6, 0.45)  # humidity
    _put_f32(frame, user + 8, emissivity)
    frame[user + 10] = distance
    return frame.reshape(r.FRAME_HEIGHT, r.WIDTH)


# --- parse_meta ------------------------------------------------------------
def test_parse_meta_accepts_plausible_frame():
    meta = r.parse_meta(make_frame())
    assert meta is not None
    assert meta.temp_shutter == pytest.approx(29.95, abs=1e-9)
    assert meta.temp_fpa == pytest.approx(31.9951170320, abs=1e-6)
    assert meta.temp_core == pytest.approx(310.0 - r.ZEROC, abs=1e-4)
    assert meta.max_raw == 9000
    assert meta.min_raw == 7000
    assert meta.center_raw == 8000
    assert meta.fpa_average == 15000
    assert meta.cal00 == 8000
    assert meta.cal01 == pytest.approx(4.0)
    assert meta.cal02 == pytest.approx(100.0)
    assert meta.cal03 == pytest.approx(0.01)
    assert meta.cal05 == pytest.approx(800.0)
    assert meta.emissivity == pytest.approx(EMISSIVITY)
    assert meta.humidity == pytest.approx(0.45, abs=1e-6)
    assert meta.temp_air == pytest.approx(25.0, abs=1e-6)
    assert meta.distance == 1
    assert meta.high_range is False


def test_parse_meta_rejects_full_length_video_frame():
    """A 196x256 frame without the metadata block reads as junk temperatures."""
    rng = np.random.default_rng(0)
    video = rng.integers(0, 65535, r.FRAME_U16, dtype=np.uint16)
    assert r.parse_meta(video.reshape(r.FRAME_HEIGHT, r.WIDTH)) is None


def test_parse_meta_rejects_image_only_frame():
    assert r.parse_meta(np.zeros((r.HEIGHT, r.WIDTH), dtype=np.uint16)) is None


def test_parse_meta_rejects_zero_range():
    assert r.parse_meta(make_frame(max_raw=0, min_raw=0, center_raw=0)) is None


def test_parse_meta_rejects_narrow_range():
    assert r.parse_meta(make_frame(max_raw=7010, min_raw=7000, center_raw=7005)) is None


def test_parse_meta_rejects_out_of_bounds_coordinates():
    frame = make_frame()
    frame.reshape(-1)[r.PIXELS + 2] = r.WIDTH  # max_x == WIDTH
    assert r.parse_meta(frame) is None


def test_parse_meta_rejects_implausible_fpa_temperature():
    frame = make_frame()
    frame.reshape(-1)[r.PIXELS + 1] = (8165 - int(37.682 * 300)) & 0xFFFF  # -268 C
    assert r.parse_meta(frame) is None


# --- temp_table ------------------------------------------------------------
def test_temp_table_shape_and_dtype():
    meta = r.parse_meta(make_frame())
    table = r.temp_table(meta)
    assert table.shape == (r.TABLE_SIZE,)
    assert table.dtype == np.float32


def test_temp_table_matches_vendor_golden_values():
    """Values from XthermTest.kt lines 74-78 (the Kotlin reference test)."""
    meta = r.parse_meta(make_frame())
    assert meta is not None
    table = r.temp_table(meta)
    golden = {
        4000: 890.217099,
        7000: 407.748100,
        8000: 184.576217,
        9000: 490.373019,
        12000: 931.326981,
    }
    for idx, expected in golden.items():
        assert float(table[idx]) == pytest.approx(expected, abs=0.02), idx


def test_temp_table_monotonic_in_valid_physics_window():
    """Within the physical branch (n >= cal02/(2*cal01)) temp rises with count.

    The vendor formula applies ``sqrt(abs(...))``; below the InfiRay shutter
    equation's minimum the Kotlin code takes the mirrored branch, which is not
    monotonic. That region lies outside the physically meaningful range
    (indexing it would require cal00 near 2^14), so monotonicity is asserted
    from the table minimum onwards, which is where real sensor counts land.
    """
    meta = r.parse_meta(make_frame())
    table = r.temp_table(meta).astype(np.float64)
    tail = table[int(np.argmin(table)) :]
    assert tail.size > 1000
    assert np.all(np.diff(tail[1:]) >= 0)
    # Real sensors sit far above the mirrored branch.
    assert table[7000] < table[9000] < table[12000]


def test_temp_table_never_saturates_sqrt_argument():
    """The |..| makes every 14-bit table entry finite (no NaN gaps)."""
    meta = r.parse_meta(make_frame())
    assert meta is not None
    assert np.isfinite(r.temp_table(meta)).all()


def test_frame_to_celsius_matches_lut():
    frame = make_frame()
    meta = r.parse_meta(frame)
    temps = r.frame_to_celsius(frame, meta)
    assert temps.shape == (r.HEIGHT, r.WIDTH)
    assert temps.dtype == np.float32
    table = r.temp_table(meta)
    flat = frame.reshape(-1)[: r.PIXELS].astype(np.int64)
    np.testing.assert_array_equal(temps.reshape(-1), table[flat])


def test_high_range_applies_correction_constants():
    frame = make_frame()
    meta = r.parse_meta(frame)
    normal = r.temp_table(meta, high_range=False)
    high = r.temp_table(meta, high_range=True)
    # corrM=1.17, corrB=-40.9 with identical t (only corrM/corrB differ).
    assert np.allclose(high, normal.astype(np.float64) * 1.17 - 40.9, atol=1e-3)


def test_iron_palette():
    palette = r.iron_palette()
    assert palette.shape == (256, 3)
    assert palette.dtype == np.uint8
    assert tuple(palette[0]) == (0, 0, 0)
    assert tuple(palette[255]) == (255, 255, 255)


def test_meta_json_roundtrip():
    meta = r.parse_meta(make_frame())
    payload = r.meta_to_dict(meta)
    restored = r.meta_from_dict(payload)
    assert restored == meta
    decoded = json.loads(r.meta_to_json(meta))
    assert decoded["valid"] is True and decoded["cal01"] == pytest.approx(4.0)
    assert r.meta_to_dict(None) == {"valid": False}


# --- Layout guards ---------------------------------------------------------
def test_metadata_layout_offsets():
    """Guards the offsets documented in docs/FRAME_FORMAT.md."""
    assert r.PIXELS == 49152
    assert r.FRAME_U16 == 50176
    assert r.PIXELS + r.AMOUNT_PIXELS + 127 == 49535  # user area start
    m = r.PIXELS
    frame = make_frame(emissivity=0.62, distance=37)
    meta = r.parse_meta(frame)
    # emissivity at user+8 = m+391, distance at user+10 = m+393
    assert int(np.frombuffer(frame.reshape(-1)[m + 391 : m + 393].tobytes(), "<f4")[0] * 100) == 62
    assert int(frame.reshape(-1)[m + 393]) == 37
    assert meta.emissivity == pytest.approx(0.62)
    assert meta.distance == 37


def test_parse_meta_defaults_out_of_window_floats():
    """Kotlin orDefault: implausible emissivity falls back to 0.95, frame stays valid."""
    frame = make_frame(emissivity=5.0)
    meta = r.parse_meta(frame)
    assert meta is not None
    assert meta.emissivity == pytest.approx(0.95)
