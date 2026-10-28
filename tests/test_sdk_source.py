"""Tests for SdkThermalSource (0x7fe P2P radiometric capture) and float32 LSL streaming."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hikmicro_lsl import offline
from hikmicro_lsl.lsl_streamer import (
    META_STREAM,
    RAW_STREAM,
    ROI_CHANNELS,
    ROI_STREAM,
    TEMP_STREAM,
    StreamerConfig,
    ThermalLslStreamer,
)
from hikmicro_lsl.lsl_streamer import _Recorder as DumpRecorder
from hikmicro_lsl.radiometry import HEIGHT, PIXELS, WIDTH, ZEROC
from hikmicro_lsl.sdk_source import SdkThermalSource, decode_appenddata
from hikmicro_lsl.webview import _to_celsius


class _FakeOutlet:
    def __init__(self, name: str) -> None:
        self.name = name
        self.samples: list[tuple] = []

    def push_sample(self, sample, timestamp=None) -> None:
        self.samples.append((sample, timestamp))


class _FakeSdk:
    def __init__(self, temps: np.ndarray, jpeg: bytes = b"\xff\xd8\xff\xd9") -> None:
        self.temps = np.asarray(temps, dtype="<f4").reshape(HEIGHT, WIDTH)
        self.jpeg = jpeg
        self.logged_in = False
        self.cleaned_up = False

    def init(self) -> bool:
        return True

    def get_device_count(self) -> int:
        return 1

    def login(self, index: int = 0) -> int:
        self.logged_in = True
        return 0

    def get_jpeg_with_appenddata(self, login_id: int):
        assert login_id == 0
        return self.jpeg, self.temps.tobytes(), 0x50

    def logout(self, login_id: int) -> None:
        self.logged_in = False

    def cleanup(self) -> bool:
        self.cleaned_up = True
        return True


def test_decode_appenddata_float32():
    expected = np.linspace(20.0, 36.5, PIXELS, dtype="<f4").reshape(HEIGHT, WIDTH)
    decoded = decode_appenddata(expected.tobytes())
    assert decoded is not None
    assert decoded.dtype == np.float32
    assert decoded.shape == (HEIGHT, WIDTH)
    np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-6)


def test_decode_appenddata_centikelvin_u16():
    expected_c = np.full((HEIGHT, WIDTH), 30.25, dtype=np.float32)
    ck = np.round((expected_c + ZEROC) * 100.0).astype("<u2")
    decoded = decode_appenddata(ck.tobytes())
    assert decoded is not None
    assert decoded.dtype == np.float32
    assert decoded.shape == (HEIGHT, WIDTH)
    np.testing.assert_allclose(decoded, expected_c, rtol=0, atol=0.01)


def test_decode_appenddata_rejects_unknown_size():
    assert decode_appenddata(b"too short") is None


def test_sdk_thermal_source_polls_p2p():
    expected = np.linspace(21.0, 35.0, PIXELS, dtype="<f4").reshape(HEIGHT, WIDTH)
    fake_sdk = _FakeSdk(expected)
    with SdkThermalSource(sdk=fake_sdk) as src:
        got = src.read_frame()
        assert got is not None
        frame, ts = got
        assert isinstance(ts, float)
        assert frame.shape == (HEIGHT, WIDTH)
        assert frame.dtype == np.float32
        np.testing.assert_allclose(frame, expected)
        assert src.frames_seen == 1
        assert src.frames_radiometric == 1
        assert src.last_preview == b"\xff\xd8\xff\xd9"


def test_streamer_push_float32_celsius_and_record(tmp_path: Path):
    temps = np.linspace(20.2, 36.1, PIXELS, dtype=np.float32).reshape(HEIGHT, WIDTH)
    streamer = ThermalLslStreamer(StreamerConfig(full_grid=True, record_dir=tmp_path))
    streamer._running = True
    streamer._record = DumpRecorder.start(tmp_path)
    outlets = {
        name: _FakeOutlet(name)
        for name in (ROI_STREAM, TEMP_STREAM, RAW_STREAM, META_STREAM)
    }
    streamer.outlets = outlets

    streamer.push(temps, timestamp=12.5)
    streamer._record.close()

    # Check ROI stream
    assert len(outlets[ROI_STREAM].samples) == 1
    roi, ts = outlets[ROI_STREAM].samples[0]
    assert ts == 12.5
    assert len(roi) == ROI_CHANNELS
    center_c = float(temps[HEIGHT // 2, WIDTH // 2])
    assert roi[3] == pytest.approx(float(temps.min()), abs=1e-4)
    assert roi[4] == pytest.approx(float(temps.max()), abs=1e-4)
    assert roi[5] == pytest.approx(center_c, abs=1e-4)
    assert roi[6] == pytest.approx(float(temps.mean()), abs=1e-4)

    # Check full-grid Temperature stream
    assert len(outlets[TEMP_STREAM].samples) == 1
    temp_vec, _ = outlets[TEMP_STREAM].samples[0]
    assert temp_vec.shape == (PIXELS,)
    np.testing.assert_allclose(temp_vec.reshape(HEIGHT, WIDTH), temps)

    # Check Meta stream
    meta_json, _ = outlets[META_STREAM].samples[0]
    meta_payload = json.loads(meta_json[0])
    assert meta_payload["valid"] is True
    assert meta_payload["raw_mode"] is True
    assert meta_payload["sdk_p2p"] is True
    assert meta_payload["min_c"] == pytest.approx(20.2, abs=1e-2)
    assert meta_payload["max_c"] == pytest.approx(36.1, abs=1e-2)

    # Check recorded dump round-trip
    bin_path = next(tmp_path.glob("*.bin"))
    loaded = list(offline.load_raw_dump(bin_path))
    assert len(loaded) == 1
    assert loaded[0].raw_mode is True
    assert loaded[0].centikelvin is True
    recovered = loaded[0].celsius()
    np.testing.assert_allclose(recovered, temps, rtol=0, atol=0.01)

    iter_items = list(offline.iter_temperatures(bin_path))
    assert len(iter_items) == 1
    assert iter_items[0][1] is not None
    np.testing.assert_allclose(iter_items[0][1], temps, rtol=0, atol=0.01)


def test_webview_to_celsius_accepts_float32():
    temps = np.full((HEIGHT, WIDTH), 31.5, dtype=np.float32)
    out = _to_celsius(temps, with_meta=True, high_range=False)
    assert out is not None
    assert out.shape == (HEIGHT, WIDTH)
    np.testing.assert_allclose(out, temps)
