"""Hardware-free tests for the streamer/offline glue (fake outlets, no LSL net)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_radiometry import make_frame  # noqa: E402

from hikmicro_lsl import offline  # noqa: E402
from hikmicro_lsl.lsl_streamer import (  # noqa: E402
    META_STREAM,
    PIXELS,
    RAW_STREAM,
    ROI_CHANNELS,
    ROI_LABELS,
    ROI_STREAM,
    TEMP_STREAM,
    TYPE_META,
    StreamerConfig,
    ThermalLslStreamer,
    roi_vector,
)
from hikmicro_lsl.lsl_streamer import (
    _Recorder as DumpRecorder,
)
from hikmicro_lsl.radiometry import HEIGHT, WIDTH, frame_to_celsius, parse_meta  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.calls: dict[str, list] = {}

    def push_sample(self, sample, timestamp=None):
        self.calls.setdefault(self._name, []).append((sample, timestamp))


def _streamer_with_fakes(**cfg) -> tuple[ThermalLslStreamer, dict]:
    """Build a streamer without pylsl; replace outlets with recording fakes."""
    streamer = ThermalLslStreamer(StreamerConfig(**cfg))
    streamer._running = True
    fakes = {}
    for name in (TEMP_STREAM, RAW_STREAM, META_STREAM):
        fake = _Recorder.__new__(_Recorder)
        fake._name = name
        fake.calls = {}
        fakes[name] = fake
    streamer.outlets = fakes
    return streamer, fakes


def test_push_without_timestamp_is_accepted():
    """Regression: push_sample(ts=None) used to raise inside pylsl."""
    streamer, fakes = _streamer_with_fakes(push_temperature=False)
    streamer.push(make_frame(), timestamp=None)
    assert len(fakes[RAW_STREAM].calls[RAW_STREAM]) == 1
    sample, ts = fakes[RAW_STREAM].calls[RAW_STREAM][0]
    assert ts is None
    assert sample.dtype == np.int32 and sample.size == 49152


def test_push_raw_mode_emits_all_streams():
    streamer, fakes = _streamer_with_fakes()
    meta = streamer.push(make_frame(), timestamp=1.5)
    assert meta is not None
    assert len(fakes[TEMP_STREAM].calls[TEMP_STREAM]) == 1
    assert len(fakes[RAW_STREAM].calls[RAW_STREAM]) == 1
    payload = json.loads(fakes[META_STREAM].calls[META_STREAM][0][0][0])
    assert payload["valid"] is True
    assert payload["raw_mode"] is True
    assert payload["cal01"] == pytest.approx(4.0)
    assert payload["frame"] == 1


def test_push_video_mode_skips_temperature():
    rng = np.random.default_rng(3)
    video = rng.integers(0, 65535, (196, 256), dtype=np.uint16)
    streamer, fakes = _streamer_with_fakes()
    meta = streamer.push(video, timestamp=1.0)
    assert meta is None
    assert TEMP_STREAM not in fakes[TEMP_STREAM].calls
    payload = json.loads(fakes[META_STREAM].calls[META_STREAM][0][0][0])
    assert payload["valid"] is False and payload["raw_mode"] is False


def test_record_and_load_raw_dump(tmp_path):
    streamer, _ = _streamer_with_fakes(record_dir=tmp_path)
    streamer._record = DumpRecorder.start(tmp_path)
    frames = [make_frame() for _ in range(3)]
    for i, f in enumerate(frames):
        streamer.push(f, timestamp=float(i))
    streamer._record.close()

    bin_path = next(tmp_path.glob("*.bin"))
    loaded = list(offline.load_raw_dump(bin_path))
    assert len(loaded) == 3
    for rec, src in zip(loaded, frames, strict=True):
        assert rec.frame_u16.shape == (192, 256)
        np.testing.assert_array_equal(rec.frame_u16, src[:192])
        assert rec.meta is not None
        # index is the 1-based streamer frame counter
        assert rec.timestamp == pytest.approx(float(rec.index - 1))
    temps = loaded[0].celsius()
    assert temps.dtype == np.float32 and temps.shape == (192, 256)


def test_load_raw_dump_without_sidecar(tmp_path):
    raw = np.zeros((2, 192, 256), dtype="<u2")
    raw[1, 5, 7] = 1234
    path = tmp_path / "plain.bin"
    raw.tofile(path)
    recs = list(offline.load_raw_dump(path))
    assert len(recs) == 2
    assert int(recs[1].frame_u16[5, 7]) == 1234
    assert recs[0].meta is None
    with pytest.raises(ValueError):
        recs[0].celsius()


def test_extract_radiometric_jpeg_rejects_plain_jpeg(tmp_path):
    path = tmp_path / "plain.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xd9")
    with pytest.raises(ValueError, match="APP4"):
        offline.extract_radiometric_jpeg(path)


def test_extract_radiometric_jpeg_reads_app4(tmp_path):
    import struct

    frame = make_frame()
    image = frame[:192].tobytes()
    meta_block = frame[192:].tobytes()
    body = image + meta_block
    # 100352 bytes exceeds the 65533 byte segment payload limit, so the camera
    # splits the blob over consecutive APP4 segments.
    first = struct.pack(">H", offline.RADIOMETRIC_TAG) + struct.pack(">I", len(body)) + body[:65000]
    chunks = [first] + [body[65000:]]
    segs = b"".join(
        b"\xff\xe4" + struct.pack(">H", len(c) + 2) + c for c in chunks
    )
    jpeg = b"\xff\xd8" + segs + b"\xff\xd9"
    path = tmp_path / "irt.jpg"
    path.write_bytes(jpeg)

    temps, meta = offline.extract_radiometric_jpeg(path)
    assert meta is not None
    assert temps.shape == (192, 256)
    np.testing.assert_allclose(temps, frame_to_celsius(frame, meta), rtol=0, atol=1e-6)



def test_load_raw_dump_tolerates_null_timestamp(tmp_path):
    """Regression: frames pushed without an LSL timestamp record "t": null."""
    streamer, _ = _streamer_with_fakes(record_dir=tmp_path)
    streamer._record = DumpRecorder.start(tmp_path)
    streamer.push(make_frame(), timestamp=None)
    streamer._record.close()
    rec = next(iter(offline.load_raw_dump(next(tmp_path.glob("*.bin")))))
    assert np.isnan(rec.timestamp)
    jsonl = next(tmp_path.glob("*.jsonl")).read_text().strip()
    assert json.loads(jsonl)["t"] is None


def test_roi_vector_length_matches_labels():
    """The RoI sample must be exactly as wide as its label list."""
    frame = make_frame()
    meta = parse_meta(frame)
    temps = frame_to_celsius(frame, meta) if meta is not None else None
    sample = roi_vector(frame.reshape(-1)[:PIXELS], temps, meta)
    assert len(sample) == ROI_CHANNELS == len(ROI_LABELS)
    assert all(isinstance(v, float) for v in sample)


def test_roi_vector_without_meta_is_nan_padded():
    frame = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    sample = roi_vector(frame.reshape(-1)[:PIXELS], None, None)
    assert len(sample) == ROI_CHANNELS
    assert np.isnan(sample[3:]).all()
    assert not np.isnan(sample[:3]).any()


def test_roi_stream_is_pushed_by_default():
    streamer, _ = _streamer_with_fakes()
    streamer.outlets[ROI_STREAM] = _Recorder.__new__(_Recorder)
    streamer.outlets[ROI_STREAM]._name = ROI_STREAM
    streamer.outlets[ROI_STREAM].calls = {}
    streamer.push(make_frame())
    calls = streamer.outlets[ROI_STREAM].calls[ROI_STREAM]
    assert len(calls) == 1
    assert len(calls[0][0]) == ROI_CHANNELS


def test_full_grid_off_by_default():
    assert StreamerConfig().full_grid is False


def test_meta_stream_info_is_irregular_rate():
    """Meta carries JSON blobs, not a sampled signal: nominal_srate must be 0."""
    pylsl = pytest.importorskip("pylsl")
    streamer = ThermalLslStreamer(StreamerConfig())
    info = streamer._info(
        pylsl,
        META_STREAM,
        TYPE_META,
        pylsl.cf_string,
        1,
        pylsl.IRREGULAR_RATE,
        "hikmicro-pocket2-meta",
    )
    assert info.nominal_srate() == 0.0
    assert info.type() == TYPE_META
    assert info.channel_count() == 1
    assert info.channel_format() == pylsl.cf_string


def test_roi_stream_info_labels_and_rate():
    pylsl = pytest.importorskip("pylsl")
    streamer = ThermalLslStreamer(StreamerConfig())
    info = streamer._info(
        pylsl,
        ROI_STREAM,
        "TEMPERATURE_ROI",
        pylsl.cf_float32,
        ROI_CHANNELS,
        25.0,
        "hikmicro-pocket2-roi",
    )
    assert info.channel_count() == len(ROI_LABELS)
    assert info.nominal_srate() == 25.0
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")
    assert labels == list(ROI_LABELS)
