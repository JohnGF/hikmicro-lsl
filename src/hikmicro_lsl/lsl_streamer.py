"""Publish thermal frames over Lab Streaming Layer (LSL) and optionally record.

Four streams are created with a shared ``source_id`` so LabRecorder groups them:

* ``HIKMICRO_Temperature_RoI`` - cf_float32, 13 ch, type ``TEMPERATURE_ROI``.
  Small fixed vector per frame (raw + degC extrema, mean, camera temperatures,
  emissivity/distance/humidity). This is the stream a BCI decoder consumes.
* ``HIKMICRO_Temperature`` - cf_float32, 49152 ch, type ``TEMPERATURE``, degC.
  Full 256x192 grid, opt-in (``full_grid=True``). Pushed only when the frame
  metadata parses (raw mode); otherwise absent for that frame.
* ``HIKMICRO_RawU16`` - cf_int32, 49152 ch, type ``RAW``, raw sensor counts,
  opt-in (``full_grid=True``). Always pushed.
* ``HIKMICRO_Meta`` - cf_string, 1 ch, type ``META``, IRREGULAR_RATE, JSON blob
  per frame (fpa/shutter/core temperatures, emissivity, min/max/centre and
  their coordinates).

Stream names keep the ``HIKMICRO_`` prefix (a reader opts in by name); stream
``type`` values are short stable tokens so bridges can use
``resolve_byprop("type", "TEMPERATURE_ROI")``.

Why cf_int32 for raw counts: LSL has no unsigned 16-bit format (``cf_int16``
would wrap values above 32767, and XTherm raw counts routinely exceed that),
and ``cf_float32`` cannot represent every integer above 2^24 exactly though it
is lossless for 14/16-bit counts. ``cf_int32`` is exact, compact and keeps the
raw stream distinguishable from the temperature stream in a viewer.

LSL cannot attach per-frame JSON to a sample, hence the separate meta stream;
it is pushed with the same LSL timestamp as the corresponding raw sample.

``pylsl`` is imported lazily so ``import hikmicro_lsl`` works on machines
without the LSL shared library.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .radiometry import (
    HEIGHT,
    PIXELS,
    WIDTH,
    ZEROC,
    Meta,
    frame_to_celsius,
    meta_to_dict,
    parse_meta,
)

SOURCE_ID = "hikmicro-pocket2"
ROI_STREAM = "HIKMICRO_Temperature_RoI"
TEMP_STREAM = "HIKMICRO_Temperature"
RAW_STREAM = "HIKMICRO_RawU16"
META_STREAM = "HIKMICRO_Meta"
TYPE_ROI = "TEMPERATURE_ROI"
TYPE_TEMPERATURE = "TEMPERATURE"
TYPE_RAW = "RAW"
TYPE_META = "META"
N_CHANNELS = PIXELS

# Fixed RoI vector published as the default (small) stream. Order matters: it is
# the column order of the CSV written by the Nautilus bridge.
ROI_LABELS = (
    "min_raw",
    "max_raw",
    "center_raw",
    "min_c",
    "max_c",
    "center_c",
    "mean_c",
    "fpa_c",
    "core_c",
    "shutter_c",
    "emissivity",
    "distance_m",
    "humidity_pct",
)
ROI_CHANNELS = len(ROI_LABELS)
ROI_UNITS = (
    "count",
    "count",
    "count",
    "degC",
    "degC",
    "degC",
    "degC",
    "degC",
    "degC",
    "degC",
    "ratio",
    "m",
    "%",
)

# Channel labels are generated lazily; 49152 strings is fine but building it on
# every stream start is wasteful, so cache it.
_LABELS: list[str] | None = None


def channel_labels() -> list[str]:
    global _LABELS
    if _LABELS is None:
        _LABELS = [
            f"T_r{row}_c{col}" for row in range(HEIGHT) for col in range(WIDTH)
        ]
    return list(_LABELS)


def roi_vector(
    frame_u16: np.ndarray,
    temps: np.ndarray | None,
    meta: Meta | None,
) -> list[float]:
    """Build the fixed-length RoI sample published on ``HIKMICRO_Temperature_RoI``.

    Column order matches :data:`ROI_LABELS`:
    ``[min_raw, max_raw, center_raw, min_c, max_c, center_c, mean_c, fpa_c,
    core_c, shutter_c, emissivity, distance_m, humidity_pct]``.

    ``temps`` is a ``HEIGHT x WIDTH`` degC array (``None`` when the metadata
    block did not parse, which happens when the camera is not in raw mode); the
    camera-side scalars come from ``meta``. Missing values are NaN so downstream
    consumers never silently interpret a zero as a real temperature.
    """
    image = np.asarray(frame_u16, dtype=np.uint16).reshape(HEIGHT, WIDTH)
    center_r, center_c = HEIGHT // 2, WIDTH // 2
    nan = float("nan")

    values = [
        float(image.min()),
        float(image.max()),
        float(image[center_r, center_c]),
    ]
    if temps is None:
        values += [nan] * 4
    else:
        t = np.asarray(temps, dtype=np.float64).reshape(HEIGHT, WIDTH)
        values += [
            float(t.min()),
            float(t.max()),
            float(t[center_r, center_c]),
            float(t.mean()),
        ]
    if meta is None:
        values += [nan] * 6
    else:
        values += [
            float(meta.temp_fpa),
            float(meta.temp_core),
            float(meta.temp_shutter),
            float(meta.emissivity),
            float(meta.distance),
            float(meta.humidity),
        ]
    return values


class _RateMeter:
    """Rolling nominal sample rate estimate from capture timestamps."""

    def __init__(self, window: int = 128) -> None:
        self._ts: deque[float] = deque(maxlen=window)

    def tick(self, ts: float | None = None) -> None:
        self._ts.append(time.monotonic() if ts is None else ts)

    def rate(self, default: float = 25.0) -> float:
        if len(self._ts) < 2:
            return default
        span = self._ts[-1] - self._ts[0]
        if span <= 0:
            return default
        return (len(self._ts) - 1) / span


@dataclass
class StreamerConfig:
    name_prefix: str = ""
    push_roi: bool = True
    push_temperature: bool = True
    push_raw: bool = True
    push_meta: bool = True
    high_range: bool | None = None
    nominal_srate: float = 0.0
    record_dir: Path | None = None
    chunk_size: int = 1
    # Full 256x192 grid streams (Temperature + RawU16) are 49152 channels each
    # (~1.2 MB/frame at 25 Hz). They are useful for offline work but overwhelm
    # BIDS/TSV writers, so they are opt-in.
    full_grid: bool = False


class ThermalLslStreamer:
    """Owns the LSL outlets, the frame loop and optional on-disk recording.

    ``config.nominal_srate == 0`` means "measure from capture": outlets are
    created on the first frame with a provisional rate and the measured rate is
    reported in the stream desc. Because LSL fixes ``nominal_srate`` at stream
    creation, the rate is estimated from the first few frames in ``start()``.
    """

    def __init__(self, config: StreamerConfig | None = None) -> None:
        self.config = config or StreamerConfig()
        self.outlets: dict[str, object] = {}
        self._rate = _RateMeter()
        self._running = False
        self._frames = 0
        self._record: _Recorder | None = None
        self.nominal_srate = float(self.config.nominal_srate)

    # -- stream creation ---------------------------------------------------
    def _pylsl(self):
        try:
            import pylsl
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "pylsl is not installed. Install the project with "
                "`uv pip install -e .` (pylsl is a declared dependency)."
            ) from exc
        return pylsl

    def _info(
        self,
        pylsl,
        name: str,
        stream_type: str,
        fmt,
        channels: int,
        srate: float,
        source_id: str,
    ):
        info = pylsl.StreamInfo(
            name=name,
            type=stream_type,
            channel_count=channels,
            nominal_srate=srate,
            channel_format=fmt,
            source_id=source_id,
        )
        desc = info.desc()
        desc.append_child_value("manufacturer", "HIKMICRO")
        desc.append_child_value("model", "Pocket 2 / XTherm 256x192")
        desc.append_child_value("geometry", f"{WIDTH}x{HEIGHT}")
        desc.append_child_value("rows", str(HEIGHT))
        desc.append_child_value("cols", str(WIDTH))
        channels_desc = desc.append_child("channels")
        if name == ROI_STREAM:
            for label, unit in zip(ROI_LABELS, ROI_UNITS, strict=True):
                ch = channels_desc.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", unit)
                ch.append_child_value("type", "misc")
        elif name == TEMP_STREAM:
            for label in channel_labels():
                ch = channels_desc.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", "degC")
        elif name == RAW_STREAM:
            for label in channel_labels():
                ch = channels_desc.append_child("channel")
                ch.append_child_value("label", label)
                ch.append_child_value("unit", "count")
        elif name == META_STREAM:
            ch = channels_desc.append_child("channel")
            ch.append_child_value("label", "meta_json")
            ch.append_child_value("unit", "json")
            ch.append_child_value("type", "misc")
        return info

    def _create_outlets(self, srate: float) -> None:
        pylsl = self._pylsl()
        prefix = self.config.name_prefix
        self.outlets = {}
        if self.config.push_roi:
            info = self._info(
                pylsl,
                f"{prefix}{ROI_STREAM}",
                TYPE_ROI,
                pylsl.cf_float32,
                ROI_CHANNELS,
                srate,
                f"{SOURCE_ID}-roi",
            )
            self.outlets[ROI_STREAM] = pylsl.StreamOutlet(info)
        if self.config.full_grid and self.config.push_temperature:
            info = self._info(
                pylsl,
                f"{prefix}{TEMP_STREAM}",
                TYPE_TEMPERATURE,
                pylsl.cf_float32,
                N_CHANNELS,
                srate,
                f"{SOURCE_ID}-temperature",
            )
            self.outlets[TEMP_STREAM] = pylsl.StreamOutlet(info)
        if self.config.full_grid and self.config.push_raw:
            info = self._info(
                pylsl,
                f"{prefix}{RAW_STREAM}",
                TYPE_RAW,
                pylsl.cf_int32,
                N_CHANNELS,
                srate,
                f"{SOURCE_ID}-raw",
            )
            self.outlets[RAW_STREAM] = pylsl.StreamOutlet(info)
        if self.config.push_meta:
            # cf_string has no nominal sample rate; IRREGULAR_RATE (0.0) tells
            # consumers not to assume one and keeps the recorder's srate field
            # honest.
            info = self._info(
                pylsl,
                f"{prefix}{META_STREAM}",
                TYPE_META,
                pylsl.cf_string,
                1,
                pylsl.IRREGULAR_RATE,
                f"{SOURCE_ID}-meta",
            )
            self.outlets[META_STREAM] = pylsl.StreamOutlet(info)

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._running:
            return
        if self.nominal_srate <= 0:
            self.nominal_srate = 25.0  # provisional until frames arrive
        self._create_outlets(self.nominal_srate)
        self._running = True
        if self.config.record_dir is not None:
            self._record = _Recorder.start(Path(self.config.record_dir))

    def stop(self) -> None:
        self._running = False
        if self._record is not None:
            self._record.close()
            self._record = None
        # Drop references so LSL flushes and the outlets leave the resolve map.
        self.outlets = {}

    @property
    def running(self) -> bool:
        return self._running

    def __enter__(self) -> ThermalLslStreamer:
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # -- publishing --------------------------------------------------------
    def push(self, frame_u16: np.ndarray, timestamp: float | None = None) -> Meta | None:
        """Publish one thermal frame; return parsed ``Meta`` or ``None``.

        ``frame_u16`` may be either:
        - a raw ``uint16`` array of shape ``(196, 256)`` or ``(192, 256)``, or
        - a pre-calibrated ``float32`` Celsius array of shape ``(192, 256)``
          (as returned by :class:`SdkThermalSource` via ``0x7fe``).

        ``timestamp`` should be an LSL clock value (``pylsl.local_clock()``);
        the driver monotonic timestamp is close enough as a fallback.
        """
        if not self._running:
            raise RuntimeError("streamer is not running; call start() first")
        arr_in = np.asarray(frame_u16)
        is_celsius_frame = np.issubdtype(arr_in.dtype, np.floating)
        if is_celsius_frame:
            flat_f = arr_in.reshape(-1)
            if flat_f.size < PIXELS:
                raise ValueError(f"frame has {flat_f.size} floats, need {PIXELS}")
            temps = flat_f[:PIXELS].astype(np.float32).reshape(HEIGHT, WIDTH)
            frame = np.clip(
                np.round((temps.astype(np.float64) + ZEROC) * 100.0), 0.0, 65535.0
            ).astype("<u2").reshape(-1)
            meta = None
            high_range = bool(self.config.high_range)
        else:
            frame = arr_in.astype("<u2", copy=False).reshape(-1)
            if frame.size < PIXELS:
                raise ValueError(f"frame has {frame.size} u16, need {PIXELS}")
            meta = parse_meta(frame)
            high_range = (
                meta.high_range if self.config.high_range is None and meta else
                bool(self.config.high_range)
            )
            if meta is not None:
                meta.high_range = high_range
            temps = None
            if meta is not None and (
                self.config.push_temperature or self.config.push_roi
            ):
                temps = frame_to_celsius(frame, meta, high_range=high_range)

        self._rate.tick(timestamp)
        self._frames += 1

        def push_sample(outlet, sample) -> None:
            # pylsl rejects an explicit None timestamp; omit it so LSL stamps.
            if timestamp is None:
                outlet.push_sample(sample)
            else:
                outlet.push_sample(sample, timestamp)

        raw = self.outlets.get(RAW_STREAM)
        if raw is not None:
            push_sample(raw, frame[:PIXELS].astype(np.int32))

        if temps is not None and self.config.push_temperature:
            out = self.outlets.get(TEMP_STREAM)
            if out is not None:
                push_sample(out, temps.astype(np.float32).reshape(-1))

        roi_out = self.outlets.get(ROI_STREAM)
        if roi_out is not None:
            push_sample(roi_out, roi_vector(frame[:PIXELS], temps, meta))

        if self.config.push_meta:
            payload = meta_to_dict(meta)
            if is_celsius_frame and temps is not None:
                payload["valid"] = True
                payload["sdk_p2p"] = True
                payload["min_c"] = round(float(np.nanmin(temps)), 3)
                payload["max_c"] = round(float(np.nanmax(temps)), 3)
                payload["mean_c"] = round(float(np.nanmean(temps)), 3)
                payload["center_c"] = round(
                    float(temps[HEIGHT // 2, WIDTH // 2]), 3
                )
            payload["frame"] = self._frames
            payload["raw_mode"] = meta is not None or is_celsius_frame
            payload["high_range"] = bool(high_range)
            payload["srate_est"] = round(self._rate.rate(), 3)
            mout = self.outlets.get(META_STREAM)
            if mout is not None:
                push_sample(mout, [json.dumps(payload, separators=(",", ":"))])

        if self._record is not None:
            self._record.write(
                self._frames,
                timestamp,
                frame[:PIXELS],
                meta,
                centikelvin=is_celsius_frame,
            )

        return meta

    def run(
        self,
        source,
        duration: float | None = None,
        on_frame=None,
        metrics_every: int = 100,
    ) -> int:
        """Pull frames from ``source`` until ``duration`` elapses.

        ``source`` is a ``UvcThermalSource`` (or anything with
        ``read_frame() -> (frame, ts) | None``). Returns the frame count.
        """
        pylsl = self._pylsl()
        t0 = time.monotonic()
        count = 0
        try:
            while True:
                if duration is not None and time.monotonic() - t0 >= duration:
                    break
                got = source.read_frame()
                if got is None:
                    continue
                frame, _ts = got
                self.push(frame, timestamp=pylsl.local_clock())
                count += 1
                if on_frame is not None:
                    on_frame(frame)
                if metrics_every and count % metrics_every == 0:
                    self.last_metrics = self.metrics(frame)
        finally:
            if self._record is not None:
                self._record.flush()
        return count

    def run_forever(self, source, duration: float | None = None) -> int:
        """Start (idempotently), stream until interrupted, then stop."""
        started_here = not self._running
        if started_here:
            self.start()
        try:
            return self.run(source, duration=duration)
        except KeyboardInterrupt:
            return 0
        finally:
            if started_here:
                self.stop()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def metrics(frame_u16: np.ndarray) -> dict:
        """Cheap per-frame statistics for the meta stream / console."""
        arr = np.asarray(frame_u16)
        if np.issubdtype(arr.dtype, np.floating):
            image = arr.reshape(-1)[:PIXELS].reshape(HEIGHT, WIDTH)
            return {
                "min_c": float(np.nanmin(image)),
                "max_c": float(np.nanmax(image)),
                "mean_c": float(np.nanmean(image)),
            }
        image = arr.astype(np.uint16).reshape(-1)[:PIXELS].reshape(HEIGHT, WIDTH)
        return {
            "raw_min": int(image.min()),
            "raw_max": int(image.max()),
            "raw_mean": float(image.mean()),
        }


class _Recorder:
    """Writes raw frames to ``frames.bin`` plus ``frames.jsonl`` sidecar."""

    def __init__(self, directory: Path, bin_file, jsonl_file) -> None:
        self.directory = directory
        self._bin = bin_file
        self._jsonl = jsonl_file
        self.count = 0

    @classmethod
    def start(cls, directory: Path) -> _Recorder:
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        bin_path = directory / f"frames-{stamp}.bin"
        jsonl_path = directory / f"frames-{stamp}.jsonl"
        return cls(directory, open(bin_path, "wb"), open(jsonl_path, "w", encoding="utf-8"))

    def write(
        self,
        index: int,
        ts,
        frame: np.ndarray,
        meta: Meta | None,
        centikelvin: bool = False,
    ) -> None:
        arr = np.ascontiguousarray(frame, dtype="<u2").reshape(PIXELS)
        self._bin.write(arr.tobytes())
        rec = {
            "index": index,
            "t": ts,
            "offset": self.count * PIXELS * 2,
            "frame_u16": PIXELS,
            "meta": meta_to_dict(meta),
        }
        if centikelvin:
            rec["centikelvin"] = True
        self._jsonl.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self.count += 1

    def flush(self) -> None:
        self._bin.flush()
        self._jsonl.flush()

    def close(self) -> None:
        self.flush()
        self._bin.close()
        self._jsonl.close()
        self.directory.joinpath("frames.latest").write_text(
            self._bin.name, encoding="utf-8"
        )
