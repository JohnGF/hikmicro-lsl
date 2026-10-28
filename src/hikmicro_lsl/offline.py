"""Offline extraction: recorded raw dumps and radiometric still images.

Two independent paths:

1. ``load_raw_dump`` reads the ``frames-*.bin`` + ``frames-*.jsonl`` pairs that
   ``ThermalLslStreamer --record`` writes (see ``lsl_streamer._Recorder``).
2. ``extract_radiometric_jpeg`` reads a still image produced by the Pocket 2.
   HIKMICRO's radiometric JPEGs store the thermal matrix as a 256x196 uint16
   blob in an APP4 (tag 0x4064) segment, which is the path implemented here.
   Vendor CSV exports go through ``extract_vendor_csv`` which delegates to
   meltyfat's ``HikExcelExtractor``.

Nothing here fabricates temperatures: if a file has no radiometric payload the
functions raise ``ValueError`` naming the fallback path.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .radiometry import (
    FRAME_U16,
    HEIGHT,
    PIXELS,
    WIDTH,
    ZEROC,
    Meta,
    frame_to_celsius,
    meta_from_dict,
    parse_meta,
)

RAW_FRAME_BYTES = PIXELS * 2
UVC_FRAME_BYTES = FRAME_U16 * 2


@dataclass
class RecordedFrame:
    index: int
    timestamp: float
    frame_u16: np.ndarray
    meta: Meta | None
    centikelvin: bool = False

    @property
    def raw_mode(self) -> bool:
        return self.meta is not None or self.centikelvin

    def celsius(self, high_range: bool | None = None) -> np.ndarray:
        if self.centikelvin:
            return (
                self.frame_u16.astype(np.float32) * 0.01 - float(ZEROC)
            ).astype(np.float32)
        if self.meta is None:
            raise ValueError(
                f"frame {self.index} has no valid metadata (camera not in raw mode)"
            )
        return frame_to_celsius(self.frame_u16, self.meta, high_range=high_range)


def _iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - corrupt file
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc


def load_raw_dump(path: str | Path) -> Iterator[RecordedFrame]:
    """Yield frames from a ``.bin``/``.jsonl`` pair written by the streamer.

    ``path`` may be either member of the pair; the sibling is derived by
    replacing the suffix. Each record's ``offset``/``frame_u16`` fields are
    honoured so dumps containing metadata rows also work.
    """
    path = Path(path)
    bin_path = path.with_suffix(".bin")
    jsonl_path = path.with_suffix(".jsonl")
    if not bin_path.exists():
        raise FileNotFoundError(f"raw frame dump not found: {bin_path}")
    if not jsonl_path.exists():
        # No sidecar: still allow reading whole 49152-u16 frames.
        size = bin_path.stat().st_size
        if size % RAW_FRAME_BYTES:
            raise ValueError(
                f"{bin_path} is {size} bytes, not a multiple of {RAW_FRAME_BYTES}"
            )
        count = size // RAW_FRAME_BYTES
        with open(bin_path, "rb") as fh:
            for i in range(count):
                data = fh.read(RAW_FRAME_BYTES)
                frame = np.frombuffer(data, dtype="<u2").reshape(HEIGHT, WIDTH).copy()
                yield RecordedFrame(
                    index=i, timestamp=float("nan"), frame_u16=frame, meta=None
                )
        return

    with open(bin_path, "rb") as fh:
        for rec in _iter_jsonl(jsonl_path):
            n = int(rec.get("frame_u16", PIXELS))
            off = int(rec.get("offset", 0))
            fh.seek(off)
            data = fh.read(n * 2)
            if len(data) != n * 2:
                raise ValueError(
                    f"{bin_path}: truncated frame at offset {off} (record {rec.get('index')})"
                )
            frame = np.frombuffer(data, dtype="<u2").copy()
            meta = meta_from_dict(rec.get("meta", {"valid": False}))
            if meta is None and frame.size >= FRAME_U16:
                meta = parse_meta(frame)
            ts = rec.get("t")
            yield RecordedFrame(
                index=int(rec.get("index", 0)),
                timestamp=float(ts) if ts is not None else float("nan"),
                frame_u16=frame.reshape(HEIGHT, WIDTH),
                meta=meta,
                centikelvin=bool(rec.get("centikelvin", False)),
            )


def list_dumps(directory: str | Path) -> list[Path]:
    """Return ``*.bin`` dumps in ``directory``, newest first."""
    return sorted(Path(directory).glob("*.bin"), key=lambda p: p.stat().st_mtime, reverse=True)


# --- Radiometric JPEG -------------------------------------------------------
# HIKMICRO/InfiRay stills embed the thermal matrix in APP4 (marker 0xE4) under
# the Exif-style tag 0x4064, followed by a big-endian uint32 length.
RADIOMETRIC_TAG = 0x4064
APP4_MARKER = 0xE4


def _iter_app4(payload: bytes) -> Iterator[bytes]:
    """Yield every APP4 segment body, in order, as raw bytes."""
    pos = 2 if payload[:2] == b"\xff\xd8" else 0
    n = len(payload)
    while pos + 4 <= n:
        if payload[pos] != 0xFF:
            return
        marker = payload[pos + 1]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            pos += 2
            continue
        seg_len = struct.unpack(">H", payload[pos + 2 : pos + 4])[0]
        seg = payload[pos + 4 : pos + 2 + seg_len]
        if marker == APP4_MARKER:
            yield seg
        if marker == 0xDA:  # start of scan: no more metadata segments
            return
        pos += 2 + seg_len


def _collect_radiometric(payload: bytes) -> bytes | None:
    """Assemble the radiometric blob from one or more APP4 segments.

    The blob is 100352 bytes (image + metadata rows), which exceeds the 65533
    byte maximum segment payload, so cameras split it across consecutive APP4
    segments. The first APP4 segment must start with tag 0x4064 followed by a
    big-endian uint32 declared length; following APP4 segments are treated as
    continuation chunks until that length is satisfied.
    """
    declared: int | None = None
    parts: list[bytes] = []
    for i, seg in enumerate(_iter_app4(payload)):
        if i == 0:
            if len(seg) < 6 or struct.unpack(">H", seg[:2])[0] != RADIOMETRIC_TAG:
                return None
            declared = struct.unpack(">I", seg[2:6])[0]
            if declared > UVC_FRAME_BYTES:
                return None
            parts.append(seg[6:])
        elif declared is None:
            return None
        else:
            parts.append(seg)
        if declared is not None and sum(len(p) for p in parts) >= declared:
            break
    if declared is None:
        return None
    joined = b"".join(parts)
    return joined[:declared]


def extract_radiometric_jpeg(path: str | Path) -> tuple[np.ndarray, Meta | None]:
    """Return ``(temperature_c, meta)`` for a HIKMICRO radiometric JPEG.

    Raises ``ValueError`` when the file has no APP4 radiometric segment. In
    that case export the vendor CSV (HIKMICRO Analyzer / PC software) and use
    ``extract_vendor_csv`` (meltyfat ``HikExcelExtractor``) instead; decoding a
    plain JPEG cannot recover temperatures.
    """
    path = Path(path)
    payload = path.read_bytes()
    if payload[:2] != b"\xff\xd8":
        raise ValueError(f"{path} is not a JPEG file")
    blob = _collect_radiometric(payload)
    if blob is None:
        raise ValueError(
            f"{path} has no APP4/0x{RADIOMETRIC_TAG:04x} radiometric segment; "
            "this is a plain JPEG with no temperature data. Fallback: export the "
            "vendor CSV from HIKMICRO Analyzer and use extract_vendor_csv() "
            "(meltyfat HikExcelExtractor)."
        )
    if len(blob) < PIXELS * 2:
        raise ValueError(
            f"{path}: radiometric payload is {len(blob)} bytes, need >= {PIXELS * 2}"
        )
    values = np.frombuffer(blob[: PIXELS * 2], dtype="<u2").copy()
    if len(blob) >= UVC_FRAME_BYTES:
        meta_block = blob[PIXELS * 2 : UVC_FRAME_BYTES]
        full = np.concatenate([values, np.frombuffer(meta_block, dtype="<u2")])
        meta = parse_meta(full)
    else:
        full = values
        meta = None
    if meta is None:
        raise ValueError(
            f"{path}: radiometric payload present but no calibration metadata; "
            "cannot convert to degC. Fallback: vendor CSV via extract_vendor_csv()."
        )
    return frame_to_celsius(full, meta), meta


# --- Vendor CSV (meltyfat bridge) ------------------------------------------
def _ensure_meltyfat(vendor_path: str | Path | None = None) -> None:
    """Put ``vendor/meltyfat/src`` (or an explicit path) on ``sys.path``."""
    import sys

    candidates: list[Path] = []
    if vendor_path is not None:
        candidates.append(Path(vendor_path))
    repo = Path(__file__).resolve().parents[2]
    candidates.append(repo / "vendor" / "meltyfat" / "src")
    candidates.append(repo / "vendor" / "meltyfat")
    for cand in candidates:
        if (cand / "meltyfat").is_dir() and str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
            return


def extract_vendor_csv(csv_path: str | Path, vendor_path: str | Path | None = None):
    """Delegate to meltyfat's ``HikExcelExtractor`` (requires pandas + vendor submodule)."""
    _ensure_meltyfat(vendor_path)
    try:
        from meltyfat.csvextractor import HikExcelExtractor
    except ImportError as exc:
        raise RuntimeError(
            "meltyfat is not importable. Initialise the submodule "
            "(`git submodule update --init vendor/meltyfat`) and install it with "
            "`uv pip install -e vendor/meltyfat`."
        ) from exc
    return HikExcelExtractor(str(csv_path))


def _default_detector():
    detector_cls, _, _ = _load_meltyfat_classes()
    return detector_cls()


def _load_meltyfat_classes(vendor_path: str | Path | None = None):
    _ensure_meltyfat(vendor_path)
    try:
        from meltyfat.wellanalyzer import WellAnalyzer
        from meltyfat.welldetector import WellDetector
    except ImportError as exc:
        raise RuntimeError(
            "meltyfat WellDetector/WellAnalyzer unavailable (needs torch, "
            "ultralytics, opencv and the vendor models). Initialise and install "
            "vendor/meltyfat, then retry."
        ) from exc
    return WellDetector, WellAnalyzer, None


def extract_wells(
    frames,
    detector=None,
    analyzer=None,
    vendor_path: str | Path | None = None,
    method: str = "hough",
    reference_size: tuple[int, int] = (640, 480),
):
    """Detect a 96-well plate and read each well's temperature per frame.

    ``frames`` are 192x256 float32 degC images (``frame_to_celsius`` output).
    Detection runs once on the first frame and is reused for the rest, because
    a plate does not move between frames.

    Delegates to meltyfat ``WellDetector`` + ``WellAnalyzer``:
    ``set_image`` -> ``detect_YOLOv8``/``detect_HoughCircles`` ->
    ``WellAnalyzer.set_detected_dict`` -> ``map_well_ids`` ->
    ``get_sensor_temp``. ``reference_size`` describes the image the detector
    sees; the 256x192 temperature image is upscaled to it first so the
    detector's pixel-scale assumptions hold.

    Returns a list (one per frame) of ``{well_id: temperature_c}``.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dep
        raise RuntimeError(
            "extract_wells needs opencv (`uv pip install opencv-python-headless`)"
        ) from exc

    WellDetector, WellAnalyzer, _ = _load_meltyfat_classes(vendor_path)
    det = detector if detector is not None else WellDetector()
    ana = analyzer if analyzer is not None else WellAnalyzer()

    results: list[dict[str, float]] = []
    mapped = None
    for frame in frames:
        img = np.asarray(frame, dtype=np.float32)
        norm = img - float(img.min())
        peak = float(norm.max()) or 1.0
        img8 = (norm / peak * 255.0).astype(np.uint8)
        ref = cv2.resize(
            img8, (reference_size[0], reference_size[1]), interpolation=cv2.INTER_CUBIC
        )
        ana.set_image(ref)

        if mapped is None:
            det.set_image(ref)
            if method == "yolo":
                wells = det.detect_YOLOv8()
            elif method == "hough":
                wells = det.detect_HoughCircles()
            else:
                raise ValueError(f"unknown detection method: {method}")
            if not wells:
                raise RuntimeError(
                    "well detection found no wells; adjust the detector or pass "
                    "a pre-fitted detector instance"
                )
            ana.set_detected_dict(wells)
            mapped = ana.map_well_ids()

        row: dict[str, float] = {}
        sensor_df = None
        for well in mapped:
            x, y = well["well_center"]
            if sensor_df is None:
                import pandas as pd

                sensor_df = pd.DataFrame(img)
            row[well["well_id"]] = float(WellAnalyzer.get_sensor_temp(sensor_df, x, y))
        results.append(row)
    return results



def iter_temperatures(path: str | Path, high_range: bool | None = None):
    """Convenience generator: ``(frame, degC image | None)`` from a dump."""
    for rec in load_raw_dump(path):
        if not rec.raw_mode:
            yield rec, None
        else:
            yield rec, rec.celsius(high_range=high_range)
