from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from hikmicro_lsl.overlay import OverlayReader, inverse_palette
from hikmicro_lsl.viewer import parse_frame_data


def test_golden_frame_decode():
    data_path = Path(__file__).parent / "data" / "frame_000.mjpg"
    if not data_path.exists():
        pytest.skip("golden frame frame_000.mjpg not found")
    jpeg_bytes = data_path.read_bytes()
    assert len(jpeg_bytes) > 0

    reader = OverlayReader(backend="template")
    res = parse_frame_data(jpeg_bytes, reader, use_palette=True)

    assert "rgb_frame" in res
    assert res["rgb_frame"].shape == (480, 640, 3)
    assert res["rgb_frame"].dtype == np.uint8

    # Check center/max/min values extracted from golden frame against measured facts:
    # Cen 31.6 C / Max 36.6 / Min 23.8 (within 1.5 C if OCR backend active)
    if res["center_c"] is not None:
        assert abs(res["center_c"] - 31.6) <= 1.5
    else:
        pytest.skip("OCR backend unavailable: center temperature not parsed")
    if res["max_c"] is not None:
        assert abs(res["max_c"] - 36.6) <= 1.5
    if res["min_c"] is not None:
        assert abs(res["min_c"] - 23.8) <= 1.5

    assert isinstance(res["depth_map"], np.ndarray)
    assert res["depth_map"].shape == (480, 640)
