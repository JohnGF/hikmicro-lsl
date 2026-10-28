"""Tests for the browser MJPEG viewer: no camera, no Qt, no fixed port."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from hikmicro_lsl import radiometry, webview


class _StubSource:
    """Synthetic 196x256 uint16 gradient source."""

    def __init__(self, period: float = 0.02) -> None:
        self.period = period
        self.modes: list[int] = []
        self.closed = False
        self._n = 0
        self._grid = (
            np.linspace(0, 4000, radiometry.WIDTH, dtype=np.uint16)[None, :]
            .repeat(radiometry.HEIGHT, axis=0)
            .copy()
        )

    def open(self) -> None:
        return None

    def read_frame(self) -> tuple[np.ndarray, float] | None:
        time.sleep(self.period)
        self._n += 1
        offset = np.uint16((self._n % 50) * 10)
        grid = self._grid + offset
        return grid.reshape(radiometry.HEIGHT, radiometry.WIDTH), time.monotonic()

    def set_mode(self, command: int) -> None:
        self.modes.append(command)

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _StubSource:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def test_run_web_viewer_is_callable() -> None:
    assert callable(webview.run_web_viewer)


def test_frame_holder_keeps_only_latest() -> None:
    holder = webview.FrameHolder()
    holder.publish(b"first")
    first_seq = holder.sequence()
    holder.publish(b"second")
    holder.publish(b"third")
    assert holder.sequence() == first_seq + 2

    frame, seq = holder.wait(first_seq, timeout=0.5)
    assert frame == b"third"
    assert seq == first_seq + 2
    assert holder.current() == b"third"


def test_control_merge_clamps_and_validates() -> None:
    control = webview.ControlState(scale_min=20.0, scale_max=40.0)
    merged = control.apply({"scale_min": 55.0, "scale_max": 10.0})
    assert merged["scale_min"] == 10.0
    assert merged["scale_max"] == 55.0
    assert control.snapshot()["scale_min"] == 10.0

    with pytest.raises(ValueError):
        control.apply({"range_mode": "turbo"})
    with pytest.raises(ValueError):
        control.apply({"scale_min": "hot"})
    with pytest.raises(ValueError):
        control.apply({"palette": "yes"})

    merged = control.apply({"range_mode": "high", "palette": False})
    assert merged["range_mode"] == "high"
    assert merged["palette"] is False
    assert control.take_range_change() == "high"
    assert control.take_range_change() is None


def _get(url: str, timeout: float = 2.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - local test URL
        return resp.status, resp.read()


def test_http_routes_serve_page_state_and_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _StubSource(period=0.02)
    monkeypatch.setattr(webview, "_default_source_factory", lambda device: stub)

    holder = webview.FrameHolder()
    control = webview.ControlState()
    capture = webview.WebCapture(device="/dev/null", holder=holder, control=control)
    server = webview.make_server("127.0.0.1", 0, holder, control, capture)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"

    capture.start()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 3.0
        while holder.current() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert holder.current() is not None, "capture thread produced no frame"

        status, body = _get(base + "/")
        assert status == 200
        assert b"/stream" in body and b"<canvas" in body

        status, body = _get(base + "/state")
        assert status == 200
        state = json.loads(body)
        for key in (
            "center_c",
            "min_c",
            "max_c",
            "palette_top_c",
            "palette_bottom_c",
            "fps",
            "color_scale_min",
            "color_scale_max",
            "range_mode",
            "frame_count",
            "history",
        ):
            assert key in state
        assert state["frame_count"] > 0
        assert isinstance(state["history"], list)

        status, body = _get(base + "/frame.jpg")
        assert status == 200
        assert len(body) > 100
        assert body[:2] == b"\xff\xd8"

        req = urllib.request.Request(
            base + "/control",
            data=json.dumps({"scale_min": 5, "scale_max": 60}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2.0) as resp:  # noqa: S310 - local
            assert resp.status == 200
            updated = json.loads(resp.read())
        assert updated["color_scale_min"] == 5.0
        assert updated["color_scale_max"] == 60.0

        req = urllib.request.Request(
            base + "/control",
            data=json.dumps({"range_mode": "ludicrous"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(req, timeout=2.0)  # noqa: S310 - local
        assert err.value.code == 400
    finally:
        webview.close_server(server, capture)
        thread.join(timeout=2.0)
    assert stub.closed


class _FailingThermalSource:
    """Thermal source whose open() rejects the node, as a non-XTherm camera does."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True
        raise self.exc

    def read_frame(self) -> None:  # pragma: no cover - thermal path must not be used
        raise AssertionError("thermal read_frame must not be called after fallback")

    def close(self) -> None:
        self.closed = True


class _FakeVideoCapture:
    """Minimal cv2.VideoCapture stand-in that yields synthetic 640x480 BGR frames."""

    instances: list[_FakeVideoCapture] = []

    def __init__(self, device: str, backend: int | None = None) -> None:
        self.device = device
        self.backend = backend
        self.settings: list[tuple[int, float]] = []
        self.released = False
        self._frame = np.zeros((webview.VIDEO_HEIGHT, webview.VIDEO_WIDTH, 3), dtype=np.uint8)
        self._frame[:, :, 1] = 96
        type(self).instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return True

    def set(self, prop: int, value: float) -> bool:
        self.settings.append((prop, value))
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        if self.released:
            return False, np.zeros((1, 1, 3), dtype=np.uint8)
        time.sleep(0.01)
        return True, self._frame.copy()

    def release(self) -> None:
        self.released = True


_EXPECTED_STATE_KEYS = {
    "mode",
    "center_c",
    "min_c",
    "max_c",
    "palette_top_c",
    "palette_bottom_c",
    "fps",
    "color_scale_min",
    "color_scale_max",
    "range_mode",
    "palette",
    "frame_count",
    "history",
}

_TEMPERATURE_KEYS = ("center_c", "min_c", "max_c", "palette_top_c", "palette_bottom_c")


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


@pytest.fixture
def fake_video_capture(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVideoCapture]:
    _FakeVideoCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", _FakeVideoCapture)
    return _FakeVideoCapture


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("/dev/video9: VIDIOC_S_FMT(256x196 YUYV) failed: Invalid argument"),
        NotImplementedError("/dev/video9: cannot set raw mode (vendor command 0x0001)"),
    ],
)
def test_open_failure_falls_back_to_video(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_capture: type[_FakeVideoCapture],
    exc: Exception,
) -> None:
    stub = _FailingThermalSource(exc)
    monkeypatch.setattr(webview, "_default_source_factory", lambda device: stub)
    holder = webview.FrameHolder()
    control = webview.ControlState()
    capture = webview.WebCapture(device="/dev/video9", holder=holder, control=control)

    capture.start()
    try:
        assert _wait_for(lambda: holder.sequence() > 0)
        assert _wait_for(lambda: capture.state()["frame_count"] > 1)
        state = capture.state()
        assert capture.error is None
        assert state["mode"] == "video"
        assert capture.mode == "video"
        assert set(state) == _EXPECTED_STATE_KEYS
        assert state["history"] == []
        for key in _TEMPERATURE_KEYS:
            assert state[key] is None
        assert isinstance(state["fps"], float)
        assert isinstance(state["frame_count"], int)
        frame = holder.current()
        assert frame is not None
        assert frame[:2] == b"\xff\xd8"
        assert stub.opened is True
        assert stub.closed is False
        assert len(fake_video_capture.instances) == 1
        used = fake_video_capture.instances[0]
        assert used.device == "/dev/video9"
        props = [prop for prop, _ in used.settings]
        assert cv2.CAP_PROP_FOURCC in props
        assert cv2.CAP_PROP_FRAME_WIDTH in props
        assert cv2.CAP_PROP_FRAME_HEIGHT in props
        assert (cv2.CAP_PROP_FRAME_WIDTH, float(webview.VIDEO_WIDTH)) in used.settings
        assert (cv2.CAP_PROP_FRAME_HEIGHT, float(webview.VIDEO_HEIGHT)) in used.settings
    finally:
        capture.stop()
        capture.join(timeout=3.0)
    assert not capture.is_alive()
    assert fake_video_capture.instances[0].released is True


def test_thermal_source_never_constructs_video_capture(
    monkeypatch: pytest.MonkeyPatch, fake_video_capture: type[_FakeVideoCapture]
) -> None:
    stub = _StubSource(period=0.01)
    monkeypatch.setattr(webview, "_default_source_factory", lambda device: stub)
    holder = webview.FrameHolder()
    control = webview.ControlState()
    capture = webview.WebCapture(device="/dev/null", holder=holder, control=control)

    capture.start()
    try:
        assert _wait_for(lambda: capture.state()["frame_count"] > 0)
        state = capture.state()
        assert capture.error is None
        assert state["mode"] == "thermal"
        assert capture.mode == "thermal"
        assert set(state) == _EXPECTED_STATE_KEYS
        assert state["center_c"] is not None
        assert fake_video_capture.instances == []
    finally:
        capture.stop()
        capture.join(timeout=3.0)
    assert stub.closed


def test_page_shows_mode_and_suppresses_nan() -> None:
    assert 'id="mode"' in webview._PAGE
    assert "Number.isFinite" in webview._PAGE
    assert "s.mode" in webview._PAGE


class _BrokenVideoCapture(_FakeVideoCapture):
    """VideoCapture that never opens, used to check the error path."""

    instances: list[_BrokenVideoCapture] = []

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the cv2 API
        return False


def test_video_failure_reports_error_and_no_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _BrokenVideoCapture.instances = []
    monkeypatch.setattr(cv2, "VideoCapture", _BrokenVideoCapture)
    stub = _FailingThermalSource(
        RuntimeError("/dev/video9: VIDIOC_S_FMT(256x196 YUYV) failed: Invalid argument")
    )
    monkeypatch.setattr(webview, "_default_source_factory", lambda device: stub)
    holder = webview.FrameHolder()
    control = webview.ControlState()
    capture = webview.WebCapture(device="/dev/video9", holder=holder, control=control)

    capture.start()
    try:
        assert _wait_for(lambda: capture.error is not None)
        assert "video" in capture.error
        assert holder.current() is None
        assert capture.state()["mode"] == "video"
    finally:
        capture.stop()
        capture.join(timeout=3.0)
    assert _BrokenVideoCapture.instances[0].released is True
