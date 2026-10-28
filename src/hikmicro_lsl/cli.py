"""Command line entry points: ``hikmicro-lsl <subcommand>``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import offline, radiometry


def _cmd_list_devices(args: argparse.Namespace) -> int:
    from .uvc_source import list_video_devices

    devices = list_video_devices()
    if not devices:
        print("no V4L2 capture nodes found under /dev/video*", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps([d.__dict__ for d in devices], indent=2))
        return 0
    for dev in devices:
        print(f"{dev.path}\t{dev.card}\t{dev.bus_info}")
    return 0


def _open_source(args):
    if getattr(args, "backend", "v4l2") == "hcusbsdk":
        from .sdk_source import SdkThermalSource

        source = SdkThermalSource()
        source.open()
        return source

    from .uvc_source import (
        CMD_RANGE_HIGH,
        CMD_RANGE_NORMAL,
        UvcThermalSource,
    )

    # raw_mode=True makes open() send the 0x8004 vendor command before STREAMON
    # and raise NotImplementedError if the host cannot issue it.
    source = UvcThermalSource(device=args.device, raw_mode=args.raw)
    source.open()
    command = CMD_RANGE_HIGH if args.range == "high" else CMD_RANGE_NORMAL
    try:
        source.set_mode(command)
    except NotImplementedError as exc:
        print(f"warning: could not set {args.range} range: {exc}", file=sys.stderr)
    return source


def _make_streamer(args: argparse.Namespace):
    from .lsl_streamer import StreamerConfig, ThermalLslStreamer

    record_dir = None
    if getattr(args, "record", None):
        record_dir = Path(args.record).expanduser()
    full_grid = bool(getattr(args, "full_grid", False))
    roi_only = bool(getattr(args, "roi_only", False))
    no_raw = bool(getattr(args, "no_raw_stream", False))
    if no_raw and full_grid:
        # --no-raw-stream and --full-grid conflict only for the raw grid stream;
        # honour the explicit opt-out.
        push_raw = False
    else:
        push_raw = True
    cfg = StreamerConfig(
        high_range=(True if args.range == "high" else False),
        record_dir=record_dir,
        full_grid=full_grid,
        push_roi=True,
        push_temperature=not roi_only,
        push_raw=push_raw,
        push_meta=True,
    )
    return ThermalLslStreamer(cfg)


def _cmd_stream(args: argparse.Namespace) -> int:
    source = _open_source(args)
    streamer = _make_streamer(args)
    try:
        with source:
            streamer.start()
            if getattr(args, "backend", "v4l2") == "hcusbsdk":
                mode, args.device = "hcusbsdk", "HIKMICRO SDK device 0"
            else:
                mode = "raw" if args.raw else "video"
            print(
                f"streaming {args.device} ({mode}, {args.range} range) on "
                f"{streamer.nominal_srate:.1f} Hz; Ctrl-C to stop",
                file=sys.stderr,
            )
            print(
                "streams: " + ", ".join(sorted(streamer.outlets)),
                file=sys.stderr,
            )
            count = streamer.run(source, duration=args.duration)
            print(f"pushed {count} frames", file=sys.stderr)
            if getattr(args, "backend", "v4l2") == "hcusbsdk" and count == 0:
                print(
                    f"warning: {source.frames_seen} SDK frames received, none with "
                    "radiometric data; put the camera on its live thermal view "
                    "(not a menu) and retry",
                    file=sys.stderr,
                )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        streamer.stop()
    return 0


def _cmd_record(args: argparse.Namespace) -> int:
    args.record = args.output
    return _cmd_stream(args)


def _cmd_replay(args: argparse.Namespace) -> int:
    from .lsl_streamer import StreamerConfig, ThermalLslStreamer
    from .radiometry import FRAME_U16, HEIGHT, WIDTH

    path = Path(args.file).expanduser()
    frames = []
    if path.suffix in (".bin", ".jsonl"):
        for rec in offline.load_raw_dump(path):
            frame_arr = rec.celsius() if rec.centikelvin else rec.frame_u16
            frames.append((frame_arr, rec.timestamp))
    else:
        payload = path.read_bytes()
        if len(payload) % (FRAME_U16 * 2) == 0:
            u16 = np.frombuffer(payload, dtype="<u2").reshape(-1, FRAME_U16)
            for frame in u16:
                frames.append((frame.reshape(HEIGHT, WIDTH), float("nan")))
        elif len(payload) % (radiometry.PIXELS * 2) == 0:
            u16 = np.frombuffer(payload, dtype="<u2").reshape(-1, radiometry.PIXELS)
            for frame in u16:
                frames.append((frame.reshape(HEIGHT, WIDTH), float("nan")))
        else:
            print(
                f"{path}: {len(payload)} bytes is neither a 196x256 nor a 192x256 uint16 dump",
                file=sys.stderr,
            )
            return 2

    if not frames:
        print(f"{path}: no frames", file=sys.stderr)
        return 1

    streamer = ThermalLslStreamer(StreamerConfig(high_range=None))
    streamer.start()
    try:
        import pylsl

        start = pylsl.local_clock()
        delay = (1.0 / args.rate) if args.rate > 0 else 0.0
        for i, (frame, _ts) in enumerate(frames):
            streamer.push(frame, timestamp=start + i * delay)
            if delay:
                target = start + (i + 1) * delay
                now = pylsl.local_clock()
                if target > now:
                    time.sleep(target - now)
        print(f"replayed {len(frames)} frames", file=sys.stderr)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        streamer.stop()
    return 0


def _cmd_extract(args: argparse.Namespace) -> int:
    path = Path(args.file).expanduser()
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if path.suffix in (".bin", ".jsonl"):
        rows = []
        images = 0
        for rec, temps in offline.iter_temperatures(path, high_range=None):
            if temps is None:
                continue
            stem = f"frame-{rec.index:06d}"
            np.save(out_dir / f"{stem}.npy", temps)
            rows.append(
                {
                    "index": rec.index,
                    "t": rec.timestamp,
                    "min_c": float(temps.min()),
                    "max_c": float(temps.max()),
                    "mean_c": float(temps.mean()),
                }
            )
            images += 1
        with open(out_dir / "frames.json", "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2)
        print(f"wrote {images} temperature images to {out_dir}", file=sys.stderr)
        return 0

    if path.suffix.lower() in (".jpg", ".jpeg"):
        temps, meta = offline.extract_radiometric_jpeg(path)
        np.save(out_dir / f"{path.stem}.npy", temps)
        with open(out_dir / f"{path.stem}.json", "w", encoding="utf-8") as fh:
            json.dump(radiometry.meta_to_dict(meta), fh, indent=2)
        if args.png:
            _save_png(temps, out_dir / f"{path.stem}.png")
        print(
            f"{path.name}: min {temps.min():.2f} C, max {temps.max():.2f} C",
            file=sys.stderr,
        )
        return 0

    if path.suffix.lower() in (".csv", ".xlsx"):
        extractor = offline.extract_vendor_csv(path)
        df = extractor.get_sampled_data()
        df.to_csv(out_dir / f"{path.stem}-sampled.csv", index=False)
        print(f"wrote sampled CSV to {out_dir}", file=sys.stderr)
        return 0

    print(f"unsupported input: {path.suffix}", file=sys.stderr)
    return 2


def _save_png(temps: np.ndarray, path: Path) -> None:
    palette = radiometry.iron_palette()
    lo, hi = float(temps.min()), float(temps.max())
    span = hi - lo or 1.0
    idx = np.clip(((temps - lo) / span * 255.0).astype(np.intp), 0, 255)
    rgb = palette[idx]
    try:
        import cv2

        cv2.imwrite(str(path), rgb[:, :, ::-1])
    except ImportError:
        print("warning: opencv not installed, skipping PNG", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hikmicro-lsl",
        description="Stream HIKMICRO Pocket 2 / XTherm thermal frames into LSL.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-devices", help="list V4L2 capture nodes")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_list_devices)

    def add_capture_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", default="/dev/video0")
        p.add_argument(
            "--backend",
            choices=("v4l2", "hcusbsdk"),
            default="v4l2",
            help="frame source: V4L2 UVC (default) or vendor libHCUSBSDK",
        )
        p.add_argument(
            "--raw",
            action="store_true",
            default=True,
            help="request raw 16-bit mode (0x8004); default on",
        )
        p.add_argument(
            "--no-raw",
            dest="raw",
            action="store_false",
            help="do not send the raw-mode command",
        )
        p.add_argument("--range", choices=("normal", "high"), default="normal")
        p.add_argument("--record", metavar="DIR", default=None)
        p.add_argument("--duration", type=float, default=None, metavar="S")
        p.add_argument(
            "--full-grid",
            action="store_true",
            help="also publish the 49152-channel Temperature and RawU16 streams "
            "(default off; heavy for BIDS/TSV consumers)",
        )
        p.add_argument(
            "--roi-only",
            action="store_true",
            help="publish only the small HIKMICRO_Temperature_RoI and Meta streams",
        )
        p.add_argument(
            "--no-raw-stream",
            action="store_true",
            help="do not publish HIKMICRO_RawU16 even with --full-grid",
        )

    p = sub.add_parser("stream", help="stream live frames to LSL")
    add_capture_opts(p)
    p.set_defaults(func=_cmd_stream)

    p = sub.add_parser("record", help="stream and write frames to disk")
    add_capture_opts(p)
    p.add_argument("--output", "-o", required=True, help="output directory")
    p.set_defaults(func=_cmd_record)

    p = sub.add_parser("replay", help="replay a dump into LSL")
    p.add_argument("file")
    p.add_argument("--rate", type=float, default=0.0, help="0 = as fast as possible")
    p.set_defaults(func=_cmd_replay)

    p = sub.add_parser("extract", help="extract temperatures from a file")
    p.add_argument("file")
    p.add_argument("--out", required=True)
    p.add_argument("--png", action="store_true", help="also write an ironbow PNG")
    p.set_defaults(func=_cmd_extract)

    p = sub.add_parser("view", help="serve a live thermal viewer in the browser")
    p.add_argument(
        "--device",
        default=None,
        help="video capture device path or LSL stream name (e.g. lsl:HIKMICRO_Temperature). Default: auto-detect",
    )
    p.add_argument(
        "--backend",
        choices=("v4l2", "hcusbsdk", "lsl"),
        default="lsl",  # Make LSL the default for the viewer
        help="frame source: V4L2 UVC, vendor libHCUSBSDK, or existing LSL stream (default: lsl)",
    )
    p.add_argument("--host", default="127.0.0.1", help="address to bind the web server to")
    p.add_argument("--port", type=int, default=8000, help="port to serve on (default: 8000)")
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser window automatically",
    )
    p.add_argument(
        "--range",
        dest="range_mode",
        choices=("normal", "high"),
        default="normal",
        help="temperature range mode (default: normal)",
    )
    p.add_argument(
        "--no-palette",
        action="store_true",
        help="start with the colour palette disabled (greyscale)",
    )
    p.add_argument(
        "--fullscreen",
        action="store_true",
        help="no-op, kept for compatibility; only relevant to the desktop UI",
    )
    p.set_defaults(func=_cmd_view)

    return parser


def _resolve_device(device_arg: str | None) -> str:
    """Resolve device path: explicit arg if valid, else auto-detect from list_video_devices()."""
    import os

    from .uvc_source import list_video_devices

    # If user explicitly provided a device, validate it exists
    if device_arg is not None:
        if os.path.exists(device_arg):
            return device_arg
        # Device doesn't exist - fall through to auto-detect with a warning
        print(f"warning: device '{device_arg}' not found, auto-detecting...", file=sys.stderr)

    # Auto-detect: prefer HIKMICRO, then first capture node, then /dev/video0, then /dev/video2
    devices = list_video_devices()
    if not devices:
        # Ultimate fallback
        for fallback in ("/dev/video0", "/dev/video2"):
            if os.path.exists(fallback):
                return fallback
        raise RuntimeError("no video capture devices found")

    # Prefer HIKMICRO device (card name contains HIKMICRO or similar)
    for dev in devices:
        if "hikmicro" in dev.card.lower() or "hikmicro" in dev.driver.lower():
            return dev.path

    # Return first valid capture node
    return devices[0].path


def _cmd_view(args: argparse.Namespace) -> int:
    from .webview import run_web_viewer

    backend = getattr(args, "backend", "v4l2")

    if backend == "lsl":
        # If --device was explicitly set (e.g., "lsl:Custom_Stream"), use it,
        # otherwise default to the standard string the factory expects.
        device = args.device if args.device is not None else "lsl"
    elif backend == "hcusbsdk":
        device = "hcusbsdk"
    else:
        # V4L2 fallback
        device = _resolve_device(args.device)

    return run_web_viewer(
        device=device,
        host=args.host,
        port=args.port,
        use_palette=not args.no_palette,
        open_browser=not args.no_browser,
        range_mode=args.range_mode,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, RuntimeError, NotImplementedError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
