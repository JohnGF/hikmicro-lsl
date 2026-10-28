#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pylsl>=1.16",
#     "numpy>=1.26",
# ]
# ///
"""
HIKMICRO Thermal (RoI) to Lab Streaming Layer (LSL) Bridge
==========================================================

Launches the ``hikmicro_lsl`` streamer against a V4L2 thermal camera and
publishes the small ``HIKMICRO_Temperature_RoI`` vector (13 channels) plus the
``HIKMICRO_Meta`` JSON stream. The 49152-channel grid streams are opt-in.

This is the standalone HIKMICRO-side launcher. The Nautilus-side wrapper lives
at ``nautilus_bci/scripts/bridges/hikmicro_thermal_bridge.py``.

Runs either with ``uv run --script`` (deploys pylsl + numpy into an ephemeral
env) or from a venv that already has ``hikmicro_lsl`` installed.

Usage:
    uv run --script scripts/hikmicro_lsl_bridge.py --device /dev/video0
    uv run --script scripts/hikmicro_lsl_bridge.py --device /dev/video0 --duration 30
    uv run --script scripts/hikmicro_lsl_bridge.py --device /dev/video0 --full-grid
    uv run --script scripts/hikmicro_lsl_bridge.py --list-devices
"""
import argparse
import os
import sys

_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def discover_streams(timeout=2.0):
    """Return ``(name, type, channels, srate)`` tuples for active LSL streams."""
    try:
        from pylsl import resolve_streams
    except ImportError:
        return []
    return [
        (s.name(), s.type(), s.channel_count(), s.nominal_srate())
        for s in resolve_streams(wait_time=timeout)
    ]


def print_streams(streams):
    if not streams:
        print("[!] No LSL streams visible yet (camera may still be starting).")
        return
    print("\n[+] Discovered LSL streams:")
    for name, stype, channels, srate in streams:
        print(f"    - '{name}' | type '{stype}' | {channels} ch | {srate:.1f} Hz")


def streamer_args(args):
    """Translate bridge flags into ``hikmicro-lsl stream`` arguments."""
    cmd = [args.device, "--range", args.range]
    if args.duration is not None:
        cmd += ["--duration", str(args.duration)]
    if args.full_grid:
        cmd += ["--full-grid"]
    if args.no_raw_stream:
        cmd += ["--no-raw-stream"]
    return cmd


def run_bridge(args):
    from hikmicro_lsl.lsl_streamer import (
        META_STREAM,
        ROI_CHANNELS,
        ROI_LABELS,
        ROI_STREAM,
        StreamerConfig,
        ThermalLslStreamer,
    )
    from hikmicro_lsl.uvc_source import (
        CMD_RANGE_HIGH,
        CMD_RANGE_NORMAL,
        UvcThermalSource,
    )

    source = UvcThermalSource(device=args.device, raw_mode=True)
    try:
        source.open()
    except (OSError, RuntimeError, NotImplementedError) as exc:
        print(f"[-] Could not open thermal camera {args.device}: {exc}")
        return 1

    command = CMD_RANGE_HIGH if args.range == "high" else CMD_RANGE_NORMAL
    try:
        source.set_mode(command)
    except NotImplementedError as exc:
        print(f"[!] Could not set {args.range} range: {exc}")

    config = StreamerConfig(
        high_range=(args.range == "high"),
        full_grid=args.full_grid,
        push_roi=True,
        push_raw=not args.no_raw_stream,
    )
    streamer = ThermalLslStreamer(config)

    print("=" * 70)
    print(" HIKMICRO Thermal -> LSL Bridge ".center(70, "="))
    print("=" * 70)
    print(f"Device: {args.device} | range: {args.range} | fps hint: {args.fps:g}")
    print(f"RoI stream: '{ROI_STREAM}' ({ROI_CHANNELS} ch: {', '.join(ROI_LABELS)})")
    print(f"Meta stream: '{META_STREAM}' (JSON per frame, IRREGULAR_RATE)")
    if args.full_grid:
        print("Full grid: ON (49152-channel Temperature / RawU16 streams)")
    else:
        print("Full grid: OFF (use --full-grid for 49152-channel streams)")

    try:
        with source:
            streamer.start()
            print_streams(discover_streams())
            count = streamer.run(source, duration=args.duration)
            print(f"[*] Pushed {count} frames")
    except KeyboardInterrupt:
        print("\n[*] Manual stop requested.")
    finally:
        streamer.stop()
        print("[*] Streamer stopped.")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Stream HIKMICRO thermal RoI vectors into LSL.",
    )
    parser.add_argument("--device", default="/dev/video0", help="V4L2 capture node")
    parser.add_argument(
        "--range", choices=("normal", "high"), default="normal", help="sensor range"
    )
    parser.add_argument(
        "--duration", type=float, default=None, metavar="S",
        help="stop after S seconds (default: run until Ctrl-C)",
    )
    parser.add_argument(
        "--fps", type=float, default=25.0, help="expected frame rate hint (default 25)"
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="also publish the 49152-channel Temperature/RawU16 streams",
    )
    parser.add_argument(
        "--no-raw-stream", action="store_true",
        help="do not publish HIKMICRO_RawU16 even with --full-grid",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="list V4L2 capture nodes and exit"
    )
    return parser


def main():
    args = build_parser().parse_args()

    if args.list_devices:
        from hikmicro_lsl.cli import main as cli_main

        return cli_main(["list-devices"])

    return run_bridge(args)


if __name__ == "__main__":
    sys.exit(main())
