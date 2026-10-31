#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pylsl",
# ]
# ///
"""Check that the HIKMICRO thermal stream is alive: connects to HIKMICRO_Temperature_RoI and
prints frame rate, latency and min / max / centre / mean temperature on one updating line.

    uv run scripts/consume_lsl.py [--timeout 5]
"""
import argparse
import math
import sys
import time

from pylsl import StreamInlet, local_clock, proc_clocksync, resolve_byprop

ROI_STREAM = "HIKMICRO_Temperature_RoI"
# Channel order published by hikmicro_lsl.lsl_streamer (used if the stream header has no labels)
DEFAULT_LABELS = ("min_raw", "max_raw", "center_raw", "min_c", "max_c", "center_c", "mean_c",
                  "fpa_c", "core_c", "shutter_c", "emissivity", "distance_m", "humidity_pct")


def channel_labels(info) -> list[str]:
    labels = []
    ch = info.desc().child("channels").child("channel")
    while not ch.empty():
        labels.append(ch.child_value("label"))
        ch = ch.next_sibling("channel")
    if len(labels) == info.channel_count() and all(labels):
        return labels
    return list(DEFAULT_LABELS[:info.channel_count()])


def fmt(value: float) -> str:
    return "  n/a" if value is None or math.isnan(value) else f"{value:5.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for the stream")
    args = ap.parse_args()

    print(f"Looking for '{ROI_STREAM}' (start the bridge: hikmicro_lsl_bridge.py) ...")
    streams = resolve_byprop("name", ROI_STREAM, timeout=args.timeout)
    if not streams:
        print(f"No '{ROI_STREAM}' stream found.")
        return 1

    # proc_clocksync puts timestamps on this machine's clock, so latency is meaningful
    # even when the camera bridge runs on another host
    inlet = StreamInlet(streams[0], processing_flags=proc_clocksync)
    info = inlet.info(timeout=args.timeout)
    labels = channel_labels(info)
    idx = {name: labels.index(name) for name in ("min_c", "max_c", "center_c", "mean_c") if name in labels}
    print(f"Connected: '{info.name()}' ({info.type()}, {info.channel_count()} ch) from {info.hostname()}. "
          "Ctrl+C to stop.")

    frames, window_start = 0, time.monotonic()
    try:
        while True:
            sample, ts = inlet.pull_sample(timeout=2.0)
            if ts is None:
                sys.stdout.write("\rNo frames for 2 s ...".ljust(100))
                sys.stdout.flush()
                continue
            frames += 1
            now = time.monotonic()
            if now - window_start >= 1.0:
                fps = frames / (now - window_start)
                latency_ms = (local_clock() - ts) * 1e3
                temps = {k: sample[i] for k, i in idx.items()}
                sys.stdout.write(
                    f"\rFPS {fps:5.1f} | latency {latency_ms:6.1f} ms | degC min {fmt(temps.get('min_c'))} "
                    f"max {fmt(temps.get('max_c'))} centre {fmt(temps.get('center_c'))} mean {fmt(temps.get('mean_c'))}")
                sys.stdout.flush()
                frames, window_start = 0, now
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
