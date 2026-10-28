#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pylsl",
# ]
# ///
import sys
import time
import signal
import numpy as np
from pylsl import resolve_byprop, StreamInlet, local_clock

def signal_handler(sig, frame):
    print("\nExiting cleanly...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)

    print("Looking for an LSL stream...")
    # Resolve stream by properties or just any stream
    # The prompt mentions "discovers and subscribes to the HIKMICRO LSL thermal stream"
    streams = resolve_byprop('name', 'HIKMICRO', timeout=2.0)

    if not streams:
        print("No HIKMICRO stream found.")
        sys.exit(1)

    inlet = StreamInlet(streams[0])
    info = inlet.info()

    name = info.name()
    type_ = info.type()
    channels = info.channel_count()
    rate = info.nominal_srate()
    print(f"Connected to stream '{name}' ({type_}) with {channels} channels at {rate} Hz.")

    frames = 0
    start_time = time.time()
    last_print_time = start_time

    print("Starting stream consumption (Press Ctrl+C to stop)...")

    try:
        while True:
            sample, timestamp = inlet.pull_sample()

            if timestamp is not None:
                current_time = time.time()
                frames += 1

                # Calculate latency
                latency = local_clock() - timestamp

                # Calculate FPS
                elapsed = current_time - last_print_time
                if elapsed >= 1.0:
                    fps = frames / elapsed

                    # Compute min/max/center
                    # If sample is a list, convert to numpy array
                    arr = np.array(sample)

                    min_temp = np.min(arr)
                    max_temp = np.max(arr)

                    # Assuming a 2D grid, but LSL provides a flattened array
                    # Center index would be len(arr) // 2
                    center_temp = arr[len(arr) // 2] if len(arr) > 0 else 0

                    # Overwrite the line in terminal
                    # using \r and end='' to update inplace
                    sys.stdout.write(f"\rFPS: {fps:5.1f} | Latency: {latency*1000:6.1f} ms | Temp (Min/Max/Center): {min_temp:6.2f} / {max_temp:6.2f} / {center_temp:6.2f}")
                    sys.stdout.flush()

                    frames = 0
                    last_print_time = current_time

    except KeyboardInterrupt:
        print("\nExiting cleanly...")
        sys.exit(0)

if __name__ == '__main__':
    main()
