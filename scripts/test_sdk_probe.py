#!/usr/bin/env python3
"""Test script for HIKMICRO Pocket 2 using native libHCUSBSDK.so."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import ctypes
from hikmicro_lsl.hcusbsdk import (
    HcUsbSdk,
    USBFrameInfo,
    STREAM_TYPE_MIXED,
    STREAM_TYPE_RAW_RADIOMETRIC,
    STREAM_TYPE_THERMAL_Y16,
    STREAM_TYPE_YUYV_PREVIEW,
)

sample_saved = False

def frame_callback(lRealHandle, pFrameInfo, pUser):
    global sample_saved
    if not pFrameInfo:
        return
    info = pFrameInfo.contents
    raw_data = ctypes.string_at(info.pBuf, info.dwBufSize) if info.pBuf and info.dwBufSize > 0 else b""
    prefix = raw_data[:16].hex(" ")
    print(
        f"[FRAME] Handle={lRealHandle} Num={info.dwFrameNum} "
        f"W={info.dwWidth} H={info.dwHeight} "
        f"FPS={info.dwFramerate} BufSize={info.dwBufSize} bytes | {prefix}"
    )
    if not sample_saved and len(raw_data) > 0:
        sample_saved = True
        sample_path = Path("/tmp/hikmicro_sample_frame.bin")
        sample_path.write_bytes(raw_data)
        print(f"--> Saved sample frame to {sample_path} ({len(raw_data)} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Probe HIKMICRO Pocket 2 via native libHCUSBSDK.")
    parser.add_argument("--stream-type", type=lambda x: int(x, 0), default=STREAM_TYPE_MIXED,
                        help="Stream type (0x65=Preview, 0x66=Y16, 0x67=Mixed, 0x68=Raw radiometric)")
    parser.add_argument("--duration", type=float, default=5.0, help="Stream probe duration in seconds")
    args = parser.parse_args()

    print("[1] Initialising HCUSBSDK...")
    sdk = HcUsbSdk()
    if not sdk.init():
        print("ERROR: USB_Init() failed.")
        return 1

    sdk.set_log(level=3, log_dir="/tmp/hcusbsdk_log")

    print("[2] Enumerating USB devices...")
    count = sdk.get_device_count()
    print(f"Device count reported: {count}")

    if count == 0:
        print("\nNote: USB_GetDeviceCount returned 0.")
        print("Checking USB endpoint permissions for 2bdf:017f...")
        try:
            import subprocess
            lsusb_out = subprocess.run(["lsusb", "-d", "2bdf:"], capture_output=True, text=True).stdout
            print(f"lsusb check:\n{lsusb_out.strip() or 'No 2bdf devices in lsusb'}")
        except Exception:
            pass
        print("\nIf the device is visible in lsusb but device count is 0, ensure non-root read/write permissions:")
        print("  sudo cp udev/99-hikmicro.rules /etc/udev/rules.d/")
        print("  sudo udevadm control --reload && sudo udevadm trigger")
        sdk.cleanup()
        return 1

    devices = sdk.enum_devices()
    for idx, d in enumerate(devices):
        print(f"  Device {idx}: VID={d.vid_hex()} PID={d.pid_hex()} Name={d.name()} SN={d.serial()} Path={d.path()}")

    print("\n[3] Logging in to Device 0...")
    try:
        login_id = sdk.login(0)
        print(f"Login success! Login handle ID: {login_id}")
    except Exception as e:
        print(f"Login failed: {e}")
        sdk.cleanup()
        return 1

    print("\n[4] Capturing radiometric P2P frame (0x7fe USB_GET_JPEGPIC_WITH_APPENDDATA)...")
    try:
        from hikmicro_lsl.sdk_source import decode_appenddata

        jpeg, app, _first = sdk.get_jpeg_with_appenddata(login_id)
        temps = decode_appenddata(app)
        if temps is not None:
            h, w = temps.shape
            center_c = float(temps[h // 2, w // 2])
            print(
                f"  Radiometric frame: {w}x{h} ({temps.dtype}), "
                f"min={float(temps.min()):.2f} C, max={float(temps.max()):.2f} C, "
                f"mean={float(temps.mean()):.2f} C, center={center_c:.2f} C"
            )
        else:
            print(f"  Received {len(app)} append bytes, {len(jpeg)} JPEG bytes")
    except Exception as e:
        print(f"  P2P snapshot error: {e}")

    print(f"\n[5] Starting stream callback (type 0x{args.stream_type:02x})...")
    try:
        stream_token = sdk.start_stream(login_id, frame_callback, stream_type=args.stream_type)
        print(f"Stream callback active! Streaming for {args.duration}s...")
        time.sleep(args.duration)
        sdk.stop_stream(login_id)
    except Exception as e:
        print(f"Streaming error: {e}")

    print("\n[6] Logging out and cleaning up...")
    sdk.logout(login_id)
    sdk.cleanup()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
