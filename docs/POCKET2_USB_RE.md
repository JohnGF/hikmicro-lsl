# HIKMICRO Pocket 2 — live radiometric over USB: investigation & findings

**Date:** 2026-10-27
**Goal:** get live per-pixel temperature (radiometric) data off a HIKMICRO Pocket 2
over USB on Linux, time-synced into LSL/UDP for the BCI stack — originally framed as
"modify the firmware to stream raw data", which turned out to be unnecessary.

---

## TL;DR

- **Firmware modification is unnecessary *and* infeasible** — dropped.
- The Pocket 2 already exposes radiometric data over USB; it's just gated behind a
  vendor command protocol.
- That protocol is **Hikvision's `HCUSBSDK`** (UVC extension-unit command channel).
- We reverse-engineered the command map + wire-framing, but driving it by hand on
  Linux **wedges the camera on every malformed command** (one attempt per physical
  replug), so hand-rolling the protocol is impractical.
- **Best path:** the SDK exports a clean C API and has a **Linux x86_64 build**
  (`libHCUSBSDK.so`). Obtain it from HIKMICRO and call `HC_USBCamera_*` directly —
  the SDK does the hard part. This is a procurement step (request-only, not a public
  download).

---

## 1. Firmware route — ruled out

`firmware/Firmware_V5.5.112_260624_EN.zip` → `digicap.dav` is a Hikvision package.
Its body is **encrypted end-to-end** (entropy ~7.99 bits/byte across all 144 MB; no
carvable filesystem — no SquashFS/UBI/cramfs magic) and the camera **signature-checks**
flashed images. Without the vendor decryption + signing keys you can't produce an
accepted modified image, and a bad flash bricks the unit. No public V5.5.112 release
notes mention streaming. **Not a path.**

## 2. How the camera presents over USB

On-device **USB mode** setting changes the USB PID:
- "USB drive"       → `2bdf:0167` (mass storage)
- "USB cast screen" → `2bdf:017f` (UVC Camera; vendor `2bdf` = InfiRay/iRay)

In cast-screen mode it enumerates as a UVC device on `/dev/video2` offering:
- `MJPG` 640×480 / 320×240 — the rendered/colourised display image (works immediately)
- `YUYV` **640×256** — the radiometric channel; **advertised but delivers 0 bytes
  until a vendor command enables it**
- a **UVC extension unit**: `bUnitID 17 (0x11)`, GUID `{9a1e7291-6843-4683-6d92-39bc7906ee49}`,
  10 controls — this is the command channel. On Linux only **selector 1** responds:
  a **512-byte GET|SET control** (`UVCIOC_CTRL_QUERY`, `info=0x03 len=512`).

Note: `/dev/video0` ("USB Video", MacroSilicon `534d:2109`) is a separate 4K microscope,
not the Pocket 2.

## 3. The command protocol (reverse-engineered from `HCUSBSDK.dll`)

Source: HIKMICRO Analyzer 2.1.1.3 (Windows) ships `HCUSBSDK.dll` — Hikvision's USB
camera SDK. Analyzed with `strings`/`objdump`/Ghidra. It is **cross-platform**
(DirectShow, **libusb** and libuvc backends; contains `libusb1.0.so`,
`libusb_init_with_usbfs`, `detach_kernel_driver` → a Linux/Android build exists) with a
clean public C API: `HC_USBCamera_Init / EnumDevice / Open / OpenEx / SetConfig /
GetConfig / StartPreview / StopPreview / Capture / Close / Fini / GetLastError /
GetSDKVersion / SetLogToFile` (all **exported by name** → callable via ctypes).

### Command map
`CUsbCommandBuilder::CreateCommand(cmdID, …)` is a big switch; each entry validates the
caller buffer size == `sizeof(struct)` and builds a 0x28-byte command object. Extracted
506 command IDs (MSVC `$0` nibble encoding A–P = 0–15, verified by consecutive IDs).
The ones relevant to radiometry:

| Command | ID get/set | struct | meaning |
|---|---|---|---|
| `USB_SYSTEM_DEVICE_INFO` | `0x7DB` | — | device info (safe read) |
| `USB_THERMOMETRY_BASIC_PARAM` | `0x7EE/0x7EF` | — | emissivity/distance/reflect temp |
| `USB_THERMAL_STREAM_PARAM` | `0x7F6/0x7F7` | — | thermal stream params |
| `USB_P2P_PARAM` | `0x800/0x801` | — | **per-pixel temperature data ("P2P")** |
| `USB_CTRL_THERMAL_STREAM_PARAM` | `0x83E/0x83F` | 32 B (0x20) | **start/ctrl the thermal stream** |
| `USB_THERMAL_STREAM_YUV_RESOLUTION` | `0x853/0x854` | — | selects the 640×256-type mode |
| `USB_JPEGPIC_WITH_APPENDDATA` | `0x7FE` | — | radiometric still (jpeg + p2p) |

(get = even id, set = odd id.) Video "raw mode" is also selectable via
`CUsbDriverDirectShow::SetRawVideoType(dwVideoType)` = a UVC `SetFormat`.

### Wire framing (`CUsbProtocolUVC`)
- Command channel = **XU unit `0x11`, selector `1`, 512-byte buffer**.
- Transfer = a UVC extension-unit property access: on Windows via `IKsControl::KsProperty`
  with a `KSP_NODE` (XU GUID + `Id`=selector + `Flags` + NodeId), on Linux the equivalent
  XU control transfer. **GET flag `0x10000001`** (`KSPROPERTY_TYPE_GET|TOPOLOGY` → UVC
  `GET_CUR`), **SET flag `0x10000002`** (→ UVC `SET_CUR`).
- Sequence: **`SET_CUR` the 512-byte command packet, then `GET_CUR` the 512-byte response.**
  A data SET is **two-step**: first a 16-byte "set length" `SET_CUR`, then the data `SET_CUR`.
- Packet = ~**11-byte header** (command id + direction byte + length; cmdID lives at command-
  object offset `0xc`, direction at byte 9, subtype at byte 10) **+ payload**. The exact
  wire byte offsets of the header are the one piece the decompile left ambiguous (C++
  object aliasing).

Ghidra project + scripts and raw decompile dumps were produced under the session
scratchpad (`ghidra_proj`, `DumpXref.java`, `DumpAddrs.java`, `DumpB.java`, `DumpC.java`,
`decomp*.txt`); the DLLs analyzed were `HCUSBSDK.dll`, `FormatConversion.dll`, `Identify.dll`.

## 4. Why hand-driving it on Linux didn't work

Harness `scratch_hik_cmd.py` (repo root) drives XU unit `0x11`/sel `1` via
`UVCIOC_CTRL_QUERY`. Results on a fresh camera (`/dev/video2`, cast-screen):
- `GET_INFO`/`GET_LEN` work (`0x03`, 512).
- `GET_CUR` → **EIO**; every `SET_CUR` → **ETIMEDOUT**, whether idle, with the MJPG
  preview held open, single or two-step.
- **A single malformed `SET_CUR` wedges the device** — afterwards even `GET_INFO` EIOs,
  and only a **physical unplug/replug** recovers it.

So empirical header-guessing is one attempt per replug, and every miss hangs the camera.
Not viable without the exact wire bytes — which the RE left fuzzy. Hence the pivot.

## 5. Recommended path — use the vendor SDK

`HCUSBSDK.dll` exports the API by name, and a **Linux x86_64 build (`libHCUSBSDK.so`)
exists** — confirmed by the GitHub project `arsdrtrt/HJK-SDK`, which drives a sibling
camera (`2bdf:0102`, 640×512 Y16) using `libHCUSBSDK.so` + `libIRSDK.so` + `libhpr.so`
(+ `libHJKUSBSDK.so`) and includes a `docs/PROTOCOL.md` (its own HCUSBSDK reverse:
command map / firmware / crypto). **The `.so` is proprietary and not shipped publicly.**

**To proceed:** obtain the Linux SDK from HIKMICRO — email `support@hikmicrotech.com`
and request the **Linux x86_64 USB Camera SDK** (`libHCUSBSDK.so`, `libIRSDK.so`,
`libhpr.so`, and headers); mention the Pocket 2. Their SDK / "UVC Client" is request-only,
not a public download. Caveat: the one known Linux user targets `2bdf:0102`; our Pocket 2
is `2bdf:017f` — same SDK family, expected to work but unverified until tried.

Once the `.so` + headers are in hand: a thin `ctypes`/CFFI wrapper on Linux —
`HC_USBCamera_Init → EnumDevice → Open → SetConfig(USB_CTRL_THERMAL_STREAM_PARAM /
start) → StartPreview(frame callback)` — feeds the existing `lsl_streamer` / a UDP sink.
The SDK performs the XU command dance correctly, so no camera-wedging.

### Alternatives considered
- **(A)** Drive `HCUSBSDK.dll` via ctypes on **Windows** (we have the DLL), `StartPreview`
  → frames, and bridge them to the Linux BCI over **LSL/UDP** (Windows = capture node;
  this directly realises the original "UDP raw data to a server" idea). Viable now; needs
  the SDK function signatures worked out.
- **(B)** Finish the wire-protocol RE and reimplement in Python — blocked by the
  wedging + fuzzy header bytes above. Not recommended.
- **Bare-core tooling** `Krastanov/ht-203u-linux` is for the raw InfiRay HT-203U module,
  a different (non-SDK) path.

## 6. Pragmatic fallback (no live stream)

If live radiometry stalls: record on-device, pull the radiometric MP4/JPEG (the APP4
`0x4064` blob carries real temperatures), and extract offline — the repo already does
this (`offline.py`, `meltyfat`). BCI time-sync would then be solved separately (e.g. a
shared thermal event to cross-correlate), which is the inconvenience that motivated
wanting a live stream in the first place.

## 7. How to resume

- Camera must be in **USB cast-screen** mode (`2bdf:017f`, `/dev/video2`); it reverts to
  USB-drive mode on sleep/replug and dozes quickly — keep it awake.
- If a command attempt hangs it, **replug** to recover.
- Durable notes also in Claude memory (`pocket2-usb-radiometric`).
- Experimental harness: `scratch_hik_cmd.py` (repo root) — safe device-info GET sweep.
