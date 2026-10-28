# hikmicro-lsl

Stream HIKMICRO Pocket 2 / XTherm-family (256x192 InfiRay core) thermal frames
into [Lab Streaming Layer](https://labstreaminglayer.org) for synchronised
recording alongside EEG/BCI data, and extract temperatures from recorded files
offline.

Linux only for live capture. The camera enumerates as a plain UVC device
(`/dev/videoN`); no vendor SDK is required.

## Install

```fish
uv venv
uv pip install -e ".[dev]"
```

Optional extras:

```fish
uv pip install -e ".[wells]"         # opencv + pandas for 96-well plate analysis
git submodule update --init vendor/meltyfat
uv pip install -e vendor/meltyfat
```

With `make`:

```fish
make setup      # uv venv + uv pip install -e .[dev]
make test       # uv run pytest
make lint       # uv run ruff check
```

## Quickstart

1. Plug in the camera and confirm the node:

   ```fish
   hikmicro-lsl list-devices
   ```

   Expect a UVC node whose card name mentions the camera (e.g. `USB Video`).

2. Start LabRecorder, then stream:

   ```fish
   hikmicro-lsl stream --device /dev/video0 --raw --range normal
   ```

   In LabRecorder press `Update` and select:

   - `HIKMICRO_Temperature_RoI` (recommended: 13 channels)
   - `HIKMICRO_Meta`

   Add `--full-grid` if you also want the 49152-channel
   `HIKMICRO_Temperature` / `HIKMICRO_RawU16` grid streams.

   All streams share `source_id` prefix `hikmicro-pocket2`, so they stay aligned
   on the LSL clock.

3. Stream and record at the same time:

   ```fish
   hikmicro-lsl record -o recordings --duration 60
   ```

   Writes `recordings/frames-YYYYmmdd-HHMMSS.bin` (192x256 uint16 per frame)
   plus a `.jsonl` sidecar with index, timestamp, byte offset and the parsed
   metadata.

## LSL streams

| Stream | Type | Format | Channels | srate | Content |
|---|---|---|---|---|---|
| `HIKMICRO_Temperature_RoI` | `TEMPERATURE_ROI` | `cf_float32` | 13 | measured (~25 Hz) | `min_raw, max_raw, center_raw, min_c, max_c, center_c, mean_c, fpa_c, core_c, shutter_c, emissivity, distance_m, humidity_pct` |
| `HIKMICRO_Temperature` | `TEMPERATURE` | `cf_float32` | 49152 | measured (~25 Hz) | degC, row-major, labels `T_r{row}_c{col}`; opt-in `--full-grid` |
| `HIKMICRO_RawU16` | `RAW` | `cf_int32` | 49152 | measured (~25 Hz) | raw sensor counts; opt-in `--full-grid` |
| `HIKMICRO_Meta` | `META` | `cf_string` | 1 | `IRREGULAR_RATE` (0.0) | JSON blob per frame |

The default stream set is the RoI vector + Meta. The two 49152-channel grid
streams are off by default because a single frame is ~1.2 MB per stream at
25 Hz and a 49152-column table is unusable in BIDS/TSV consumers. Enable them
with `--full-grid`; `--no-raw-stream` suppresses the raw grid while keeping
the temperature grid; `--roi-only` publishes only RoI + Meta.

`cf_int32` is used for raw counts because LSL has no unsigned 16-bit type and
counts exceed the `cf_int16` range; `cf_int32` represents them exactly.

LSL cannot attach per-frame JSON to a sample, hence the separate meta stream.
Each meta string carries `fpa`/shutter/core temperatures, emissivity, min/max/
centre values with their coordinates, the frame counter, a `raw_mode` flag and
a running sample-rate estimate.

Temperature values (both RoI `*_c` columns and the full grid) are only pushed
when the metadata block parses (camera in raw mode). RoI `*_raw` columns are
always valid; the temperature/meta columns are `NaN` if the block does not
parse.

## Use with Nautilus BCI

The package is a drop-in source for `nautilus_bci`. A Nautilus-side wrapper
lives at `nautilus_bci/scripts/bridges/hikmicro_thermal_bridge.py`; a
standalone launcher lives here at `scripts/hikmicro_lsl_bridge.py`:

```fish
# standalone (uv deploys pylsl + numpy into an ephemeral env)
uv run --script scripts/hikmicro_lsl_bridge.py --device /dev/video0

# from a venv that already has hikmicro_lsl installed
uv run scripts/hikmicro_lsl_bridge.py --device /dev/video0 --duration 30
```

Consumer snippet:

```python
from pylsl import StreamInlet, resolve_byprop

inlet = StreamInlet(resolve_byprop("type", "TEMPERATURE_ROI", timeout=5.0)[0])
sample, ts = inlet.pull_sample()
```

The Nautilus `multimodal_bids_recorder.py` discovers streams dynamically and
keys them by name, so `HIKMICRO_Temperature_RoI` and `HIKMICRO_Meta` are
recorded with no recorder changes. Full contract, channel table and caveats:
[`docs/NAUTILUS_INTEGRATION.md`](docs/NAUTILUS_INTEGRATION.md).

## Offline workflow

```fish
hikmicro-lsl replay recordings/frames-YYYYmmdd-HHMMSS.bin     # back into LSL
hikmicro-lsl extract recordings/frames-YYYYmmdd-HHMMSS.bin --out out/
hikmicro-lsl extract stills/IRT_0001.jpg --out out/ --png
```

`extract` writes one `.npy` float32 temperature image per frame plus a summary
JSON. Radiometric JPEGs carry a 256x196 uint16 blob in an APP4 (tag 0x4064)
segment; plain JPEGs have no temperature data and are rejected with an error
pointing at the vendor CSV export path.

Vendor CSV exports (`HIKMICRO Analyzer`) go through meltyfat:

```fish
hikmicro-lsl extract export.csv --out out/
```

96-well detection delegates to `vendor/meltyfat` (`WellDetector` +
`WellAnalyzer`), imported lazily; it needs the `wells` extra and the submodule.

## Caveats

- **Raw mode is required for radiometry.** Without the vendor command 0x8004 the
  camera streams ordinary video; frames are still published on
  `HIKMICRO_RawU16` but flagged `raw_mode: false` and no temperatures are
  produced.
- **Vendor commands may be unavailable.** `0x8000`/`0x8004`/`0x8020`/`0x8021` go
  through `UVCIOC_CTRL_QUERY`. Kernels that disable UVC extension-unit ioctls
  (or a device that does not expose the XU) raise `NotImplementedError` with
  the errno; the CLI degrades to a warning and keeps streaming video.
- **Pocket 2 vs Mini2Plus** use the same 256x192 core but different firmware
  brightness/max-range handling; the high-range correction (`x1.17 - 40.9`) is
  calibrated for the HT203U/Pocket 2 family. Verify against a blackbody or a
  known reference before using absolute values from a Mini2Plus.
- **Live capture is Linux only** (V4L2 ioctls via `ctypes`). Offline decoding is
  portable.
- **Accuracy** rests on the device's own calibration block; the port does not
  add corrections beyond `Xtherm.kt`.

## Layout

```
src/hikmicro_lsl/radiometry.py   parse_meta / temp_table / iron_palette
src/hikmicro_lsl/uvc_source.py   V4L2 mmap capture + UVCIOC_CTRL_QUERY
src/hikmicro_lsl/lsl_streamer.py LSL outlets, frame loop, .bin/.jsonl recorder
src/hikmicro_lsl/offline.py      raw dump reader, JPEG/CSV extraction, wells
src/hikmicro_lsl/cli.py          hikmicro-lsl entry point
scripts/hikmicro_lsl_bridge.py   standalone LSL launcher (uv-run, PEP 723)
docs/NAUTILUS_INTEGRATION.md     stream contract for the Nautilus BCI stack
docs/FRAME_FORMAT.md             wire format, metadata offsets, calibration
```

## Licence

MIT. The radiometry port follows `vendor/HT203U-Thermal` (see its licence).
