# Nautilus BCI integration

`hikmicro-lsl` is drop-in compatible with the Nautilus BCI stack
(`nautilus_bci/`). This document is the contract: stream names, types, formats
and how the Nautilus recorder consumes them.

## Stream table

Default invocation (`--device /dev/video0`):

| name | type | format | channels | nominal_srate | notes |
|---|---|---|---|---|---|
| `HIKMICRO_Temperature_RoI` | `TEMPERATURE_ROI` | `cf_float32` | 13 | measured (~25 Hz) | **the stream to consume** |
| `HIKMICRO_Meta` | `META` | `cf_string` | 1 | 0.0 (`IRREGULAR_RATE`) | one JSON blob per frame |

Opt-in with `--full-grid`:

| name | type | format | channels | nominal_srate | notes |
|---|---|---|---|---|---|
| `HIKMICRO_Temperature` | `TEMPERATURE` | `cf_float32` | 49152 | measured (~25 Hz) | full 256x192 grid in degC; raw-mode frames only |
| `HIKMICRO_RawU16` | `RAW` | `cf_int32` | 49152 | measured (~25 Hz) | full grid, sensor counts |

All streams share `source_id` prefix `hikmicro-pocket2` so LabRecorder groups
them. Labels and units are carried in the XML `desc` (`channels/channel`).

### RoI channel order

Channel index == column order, fixed:

| # | label | unit | source |
|---|---|---|---|
| 0 | `min_raw` | count | raw grid minimum |
| 1 | `max_raw` | count | raw grid maximum |
| 2 | `center_raw` | count | raw count at (96, 128) |
| 3 | `min_c` | degC | temperature grid minimum |
| 4 | `max_c` | degC | temperature grid maximum |
| 5 | `center_c` | degC | temperature at (96, 128) |
| 6 | `mean_c` | degC | temperature grid mean |
| 7 | `fpa_c` | degC | `Meta.temp_fpa` |
| 8 | `core_c` | degC | `Meta.temp_core` |
| 9 | `shutter_c` | degC | `Meta.temp_shutter` |
| 10 | `emissivity` | ratio | `Meta.emissivity` |
| 11 | `distance_m` | m | `Meta.distance` |
| 12 | `humidity_pct` | % | `Meta.humidity` |

When the metadata block does not parse (camera not in raw mode) columns 3-12
are `NaN`; columns 0-2 are always valid. Nothing is ever silently zero-filled.

## Consuming the stream

Resolve by type, not by name:

```python
from pylsl import StreamInlet, resolve_byprop

streams = resolve_byprop("type", "TEMPERATURE_ROI", timeout=5.0)
if not streams:
    raise SystemExit("no thermal RoI stream found; is the bridge running?")

inlet = StreamInlet(streams[0])
sample, ts = inlet.pull_sample()
# sample -> [min_raw, max_raw, center_raw, min_c, max_c, center_c, mean_c,
#            fpa_c, core_c, shutter_c, emissivity, distance_m, humidity_pct]
```

## Recorder pickup (no changes needed)

`nautilus_bci/scripts/recorders/multimodal_bids_recorder.py` discovers streams
with `resolve_streams(wait_time=...)` and keys them by `info.name()`, storing
`channel_count` and `nominal_srate` per stream. Every non-EEG, non-smartwatch
stream falls through to the generic path, which writes

```
<TIMESTAMP>_<name>.tsv          one column per channel + timestamp_sec
<TIMESTAMP>_<name>_meta.json    stream metadata
```

under `results/bids.../sub-XX/ses-YY/<type-lowercase>/`. So
`HIKMICRO_Temperature_RoI` lands in a `temperature_roi/` folder with 13
columns, and `HIKMICRO_Meta` in `meta/` with a single `meta_json` column.
No recorder edit is required. Enable the stream before starting the recorder
(`--full-grid` off is the safe default).

## Time alignment

LSL timestamps come from the shared LSL clock, the same one the g.Nautilus
outlet uses, so an inlet-side join needs no manual offset. For offline joins,
use `pylsl.local_clock()`-stamped marker streams as usual.

## Caveats

- **Do not feed the 49152-channel grid to a BIDS/TSV writer.** 49152 columns
  at ~25 Hz is roughly 1.2 MB/frame and tens of thousands of CSV columns; the
  recorder would build a huge DataFrame and `mne.io.RawArray` would allocate
  gigabytes. Use the RoI stream for recorded sessions and `--full-grid` only
  for interactive inspection or LabRecorder XDF capture.
- `HIKMICRO_Meta` is `cf_string` with `IRREGULAR_RATE` (0.0). Consumers must
  not assume a sample rate for it.
- Temperature values require the camera in raw mode. Without it, only the RoI
  raw-count columns and the meta `raw_mode: false` flag are meaningful.
- `--roi-only` publishes just `HIKMICRO_Temperature_RoI` + `HIKMICRO_Meta`.
