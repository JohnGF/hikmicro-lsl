# HIKMICRO Pocket 2 / XTherm frame format

Sources: `vendor/HT203U-Thermal/app/src/main/java/io/github/cbird/ht203u/Xtherm.kt`
(line references below) and `XthermTest.kt` (golden values, lines 20-46, 70-79).
The core is the InfiRay 256x192 sensor used in the HIKMICRO Pocket 2, Hti
HT-203U, T2S+ and T2L.

## 1. UVC frame layout

Each frame delivered by the UVC device is 256 x 196 little-endian `uint16`:

| Field | Rows | u16 offset | u16 count | Bytes |
|---|---|---|---|---|
| Thermal image | 0..191 | 0 | 49152 | 98304 |
| Metadata block | 192..195 | 49152 | 1024 | 2048 |
| **Total** | 196 | 0 | **50176** | **100352** |

- `WIDTH = 256`, `HEIGHT = 192`, `META_ROWS = 4`, `FRAME_HEIGHT = 196`
  (`Xtherm.kt:15-26`).
- `TABLE_SIZE = 16384`: raw values are 14-bit and index a 16384-entry LUT.
- The image area holds raw 14-bit counts **only after** vendor command `0x8004`
  has switched the stream to raw mode. Otherwise the same payload is ordinary
  video and the metadata block contains no calibration data.
- The UVC payload is byte-packed little-endian. In YUYV each 2-byte word is one
  sample; the byte order is identical, so the driver-side decode is
  `np.frombuffer(payload, '<u2')`.

## 2. Metadata field offsets

All offsets are relative to `m = 49152` (start of the metadata block), in u16
units (`Xtherm.kt:74-136`). `f32(off)` reads two u16 as an IEEE-754 float32.

| Offset | Type | Field | Notes |
|---|---|---|---|
| `m + 0` | u16 | `fpaAverage` | mean FPA reading |
| `m + 1` | u16 | FPA counter | `temp_fpa = 20 - (v - FPA_OFF)/FPA_DIV` |
| `m + 2` | u16 | `maxX` | must be `< 256` |
| `m + 3` | u16 | `maxY` | must be `< 192` |
| `m + 4` | u16 | `maxRaw` | `0 < maxRaw < 16384` |
| `m + 5` | u16 | `minX` | must be `< 256` |
| `m + 6` | u16 | `minY` | must be `< 192` |
| `m + 7` | u16 | `minRaw` | `minRaw <= maxRaw` |
| `m + 8` | u16 | `avgRaw` | |
| `m + 12` | u16 | `centerRaw` | `minRaw <= centerRaw <= maxRaw` |
| `m + 256` | u16 | `cal00` | table offset |
| `m + 257` | u16 /10 K | shutter temperature | `T = v/10 - 273.15` |
| `m + 258` | u16 /10 K | core temperature | `T = v/10 - 273.15` |
| `m + 259` | f32 | `cal01` | shutter quadratic A |
| `m + 261` | f32 | `cal02` | shutter quadratic B |
| `m + 263` | f32 | `cal03` | FPA quadratic, 2nd order |
| `m + 265` | f32 | `cal04` | FPA quadratic, 1st order |
| `m + 267` | f32 | `cal05` | FPA quadratic constant |
| `m + 383` | f32 | `correction` | default 0, valid [-50, 50] |
| `m + 385` | f32 | `tempReflected` | default 25, valid [-50, 300] |
| `m + 387` | f32 | `tempAir` | default 25, valid [-50, 100] |
| `m + 389` | f32 | `humidity` | default 0.45, valid [0, 1] |
| `m + 391` | f32 | `emissivity` | default 0.95, valid [0.01, 1] |
| `m + 393` | u16 | `distance` | cm |

The user area starts at `m + AMOUNT_PIXELS + 127` = `m + 383` with
`AMOUNT_PIXELS = 256` (`Xtherm.kt:33-35`, `:96`), i.e. absolute u16 offset 49535.
For reference, `cal01` is at `m + 259` (absolute 49411) and `cal05` at
`m + 267` (absolute 49419).

### Plausibility checks (all preserved in `parse_meta`)

1. Frame must contain at least `m + 4*256` u16.
2. `maxX/maxY/minX/minY` inside the image.
3. `minRaw <= maxRaw`, `maxRaw < 16384`, `maxRaw != 0`.
4. `minRaw <= centerRaw <= maxRaw`.
5. `maxRaw >= 100` and `maxRaw - minRaw >= 16`.
6. `-40 <= temp_shutter <= 150` and `-40 <= temp_fpa <= 150`.
7. Float fields out of their validity window fall back to the vendor default
   (Kotlin `orDefault`), they do not reject the frame.

Any failure returns `None`, which is how plain video frames are rejected.

## 3. Calibration formula (`tempTable`, `Xtherm.kt:140-180`)

Constants: `ZEROC = 273.15`, `FPA_OFF = 8617.0`, `FPA_DIV = 37.682`,
`CAL_00_OFFSET = 170.0`, `CAL_00_FPAMUL = 0.0`.

```
temp_fpa     = 20 - (fpaCounter - FPA_OFF) / FPA_DIV
temp_shutter = shutterRaw / 10 - ZEROC
temp_core    = coreRaw / 10 - ZEROC
distance     = u16(distance)            # cm
d            = max(distance, 20) * 1.0  # cm used by the atmosphere model

wvc(hum, air)   = hum * exp(1.5587 + 0.06939*air - 2.7816e-4*air^2 + 6.8455e-7*air^3)
atmt(hum, air, d) = 1.9  * exp(-sqrt(d) * (0.006569 + (-0.002276)*sqrt(wvc)))
                  + (1 - 1.9) * exp(-sqrt(d) * (0.01262  + (-0.00667) *sqrt(wvc)))

numerator   = (1 - emissivity) * atmt * (tempReflected + ZEROC)^4
            + (1 - atmt) * (tempAir + ZEROC)^4
denominator = emissivity * atmt
calA        = cal02 / (2*cal01)
calB        = cal02^2 / (4*cal01^2)
calC        = cal01*temp_shutter^2 + temp_shutter*cal02
calD        = cal03*temp_fpa^2 + cal04*temp_fpa + cal05
tableOffset = cal00 - max(int(CAL_00_OFFSET - temp_fpa*CAL_00_FPAMUL), 0)

for v in 0..16383:
    n   = sqrt(|((v - tableOffset)*calD + calC)/cal01 + calB|)   # NaN -> 0
    t   = ((n - calA) + ZEROC)^4
    t   = ((t - numerator)/denominator)^0.25 - ZEROC
    t   = t + (d*0.85 - 1.125)*(t - tempAir)/100 + correction
    if high_range: t = 1.17*t - 40.9
    table[v] = t
```

Notes:

- `t = (n - calA + ZEROC)^4` and the outer `^0.25` are applied **without** an
  absolute value. Where the inner argument is negative the Kotlin code yields
  `NaN`; where `n < calA` the branch is mirrored and the table is not
  monotonic. Real sensor counts sit in the monotonic region (`n >= calA`).
- `high_range` (vendor command `0x8021`) only changes the linear
  `x*1.17 - 40.9` correction; the rest of the chain is identical.
- Reference values from `XthermTest.kt:74-78`: table[4000] = 890.217099,
  table[7000] = 407.748100, table[8000] = 184.576217, table[9000] = 490.373019,
  table[12000] = 931.326981 (within 0.02). `tests/test_radiometry.py` asserts
  these.

Pixel temperature is `table[raw]`; the Python implementation builds the LUT
once per metadata block with numpy and indexes it, so no per-pixel loop runs.

## 4. Vendor extension-unit commands

Sent as a 2-byte little-endian payload via `UVCIOC_CTRL_QUERY` with
`UVC_SET_CUR`. On the HT-203U/Pocket 2 the unit is `0x02` and the selector
`0x01`; the selector value carries the command word (`Xtherm.kt:37-40`).

| Command | Meaning |
|---|---|
| `0x8000` | trigger shutter (NUC) calibration |
| `0x8004` | switch stream to raw 16-bit |
| `0x80FF` | save / persist |
| `0x8020` | normal range, -20..120 C |
| `0x8021` | high range, -20..450 C |

If the kernel disables UVC extension-unit ioctls or the device does not expose
them, `send_vendor_command` raises `NotImplementedError` with the failing errno.
It never silently no-ops; the CLI prints a warning and continues in video mode.

## 5. Recorded dump format (`.bin` + `.jsonl`)

`hikmicro-lsl record` writes, per session:

- `frames-<timestamp>.bin`: concatenated 192x256 uint16 frames, 98304 bytes
  each, image area only (metadata rows are not stored).
- `frames-<timestamp>.jsonl`: one JSON object per frame:
  `{"index", "t" (LSL clock), "offset" (byte offset in the bin), "frame_u16",
  "meta": {...,"valid": true|false}}`.
- `frames.latest`: name of the last bin file written.
