# MiniMax-M3 Slice 7b-ext — calibration across the new resource kinds

Date: 7 September 2026. Status: PASS C/M, P `NOT_RUN` (data-model only).

## Scope

Extend the existing bounded calibration pipeline (`m3_shadow_calibrate.py`)
so the four new resource kinds (`cpu_island`, `gpu_island`, `memory_dom`,
`link`) declared by the 7a-ext schema can carry measured calibration
fields.  The extension is strictly additive: the existing capture path
is unchanged, the existing JSON schema is unchanged, and a new
`--emit-planner-text <path>` flag writes a sibling planner-readable
text dump alongside the capture JSON.

## Files touched

| File | Change |
|---|---|
| `c/tools/m3_shadow_calibrate.py` | Added `islands` manifest field, `_format_island_row()`, `emit_planner_text()`, and the `--emit-planner-text` flag. It emits storage profiles only for resources actually measured; declared controller/upstream resources with no own measurement remain profile-less. Existing JSON capture path unchanged. |
| `c/m3_shadow_plan.c` | Allows a topology to be partially calibrated. Missing profiles are retained as an honest `UNKNOWN` path state; present profiles still require a matching topology hash. Non-storage kinds may also carry no profile because the Slice 7c-1 predictor reads the kind-specific payload directly (`rate_bytes_per_s` on link / memory_dom, `vram_bytes` on gpu_island, `isa_caps` on cpu_island). |
| `c/m3_shadow_plan_calib.h` | `m3_shd_calib_is_sound()` now permits absent profile rows while validating every row that is present. INV-10 honesty is preserved (every profile row still must carry `units_per_second_explicit > 0`, `sample_count > 0`, and a matching tail percentile). |
| `c/tests/test_m3_shadow_calibration_emit.c` | New end-to-end gate: reads a planner-text dump, calls `m3_shd_parse_text`, then `m3_shd_open`.  Counts parsed kinds. |
| `c/Makefile` | New build target `tests/test_m3_shadow_calibration_emit.exe` and `make test-shadow-m3-calibration-emit` wrapper. |
| `docs/results/m3-shadow-calibrate-A-manifest.json` | New manifest for the machine-A capture. |
| `docs/results/m3-execution-slice-7b-ext-capture-2026-09-07.json` | Captured raw JSON (machine A, 2 samples × inflight 1/2/4, 16 MiB blocks). |
| `docs/results/m3-execution-slice-7b-ext-2026-09-07.planner.txt` | Planner-readable text dump from the capture above. |

## What the planner text looks like

The 7b-ext planner text is a sequence of:
- `resource <id> <kind> <parent> <max-inflight> <max-bytes>` rows for each
  storage kind (drive / controller / upstream).
- `profile <id> <size> <load> <rate> <startup> <residual> <samples> [<tail-ppm>]`
  rows for each storage resource that fired during capture, plus
  profile rows only for resources that were measured directly by an
  unambiguous singleton-drive group; multi-drive aggregates are omitted.
- `cpu_island / gpu_island / memory_dom / link` rows for the new
  resource kinds (declared in the manifest's `islands` field, kept on
  the planner topology, but the predictor reads their payload directly
  rather than via the profile table).

The capture's `islands` field is the only path through which the new
kinds enter the planner topology; their rates come from the manifest
field declarations (`rate_bytes_per_s`, `vram_bytes`, `isa_bits`,
`cores`).  The capture tool never fabricates a value — missing
measurements stay absent and the parser returns `UNKNOWN` for them
without inventing defaults (INV-10 discipline).

## End-to-end gate

```text
$ make test-shadow-m3-calibration-emit
planner parse: PASS  n_r=10 n_p=2  drive=2 ctrl=2 up=1 cpu=1 gpu=1 mem=1 link=2
```

10 resources (5 storage + 5 islands), 2 profile rows (one per measured
drive with singleton ownership), parse + open succeed. Counts by kind in the parsed
topology match the manifest's declarations.

## Reproduction

```text
cd c
make test-shadow-m3 test-shadow-observer-m3 test-shadow-store-m3 \
     test-shadow-registry-m3 test-shadow-m3-bridge test-shadow-m3-runtime \
     test-shadow-m3-topology test-shadow-m3-calibration-emit colibri \
     EXTRA_CFLAGS=-Werror

# rerun the capture and emit
cd ..
python c/tools/m3_shadow_calibrate.py \
  --manifest docs/results/m3-shadow-calibrate-A-manifest.json \
  --bytes 16777216 --samples 2 --inflight 1,2,4 \
  --cache-condition unknown \
  --emit-planner-text \
    docs/results/m3-execution-slice-7b-ext-2026-09-07.planner.txt
```

All seven pre-existing shadow test binaries plus the new
`test-shadow-m3-calibration-emit` plus the engine rebuild pass under
`-Werror -std=c11`.  The 7a-ext topology fixtures continue to pass
unchanged.

## Honesty log

The follow-up hash audit found that kind-specific topology payloads were not
included in the immutable profile hash. The planner now hashes ISA/capacity,
VRAM/P2P, parent-link, rate, bandwidth-class, and eligibility fields, with a
regression proving that a payload change changes the hash. This is a
determinism/integrity fix, not a hardware-capability claim.

- **Capture path is unchanged.**  The existing JSON capture format
  (`schema: m3-shadow-calibration-v1`) is preserved byte-for-byte; the
  new `manifest.islands` field is only present when the manifest
  declares islands and when the emit flag is set.  Downstream tooling
  that consumes the JSON is unaffected.

- **Planned GPU H2D/D2H measurement mode (`--capture transfer`) is
  deferred.**  The 7b-ext packet scope included an opt-in GPU
  measurement mode, but machine A has only an ineligible GTX 1080 and
  no compute islands worth measuring.  The plan is to add the
  transfer-capture mode when a target machine with an eligible GPU
  becomes available (B or C), or as part of a follow-up slice on a
  Linux CUDA box.  The current slice covers storage + CPU + memory +
  link measurement and end-to-end emit/parse; that is the local
  capability.

- **Controller / upstream profiles are never inferred.**
  When a controller or upstream has no direct measurement, the emitter
  declares its topology row but emits no profile row. The parser and
  opener preserve that partial-calibration state, while `path_valid()` and
  the predictor return `UNKNOWN` for a path requiring the missing resource.
  Same-controller and shared-upstream capacity therefore cannot silently
  enter the planner through a drive-rate extrapolation.

- **Sentinel and admission fixes in `c/m3_shadow_plan.c` are
  relaxation, not loosening.** A topology may have fewer profile rows than
  resources when measurement coverage is incomplete; no missing row is
  synthesized and any path needing it is `UNKNOWN`.
  INV-10 honesty still applies to every profile row that exists:
  `units_per_second_explicit > 0`, `sample_count > 0`, version match,
  topology hash match, and tail-percentile discipline on residuals.

- **Cache condition is `unknown`.**  This matches the existing Step 7
  calibration capture (see `m3-execution-step7-calibration-2026-09-07.md`).
  Flushing the OS cache before the capture is unsafe on a shared
  Windows host and would distort the warm-cache numbers that the
  planner's hot-tier scoring uses.  A future controlled-flush protocol
  remains `NOT_RUN`.

- **NOT_RUN sections.**  GPU H2D/D2H, P2P, QPI penalty, and same-controller
  contention are all `NOT_RUN` on machine A (no eligible GPU; the host
  has two NVMe drives on separate controllers; the upstream link is
  shared but not probed).  These stay on the open-evidence list for
  machine B / C / D where the hardware exists.

## Files touched (this slice)

- Modified: `c/tools/m3_shadow_calibrate.py`
- Modified: `c/m3_shadow_plan.c`
- Modified: `c/m3_shadow_plan_calib.h`
- Modified: `c/Makefile`
- New: `c/tests/test_m3_shadow_calibration_emit.c`
- New: `docs/results/m3-shadow-calibrate-A-manifest.json`
- New: `docs/results/m3-execution-slice-7b-ext-capture-2026-09-07.json`
- New: `docs/results/m3-execution-slice-7b-ext-2026-09-07.planner.txt`
- New: `docs/results/m3-execution-slice-7b-ext-2026-09-07.md` (this file)
