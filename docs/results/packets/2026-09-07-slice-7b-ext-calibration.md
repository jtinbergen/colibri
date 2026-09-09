# Packet: Slice 7b-ext — calibration (storage + transfer)

Deliverable: extend the existing calibration pipeline (`c/tools/m3_shadow_calibrate.py`
+ `c/tools/m3_shadow_calibration_review.py`) to cover the new resource kinds
and link / GPU dimensions. Data only, no admission policy.

## 1. Scope and invariants

- Measure per drive: completion-latency distribution and aggregate throughput
  vs request size (64 KiB / 1 MiB / 16 MiB + actual expert tensor size) vs
  inflight 1 / 2 / 4 / 8 / 16, with cache state explicitly labelled.
- Measure per GPU / link: H2D, D2H, P2P per pair, cross-socket penalty where
  applicable. All measurements honour `DIRECT` mode if available; state the
  mode in the capture.
- `/s` units mandatory on every rate field. `units_per_second_explicit`
  stays non-zero; the validator rejects profiles that drop the field.
- p99 capture with at least 16 samples per cell (existing
  `m3-execution-step7-calibration-p99-16-2026-09-07.json` only had 2).
- All bytes/s, ms, lat fields carry explicit units.
- Honest measurement only: missing hardware → `NOT_RUN`, never extrapolation.

## 2. Tasks

### 2.1 Storage extension

1. Run the existing calibration tool at inflight 8 and 16 for the
   machine-A drives, with `cache-condition` labelled.
2. Capture same-controller contention: write a small workload where C: and
   E: are forced through one synthetic controller in the parser (this is a
   parser-only test, not a real same-controller measurement on this host;
   same-controller capture stays `NOT_RUN`).
3. Same for shared-upstream capture: `NOT_RUN` on A, parser-fixture only
   on B / C / D where the upstream is real.

### 2.2 Transfer extension

1. Add `--capture transfer --bytes <n> --device gpu0` mode to the tool. For
   A (no eligible GPU) the tool exits with `NOT_RUN gpu_island`.
2. Capture H2D and D2H timing for every eligible GPU on the host. For B /
   C / D this is the only way to seed 7c-1 with real transfer rates.
3. P2P pairs: enumerate `(gpu_i, gpu_j)`; if P2P is supported, capture
   end-to-end bytes/ns; else mark `NOT_RUN p2p`.
4. Cross-socket penalty: only on B / C. Capture as `latency-add per-1000-bytes`
   or similar unit-explicit field.

### 2.3 Profile output

The output JSON gains:

```json
{
  "machine": "A",
  "drives": [...],
  "gpu_islands": [...],
  "links": [...],
  "memory_domains": [...],
  "units": {"rate": "B/s", "latency": "ns", "bytes": "B"}
}
```

The reviewer (`m3_shadow_calibration_review.py`) gains an `--allow-p99`
flag (already exists) and a new `--allow-tail-multiplier` so the
illustrative 2x review cannot be confused with an admission bound.

### 2.4 Artifact

`docs/results/m3-execution-slice-7b-ext-<date>.md` with embedded JSON, raw
captures under `docs/results/m3-execution-slice-7b-ext-*-<date>.json`,
per-machine summary tables.

## 3. Gate 7b-ext

- C: parser refuses profiles without `units_per_second_explicit`,
  without `topology_hash`, with zero samples, or with mismatched topology
  hash.
- M: every numeric field labelled; missing hardware is `NOT_RUN`.
- P: no perf claim. Profiles are inputs to 7c-1 / 7c-2.
