# MiniMax-M3 Step 7b-0 — canonical machine profile

## Status

**IMPLEMENTED LOCALLY / NOT GATED / NOT PROMOTED.** This slice creates the
inventory and evidence bridge needed by `coli plan`; it does not claim that
Step 7 calibration or Gate 7 is complete.

## Delivered

- `c/machine_profile.py` defines `colibri-machine-profile-v1`.
- `coli profile --model <dir>` writes an atomic
  `<dir>/.coli_machine_profile.json` without scanning model shards or running a
  storage benchmark. `--output` supports a machine-level destination.
- An optional `--calibration <capture.json>` records bounded metadata from the
  existing `m3-shadow-calibration-v1` raw capture. Samples are not copied into
  the profile and are not promoted to admission limits.
- `coli plan` reads the model sidecar, when present, and renders a compact
  evidence summary. Placement calculations remain unchanged.
- Missing cache condition, shared-controller/upstream contention, NUMA, and
  GPU-transfer measurements remain `NOT_RUN`/`UNKNOWN`.

## Verification

```text
python -m unittest tests.test_machine_profile -v
  5 tests: PASS

python -m unittest tests.test_resource_plan -q
  65 tests: PASS

python c/coli profile --output <temporary-path> --json
  CLI smoke: PASS

python -m py_compile machine_profile.py coli resource_plan.py \
  tests/test_machine_profile.py
  PASS

git diff --check
  PASS
```

The bridge was also exercised against the current full M3 model directory:
`coli profile` wrote a profile with 16 logical / 10 physical cores, one socket,
32 GiB available RAM, one discovered GTX 1080, and storage discovery marked
`DISCOVERED` while the optional SSD probe remained `absent`. A following
`coli plan --machine-profile ... --json` loaded that profile and preserved the
planner warning that storage, shared-controller/upstream, NUMA, and GPU
transfer calibration are not admitted. No placement or admission bound was
changed by the profile.

The local profile is evidence plumbing only. A later 7b calibration step must
still establish cache condition, simultaneous controller/upstream contention,
tail criteria, and held-out validation before any predictor or admission bound
may consume it.
