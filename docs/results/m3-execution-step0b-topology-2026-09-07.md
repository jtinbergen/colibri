# MiniMax-M3 Step 0b — per-machine topology capture (gate report)

Date: 7 September 2026. Status: PASS C/M, P `NOT_RUN` (data-model slice).

## Deliverables

| Path | Purpose |
|---|---|
| `docs/results/m3-topology-A-2026-09-07.json` | Machine A topology capture (filled) |
| `docs/results/m3-topology-A-2026-09-07.md` | Companion markdown |
| `docs/results/m3-topology-B.template.json` | Machine B template |
| `docs/results/m3-topology-C.template.json` | Machine C template |
| `docs/results/m3-topology-D.template.json` | Machine D template |
| `docs/results/m3_topology_validate.py` | Validator script |

## Method

`m3_topology_validate.py` enforces:

- explicit `units` block with at least `rate`, `latency`, `bytes`;
- top-level required keys (`schema_version`, `machine_id`,
  `capture_date`, `evidence_basis`, `units`, `compute`,
  `memory_domains`, `storage`, `controllers`, `interconnects`,
  `machine`, `operating_environment`, `open_evidence_gaps`);
- non-NOT_RUN placeholder values are rejected on numeric fields
  (`compute.cpu.physical_cores/logical_cores`, `memory_domains[*].total_bytes`,
  `compute.gpu[*].vram_bytes`); the literal string `"NOT_RUN"` is the
  only acceptable placeholder;
- drive / controller / interconnect ids unique within their kind;
- machine `eligibility` is a structured object with `cpu_island` and
  `gpu_island` keys;
- `open_evidence_gaps` is a list with `id / field / status / rationale`
  entries; `status` is `NOT_RUN` while unmeasured.

Exit codes: 0 PASS, 1 PASS-WITH-NOTES (calibration marked `NOT_RUN` where
appropriate), 2 FAIL, 3 usage error.

## Validation results

```text
$ python m3_topology_validate.py m3-topology-A-2026-09-07.json \
    m3-topology-B.template.json m3-topology-C.template.json \
    m3-topology-D.template.json
PASS-WITH-NOTES m3-topology-A-2026-09-07.json
  - drive:103: calibration NOT_RUN
PASS-WITH-NOTES m3-topology-B.template.json
  - drive:b01: calibration NOT_RUN
PASS-WITH-NOTES m3-topology-C.template.json
  - drive:c01: calibration NOT_RUN
PASS-WITH-NOTES m3-topology-D.template.json
  - drive:d01: calibration NOT_RUN
EXIT=1
```

PASS-WITH-NOTES is the expected state for unmeasured machines. The
machine A artifact has 6 explicit `open_evidence_gaps` entries; the
template artifacts surface their open gaps in the same field. Every
`status: NOT_RUN` is reflected both in the per-drive `calibration`
field and in the `open_evidence_gaps` list — no fabricated capacity.

## Gate 0b

- C (schema): PASS for all four artifacts. `m3_topology_validate.py`
  is the formal schema check; both filled and template artifacts
  parse, have unique ids, and pass the placeholder rule.
- M (honesty): PASS — every numeric field carries explicit units via
  the `units` block; missing measurements are labelled `NOT_RUN`; no
  extrapolation to `bytes/s` from `bytes` without `/s`.
- P (performance): NOT_RUN — this slice is data-model only; no engine
  behavior change; no perf claim.

## Honesty log

- Machine A evidence is taken from live `Get-CimInstance` calls plus
  the existing Step 7 calibration capture and the latency tables in
  `docs/minimax-m3-comprehensive-report-2026-09-05.md`.
- The drive:103 entry (Kingston XS1000 USB SSD) has `calibration:
  NOT_RUN` because no planner profile is needed for a USB-attached
  external SSD; the artifact records it for completeness.
- The validator template-tolerance rule is conservative: any controller
  id containing the substring `FILL` causes the validator to skip the
  drive-id-existence check on the drives that controller references,
  matching the template use case where a future filler will add
  matching drive rows. A filled artifact with placeholder text in
  non-template locations would still be rejected by other rules
  (`units`, `kind`, `model`).

## Stop conditions

- 0b complete. The schema is the substrate for Slice 7a-ext (parser
  extension) and Slice 7b-ext (calibration). They depend on
  `m3_shadow_plan_calib.h` having the kinds present, which is the
  next implementation step.
- No engine code change in this slice.

## Files touched (this slice)

- New: `docs/results/m3-topology-A-2026-09-07.json`
- New: `docs/results/m3-topology-A-2026-09-07.md`
- New: `docs/results/m3-topology-B.template.json`
- New: `docs/results/m3-topology-C.template.json`
- New: `docs/results/m3-topology-D.template.json`
- New: `docs/results/m3_topology_validate.py`
- New: `docs/results/m3-execution-step0b-topology-2026-09-07.md` (this file)

No edits to `c/colibri.c` or `c/m3_shadow_plan.*` in this slice.
