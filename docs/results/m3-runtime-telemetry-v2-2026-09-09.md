# M3 runtime telemetry v2 — 2026-09-09

The live Windows M3 shadow seam now records an epistemic, hash-bound runtime
trace without changing the engine route or enabling admission.

## Evidence contract

Each row records:

- run ID, topology hash, planner-profile hash, and cache-condition label;
- model/forward/request identity, request type, byte size, six-component mask;
- complete resource path, explicit request concurrency, per-resource in-flight
  and byte occupancy;
- evidence state, single-observation confidence, completion result, and service latency.

Rows start `UNKNOWN` and become `OBSERVED` only after successful completion
with a complete resource path and all six weight/scale components. The report
accepts all five epistemic labels for accounting, but excludes every state
except observed successful rows from performance summaries and candidate
extraction.

## Candidate policy

`c/tools/m3_shadow_trace_report.py` accepts multiple trace files. A runtime
candidate requires at least two distinct run IDs for the same model/path/size/
request/mapping/cache/topology/profile scope, contiguous observed concurrency
levels, the declared p95/p99 tail policy, and no extrapolation. It reports both
p95 and p99 and deliberately emits no planner-rate claim from per-request
latency. All output is `CANDIDATE_ONLY` with
`active_admission_allowed: false`, and includes input/tool SHA-256 provenance.

## Profile and planner visibility

Attach a verified report to the portable machine profile with:

```text
coli profile --model <model-dir> --runtime-report <runtime-report.json>
```

The resulting `runtime_evidence` section is visible when that profile is
attached by `coli plan`. It is a bounded evidence summary, not planner
configuration: malformed or non-conservative reports become `UNKNOWN`, and
even valid candidates remain inactive until a separate, explicit promotion
gate exists. When a calibration capture is also attached, candidate topology
hashes must match it; otherwise the runtime attachment is `UNKNOWN`.

## Verification

- Focused C runtime seam, including concurrent issue/finish/trace snapshots: PASS.
- Shadow planner, observer, store, registry, bridge, topology, copy-scale, and residency tests: PASS.
- Calibration, calibration review/promote, runtime-report, and machine-profile tests: PASS.
- Rebuilt Windows shadow-off/on M3 replay: PASS; semantic stdout, exit status,
  and normalized engine trace remained unchanged, while live decision and
  runtime traces were emitted.
- TSan target was attempted but could not run because `clang` is not installed
  on this Windows machine; this is an environment limitation, not a positive
  scalability claim.

Active admission remains disabled.
