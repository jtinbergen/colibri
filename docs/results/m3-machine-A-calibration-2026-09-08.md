# Machine A M3 calibration and shadow configuration

Date: 8 September 2026. The generated artifacts are valid for shadow parsing
and replay. Active source admission remains disabled.

## Generated artifacts

- Manifest: `m3-machine-A-calibration-manifest-2026-09-08.json`
- Raw capture: `m3-machine-A-calibration-2026-09-08.json`
- Planner rows: `m3-machine-A-planner-2026-09-08.txt`
- Full model configuration: `m3-machine-A-full-model-config-2026-09-08.txt`
- Machine evidence profile: `m3-machine-A-profile-2026-09-08.json`
- Tail review evidence: `m3-machine-A-calibration-review-2026-09-08.json`

The full-model generator validated the calibration manifest hash and both safetensors replicas, found 7,296
expert identities, emitted 14,592 copy rows, and consumed the measured
planner rows instead of synthetic rates. The C parser/open gate passed with
7 resources and 6 profile rows.

## Capture contract

- Expert request size: 31,850,496 B, the actual M3 expert payload.
- Load levels: 1, 2, 4, 8, and 16; 16 samples per cell.
- I/O mode: buffered OS reads on Windows.
- Cache condition: `unknown`; no cache flush or physical-I/O claim.
- Range reuse: enabled only because the shard cannot provide disjoint ranges
  for the entire QD16 matrix. This is recorded in the raw capture.
- Storage bounds in the generated configuration: one request and one expert
  payload per resource. This is an explicit conservative shadow policy, not a
  measured capacity limit.

## Measured evidence

- Drive 101, controller 201, and drive 102, controller 202 each have
  explicitly targeted large-request profiles.
- Upstream 901 has an explicitly targeted two-drive group profile, so shared
  upstream behavior is represented as evidence rather than inferred from a
  singleton drive.
- Same-controller contention remains `NOT_RUN`: this machine has one drive
  on each NVMe controller.
- Host DRAM copy probe: 8,285,812,143 B/s p50 over 16 samples for a 256 MiB
  prefaulted buffer. This is `host_memcpy` evidence only; it is not a
  storage-controller DMA measurement and is not yet on the storage read path.
- The separate p95/2x review reports only observed candidates (controller 201:
  2, controller 202: 2, upstream 901: 4); those values were not promoted to
  admission bounds.
- GPU transfer and NUMA-specific measurements remain `NOT_RUN`.

## Promotion status

The configuration is acceptable as a machine-specific shadow artifact and
for parser/replay integration. It must not be used to claim active source
selection or end-to-end performance: cache classification, held-out
ready-time validation, direct-I/O equivalence, same-controller contention,
to-memory DMA, and sanitizer/review gates remain open. The production runtime
continues to construct admission with an untrusted policy and falls back to
the existing engine path.
