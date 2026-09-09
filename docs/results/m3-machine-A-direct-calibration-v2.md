# Machine A direct-I/O calibration v2

Date: 8 September 2026. This is a machine-specific shadow calibration, not an
active admission approval.

## Capture

- Backend: Windows `FILE_FLAG_NO_BUFFERING` (`win32-no-buffering`).
- Cache condition: `cold` was requested through direct/unbuffered reads.
- Expert request: `31,850,496` bytes.
- Scalar levels: 1, 2, 4, 8, 16.
- Shared-upstream matrix levels: 0, 1, 2, 4, 8, 16 per drive, bounded by
  total inflight 16.
- Matrix cells: 26; cells with both drives active: 16.
- Operations per worker: 2; samples per cell: 16.
- Range reuse: enabled because each selected shard is only one expert payload;
  this is recorded and prevents promotion to a physical-capacity claim.
- Quantiles: conservative nearest-rank over request completions.

## Observed results

- Shared upstream 901 review candidate: total inflight 2 under the observed
  p95 <= 2x baseline policy, with minimum per-drive retention threshold 0.50
  and normalized fairness threshold 0.80. This supersedes an earlier run that
  reported total inflight 4.
- Candidate minimum per-drive retention: approximately `0.851`.
- Host DRAM copy p50: `8,136,256,906 B/s`; this remains `host_memcpy` evidence,
  not storage-controller DMA evidence.

## Generated artifacts

- Raw capture: `m3-machine-A-direct-calibration-v2.json`
- Planner text: `m3-machine-A-direct-calibration-v2.planner.txt`
- Review: `m3-machine-A-direct-calibration-v2.review.json`
- Machine profile: `m3-machine-A-direct-profile-v2.json`
- Full model configuration: `m3-machine-A-direct-full-model-config-v2.txt`

The full-model generator emitted 7,296 expert identities and 14,592 copy rows.
The C parser/open gate passed with 7 resources and 6 profile rows.

## Remaining gates

- Review status is `CANDIDATE_ONLY`; active admission is disabled.
- Range reuse means the capture is not a clean disjoint-range physical
  capacity proof.
- Storage-to-memory DMA, ready-time validation, held-out planner validation,
  sanitizer evidence, and stronger-model final review remain open.
- The conservative generated storage policy remains one request and one expert
  payload per resource; measured candidates have not been promoted into bounds.
