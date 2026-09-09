# M3 profiler contention invariants

This document defines the contract for the shared-controller and
shared-upstream calibration capture. It is a measurement contract, not an
admission approval.

## Ownership and timing

- The profiler owns only its worker threads and read observations. It does not
  select runtime sources, publish I/O, or change planner admission.
- Every cell has one common start barrier. Its wall time is measured from that
  barrier to the latest worker completion.
- A cell's total inflight count is the sum of its per-drive queue depths. A
  drive with queue depth zero has no worker in that cell.
- Per-drive throughput is computed from that drive's request count and the
  common cell wall time. Aggregate throughput is the sum of those rates.
- Tail latency is reported both across all workers and per drive. Fairness is
  reported over active drives only.

## Matrix and range rules

- Shared groups are swept over an explicit Cartesian product of per-drive
  queue-depth levels, subject to a bounded total-inflight limit.
- `(0, 0, ...)` is not a measurement cell. Cells with only one active drive
  are retained as controls; cells with every declared drive active are the
  contention observations.
- A source range is not reused across matrix cells or samples unless the
  caller explicitly enables range reuse while cache state remains `unknown`.
- Matrix cells are bounded by the configured level set and total-inflight
  limit; the profiler rejects unbounded or invalid requests.

## Evidence and promotion

- Buffered reads, unknown cache state, or range reuse are recorded in the raw
  capture and cannot be described as physical controller/DMI bandwidth.
- The raw matrix is evidence only. A review step may derive an observed
  candidate under a predeclared tail-latency policy, but it must not silently
  turn an observed queue depth into a safe admission bound.
- To claim a storage-to-memory limit, the capture must exercise the production
  DMA/read-to-ready path. A host `memcpy` probe is a separate DRAM observation.
- Active planner admission remains disabled until direct-I/O/cache
  classification, held-out validation, and the required runtime review gates
  pass.

## Runtime evidence contract

The production M3 shadow seam is an evidence collector, not an admission
path. Each completed row records the run identity, topology/profile hashes,
cache-condition label, request type and size, complete resource path, path
occupancy vector, component mapping, service latency, and completion result.

Epistemic state is conservative: a row starts `UNKNOWN` and becomes
`OBSERVED` only after successful completion with a complete path and all six
weight/scale components. `UNAVAILABLE`, `THEORETICAL`, and `NOT_APPLICABLE`
are accepted labels for profile/report inputs but never create capacity.

The runtime report may combine independent trace files. It emits a
`CANDIDATE_ONLY` row only when at least two run IDs repeat the same scope,
the observed concurrency levels form a contiguous downward prefix, and the
declared tail policy passes. It reports both p95 and p99 but emits no
planner-rate claim from per-request latency. Candidate output is bound to the
input trace digests and remains inactive until a separate explicit review and
promotion gate.
