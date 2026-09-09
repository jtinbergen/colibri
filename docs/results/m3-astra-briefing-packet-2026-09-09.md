# MiniMax-M3 local heterogeneous DAG — independent review packet

Date: 2026-09-09  
Purpose: independent architecture and strategy review by a high-capability reviewer (Astra)

## Decision to review

Determine whether the local heterogeneous DAG should be advanced toward measurable
ROI on this machine, or whether the current approach should be stopped, narrowed,
or redesigned.

The relevant target is not merely “DAG code executes”. The target is lower
end-to-end latency or higher throughput through useful local resource placement:

```text
disk -> RAM staging -> CPU/GPU compute island -> next layer/window
```

The machine is a Windows desktop with an i5-12600K, 32 GB RAM, one GTX 1080,
one fast NVMe and one slower NVMe mirror. Network distribution is out of scope;
local NUMA, multi-GPU, CPU/GPU and storage islands are in scope.

## Verified evidence

### Existing CPU C+E baseline

File: `m3-layer-staging-ab-baseline-cap9-2026-09-09.txt`

The requested cap was 9, but runtime memory pressure reduced the effective cache
to 8 experts/layer (456 LRU experts, 14.5 GB). The useful rotating result was:

| metric | observed |
|---|---:|
| rotating median throughput | 0.46 tok/s |
| p95 request | 104.80 s |
| median hit rate | 43.3% |
| expert disk service | 133.609 s/request |
| felt expert wait | 50.218 s/request |
| expert matmul | 9.872 s/request |
| PIPE I/O-active | about 53.5 s |
| PIPE compute-active | about 9.85 s |
| PIPE concurrent overlap | about 3.30 s |

Telemetry is internally coherent for this legacy route: phase shares sum to
approximately 100%, and compute-active is nonzero.

### M3 bounded parallel-task DAG

File: `m3-dedicated-cpu-c-e-mirror-dag-parallel-2026-09-09.txt`

The log contains:

```text
[M3_DAG] active: bounded parallel tasks, private scratch, fixed reduction order
```

Therefore the DAG executor really ran. It was nevertheless a hybrid run:
layer 3 reported `parallel path ineligible` and fell back safely. The DAG changed
task orchestration, not expert residency or the amount of model data fetched.

| metric | observed |
|---|---:|
| rotating median throughput | 0.44 tok/s |
| p95 request | 106.44 s |
| median hit rate | 45.1% |
| resident LRU experts | 513 / 16.3 GB in that run |
| expert disk service | 128.604 s/request |
| felt expert wait | 53.048 s/request |

This is not a clean ROI comparison because the cache capacity differed from the
baseline. More importantly, DAG task compute was not included correctly in the
PIPE profiler: `compute-active=0` while DAG activity was present, and phase shares
could exceed 100%. This run proves activation, not performance benefit.

### Real cross-layer RAM prefetch (`PILOT_REAL`)

File: `m3-layer-staging-ab-pilot-real-cap9-2026-09-09.txt`

This is the closest existing implementation to “load the next layer into RAM
while the current layer computes”. It is value-preserving prefetch, not a new
model-output path.

The exploratory run reached 513 LRU experts / 16.3 GB and observed:

| metric | baseline cap-8 | pilot exploratory |
|---|---:|---:|
| rotating median throughput | 0.46 tok/s | 0.44 tok/s |
| p95 request | 104.80 s | 112.53 s |
| median hit rate | 43.3% | 56.6% |

The higher hit rate did not translate to lower latency. This is a warning that
real prefetch can consume queue capacity, evict useful data, or serialize work.
The exact cap-8 pilot is now complete and makes the comparison controlled:

| metric | baseline effective cap-8 | pilot exact cap-8 |
|---|---:|---:|
| rotating median throughput | 0.46 tok/s | 0.44 tok/s |
| p95 request | 104.80 s | 112.92 s |
| median hit rate | 43.3% | 54.2% |
| expert wait | 50.218 s | 50.667 s |

The current `PILOT_REAL` policy is therefore negative for end-to-end ROI on
this workload: hit rate rises, but throughput falls about 4% and p95 request
latency worsens about 7.8%. This rejects the current naive staging policy; it
does not reject every possible fixed-residency or layer-window policy.

Source: `m3-layer-staging-ab-pilot-real-cap8-2026-09-09.txt`.

## Fixed hot-store and primary-only storage control

The fixed hot-store experiment used the same 27.8 GB RAM budget, `PIN_GB=7`,
`USAGE_SAVE=0`, effective LRU cap 4/layer, and no pilot prefetch. The hot set
was 219 experts / 7.0 GB; the remaining LRU was 228 experts / 7.3 GB.

| metric | dual-drive, 84/16 split | primary-only |
|---|---:|---:|
| rotating median throughput | 0.43 tok/s | 0.42 tok/s |
| p95 request | 109.47 s | 117.36 s |
| median hit rate | 38.9% | 38.9% |
| expert wait | 54.009 s | 55.623 s |

This is a small positive signal for the second drive (about +2.4% throughput
and -6.7% p95), but it is below the 10% promotion threshold. The runs were
sequential and used `DIRECT=0` with retained page cache, so page-cache order is
a confounder; this is evidence of at most modest local benefit, not a general
dual-drive scaling claim.

Sources:

- `m3-hot-residency-pin7-cap8-2026-09-09.txt`
- `m3-hot-residency-pin7-cap8-primary-only-2026-09-09.txt`

The runtime profiler still reports aggregate I/O queue depth and aggregate
service/wait. The mirror probe reports isolated primary/mirror throughput and
the startup byte split. The profiler now additionally exposes per-window
runtime source bytes and read counts. A
decode smoke reported primary 97.622 GB / 12,260 reads and mirror1 21.308 GB /
2,676 reads (17.9% mirror share), followed by 97.208 GB / 12,208 reads and
21.117 GB / 2,652 reads (17.8% mirror share). This confirms that the mirror
participates in real serving reads. Per-drive queue depth, p95/p99 completion,
and shared-controller counters are still required before a live tail-aware
drive selector can be promoted.

## Independent Astra review

Astra's independent review agrees with the controlled interpretation:

- retain the bounded task DAG as an opt-in/reference path, but do not promote
  it for ROI;
- repair DAG phase accounting before drawing overlap or compute-efficiency
  conclusions;
- treat the current real-prefetch result as negative, partly because the
  layer handoff waits for all speculative loads while the headline expert-wait
  counter does not include that barrier;
- if another experiment is justified, use a fixed hot-expert RAM-residency
  policy under the same total memory budget, selected from a separate trace;
- evaluate live queue-/tail-aware drive selection separately, with explicit
  ownership, deduplication, lease, reservation, and fallback invariants.

Astra also noted that the existing three-request arms are sufficient to reject
the current candidate, but not to establish a precise p99 or scalability claim.

## Hardware observations

- The mirror probe consistently measures the primary NVMe around 4.4–5.1 GB/s
  and the slower mirror around 0.95–1.0 GB/s.
- The current placement is deterministic hash routing, typically about 82–84%
  of bytes on the primary and 16–18% on the mirror.
- Queue depth and tail-relevant service telemetry are recorded, but current
  runtime routing does not dynamically choose a drive from live queue depth or
  p99 latency. Queue depth is presently evidence, not a control signal.
- The workload is strongly expert-I/O-bound and the current GPU tier was not
  useful at this residency/transfer granularity.

## Current interpretation

Verified:

1. The local DAG executor can run safely for an eligible subset of M3 blocks.
2. Safe per-layer fallback exists for ineligible blocks.
3. The current task DAG alone does not materially change disk traffic.
4. Naive real cross-layer prefetch increases hit rate but has not yet shown an
   end-to-end benefit; the exploratory result is negative.

Not yet verified:

1. A corrected DAG profiler result with per-layer active/fallback counts.
2. A useful layer-window policy that stages only sufficiently predictable/hot
   experts and does not reduce I/O queue utilization.
3. Live tail-aware drive routing.
4. Benefit from multiple local compute islands after transfer and residency
   costs are included.

## Questions for Astra

Please return separate sections for verified facts, recommendations, blockers,
and unproven assumptions.

1. Given the evidence, should the current bounded parallel-task DAG be retained,
   rewritten, or treated as an instrumentation/reference path only?
2. Is layer-window/RAM staging still a credible ROI path on a 32 GB host with a
   roughly 1 GB/s mirror drive, or does the pilot result argue against it?
3. What is the smallest technically meaningful experiment that can decide this:
   fixed hot-window residency, predictive staging, or live queue/tail-aware drive
   selection?
4. What comparison controls are mandatory before accepting a positive result?
5. Should drive selection be changed from deterministic hash placement to a
   queue-/tail-aware scheduler, and what safety invariant prevents duplicate
   reads, starvation, or worse tail latency?
6. What is the explicit stop condition — the evidence that should make us give
   up on this machine/configuration and redirect effort to a larger-RAM,
   multi-GPU, or multi-island host?

## Proposed acceptance rule

Do not promote any strategy into active planning merely because its hit rate
improves. Accept a strategy only if, under equal effective cache and workload,
it produces at least 10% lower p95 request latency or expert wait, or at least
10% higher rotating throughput, with no correctness or p99 regression. Otherwise
record a negative result for that strategy and configuration.

## Files

- `m3-dedicated-cpu-c-e-mirror-dag-parallel-2026-09-09.txt`
- `m3-layer-staging-ab-baseline-cap9-2026-09-09.txt`
- `m3-layer-staging-ab-pilot-real-cap9-2026-09-09.txt`
- `docs/minimax-m3-unified-compute-islands.md`
- `docs/m3-profiler-contention-invariants.md`
