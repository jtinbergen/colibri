# CUDA graph coarse-execution spike

Date: 2026-09-03

## Result

The prototype can capture and replay the existing DP4A resident sequence on
the GTX 1080 while preserving the existing kernel boundaries and output. It
does not improve production throughput in its current form.

The opt-in path is therefore a structural architecture PASS but a performance
FAIL. It remains disabled by default and falls back to the validated resident
path on capture, instantiate, or replay failure.

Matched 500-token production-style runs, with no CUPTI, no timing dump, and
the same model/configuration:

| configuration | warmup | timed | tok/s | GPU util samples | mean power |
|---|---:|---:|---:|---:|---:|
| DP4A baseline, graph off | 8.15 s | 127.28 s | 3.928 | 8.62% | 44.99 W |
| DP4A graph prototype | 17.44 s | 191.67 s | 2.609 | 7.05% | 44.88 W |

This pair is a clear regression of 33.6%. The absolute machine state was slow
relative to earlier runs, so the result is interpreted as a matched A/B, not
as a new absolute engine baseline. The output text was exactly identical:
2,143 characters in both responses.

## Current execution boundary

The current resident call is:

```text
input P2P
  -> dynamic metadata preparation / refresh
  -> activation quantization
  -> gate+up DP4A
  -> down-input quantization
  -> down DP4A
  -> weighted_sum_rows
  -> partial-result P2P
  -> ev_done
  -> resident_take: wait, sum_slots, sync, D2H
```

The arithmetic and reduction order are unchanged. In particular, the graph
does not fuse the quantization boundary or change the fixed-order FP32 sums.

Lifetime classification:

| lifetime | examples | current handling |
|---|---|---|
| model/device lifetime | device capability, geometry, scratch buffers, streams | initialized once or reserved at fixed caps |
| residency generation | expert weight/scales pointer tables | persistent device metadata, bulk refresh on change |
| layer/token | active expert count, weights, input/output, partial slot | host supplies/updates each call |
| device-internal | quantization, projections, activation, weighting | existing CUDA kernels, unchanged |

The minimum dynamic information is the active count and routing weights plus
the already-selected input/output/partial buffers and the current metadata
generation. Expert identities are represented by the persistent device pointer
tables; they are not embedded as host-pointer arrays in the captured graph.

## Prototype topology

For `COLI_CUDA_GRAPH=1`, `COLI_CUDA_DP4A=1`, SM61, and all three projections
using `fmt=4, gs=64`, one graph executable is cached per active expert count.
The graph contains the validated sequence:

```text
quantize input
  -> gate+up DP4A
  -> quantize down input
  -> down DP4A
  -> weighted_sum_rows
```

The following remain outside the graph because they are dynamic or were not
captureable on this path:

* input peer copy;
* persistent-metadata refresh, when needed;
* routing-weight H2D update;
* partial-result peer copy;
* completion event and `resident_take` reduction/synchronization.

A short debug run captured eight count variants (counts 1 through 8) and then
replayed them without capture or replay errors. The graph smoke test and the
existing nibble/batched DP4A tests passed.

## Why the graph did not win

The graph removes repeated kernel-submission bookkeeping for the five
resident kernels and replaces it with one `cudaGraphLaunch` for that internal
sequence. It does not remove the P2P copies, the layer completion boundary,
or `sum_slots`.

The production A/B shows that this reduction is not free on this Pascal
workload. The graph run had lower GPU utilization and nearly 1.5x timed wall
time. The likely architectural explanation is that the current fine-grained
host issue pattern permits useful CPU/GPU overlap, while the graph launch is a
coarser opaque submission with substantial Pascal-era graph launch/capture
overhead. This is an attribution hypothesis, not a claim that has been
proven by a new CUPTI run.

The important negative result is therefore precise: making the GPU submission
coarser does not automatically improve the layer makespan when the graph
still ends at the same partial-copy/completion boundary.

## Strategy comparison

| strategy | semantics | dynamic routing | SM61 risk | assessment |
|---|---|---|---|---|
| CUDA Graph replay | existing kernels and reductions unchanged | count variants; weights/metadata updated outside | graph launch overhead, capture limitations | implemented as opt-in prototype; performance FAIL |
| coarse batched host dispatch | existing kernels, fewer decisions/bookkeeping | naturally flexible | limited additional reduction because the path is already batched | useful only after measuring a narrower host bottleneck |
| persistent device executor | potentially one long-lived device boundary | flexible but complex | synchronization, occupancy, watchdog, dynamic-dispatch risk | defer; too large for this spike |

## API/launch accounting

The previous diagnostic trace recorded 7,960 resident calls in 199 decode
windows, 39,560 queued kernels from resident calls, 184,514 memcpy records,
and 536,613 CUDA API records. Those counts include diagnostic CUPTI/API
activity and are not production-time measurements.

For the graph path, the internal five-kernel sequence is represented by one
graph launch after each graph variant has been captured. Dynamic copies,
peer copies, completion events, `sum_slots`, and final synchronization remain.
Thus the graph cannot by itself eliminate the largest layer boundary or the
CPU fallback lane.

## Decision

```text
Current layer critical path:
  CPU/fallback work and the existing layer completion boundary remain unchanged;
  graph replay replaces only the internal resident submission envelope.

Avoidable in this prototype:
  five resident kernel submissions become one graph submission after warmup.

Probably unavoidable here:
  input/partial P2P, dynamic routing/metadata updates, resident_take wait,
  sum_slots, D2H, and the Qwen/GDN layer dependency.

Highest-value first intervention:
  do not enable this graph implementation for production; measure the
  host-visible overlap and graph-launch cost with a focused microbenchmark.

Predicted effect:
  current measured effect is -33.6% tok/s in the matched 500-token A/B;
  no positive production prediction is justified.
```

The future multi-island design principle remains valid, but this experiment
shows that the local executor must be tuned for the actual island and must
preserve useful CPU/GPU overlap. A future layer descriptor should therefore
describe ownership and completion, while each island may choose its own
submission mechanism rather than assuming CUDA Graphs are universally faster.
