# CPU/GPU overlap diagnostic

The layer CSV now contains the following request-local cumulative fields:

- `cpu_window_ms`: host interval from completion of `qt_issue` through the
  CPU fallback and shared expert work;
- `gpu_event_ms`: CUDA-event duration for the resident GPU work;
- `gpu_cpu_overlap_ms`: intersection of the host CPU window with the GPU
  event interval;
- `gpu_sync_ms`: host wait measured around the resident CUDA synchronization;
- `qt_take_ms`: caller-visible `qt_take` wall time;
- `issue_to_complete_ms`: host wall interval from `qt_issue` entry through
  completion of `qt_take`.

The GPU interval is anchored at the host issue timestamp for this diagnostic.
That makes the overlap estimate useful for ranking, but it is not a profiler
replacement: CUDA event timestamps and host timestamps have different clocks.

## Initial short matched diagnostic

Both traces used DP4A, fixed 3,642-slot placement, `QTIER_LFRU=0`, no timing
dump, and 16 generated tokens.  They are diagnostic traces, not final
throughput measurements.

| metric | current | global-hot |
|---|---:|---:|
| GPU hitmix | 69.0% | 93.2% |
| CPU window | 50.129 ms/token | 19.651 ms/token |
| GPU event work | 17.046 ms/token | 22.016 ms/token |
| CPU/GPU overlap | 9.115 ms/token | 10.906 ms/token |
| direct `qt_take` | 4.859 ms/token | 9.662 ms/token |
| issue→completion | 62.981 ms/token | 37.051 ms/token |
| throughput | 1.024 tok/s | 1.044 tok/s |

The result has the expected shape: global-hot increases GPU work and
`qt_take`, but removes enough CPU fallback work that the combined expert
completion window becomes shorter.  The small throughput difference is not
statistically meaningful yet; a longer matched run is still required.

## Overlap-aware objective

`simulate_makespan.py` estimates each layer as:

```text
cpu = shared_cost + nonresident_routes * fallback_cost
gpu = resident_routes * gpu_cost
overlap = measured_overlap_fraction * min(cpu, gpu)
completion = cpu + gpu - overlap + measured_boundary_overhead
```

The layer completion estimates are summed because Qwen layers are sequential.
The objective can therefore prefer a placement that leaves some routes on the
CPU when that keeps CPU and GPU completion times balanced.  This is still an
offline ranking model; the next runtime experiment must validate it.

The simulator can emit the selected `makespan_balanced` set as a QTH1 heat
file with `--heat-out`.  That file is intended for a fixed-placement runtime
experiment with `QTIER_LFRU=0`; it is not yet a production scheduler.

## Balanced-placement runtime check

The first fixed `makespan_balanced` runtime check used the same short 16-token
diagnostic shape.  It is not a throughput result because the run is short and
the warmup state varied, but the phase movement is unambiguous:

| metric | current | global-hot | makespan-balanced |
|---|---:|---:|---:|
| GPU hitmix | 69.0% | 93.2% | 82.5% |
| CPU window | 50.129 ms/token | 19.651 ms/token | 28.517 ms/token |
| GPU event work | 17.046 ms/token | 22.016 ms/token | 536.219 ms/token |
| direct `qt_take` | 4.859 ms/token | 9.662 ms/token | 512.442 ms/token |
| issue→completion | 62.981 ms/token | 37.051 ms/token | 549.527 ms/token |

The regression is concentrated in late layers 36–39, where GPU sync waits
were roughly 31–35 ms/token.  This exposes a missing term in the first
overlap-aware simulator: per-layer CPU/GPU overlap is insufficient when GPU
work queues behind earlier layer work.  The next simulator must model a
device-queue/backlog timeline across sequential layers and tokens, including
the completion boundary, before it ranks a new placement.

## Per-call backlog trace

`analyze_overlap.py` consumes `QTIER_OVERLAP_FILE` and writes machine-readable
JSON plus an optional normalized call CSV. It reports individual call
windows, per-token and per-layer aggregates, and a diagnostic list-scheduling
projection for one, two, or three identical GPU islands. The projection is
not a runtime model: it uses measured host issue timestamps and CUDA event
durations, and cannot establish cross-island PCIe or CUDA-stream behavior.

The matched 16-token DP4A traces used fixed placement and `QTIER_LFRU=0`:

| metric per traced decode token (15 traced; 16 output) | current | global-hot |
|---|---:|---:|
| measured throughput | 1.137 tok/s | 1.157 tok/s |
| host trace span | 146.20 ms | 117.07 ms |
| CPU window | 45.44 ms | 17.06 ms |
| GPU event work | 16.02 ms | 21.57 ms |
| direct `qt_take` | 4.43 ms | 10.32 ms |
| inferred GPU outstanding at take | 0.41 ms | 4.91 ms |
| calls with inferred outstanding work | 37 / 600 | 387 / 600 |

The short pair is not a final throughput benchmark, but its attribution is
clear. Global-hot removes about 28.4 ms/token of CPU fallback window while
adding about 5.5 ms/token of GPU event work and 4.9 ms/token of inferred work
still outstanding when `qt_take` begins. The GPU is not idle in this policy;
it is more frequently on the completion-critical side of the boundary. This
is the measured form of the CPU/GPU load-balancing effect.

For the previously problematic tail, layers 38/39 moved from
`1.549/1.678 ms` CPU-window per traced token to `0.746/0.726 ms`, but their
`qt_take` moved from `0.108/0.112 ms` to `0.173/0.222 ms` and their inferred
outstanding GPU work rose from `0.008/0.000 ms` to `0.050/0.099 ms`.

The first timeline projection currently reports no additional *serial* queue
wait because the host issue gaps are wider than the measured event durations.
That does not contradict the outstanding-at-take result: the latter is based
on the per-call inferred interval and is sensitive to the host/CUDA clock
anchor, while the list scheduler is a deliberately conservative arrival-rate
model. A future exact version needs synchronized CUDA/host timestamps or a
CUDA-side trace of stream completion.

Example:

```text
python c/tests/analyze_overlap.py --input trace.overlap.csv \
  --out trace.overlap.json --calls-out trace.calls.csv --gpus 1 2 3
```

The next runtime experiment should therefore target scheduling/queue
placement with a longer matched trace, not replace the resident placement
policy based on hit rate alone.

## Same-clock instrumentation result

`COLI_CUDA_TIMELINE=1` now adds no stream callback and no blocking event query.
It records a host-clock lower bound at resident issue and an upper bound after
the already-required home-stream synchronization. The trace exposes these as
`gpu_end_host_lower_ms` / `gpu_end_host_upper_ms` and corresponding reduction
bounds. GPU start bounds are obtained by subtracting the CUDA event duration;
overlap is reported conservatively as a lower/upper interval.

Two stronger cross-clock mechanisms were tested and rejected for this workload:
`cudaLaunchHostFunc` added severe stream perturbation, and `cudaEventQuery` in
the take path likewise changed the Pascal run from sub-millisecond resident
events to multi-millisecond events. Neither is suitable for performance
measurement. The current bounds are therefore deliberately broad but
non-invasive. Exact narrow intervals require a lower-overhead driver/Nsight
trace or a separately validated polling mechanism; placement and scheduler
decisions must wait for that validation.

## CUPTI clock validation

The standalone [cupti_clock_probe.cu](D:/src/colibri/c/tests/cupti_clock_probe.cu)
uses CUPTI activity records rather than callbacks in the application stream.
On the GTX 1080 it produced 128/128 kernel records. Across 32 host/CUPTI
calibration samples the median offset was `0 ns`, with a `3.45 us` spread.
The host synchronization interval was `7.7 us`; CUPTI reported `7.8 us` for
the same interval. The activity trace covered `0.965 ms` of GPU execution,
with mean kernel duration `1.546 us`, mean queued-to-start latency `15.082 us`,
and submitted-to-start latency `0.207 us`.

This validates CUPTI timestamps as a viable same-clock source on this machine.
The result is still a probe result, not a Qwen result: the next instrumentation
slice should enable CUPTI activity collection in a diagnostic CUDA build and
join its absolute kernel records to the existing per-call host trace. Runtime
placement and scheduling decisions remain postponed until that join is shown
to be stable on the actual resident-expert path.

## Joined Qwen resident timeline

The diagnostic backend is built with `build_cuda_cupti.bat` and selected without
replacing the normal DLL:

```text
set COLI_CUDA_DLL_PATH=D:\src\colibri\c\coli_cuda_cupti.dll
set COLI_CUDA_CUPTI=1
set COLI_CUDA_CUPTI_FILE=D:\src\colibri\c\tests\cupti_qwen.csv
set COLI_TIMERS=1
```

The collector records concurrent kernels, host/device and peer copies, and
runtime/driver API activity. `host_*_ns` columns are CUPTI timestamps translated
to the host monotonic domain using the startup calibration. It performs no
stream callback, event query, or scheduler change. The normal backend remains
selected when `COLI_CUDA_DLL_PATH` is unset.

`join_cupti_timeline.py` joins kernels by queued timestamp inside each
resident-call issue interval, while separately listing kernels that execute in
the complete issue-to-take window. The latter includes kernels queued by an
earlier call and therefore exposes GPU backlog. It also reports D2H bytes,
device-copy bytes, API activity, queue-to-start latency, and the final issued
kernel end relative to `qt_take`.

Example:

```text
python c/tests/join_cupti_timeline.py \
  --overlap c/tests/cupti_qwen_dp4a_2.overlap.csv \
  --cupti c/tests/cupti_qwen_dp4a_2.csv \
  --out c/tests/cupti_qwen_dp4a_2.joined.json
```

The first real resident-path validation captured 33 calls from a short DP4A
run: 165 kernels were issued by those calls, all 33 calls had a measured D2H,
and 198 kernels executed in the call windows. The extra 33 window executions
are the reduction/take-side kernels, not evidence of extra expert launches.
The issued kernels totalled 14.47 ms of GPU execution; this trace was made
with diagnostic timers and CUPTI enabled, so it is an attribution artifact and
not a performance result.

The requested 200-token DP4A capture completed successfully. It contains 199
complete decode token windows and 7,960 resident calls (the final sampled
output token does not require another complete forward/MoE window), with
39,560 issued kernels and 47,472 kernel executions intersecting call windows.
The collector recorded 99,642 kernel, 184,514 memcpy, and 536,613 runtime API
records. The joined calls account for 64.81 MiB of D2H data and 123.67 MiB of
device-to-device data. Issued kernels used 6,808.07 ms of aggregate device
execution, while 1,889 calls had issued GPU work extending into their take
window; the summed extension was 1,350.54 ms.

Across the joined calls, mean queue-to-start was 567.8 us, p95 796.4 us, and
maximum 7.26 ms. These values are diagnostic observations under CUPTI and
must not be compared to the normal no-CUPTI throughput run. The machine-
readable artifacts are `cupti_qwen_dp4a_200_v2.csv`,
`cupti_qwen_dp4a_200_v2.overlap.csv`,
`cupti_qwen_dp4a_200_v2.routing.csv`, and
`cupti_qwen_dp4a_200_v2.joined.json`.

## Resident operation DAG analysis

`analyze_resident_dag.py` performs the next offline-only attribution pass. It
joins absolute CUPTI host-normalized activity with resident call windows,
maps API cbids through the installed CUPTI headers, maps memcpy correlations
back to their `cudaMemcpy*` caller, and writes representative call DAGs, a
complete 40-layer decode window, kernel breakdowns, memcpy size/direction
histograms, API breakdowns, and conservative queue-wait attribution.

```text
python c/tests/analyze_resident_dag.py \
  --overlap c/tests/cupti_qwen_dp4a_200_v2.overlap.csv \
  --cupti c/tests/cupti_qwen_dp4a_200_v2.csv \
  --out c/tests/cupti_qwen_dp4a_200_v2.dag.json \
  --summary-out c/tests/cupti_qwen_dp4a_200_v2.dag.md
```

The trace explains the resident envelope structurally: 7,912 calls with GPU
work issue five kernels each (39,560 total), while `sum_slots` is a sixth
kernel executing in the issue-to-take windows. A typical resident window also
contains about 11 memcpy records and 32.8 CUDA API records. The recurring
small transfers are six pointer-table H2D updates plus a small routing-weight
update; the 8 KiB P2P copies carry the activation to the resident island and
the partial result back home. Five event records and two elapsed-time queries
per window are diagnostic timer activity from `COLI_TIMERS=1`.

The resulting DAG JSON and Markdown report are diagnostic artifacts: CUPTI
collection substantially perturbs this small-kernel workload. The queue
analysis can distinguish time covered by prior GPU work from uncovered wait,
but uncovered wait remains a combined stream/dependency, driver/API, host,
and instrumentation category. A no-CUPTI matched run is required before
assigning any predicted ms/token gain.
