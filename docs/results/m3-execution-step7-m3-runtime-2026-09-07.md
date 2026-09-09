# MiniMax-M3 Step 7 live shadow seam — 7 September 2026

## Result

The opt-in runtime observer is wired into the real M3 `expert_load()` path.
The tiny fixture was run once with the observer disabled and once with it
enabled. Both runs passed the M3 oracle (prefill `24/24`, decode `20/20`),
and the replay test accepted equal semantic engine output, exit status, and
normalized `COLI_TRACE` lifecycle events.
The enabled run wrote decision records and a bounded runtime trace at process
exit.

Reproduction from `c/`:

```text
make PYTHON=/c/Python313/python.exe test-shadow-m3-runtime-replay
```

The convenience target invokes
`tests/test_m3_shadow_runtime_replay.py`, using the checked-in synthetic
two-path configuration in `../docs/results/m3-execution-step7-fixtures/`.
The harness pins the same serial-contract/bounded-parallel settings for both
runs (`OMP_NUM_THREADS=16`, `COLI_M3_DAG_SERIAL=1`,
`COLI_M3_DAG_PARALLEL=1`, `COLI_M3_DAG_PIPE=1`, eight compute workers), so the
occupied-to-counterfactual decision is not a host-scheduling accident.
The replay supplies `COLI_M3_SHADOW_SYNTHETIC_NEED_DELTA_NS=1000000000`
explicitly; this is fixture input, not a claim that the runtime has inferred a
DAG deadline. On the eligible production M3 serial DAG path, the runtime
instead captures one monotonic `need=now` timestamp when the current consumer
is ready except for its missing weights. It preserves that timestamp for the
block and does not refresh it during later predictions. PIPE/parallel paths
remain UNKNOWN until a provider can establish readiness after their shared
compute/preflight dependencies.
Performance/profiling lines are intentionally ignored by the comparison;
oracle counts, stable engine output, and exit status are checked.

The seam also emits an optional bounded correlation trace when
`COLI_M3_SHADOW_TRACE` is set. Each CSV row joins request/forward/generation,
consumer node and need time, actual drive plus its expanded
drive/controller/upstream path, one consistent post-registration inflight/
byte-occupancy snapshot for every resource on that path, a six-component completeness mask, the planner's
predicted ready time, issue time, and the observed `expert_load()` return
time/result. The production hook marks all three weight tensors and all three
quantization/scale tensors only when they resolve to one actual replica;
incomplete or mixed mappings remain unknown. This is separate
from the counterfactual decision log so actual completion evidence cannot be
mistaken for a planner prediction. The replay now requires a non-empty
decision log and at least one correlated trace row.

The offline [`m3_shadow_trace_report.py`](../../c/tools/m3_shadow_trace_report.py)
consumes that CSV and reports ready-time error, p50/p95/p99 error, and
actual/predicted/uncertainty-guarded deadline misses per drive/resource path.
The recorded tiny evidence report is
[`m3-execution-step7-runtime-report-2026-09-07.json`](m3-execution-step7-runtime-report-2026-09-07.json):
two successful rows with complete six-component masks, no deadline misses,
and a maximum absolute ready-time error of 54,894 ns. This is a two-row
semantic smoke test, not a performance or calibration conclusion.

## Scope and limits

- Shadow mode is opt-in: the build contains the seam, but it remains inert
  without `COLI_M3_SHADOW_CONFIG`, a topology hash, and a replica-drive map.
- If the current consumer's readiness fact is unavailable, the runtime still
  refuses to invent a deadline and correctly reports no decision. The
  observed `need=now` path is limited to the eligible serial M3 DAG contract;
  other paths remain UNKNOWN.
- Equivalent-copy rows use an explicit stable `COLI_M3_SHADOW_MODEL_ID`; the
  runtime falls back to the process-local model pointer only when a configured
  copy identity is unavailable.
- Context is copied per PIPE job and pushed/restored per OpenMP iteration. A
  load observation token is stack-owned and is finished exactly once on normal
  returns.
- The production tiny replay now declares two complete equivalent copies on
  independent controller/upstream paths for both observed experts. With
  replica 0 occupied, the atomic shadow operation records replica 1 as the
  planner's counterfactual candidate while the actual engine path remains
  replica 0. This proves production-seam candidate ranking and output
  transparency, but not that the active engine has routed a real read to the
  counterfactual copy.
- Missing deadlines, mixed tensor residency, direct multi-replica striping,
  mmap physical traffic, and io_uring loads are not treated as known planner
  observations. No hardware contention or performance claim is made.
- The trace is only populated for a complete physical replica path; it does
  not make mmap, striped direct reads, or PIPE/parallel readiness known.

This is meaningful live-integration evidence, but it does not close Gate 7:
complete physical mappings, real drive/controller calibration, hosted
sanitizer evidence, and the stronger-model final diff review remain required.
