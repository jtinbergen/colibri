# M3 execution-DAG step 1: observability

Date: 5 September 2026. Step 1 adds observability only: it does not alter M3
task order, OpenMP configuration, residency policy, kernel selection, routing,
or numerical operations.

## Delivered interface

`PROF=1` now reports the following exclusive M3-attention components in both
the ordinary profile and the per-run `[PROF]` report:

- projection;
- norm/RoPE/KV-cache update;
- MSA block-score scan (decode, `S=1`);
- MSA block selection (decode, `S=1`);
- score/softmax/value core; and
- output projection.

The MSA scan and selection counters are subsets of the existing inclusive
attention wall time, but are disjoint from the other new attention components.
They are deliberately not added to the pre-existing top-level `PROFILE`
accounting total, so that report's semantics remain unchanged.

`COLI_TRACE=<csv-path>` enables trace format `colibri-trace,v1`. It uses at
most 64 fixed per-thread buffers of 512 events. The hot event path has no I/O,
heap allocation, mutex, or scheduler action. At process exit it writes a CSV
header followed by:

```text
ns,forward,thread,kind,layer,eid,generation,resource
```

`ns` is a monotonic timestamp; `forward` is the active forward sequence;
`generation` identifies a pthread PIPE load batch; and `resource` is the
PIPE slot, routed-row ordinal, or MSA block-count according to `kind`.
Implemented event names are `forward_begin/end`, `msa_scan_begin/end`,
`msa_select_begin/end`, `route_ready`, `weight_pin`, `weight_lru`,
`load_queued`, `load_start`, `load_complete`, `consumer_wait_begin/end`, and
`compute_start/end`. `weight_pin` and `weight_lru` are the v1 forms of
weights-acquired. Gate/up, activation, down, reduction and release events are
not yet reachable because the current executor does not expose those stages as
separate jobs; their absence is explicit rather than inferred.

An exit message reports records written and dropped records. Overflow includes
both a full per-thread buffer and a thread for which no one of the 64 buffers
is available. A trace reporting dropped records is a partial timeline, not a
dependency proof. Scope v1 is one active model forward per process; it must
not be used to infer a total order between concurrent model forwards.

## Validation

All commands ran from `colibri-dev/c` on the Step 0 host and used its existing
tiny fixture. The tiny fixture is a numerical/scheduling smoke test, not a
Naples, Rome, or full-checkpoint performance result.

```powershell
& 'C:\msys64\usr\bin\bash.exe' -lc "cd /c/Users/jaapj/.lmstudio/models/colibri-dev/c && make colibri"
$env:COLI_TRACE='..\docs\results\m3-execution-step1-oracle-trace.csv'
& 'C:\Python313\python.exe' '.\tests\test_m3_tiny.py' --binary '.\colibri.exe' --snap '.\m3tiny_i8' --ref '.\ref_m3.json'
& 'C:\msys64\usr\bin\bash.exe' -lc "cd /c/Users/jaapj/.lmstudio/models/colibri-dev/c && make tests/test_pipe_block.exe && ./tests/test_pipe_block.exe"
```

The trace-on oracle passed exactly as trace-off: prefill `24/24` and decode
`20/20` positions. The rebuilt PIPE regression printed `test_pipe_block: ok`.
The oracle trace contains 344 events, 19 MSA scan pairs, 19 MSA selection
pairs, and zero dropped records.

For a fixed 20-token replay (`SNAP=.\m3tiny_i8`, `REF=.\ref_m3.json`,
`REPLAY=1`, `PROF=1`, `IDOT=0`, `PIPE=1`) the two trace-off runs were 1295.40
and 1148.91 tok/s; the two trace-on runs were 1121.78 and 1126.18 tok/s. The
short run is noisy, but this measures approximately 7.5% lower throughput with
full trace recording. Tracing is therefore off by default and must remain off
for speed benchmarks.

The retained example trace is
[`m3-execution-step1-tiny-trace.csv`](m3-execution-step1-tiny-trace.csv). Its
last run contains 300 events and zero drops:

| Evidence | Observed events |
|---|---|
| Initial PIPE miss and felt wait | 2 `load_queued`, 2 `load_start`, 2 `load_complete`, 1 matching `consumer_wait_begin/end` |
| Resident hit | 40 `weight_lru` |
| Decode MSA lifetime | 20 `msa_scan_begin/end` and 20 `msa_select_begin/end` |
| Expert compute boundaries | 42 `compute_start/end` |

The wait is for the first prefill's first pipe slot: its `load_queued` event
precedes `consumer_wait_begin`, and both worker `load_complete` events precede
`consumer_wait_end`. Subsequent decode forwards acquire the same two experts
through `weight_lru`; each has one complete MSA scan followed by one selection
pair. This establishes the required explainable miss, residency hit, and MSA
timeline within the tiny fixture's scope.

## Gate 1 status

| Gate | Status | Evidence / limitation |
|---|---|---|
| C | PASS, tiny CPU scope | Trace-on M3 oracle remains `24/24` prefill and `20/20` decode; rebuilt PIPE regression passes. New timers are exclusive components of inclusive attention and do not alter the old total. |
| M | PASS, tiny PIPE scope | The retained trace shows a waited load, LRU resident hits, complete decode MSA pairs, lifecycle order, and zero drops. URING load-queued/wait events are emitted too, but were not exercised on this Windows host. |
| P | PASS, bounded-trace smoke scope | The measured short-run trace-on slowdown is about 7.5%; it is opt-in and disabled by default. No full-model, long-context, Naples, or Rome overhead measurement has been made. |

Step 1 can now provide baseline observability for the MSA-parallelization and
task-executor steps. Do not treat the tiny MSA times (below the three-decimal
reporting resolution) as a long-context MSA cost measurement.
