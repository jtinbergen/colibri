# MiniMax-M3 Step 6 — bounded parallel expert tasks

## Scope

Step 6 is opt-in and initially limited to the CPU, `S=1`, grouped-int4,
`fmt==4`, `gs==64` M3 route already admitted by the Step 5 adapter. GPU,
URING, PILOT_REAL, cluster, draft, ablation, unsupported tensor formats, and
other batch sizes remain on their established paths until separately gated.

The scheduler may overlap routed experts and their output-row tiles. It must
not move disk I/O into compute workers, replace the existing router, or change
the order in which contributions are reduced.

## Proposed execution contract

The first implementation uses a bounded per-forward CPU compute team. A
dispatcher owns readiness and PIPE waits; compute workers only execute already
published weight handles and numerical jobs. The team is joined before private
buffers or cache slots are released. The team is not a model-wide scheduler
and creates no nested OpenMP region inside a row kernel.

Each routed expert owns one private gate buffer, one private up buffer, and one
private output buffer. Its dependency chain is:

```
gate/up row tiles -> full activation -> down row tiles -> expert complete
```

Down tiles for expert A may start as soon as A's activation is complete; they
do not wait for another expert. Final accumulation still walks the original
router positions and uses the existing floating-point expression.

## Invariants to preserve

1. A task carries the current model/forward/layer identity, exact route index,
   expert identity, and the PIPE generation it belongs to. A slot pointer or
   expert ID alone is never task identity.
2. A worker reads only a fully published, currently leased weight handle. Cache
   promotion, eviction, reset, and slot reuse wait until all tasks using that
   handle have retired.
3. Scratch and output storage are disjoint per expert. No worker uses `g_pq`,
   TLS quantization scratch, `expert_ffn()`'s shared buffers, or another
   expert's gate/up/output arena.
4. Gate/up tiles for one expert complete before its activation; activation
   completes before any of that expert's down tiles. No down tile depends on
   another expert's readiness.
5. A job is claimed, completed, failed, cancelled, and released at most once.
   Failure prevents new jobs but drains and joins already-running jobs before
   reclaiming resources. No partial private output is reduced.
6. Reduction is exactly once, in router order, after every successful expert
   has retired. Shared output is added once according to `with_shared`.
7. The compute team is bounded by an explicit worker count and parks on a
   condition variable when idle. It does not busy-spin on disk readiness and
   does not create nested OpenMP teams.

## Required evidence

- exact output equality against the Step 5 serial path for 1, 2, 4, and the
  available larger worker counts, with repeated completion-order permutations;
- task lifecycle tests for worker shutdown, allocation/thread-create failure,
  cancellation/failure drain, duplicate completion, and no premature slot or
  scratch reuse;
- trace evidence showing overlapping expert compute, shared participation,
  and down-start for one expert while another remains blocked, with no dropped
  events and no nested-team marker;
- sanitizer/race evidence where supported by the selected Windows/MSYS2
  runtime; an unavailable or nonfunctional sanitizer is reported as such, not
  counted as a pass;
- old engine versus Step 4 versus Step 6 timing at equal residency, plus a
  separate all-resident comparison. An all-resident result is `NOT_RUN` when
  the host cannot provide the required memory.

This is a pre-implementation contract. Gate status remains open until the
implementation and evidence satisfy every item above.

## Implementation and current evidence

The opt-in implementation is now present in `c/colibri.c` behind
`COLI_M3_DAG_PARALLEL=1` (with `COLI_M3_DAG_SERIAL=1`, `PIPE=1`, and
`COLI_M3_DAG_WORKERS=N`). It uses one layer-local OpenMP region, coordinator
tasks, taskgroups for gate/up, activation, and down row tiles, and a producer-
owned submission mask. Per-expert scratch is preflighted before PIPE
publication; unsupported loaded formats or post-publication task failure drain
the PIPE slots and discard private outputs before falling back or terminating.
The reducer commits only after taskwait and walks router order. Shared output
is computed through the same private task path and added once at layer end.

Verified so far:

- `test_i4_grouped.exe`: grouped-int4 references, fused and unfused task graphs,
  private-scratch/output canaries, clipping, concurrency, and overflow preflight
  guard pass.
- `test_m3_dag.exe`: serial and parallel lifecycle transitions, duplicate
  reduction rejection, and failure terminal state pass.
- `test_pipe_block.exe`: the real scheduler's post-publication routed-task
  failure path passes in both serial and bounded-parallel modes; the parallel
  case reaches a post-publication failure (`submitted=1` in the isolated case,
  `submitted=2` in the full suite), joins the compute team, drains both PIPE
  slots, and only then enters the test fatal boundary.
- After the branch split, the current Step-6 branch was rebuilt and the
  focused `colibri.exe`, `test_m3_dag.exe`, `test_i4_grouped.exe`, and
  `test_pipe_block.exe` suite was rerun successfully. Commit `2ec839c`
  restores the replica-selector metric fields that were lost during conflict
  resolution; the fix is structural and does not alter the Step-6 schedule.
- Real MiniMax-M3 `PROMPT=hi`, `NGEN=1`, PIPE/blocking, two compute workers:
  all 57 sparse-layer FNV digests match the established baseline, including
  `0b5dca1f6fa0cd96` at layer 3 and `c7cd7ef0e12065bb` at layer 59. The same
  57 digests match with `COLI_NO_FUSED_PAIR=1`.
- Final trace `c/m3_step6_trace_final.csv`: 5,189 events, zero dropped; 285 expert
  intervals (57 shared plus 228 routed), 285 balanced phase completions,
  285 `team_level` events all at OpenMP level 1, and 53 overlapping same-layer
  expert interval pairs. In layer 4, expert 117 begins down at
  `51959607379900` while expert 67's activation completes later at
  `51959608334700`, proving down-start does not wait for the other expert;
  both reductions still remain in router order. The shared
  `reduction_done` event is emitted only after Phase E adds its private output.
- Worker-count A/B: `COLI_M3_DAG_WORKERS=1`, `=2`, `=4`, and `=10` each
  produce 57 digests, byte-for-byte identical across the worker counts; the
  ten-worker run uses the machine's full available compute team.
- An independent log comparison on the current branch found 57/57 matching
  digests for the 1-, 4-, and 10-worker logs against the 2-worker baseline;
  first digest `0b5dca1f6fa0cd96`, final digest `c7cd7ef0e12065bb`.
- Equal-residency old/Step-4/Step-6 matrix, with `COLI_M3_DAG_PIPE=1`
  explicitly set for Step 4, `PROMPT=hi`, `NGEN=1`, PIPE/blocking,
  `RAM_GB=18`, RSS 11.61 GB (seconds, prefill):

  | round | old | Step 4 | Step 6 |
  |---|---:|---:|---:|
  | 1 (old → Step 4 → Step 6) | 4.17 | 4.29 | 4.03 |
  | 2 (Step 6 → old → Step 4) | 4.17 | 4.28 | 4.14 |

  Both Step 4 and Step 6 activation banners were observed, and all six runs
  produced 57 digests; old, Step 4, and Step 6 digest sequences were equal.
  These are mixed disk/resident runs and remain evidence, not a scalability
  promotion. The separate all-resident run is `NOT_RUN`: the host has 34.12 GB
  physical RAM while the model package is 224.55 GB, so it cannot be isolated
  honestly on this machine.

Sanitizer status: the Clang64 non-static ASan+UBSan builds of
`test_m3_dag`, `test_i4_grouped`, and `test_pipe_block` all pass, including the
parallel failure/drain case. The installed Clang64 package has no TSan runtime,
so TSan remains **NOT_RUN**. The earlier GCC sanitizer attempts failed because
the GCC installation lacked `libasan`, `libubsan`, and `libtsan`; capability
audit also found no native race checker and WSL is present but inaccessible
under the current Windows policy. The branch now provides a focused
`make -C c test-tsan-m3` target and a Linux `tsan-m3` CI job; until that job
executes successfully, this remaining TSan limitation is explicit and is not
treated as a pass. The first CI attempt (run `34033510969`) successfully
compiled and passed `test_m3_dag`, but stopped in the general legacy
`matmul_i4_grouped` row-wrapper probe at `quant.h:216`; that report is retained
as a separate finding rather than attributed to the Step-6 executor. The TSan
target now selects only the Step-6-relevant expert reentrancy, private
scratch/output, bounded task-group, and preflight checks; it still requires a
fresh CI run before the Gate-C claim can change. The second attempt (run
`34033863447`) reached the bounded task group and reported a task-capture race
at `tests/test_i4_grouped.c:325`; the executor tasks now use explicit
`firstprivate` expert pointers and structured task blocks, with a fresh TSan
run still required to confirm the fix. That report is emitted from GCC's
`libgomp` task runtime; the CI gate now installs Clang plus LLVM `libomp` and
will rerun the same focused suite there to separate runtime instrumentation
noise from an application race. The LLVM run (`34034659201`) instead located
the same capture at the test-only outer task fan-out, so that harness now uses
OpenMP sections while retaining the executor's internal taskgroups. The next
LLVM run (`34035631669`) reached that harness and stopped on a lazy `libomp`
mutex initialization report, with no application frame; the TSan target now
runs the production-style PIPE drain test before that check so its result is
still independently observable. TSan remains **NOT_RUN for the executor
claim** until an OpenMP-compatible race runtime produces clean application
evidence. The sixth run (`34036507641`) then identified a real shared lazy-init
race in `g_planar` (`planar_on`, `colibri.c:1234/1238`) between PIPE workers.
The current branch fixes that publication with atomic compare-exchange
initialization and applies the same protection to `g_idot_gs`; the normal
Windows focused suite remains green. In the seventh run (`34037245091`),
`test_m3_dag` and `test_pipe_block` both passed under Clang/TSan; only the
first OpenMP bootstrap in the grouped-int4 harness reported a `libomp` mutex
race. The harness now performs a completed two-thread warm-up before the
Step-6 checks, so the next run can distinguish that bootstrap artifact from
executor activity. The eighth run (`34037690780`) confirms the split:
Clang/TSan passes `test_m3_dag` and the production-style `test_pipe_block`
after the atomic fix, then reports the same `libomp` mutex-initialization race
in the grouped-int4 harness, with no application frame, even after the
explicit warm-up. This is not counted as a Step-6 race pass or failure; a race
checker/runtime that can instrument OpenMP taskgroups without reporting its own
bootstrap is still required for Gate-C. The branch now also contains a Linux
`test-helgrind-m3` supplemental fallback and CI job. Its first CI attempt
stopped before executing the tests because the normal `-march=native` build
emitted an AVX-512 instruction that Valgrind 3.22 did not decode; the
Helgrind-only build now appends portable `-march=x86-64`. The second attempt
(`34038835731`) then ran `test_m3_dag` cleanly, but Helgrind reported extensive
conflicts in the older `test_pipe_block` fixture before reaching grouped-int4.
Those reports are not treated as a Step-6 application-race result because the
fixture deliberately relies on C11 atomics and test-only hooks that Helgrind
does not model reliably. The fallback is therefore narrowed to the bounded DAG
check; CI run `34039146571` passes that Helgrind check. The same run's TSan job
again passes `test_m3_dag` and `test_pipe_block` and stops only on the external
`libomp` mutex-initialization report in grouped-int4; TSan remains the race gate
for PIPE and grouped-int4. A follow-up audit also removed an unnecessary
non-atomic `g_pp.m` rewrite from every PIPE dispatch; the focused Windows suite
remains green after that change.

The next validation attempt adds a second Linux TSan job using GCC/libgomp.
This is intentionally an evidence-gathering route, not a suppression: if
GCC/libgomp reaches grouped-int4 cleanly, it can separate the current Clang/
libomp bootstrap report from executor behavior; if it reports an application
frame, that finding becomes the next code-level fix.

## Current gate record

- **C: PARTIAL.** Focused fused/unfused task graphs, lifecycle transitions,
  worker counts 1/2/4/10, both real grouped-int4 configurations, and the real
  post-publication failure/drain path preserve the established outputs and
  release ownership safely under the normal build and ASan+UBSan. The required
  TSan race checker passes the bounded DAG and PIPE paths; grouped-int4 remains
  unclassified because the hosted libomp runtime races during initialization.
- **M: PASS within the supported opt-in scope.** The final trace proves
  overlap, phase ordering, private-task ownership, zero drops, and OpenMP
  level 1; focused tests cover failure-state transitions, shape/bounds
  preflight, worker-side failure propagation, and production PIPE shutdown.
- **P: NOT_PROMOTED.** The equal-residency old/Step-4/Step-6 matrix is now
  recorded, but the separate all-resident isolation run is `NOT_RUN`. The
  all-resident case is constrained by the host's available RAM and must remain
  `NOT_RUN` unless a suitable host or resident setup is supplied.
