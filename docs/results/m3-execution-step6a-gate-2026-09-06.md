# MiniMax-M3 Step 6a — correctness recovery and gate reconciliation

## Scope

This report closes the remaining evidence work before Step 7. The supported
scope remains CPU-only, `S=1`, grouped-int4 `fmt==4`, `gs==64`, with the
bounded Step-6 executor opt-in. No storage planner or new scheduling policy is
introduced here.

Baseline code is `b06130b` (`34050827680`), which already contains the atomic
failure-publication fix and the blocking Archer/ASan/UBSan CI result. The
changes in this report add only test coverage and a test-only production-path
coordination seam.

## Remaining gaps addressed

1. **All preflight allocations.** The grouped executor preflight has eight
   owned allocations when the shared expert is enabled: the block, expert
   records, three routed arenas, shared output, shared gate, and shared up.
   `test_i4_grouped.c` now injects failure at offsets `0..7`, verifies every
   partial-cleanup path rejects the allocation, and then verifies a valid
   allocation succeeds.
2. **Production dispatcher interleaving.** `test_pipe_block.c` now gates the
   real `moe()` dispatcher immediately before its `m3_dag_task_ready()` call
   for the second routed task. A previously submitted routed task is held at
   its worker entry, the dispatcher reaches the readiness probe, the worker is
   released to publish failure, and the readiness result is checked after that
   publication. The test asserts no readiness acceptance and unchanged caller
   output. It uses atomic latches and no sleeps. The test-only fatal boundary
   returns normally after production drain, avoiding `longjmp` from this new
   path; the existing longjmp-based PIPE drain test remains covered separately.
3. **Lifecycle and completion-order evidence.** The grouped executor harness
   runs fused and unfused numerical task graphs at worker counts 1, 2, and 4
   when available. A deterministic gate runs both completion permutations at
   two workers and checks numerical output and canaries. The production PIPE
   failure/drain test continues to hold an active load while the failure path
   drains and joins the compute work before slot reuse.

## Commands and results

From `c/`:

```text
make tests/test_i4_grouped.exe tests/test_pipe_block.exe tests/test_m3_dag.exe
./tests/test_m3_dag.exe                                      PASS
./tests/test_i4_grouped.exe                                  PASS
./tests/test_pipe_block.exe                                  PASS
COLI_TEST_DISPATCH_INTERLEAVE_ONLY=1 ./tests/test_pipe_block.exe PASS
COLI_TEST_PARALLEL_FAILURE=1 ./tests/test_pipe_block.exe     PASS
```

The complete `test_pipe_block.exe` was repeated ten times without failure.
The grouped test reported:

```text
parallel expert task graph ok ... workers=1, workers=2, workers=4
repeated numerical completion permutations ok (fused, workers=2)
repeated numerical completion permutations ok (separate, workers=2)
parallel preflight cleans up every injected allocation failure
test_i4_grouped: ok
```

The production dispatcher regression reported:

```text
production dispatcher rejects readiness after published worker failure
test_pipe_block: dispatcher interleave ok
```

## Explicit limitations

The runtime does not expose an injectable thread-creation failure seam, and
the supported cancellation contract is not a separate production M3
parallel-executor API. This report therefore claims the tested allocation,
failure publication, active-load drain, task-team wait, duplicate-completion,
and numerical completion-order cases only; it does not claim untested
thread-create or unsupported cancellation failures.

The local MSYS2 GCC installation cannot provide ASan/UBSan or TSan runtimes.
The prior authoritative CI run passed Archer, ASan/UBSan, and the Linux suite;
a fresh CI run is required after this report's new test changes before the
race portion of Gate 6a is finally closed.

## Gate status

- **C: locally PASS, CI confirmation pending.** The mixed atomic-access fix is
  present, the production readiness/failure interleaving is exercised, all
  eight preflight allocation failures are covered, active PIPE failure/drain
  is covered, and fused/unfused worker-count and completion-order checks pass.
- **M: PASS within the supported scope.** The production dispatcher reaches
  the post-publication rejection and drains before the test returns; numerical
  task completion permutations preserve outputs and canaries.
- **P: NOT_PROMOTED.** No new performance claim is made. The existing
  all-resident limitation remains unchanged.

After a fresh blocking Archer/ASan/UBSan CI run and final reviewer check, the
plan may mark 6a C/M complete and permit Step 7. Step 7 itself is not started
by this report.
