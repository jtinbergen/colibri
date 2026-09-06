# MiniMax-M3 step 4 — PIPE ready-first CPU reference

## Scope

`COLI_M3_DAG_PIPE=1` is an additional opt-in on top of
`COLI_M3_DAG_SERIAL=1`.  It is deliberately limited to the CPU, `S=1`,
pthread-PIPE, grouped-int4/group-size-64 M3 path.  GPU, URING, cluster,
PILOT_REAL, draft, ablation and mmap routes keep the legacy path.

After PIPE dispatch, shared computation runs once while workers load routed
experts.  The compute thread runs at most one routed expert at a time.  It
selects a ready expert first; if none is ready, it waits on a generation-bound
predicate covering every unfinished PIPE slot, then rescans.  Each routed
result is copied to a private buffer and reduction remains in original router
order.  Every published slot is acknowledged before promotion or slot reuse.

## Targeted evidence

`tests/test_pipe_block.exe` passes in spin and `COLI_PIPE_BLOCK=1` modes.  Its
ready-first latch holds A unpublished, enters the consumer predicate wait, then
publishes B; the production scheduler selection helper must return B while A
remains unavailable.  It also proves an already-ready task wins over pending
I/O.  The mode flag is atomic, so this test does not race a worker observing
the spin/block policy.  The existing 200-generation slot-content checks remain
part of the same test.

`tests/test_m3_dag.exe` passes its serial-order, shared-once, stale,
cancellation and failure checks.

On the local `minimax_m3_i4` grouped-int4 checkpoint, matched CPU-only
`PROMPT=hi NGEN=1 TEMP=0 PIPE=1 PIPE_WORKERS=2 COLI_PIPE_BLOCK=1 RAM_GB=18
COLI_RAM_OVERCOMMIT=1 AUTOPIN=0` runs produced `{tabular` and identical FNV-64
layer-output digests for all 57 sparse layers (3--59), with and without the
two step-4 flags.  The ready-first run took 3.98 s prefill versus 3.87 s for
the single legacy measurement; this is not a performance claim.

A matched `NGEN=3` pair also produced identical digests for all 171 sparse
layer forwards (prefill plus two decode forwards) and the same `{tabular`
continuation.  Both observed the same 7.9% LRU hit rate under cap 2.  Ready-
first measured 3.89 s prefill and 6.22 s decode; the legacy sample measured
5.61 s and 9.23 s.  This one pair is affected by normal run-to-run I/O and
CPU variance, so it demonstrates output equivalence and multi-generation
completion, not a speedup claim.

## Remaining gate work

The exact route fixture at `c/tests/fixtures/m3_step4_hi_routes.txt` covers a
real all-hit prefill.  It was derived from a complete, zero-drop route trace
for `PROMPT=hi`, pins all 228 routed experts (four for each sparse layer), and
was run with `RAM_GB=26 PIN_GB=8`.  Both legacy PIPE and ready-first PIPE
reported 0.000 s felt prefill wait, produced `{tabular`, and had identical
digests for layers 3--59.  The final decode-only hit summary resets counters
after prefill and therefore reads 0.0%; it is not a prefill residency measure.
The retained ready-first pinned trace has 228 `weight_pin` events across all
57 sparse layers, zero `load_queued`/`load_start` events, and zero drops.  It
therefore directly establishes all-hit residency for this prefill.

For a scheduler/DAG failure after PIPE publication, the supported ownership
argument is that the error path acknowledges every `q < nmiss` before it frees
private output or exits; workers publish readiness only after completing their
slot writes.  Thus no supported cache promotion, next dispatch or slot reuse
can follow a partial scheduler failure.  Fatal demand-I/O errors remain process
termination, not recoverable cancellation: they deliberately provide no
post-error reuse guarantee.  `test_pipe_block` latches a real worker-held A
slot, selects ready B, then proves the exact production drain helper reaches a
real not-ready wait before A is released, in spin and blocking modes.  Its
synthetic M3 `moe()` integration also executes the ready-first scheduler:
shared runs, B is selected and test-injected to fail through the real task-run
operation, and the production fatal branch must wait for held A before its
test-only fatal callback can run.  The test verifies both published slots after
that drain in both modes.  `test_m3_dag` supplies the paired task-run failure
evidence: lease release, no commit and duplicate rejection.  Live-I/O
cancellation/reset is unsupported in this phase.

The real all-hit and all-miss prefill checks were both repeated with
`COLI_PIPE_BLOCK=0` and reproduced the same 57 output digests and `{tabular`.
The all-miss spin run recorded 3.417 s felt wait.  Together with the matched
cap-2 three-token pair (mixed cache, prefill plus two decode forwards), this
closes the intended output/lifecycle modes.  The three-token pair is the
available true-PIPE A/B performance sample; it uses identical prompt, token
count, two workers, cache budget and CPU settings.  Its timing is explicitly
not treated as a speedup claim.

The mixed-cache row is covered by the matched three-token cap-2 pair above.
An attempted `PIN=auto PIN_GB=10` first-pass run loaded 313 historical experts
but still observed 0.0% pin hits for this prompt, so it is not used as all-hit
evidence.  The trace buffer was expanded from 512 to 2048 events so the route
fixture could be derived from the subsequent complete, zero-drop capture.
