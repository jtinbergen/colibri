# MiniMax-M3 Step 7 — planner in shadow mode

## Scope

Step 7 introduces a planner that **proposes** reads on real traces without
changing reads, placement, routing, or compute. It predicts availability and
consumer-wait, never hitrate as a proxy success measure. It is observer-only:
its existence, scope, parsing failures, or `UNKNOWN` outputs must not alter any
existing path, order, byte, or output.

The supported scope follows 7a/7b/7c/7d of
[`docs/minimax-m3-execution-plan-2026-09-05.md`](../minimax-m3-execution-plan-2026-09-05.md):

- 7a: explicit topology + bounded data model + per-resource counters + bounded
  queue/bytes admission caps. OS-discovery is permitted as a helper, never a
  precondition. Invalid or contradictory topology yields `UNKNOWN` and the
  existing read path is left untouched.
- 7b: calibration of per-drive completion latency and aggregate throughput
  versus request size and in-flight count; contention across drives on the same
  controller / upstream. Profiles carry their own topology/config hash, sampled
  sizes, latency distributions, and **explicit `/s` units** on bandwidth. A
  bytebudget is never a bandwidthbudget.
- 7c: a pure predictor (`predict(snapshot, request, candidate, dispatch_time)`)
  derived from those profiles, plus a deterministic, event-driven decision
  pipeline. Numerical replica of the existing predictor is the fluid simulation
  that fits calibration residuals; the bounded policy is the five-step local
  proxy already specified and never a global optimizer.
- 7d: deterministic fixtures under a fake monotone clock + fixed service
  curves + fixed event queue. No sleep, no wall-clock timing assertion, same
  numeric precision across every fixture.

Hardware-specific P-claims (six- and seven-drive topologies, real
`NOT_RUN` measurements, cross-controller contention) are explicitly **not**
made in this step. `NOT_RUN` is reported where hardware is absent or
calibration is unsound; fixtures prove the mechanism, not a hardware win.

`COLI_PLANNER_SHADOW=1` opt-in. With the variable unset, the planner is
compiled out of the read path entirely.

## Proposed execution contract

### Insertion point

The shadow planner is a thin observer in front of `ColiExpertStoreOps.lookup`
and `.release` (and, for the new advisory flow, `.prefetch`). The existing
`lookup`/`release` results are returned unchanged to the existing callers
(`c/deepseek_v4.c:4349/4361/4471/4562/4681/4718/5026/5257/5287/5680/5690/5699`,
plus the parallel and pipe paths that share the same vtable). The observer
**never** mutates:

- the order of routings, expert selections, or reductions;
- the placement, eviction, or residency of expert weights;
- the bytes the disk read actually issues;
- the weight handle or lease already held by the caller;
- the caller's view of completion, failure, or cancellation.

This is the *only* way shadow semantics satisfy the user's "shadow aan/uit geeft
dezelfde route, reads, plaatsing en output" Gate-7 requirement.

### Module layout (proposed, not committed)

```
c/m3_shadow_plan.h            -- immutable types + documented contracts
c/m3_shadow_plan.c            -- topology/profile parser, predictor, decision
c/m3_shadow_plan_log.h        -- per-decision log record layout
c/m3_shadow_plan_calib.h      -- synthetic + measured profile representation
c/m3_shadow_m3_bridge.c/.h    -- pure M3 load-snapshot adapter
c/tests/test_m3_shadow_plan.c -- deterministic 7d fixtures
docs/results/m3-execution-step7-invariants-2026-09-06.md     -- this document
docs/results/m3-execution-step7-gate-2026-09-06.md          -- gate evidence
```

The planner module owns all its state behind a single opaque handle. The
observer supplies the snapshot on `lookup` entry and tears it down on
`release`. Memory is owned by the planner module; the engine never frees a
planner allocation through its own allocator.

### Snapshot model

A snapshot is a fully immutable view of:

- parsed topology: drives, controllers, upstream links with their
  `max_inflight`, `max_inflight_bytes`, calibration-profile references;
- provider drive-only paths are deterministically expanded through the declared
  parent chain before admission; contradictory or cyclic chains are rejected;
- parsed weight copies: model/tensor identity, byte length, drive;
- each request's identity (`model_id`, `forward_id`, `generation`,
  `consumer-need` time, demand/prefetch flag, enqueue time, chosen and
  alternative copies so far, state, reserved resources).

The planner takes a copy before evaluating any candidate. Two evaluations of
the same `(snapshot, request, candidate)` produce byte-identical decisions.
Echte completions can update a private `n_completed[request_id]` counter only;
they never rewrite the planner's `remaining_bytes` ledger for a real request.

### Predictor

Per resource `r`, the service profile yields a startup duration
`S_r(size_class)` and a transfer rate `C_r(load_class, size_class)`. With
`n_r` active byte-flows through `r`, request `i` receives rate
`min_r(C_r / n_r)` over its full path; load_class includes startup-stage
issued requests, `n_r` only counts flows that have actually entered the
transfer phase. Rates are recomputed at every dispatch / startup-end /
completion event. Bytes integrate forward against an event queue; concurrent
events resolve by request ID. Numerical integration precision and time
quantisation are fixed per planner version and shared across every fixture.

The first predictor is the conservative fluid simulation. Bad held-out
accuracy blocks activation and demands a model review — it does not silently
add hidden producer heuristics.

A completion predicted to zero bytes while the real completion has not been
observed returns `OVERDUE / UNKNOWN` for the affected candidates. Other,
independent groups remain planable.

When calibration declares the optional conversion profile, I/O completion is
not yet `weight_ready_time`: flows enter one bounded serial conversion stage
in `(io_ready_time, request_id)` order. Its duration and residual are part of
the immutable profile hash; a missing duration class or conversion bound
overflow returns `UNKNOWN`. With no conversion profile, the storage-only model
is retained for backward-compatible fixtures.

### Decision

The bounded policy is exactly the five steps in the plan:

1. Plan only demand requests; prefetch is left to 8b. Sort by
   `consumer_need`, then enqueue time, then stable request ID. Apply an
   age-priority bound before forwarding.
2. Evaluate every candidate source with a complete profile; keep the drive
   *and* every shared resource in the score. Compute a defer candidate on
   the next predicted resource completion without reserving.
3. Exclude candidates without a complete profile or with hard budgets that
   cannot fit on the proposed dispatch moment. Score is the sum of predicted
   consumer-wait including the same uncertainty margins, evaluated on the
   union of the candidate-relevant admitted requests plus the new request.
   Independent requests outside that union are irrelevant.
4. Admit only when every hard resource and buffer bound fits. If the head
   fits nowhere or defer wins, advance to the next waiting request whose
   candidate paths are independent of every candidate path of that head.
   Stay within the bounded queue. Even when no candidate meets its deadline,
   schedule the best feasible one and record the miss.
5. Recompute on enqueue, completion, failure, and DAG readiness. Event-driven
   only; never poll on predicted times. Defer is bounded: for an
   age-priority request with currently available capacity, the defer
   candidate expires and the earliest weight-ready wins on score. Predicted
   completion **never** certifies that a real read is finished or that its
   budget may be released.

### Logging

Per decision, the planner logs:

- snapshot hash + profile hash + planner version;
- request and consumer identifiers;
- candidate set and resource paths;
- occupancy before/after the hypothetical admission;
- predicted `weight_ready_time` and `consumer_need`, uncertainty margin;
- impact on existing reads;
- chosen candidate (or `DEFER`/`UNKNOWN`) and reason code.

A future step links these to real dispatch / completion / weight-ready events
using the same IDs. Until then the log is the shadow's only observable;
counterfactual gain stays a prediction. Hypothetical completions never appear
as measured data.

## Invariants to preserve

1. **Shadow is opt-in.** With `COLI_PLANNER_SHADOW` unset, the planner is
   compiled out. `lookup`, `release`, `prefetch`, `stats` returns are
   byte-identical to the corresponding pre-Step-7 paths.
2. **Reads and placement do not change.** `shadow on/off` produces the same
   routing, the same expert selections, the same disk-read bytes, the same
   weight leases, the same reductions, and the same numerical output for
   every fixture and every replayed trace.
3. **No mutation outside the planner handle.** `predict`, `record_completion`
   and friends write only to planner-owned storage. The engine never observes
   the planner; the planner never observes engine storage beyond what the
   observer hands it.
4. **Predictor purity.** `predict` is a pure function of the immutable
   snapshot, the request, the candidate, and the proposed dispatch time. No
   I/O. No global state mutation. No future-replay information. Same inputs
   ⇒ same outputs, within planner-version numerics.
5. **Counterfactual containments.** Hypothetical completions update the
   predictor's own `remaining_bytes` copy and release capacity for other
   *predicted* flows; they do not change the engine's real counters or
   readiness. Real completion does not retroactively "fix" a hypothetical
   prediction.
6. **Bounded work.** Simulator events are bounded by the per-snapshot queue
   and byte caps declared in 7a. Overflow returns `UNKNOWN`; the function
   never busy-loops and never blocks waiting for a real completion to
   validate a prediction.
7. **Determinism.** Replay of the same snapshot under the same profile hash
   and planner version produces the same decision log byte-for-byte, with
   the same candidate ranking, the same tie-break, and the same
   occupancy/impact values.
8. **No global timer.** Wall-clock dependencies are forbidden in the
   predictor and the decision step. Only the fake monotone clock supplied
   per fixture, plus future-replay snapshot timestamps, advance the
   simulator.
9. **Failure / cancellation are independent.** A `FAILED` or `CANCELLED`
   request never occupies the simulator ledger after the event has been
   recorded on the real side. `OVERDUE / UNKNOWN` may apply only to
   candidates that the real side still considers inflight.
10. **Hardware is honest.** A claim of measured capacity, contention, or
    latency is only made against a recorded profile with explicit samples,
    units, and topology hash. Anything else is `UNKNOWN`, and the missing
    six- or seven-drive machine is `NOT_RUN`, not extrapolated.

## Required evidence (pre-impl contract)

The implementation is not started. The following evidence is required before
any "shadow on/off equal" claim, and every item is acknowledged in the gate
report that closes the step:

- A pre-recorded baseline trace (`PROMPT=hi`, `NGEN=1`, PIPE/blocking,
  grouped-int4 M3 route) is replayed with `COLI_PLANNER_SHADOW=0` and
  `=1`. The route digest list, expert selection counts, byte totals, and
  `redaction_done` ordering are byte-equal across the two runs.
- The 7d fixtures pass: two drives with latency asymmetry, four drives on
  one shared 500 MiB/s controller with candidate admission, six + seven
  drives across three controller groups with optional upstream sharing,
  contention under sustained load, age-priority under sustained demand,
  unknown profile absence, missing replica, deterministic tie-break, and
  "completion in predictor must not mark real readiness". Each fixture
  names its published profiles, request set, expected decision log, and
  expected occupancy vector. Calculations are checked with a 1 ns
  tolerance on shared time arithmetic.
- A repro command per fixture plus an output capture under
  `docs/results/m3-execution-step7-fixtures/<name>/{inputs,expected,actual}`.
- Honest gate status:
  - **C:** shadow on/off equality verified against the recorded baseline;
    fixtures simulate the bounded policy without mutating real state.
  - **M:** every 7d fixture's expected log matches; replay is bit-exact;
    counterfactual completions never change real occupancy or readiness.
  - **P:** predictive accuracy on a held-out prompt set is reported with
    explicit acceptance thresholds and uncertainty coverage, separated by
    drive group. Missing six- or seven-drive hardware and any unsound
    calibration remain `NOT_RUN`; no microbenchmark is treated as a
    scalability claim.

## Implementation status (updated 2026-09-07)

The plan-level GO recorded below was followed for the first implementation
slice. The following files now exist:

- `c/m3_shadow_plan_calib.h`, `c/m3_shadow_plan_log.h`,
  `c/m3_shadow_plan.h`, and `c/m3_shadow_plan.c`;
- `c/tests/test_m3_shadow_plan.c` and
  `c/tests/test_m3_shadow_plan_observer.c`;
- `c/m3_shadow_store.c/.h` plus store-forwarding and registry-boundary tests;
- `c/m3_shadow_m3_bridge.c/.h` plus a pure M3 snapshot-adapter test;
- the focused Makefile targets and
  `.github/workflows/m3-step7-shadow.yml`;
- deterministic fixture input/expected/actual captures under
  `docs/results/m3-execution-step7-fixtures/`.

This slice is intentionally still **NOT GATED**, but the live boundary has
since been added. The opt-in M3 runtime seam wraps the real `expert_load()`
route under `COLI_M3_SHADOW_RUNTIME`; the tiny production replay verifies
shadow-on/shadow-off semantic stdout, exit status, normalized engine trace,
and non-empty request-ID-joined decision/runtime evidence. This is still a
small two-expert replay, not proof of full-model route/lease/read equality,
mixed residency, hardware contention, or performance promotion.

The bounded queue also supports an optional calibrated `queue_age_ns` bound.
The fake-clock fixture proves that an aged request outranks newer demand and
that aged ties are deterministic. The bound is part of the profile hash.

This is the pre-implementation contract required by `colibri-dev/AGENTS.md`.
Remaining closure conditions are:

1. A reviewer at or above the high-risk model tier for the final diff. The
   available review paths surface `coder → coder ESCALATE → pro-coder`. If
   no stronger model approves the final diff before merge, the missing
   review is recorded as a blocker and the build is left as
   `NIET GEÏMPLEMENTEERD`, consistent with `AGENTS.md`'s missing-review rule.
2. A CI job (`ci.yml` or `m3-step7-shadow.yml`) that runs the focused
   fixtures plus `colibri.exe` and the existing DAG/PIPE/grouped-int4 tests
   under TSAN/ASan on the supported hosts. Failed CI is not a Gate-7 PASS.

3. Full-model tensor/replica mapping, real same-controller/upstream
   contention calibration, held-out/all-resident performance evidence, and
   the remaining lifecycle/resource-failure evidence. None may be inferred
   from the tiny replay or standalone fixtures.
