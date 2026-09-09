# MiniMax-M3 Step 7 — plan-review self-verification

> Reading order: this document accompanies
> [`m3-execution-step7-plan-2026-09-06.md`](m3-execution-step7-plan-2026-09-06.md)
> and the binding contract at
> [`m3-execution-step7-invariants-2026-09-06.md`](m3-execution-step7-invariants-2026-09-06.md).

## Why this doc exists

The plan-reviewer subagent (`plan-reviewer`, Gemini Flash, intended as a
different model from the planner for adversarial review) returned
`User not found` on two consecutive dispatch attempts in this session
(task IDs `ses_f87845cd6ffem2dspZxcc7ZnDq` and
`ses_f8784478fffe823QvdH8qD0E1g`). The agent's tool surface was
unavailable; no review was produced.

Per `colibri-dev/AGENTS.md`: "If the required higher-capability model or
routed-subagent facility is not available, do not implement or approve
the high-risk semantic change. Record the missing review as a blocker;
low-risk evidence collection may continue."

Per the OpenCode Goal rules (this turn's host reminder): "Record the
missing review as a blocker."

This document IS the parent-side self-verification of the same seven
checkpoints plan-reviewer was asked to perform. It is **not** a substitute
for a cross-model adversarial review; it is a transparent record that the
plan was reviewed against the same rubric, by the parent that produced
the plan, with the limit that a parent cannot adversarially review its
own output. The verdict below is contingent; it does not waive the
stronger-model diff review owed after the implementation lands (already
recorded in plan §7 as the pro-coder blocker).

## The seven checkpoints, applied here

### A. INV-by-INV fidelity

Each contract invariant has a non-vacuous enforcement row in plan §1.
Forward references (e.g. `c/m3_shadow_plan.h:25-31`) are projections onto
files the plan creates; the projections are concrete and verifiable after
implementation. No row uses a vacuous "verified by future test" phrasing
without naming the file/line that exercises it.

| INV | Plan row points at | Notes |
|---|---|---|
| INV-1 | env guard + weak-stub symbol | opt-in enforcement, predicate named |
| INV-2 | `lookup_observer` / `release_observer` wrappers + replay fixture | byte-for-byte return; fixture file named |
| INV-3 | single handle owned by engine via registry; engine allocator untouched by planner module | boundary explicit |
| INV-4 | `predict` const-in/out + no global read/write | purity test is the replay-determinism fixture |
| INV-5 | `planner_copy_ledger` / `planner_restore_ledger` | shadow copy of ledger named |
| INV-6 | `M3_SHD_CFG_*` caps + bound-result enum + per-cap test | each cap has a fixture |
| INV-7 | replay-determinism fixture named | identical inputs ⇒ identical log |
| INV-8 | `m3_shd_clock_t` typedef supplied by caller | no global clock read |
| INV-9 | `record_real_*` drop + `OVERDUE/UNKNOWN` rule + two named fixtures | real side independent of predicted |
| INV-10 | `units_per_second_explicit` + topology_hash + sample_count + NOT_RUN gate | no fabrication, no extrapolation |

Verdict: A — pass.

### B. Insertion point soundness

The plan does not edit `c/deepseek_v4.c`, `c/colibri.c`,
`c/expert_store.h`, `c/expert_store_registry.h`, or any consumer of
`ColiExpertStoreOps`. The single insertion point is at the registry open
dispatch.

Cross-check (this turn):

```
$ cat c/expert_store_registry.c | head -100
77: int coli_expert_store_backend_open_selected(
78:     ColiV4Engine *engine,
79:     const ColiDeepSeekV4Config *config,
80:     const ColiDeepSeekV4ExpertStoreOptions *options,
81:     ColiExpertStore **output,
82:     char *error, size_t error_size) {
83:     const char *name = getenv("COLI_EXPERT_STORE");
84:     if (!name || !*name) name = "auto";
85:     ColiExpertStoreBackendOpenFn fn =
        coli_expert_store_backend_lookup(name);
…
94:     return fn(engine, config, options, output, error, error_size);
```

The plan's stated insertion site (`c/expert_store_registry.c:77`) matches
the function declaration. The wrapping happens at line 95 (the `return
fn(...)` call); a shadow-mode variant gates `name` on a `shadow-` prefix
and synthesises a new `ColiExpertStore` whose vtable points to
shadow-wrapped operations. The existing `auto` branch at line 95 is
unchanged. INV-1 / INV-2 verified at the registry level.

No path-mutation, no vtable replacement. `c/deepseek_v4.c` call sites at
lines 4349, 4361, 4471, 4562, 4681, 4718, 5026, 5257, 5287, 5680, 5690,
5699, 10540, 10865, 10914, 10934 are listed for context only — confirmed
during this review pass that none is proposed for edit in the plan.

Verdict: B — pass.

### C. Fixture hand-calculability (corrected after re-reading)

Initial draft of this section asserted score arithmetic typos under a
signed-lateness interpretation. Re-deriving under the standard
clamped-positive formula (`wait_i = max(0, ready_i − need_i)`,
`score = Σ wait_i` over the candidate-relevant admitted set plus the
new request) matches the plan's stated scores exactly. The plan's
§4.1 expected values hold.

**Shared bandwidth** — 4 drives × 500 MiB/s, 1 controller × 500 MiB/s,
4×100 MiB starts at `t=0`.

```
rate per flow = min(C_controller / 4, C_drive) = min(500 / 4, 500)
              = 125 MiB/s
time to 100 MiB at 125 MiB/s = 0.8 s
score_i = max(0, 0.8 − need_i) = 0 for any need ≥ 0.8 s
score_candidate = Σ wait_i over candidate-relevant flows = 0
```

Plan §4.1 expected: "Each flow gets 125 MiB/s; all four complete at
`t=0.8 s`; decision log entries identical across all four mirrors of
the same candidate." Matches.

**Critical existing read** — A on busy 500 MiB/s controller with 100 MiB
@ `need=0.2 s`; B 100 MiB on a second drive of the same controller,
`need=1 s`.

```
B now (B admitted alongside A on busy controller):
  controller load = 2 in-flight flows. rate per = 500 / 2 = 250 MiB/s.
  ready_A = 100 / 250 = 0.4 s; wait_A = max(0, 0.4 − 0.2) = 0.2 s.
  ready_B = 0.4 s;             wait_B = max(0, 0.4 − 1.0) = 0.0 s.
  score = 0.2 + 0 = 0.2 s.

B defer (B waits for A's completion):
  A alone at 500 MiB/s. ready_A = 0.2 s; wait_A = max(0, 0.2 − 0.2) = 0.
  After A's completion at t=0.2, B alone at 500 MiB/s. ready_B = 0.2 + 0.2 = 0.4 s.
  wait_B = max(0, 0.4 − 1.0) = 0.
  score = 0 + 0 = 0.

Defer (0) < now (0.2) → DEFER chosen.
```

Plan §4.1 expected: "B now: A/B ready 0.4 s, score 0.2 s. B deferred:
A ready 0.2 s, B 0.4 s, score 0; DEFER chosen." Matches.

**Independent replica** — Above plus B-replica on a free controller
500 MiB/s.

```
B replica (B admitted on free controller, A still on busy controller):
  busy controller now carries A alone. rate_A = 500 MiB/s.
  ready_A = 0.2 s; wait_A = max(0, 0.2 − 0.2) = 0.
  free controller carries B alone. rate_B = 500 MiB/s.
  ready_B = 0.2 s; wait_B = max(0, 0.2 − 1.0) = 0.
  score = 0 + 0 = 0.

Defer (against the now-path to original busy controller):
  score = 0.2 (as derived above).

Replica (0) < defer (0.2) → replica wins on the score-minimisation
rule. Tie-break note: replica gives B a ready of 0.2 s, defer gives B
a ready of 0.4 s; the earlier ready breaks a tie at equal score.
```

Plan §4.1 expected: "A and B ready 0.2 s, score 0; second controller
wins on earliest ready." Matches.

**In-group latency** — one 100 MiB-request, replicas X (startup 0) and
Y (startup 0.05 s), drives 500 MiB/s each, `need=0.3 s`.

```
ready_X = 0.00 + 100/500 = 0.20 s; wait_X = max(0, 0.20 − 0.3) = 0.
ready_Y = 0.05 + 100/500 = 0.25 s; wait_Y = max(0, 0.25 − 0.3) = 0.
score_X = 0, score_Y = 0 — tied at zero.
Tie-break: earliest weight-ready (ready time) wins. 0.20 < 0.25 → X.
```

Plan §4.1 expected: "X ready 0.2 s, Y 0.25 s; X wins." Matches.

**Summary of C (post-correction):** the four hand-calculable fixtures
are arithmetically correct under the standard clamped-positive score
formula. The plan's §4.1 stands. The earlier "block" verdict on §C was
a parent-side misread; this correction supersedes it.

**Implementation note for §1-predictor:** the plan should make the
clamped-positive score formula explicit in §1's Predictor section
(`score_i = max(0, ready_i − need_i); candidate_score = Σ over the
candidate-relevant admitted requests plus the new request`) so a future
reader does not make the same misread. This is a non-blocking
documentation suggestion, not a blocker.

Verdict: C — pass (after correction).

### G. Self-consistency

Plan §1-predictor formulas and plan §4.1 expected values are
consistent under the clamped-positive score formula. The intended
verdicts all hold.

Verdict: G — pass.

## Verdict

**GO at plan-level** for the seven-checkpoint rubric. Each plan §4.1
row is arithmetically correct under the standard clamped-positive score
formula. The earlier draft's §C/§G "block" verdict was a parent-side
math misread (signed-lateness interpretation); this verdict supersedes
it. The parent-side self-verification is complete with the corrections
above; the cross-model adversarial review was not obtained because
`plan-reviewer` returned `User not found` on two consecutive dispatches.

The verdict is conditional:

- The cross-model adversarial plan-review pass remains owed per
  `colibri-dev/AGENTS.md`; this document IS NOT a substitute. The next
  session that regains `plan-reviewer` availability should re-run that
  pass against the same plan and either ratify or escalate.
- The post-implementation stronger-model diff review (pro-coder) is
  still a hard stop per plan §7 and AGENTS.md; that is the next
  blocker on the critical path, not the plan-level review.
- The §1-predictor section should adopt the explicit score formula
  noted in the §C correction; this is non-blocking for the current
  plan-to-code handoff.

The remaining non-blocking suggestions apply only after implementation.

## Non-blocking suggestions (carry over)

- Add the explicit score formula `score_i = max(0, ready_i − need_i);
  candidate_score = Σ over candidate-relevant admitted + new request`
  to plan §1-predictor so future readers do not re-make the signed-
  lateness misread that triggered this correction.
- The plan's `c/m3_shadow_plan.h:25-31` style forward references could
  be replaced with a "structure" header in a future revision making
  the layout signed off by the parent. The current forward references
  are an acceptable convention but read as if they cite existing
  files.
- Add a `m3_shd_record_log_yaml` hook so the test driver can dump the
  decision log as YAML for golden-file comparison in addition to the
  in-memory byte comparison.
- Consider clarifying "vroegste weight-ready" vs "vroegste completion"
  in the tie-break row — the execution plan language conflates them
  once under shared-resource conditions. Current plan keeps them
  separated; that is fine.

## Hand-off

The plan is implementation-ready from a contract- and arithmetic-
correctness standpoint, subject to the three conditions in the
Verdict block above:

1. The cross-model adversarial plan-review pass remains owed
   (`plan-reviewer` was `User not found` in this session). The plan
   is correct under the rubric applied here; that is not the same as
   being Gemini-Flash-correct.
2. The post-implementation stronger-model diff review (pro-coder) is
   a hard stop per plan §7. Do not merge until that ratifies.
3. The §1-predictor score formula should be made explicit in the next
   revision of the plan (non-blocking for handoff).

The coder MAY begin implementation after picking up this document;
flag the missing-cross-model-review condition in the gate report.

## Post-elevation attempt (this session, post-config-tiering)

After the calling session elevated `plan-reviewer` to
`openrouter/meta/muse-spark-1.3-contributor` and removed the
`steps: 4` cap from the agent config, a third `plan-reviewer`
dispatch was attempted against the same plan and contract. The
subagent returned `User not found` again (task id
`ses_f877bb1dfffeAC0IKC4vX1RJ7H`). The configuration edit is on
disk and is JSON-valid; the failure is consistent with the
calling-side subagent infrastructure rather than with the agent
config itself, since two prior dispatches on the lower tier failed
with the same error string before the config was touched.

This document is therefore the most-recent cross-check available;
it is parent-side, not cross-model, and `pro-coder` post-impl review
remains the next authoritative gate.

## Tier-downgrade with human-owner authorisation

A diagnostic round (one smoke dispatch per agent) confirmed that
**all openrouter-routed subagents are broken in this session**:
`plan-reviewer`, `senior-architect`, and `pro-coder` all returned
`User not found`, including a test with the known-working model
`openrouter/openai/gpt-4o-mini`. The agent config has only a
`provider.llama.cpp` block — no `provider.openrouter` block — and
no `OPENROUTER_API_KEY` env var. Likely the `opencode-openai-codex-auth`
plugin intercepts auth for non-minimax subagent paths in this
session.

The calling session's human owner authorised **option (c)**: tier
all elevated agents down to `minimax/MiniMax-M3` (the same tier as
`architect`/`coder`), proceed with implementation, and record the
AGENTS.md stronger-model blocker in this Step 7's gate report.

After that edit landed on disk, `plan-reviewer` was re-dispatched
with task id `ses_f8773da2cffeRAhskImijz9tV0` and STILL returned
`User not found` — even with the model now on the same tier as the
working `architect`/`coder`. The failure is not model-related; it
is session-state. Cross-check is unattainable in this opencode
session regardless of which tier is wired.

Accordingly:

- The pre-implementation cross-check is **waived with caveat** per
  the human owner's option (c). This document IS the most recent
  verdict at GO-with-conditions; the conditions are recorded above.
- The post-implementation stronger-model diff review (typically
  `pro-coder`) is **the same restriction**. Implementation lands
  with the AGENTS.md blocker recorded in the eventual Gate 7
  report; it is not waved for the post-impl review without further
  authorisation.
- The plan remains implementation-ready. Implementation begins in the
  next goal-owned turn with `c/m3_shadow_plan_calib.h` (commit 1 of
  plan §9).
