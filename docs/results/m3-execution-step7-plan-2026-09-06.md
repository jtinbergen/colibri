# MiniMax-M3 Step 7 — implementation plan

> Reading order: this plan assumes the binding invariant contract at
> [`docs/results/m3-execution-step7-invariants-2026-09-06.md`](m3-execution-step7-invariants-2026-09-06.md)
> is in hand. Every section below cites the contract section it implements.

This plan was synthesised after two architect-pass attempts returned NO-GO
under the available step budget. The parent (`coder` invocation) reads it in
place of a fresh plan synthesis. Every existing-code touch carries a
`path:line` citation; every invariant carries a `INV-N` cross-ref.

## 1. Scope and contract anchor

Each invariant from the contract maps to a concrete enforcement. The contract
is preserved verbatim — no new invariants, none removed.

| Contract # | Title (one line) | Enforced by |
|---|---|---|
| INV-1 | Shadow is opt-in (`COLI_PLANNER_SHADOW=1`) | `c/m3_shadow_plan.h:25-31` env guard; planner symbol is `weak` and a stub when the macro is not defined (sketch at `c/m3_shadow_plan.c` header guard around the implementation). |
| INV-2 | Shadow on/off produces identical routing, expert selection, disk bytes, leases, reductions, outputs | The observer (`c/m3_shadow_plan.c:lookup_observer/release_observer`) calls the underlying `ops->lookup` / `ops->release` and returns its result byte-for-byte. Verified by `c/tests/test_m3_shadow_plan_observer.c` on the recorded baseline trace (`docs/results/m3-execution-step7-fixtures/baseline_replay/`). |
| INV-3 | No mutation outside the planner handle | All planner state lives in a single `m3_shadow_plan_t` allocated by the engine in `c/expert_store_registry.c:77` before backend open and freed in `c/expert_store_registry.c:backend_unregister` (or at process exit). The engine's own allocator is never called by the planner module. |
| INV-4 | Predictor purity | `predict(snapshot, request, candidate, dispatch_time)` in `c/m3_shadow_plan.c:predict()` is a pure function: all inputs are const pointers, the only output is a value struct on the caller's stack, no global state is read or written. Verified by replay-determinism fixture. |
| INV-5 | Counterfactual containments | `predict()` operates on a per-snapshot copy of the ledger. The functions `planner_copy_ledger` / `planner_restore_ledger` (in `c/m3_shadow_plan.c`) explicitly shuttle private state across hypothetical admission. The engine's real `ColiExpertStoreStats` are never inspected by the planner — the observer records its own shadow stats. |
| INV-6 | Bounded work | `c/m3_shadow_plan.h:M3_SHD_CFG_MAX_RESOURCES` / `MAX_REQUESTS_INFLIGHT` / `MAX_BYTES_INFLIGHT` constants declare the caps. Every public planner entry checks the bound before allocation and returns `M3_SHD_RESULT_UNKNOWN_BOUND` if exceeded. Tests in `c/tests/test_m3_shadow_plan.c` cover each cap. |
| INV-7 | Determinism | Identical `(snapshot hash, profile hash, planner version)` ⇒ identical decision log. Tested by `test_m3_shadow_plan_replay` (twice the same input → byte-equal log). |
| INV-8 | No global timer | `c/m3_shadow_plan.h` exposes a `m3_shd_clock_t` typedef the caller must supply (the fake monotone clock fixture provides one; the engine-supplied clock also wraps `v4_now_mono()` if `COLI_PLANNER_SHADOW_USE_MONO_CLOCK=1`). The planner reads only this clock. |
| INV-9 | Failure / cancellation are independent | `record_real_completion` / `record_real_failure` (`c/m3_shadow_plan.c`) drop the request from the simulator ledger after a real event. `OVERDUE/UNKNOWN` returns only consider real-side in-flight requests. Tested by `test_m3_shadow_plan_real_failure` and `test_m3_shadow_plan_predicted_overdue`. |
| INV-10 | Hardware is honest | Profiles carry `topology_hash`, `units_per_second_explicit`, `sample_count`, `version`. Loading a profile with `units_per_second_explicit == 0` puts every prediction into `UNKNOWN`. Missing 6/7-drive hardware is reported as `NOT_RUN` by the gate evidence; never fabricated. |

### 1.1 Predictor and scoring (formula explicit)

Carrying the formula explicitly here so a future reader (and a future
re-test) does not re-make the signed-lateness misread recorded in
[`m3-execution-step7-plan-review-2026-09-06.md`](m3-execution-step7-plan-review-2026-09-06.md)
§C original draft.

For a candidate `k` over a snapshot of admitted requests `S`:

```
rate_r     = C_r(load_class, size_class) / n_r   (per resource r in the path)
ready_i    = dispatch_time + startup_i + bytes_i / min_r(rate_r)
wait_i     = max(0, ready_i - need_i)            (clamped-positive, never negative)
score_k    = Σ wait_i over S ∪ {new request}    (the candidate-relevant union)
slack_k    = need_new - ready_new - uncertainty_margin
predicted_wait_k = max(0, -slack_k)
uncertainty_margin = pre-recorded calibration residual (no per-call tuning)
```

`need_i` is a soft deadline derived from the DAG and consumer-side
dependencies; it is **never** derived from the predictor's own I/O
completion (that would conceal the very stall the predictor measures).
Admit the candidate with the smallest `score_k`; ties break on earliest
`ready_new` then lowest stable drive-ID. Defer is bounded: when an
age-priority request has now-available capacity, the defer candidate
expires and `ready_new` wins on `score_k`.

All arithmetic uses fixed precision and time quantisation per planner
version. Fixtures in §4 must match these formulas; future changes to the
scoring rule are an invariant amendment requiring a contract update.

## 2. Files touched

### 2.1 New files

```
c/m3_shadow_plan.h                      m3-shad-pub API; opaque types; env guard
c/m3_shadow_plan.c                      observer + predictor + bounded decision
c/m3_shadow_plan_log.h                  per-decision log record format
c/m3_shadow_plan_calib.h                calibration profile representation
c/tests/test_m3_shadow_plan.c           7d fixtures (deterministic)
c/tests/test_m3_shadow_plan_observer.c  shadow on/off replay-equality test
docs/results/m3-execution-step7-fixtures/<name>/{inputs,expected,actual}
                                        fixture captures (per the contract)
docs/results/m3-execution-step7-gate-2026-09-06.md
                                        gate-evidence report (final closure)
```

### 2.2 Build wiring (Makefile changes, `path:line`)

- `c/Makefile:1404` — add `tests/test_m3_shadow_plan$(EXE)` rule following the
  shape of `tests/test_expert_store_ops$(EXE)` (one-file rule linking only
  the planner + fixture sources). Same for
  `tests/test_m3_shadow_plan_observer$(EXE)`.
- `c/Makefile:1423-1491` (segment block) — add `m3_shadow_plan.o` to the
  segment CPU object list and `(SEGMENT_BUILD_DIR)/m3_shadow_plan.o` build
  rule, mirroring `expert_store_registry.o`.
- `c/Makefile:1620-1660` (`V4_OWN_DIR`/`expert_store_registry.o` block) —
  add the analogous v4-own rule for `m3_shadow_plan.o`. Pattern:
  ```
  $(V4_OWN_DIR)/m3_shadow_plan.o: m3_shadow_plan.c m3_shadow_plan.h \
          m3_shadow_plan_log.h m3_shadow_plan_calib.h tensor.h | $(V4_OWN_DIR)
          $(CC) $(V4_OWN_CFLAGS) -c m3_shadow_plan.c -o $@
  ```
- `c/Makefile.deepseek-v4:204` — extend `V4_OBJS` to include the
  v4-built `m3_shadow_plan.o` (already produced by the segment rule when
  `COLI_V4_SUPPORTED=1`; `V4_OBJS` only appends it unconditionally in the
  supported build).

### 2.3 CI additions

- New file `.github/workflows/m3-step7-shadow.yml`. Pattern copied from
  `.github/workflows/m3-step6-tsan.yml:1-24` (one workflow per step is
  precedent). Triggers on `paths: ['c/m3_shadow_plan*', 'c/Makefile*',
  '.github/workflows/m3-step7-shadow.yml', 'docs/results/m3-execution-step7*']`.
  Job: build the four focused executables
  (`test_m3_dag.exe`, `test_i4_grouped.exe`, `test_pipe_block.exe`,
  `test_m3_shadow_plan.exe`, `test_m3_shadow_plan_observer.exe`) and run
  them. Failure is a hard fail.

### 2.4 Existing source — what is NOT edited

The plan does not edit `c/deepseek_v4.c`, `c/colibri.c`, `c/expert_store.h`,
`c/expert_store_registry.h`, or any consumer of `ColiExpertStoreOps`. The
observer is plugged at the `backend_open` site (`c/expert_store_registry.c:77`),
which synthesises a new `ColiExpertStore` whose vtable points to
`shadow_{lookup,release,prefetch,stats,destroy}` wrappers. The wrappers call
the underlying backend first, then call the observer's record hooks. Engine
code never sees the difference.

## 3. Module boundaries (header / opaque types / inline rules)

Modelled on `c/expert_store.h`:

```c
#ifndef COLIBRI_M3_SHADOW_PLAN_H
#define COLIBRI_M3_SHADOW_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include "expert_store.h"
#include "m3_shadow_plan_calib.h"

#define M3_SHD_MAGIC 0x53484450u     /* "SHDP" */
#define M3_SHD_VERSION 1

#ifndef COLI_PLANNER_SHADOW
/* Symbol-stub form. All APIs return UNKNOWN; nothing else happens. */
#endif

typedef struct m3_shadow_plan m3_shadow_plan_t;

typedef enum { M3_SHD_OK = 0,
               M3_SHD_UNKNOWN = 1,
               M3_SHD_OVERDUE_UNKNOWN = 2,
               M3_SHD_DEFER = 3,
               M3_SHD_RESULT_UNKNOWN_BOUND = 4 }
m3_shd_result_t;

m3_shadow_plan_t *m3_shd_open(const void *topology, size_t topology_bytes,
                             const void *profiles, size_t profiles_bytes,
                             char *err, size_t errsz);
void m3_shd_close(m3_shadow_plan_t *p);

void m3_shd_observe_lookup_begin(m3_shadow_plan_t *p,
                                 const ColiExpertKey *key,
                                 uint64_t consumer_need_ns);
void m3_shd_observe_lookup_end(m3_shadow_plan_t *p,
                               const ColiExpertKey *key,
                               int result);
void m3_shd_observe_release(m3_shadow_plan_t *p,
                            const ColiExpertKey *key);

m3_shd_result_t m3_shd_predict(const m3_shadow_plan_t *p,
                               const ColiExpertKey *key,
                               const void *candidate_paths,
                               size_t n_candidates,
                               m3_shd_decision_record_t *out);

const char *m3_shd_decision_log_path(const m3_shadow_plan_t *p);
#endif
```

Style rules:
- Header guards `COLIBRI_*` (matches `c/expert_store.h:1`).
- Opaque types hide fields (matches `ColiExpertStore` in `expert_store.h:14-78`).
- Static inlines live in headers (matches `c/expert_store.h:80-97`).
- No global state beyond the `m3_shadow_plan_t` handle and `m3_shd_clock_t`
  the caller supplies.

## 4. Test surface

### 4.1 Fixtures (7d hand-calculable, listed in `m3-execution-plan:772-778`)

For each row below, the fixture file holds the topology, profile, request
list and the *expected* decision record (one line per decision). Tolerated
on shared time arithmetic: ≤ 1 ns.

| Fixture name (file: test name) | Inputs | Expected outcome |
|---|---|---|
| `test_m3_shadow_plan_shared_bandwidth` | 4 drives × 500 MiB/s + 1 controller × 500 MiB/s; 4×100 MiB starts at `t=0` | Each flow gets 125 MiB/s; all four complete at `t=0.8 s`; decision log entries identical across all four mirrors of the same candidate. |
| `test_m3_shadow_plan_critical_existing_read` | A on busy controller 500 MiB/s with 100 MiB @ `need=0.2 s`; new B 100 MiB on a second drive of the same controller, `need=1 s` | B now: A/B ready `t=0.4 s`, score `0.2 s`. B deferred: A ready `0.2 s`, B `0.4 s`, score 0 → `DEFER` chosen. |
| `test_m3_shadow_plan_independent_replica` | Above plus B-replica on a free controller 500 MiB/s | A and B ready `0.2 s`, score 0 → second controller wins on earliest ready. |
| `test_m3_shadow_plan_in_group_latency` | One 100 MiB-request; replicas X (startup 0) and Y (startup 0.05 s); both 500 MiB/s; `need=0.3 s` | X ready `0.2 s`, Y `0.25 s`; X wins. |

### 4.2 Named-but-uncalculated 7d cases (per the plan §7d bullet list)

Each becomes its own fixture:

| Fixture | Verdict asserted |
|---|---|
| `test_m3_shadow_plan_six_seven_drives_three_controllers` | A controller-0 saturating request leaves groups on controllers 1 and 2 planable. Adding a shared upstream resource couples them. |
| `test_m3_shadow_plan_two_drives_latency_asym` | A holds its latency at 4 reads, B spikes after 1; A admits 4, B never backfills proportionally. With A busy, B can still be the best candidate. |
| `test_m3_shadow_plan_low_latency_high_contention` | Lower-latency source with high contention loses to a slower free source. |
| `test_m3_shadow_plan_unknown_profile` | Missing profile for candidate ⇒ `UNKNOWN`, never a default-capacity fill. |
| `test_m3_shadow_plan_missing_replica` | Replica unavailable ⇒ that candidate is excluded; no implicit fallback. |
| `test_m3_shadow_plan_no_deadline_hit` | No candidate hits; best feasible is chosen; miss is recorded. |
| `test_m3_shadow_plan_tie_break` | Stable tie-break on score; on equal score, earliest weight-ready; on equal weight-ready, lowest drive-ID. |
| `test_m3_shadow_plan_age_priority` | Sustained demand with bounded queue; age-priority request gets the next empty slot ahead of fresher ones at equal need. |
| `test_m3_shadow_plan_predicted_completion_does_not_mark_real_readiness` | Real occupancy counter is unchanged across a hypothetical completion in the predictor; only the shadow copy moves. |

### 4.3 Observer equality test (`test_m3_shadow_plan_observer.c`)

Uses the existing `c/colibri.exe` baseline capture at
`docs/results/m3-execution-step6a-gate-2026-09-06.md` (57-digest FNV chain).
Runs `PROMPT=hi`, `NGEN=1`, PIPE/blocking, grouped-int4 with
`COLI_PLANNER_SHADOW=0` and `=1`. Asserts byte-equal:
- FNV digest list (57 layers);
- `ColiExpertStoreStats` snapshots (requests, misses, prefetched, bytes_read);
- per-step `redaction_done` ordering.

### 4.4 Determinism test

`test_m3_shadow_plan_replay_determinism` runs the entire 7d fixture suite
twice with seeded randomness neutralised (zero-length `m3_shd_random_seed`)
and asserts the decision log is byte-equal.

## 5. Insertion point — citations only

The observer does not touch any of these call sites. They are listed to
confirm the design hooks at the registry instead of the engine.

```
c/deepseek_v4.c:4349   /* shared_expert_path forward */
c/deepseek_v4.c:4361   /* same block, release */
c/deepseek_v4.c:4471   /* expert_load_worker body */
c/deepseek_v4.c:4562   /* dual_expert_loader_worker body */
c/deepseek_v4.c:4681   /* persistent_expert_loader_worker body */
c/deepseek_v4.c:4718   /* expert_load_start, sync variant */
c/deepseek_v4.c:5026   /* batched lookup, release */
c/deepseek_v4.c:5257   /* batched lookup, release (sequential loop) */
c/deepseek_v4.c:5287-5296 /* sequential add_matmul + release */
c/deepseek_v4.c:5680   /* multi-expert grouped release */
c/deepseek_v4.c:5690-5699 /* multi-expert grouped lookup + release */
c/deepseek_v4.c:10540 / c/deepseek_v4.c:10865 / 10914 / 10934
                       /* additional expert_pool-style call sites */
```

The single insertion point is `c/expert_store_registry.c:77`
(`coli_expert_store_backend_open_selected`). The wrapping
`ColiExpertStore` whose `ops` member points to
`m3_shd_{lookup,release,prefetch,stats,destroy}` is built only when the
backend name begins with `shadow-` (e.g. `COLI_EXPERT_STORE=shadow-auto`).
A new entry on `COLI_EXPERT_STORE=auto` is NOT added — the parent's existing
behaviour is preserved bit-for-bit when `COLI_PLANNER_SHADOW` is unset, per
INV-1.

## 6. CI integration

```
.github/workflows/m3-step7-shadow.yml     (NEW)
  on:
    workflow_dispatch:
    pull_request:
      branches: [dev, main]
      paths:
        - 'c/m3_shadow_plan*'
        - 'c/Makefile*'
        - 'docs/results/m3-execution-step7*'
        - '.github/workflows/m3-step7-shadow.yml'
  permissions:
    contents: read
  jobs:
    shadow:
      name: MiniMax-M3 Step 7 focused gate
      runs-on: ubuntu-latest
      timeout-minutes: 20
      steps:
        - uses: actions/checkout@v4
        - name: Install build deps
          run: sudo apt-get install -y --no-install-recommends make gcc
        - name: Build the planner test executables
          run: make -C c tests/test_m3_shadow_plan.exe tests/test_m3_shadow_plan_observer.exe
        - name: Run the 7d fixtures + replay determinism
          run: make -C c test-shadow-m3
        - name: Run the shadow on/off observer equality test
          run: make -C c test-shadow-observer-m3
        - name: Run the focused Step-6 + step-7 suite under TSAN
          run: make -C c test-tsan-m3 CC=clang TSAN_TARGETS=shadow
```

The Makefile gains `test-shadow-m3`, `test-shadow-observer-m3`, and a
`TSAN_TARGETS=shadow` overlay on `test-tsan-m3`.

## 7. Missing stronger-model review (BLOCKER)

`AGENTS.md` requires a high-risk-model diff review before merging a
scheduler-touching change. The available escalation path in this repo is
`coder → coder.ESCALATE → pro-coder` (per the agent's tool tier roster).
`pro-coder` is reached only via coder's internal escalation hook and cannot
be invoked directly by the parent.

If `pro-coder` returns NO-ESCALATE, NO-REVIEW, or otherwise refuses to ratify
the final diff, the merge MUST be blocked. The gate report records the
blocker in a dedicated "Stronger-model review" section. The build remains
`NIET GEÏMPLEMENTEERD / NIET GEGATED` per the existing §7 status text.

Specifically, this blocker is not a paperwork item — it is a literal hard
stop. The repo's gate text already says "Step 7 mag nu beginnen; Step 7 zelf
blijft `NIET GEÏMPLEMENTEERD / NIET GEGATED`"; it must not become
`GEÏMPLEMENTEERD / GEGATED` until that review ratifies.

## 8. Blocking ambiguities

No contract gap forces the coder to invent scheduler policy. The plan covers
every clause the contract enumerates. Two operational caveats are recorded
here so the human owner is not surprised:

1. **Hardware-clause P-claims remain `NOT_RUN`.** The plan does not provide
   a 6/7-drive machine. Gate 7 §P must stay `NOT_RUN` for cross-controller
   contention and six- or seven-drive topology claims. The plan enforces
   this by the profile-loader check on `units_per_second_explicit == 0 ⇒
   UNKNOWN`, never a fallback to a default capacity.

2. **Tie-break on equal `weight_ready_time`.** The plan specifies
   "lowest drive-ID" as the secondary tie-break. The contract does not state
   this verbatim, but it is consistent with INV-7 (determinism). If the human
   owner prefers another ordering (e.g. lowest resource-ID, lowest
   controller-ID), it is a one-line change in
   `c/m3_shadow_plan.c:m3_shd_decide()`. No invented policy.

## 9. Coder's first commit, in file order

The coder is asked to land the following files in this exact order so a
bisect lands on a green build at every step:

1. `c/m3_shadow_plan_calib.h` — profile struct, units/s mandate,
   `M3_SHD_VERSION`, sample-count typing.
2. `c/m3_shadow_plan_log.h` — decision record format.
3. `c/m3_shadow_plan.h` — public API.
4. `c/m3_shadow_plan.c` — observer, predictor, five-step decision.
5. `c/tests/test_m3_shadow_plan.c` — fixtures.
6. `c/tests/test_m3_shadow_plan_observer.c` — replay equality.
7. `c/Makefile` — three new build rules + the three new phony test targets.
8. `c/Makefile.deepseek-v4` — minimal: include the segment rule's output
   into `V4_OBJS` only when `COLI_V4_SUPPORTED`.
9. `.github/workflows/m3-step7-shadow.yml` — workflow file.
10. `docs/results/m3-execution-step7-fixtures/<name>/{inputs,expected,actual}`
    — captures from running each fixture before merge.
11. `docs/results/m3-execution-step7-gate-2026-09-06.md` — gate-evidence
    report (final closure, includes the missing-review section even if it
    is just the explicit "BLOCKED pending stronger-model review").

## 10. Self-checks before merge

- `make -C c tests/test_m3_shadow_plan.exe tests/test_m3_shadow_plan_observer.exe`
  (Linux/Windows/macOS as supported) — must build cleanly.
- The four focused-step fixtures plus the existing
  `tests/test_m3_dag.exe`, `tests/test_i4_grouped.exe`,
  `tests/test_pipe_block.exe` — all PASS.
- Shadow on/off equality test on the recorded baseline — all 57 FNV
  digests match, byte-equal stats.
- Pre-existing CI suite (`ci.yml` Step-6 + engine tests) — green.

## 11. Hand-off

This plan is the parent of the `coder` invocation. If `coder` exits with
ESCALATE, the next reviewer is `pro-coder`. If `pro-coder` exits with
NO-GO, return NO-GO to the parent immediately with the reviewer's reason
attached; do not silently re-route. Do not invent scheduler policy to
satisfy a pro-coder review.

If this plan is rejected by `plan-reviewer`, refine once and re-submit. If
it is rejected twice, return NO-GO to the parent with both rounds' blocker
list.
