# MiniMax-M3 Step 7 — compact stronger-review packet

This packet is intentionally self-contained and narrow. It is for an
independent Sol/Astra review of the current uncommitted Step-7 diff; it is not
an approval by the implementing agent.

## Requested decision

Report blocking findings only for the bounded shadow planner and transparent
store boundary. Do not infer Gate-7 promotion from the local fixture passes.
Classify each finding as verified, suspected, or unproven.

## Scope

- `c/m3_shadow_plan.{c,h}`: bounded event-driven predictor, queue, active
  ledger, deterministic decision log, atomic observer guard, and candidate
  parent-chain expansion.
- `c/m3_shadow_plan_calib.h`: explicit drive/controller/upstream topology and
  measured-profile representation. Rates, startup, and residual values are
  `uint64_t`; missing units or samples are unsound.
- `c/m3_shadow_store.{c,h}`: transparent forwarding wrapper. Planner state is
  shadow-only; real leases, views, return values, store occupancy, and
  destruction remain owned by the underlying store.
- `c/expert_store_registry.c`: opt-in `shadow-` backend wrapper. The default
  `auto` path is unchanged.
- `c/colibri.c`: authoritative MiniMax-M3 path. It uses `expert_load()` and
  `ESlot` directly; the opt-in `COLI_M3_SHADOW_RUNTIME` build now instruments
  this path without changing route, lease, or reduction behavior.
- `c/m3_shadow_m3_bridge.{c,h}`: pure translation seam for M3 load facts; no
  global state or I/O.
- `c/m3_shadow_m3_runtime.{c,h}`: process-lifetime bounded runtime adapter,
  copied PIPE/OpenMP job context, explicit load observation token, and
  opt-in decision-log output. It refuses missing deadlines and incomplete or
  striped physical mappings. Its live decision and actual-start registration
  share one planner guard, preventing concurrent observer calls from deciding
  against the same stale occupancy snapshot.
- `c/tests/test_m3_shadow_*.c`: deterministic planner, observer, store, and
  registry-boundary fixtures.
- 7b-ext calibration artifacts: `c/tools/m3_shadow_calibrate.py`, the
  topology-kind fixture, and the planner-text emit/parse gate for CPU/GPU,
  memory-domain, and link rows. These are data-model evidence only; they do
  not promote unavailable transfer or contention hardware. The emitter now
  accepts only unambiguous singleton-drive groups, never infers
  controller/upstream capacity, and leaves incomplete paths `UNKNOWN`.
- 7b-0 machine-profile bridge: `c/machine_profile.py`, `coli profile`, and the
  read-only `coli plan` attachment. It records inventory and raw-capture
  metadata only; it does not alter placement or promote calibration evidence.

## Invariants to audit

1. `m3_shd_predict()` is pure: no planner log, real ledger, queue, or
   observer-counter mutation.
2. `m3_shd_decide()` may append only a bounded deterministic record; it must
   not reserve real store resources.
3. Every admitted drive-only candidate is expanded through its declared
   controller/upstream ancestors exactly once. Missing, cyclic, duplicate, or
   overlong chains must become unavailable/unknown, never independent
   capacity.
4. All planner mutable state is protected by its bounded atomic guard; no
   ordinary field is concurrently read while an observer or decision path can
   write it. Live decision plus actual-start registration is one guarded
   operation, and its returned actual-path occupancy is one guarded snapshot.
5. Store wrapper destruction releases the wrapper-owned planner exactly once
   and destroys the inner store exactly once, after normal lease ownership has
   been preserved.
6. A provider failure, invalid candidate count, or unknown profile cannot
   alter the underlying lookup result or view.
7. Fixed-width arithmetic remains conservative on overflow and on missing
   calibration; no default bandwidth is invented.
8. The immutable profile hash covers kind-specific topology payloads, not only
   resource IDs and generic storage bounds.

## Verified local evidence

```text
make -C c test-shadow-m3 test-shadow-observer-m3 \
  test-shadow-store-m3 test-shadow-registry-m3 test-shadow-m3-bridge \
  test-shadow-m3-runtime EXTRA_CFLAGS=-Werror
  planner fixtures: PASS
  observer boundary: PASS
  store forwarding: PASS
  registry boundary: PASS
  M3 bridge: PASS
  M3 runtime seam: PASS

make -C c PYTHON=/c/Python313/python.exe test-shadow-m3-runtime-replay
  shadow-off/on semantic stdout, exit status, normalized COLI_TRACE: PASS
  complete six-component rows: PASS
  counterfactual replica 102 under occupied replica 101: PASS
  four controlled completion-order permutations: PASS
  clock-callback re-entry, automatic-ID, and invalid-bound regressions: PASS
  concurrent combined decide/start callers with consistent occupancy: PASS
  decision-log/runtime-trace request-ID join for the live replay: PASS
  replay uses a pinned bounded-parallel contract to make overlap non-vacuous: PASS
  invalid weight-copy resource and unavailable-copy fallback regressions: PASS
  optional serial conversion-stage ordering and bounded admission: PASS
  configured queue-age priority under a fake monotone clock: PASS

make -C c -B test-shadow-m3-topology test-shadow-m3-calibration-emit \
  EXTRA_CFLAGS=-Werror
  7a-ext topology fixtures: PASS
  7b-ext calibration planner-text emit/parse: PASS
  sparse profile ownership, aggregate-order, and UNKNOWN-path regressions: PASS
  Python calibration/profile/resource-plan tests: PASS (71 tests)
  inflight 8/16 capture and p99/2x sensitivity review: PASS (NOT_PROMOTED)
  7b-0 machine profile/unit tests and CLI smoke: PASS (NOT_GATED)
  full M3 copy-table scale (15,360 rows, 3,000 lookups): PASS (NOT_PROMOTED)
  full-model runtime shadow-off/on route/read smoke: PASS for 444 observed
  identities (synthetic profiles; C-evidence only, NOT_PROMOTED)
  The trace joined consumer need, predicted ready, actual ready, and per-path
  load for 453 completed rows; synthetic-profile diagnostic only. The
  machine-readable drive/controller aggregation is in
  `m3-execution-step7-full-model-runtime-report-2026-09-07.json`.

make -C c -f Makefile.deepseek-v4 deepseek-v4
  Windows engine link: PASS

git diff --check
  PASS
```

The store fixture explicitly checks `{drive}` → `{drive, controller}`
The bounded calibration-semantics patch received a final scoped Sol review:
approved. It checked singleton ownership, aggregate exclusion, non-inferred
controller/upstream capacity, sparse-map integrity, and `UNKNOWN_BOUND`
propagation. This is not the still-required full Step-7 diff review or Gate-7
promotion.

The model-aware full-model config generator received a follow-up scoped Sol
review after its initial blockers were corrected: no blocking findings. The
review confirmed exact config-derived expert inventory, deep safetensors
container validation, topology/resource-ID consistency, and copy-cap bounds.
This remains a route/read integration artefact with synthetic profiles, not a
hardware-calibration or performance approval.

The expansion confirms shadow decisions do not leave active or queued real
planner entries.
The M3 bridge fixture checks snapshot-field preservation, rejects incomplete
load facts (including the model identity required to correlate a request with
the M3 forward), and passes a translated drive candidate through declared
controller/upstream expansion into the pure planner.
The production tiny replay configures two equivalent copies for both observed
experts on independent drive/controller/upstream paths. The actual engine
continues to use replica 101; the second shadow decision selects replica 102
after the first path is registered. The replay therefore proves candidate
enumeration and stale-snapshot prevention at the live seam, not an active
engine route change.
The Windows capture maps `C:` and `E:` to distinct NVMe PCI parent devices;
the 16-sample p99 matrix and its 2x sensitivity review are recorded, but no
same-controller or shared-upstream contention is claimed.

An Astra review was requested against this packet, but the stronger-model
review facility returned a usage-limit error before producing findings. This
is recorded as missing review evidence, not as an approval or a clean result.

A later compact Astra design review covered the live seam. Its actionable
points were applied locally: load observations use stack-owned tokens, job
context is copied before PIPE publication and installed per worker job,
counterfactual decisions are separated from actual-load occupancy, missing
deadlines/mappings return unknown, and io_uring/direct-striped coverage is not
claimed. A further narrow Astra review of the calibration-tail patch found no
blocker: non-zero residuals now accept only p95/p99 labels, and unsupported
percentiles are rejected by tests. This remains a scoped review, not full
Gate-7 approval.

A subsequent narrow Astra review of the live consumer-need seam also found no
blocker after the scope was narrowed to the serial M3 DAG path. The observed
`need=now` timestamp is captured once per block only where current readiness is
established; PIPE/parallel paths remain UNKNOWN until their post-shared-
compute readiness can be supplied explicitly.

The latest narrow Astra follow-up reviewed only the two newly added evidence
items: concurrent combined decide/start contention and request-ID joining of
the live decision log to the runtime trace. It found no blocker. This closes
those scoped review questions, but does not approve Gate 7 or the remaining
full-model, contention-calibration, held-out-performance, and sanitizer gates.

## Questions requiring independent review

- Does the atomic guard cover every planner field read/write pair, including
  pure-vs-mutating decision paths, observer telemetry, and the combined
  decide/start operation?
- Can queue holes, active-ledger holes, or simultaneous completion events cause
  a stale occupancy/rate decision or non-deterministic log?
- Is wrapper/provider lifecycle safe if lookup fails, provider fails, inner
  store destruction fails indirectly, or a release races a lookup under the
  backend's documented threading contract?
- Are the parent-chain expansion and shared-resource admission semantics sound
  for multiple drives sharing one controller and an optional upstream link?
- Is the proposed M3 integration placed at the `colibri.c` load/DAG boundary,
  rather than incorrectly treating the V4 store wrapper as M3 coverage?
- Does the `uint64_t` profile representation and parser reject malformed or
  overflowing input without silently truncating it?

## Implementer self-audit findings — addressed locally, not independent approval

The local self-audit found and then addressed these issues:

- `m3_shd_submit()` now checks active capacity under the planner guard. An
  OpenMP submit/complete regression covers concurrent bounded admission and
  final zero occupancy.
- `m3_shd_decision_log_copy()` now copies a committed, NUL-terminated prefix
  under the guard and reports the required size for undersized buffers. The
  legacy pointer accessor is explicitly quiescent-only.
- A guarded request-ID helper now rejects live duplicates, preserves valid
  queued-to-active promotion, advances explicit IDs without wrapping, and
  rejects allocator exhaustion. Tests cover duplicate/mismatched promotion,
  explicit/automatic allocation, and `UINT64_MAX` boundaries.

The implementation is locally verified, including the combined-operation
contention regression and request-ID evidence join. A narrow independent
concurrency review is still useful; local passes are not being represented as
independent approval.

Additional low-risk self-checks now pass locally: store byte-accounting
overflow saturates before aggregation, duplicate real completion does not
double-retire an active request, duplicate resource IDs and cyclic topology
are rejected, and both GCC `-fanalyzer` and Clang warning-only syntax checks
are clean for the shadow modules.

## Explicitly unproven gates

Measured controller/upstream contention calibration, all-resident shadow-on/off
replay, 57-digest replay evidence, and hosted Linux ASan/UBSan/TSan results are
still absent. Full-model synthetic-profile C-evidence is now recorded, but it
observes only 444 of 7,296 identities and makes no hardware/P claim. Hardware
performance claims remain `NOT_RUN`; this packet must not be read as Gate-7
closure.
