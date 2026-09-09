# Packet: Slice 7c-2 — fluid sim (extends m3_shd_predict)

Deliverable: harden the existing event-driven fluid sim in `m3_shd_predict`
to consume 7b-ext profiles. Same algorithm; broader input space.

## 1. Scope and invariants

- Carries forward the existing fluid-sim model from the base plan §7c:
  startup phase, transfer phase, completion phase, serial conversion
  stage, `queue_age_ns` priority.
- Uses 7b-ext profiles (drive / controller / upstream / link / GPU / memory).
- Held-out accuracy is the activation gate. Bad accuracy → model review,
  no hidden heuristic.
- All rate arithmetic in `uint64_t`. The existing
  5,000,000,000-byte/s regression is preserved as a guard.
- Same precision / rounding as 7d / 7d-ext.

## 2. Tasks

1. Extend the per-resource rate resolution to consume the new resource
   kinds. Drive / controller / upstream remain identical; new kinds
   contribute to `min_r(rate_r)` along the same code path.
2. Extend the cost model with the two functions
   (streamed-weights / resident-compute) and integrate the choice into
   the score, not the decision (decision is still local proxy).
3. Add per-fixture accuracy reporting: per-drive, per-controller, per-link
   ready-time error, deadline-miss counts, plus aggregate bytes/s.
4. Add a held-out evaluation harness `c/tools/m3_shadow_eval.py` that
   consumes the runtime CSV from
   `docs/results/m3-execution-step7-m3-runtime-2026-09-07.md` and produces
   per-machine accuracy summaries.

## 3. Evidence

- All 7d + 7d-ext fixtures green.
- Held-out evaluation on machine A: per-drive ready-time error, deadline
  misses (actual / predicted / guarded).
- Shadow on/off equality preserved.

Artifact: `docs/results/m3-execution-slice-7c-2-<date>.md`.

## 4. Gate 7c-2

- C: identical outputs with predictor on/off.
- M: per-fixture decision log; replay determinism.
- P: held-out accuracy within a pre-registered threshold; otherwise
  `INCONCLUSIVE`, never promoted silently. Out-of-scope routes (`S>1`,
  spec, URING, GPU execution on A) tested for fallback only.
