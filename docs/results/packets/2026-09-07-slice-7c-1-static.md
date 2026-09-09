# Packet: Slice 7c-1 — queue-aware static predictor

Deliverable: a small, static predictor that selects among equivalent drive
copies using the calibration profiles from 7b-ext. No fluid sim. Shipped
together with A2.

## 1. Scope and invariants

- Predictor is a pure function `predict(snapshot, request, candidate,
  dispatch_time)`. Inputs are const; no planner-owned state mutation.
- Formula (fixed precision, 1 ns tolerance):

  ```text
  rate_r       = C_r(load_class, size_class) / n_r
  arrival      = dispatch_time + startup_r + bytes / min_r(rate_r)
  slack        = consumer_need - arrival - uncertainty_margin
  wait         = max(0, -slack)
  score_k      = Σ wait over candidate-relevant union (new + affected)
  predicted_wait_k = max(0, -slack_k)
  uncertainty_margin = pre-recorded calibration residual (no per-call tuning)
  ```

- `consumer_need` is a soft deadline from the DAG; never from the
  predictor's own I/O completion.
- Score tie-break: earliest `arrival`, then stable drive-ID.
- Score is a local proxy; not a global optimum claim.
- Defer is a private counterfactual; re-evaluated on real completion.
- Same precision / rounding as fixtures. Replay determinism preserved.

## 2. Tasks

1. Wire `m3_shd_predict` for equivalent drive candidates only. Compute
   islands / transfer-cost scoring live in 7c-2.
2. Emit the decision record with snapshot / profile version, request id,
   candidates, resource paths, occupancy before / after hypothetical
   admission, predicted ready / need, uncertainty, impact on existing
   reads, chosen candidate, defer reason.
3. Add an `m3_shd_apply_to_active_path` that records the counterfactual
   without touching the engine route (existing `m3_shadow_m3_runtime` is
   the seam).
4. Add fixtures: two equivalent drives with measured tail difference,
   plus same-controller contention case (parser-only, real capture
   `NOT_RUN` on A).

## 3. Evidence

- Existing 7d fixtures stay green (replay determinism).
- New fixtures: contended siblings, contention with defer, no-profile
  fallback.
- Shadow on/off replay equality.

Artifact: `docs/results/m3-execution-slice-7c-1-<date>.md`.

## 4. Gate 7c-1

- C: identical outputs with predictor on/off.
- M: decision log carries every documented field; replay determinism.
- P: held-out prompt set (not used for calibration) reports paired
  tok/s + felt wait on the A-track. Bad accuracy → `INCONCLUSIVE`, no
  silent fallback.
