# Packet: Slice 11 — heterogeneous placement (shadow)

Deliverable: placement-aware DAG, shadow-only, behavior unchanged. Carries
the placement cost from 7a-ext into a real decision, log only.

## 1. Scope and invariants

- Decision = (expert, island, source copy, streamed-weights vs
  resident-compute).
- Counterfactual only; the engine route is untouched in shadow.
- DAG edges carry transfer cost + contention. Node ready = max over
  dependencies of (completion + transfer).
- Reduction point is a priced join: planner picks the island that
  minimises inbound transfer subject to the existing router order.
- Per-machine eligibility (zero `gpu_island` on D → CPU-only fallback).
- No claim of minimal total DAG latency; the placement is a local
  proxy, scored under the same precision as 7c-1 / 7c-2.

## 2. Tasks

1. Extend `m3_shd_predict` (or add a sibling) to score placement
   candidates per the cost model in 7a-ext.
2. Add the bridge translation in `m3_shadow_m3_bridge.{c,h}` so the
   placement decision sees the consumer node and the join cost.
3. Add fixtures from 7d-ext (`slow-link`, `pinned-vs-streamed`, `no-GPU`,
   `priced-join`) wired into the placement decision.
4. Add a counterfactual log line per forward: chosen placement, why,
   expected reduction cost.
5. Add a regression: shadow-on with placement enabled vs shadow-off
   must produce byte-equal engine output.

## 3. Evidence

- All 7d + 7d-ext fixtures green.
- Shadow on/off replay equality.
- Counterfactual log: chosen placement correlates with predicted best
  on held-out prompts.

Artifact: `docs/results/m3-execution-slice-11-placement-shadow-<date>.md`.

## 4. Gate 11

- C: identical outputs.
- M: counterfactual log carries resources, costs, reason; replay
  determinism; no engine-route change.
- P: held-out accuracy report; per-machine gate. Activation of any
  placement behavior change requires a separate explicit approval
  outside this slice.
