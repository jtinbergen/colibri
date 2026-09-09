# Packet: Slice 9b — affinity for all islands (NUMA / GPU)

Deliverable: extend Step 9 NUMA ownership to cover CPU + GPU + memory
islands. Per-machine A/B; absent hardware stays `NOT_RUN`.

## 1. Scope and invariants

- Tile ownership, worker placement, and weight placement are decided
  per island.
- Gate/up-intermediate sharing cost (shared vs replicated vs transferred)
  is counted explicitly.
- Remote work-stealing is bounded and visible in the trace.
- Allocator placement and effective thread affinity are both measured.
- `COLI_NUMA=1` alone does not prove local ownership; the slice measures
  the launch mask and effective binding.

## 2. Tasks

1. Extend the existing NUMA ownership path to consume the new resource
   kinds.
2. Add per-machine A/B: one island vs two islands at the same total
   residency. Compare interleave vs ownership.
3. Add a measurement of launch-mask and effective binding for both
   CPU and GPU kernels.
4. Add a regression that asserts no in-use weight is moved across
   islands after a placement decision.

## 3. Evidence

- Per-machine NUMA evidence report (existing benchmarks on the same
  Xeon Silver 4510 box already measured +13% / +40% on 2-socket /
  4-socket CPU-only; this slice extends the methodology).
- Held-out tok/s + felt wait, paired 5x.
- Trace evidence: no remote steal in steady state.

Artifact: `docs/results/m3-execution-slice-9b-affinity-<date>.md` and
per-machine appendices.

## 4. Gate 9b

- C: identical outputs.
- M: NUMA / affinity measurements show the chosen placement.
- P: one vs two islands A/B; absent hardware `NOT_RUN` per machine.
