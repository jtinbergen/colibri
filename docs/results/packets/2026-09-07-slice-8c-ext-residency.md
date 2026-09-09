# Packet: Slice 8c-ext — residency optimizer across DRAM/VRAM/disk

Deliverable: residency optimizer that weighs expected critical-path win per
byte against movement cost, across all eligible memory kinds.

## 1. Scope and invariants

- Optimizes expected critical-path win per byte, minus movement cost,
  across DRAM / VRAM / disk memory domains.
- Variable per-machine cache budget requires an audit of every
  `ecap/ecache` allocation; not introduced with a planner flag alone.
- Eviction / slot reuse rules from the existing ESlot contract are
  unchanged: no in-use weights are evicted.
- Decision is shadow-only initially; activation only after paired P
  passes the pre-registered threshold.

## 2. Tasks

1. Audit and document every `ecap/ecache` site touched by residency
   changes. Use a dedicated packet; no changes to allocation in this
   slice.
2. Add the residency optimizer as a planner-side function that returns
   a list of move / evict / pin actions per forward.
3. Add fixtures for the convergence behaviour:
   - on machine D (no GPU, large DRAM): converges to ~resident.
   - on machine A (32 GB DRAM, 0 GB VRAM eligible): hot set stays in
     DRAM + a small spill on E: with prefetch.
   - on machine B (6×1080Ti 66 GB VRAM): hot set in VRAM, cold on disk
     with prefetch on heterogenous drives.

## 3. Evidence

- Audit report: every allocation site named.
- Fixture traces: actions per forward, expected critical-path win.
- Held-out tok/s + felt wait on machine A: residency changes do not
  regress below pre-registered floor.

Artifact: `docs/results/m3-execution-slice-8c-ext-residency-<date>.md`.

## 4. Gate 8c-ext

- C: identical outputs with optimizer on/off.
- M: no in-use weights evicted; no duplicate loads; no premature slot
  reuse.
- P: paired tok/s + felt wait; pre-registered threshold; per-machine
  verdict (D converges, A bounded, B banked).
