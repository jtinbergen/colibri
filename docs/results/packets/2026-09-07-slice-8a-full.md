# Packet: Slice 8a-full — bounded active planning, source + admission + defer

Deliverable: activate the bounded planner behind 7c-1 + 7c-2 + 8a-lite.
One behavior change (the activation flag) carries the package.

## 1. Scope and invariants

- Admission contract from base plan §8 unchanged: dispatcher owns real
  counters; workers report completion through the existing path; no extra
  lock-free scheduler.
- State path: `QUEUED -> RESERVED -> ISSUED -> COMPLETED/FAILED -> RETIRED`,
  with cancel and submit-error handling as written.
- I/O resource budget and weight buffer lease have separate endpoints; the
  existing `m3_shadow_m3_runtime` seam is the carrier.
- Existing PIPE-batch slots are not repurposed as cross-layer prefetch
  storage.
- First active integration stays within the supported PIPE generation;
  unknown mappings fall back before publication.

## 2. Tasks

1. Wire the planner’s bounded decision log into the live dispatcher. The
   existing `m3_shadow_m3_runtime.c` already records one row per
   `expert_load()`; extend it to record the chosen path, the
   counterfactual, and the impact summary.
2. Add a one-shot A/B between planner-off (existing) and planner-on on
   machine A with the A2 winner configuration.
3. Add a regression: when no profile is configured, the dispatcher
   must take the existing path; assert byte-equal output.
4. Add a regression: when admission is exceeded, the request waits; no
   overbooking, no double-free.
5. Capture the per-resource path occupancy snapshot
   (`m3_shadow_m3_runtime.c:42`) and verify it matches the planner’s
   predicted occupancy for the same admission.

## 3. Evidence

- Shadow on/off equality at replay.
- Held-out tok/s + felt wait, paired 5x alternating.
- Decision-log inspection: chosen path matches predicted best in
  > 80% of demand reads on held-out prompts.

Artifact: `docs/results/m3-execution-slice-8a-full-<date>.md`.

## 4. Gate 8a-full

- C: identical outputs with planner on/off.
- M: traces show bounded admission, no overbooking, demand not starved
  by prefetch (8b).
- P: pre-registered threshold (≥5% median ms/token in the A-track
  target regime, paired 95% interval excludes zero, ≤3% regression in
  controls). Otherwise `INCONCLUSIVE`.
