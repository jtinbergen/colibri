# MiniMax-M3 Step 7 C/M transition — 2026-09-07

## Decision

Step 7 C/M is complete for the bounded shadow-only dependency. Step 8a may
begin as shadow-only preparation. The overall Step-7 Gate remains
`NOT_GATED / NOT_PROMOTED`; no active planner policy is enabled.

This follows the authoritative dependency table in
`docs/minimax-m3-execution-plan-2026-09-05.md`: Step 8 requires Step 7 C/M,
not Step-7 P. The decision was independently checked in a compact Sol review
against the evidence below.

## C evidence

- All 7d deterministic planner fixtures pass, including bounds, shared
  resources, ties, deadline/age policy, unknown calibration, failure cleanup,
  deterministic replay, and unchanged real occupancy.
- The fixed-seed full-model shadow-off/on smoke exits successfully. Both engine
  traces contain 10,366 events and their semantic
  `(kind, layer, expert, resource)` multisets are equal.
- The model-aware generator validates both replicas against the M3 config,
  emits exactly 7,296 expert identities / 14,592 copy rows, and the runtime
  loads all rows. The shadow-on run completes 453 rows across 444 observed
  identities using complete physical paths.
- The tiny runtime replay remains green with shadow off/on semantic output,
  exit status, and trace equality.

## M evidence

- The runtime trace contains actual load, consumer need, predicted ready, and
  actual ready fields. The reproducible report aggregates by drive/path and by
  controller; it has 306 samples on controller 201 and 147 on controller 202.
- The machine profile bridge and `coli plan` read-only integration are green;
  missing calibration stays explicitly `UNKNOWN`/`NOT_RUN`.
- The full 7d fixture set and the bounded runtime/report tests pass. No future
  router choice is injected into the shadow decision.

## Explicit limits

The following remain open and prevent Gate-7 closure or active Step-8a
admission: same-controller/shared-upstream contention, cache-classified and
held-out calibration, all-resident performance, mixed-residency/direct-striped
reads, mmap/io_uring routes, hosted ASan/UBSan/TSan, and the final full Step-7
diff review. Synthetic profiles are evidence-only and cannot drive scheduling.

## Allowed next stage

Prepare 8a shadow-only fixtures and logging against the existing admission
contract. Keep the activation flag off, preserve fallback when no admitted
profile exists, and do not claim tok/s or felt-wait improvement until the
pre-registered P evidence is available.
