# Step 8c shadow-residency action invariants — 2026-09-07

## Scope

The first 8c implementation is a pure planner function.  Given immutable
candidate facts and a fixed per-domain slot/byte budget, it returns a bounded,
deterministically ordered list of `PIN`, `MOVE`, or `EVICT` proposals.  It
does not access `Model`, `ESlot`, `ecache`, `ecap`, a buffer pointer, or an
I/O request.  It cannot cause an engine action.

## Action eligibility and scoring

1. Every candidate has immutable content/copy identity and content generation,
   source and target domain, bytes, expected critical-path wait saved,
   movement cost (both in ns), and an `in_use` fact.  A resident candidate
   additionally names its exact resident domain/slot, cache snapshot
   generation, and explicit `evictable` fact (published, unpinned,
   non-reserved, non-busy, non-pilot-owned).
2. Score is `max(0, expected_wait_saved - movement_cost) / bytes`; zero-byte,
   unavailable, unsupported, or in-use candidates cannot propose a move,
   eviction, or pin.  Integer arithmetic and a stable tie break make output
   reproducible.
3. `MOVE` requires available immutable source content/generation, an eligible
   target, and target slot/byte demand that fits the caller-supplied available
   snapshot budget.  `PIN` requires an existing resident target entry.  The
   planner never invents a capacity, changes a budget, or claims a
   cross-domain transfer is free.
4. `EVICT` is legal only for an explicitly resident, evictable, non-in-use,
   unpinned candidate in the same cache snapshot.  It returns its exact
   resident slot/bytes as evidence only; it never increases the available
   budget or makes a move/pin fit in the same planner call.  The shadow
   function does not sequence live reuse.

## Ordering and unsupported topology

1. Domain IDs are opaque.  NUMA/Infinity Fabric/QPI/PCIe/NVLink costs must
   arrive as measured movement-cost facts; there is no platform-name default.
2. Actions sort by descending rational score without floating point, using an
   overflow-safe fraction comparison, then by critical-path win, content ID,
   content generation, source domain, and target domain.  Input order cannot
   affect the result.
3. Unknown target domain/budget, candidate identity collision, overflow, or
   unsupported domain kind produces no action for that candidate.

## Integration boundary

The existing `ecap`/`ecache` audit remains binding.  A future live consumer
must map an action to the existing cache index/reserve/publish/hide helpers
under their ownership and safe-point rules, and must prove that the target
slot is not busy.  This 8c slice produces evidence only; activation remains
blocked on that integration, Step-7 calibration, and held-out P evidence.
