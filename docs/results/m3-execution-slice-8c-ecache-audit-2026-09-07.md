# Step 8c prerequisite — `ecap` / `ecache` audit — 2026-09-07

## Verdict

**Audit complete; no cache-capacity or residency behavior changed.**  A
Step-8c residency policy may initially emit shadow actions only.  It must not
change global/per-layer capacity, reallocate `ecache`, or reuse an `ESlot`
until the sites below are designed and tested together.

## Owned state and fixed-size allocation

| Site | Contract that a variable residency policy must preserve |
|---|---|
| `c/colibri.c:481-482` | `Model` owns `ecache` rows, `ecn` published-row counts, global `ecap`, and the `expert -> slot` index.  They are one coupled representation. |
| `:2637-2653` | Startup records `ecap`, allocates row-pointer/count arrays, then full expert-index arrays.  Index entries initialise to `-1`; a policy cannot make a row capacity differ without preserving this mapping. |
| `:2746`, `:2799` | M3/GLM layer paths allocate each `ecache[i]` at the startup `cap`. |
| `:11944-11977` (`cap_for_ram`) | RAM accounting may shrink `ecap`; optional `CAP_RAISE` reallocates every row and zeroes only the newly added range.  It must run where no pilot/worker holds an `ESlot *`. |
| `:13332-13372` | Destruction iterates to the current `ecap`, destroys segment-owned slot contents, then frees rows, indexes, and count arrays. |

## Lookup, publication, and reuse

| Site | Safety property |
|---|---|
| `:438-449` (`eslot_lru_victim`) | A slab-less hole is reusable only while live slabs are below `ecap`; reserved or busy slots are not eviction victims. |
| `:5917-5927` (`ecache_indexed`) | The expert-index entry, `ecn` bound, and actual slot tag must agree.  Reserved tags are visible only to callers that request them. |
| `:5936-5962` (`unindex`, `publish`, `reserve`, `hide`) | These are the only coherent transitions for the index and `ESlot.eid`: published `eid`, reservation `-(eid+2)`, hidden `-1`. |
| `:6681-6847` | MoE demand pre-scan and hit accounting treat only published pins/LRU entries as resident. |
| `:7866-7878` | DAG promotion swaps an `ESlot` with workspace and updates index ownership before publication; it is bounded by `ecap`. |

## Cross-layer pilot / real-I/O ownership

| Site | Safety property |
|---|---|
| `:1684-1694` | `g_pilot_mx` serializes the future-layer pilot writer with scans of that same row; current-layer execution is separated by the existing barrier. |
| `:8090-8145` | Synchronous pilot path reserves a visible slot under `g_pilot_mx`, publishes only after success, and hides a failed reservation. |
| `:8171-8221` | uring pilot path uses the same reservation/index protocol; in-flight slots are not victims and a failed submission or completion hides the slot. |
| `:8397-8411` | Future-layer residency scans lock before reading `ecache`/`ecn` while the pilot can mutate them. |
| `:10060-10132` (`rss_guard`) | RSS eviction runs at a safe point, locks the row, skips `eid < 0` and busy slots, hides before releasing slab data, and can lower `ecap`.  It expressly avoids compaction because a pilot read can retain an `ESlot *`. |

## Consequences for Step 8c

1. The existing Python `c/tools/residency_sim.py` is an offline replay model;
   it is not an engine ownership authority and cannot directly choose live
   `ecap` values or slot pointers.
2. A first optimizer may rank shadow `pin`, `move`, and `evict` proposals by
   expected critical-path win per byte minus movement cost.  It must output no
   live action until a separate integration establishes which existing helper
   performs each transition.
3. Any variable cache budget must include all rows, `ecn`, index arrays,
   startup allocation, `cap_for_ram`, `CAP_RAISE`, `rss_guard`, DAG promotion,
   pilot reservation, and destruction in one ownership review.  A planner
   flag alone is insufficient.
4. No in-use, reserved, or pilot-owned `ESlot` may be evicted/reused.  PIPE
   batch slots remain excluded from cross-layer prefetch storage.
