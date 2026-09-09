# Step 8b prefetch/promotion invariants — 2026-09-07

## Scope

8b extends the reviewed 8a dispatcher.  It may create a bounded prefetch only
through that same dispatcher and only in deterministic fake-I/O fixtures.
Production remains untrusted/fallback-only until the explicit Step-7 and
Step-8 activation gates are met.

## Demand reserve and admission

1. Demand and prefetch use the same resource counters and the same full-path
   reservation transaction.  There is no independent prefetch bandwidth,
   controller, upstream, byte, or request budget.
2. Each storage resource may declare a demand reserve in requests and bytes.
   A prefetch admission must leave at least that reserve after accounting for
   already reserved demand work.  A demand request is not rejected merely to
   preserve a prefetch reserve.
3. The admission predictor is a pure, injected fixture function.  A prefetch
   is deferred when its predicted addition raises the projected demand wait;
   a missing prediction is fail-closed for prefetch.  Demand retains the
   established 8a behavior.
4. The dispatcher logs/reports the exact reserve and predicted-wait decision
   used for the prefetch admission.  A test may model bad prediction, but it
   cannot treat a synthetic profile as an active production capacity.

## Deduplication and promotion

1. A prefetch has a stable coalescing key for the selected immutable copy and
   byte range.  A demand for the same key promotes the existing request; it
   does not enqueue, reserve, issue, or load a second copy.
2. Promotion adds a demand consumer lease to the existing request.  The I/O
   reservation and destination buffer remain owned by that original request.
   The buffer cannot be released until I/O is terminal and every attached
   consumer lease is retired.
3. Promotion is valid from `QUEUED`, `RESERVED`, and `ISSUED`.  It cannot
   resurrect a cancelled/failed/retired prefetch, and it may not silently
   switch source/path after any I/O part is issued.

## Cancellation and reset

1. A speculative prefetch may be cancelled only before issue: `QUEUED` or
   `RESERVED` transitions to `CANCELLED` and releases no / exactly one
   unissued transaction respectively.
2. An issued prefetch may only be marked no-longer-wanted.  It keeps the full
   resource reservation and buffer lease until the last terminal I/O event;
   an attached demand consumer still receives its normal result.
3. Reset follows 8a: no new admission, queued/reserved rollback, issued
   drain, then all producer and consumer leases retired before reuse.

## Required fixture evidence

- prefetch cannot consume configured demand reserve on a shared resource;
- a saturated shared group does not block an independent group;
- missing or regressing demand-wait prediction defers prefetch;
- promotion from queued, reserved, and issued performs no second issue or
  resource reservation;
- pre-issue cancel releases once; issued cancel/no-longer-wanted does not;
- stale/double terminal events and reset preserve zero post-drain occupancy.
