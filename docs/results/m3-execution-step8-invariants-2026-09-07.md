# Step 8 admission/lifetime invariants — 2026-09-07

## Scope and status

This document fixes the ownership contract before implementation of Step 8.
Step 7 is C/M-complete only in its bounded shadow scope.  Consequently this
work begins with deterministic fake-I/O fixtures and an inactive production
seam.  No profile currently available on this host is admitted to control
live source selection, prefetch, or residency.

The first implementation slice is 8a: demand-source selection plus admission
and deferral.  It has fixed residency and creates no new prefetch.  8b adds
prefetch/promotion only after 8a's dispatcher contract is exercised.  8c is
initially planner-side and shadow-only; it must not alter `ecap` or `ecache`
allocation until its separate allocation/index audit is complete.

## Dispatcher ownership

1. Exactly one bounded dispatcher owns every mutable request state and every
   drive/controller/upstream in-flight counter.  A worker or I/O callback may
   report an event but never adjusts a resource counter itself.
2. Admission reserves the complete selected path and the destination-buffer
   lease in one all-or-nothing dispatcher transaction before I/O publication.
   If any resource or lease is unavailable, the request remains `QUEUED` and
   no counter has changed.
3. The source predictor is advisory until a trusted profile is explicitly
   admitted.  A missing mapping, profile, or trusted active seam takes the
   existing engine path before publication.  It never starts a second load.
4. The dispatcher is a single guarded bounded ledger, not another lock-free
   scheduler.  Capacity is explicit and rejects/defer requests before array
   or counter overflow.

## Request identity and lifecycle

Each request identity is `(request_id, generation)`.  Completion, failure,
cancel, reset, and retirement validate both fields; a terminal I/O event also
validates its `part_id`.  An event from a prior generation or an already
terminal part is stale and makes no state or counter change.
The dispatcher additionally returns a private monotone ticket nonce and
matches it on every operation.  External identities may be reused after
retirement, but an old ticket cannot match the replacement request.

The normal state path is:

```
QUEUED -> RESERVED -> ISSUED -> COMPLETED / FAILED -> RETIRED
```

`QUEUED -> CANCELLED -> RETIRED` has acquired no I/O or lease reservation.
`RESERVED -> CANCELLED / FAILED -> RETIRED` releases the unused transaction
once in the dispatcher.  Once `ISSUED`, cancellation is only a pending mark:
the full reservation remains until all issued parts become terminal.  A
per-request release marker makes every duplicate terminal event, terminal
event after reset, or combined submit-failure/reset path idempotent.

The adapter retains `issued_parts`, `terminal_parts`, and a terminal-bitset
for its bounded part domain.  A partial issue keeps the request `ISSUED`,
retains its full reservation, and must drain issued parts before retirement.

## Resource and buffer lifetime

Resource reservations and the destination-buffer lease have separate
endpoints.  A terminal I/O completion may release transfer capacity, but it
does not make a weight buffer reusable: conversion, publication, and all
consumers must be retired first.  Failure/cancel publishes no partial weight
and frees its lease only when no issued part, conversion, or consumer owns it.
Every request therefore carries a lease ID and lease generation, and the 8a
fixture models lease ownership as a distinct counter even though it performs
no conversion or cache placement.

## Reset and isolation

Reset first closes admission, cancels queued work, rolls back unissued
reservations, and drains issued I/O before topology change or lease reuse.
Every resource group is admitted independently under the same dispatcher;
saturation of one drive/controller/upstream path must not prevent an eligible
request on a disjoint resource group.  All tests assert that post-drain
resource and lease occupancy are zero.
Dispatcher teardown first closes admission and refuses to free a live ledger.
The owner must externally quiesce all other API calls around the final close;
the dispatcher pointer itself cannot be safely used after close.

## Required 8a evidence

The dispatcher tests must cover: full queue and full resource path; atomic
multi-resource rollback; bad prediction with a valid fallback; failed source;
stale and double terminal events; cancel before and after issue; reset during
an issued read; no negative counters/double release/early lease reuse; and
an independent group making progress while another is saturated.  Integration
remains disabled by default and must prove the no-profile existing-path
fallback is byte-identical before an active gate can be claimed.
