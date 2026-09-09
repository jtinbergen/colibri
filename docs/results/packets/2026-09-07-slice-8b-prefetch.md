# Packet: Slice 8b — bounded prefetch

Deliverable: bounded prefetch behind demand reserve, promote-without-
duplicate, cancel-before-start only.

## 1. Scope and invariants

- Bytes + inflight caps enforced via the same bounded dispatcher from 8a.
- Demand reserve per shared resource is a separate counter; prefetch
  admission reads it before publishing.
- Promote existing prefetch to demand without re-issuing the read.
- Cancel-pending only on requests that have not started; an already
  running read keeps its full byte reservation until completion.
- Same admission contract as 8a; same precision; same logging.

## 2. Tasks

1. Add a prefetch admission path that calls `m3_shd_predict` with the
   `demand_reserve` adjustment; reject if projected demand-wait would
   rise.
2. Add a promote-on-demand flag: if the next request hits a previously
   prefetched expert, the read is not re-issued; the existing lease is
   handed to the demand task.
3. Add a cancel-pending-only path: requests in `QUEUED` or early
   `RESERVED` can be cancelled; requests past `ISSUED` cannot.
4. Add a regression for each invariant in §1.

## 3. Evidence

- 8a fixtures stay green.
- Held-out tok/s + felt wait with and without prefetch enabled.
- Decision-log inspection: no double-loads, no demand starvation.

Artifact: `docs/results/m3-execution-slice-8b-prefetch-<date>.md`.

## 4. Gate 8b

- C: identical outputs.
- M: prefetch admission traced; demand-reserve respected.
- P: paired tok/s + felt wait with prefetch on/off; pre-registered
  threshold. E:-prefetch remains experimental until A-Track P passes.
