# Packet: Slice 7d-ext — deterministic fixtures (fake clock)

Deliverable: extend the existing 7d fixtures
(`c/tests/test_m3_shadow_plan.c`) with compute-island scenarios.

## 1. Scope and invariants

- Fake monotone clock, fixed service curves, fixed event queue. No sleeps,
  no real I/O, no wall-clock assertions.
- Time precision 1 ns everywhere; same precision in predictor and fixture.
- Coverage includes: heterogeneous drive tails (already in scope), per-island
  transfer cost, residency choice, no-GPU fallback, priced join.
- Decision logs are deterministic: identical snapshot+topology+profile hash
  yields byte-equal log.

## 2. Tasks

Add the following fixtures. Each is one entry in
`m3_shadow_plan_test_fixtures[]` plus a corresponding golden log.

### 2.1 Slow-link island

Inputs:

- expert 200 MiB
- island X: link 12 GB/s, source drive 5 GB/s
- island Y: link 3 GB/s, source drive 5 GB/s
- activation_in + result_out: 2 MiB

Expected:

- compute on X is preferred
- if X is busy past `consumer_need`, planner evaluates Y; if Y also missed,
  log shows `cost cited` for both link and drive
- no implicit assumption that streaming is faster than transfer-of-activation

### 2.2 Pinned-vs-streamed

Inputs:

- hot expert, 200 MiB, requested twice (T0 and T0+Δ)
- island G0: VRAM-resident copy
- island G1: disk source 5 GB/s, link 12 GB/s

Expected:

- first request: planner logs `streamed-weights cost X`, `resident-compute
  cost Y`, picks whichever fits `consumer_need`
- second request: G0 admission unchanged; planner does not duplicate the
  load
- log includes both costs explicitly

### 2.3 No-GPU fallback

Inputs:

- topology with zero `gpu_island` rows
- one CPU island, one DRAM, one drive

Expected:

- planner admits a CPU-only plan
- no `UNKNOWN` collapse on the missing kind
- shadow on/off decision logs are byte-equal in shape

### 2.4 Priced join

Inputs:

- two experts A, B on different islands
- one consumer (down-projection) on island C
- edges: A→C transfer cost t1, B→C transfer cost t2, with t1 < t2

Expected:

- planner places reduction on C only when both ready
- if t1 + t2 > slack, planner logs `priced-join miss` and considers
  alternative consumer island
- cost cited for each edge

## 3. Gate 7d-ext

- C: each fixture returns the documented decision byte-for-byte; replay
  determinism test passes for every fixture.
- M: each decision log cites the resources, costs, and reason.
- P: no perf claim. The fixtures exist to prove the placement rule, not
  hardware.
