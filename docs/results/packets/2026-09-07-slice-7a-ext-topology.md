# Packet: Slice 7a-ext — topology + bounded data model with compute islands

Deliverable: extend `c/m3_shadow_plan_calib.h` resource + `c/m3_shadow_plan.h`
text format with compute-island and memory-domain kinds. Data-model only; no
read-path touch.

## 1. Scope and invariants

- Kinds added alongside existing `drive/controller/upstream`:
  `cpu_island`, `gpu_island`, `memory_domain`, `link/interconnect`.
- `cpu_island`: ISA caps bitfield (`AVX1`, `AVX2`, `FMA`, `AVX512`, `VNNI`,
  `SVE2`, `i8mm`), physical cores, parent memory-domain id.
- `gpu_island`: arch (e.g. `sm_61`, `sm_120`, `cdna2`, `gfx1100`), VRAM
  bytes, parent PCIe link id, P2P class id, eligible flag.
- `memory_domain`: total bytes, bandwidth class (`LPDDR5x-8533`, `DDR4-3200`,
  `HBM2e`, etc.), kind (`DRAM`, `VRAM`).
- `link/interconnect`: rate B/s, latency class, shared-group id (anything
  sharing the same root complex or NUMA hop is in one shared group).
- Per-machine `eligible/ineligible` is a planner-level policy, not a kind
  field. Machines with zero eligible `gpu_island` rows must collapse to
  CPU-only paths.
- Invariant INV-10 from `m3_shadow_plan_calib.h` (no fabricated capacity) is
  preserved verbatim. Missing measurements stay `UNKNOWN`.
- Shadow on/off equality holds: the planner is observer-only.

## 2. Tasks

### 2.1 Header extensions

1. Add the four kinds to `m3_shd_resource_kind_t` and the parser enum.
2. Extend `m3_shd_resource_t` with kind-specific payloads:
   - ISA caps (uint32_t bitfield).
   - VRAM (uint64_t bytes).
   - PCIe parent id (resource id; no implicit inference).
   - P2P class (uint16_t).
3. Add `m3_shd_eligibility_t` per machine entry, kept in the planner handle
   (lives next to the existing `m3_shd_config_t`).

### 2.2 Text format extensions

Extend `m3_shadow_plan.h:178-191` format. New lines:

```text
cpu_island  <id> <parent-mem> <isa-bits> <cores>
gpu_island  <id> <parent-link> <arch> <vram-bytes> <p2p-class> <eligible>
memory_dom  <id> <total-bytes> <bandwidth-class> <kind>
link        <id> <parent-cpu> <rate-bps> <latency-class> <shared-group>
machine     <machine-id> <group-of-cpu-island-ids> <group-of-gpu-island-ids>
```

Parser rejects unknown kind, duplicate id, cyclic parent, length-overflow
chain. Acceptance of any new kind is gated on §3 fixtures.

### 2.3 Fixtures

Add four new parser fixtures under
`c/tests/test_m3_shadow_plan_topology.c`:

- Slow-link island: two `gpu_island` rows on links 12 GB/s vs 3 GB/s.
- Pinned-vs-streamed: `gpu_island` with VRAM-resident copy + disk source
  on the same machine.
- No-GPU: topology with zero `gpu_island` rows; planner still admits the
  CPU-only path without `UNKNOWN` collapse.
- Cyclic parent rejection: parser returns error.

### 2.4 Cost model (shadow only)

Add `m3_shd_cost_model_t` carrying per-(expert, island, source) bytes:

```text
streamed_weights_bytes = expert_bytes                    (GB-scale)
resident_compute_bytes = activation_in + result_out      (KB-MB)
transfer_cost_ns = bytes / link.rate_bps + link.startup_ns
```

This is informational only in this slice; no admission or policy decision
is built on it.

## 3. Gate 7a-ext

- C: parser rejects invalid / cyclic / duplicate / overlong chains to
  `UNKNOWN`. Existing 7d fixtures stay green.
- M: replay-determinism (byte-equal log on identical snapshot+topology+profile
  hash). Shadow on/off equality.
- P: no perf claim. The cost model is shadow-only.

## 4. Files

- Touch: `c/m3_shadow_plan.h`, `c/m3_shadow_plan_calib.h`,
  `c/m3_shadow_plan.c` (parser only).
- New: `c/tests/test_m3_shadow_plan_topology.c`,
  `docs/results/m3-execution-slice-7a-ext-<date>.md`.
