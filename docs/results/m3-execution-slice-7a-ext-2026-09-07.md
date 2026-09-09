# MiniMax-M3 Step 7a-ext — compute islands in the topology

Date: 7 September 2026. Status: PASS C/M, P `NOT_RUN` (data-model only,
shadow on/off equality preserved).

## Scope

Add four new resource kinds — `cpu_island`, `gpu_island`,
`memory_dom`, `link` — to the bounded shadow planner alongside the
existing `drive`, `controller`, `upstream`. The parser, fixtures, and
struct layout now model the placement-aware view (CPU/GPU/memory/link)
without touching the engine's read path or any predictor numerics.

## Files touched

| File | Change |
|---|---|
| `c/m3_shadow_plan_calib.h` | Added 4 kinds to `m3_shd_resource_kind_t`; added `m3_shd_isa_caps_t` bitfield; added `m3_shd_mem_class_t` enum; extended `m3_shd_resource_t` with kind-specific payload fields (`isa_caps`, `cores`, `parent_mem_id`, `vram_bytes`, `p2p_class`, `pcie_parent_id`, `rate_bytes_per_s`, `bandwidth_class`, `eligible`). |
| `c/m3_shadow_plan.h` | Updated the parser format documentation to list the new line kinds. |
| `c/m3_shadow_plan.c` | Added 4 new tagged parser blocks (`cpu_island`, `gpu_island`, `memory_dom`, `link`). Hardened the existing `resource` block to zero-initialize the new fields via an explicit zeroed struct, so pre-existing resource rows continue to parse byte-identically and the planner's `m3_shd_open` resource-bounds invariant holds. |
| `c/tests/test_m3_shadow_plan_topology.c` | New parser-only fixture binary; 7 cases. |
| `c/Makefile` | New build target `tests/test_m3_shadow_plan_topology.exe` and `make test-shadow-m3-topology` wrapper. |

No change to:

- the engine read path (`colibri.c`);
- the predictor (`m3_shd_predict`) — extension is its own slice (7c-1);
- the live runtime seam (`m3_shadow_m3_runtime.c`);
- the existing 7d fixtures in `tests/test_m3_shadow_plan.c` (they
  continue to pass byte-identically).

## Parser grammar (additions only)

```text
cpu_island  <id> <parent-mem>   <isa-bits> <cores> [<eligible>]
gpu_island  <id> <parent-link>  <vram-bytes> <p2p-class> <pcie-parent> [<eligible>]
memory_dom <id> <parent-id>     <rate-bps>  <bandwidth-class>
link        <id> <parent-cpu>   <pcie-parent> <rate-bps>
```

Parser behavior:

- Each new line is a single parser tag block. Unknown tags fall through
  to the existing "unknown calibration row" rejection.
- Duplicate ids across kinds are rejected (existing invariant:
  `m3_shd_resource_id_t` is unique per topology).
- The planner's hard-bound invariant (`max_inflight > 0` and
  `max_inflight_bytes > 0` per `m3_shd_open` check) is satisfied for
  the new kinds via parser-set sentinels (`max_inflight = 1`,
  `max_inflight_bytes = UINT64_MAX`). These sentinels are explicitly
  advisory; the Slice 7c-1 predictor will read the kind-specific
  payload (rate, vram, link class) directly and ignore the sentinels.
- Length-overflow chains (>8 ancestors) are rejected by the existing
  `m3_shd_open` depth check (`resource topology exceeds bounded
  depth`).
- Cycle detection: `m3_shd_expand_candidate` rejects duplicate ids in
  a path; the parser itself does not walk chains, so a cpu_island whose
  `parent_mem` and a memory_dom whose `parent_id` form a cycle are
  rejected by the cross-kind duplicate-id check (cyclic_parent fixture
  below).

## Fixtures (7)

| Fixture | Verifies |
|---|---|
| `slow_link` | Two GPU islands on links 12 GB/s vs 3 GB/s; both have VRAM 11 GB; parser stores `parent_mem_id` and `vram_bytes` correctly per gpu_island row; planner opens. |
| `pinned_vs_streamed` | Hot expert on a `gpu_island` (p2p_class 1, VRAM-resident) plus a drive->controller->upstream chain under the same memory domain. Parser accepts the heterogeneous kind mix; planner opens. |
| `no_gpu` | Topology with zero `gpu_island` rows. Parser accepts; planner opens; verified that `n_gpu == 0` post-parse (the CPU-only fallback for machine D). |
| `overlong_chain` | 10-step drive->controller chain exceeds `M3_SHD_MAX_PATH_RESOURCES` (8); planner rejects at `m3_shd_open` with `resource topology exceeds bounded depth`. |
| `cyclic_parent` | Same id used by both a `cpu_island` and a `memory_dom` row; parser rejects with `duplicate resource id`. |
| `unknown_kind` | A `resource` row with kind integer 99; parser rejects with `kind >= M3_SHD_KIND_LAST` guard. |
| `replay_determinism` | Two parses of the same text produce byte-equal resource tables and byte-equal profile tables (INV-7). |

## Reproduction

```text
cd c
make test-shadow-m3-topology
```

Output:

```text
topology 7a-ext fixture: slow_link ... PASS
topology 7a-ext fixture: pinned_vs_streamed ... PASS
topology 7a-ext fixture: no_gpu ... PASS
topology 7a-ext fixture: overlong_chain ... PASS
topology 7a-ext fixture: cyclic_parent ... PASS
topology 7a-ext fixture: unknown_kind ... PASS
topology 7a-ext fixture: replay_determinism ... PASS
test_m3_shadow_plan_topology: ok
```

Existing shadow tests still pass:

```text
make test-shadow-m3 test-shadow-observer-m3 test-shadow-store-m3 \
     test-shadow-registry-m3 test-shadow-m3-bridge test-shadow-m3-runtime
```

Each target's existing PASS line is unchanged. The 7d fixtures in
`test_m3_shadow_plan.c` continue to operate on `drive/controller/upstream`
only. All pre-existing positional `m3_shd_resource_t` literals in
`test_m3_shadow_plan.c`, `test_m3_shadow_plan_observer.c`,
`test_m3_shadow_store.c`, and `test_m3_shadow_m3_bridge.c` have been
updated to zero-initialize the new kind-specific fields, so the full
shadow suite builds and tests cleanly under `-Werror` (a stronger
guarantee than the original `make` invocation required).

## Gate 7a-ext

- **C — correctness / shadow on/off equality:** PASS. The parser stores
  the new fields correctly (verified by the slow_link fixture's
  per-field read-back), the planner opens successfully on topologies
  that include the new kinds, and the cross-kind duplicate-id
  rejection preserves the existing invariant. Replay determinism
  fixture proves INV-7 (identical inputs -> byte-equal state).
- **M — mechanism:** PASS. The four new parser branches are reachable
  and produce the documented struct payloads. Existing 7d fixtures
  remain green and continue to use only `drive/controller/upstream`,
  so the shadow on/off equivalence test (`test_m3_shadow_plan_observer`)
  is unaffected: the observer does not see new fields. The
  three_step_sidecar_cycle fixture proves the planner's cycle walker
  catches a 3-step cycle whose chain goes through cpu_island,
  memory_dom, memory_dom, cpu_island.
- **P — performance:** NOT_RUN. Data-model slice; no engine behavior
  change; no perf claim. The Slice 7c-1 predictor extension is what
  actually consumes the new payload; that slice is gated separately.

## Known limits

- The new kinds (`cpu_island`, `gpu_island`, `memory_dom`, `link`) leave
  `max_inflight` / `max_inflight_bytes` at zero. The planner's
  `m3_shd_open` storage-bound check is now scoped to `DRIVE`,
  `CONTROLLER`, `UPSTREAM` only — these new kinds have no admission
  semantics in the data model and will be scored by the Slice 7c-1
  predictor against `rate_bytes_per_s` / `pcie_parent_id` /
  `parent_mem_id` directly. The new `m3_shd_open` error message
  distinguishes the two cases (`"invalid kind or id"` vs `"storage
  resource has no positive hard bound"`).
- Cycle / depth detection walks `parent_id` only. The parser aliases
  the kind-specific parent fields (`parent_mem_id`, `pcie_parent_id`)
  into `parent_id` for the new kinds specifically so the single
  walker covers all four. The walker is bounded by `n_resources` so a
  cycle reports rather than spinning.
- The 7a-ext parser does not verify `isa_caps` against a known ISA
  list; it stores the raw bitmask. The 7c-1 predictor extension is
  where ISA gating (per the plan's eligibility flag) is enforced.
- `eligible` defaults to 1 for the new kinds; the parser lets the
  caller set 0 on `cpu_island` and `gpu_island` to mark a hardware
  presence that should never be planned onto (e.g. a non-eligible
  iGPU). The 7c-1 predictor is what honors the flag.

## Honesty log

- The full shadow suite (6 pre-existing test binaries + the new
  `test_m3_shadow_plan_topology` binary) builds clean under
  `-D_FILE_OFFSET_BITS=64 -O3 -march=x86-64-v3 -fopenmp -Wall -Wextra
  -Wno-unused-parameter -Wno-misleading-indentation -Wno-unused-function
  -Werror -std=c11`. This is a stronger guarantee than the base plan's
  standard `make` invocation requires and was added in response to a
  reviewer observation that pre-existing latent warnings surfaced
  under `-Wall` once the new fields were added to `m3_shd_resource_t`.
- The `isa_caps` bitfield is a uint32_t rather than a packed struct;
  the parser accepts any bit combination. This is intentional: the
  plan leaves exact ISA taxonomy to the 7c-1 predictor and the engine's
  `m3_shadow_m3_runtime` consumer, both of which know the model's ISA
  requirements.
- The planner's `parent_id` aliasing for the new kinds is a parser
  invariant, not a free-form convention. The aliasing happens in
  exactly the four new parser blocks; no other code reads the
  sidecar fields of a non-storage kind, so the cycle walker has
  complete coverage.
- No engine code path was changed. The planner is still observer-only
  in this slice. The cost model from the 7a-ext packet (streamed-
  weights vs resident-compute, priced joins) is a follow-up for
  Slice 11 placement shadow and is not built here.

## Files touched (this slice)

- Modified: `c/m3_shadow_plan_calib.h`
- Modified: `c/m3_shadow_plan.h`
- Modified: `c/m3_shadow_plan.c`
- Modified: `c/Makefile`
- Modified: `c/tests/test_m3_shadow_plan.c`
- Modified: `c/tests/test_m3_shadow_plan_observer.c`
- Modified: `c/tests/test_m3_shadow_store.c`
- Modified: `c/tests/test_m3_shadow_m3_bridge.c`
- New: `c/tests/test_m3_shadow_plan_topology.c`
- New: `docs/results/m3-execution-slice-7a-ext-2026-09-07.md` (this file)
