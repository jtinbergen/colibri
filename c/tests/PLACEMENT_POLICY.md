# Qwen3.6 resident-expert placement

## Current policy

The model has 40 layers and 256 experts per layer: 10,240 possible
`(layer, expert)` blocks.  On the current single-GPU desktop run the tier
budget admits exactly 3,642 expert blocks in VRAM.  The CUDA tier fills this
budget in one global order, not with a per-layer quota:

1. `qt_plan_fill()` creates flattened keys `layer * n_experts + expert`.
2. With `HEAT_FILE`, it sorts the complete list by the persisted heat value
   descending.  Without a heat file, the fallback is layer-major, expert-
   ascending order.
3. Candidates are admitted while the home-device budget has room.  The home
   device is `expert % gpu_count`; with one GPU all candidates share GPU0.

The persisted heat is updated by `qt_note()` for routed experts.  It decays
by half every 1,024 tier ticks.  Every 16 ticks the adaptive LFRU path compares
the hottest nonresident loaded expert with the coldest resident expert and
swaps only when `tier_should_promote()`'s 25% plus 4 hysteresis is satisfied.
Thus the warmstart is global heat-ordered and the later policy is adaptive
global LFRU; neither is layer-balanced.

`HOT` and `COLIBRI_RESIDENT` are separate CPU/prompt expert-cache mechanisms;
they do not define the CUDA VRAM placement set.  The current one-GPU topology
therefore places every resident block on GPU0.  Multi-GPU home mapping is
`expert % gpu_count`, but it is intentionally out of scope for this study.

## Measurements

Set `QTIER_ROUTING_FILE` for a decode request.  The engine writes:

- `<file>`: one row per routed `(layer, expert)`, with invocation, GPU-hit,
  CPU-fallback, and CPU lookup/matmul totals;
- `<file>.layers.csv`: per-layer route/hit/fallback counts and cumulative
  MoE, issue, CPU-fallback, and direct `qt_take` timings;
- `<file>.placement.csv`: all 10,240 `(layer, expert)` keys and the exact
  resident snapshot at export time.

The request reset makes these counters request-local.  Divide cumulative
layer timings by `calls` (and `qt_take_ms` by `qt_take_calls`) for ms/token.
Use `COLI_CUDA_TIMING_DUMP=0` and `QTIER_TIMING_DUMP=0` for performance runs;
the CSV timing dump changes scheduling and is diagnostic only.

## Offline policies

`c/tests/simulate_placement.py` keeps capacity fixed at 3,642 and compares:

- `current`: the exported exact resident snapshot;
- `global_hot`: the 3,642 keys with the highest invocation count;
- `layer_balanced`: per-layer quotas (`3642 // 40`, with the remainder given
  to the first layers), then hottest experts inside each layer;
- `critical_path`: global top score
  `invocations * fallback_cost_ms_per_route * (1 + layer_penalty)`.

For an expert, `fallback_cost_ms_per_route` is measured CPU get plus matmul
time divided by observed fallback count; missing values use the layer median,
then the global median.  `layer_penalty` is
`max(0, layer_qt_take - median_layer_qt_take) / median_layer_qt_take`.

The makespan prediction is intentionally a conservative ranking heuristic:
non-MoE work stays fixed, GPU `qt_take` scales with resident-route count,
CPU fallback scales with measured fallback cost, and the two paths overlap as
`max(GPU, CPU)`.  Since Qwen layers are sequential, predicted per-layer costs
are summed.  It predicts opportunity, not guaranteed throughput; a matched
runtime run is required before changing the placement policy.

For a controlled A/B test, the simulator can emit QTH1 heat files with
`--heat-dir`.  The runtime accepts these through its existing `HEAT_FILE`
interface.  Set `QTIER_LFRU=0` for the test so adaptive swaps do not change
the selected set during the request; this is a measurement control, not a
new production policy.  The engine writes the heat file back on shutdown,
so use a copy of the generated file for each run.
