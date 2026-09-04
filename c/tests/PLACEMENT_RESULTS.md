# Placement investigation result

The 64-token DP4A export used for the offline study contained 20,160 routed
expert invocations over 40 layers and 4,116 distinct expert blocks.  The
runtime snapshot contained exactly 3,642 resident blocks.

Offline prediction from `simulate_placement.py`:

| policy | slots | predicted hitmix | predicted CPU fallbacks | predicted MoE improvement |
|---|---:|---:|---:|---:|
| current | 3,642 | 64.43% | 7,170 | +14.825 ms/token |
| global_hot | 3,642 | 97.65% | 474 | +57.410 ms/token |
| layer_balanced | 3,642 | 97.00% | 604 | +54.438 ms/token |
| critical_path | 3,642 | 97.53% | 498 | +58.014 ms/token |

The prediction is a ranking heuristic, not a calibrated full-token model.
It assumes the measured routing distribution repeats, scales resident
`qt_take` with resident route count, overlaps CPU and GPU expert work with a
`max()` model, and holds non-MoE work constant.

## Matched runtime check

The first runtime candidate was `global_hot`, encoded through the existing
QTH1 `HEAT_FILE` interface.  Both runs used DP4A, 32 generated tokens, the
same model/configuration, 3,642 slots, `QTIER_LFRU=0`, and no timing dump.

| metric | current snapshot | global_hot |
|---|---:|---:|
| throughput | 1.629 tok/s | 1.545 tok/s |
| latency | 613.9 ms/token | 647.4 ms/token |
| GPU hitmix | 66.4% | 93.5% |
| CPU fallback routes | 3,330 | 640 |
| MoE wall | 80.5 ms/token | 57.5 ms/token |
| direct `qt_take` | 15.2 ms/token | 20.4 ms/token |
| GPU sampled utilization | 35.5% average | 34.1% average |
| GPU sampled power | 51.0 W average | 51.8 W average |

Problem layers improved in isolation:

| layer | current hitmix / MoE | global_hot hitmix / MoE |
|---:|---:|---:|
| 38 | 53.6% / 2.675 ms/token | 87.1% / 1.802 ms/token |
| 39 | 50.0% / 2.511 ms/token | 87.5% / 1.638 ms/token |

This is currently a placement-only FAIL for the end-to-end PASS criterion,
but it does not falsify `(layer, expert)` placement.  It falsifies the simpler
objective of maximizing resident hit rate on one GPU.  The result is:

- resident-expert placement hypothesis: strengthened;
- placement-only, single-GPU hit-rate objective: failed;
- CPU/GPU compute-island balancing hypothesis: strengthened.

The CPU fallback path is not merely a slow miss path.  In the current system
it can overlap with resident GPU work.  Moving almost all routes onto one
Pascal reduces CPU fallback time but raises `qt_take` enough to increase the
completion-critical makespan.  The 32-token throughput difference is
directional only and needs a longer matched run; the MoE and `qt_take`
changes are already large enough to establish the load-balancing effect.

The next model and experiment must therefore minimize expected per-layer
completion makespan given CPU capacity, GPU capacity, resident placement, and
overlap.  With multiple Pascal islands, the objective should distribute
resident routes across GPU0/GPU1/GPU2 while retaining CPU work where that
reduces the latest completion time, rather than concentrating all resident
work on one card.

The first overlap-aware `makespan_balanced` runtime candidate also exposed a
missing system term.  Although it reduced the CPU window to 28.5 ms/token,
GPU event work grew to 536.2 ms/token and `qt_take` to 512.4 ms/token, with
the backlog concentrated in layers 36–39.  The local per-layer model did not
account for a GPU queue accumulating across sequential layers.  The next
simulator must therefore be a timeline model with device backlog and
completion boundaries, not another independent layer ranking.
