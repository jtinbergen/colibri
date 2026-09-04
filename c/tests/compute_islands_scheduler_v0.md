# Compute-island scheduler simulation

Offline first-order model; it does not change runtime placement or scheduling.

## Inputs and model

- Expert rows: 4116; routed invocations: 20160
- Capacity: **3642** resident expert slots
- Timing calibration: `c/tests/compute_islands_v0_balanced_200.csv`
- Measured trace mean layer makespan: **10.845 ms/layer-record**
- Layer cost: `measured pre + dispatch + max(CPU0, GPU0) + measured merge + post`.
- Current kernels, quantization, and reduction order are not changed.

## Policy predictions

| Policy | Slots | GPU hit | CPU fallbacks | Predicted ms/token | vs current |
|---|---:|---:|---:|---:|---:|
| current | 3642 | 64.43% | 7170 | 417.544 | +0.000 |
| global_hot | 3642 | 97.65% | 474 | 432.036 | -14.491 |
| layer_balanced | 3642 | 97.00% | 604 | 430.619 | -13.075 |
| critical_path | 3642 | 97.61% | 481 | 431.353 | -13.808 |
| makespan_balanced | 3642 | 78.73% | 4289 | 386.409 | +31.135 |

## Makespan-balanced policy by layer

| Layer | Slots | GPU hit | CPU ms | GPU ms | Imbalance | Makespan | GPU slack | Class |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 83 | 75.99% | 3.917 | 4.792 | 0.874 | 12.172 | 0.156 | GPU-critical |
| 1 | 256 | 100.00% | 0.000 | 4.064 | 4.064 | 9.802 | 0.285 | GPU-critical |
| 2 | 77 | 76.39% | 3.304 | 4.043 | 0.739 | 9.825 | 0.180 | GPU-critical |
| 3 | 48 | 69.44% | 4.435 | 4.552 | 0.117 | 8.821 | 0.188 | balanced |
| 4 | 57 | 83.93% | 3.681 | 3.709 | 0.029 | 9.554 | 0.201 | balanced |
| 5 | 256 | 100.00% | 0.000 | 5.365 | 5.365 | 11.100 | 0.162 | GPU-critical |
| 6 | 56 | 86.31% | 2.272 | 3.626 | 1.353 | 9.492 | 0.269 | GPU-critical |
| 7 | 28 | 73.81% | 3.523 | 3.673 | 0.150 | 7.779 | 0.228 | balanced |
| 8 | 18 | 69.44% | 4.488 | 4.515 | 0.027 | 10.370 | 0.077 | balanced |
| 9 | 30 | 81.35% | 4.123 | 4.567 | 0.444 | 10.617 | 0.129 | GPU-critical |
| 10 | 256 | 100.00% | 0.000 | 3.719 | 3.719 | 9.887 | 0.302 | GPU-critical |
| 11 | 51 | 80.95% | 3.141 | 3.632 | 0.491 | 8.010 | 0.287 | GPU-critical |
| 12 | 27 | 68.06% | 4.200 | 4.270 | 0.070 | 10.297 | 0.178 | balanced |
| 13 | 256 | 100.00% | 0.000 | 4.666 | 4.666 | 10.810 | 0.199 | GPU-critical |
| 14 | 24 | 75.40% | 3.946 | 4.107 | 0.161 | 10.192 | 0.167 | balanced |
| 15 | 256 | 100.00% | 0.000 | 3.774 | 3.774 | 8.136 | 0.347 | GPU-critical |
| 16 | 21 | 58.33% | 3.868 | 3.918 | 0.050 | 9.792 | 0.233 | balanced |
| 17 | 31 | 68.06% | 3.682 | 3.855 | 0.173 | 9.782 | 0.224 | balanced |
| 18 | 29 | 69.05% | 7.870 | 7.802 | 0.068 | 13.729 | 0.282 | balanced |
| 19 | 35 | 78.17% | 4.204 | 4.275 | 0.071 | 8.520 | 0.168 | balanced |
| 20 | 25 | 75.99% | 3.970 | 4.174 | 0.204 | 10.077 | 0.126 | balanced |
| 21 | 24 | 69.25% | 3.907 | 3.999 | 0.092 | 9.689 | 0.088 | balanced |
| 22 | 39 | 73.21% | 3.368 | 3.553 | 0.185 | 9.463 | 0.285 | balanced |
| 23 | 256 | 100.00% | 0.000 | 4.104 | 4.104 | 8.386 | 0.261 | GPU-critical |
| 24 | 28 | 67.06% | 4.006 | 4.102 | 0.096 | 9.997 | 0.203 | balanced |
| 25 | 22 | 64.09% | 4.194 | 4.289 | 0.095 | 10.237 | 0.139 | balanced |
| 26 | 18 | 57.54% | 4.185 | 4.113 | 0.072 | 10.316 | 0.167 | balanced |
| 27 | 42 | 81.75% | 2.272 | 2.850 | 0.578 | 7.191 | 0.369 | GPU-critical |
| 28 | 31 | 64.68% | 4.025 | 4.055 | 0.030 | 10.161 | 0.210 | balanced |
| 29 | 32 | 65.67% | 3.521 | 3.554 | 0.033 | 9.629 | 0.261 | balanced |
| 30 | 256 | 100.00% | 0.000 | 3.692 | 3.692 | 9.834 | 0.306 | GPU-critical |
| 31 | 30 | 67.66% | 3.334 | 3.435 | 0.101 | 7.859 | 0.333 | balanced |
| 32 | 38 | 68.06% | 3.513 | 3.550 | 0.036 | 9.647 | 0.278 | balanced |
| 33 | 256 | 100.00% | 0.000 | 4.490 | 4.490 | 10.541 | 0.241 | GPU-critical |
| 34 | 43 | 73.02% | 3.968 | 4.013 | 0.045 | 9.935 | 0.197 | balanced |
| 35 | 25 | 63.49% | 4.197 | 4.181 | 0.016 | 8.407 | 0.123 | balanced |
| 36 | 256 | 100.00% | 0.000 | 3.683 | 3.683 | 9.616 | 0.355 | GPU-critical |
| 37 | 256 | 100.00% | 0.000 | 2.935 | 2.935 | 9.106 | 0.354 | GPU-critical |
| 38 | 33 | 71.83% | 3.492 | 3.482 | 0.010 | 9.394 | 0.254 | balanced |
| 39 | 37 | 71.03% | 3.963 | 3.998 | 0.034 | 8.236 | 0.248 | balanced |

## Problem layers

- Layer 38: 33 slots, 71.83% GPU hit, 9.394 ms, balanced.
- Layer 39: 37 slots, 71.03% GPU hit, 8.236 ms, balanced.

## Top marginal GPU moves from current placement

- (18,143): +0.817 ms/layer; 20 invocations; CPU removed 0.817 ms, GPU added 0.448 ms.
- (18,161): +0.734 ms/layer; 19 invocations; CPU removed 0.734 ms, GPU added 0.426 ms.
- (13,28): +0.633 ms/layer; 25 invocations; CPU removed 0.633 ms, GPU added 0.231 ms.
- (18,226): +0.604 ms/layer; 15 invocations; CPU removed 0.604 ms, GPU added 0.336 ms.
- (39,70): +0.572 ms/layer; 29 invocations; CPU removed 0.572 ms, GPU added 0.324 ms.
- (39,78): +0.552 ms/layer; 25 invocations; CPU removed 0.552 ms, GPU added 0.279 ms.
- (11,111): +0.543 ms/layer; 20 invocations; CPU removed 0.543 ms, GPU added 0.178 ms.
- (18,72): +0.522 ms/layer; 13 invocations; CPU removed 0.522 ms, GPU added 0.291 ms.
- (37,21): +0.522 ms/layer; 24 invocations; CPU removed 0.522 ms, GPU added 0.140 ms.
- (14,106): +0.481 ms/layer; 19 invocations; CPU removed 0.481 ms, GPU added 0.205 ms.

The policy ranking is predictive only. Compare deltas with matched runtime A/B runs before changing placement or scheduling.

The generic CPU0/GPU0 labels are intentional: future NUMA and GPU islands can use the same schema once measured cost profiles exist.
