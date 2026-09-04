# Resident-memory pressure sweep

Result: **no crossover observed**; pressure axes are explicitly non-equivalent.

## Colibri

| Capacity | Slots | Routed hit | Run 1 tok/s | Run 2 tok/s | Median tok/s |
|---:|---:|---:|---:|---:|---:|
| 100.0% | 3642 | 64.4% | 4.396 | 1.824 | 3.110 |
| 75.0% | 2732 | 64.4% | 4.360 | 2.120 | 3.240 |
| 50.0% | 1821 | 63.0% | 3.004 | 1.638 | 2.321 |
| 37.5% | 1366 | 59.3% | 1.679 | 1.686 | 1.682 |
| 25.0% | 910 | 51.6% | 2.869 | 1.574 | 2.221 |
| 12.5% | 455 | 38.4% | 2.883 | 1.838 | 2.361 |

## llama.cpp

| GPU layers | Proxy fraction | Generation tok/s | GPU util | Power | Status |
|---:|---:|---:|---:|---:|---|
| 0 | 0.0% | 9.900 | 2.9% | 44.1 W | valid |
| 2 | 5.0% | 11.800 | 16.8% | 60.0 W | valid |
| 4 | 10.0% | 12.100 | 15.8% | 58.3 W | valid |
| 6 | 15.0% | 10.900 | 17.2% | 64.4 W | valid |
| 8 | 20.0% | 13.100 | 16.8% | 70.9 W | valid |
| 10 | 25.0% | 13.300 | 17.9% | 63.9 W | valid |
| 12 | 30.0% | 13.800 | 18.0% | 73.8 W | valid |
| 14 | 35.0% | 14.500 | 22.9% | 70.2 W | valid |
| 16 | 40.0% | n/a | 7.0% | 65.3 W | infeasible |
| 24 | 60.0% | n/a | 8.7% | 64.7 W | infeasible |
| 32 | 80.0% | n/a | 4.6% | 64.4 W | infeasible |
| auto | n/a | 23.100 | 27.7% | 75.7 W | valid |

## Interpretation

Colibri's measured maximum is 4.396 tok/s; llama.cpp's slowest valid generation point is 9.900 tok/s (CPU-only).
llama.cpp remains executable through 14 offloaded layers and fails at 16 because Vulkan cannot allocate the required device buffer.
The tested Qwen3.6/model-size regime therefore does not expose a Colibri crossover. The result does not answer the oversized-model regime because the two pressure controls are not physically equivalent.

Recommended next experiment: repeat this protocol on a model/working set that exceeds llama.cpp's feasible offload boundary while keeping Colibri's storage-backed expert pool operational.
