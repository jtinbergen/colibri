# llama.cpp comparison for Qwen3.6 on GTX 1080

Date: 2026-09-04

## Setup

* Model: `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
* llama.cpp build: local Vulkan build, `Vulkan0 = NVIDIA GeForce GTX 1080`.
* CUDA is disabled in this llama.cpp build; GPU results use the normal Vulkan
  backend.
* 10 CPU threads, context 16,384, batch 512, ubatch 512, seed 1,
  temperature 0, reasoning off.
* Same 500-token technical MoE prompt for all configurations.
* llama.cpp's internal warmup remained enabled.

The GGUF is Q4_K_M, while Colibri's current native model path uses its own
group-scaled representation. This is a useful engine/hardware comparison, but
not a bit-identical quantization comparison.

## Results

The llama.cpp timing line reports decode generation speed; wall time includes
model loading and warmup. GPU, CPU, VRAM and PCIe values are averages/maxima
from the separate 0.5-second sampler during each process.

| configuration | llama decode tok/s | process wall s | GPU util | GPU power | CPU util | max VRAM |
|---|---:|---:|---:|---:|---:|---:|
| CPU-only (`--device none`) | 10.8 | 77.93 | 4.52% | 43.5 W | 38.4% | 1,508 MiB |
| maximum normal offload (`Vulkan0`, `--gpu-layers auto`) | 22.0 | 46.90 | 32.86% | 80.4 W | 32.8% | 7,468 MiB |
| hybrid (`--cpu-moe`, Vulkan dense/offload) | 19.8 | 38.75 | 40.91% | 66.7 W | 43.0% | 3,945 MiB |

All observed runs reported PCIe Gen3 x16. The CPU-only VRAM number is the
desktop/driver baseline, not model residency.

Relative to CPU-only, maximum normal offload was approximately 2.04x faster;
the explicit CPU-MoE hybrid was approximately 1.83x faster. Maximum offload
was about 11% faster than the explicit CPU-MoE hybrid, but used substantially
more VRAM and power.

## Configurations

```text
cpu_only:
  --device none --gpu-layers 0

gpu_max:
  --device Vulkan0 --gpu-layers auto --fit on --fit-target 1024

hybrid_cpu_moe:
  --device Vulkan0 --gpu-layers auto --fit on --fit-target 1024 --cpu-moe
```

The first attempted `--gpu-layers all` configuration was rejected by Vulkan
with `ErrorOutOfDeviceMemory`; it is not included as a performance result.
`auto` is therefore the maximum sensible offload for this 8-GB card.

## Interpretation for Colibri

This is strong evidence that the Pascal is not inherently underpowered for
this model shape. llama.cpp's ordinary execution path keeps the GTX 1080 at
roughly one-third to two-fifths utilization and 67–80 W, while Colibri's
resident DP4A path has recently shown only roughly 5–12% utilization and
44–46 W. The comparison points at Colibri's execution envelope, CPU/GPU
work partition, and completion scheduling rather than DP4A arithmetic alone.

The explicit hybrid result is especially relevant: leaving MoE work on the
CPU while offloading dense work can be faster than starving a GPU resident
island, but it is still slightly slower here than llama.cpp's maximum normal
offload. This supports the existing design direction of treating CPU and GPU
as parallel compute islands, while showing that Colibri currently feeds its
GPU island less effectively.

The result should not be interpreted as proof that Colibri can simply copy
llama.cpp's tensor placement. The model quantization formats differ, and
llama.cpp is executing a conventional whole-model graph rather than Colibri's
fine-grained expert streaming/residency path.

Raw machine-readable results are in `llama_cpp_qwen_500d/all.summary.json` and
the per-run `*.metrics.csv` files next to this report.
