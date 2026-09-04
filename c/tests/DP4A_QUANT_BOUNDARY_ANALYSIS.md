# DP4A resident-path quantization boundary analysis

Status: **FAIL for this intervention; production baseline restored**.

This report records why the separate gate-output/down-input quantization launch
cannot safely be removed by a small producer-side change while preserving the
current quantization semantics and DP4A kernels.

## Current operation

For the resident DP4A path in `backend_cuda.cu`, the relevant sequence is:

```text
ctx->x (FP32, D values)
  -> quantize_row_g_s(gs=64)
ctx->qx / ctx->qscale (INT8 + per-group FP32 scales)
  -> gate_up_meta_s / gate_up_all_s
ctx->gate (FP32, count * I values; silu(gate) * up)
  -> quantize_rows_g_s(gs=64)
ctx->qx / ctx->qscale (INT8 + per-group FP32 scales)
  -> down_meta_s / down_all_s
ctx->y (FP32, count * D values)
```

The gate/up kernel computes the exact FP32 value

```text
gate[e,o] = silu(dot(gate_weight[e,o], x)) * dot(up_weight[e,o], x)
```

and writes one value per `(expert, o)`. The down kernel consumes an INT8 row
of those values plus one scale for every group of 64 input values.

The quantizer uses, for each row/group:

```text
s = amax(abs(gate[e, group*64 : group*64+64])) / 127, or 1 when amax == 0
q = rint(clamp(gate / s, -127, 127))
```

The FP32 gate buffer has no other consumer on this resident path. Its only
purpose between the two DP4A projections is to provide the complete row to
the groupwise quantizer.

## Why producer-side fusion is not local

`dp4a_gate_up_all_kernel` uses `grid=(O,count)` and one 32-thread block for
each scalar output `gate[e,o]`. The group scale for down input is a reduction
over 64 consecutive `o` values. Therefore the block producing `o` does not
have the other 63 values needed to compute the exact scale or quantized bytes.

For the Qwen shape (`O=I=512` in gate/up, `gs=64`), this is eight independent
cross-block reductions per expert. CUDA block synchronization cannot provide
this reduction, and a grid-wide barrier would be invalid for the current grid:
the complete grid cannot be resident simultaneously on SM61. Atomics would
also require a second phase and careful reset/publication ordering, so they do
not remove the execution boundary.

Changing gate/up to produce 64 output values per block could make the reduction
local, but that changes the current DP4A arithmetic geometry and likely requires
large blocks or a different tiling. It is a separate kernel redesign, not a
safe metadata/launch cleanup.

## Candidate classification

| Candidate | Exact semantics | Launch reduction | Decision |
|---|---:|---:|---|
| Extend current scalar gate/up blocks | No: each block lacks 63 values | N/A | Reject |
| Grid-wide barrier/atomics | Possible only with a second phase or restrictive residency | No meaningful reduction | Reject on SM61/current grid |
| Quantize inside down kernel | Yes, if repeated per down tile | 1 launch removed | Reject: repeats input quantization for every down output tile |
| Tiled gate/up producing 64-value groups | No in tested form | 1 launch removed | Rejected: numerical divergence and regression |
| Keep the existing quantizer | Yes | 0 | Current safe path |

For the current down kernel, one block covers four output rows. If it performs
the complete gate-input quantization itself, the same `count * I` activation
values are revisited once per down output tile. With `D=2048`, that is
`D/4 = 512` repetitions per expert, versus one quantization pass today. This
would replace one short launch with roughly 512x repeated amax/scale/INT8
work and is not a valid optimization.

## Current topology and unavoidable dependency

The applicable resident topology remains:

```text
L1  quantize input                 (one launch)
L2  gate+up DP4A + SiLU            (one launch)
L3  quantize gate/down input       (one launch)
L4  down DP4A                      (one launch)
```

The L3 boundary is mathematically required by the current producer geometry,
not by pointer metadata or host API design. The metadata cleanup is independent
and remains valid: it removes six pointer-table H2D updates without changing
this quantization dependency.

## Result

An isolated 512-thread, 64-output-tile producer was implemented behind
`COLI_CUDA_DP4A_FUSED_Q=1` and tested, then removed after the gate/up reduction
order changed generated output and the geometry regressed. The production
four-launch path is restored.

The candidate did build for `sm_61`, passed the existing low-level tests and a
resident smoke test, and produced identical content in a short 32-token run.
The decisive 500-token run did not: baseline was 5.6375 tok/s versus 4.7350
tok/s fused (-16.0%), and the generated message content differed. Therefore
the candidate fails the correctness/performance gate despite removing one
launch.

The matched run's lightweight system observations were also consistent with
the regression rather than a GPU fault:

| metric | baseline | fused candidate |
|---|---:|---:|
| GPU utilization sample mean | 12.16% | 11.66% |
| GPU memory utilization | 2.84% | 3.23% |
| VRAM used | 7569 MiB | 7634 MiB |
| GPU power | 46.27 W | 45.59 W |
| qwen36 CPU sample mean | 23.25% | 21.50% |
| qwen36 working set | 21.84 GB | 22.11 GB |

PCIe remained Gen3 x16. The lower fused GPU power/utilization and higher
working set are compatible with the 512-thread producer geometry being a poor
fit for this workload; they do not justify enabling it.

The current four-launch path is the accepted implementation. A future launch
removal must preserve the original FP32 reduction semantics, likely requiring
a substantially different cooperative/two-stage design; it should not be
attempted as a local scalar-to-tile rewrite.
