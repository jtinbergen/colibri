# Qwen3.6 execution-granularity diff: llama.cpp vs Colibri

Date: 2026-09-04  
Scope: source-level execution comparison and existing production measurements  
Status: discovery complete; no scheduler, placement, or arithmetic change made

## Executive result

The smallest useful unit to transplant from a llama-like execution model is not
an individual expert call and not another fused microkernel. It is one
`(layer, compute-island)` work batch:

```text
LayerWorkDescriptor
  layer, token/batch count
  input and output buffers
  selected expert ids and routing weights
  compact resident expert records for this island
  metadata/residency generation
```

The control plane may still split the eight routed experts between CPU and one
or more GPUs. Once split, each island should receive its complete local batch
and execute its internal gate/up -> activation -> down -> weighting sequence.
The host should observe only the mathematically required layer completion.

This is different from the current Colibri boundary:

```text
router -> expert lookup/classification -> GPU issue
       -> several CUDA launches and transfers
       -> GPU event / qt_take / reduction / D2H
       -> CPU fallback and shared expert merge
```

Colibri already groups resident experts once per device in `qt_issue`; it is
therefore not suffering primarily from eight independent GPU expert launches.
Its remaining weakness is that the grouped island call still exposes a
micro-operation envelope to the host and that the CPU path is still expressed
as per-expert work.

## Evidence and caveat

This document combines source inspection with the already collected aggregate
CUPTI trace and non-instrumented benchmarks. It is not a new CUPTI run. The
previous trace measured 200 decode tokens, 7,960 resident calls, 39,560 queued
kernels from resident calls, 184,514 memcpy records, and 536,613 CUDA API
records. Its instrumentation was known to perturb this workload, so those
counts are used structurally, not as production-time latency.

The llama comparison used a local Vulkan build and the same GTX 1080, but its
GGUF Q4_K_M representation is not bit-identical to Colibri's native
group-scaled representation. It is therefore an execution/hardware
comparison, not an accuracy comparison.

## 1. Control-flow comparison

### llama.cpp Qwen3.5/3.6 MoE layer

Relevant sources:

* `C:\Users\jaapj\src\llama.cpp-vulkan\src\models\qwen35moe.cpp:497`
  builds the Qwen MoE layer and passes the merged `ffn_gate_up_exps` tensor.
* `C:\Users\jaapj\src\llama.cpp-vulkan\src\llama-graph.cpp:1917`
  builds routing, expert selection, indexed matmuls, weighting, and reduction.
* `C:\Users\jaapj\src\llama.cpp-vulkan\src\llama-graph.cpp:2095` emits
  one `MUL_MAT_ID` for merged gate/up; `:2231` emits one indexed down matmul.
* `C:\Users\jaapj\src\llama.cpp-vulkan\src\llama-context.cpp:2494`
  submits the complete ggml graph to the backend scheduler asynchronously.

For the decode case (`T=1`, `K=8`), the logical MoE DAG is:

```text
cur [D,T]
  |
  +-- router matmul -> probabilities -> top-k ids [K,T]
  |                                  -> weights [1,K,T]
  |
  +-- reshape cur [D,1,T]
       |
       +-- one indexed merged gate/up matmul
       |      gate_up [2I,K,T]
       |        |-- gate view [I,K,T]
       |        `-- up view   [I,K,T]
       |
       +-- SwiGLU/SILU activation [I,K,T]
       |
       +-- one indexed down matmul [D,K,T]
       |
       +-- routing-weight multiply
       |
       +-- expert-output aggregation
       |
       `-- add shared expert result
```

`ggml_mul_mat_id()` carries the expert-id tensor into the backend. The Vulkan
backend dispatches `GGML_OP_MUL_MAT_ID` through a single indexed-matmul
operation (`ggml_vk_mul_mat_id`), with the selected ids and expert count in
device-visible buffers. It does not call the host once for each selected
expert. The Vulkan backend may additionally fuse compatible elementwise nodes;
that is an execution detail and does not change the logical DAG.

The graph is not necessarily one hardware dispatch: Vulkan may record and
submit several command-buffer batches based on node count/flop thresholds.
The important property is that the host submits a graph and the backend owns
the internal node sequencing. The CPU backend has the same `MUL_MAT_ID`
operation and uses its threadpool for the indexed matmul (`ggml-cpu.c:1534`),
so the CPU-MoE configuration remains a backend execution choice rather than a
host loop in the Qwen model code.

### Colibri Qwen3.6 layer

Relevant sources:

* `D:\src\colibri\c\qwen36.c:2333` performs router matmul, softmax, host
  top-k selection, and CPU/GPU split.
* `D:\src\colibri\c\qwen36.c:2417` calls `qt_issue()` after expert lookup.
* `D:\src\colibri\c\qwen36.c:2420` onward computes every nonresident expert
  with `slot_ensure_int8`, three `matmul_qe` operations, a SILU epilogue, and
  weighted accumulation on the CPU.
* `D:\src\colibri\c\qwen36.c:2492` calls `qt_take()` at the layer's existing
  consumption point.
* `D:\src\colibri\c\qwen36_tier.c:480` classifies routed experts by resident
  island and calls one resident issue per active device.
* `D:\src\colibri\c\backend_cuda.cu:3526` is the resident GPU entry point.

Current resident GPU DAG for one active device is:

```text
host/router input x
  |
  +-- P2P x: home -> island
  +-- optional compact metadata H2D refresh
  +-- routing weights H2D
  |
  +-- quantize activation [D/64 groups]
  +-- gate/up DP4A for all resident experts [O,I,count grid]
  +-- quantize gate output/down input [count,I/64 groups]
  +-- down DP4A for all resident experts [O,D,count grid]
  +-- weighted_sum_rows -> one island partial [D]
  |
  +-- partial P2P: island -> home
  `-- completion event

CPU misses run in parallel, but are expressed as a loop over individual
experts. At qt_take:

  completion events -> legacy-stream wait -> sum_slots -> sync -> D2H
```

The current DP4A path therefore has four DP4A-related launches plus the
weighted aggregation launch per resident issue; `sum_slots` is a sixth kernel
in the issue-to-take window. The host has already removed the six old pointer
table copies in the persistent-metadata cleanup, but the execution boundary
still exposes the sequence above.

## 2. Shape and layout diff

| Concern | llama.cpp | Colibri today | Architectural consequence |
|---|---|---|---|
| Router/top-k | ggml graph tensors; backend-visible ids and weights | host C arrays `idx[K]`, `val[K]` | descriptor must carry compact ids/weights, not re-run routing |
| Expert grouping | one `MUL_MAT_ID` sees the selected expert set | one `resident_issue` per island sees only that island's resident subset | the island descriptor should represent a subset of the routed set |
| Gate/up weights | one stacked expert tensor with indexed rows | per-expert tensor pointers/metadata | descriptor needs stable island-local expert records |
| Gate/up result | `[2I,K,T]`, split by views | `[count,I]` gate output | either representation can remain; no arithmetic rewrite is required |
| Down input | activation/SwiGLU tensor consumed by indexed down matmul | separately quantized `count x I` INT8 plus scales | preserve this boundary initially; executor owns it internally |
| Down result | `[D,K,T]` followed by graph weighting/reduction | `[count,D]`, weighted to `[D]`, then cross-island `sum_slots` | island-local partial should be one explicit output |
| CPU execution | CPU `MUL_MAT_ID` uses ggml threadpool | `slot_ensure_int8` plus per-expert `matmul_qe` loop | create an analogous CPU batch executor later |
| Host/device lifetime | graph allocator and backend buffers | Colibri buffers, events, residency generation | descriptor must include generation and buffer ownership/lifetime |
| Completion | graph scheduler/backend synchronization | explicit `qt_take` event/reduction/sync | retain one layer boundary, move internals below it |

For Colibri's current decode dimensions, a useful descriptor is therefore
closer to `[resident_count, D]`/`[resident_count, I]` island-local storage
than to llama's full `[K,T]` indexed graph tensor. It must not assume all eight
experts are on the same island.

## 3. What is necessary versus overhead

### Mathematically necessary

* router computation and exact top-k/weight semantics;
* gate and up projection for each selected expert;
* SILU/gating and down projection;
* routing-weight application and summation in the established order;
* CPU fallback work for experts not assigned to the GPU;
* the GDN/layer dependency before the next layer consumes the result.

### Required by the current representation

* input replication into a resident-expert row layout;
* per-group activation quantization for DP4A;
* separate gate-output/down-input quantization because the validated kernels
  consume INT8 plus group scales;
* P2P movement between the home device and an island;
* a partial buffer for each active GPU island;
* a final `sum_slots` because partials are produced independently.

### Host/API envelope or implementation overhead

* deciding and preparing the same GPU sequence at every resident issue;
* dynamic metadata refreshes when the residency generation is unchanged;
* per-call routing-weight upload rather than a device-visible descriptor;
* explicit event and stream bookkeeping for every island call;
* the legacy-stream synchronization before host D2H;
* per-expert CPU lookup/rematerialization setup;
* device-property queries (already removed from the hot path);
* diagnostic event/CUPTI calls (must not be optimized as production work).

The prior persistent-metadata result is direct evidence that this category is
worth attacking: the matched 500-token cleanup moved 2.015 to 2.318 tok/s in
one run, with a pooled indication around +9.9%, while preserving output.

## 4. Why llama.cpp feeds Pascal better

The local 500-token comparison on the same GTX 1080 reported:

| Configuration | Decode tok/s | GPU utilization | GPU power | CPU utilization | Max VRAM |
|---|---:|---:|---:|---:|---:|
| llama CPU-only | 10.8 | 4.52% | 43.5 W | 38.4% | 1,508 MiB |
| llama maximum normal offload | 22.0 | 32.86% | 80.4 W | 32.8% | 7,468 MiB |
| llama explicit CPU-MoE hybrid | 19.8 | 40.91% | 66.7 W | 43.0% | 3,945 MiB |

The relevant conclusion is not that Colibri should copy llama.cpp's tensor
placement. It is that the conventional graph lets the backend see enough
work at once. Colibri's recent resident DP4A runs used roughly 5--12% GPU
utilization and 44--46 W despite fast arithmetic. The missing unit is a local
executor that receives an already-decided island batch and owns its internal
work window.

## 5. Evaluation of execution strategies

| Strategy | Semantics | Dynamic routing | Current evidence | Future islands | Decision |
|---|---|---|---|---|---|
| CUDA Graph replay of current five-kernel envelope | preserves kernels/order | requires variants and mutable pointers/weights | existing opt-in prototype was exact but -33.6% in matched 500-token A/B | possible, but each island needs its own graph/lifetime | keep disabled; not first transplant |
| coarse host dispatch around existing sequence | preserves kernels/order | high; descriptor fields remain dynamic | no prototype yet | good if descriptor is island-neutral | selected first |
| persistent device executor | potentially lowest host work | high in theory | not measured; high SM61 complexity/risk | conceptually good, implementation-heavy | defer |
| another local kernel fusion | risks reduction/order and geometry | high | rejected fusion was -16% and divergent | weak abstraction for multiple islands | explicitly defer |

The selected prototype is a thin layer-executor interface around the existing
validated operations, not a new scheduler:

```text
current control plane
  route -> lookup -> classify resident/fallback -> build descriptor

GPU island executor
  descriptor -> stage input/metadata -> existing DP4A sequence
           -> island partial -> one completion token

CPU island executor (follow-up)
  descriptor -> packed fallback batch -> existing numerical operations

layer merge
  GPU partials + CPU result + shared expert -> next GDN boundary
```

The first implementation can call the existing DP4A group path through this
interface. That is intentionally useful: it makes ownership and lifetime
explicit before changing launch topology. A second implementation can then
replace only the internals of the GPU executor with a coarse command/descriptor
submission mechanism while keeping the control plane and arithmetic kernels
fixed.

## 6. Proposed descriptor contract

The contract should be opaque to routing and storage code and explicit about
ownership:

```text
LayerWorkDescriptor {
    uint32 layer;
    uint32 token_count;              // decode is 1 initially
    uint32 resident_count;
    uint32 metadata_generation;
    IslandId island;

    const float * input;             // island-visible or source device buffer
    float * partial_output;          // exactly one [D] island result
    const uint32 * expert_id;        // compact subset, stable order
    const float * route_weight;      // same order as expert_id
    const ExpertRecord * experts;    // gate/up/down + scale metadata

    uint32 hidden_dim;
    uint32 intermediate_dim;
    uint32 group_size;
    uint32 flags;                    // decode/prefill, DP4A eligibility, etc.
}
```

Rules for the prototype:

1. The control plane owns `expert_id` ordering and routing weights.
2. The executor owns all temporary buffers and internal launch ordering.
3. The executor returns only an island partial plus a completion token.
4. No pointer survives a residency-generation change unless its generation is
   still valid.
5. The executor must use the existing DP4A kernels and fixed-order reductions.
6. CPU fallback assignment and the layer merge remain unchanged.

`ExpertRecord` is deliberately a logical contract at this stage; it does not
require moving the current tensor representation or introducing a broad
abstraction layer.

## 7. Smallest next implementation and test

The next code change should be a Qwen-only adapter with three phases:

1. Build the descriptor in `qt_issue` from the already computed
   `G.is_k[]`, `G.is_tg[]`, `G.is_tu[]`, `G.is_td[]`, and `wbuf[]` arrays.
2. Route the descriptor to a new internal GPU-island executor function that
   initially delegates to the current `coli_cuda_expert_group_resident_issue`.
3. Add structural counters for descriptors, resident experts, completion
   tokens, and CPU-fallback experts; verify the baseline output and counts are
   unchanged.

Only after this adapter is exact should the executor internals be changed to
submit a larger island-local command sequence. This gives a clean A/B boundary
without changing placement, scheduling, kernels, or numerical ordering.

## Current architectural verdict

```text
llama-like coarse execution:      PASS as the target abstraction
Colibri fine-grained placement:   retain
individual host expert execution: replace inside each island
CUDA Graph replay prototype:      structural PASS, performance FAIL
local fusion as main strategy:    reject for now

Highest-value first intervention:
  introduce a per-layer/per-island LayerWorkDescriptor and executor boundary,
  initially as an exact adapter around the validated resident DP4A path.

Expected immediate throughput effect:
  approximately zero; this first change is an isolation/ownership step.

Expected later payoff:
  one place to batch or asynchronously submit the full island sequence, and
  the same contract can feed CPU, GPU0, GPU1, and future NUMA-local islands.
```

