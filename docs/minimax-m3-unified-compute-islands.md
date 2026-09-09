# MiniMax-M3 unified local compute-island DAG

Status: design/status note, 2026-09-09

## Purpose

The project must use a single dependency-aware DAG to schedule local CPU,
GPU, RAM, and storage resources as compute islands. The goal is not merely to
add more GPU devices or to distribute a global expert prefix. The goal is to
place and execute routed experts where their weights and compute are useful,
while accounting for movement, queueing, bandwidth, and tail latency.

The target execution shape is:

```text
router
  -> routed expert tasks
  -> island/resource assignment
  -> per-island I/O and compute queues
  -> dependency-aware execution
  -> deterministic weighted reduction
```

This note concerns multiple islands in one machine. The existing TCP expert
worker/cluster path is a separate feature and is deliberately out of scope.

Compute islands are intentionally heterogeneous. An island may be an AMD or
Intel CPU/NUMA domain, an NVIDIA CUDA device, an Intel/AMD GPU backend, a
different accelerator backend, or a storage-backed CPU resource. The planner
must use discovered capabilities, measured links, supported kernels, memory,
queue depth, and tail behavior. It must not assume that islands share an
instruction set, backend, memory model, precision support, or transfer path.

## Verified current state

### Already present

- The M3 code has a task contract with weight and compute states, route
  commitment, readiness, leases, cancellation, failure, output publication,
  and fixed reduction hooks (`c/m3_dag.h`).
- M3 has serial, PIPE ready-first, and bounded parallel CPU DAG paths in
  `c/colibri.c`.
- The CUDA backend accepts a list of local devices through `COLI_GPUS` and
  contains per-device expert-group dispatch and optional peer/device-resident
  paths.
- Expert pinning can upload a usage-ranked prefix across local CUDA devices and
  reports per-device resident counts and bytes.
- Runtime profiling already reports expert I/O, wait, CPU expert time, GPU
  critical time, device placement, queue depth, overlap, cache hits, and tails.

### Not yet integrated

The M3 DAG eligibility functions explicitly reject the CUDA, Metal, Vulkan, and
cluster paths. For example, the CUDA-enabled guard is in
`c/colibri.c:6340` and the parallel guard is in `c/colibri.c:6359`.

The local multi-GPU executor therefore runs through a separate conventional
MoE dispatch path (`c/colibri.c:7073` and `c/colibri.c:7751`). It is not yet a
resource-aware M3 DAG in which CPU and GPU islands participate in one task
graph.

Current GPU expert placement is still a global popularity-ranked prefix. It is
not yet layer-affine, coverage-aware, or selected by a unified DAG resource
planner (`c/colibri.c:11549`).

The current 80%-coverage experiments are therefore placement experiments on a
single GPU island. They do not yet prove unified multi-island DAG execution.

The completed single-GPU 80%-coverage run used 142 experts from contiguous
layers 17--22 (4.52 GB in VRAM). It measured 0.47 tok/s median on the rotating
workload, 46.8% median cache hit, and 70% expert-I/O time. This is evidence that
the candidate loaded and executed as a GPU-resident placement, not evidence
that the unified local DAG is already scheduling multiple islands.

## Required unified model

Every routed task must carry or resolve the following information before
execution:

- model layer and expert identity;
- route row(s) and mixture weight(s);
- weight owner and residency state;
- assigned compute island;
- input owner and required transfers;
- I/O queue and compute queue;
- capacity/bandwidth evidence and epistemic state;
- cancellation, failure, and fallback policy;
- timestamps for ready, submitted, started, output-ready, and committed.

An island may be:

- a CPU/NUMA resource set;
- a CUDA device;
- a group of CUDA devices with an explicit home device;
- a RAM-resident weight tier;
- a storage/controller path used for loading and prefetch.

The planner must also represent island compatibility and conversion costs:

- CPU ISA and kernel family (for example AVX2, AVX-512, VNNI, or scalar);
- accelerator backend and supported quantization formats;
- device-local memory versus host memory;
- peer-to-peer support and direction-specific transfer limits;
- NUMA ownership and remote-memory cost;
- whether an island can execute a task or only provide storage/staging.

RAM and storage are not compute islands by themselves. They are resource
owners and transfer paths attached to compute islands.

## Invariants before implementation

1. **One owner per task.** A routed expert task has one active compute owner at
   a time. Replication or speculative duplicate work must be explicit.
2. **No use before residency.** A device cannot start an expert task until its
   required weight handle is resident and its input is ready on that device.
3. **Bounded queues.** Per-island I/O, transfer, and compute queues are bounded;
   admission cannot create unbounded prefetch or staging memory.
4. **No hidden transfers.** Host/device and device/device transfers are recorded
   as DAG edges or explicit transfer tasks. They may not be hidden inside a
   generic matmul call when they affect scheduling or tails.
5. **Deterministic reduction.** Expert outputs are reduced in a defined order
   with the same numeric contract as the reference path.
6. **Failure is complete.** A failed island task either retries under an
   explicit policy or falls back to a known-good owner; partial output is never
   committed.
7. **Evidence is scoped.** Measured capacity is bound to device, direction,
   request shape, concurrency, residency, and tail statistics. Unknown or
   unavailable capacity never creates optimistic admission.
8. **NUMA locality is explicit.** CPU staging and fallback work must identify
   their NUMA owner. Remote memory is not treated as local memory.
9. **No scalability claim from one microbenchmark.** Promotion requires
   matched end-to-end DAG runs and tail evidence.

## Required implementation stages

### Stage A: CPU + one CUDA island in one M3 DAG

Keep the current machine as the proof platform. Represent the CPU fallback and
the GTX 1080 as two islands. A routed task may execute on the GPU only when its
expert is resident there; other tasks remain CPU/DAG tasks.

The first gate is not a large tokens/s gain. It is proof that:

- M3 DAG task states remain active with CUDA enabled;
- GPU and CPU tasks execute from one graph;
- GPU hits, CPU fallbacks, transfers, and reductions are counted separately;
- output matches the reference path;
- the graph does not deadlock, leak leases, or double-reduce outputs.

### Stage B: Layer-affine placement

Add a placement policy that can select complete or coverage-based contiguous
layer blocks. The 80%-coverage candidate is an input to the policy, not an
automatic admission decision.

The policy must compare the cost of:

- keeping a frequent expert on a GPU island;
- keeping it in RAM;
- loading it from storage;
- moving its activation to another island;
- leaving the task on the CPU.

### Stage C: Multiple local GPU islands

Use `COLI_GPUS` and the existing per-device CUDA dispatch only through the
unified DAG. Add per-device task queues, residency maps, transfer edges, and
per-device telemetry. Existing round-robin or global-prefix behavior may remain
as a baseline, but must not be called the final island planner.

### Stage D: Storage/controller parallelism

Only after the task graph is correct, add multiple storage paths. Expert loads
must be assigned to actual drives/controllers and measured with per-path queue
depth, service time, wall time, and p95/p99 tails. Additional drives are not
assumed useful unless the graph issues independent work to them.

## Acceptance gates

The unified local-island feature is not ready for hardware ROI claims until:

1. CPU-only and unified CPU+GPU outputs pass the existing correctness oracle;
2. telemetry proves nonzero routed GPU execution and separate CPU fallback;
3. all tasks have visible owner/residency/transfer states;
4. repeated runs show no lease, cancellation, or reduction errors;
5. matched baseline versus unified-DAG runs report tokens/s, request p95/p99,
   expert bytes, I/O wait, queue depth, transfer time, GPU idle time, and
   cache-hit rate;
6. a full-layer and an 80%-coverage placement are compared under the same
   workload;
7. promotion remains disabled unless the evidence passes the scoped,
   tail-safe candidate rules.

## Hardware implication

Do not select a six-GPU or multi-socket target solely from aggregate VRAM or
core count before Stage A succeeds. The current desktop is sufficient to prove
the CPU-island plus one-GPU-island contract. A future many-GPU host should
then be chosen for PCIe topology, memory channels, RAM capacity, and storage
parallelism—not just nominal accelerator count.

The likely useful role for a future RD550 is a separate CPU/RAM/storage island,
but that is a later extension. It must not be confused with the local unified
island path described here.
