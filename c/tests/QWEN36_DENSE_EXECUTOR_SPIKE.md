# Qwen3.6 coarse dense-island spike

This spike adds an execution-plane boundary below the existing Colibri
control plane. Routing, expert placement, LFRU, CPU fallback, and the DP4A
resident path are unchanged. The calls are opt-in through `COLI_DENSE_GPU=1`.

The projection mode batches DeltaNet qkv/z on one CUDA submission. The full
mode (`COLI_DENSE_FULL=1`) hands a complete DeltaNet layer to one CUDA call,
including persistent device state and the convolution ring. The LM-head and
attention projection experiments use the same dense-island boundary.

`COLI_CUDA_DENSE_GB` reserves VRAM for persistent dense weights before expert
warmstart. It must be identical between A/B runs when comparing placement.

## Numerical result

The initial GPU reduction tree changed the 32-token output and was removed.
The current dense int8 projection kernel mirrors the four FMA streams used by
Qwen's AVX2 `matmul_q`; all 30 DeltaNet qkv/z checks in the direct probe were
zero at the checked calls.

The full DeltaNet path is not yet an accuracy candidate. A layer dump of the
ordinary CPU path versus the full device path showed a difference already in
layer 0, with maximum about 0.14 in the first 64-token diagnostic run. The
difference grows through the recurrent stack. Keeping the state and norm
kernels scalar does not remove it, so the full path is not simply a state
parallelization issue; the f32 b/a and surrounding device reductions do not
share the CPU `matmul()` instruction-level reduction contract.

The 2 GB dense-reserve experiment separated capacity from short-run behavior.
LM head plus all ten attention layers was exact for 64 tokens and measured
5.373 tok/s versus 4.556 tok/s for its matched baseline. At 200 tokens it
measured 5.501 versus 4.536 tok/s, but the output hashes differed. That is a
performance signal, not an accuracy PASS.

The LM-head-only path remains the strongest *short-run* result: the earlier
200-token resident A/B was bit-identical and measured 5.396 tok/s versus
4.632 tok/s. The newer 500-token matched pair measured 4.404 versus 4.606
tok/s and produced a different output hash. It is therefore still useful as
an execution experiment, but not a production accuracy/performance PASS.

## Performance observations

Representative direct-engine runs, no CUPTI/timing dump, one GTX 1080:

| configuration | decode tok/s | GPU avg | power avg | result |
|---|---:|---:|---:|---|
| persistent-metadata DP4A baseline | 4.632 | 8.72% | 20.0 W | reference |
| LM head only, 1 GB reserve, 200 | 5.396 | 9.23% | 46.9 W | exact at 200 |
| LM head only, 1 GB reserve, 500 | 4.404 | 24.17% | 32.6 W | hash differs |
| LM + all attention, 2 GB reserve, 200 | 5.501 | 10.36% | 43.2 W | hash differs |
| full DeltaNet, no resident, scalar state | 4.367 | 17.58% | 23.9 W | not accepted |
| full DeltaNet, parallel state experiment | faster short run | higher | higher | hash differs |
| qkv/z/out dense projections, 1 GB reserve, 500 | 5.142 | 18.19% | 33.3 W | hash differs; not accepted |

The full-layer executor proves that the coarse island boundary can be reached,
but simply moving more operations to Pascal does not produce the desired
throughput. It also shows that the numerical contract must be specified before
an execution backend is allowed to replace the CPU path.

The projection-only executor is a useful load probe: it raises Pascal from
roughly 10% to 18% utilization and from about 18 W to 33 W, but its 500-token
hash differs and its speed is below the exact packed-CPU baseline. It is not a
production candidate until the dense projection path reproduces the existing
CPU numerical contract.

An opt-in transient-GPU-miss experiment was also tested. It reserved eight
execution-only expert slots per device and streamed nonresident routed experts
into the existing DP4A group. Consolidating the three weight refreshes into one
stream-ordered expert update improved the 32-token probe from 2.22 to 4.01
tok/s, but the exact packed-CPU baseline was 5.37 tok/s. The experiment did
raise Pascal activity to about 25%, yet PCIe/refresh work remained on the
critical path. It is therefore retained only as a diagnostic branch, disabled
by default with `COLI_CUDA_STAGE_MISSES` unset.

## Architecture decision

The smallest useful transplantable unit remains `(layer, compute-island)`, not
an individual projection. The descriptor adapter and GPU executor boundary are
implemented; initially the executor delegates to the existing validated
resident group path. This is intentionally a neutral ownership step for later
CPU, GPU0, GPU1, and NUMA-local executors.

CUDA Graph replay of the existing resident envelope was exact but was a matched
performance failure, so it remains disabled. Another local fusion is deferred:
the rejected gate/down quantization fusion regressed 16% and changed output
because of FP32 reduction ordering.

The next viable implementation must preserve the CPU reduction contract for
the full dense layer, or define an explicit numerical tolerance and validate it
with layer dumps/perplexity. It should keep the control-plane descriptor stable
while replacing only the inefficient internal executor.

## Current status

```text
coarse layer/island abstraction: PASS
exact full DeltaNet replacement: FAIL
safe LM-head acceleration: PASS (opt-in)
safe all-attention replacement: not established
end-to-end 10 tok/s goal: not reached
CPU fallback batch default: yes (opt-out: COLI_CPU_EXPERT_BATCH=0)
```

The validated default remains persistent-metadata DP4A resident execution.
The decode-time CPU fallback now uses the packed CPU-island executor by default;
`COLI_CPU_EXPERT_BATCH=0` restores the scalar A/B path.

## CPU island batch follow-up

The first safe coarse execution change was applied to the CPU fallback island,
not to the GPU kernels.  All misses for one decode layer are now gathered,
their gate/up/down work is executed through packed batch buffers, and route
order is retained for the final weighted accumulation.  The existing scalar
path remains available with `COLI_CPU_EXPERT_BATCH=0`.

The first implementation exposed a real lifetime hazard: the eight-entry
per-layer int8 scratch LRU could evict a pointer that a batch had already
collected.  That was fixed with temporary scratch leases.  If the configured
subpool cannot hold a complete batch, execution falls back to the established
scalar path.  This preserves the LRU/residency policy and makes replacement
safe.

Matched non-instrumented 500-token runs, same model/prompt/residency and
`COLI_CUDA_DP4A=1`:

| path | engine tok/s | request tok/s | output | GPU avg | power avg |
|---|---:|---:|---|---:|---:|
| scalar CPU fallback (`COLI_CPU_EXPERT_BATCH=0`) | 4.630 | 4.550 | exact | 9.82% | 18.1 W |
| packed CPU island (default) | 5.321 | 5.215 | exact | 10.14% | 18.4 W |
| packed CPU island repeat | 5.320 | 5.212 | exact | 10.77% | 18.3 W |

This is approximately a reproducible 15% end-to-end improvement, but it does
not materially raise Pascal utilization.  It confirms that coarse execution
also helps the CPU island, while the remaining path to the 10 tok/s target is
still the GPU/dense execution plane.

The decode thread-pool follow-up found that the process-wide ten-thread pool
is also too wide for this single-row workload.  With the same packed CPU
fallback, placement, model, prompt, and 500-token generation, applying
`COLI_QWEN_DECODE_THREADS=4` measured 6.103 engine tok/s (5.975 request
tok/s), versus the previous 5.32 tok/s packed-CPU reference family.  CPU
utilization was 216% and the output hash was
`d1f0e02ac03e96d04d57cb91ff819abc155e30a62f9c9a4c27a6123fc032f0aa`, equal
to the matched baseline hash.  The setting is currently opt-in pending a
broader thread-count A/B; it is a CPU-island scheduling knob, not a GPU or
numerical change.

## Execution-cache and capacity follow-up

The opt-in transient miss staging was changed into a bounded LRU execution
cache keyed by `(layer, expert)`.  This preserves the normal residency table:
staged entries are execution buffers only and are not admitted to LFRU.  The
64-token run with 32 slots/device was correct, but produced 18,231 new uploads
and only 2,100 cache hits after warmup.  It reached 28.4% GPU utilization but
only 3.825 tok/s.  The cache is therefore an explicit negative result: a small
cross-layer cache cannot replace resident placement when each token routes
hundreds of nonresident experts, and PCIe refresh remains on the critical path.

The batch-refresh implementation was then corrected and rechecked on the same
32-token probe.  It successfully refreshed 2,785 new execution slots, served
1,050 cache hits, and reported zero update failures; the output hash was
exactly the established baseline hash.  The run measured only 3.134 engine
tok/s versus 5.391 tok/s for the matched cache-disabled run, with 26.0% GPU
utilization versus 7.2%.  This is a useful negative result: the extra GPU
activity is paid for by upload and completion-path work, not by useful resident
reuse.  The staging branch remains opt-in and is not a candidate for the
production execution plane.

The benchmark also exposed an avoidable comparison reserve.  With dense
experiments disabled, leaving the default 1 GB dense reserve reduced the expert
budget to about 5 GB.  Removing that reserve loaded 3,642 experts instead of
about 3,037 and measured 5.415 tok/s at 35.5% hitmix.  A cautious explicit
6.5-GB expert budget loaded 3,935 experts, measured 5.479 tok/s at 38.3%
hitmix, and used 7.51 GiB peak VRAM.  Pascal utilization remained only 9.75%
and power 24.96 W, so capacity alone is not the route to 10 tok/s.  These
measurements include GPU utilization/power/VRAM, process CPU/RSS/system RAM,
and the machine's Gen3 x16 PCIe link.  Raw values are in
`qwen36_execution_cache_results.json`.

The larger-capacity hashes are recorded but are not claimed bit-identical to
the smaller-capacity configuration: changing CPU/GPU expert ownership can
change floating-point accumulation paths and therefore sampled output.  A
same-capacity deterministic reference is required before promoting this to an
accuracy candidate.

## Placement-informed coarse-island follow-up

The runtime now accepts `QTIER_PLACEMENT=layer_balanced`.  With a QTH1 heat
file it interleaves the heat-ranked `(layer, expert)` lists instead of
consuming the warmstart order globally.  `QTIER_HEAT_READONLY=1` prevents a
benchmark from overwriting the input heat file at shutdown, which makes
matched placement A/B runs reproducible.  The policy is still a control-plane
placement choice; it does not change the GPU kernels or CPU fallback logic.

Using the same 3,037-slot routing-informed balanced heat file, a 500-token
resident DP4A run reached 7.103 engine tok/s (6.983 request tok/s), with
25.19% sampled GPU utilization, 21.35 W average power, 5,344 MiB average
VRAM, and 64.35% resident-route coverage.  This reproduces the earlier
approximately 7.1--7.2 tok/s placement result.  The 200-token route export
showed 76.8% resident coverage in layer 38 and 73.1% in layer 39.

## Dense-island decomposition on the same placement

The existing dense execution plane was tested without CUPTI or timing dump.
QKV/Z-only was locally projection-exact at 128 tokens, but its 500-token
output hash did not match the CPU baseline and its single run was 6.885 versus
7.131 engine tok/s.  It therefore remains a load probe, not a production
optimization.

Enabling all ten attention layers and all shared experts raised Pascal power
to 46.6 W average and reached 8.501 engine tok/s over 500 tokens, but the
output differed from the exact baseline.  The 200-token run showed the same
direction (9.264 versus 7.185 engine tok/s) and also differed in output.  The
attention/shared projection checks can report zero local matrix error while
the end-to-end generated stream still diverges; local projection equality is
therefore insufficient as the acceptance test.

The dense parts remain opt-in (`COLI_DENSE_GPU`, `COLI_DENSE_ATTN`, and
`COLI_DENSE_SHARED`).  The current exact production configuration is still
the persistent-metadata DP4A resident path plus the packed CPU expert island.
The next required engineering task is to identify and eliminate the
long-run state/ownership determinism gap before promoting any dense island
work.  The 10 tok/s goal is not yet reached without accepting an accuracy
change.

## State-only and isolated dense-island probes

The state-only DeltaNet probe was intentionally kept diagnostic.  It leaves
the four projections on the established CPU path and moves only convolution,
recurrent state, and gated normalization to the GPU.  The scalar GPU state
variant reached 3.091 engine tok/s at 41.68% sampled GPU utilization and
50.13 W, but was far slower than the exact baseline.  The parallel state
variant reached 6.803 tok/s versus a matched 7.668 tok/s baseline, with
18.56% versus 12.60% GPU utilization, and changed the output.  Higher GPU
activity therefore did not establish a useful or exact execution boundary.

An attention/shared-only probe was also negative: 6.383 versus 7.467 tok/s
on the matched 32-token pair, with output divergence.  A `COLI_DENSE_I8=0`
control reduced the engine to 2.309 tok/s, confirming that the existing CPU
int8 dense path is materially important and that replacing it with the f32
path is not a viable route.

These results reject another partial island split.  The next useful dense
executor must own a complete numerically validated consumer chain, or it
must remain on the CPU.  Moving a recurrent subgraph or a projection pair
independently adds transfers and state-ownership hazards without delivering
the desired makespan improvement.

### Full DeltaNet first-boundary probe

The first one-token layer dump showed that the complete device-owned
DeltaNet executor already differed in layer 0 (`max abs` about `0.215`), so
the error is not a late routing/residency effect.  A temporary probe then
replaced only the device-side `b/a` projections with the established CPU
matmul results while leaving the device conv/state/norm/out chain unchanged.
The first-layer difference was unchanged within dump noise (`max abs` about
`0.215`, identical RMS to the displayed precision).  The probe was removed
afterward.  This rejects `b/a` as the first useful repair target and narrows
the parity investigation to the device-side convolution, recurrent update,
normalization, or their input/reduction contract.  No dense executor is
promoted until that chain has component-level parity.

A one-shot component dump (`COLI_DENSE_FULL_DBG`, disabled by default) gives
a more precise result for the first full call.  Compared with the matching
CPU `DN_DBG` record, conv differed by `2.38e-7`, q by `3.73e-9`, recurrent
`outv` by `2.24e-8`, z was identical, gated norm differed by `1.49e-8`, and
the device output differed by `5.59e-9`.  The b/a vectors differed by
`7.15e-7` and `9.54e-7`.  These are local near-parity results, not an
end-to-end acceptance result: tiny differences can still accumulate through
the recurrent stack and change the sampled stream.  The dump is retained as
a diagnostic-only tool; its record and comparison are in
`dn_full_debug_comparison.json`.

## Single-island resident reduction cleanup

The first safe execution-envelope change after the rejected local fusion is
now implemented.  During one-row decode, when exactly one GPU island
contributes to a resident layer, the sole partial-result buffer is aliased as
the accumulator.  The existing `sum_slots` kernel is an identity in this
case and is skipped.  Multi-row prefill deliberately remains on the old
separate-accumulator path because its reduction ordering is not yet proven
equivalent.  The CUDA event and stream-wait semantics remain intact, so
decode arithmetic, routing, placement, CPU fallback, and the multi-island
reduction path are unchanged.

`COLI_CUDA_DIRECT_SINGLE=1` enables the direct path; unset or `0` restores the
old separate accumulator for a matched A/B.  The tier reports the
number of direct accumulations as `[qtier-exec] single-island direct
accumulators ...`.

This is deliberately a narrow coarse-boundary cleanup, not a new scheduler,
kernel fusion, graph implementation, or placement policy.  Its expected
benefit is limited to one reduction launch and its dependency per decode
GPU-active layer; it is a prerequisite sanity check before attempting a
larger device-owned layer executor.

The rebuilt 32-token structural checks exercised the new path on 2,306
resident calls and reported no reduction/fallback failure.  The old control
reported 2,305 resident calls and no direct accumulations.  A 500-token pair
also exercised the path, but the two fresh processes followed materially
different route/hit streams (46.2% versus 69.6% resident hit rate and
different output lengths).  The resulting 6.610 versus 7.524 tok/s numbers
are therefore not a valid performance A/B.  The implementation has a
structural PASS, but no performance PASS is claimed.

This also reinforces the architecture requirement: performance experiments
must compare the same routed work, not merely the same prompt and capacity.
The next executor prototype needs a replayable layer-work trace or a
deterministic route/state harness so that CPU/GPU execution changes can be
compared without the current fresh-process stream variance.

The standalone backend test `test_resident_direct.exe` now runs the same
resident expert through both accumulator modes and compares the downloaded
vectors byte-for-byte.  It reports `differing=0 max_abs=0`; the direct
accumulator therefore preserves the resident arithmetic contract.  A
one-token prompt route-replay check also produced byte-identical output in
old and direct modes, covering the Qwen decode path with `S=1`.  The
remaining uncertainty is performance attribution, not correctness of this
specific reduction removal.  A long matched A/B still requires replaying the
same route stream; fresh processes are not sufficient because routing and
residency state vary materially.

That matched test is now complete.  With a 500-token `x` prompt, identical
route records, 3,037 balanced resident slots, no timers/CUPTI, and the same
warmup, the old accumulator measured 4.413 engine tok/s and the direct
accumulator 4.395 tok/s.  Both produced SHA-256
`091e895cbb798035205783d706a7d667e922f0fd966164b6f92a3d52a2ca824f`.
GPU telemetry was also indistinguishable: 7.68% versus 8.01% average
utilization and 21.69 versus 21.69 W.  This is a correctness and structural
PASS, but a neutral performance result (about -0.4%), so this alone is not
the route to 10 tok/s.

The same replay isolated the CPU decode-team setting.  Four OpenMP decode
threads (`OMP_NUM_THREADS=4`, `COLI_QWEN_DECODE_THREADS=4`) measured 5.954
engine tok/s versus 4.413 with the otherwise identical ten-thread
configuration.  The output hash remained identical, GPU utilization was
8.93%, and average power was 21.71 W.  This is a real CPU-island baseline
improvement, but Pascal remains lightly loaded and the 10 tok/s target is
still open.

The four-thread policy is now the engine default for `S=1` decode, while
`COLI_QWEN_DECODE_THREADS` remains the override for a NUMA-local CPU island.
Two fresh 500-token production-prompt runs with the same balanced 3,037-slot
placement measured 7.572 and 7.514 engine tok/s (mean 7.543), with identical
output hash `2fda9ea8410f45ab2fa5fcc537520457fea6fe562a5fd71673db47e1babd26ff`.
Average GPU utilization was 25.6--25.9% and power 23.75--23.92 W.  This
closes the safe CPU-island tuning step; the remaining gap to 10 tok/s still
requires a useful GPU execution-plane change.

As a deliberately non-production load probe, the same default-four-thread
configuration was run with all attention and shared-expert dense projections
on the GPU (`COLI_DENSE_GPU=1`, `COLI_DENSE_ATTN=1`, and
`COLI_DENSE_SHARED=1`; full DeltaNet remained disabled).  It reached 8.118
engine tok/s (7.979 request tok/s), with 27.85% average GPU utilization and
42.37 W, but its 500-token output was not exact.  This is evidence that
feeding the Pascal a coarser dense workload raises activity and moves toward
the target, not an accepted execution path.  The next implementation must
preserve the complete numerical consumer chain and validate long-run output
before any such dense island is promoted.

## Resident input-alias coarse-boundary probe

The first small executor-envelope change after the descriptor adapter removes
one redundant activation copy on the single-GPU SM61 path.  The resident
expert executor normally copies the home-device activation into its own CUDA
scratch buffer before quantizing it.  With `COLI_CUDA_DIRECT_INPUT=1`, and
only when every routed resident tensor satisfies the already validated DP4A
`fmt=4`, `gs=64`, decode geometry, the DP4A quantizer reads the home-device
row directly.  W4, mixed geometry, multi-device, and fallback paths retain the
old copy.

The 500-token route-replay A/B used the same `x` trace, no warmup, 3,037
balanced slots, DP4A, and no timers/CUPTI.  The old path reached 6.005 engine
tok/s and the alias path 5.973 tok/s.  Both outputs were byte-identical
(`ccec6f1737048aa5df8c4dc5217be2c1e7fbece6a3998cf3779ffe7dfcb4ea8c`).
GPU utilization was 10.97% versus 10.88%, power 22.58 versus 22.78 W, and
VRAM 5,620 versus 5,612 MiB.  This is a correctness/structural PASS but a
neutral performance result (about -0.5%); the copy is not the dominant
single-GPU bottleneck.  The opt-in path remains useful as a clean island
ownership primitive, but it is not promoted as the route to 10 tok/s.

## Long-run exactness boundary trace

To separate numerical drift from route/residency variance, the dense probes
were rerun with the same 500-step route replay and per-step layer dumps.  The
dump writer was corrected to append rather than overwrite each step.  The
combined attention+shared probe remained bit-identical through tokens 0 and 1;
its first nonzero record was token 2, layer 15, after the attention sublayer,
with maximum absolute difference `0.022494`.  Shared-only remained identical
through tokens 0 and 1 and first diverged at token 2, layer 19, with maximum
absolute difference `0.015521`.  The complete machine-readable comparison is
in `dense_layertrace_128_comparison.json`.

The existing attention projection spot-checks for q/k/v and o reported zero
error on their first checked calls.  Therefore the next exact executor work
must trace attention/cache state and the complete layer ownership boundary;
it is not justified to replace another individual projection based only on a
local matrix check.  The split probe narrows this further: attention-only was
bit-identical to the baseline for all 128 tokens, while shared-only first
drifted at token 2/layer 19.  The combined attention+shared run first drifted
at token 2/layer 15.  This interaction is consistent with an execution/state
ownership or scheduling race being exposed when two GPU work classes are
active, rather than proving that the attention arithmetic itself is wrong.
The dense paths remain diagnostic and disabled by default.  This trace is the
gate before implementing a production coarse layer executor.

## Coarse shared-MLP boundary spike

The first architecture prototype moved the complete shared MLP call behind an
optional `coli_cuda_pipe_dense_mlp()` backend entry point.  The existing gate
and up projection kernels, SiLU operation, and down projection remain separate;
the host-side qwen code only selects the executor.  This preserves the
projection reduction boundaries and leaves the original path available with
`COLI_DENSE_SHARED_COARSE=0`.

The first GPU-local version used the device `silu_mul` kernel.  It ran at 6.895
engine tok/s on a 128-token route-replay probe, but that first record was later
invalidated because the prototype used scratch slot 25, which overlapped the
resident hub.  Independently, the GPU-SiLU path is not an accepted numerical
route: the first eight tokens were equal, but small device-versus-host `expf`
differences can accumulate after recurrent state updates.

An exactness control kept the existing host `expf` semantics inside the
backend.  With dedicated scratch slots 27--29 it reproduced the current
dense-shared reference hash
`95f55cb187fc1425a9dad210a39de043f8775ac767946e7e3117d5e9c066bf8b`.  It
reached 4.827 engine tok/s versus 5.464 tok/s for the matched existing
dense-shared route, an 11.7% regression.  The extra device-to-host activation
copy, CPU SiLU, and host-to-device upload are therefore real cost, not merely
an API accounting artifact.

The machine-readable records are in `qwen36_coarse_island_results.json`.
Conclusion: the coarse boundary abstraction is viable, but this particular
partial shared-MLP transplant is rejected for production.  The next useful
executor must keep the complete numerically validated consumer chain local to
an island, or provide a separately validated bit-compatible activation
implementation; merely hiding the old host/device round trips behind one
function is insufficient.  No placement, scheduler, DP4A arithmetic, or
default execution policy was changed.

## Transient execution-cache control

As a separate load-versus-throughput control, eight temporary GPU execution
slots were allowed to stage routed CPU misses without changing the persistent
3,037-expert placement table.  On the same 128-token route replay this raised
average Pascal utilization from 8.89% to 37.24%, but reduced engine throughput
from 6.075 to 3.761 tok/s.  The output hash stayed identical.  The extra
per-layer expert uploads and staging traffic therefore buy GPU activity, not
useful throughput; this is not the path to the 10 tok/s target.

## Exact dense GEMV execution-shape probe

The execution-plane question was also tested below the full model layer.  An
opt-in `COLI_DENSE_EXACT=1` kernel mirrors the existing CPU Qwen dense GEMV
reduction shape: 32 logical accumulation streams followed by the same ordered
FP32 reduction and round-to-nearest operations.  The model-free parity test
passed for `1024x257`, `2048x1024`, and `1024x4096` matrices with zero
differences.  This is an exactness control, not a new default kernel.

On a corrected absolute route replay, with attention and shared projections
using this exact path, the 500-token output was byte-identical to the existing
reference (`091e895cbb798035205783d706a7d667e922f0fd966164b6f92a3d52a2ca824f`).
The run reached 2.308 engine tok/s, with 15.71% average GPU utilization and
45.38 W average power.  The result is structurally correct but slower/neutral
against the matched control; it is not a route to 10 tok/s and remains
diagnostic-only.

## Compute Islands v0: current boundary classification

The existing `QtIslandWork` and `qt_execute_gpu_island` boundary was inspected
without changing routing, placement, or arithmetic.  It represents one
`(layer, device)` GPU island batch: routed expert tensors, normalized routing
weights, a home-device activation pointer, and one partial-result destination.
The executor submits the already validated resident sequence.  The CPU
fallback is still assembled by Qwen's host-side miss loop/batch path; it is not
yet represented by a symmetric `CpuIslandWork` object.

The current decode order is therefore:

```text
router/top-k and residency classification
       |
       +--> qt_note / resident metadata publication
       |
       +--> GPU island work: input upload -> resident sequence -> partial
       |
       +--> CPU fallback work: lookup/rematerialize -> 3 projections -> SiLU -> weighted add
       |
       +--> shared expert on the host
       |
       `--> qt_take: GPU completion/reduction/D2H -> layer merge
```

The GPU work is already submitted before the CPU fallback loop, so a wrapper
that merely renames `qt_issue` is not a new coarse executor.  The missing
architectural piece is a symmetric, explicit per-layer work descriptor whose
CPU and GPU local executors expose start/finish timestamps and one common
completion contract.

The first machine-readable classification is
`compute_islands_v0_current16.json`, generated from the existing
`overlap_diag_dp4a_16.csv.layers.csv`; the Markdown summary is
`compute_islands_v0_current16.md`.  It reports 40 layers, 4,800 routed expert
assignments, 3,312 GPU hits, and 1,488 CPU fallbacks.  Mean measured values
are:

```text
layer envelope       23.618 ms
CPU window            18.798 ms
GPU event work         6.392 ms
CPU/GPU overlap        3.418 ms
residual envelope      4.820 ms
absolute imbalance    12.406 ms
```

The 4.820 ms residual is deliberately labelled boundary/measurement territory,
not proven mutex or synchronization time.  The source trace combines host
timestamps and CUDA-event durations without a synchronized clock.  A proper
cross-clock per-layer trace is required before turning that residual into a
scheduler cost.

### Compute Islands v0 gate status

* Gate A — current boundary understood: **PASS-provisional**.
* Gate B — explicit GPU boundary with unchanged kernels: **PASS-structural**;
  the abstraction exists, but CPU symmetry is incomplete.
* Gate C — CPU/GPU overlap: **PASS-observed**, not yet cross-clock exact.
* Gate D — max-island makespan model: **PASS-provisional**; the residual is
  still unassigned measurement territory.
* Gate E — performance improvement: **NOT TESTED by this analysis**.
* Gate F — scheduler readiness: **NOT READY**; per-island costs and the
  residual boundary need a matched controlled executor trace.

The first narrow implementation now adds an opt-in `CpuIslandWork` descriptor
and routes it to the existing exact CPU batch executor.  The GPU side now has
the corresponding `QtLayerWork` descriptor containing all active local
`QtIslandWork` batches; its executor still calls the same validated resident
entry point in the same order.  GPU submission, CPU assignment, kernels, and
reduction order are unchanged.

The matched 500-token A/B is recorded in `compute_islands_v0_results.json`:
both outputs are bit-identical, but one pair is insufficient to distinguish
the 2.931 versus 3.180 tok/s result from runtime-state variance.  This is a
structural Gate B pass, not yet a performance pass.  The next step is to
repeat that A/B alternately with lightweight descriptor start/finish
accounting; work assignment and multi-GPU scheduling remain frozen.

## Compute Islands v0 lightweight timing run

The requested timing-only run was then executed with `COLI_ISLAND_TIMING=1`,
without `COLI_TIMERS`, CUPTI, or the timing dump.  It produced exactly 8,000
records: 200 decode tokens by 40 layers.  The generated output remained exact
with SHA-256
`bc6c9d1b920480387d58e3c51a973e8cac24b61d1b1d8e37e1640e54eea613fa`.

The raw per-layer/tokens CSV is `compute_islands_v0_200.csv`; the aggregate
and per-layer analysis is in `compute_islands_v0_200.json` and
`compute_islands_v0_200.md`.

```text
mean layer makespan       9.908 ms
median layer makespan    10.559 ms
p95 layer makespan       14.007 ms
mean GPU dispatch delay   0.077 ms
mean GPU lane              0.220 ms
mean CPU lane              3.921 ms
mean exposed merge         0.404 ms
```

Across individual token/layer samples, the conservative diagnostic classifier
reported 5,550 CPU-critical, 2,399 GPU-critical, and 51
merge/synchronization-critical samples.  This is already useful scheduler
evidence: the current route-replay placement does not leave a uniformly
GPU-critical system.  Most assigned fallback work is CPU-dominant, while the
early resident-only layers are GPU-dominant under this placement.

The layer makespan intentionally starts at the complete model-layer entry,
therefore it includes the preceding DeltaNet/attention work.  CPU/GPU lane
durations begin at the island assignment boundary and describe only the
assigned resident/fallback work.  This distinction is required: the
optimization target is full layer completion, while the assignment signal is
the relative island completion time.  `gpu_slack_ms` is measured from GPU
completion to full layer completion; it is zero for CPU-only samples and is
not interpreted as a cross-GPU slack value yet.

### Revised Compute Islands v0 gates

* Gate A — current boundary understood: **PASS**.
* Gate B — explicit CPU/GPU layer boundary with unchanged arithmetic:
  **PASS**.
* Gate C — overlap exists before the layer boundary: **PASS-observed**; the
  trace uses one host clock and host-clocked post-synchronization GPU bounds.
* Gate D — max-island makespan model: **PASS-provisional**; the measured
  residual and non-MoE layer work are now explicit, but not yet independently
  validated against a synchronized hardware timeline.
* Gate E — performance benefit: **OPEN/FAIL for this structural spike**; no
  performance improvement was expected or demonstrated.
* Gate F — scheduler readiness: **HOLD**; the signal exists, but the current
  route-replay placement and single-GPU data do not yet provide a calibrated
  multi-island cost model.

## Balanced-placement rerun

The final authoritative v0 trace uses the existing
`layer_balanced.stable.heat` placement file, still with 200 tokens and the
same route replay.  It includes corrected handling of layers with no CPU
fallback.  It produced 8,000 records and the same exact output hash.

```text
mean layer makespan      10.845 ms
median layer makespan    10.892 ms
p95 layer makespan       13.916 ms
mean pre-assignment       5.139 ms
mean dispatch delay       0.172 ms
mean CPU lane             4.836 ms
mean GPU lane             3.429 ms
mean exposed merge        0.222 ms
mean post-merge tail      0.043 ms
```

The per-sample classification is 4,796 GPU-critical, 2,637 CPU-critical,
535 balanced, and 32 merge-critical.  The important result is not that one
island universally dominates: the placement produces both CPU- and
GPU-critical work across layers, with a relatively small merge tail.  This
supports a future makespan-aware assignment signal, but does not yet justify
changing assignment policy.  The 5.139 ms pre-assignment component also
remains outside the MoE island assignment and must be kept separate when
estimating scheduler payoff.

## Makespan-balanced runtime A/B

The generated 3,642-slot makespan-balanced heatfile was tested against a
current-placement heatfile with identical 500-token requests, 32-token
warmup, DP4A configuration, and frozen LFRU.  Timers, timing CSV, timing dump,
and CUPTI were disabled.  The complete resource report is
`compute_islands_scheduler_ab_500_v3.json` and the interpretation is in
`compute_islands_scheduler_ab_500.md`.

```text
                    tok/s    ms/token   GPU util   power
current             2.2317     448.081      6.26%   44.63 W
makespan-balanced   2.2128     451.918      6.19%   44.70 W
```

The candidate was 0.85% slower, within observed run variance.  CPU load,
VRAM, and power were also effectively unchanged.  A second complete pair
showed the reverse direction, confirming that this is not a reproducible
performance win.  More importantly, a 64-token deterministic smoke produced
different output text/hashes for the two placements.  This heatfile therefore
does not pass the exact-output correctness gate and is not promoted to a
runtime policy.

## Offline scheduler simulator v0

`simulate_scheduler.py` now evaluates the same fixed resident budget as an
offline two-island model (`CPU0` and `GPU0`).  It consumes expert invocation
counts, the placement snapshot, and the lightweight layer-boundary trace.  It
keeps measured pre-assignment, dispatch, merge, and post-merge costs fixed and
models the assignment-sensitive part as:

```text
layer = pre + dispatch + max(CPU0 service, GPU0 service) + merge + post
```

The current run is based on the run-matched `placement_run5_dp4a_64` expert
export and the 3,642-slot placement, calibrated with the authoritative 200
token Compute Islands trace.  The model’s current placement is 10.439
ms/layer versus 10.845 ms/layer in the timing trace, a -3.74% calibration
error.  This is acceptable for a first-order policy comparison but not for an
absolute performance claim.

```text
policy                GPU hit   CPU fallback   predicted ms/token   vs current
current                 64.43%          7170              417.544       +0.000
global_hot              97.65%           474              432.036      -14.491
layer_balanced          97.00%           604              430.619      -13.075
critical_path           97.61%           481              431.353      -13.808
makespan_balanced       78.73%          4289              386.409      +31.135
```

The simulator therefore reproduces the central architectural result: hit
rate alone predicts the wrong direction, while a makespan-balanced policy
retains CPU work deliberately.  It also ranks marginal current-placement
moves and emits a heat file, but no runtime placement or scheduler change has
been made.  The machine-readable result is
`compute_islands_scheduler_v0.json`; the full layer table is in
`compute_islands_scheduler_v0.md`.
