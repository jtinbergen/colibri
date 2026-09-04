# Qwen3.6 SM61 resident-call operation DAG analysis

## Trace quality

This is a structural CUPTI analysis. CUPTI collection materially perturbed the 200-token run; absolute host wall-clock values are diagnostic only.

## Counts

- Complete decode windows: **199**; resident calls: **7960**
- Issued resident kernels: **39560** (4.970/call); all kernel records: **99642**
- Memcpy records: **184514**; CUDA API records: **536613**

## Kernel breakdown

| kernel type | launches | /resident call | mean us | queue->start mean us | GPU ms/token |
|---|---:|---:|---:|---:|---:|
| down_dp4a | 7912 | 0.994 | 239.850 | 762.978 | 9.536 |
| gate_up_dp4a | 7912 | 0.994 | 593.410 | 163.433 | 23.593 |
| quantize_down_activation | 7912 | 0.994 | 10.938 | 754.458 | 0.435 |
| quantize_gate_up_activation | 7912 | 0.994 | 8.552 | 160.010 | 0.340 |
| weighted_sum_rows | 7912 | 0.994 | 7.724 | 998.114 | 0.307 |

## Representative operation timelines

Times below are relative to `qt_issue` entry and are diagnostic CUPTI host-normalized timestamps.

### cheap_normal (row 7395, token 185, layer 35)

| relative start us | end us | type | operation | bytes | caller API |
|---:|---:|---|---|---:|---|
| 265.8 | 268.1 | memcpy | memcpy | 8192 | cudaMemcpy_v3020 |
| 272.7 | 275.6 | memcpy | memcpy | 8192 | cudaMemcpyPeerAsync_v4000 |
| 283.2 | 283.8 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 286.7 | 287.3 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 290.1 | 290.7 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 293.5 | 294.1 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 296.9 | 297.5 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 300.2 | 300.8 | memcpy | memcpy | 48 | cudaMemcpyAsync_v3020 |
| 304.6 | 312.1 | kernel | quantize_gate_up_activation | 0 |  |
| 317.1 | 953.4 | kernel | gate_up_dp4a | 0 |  |
| 958.5 | 969.3 | kernel | quantize_down_activation | 0 |  |
| 974.3 | 1260.3 | kernel | down_dp4a | 0 |  |
| 1264.9 | 1265.6 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 1269.4 | 1277.3 | kernel | weighted_sum_rows | 0 |  |
| 1284.6 | 1287.9 | memcpy | memcpy | 8192 | cudaMemcpyPeerAsync_v4000 |
| 1637.6 | 1640.5 | memcpy | memcpy | 8192 | cudaMemcpy_v3020 |

Host: issue 171876787.5185→171876787.6910 ms; CPU parallel window 171876787.6910→171876788.9600 ms; take 171876788.9600→171876789.1656 ms.

### expensive_problem_38_39 (row 3599, token 90, layer 39)

| relative start us | end us | type | operation | bytes | caller API |
|---:|---:|---|---|---:|---|
| 341.0 | 343.4 | memcpy | memcpy | 8192 | cudaMemcpy_v3020 |
| 348.3 | 351.2 | memcpy | memcpy | 8192 | cudaMemcpyPeerAsync_v4000 |
| 358.8 | 359.4 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 362.3 | 362.9 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 365.7 | 366.3 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 369.2 | 369.8 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 372.6 | 373.2 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 376.1 | 376.6 | memcpy | memcpy | 24 | cudaMemcpyAsync_v3020 |
| 380.4 | 387.6 | kernel | quantize_gate_up_activation | 0 |  |
| 392.8 | 779.1 | kernel | gate_up_dp4a | 0 |  |
| 784.2 | 793.9 | kernel | quantize_down_activation | 0 |  |
| 799.1 | 961.2 | kernel | down_dp4a | 0 |  |
| 965.8 | 966.4 | memcpy | memcpy | 12 | cudaMemcpyAsync_v3020 |
| 970.2 | 976.3 | kernel | weighted_sum_rows | 0 |  |
| 983.6 | 986.9 | memcpy | memcpy | 8192 | cudaMemcpyPeerAsync_v4000 |
| 3255.9 | 3258.8 | memcpy | memcpy | 8192 | cudaMemcpy_v3020 |

Host: issue 171407532.9744→171407533.2884 ms; CPU parallel window 171407533.2884→171407535.8691 ms; take 171407535.8691→171407536.2429 ms.

## One complete 40-layer decode window

Selected median complete decode token: **100**.

| layer | GPU routes | CPU routes | resident GPU service ms | issue→take host ms | executed kernel types |
|---:|---:|---:|---:|---:|---|
| 0 | 7 | 1 | 1.141 | 1.806 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 1 | 7 | 1 | 1.079 | 1.545 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 2 | 5 | 3 | 0.811 | 2.312 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 3 | 5 | 3 | 0.808 | 3.259 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 4 | 7 | 1 | 1.098 | 1.464 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 5 | 3 | 5 | 0.566 | 2.078 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 6 | 4 | 4 | 0.713 | 2.351 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 7 | 6 | 2 | 0.944 | 2.553 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 8 | 5 | 3 | 0.807 | 1.961 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 9 | 2 | 6 | 0.403 | 2.454 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 10 | 4 | 4 | 1.850 | 3.359 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 11 | 4 | 4 | 0.701 | 2.800 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 12 | 5 | 3 | 0.820 | 1.906 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 13 | 6 | 2 | 0.965 | 1.519 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 14 | 6 | 2 | 0.966 | 1.815 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 15 | 5 | 3 | 0.801 | 1.324 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 16 | 4 | 4 | 0.716 | 2.079 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 17 | 3 | 5 | 0.564 | 2.576 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 18 | 4 | 4 | 0.706 | 2.371 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 19 | 6 | 2 | 0.937 | 1.292 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 20 | 4 | 4 | 0.690 | 2.361 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 21 | 3 | 5 | 0.564 | 2.552 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 22 | 3 | 5 | 1.566 | 4.042 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 23 | 3 | 5 | 0.575 | 3.548 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 24 | 4 | 4 | 0.705 | 2.095 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 25 | 3 | 5 | 0.568 | 2.593 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 26 | 3 | 5 | 0.563 | 2.580 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 27 | 4 | 4 | 0.699 | 2.143 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 28 | 3 | 5 | 0.558 | 2.577 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 29 | 1 | 7 | 0.216 | 3.811 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 30 | 3 | 5 | 0.562 | 2.607 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 31 | 4 | 4 | 0.673 | 2.315 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 32 | 4 | 4 | 0.780 | 3.345 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 33 | 3 | 5 | 0.567 | 3.215 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 34 | 5 | 3 | 0.806 | 1.840 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 35 | 4 | 4 | 0.717 | 1.727 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 36 | 4 | 4 | 0.695 | 1.918 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 37 | 5 | 3 | 0.823 | 2.461 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 38 | 4 | 4 | 0.713 | 2.979 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |
| 39 | 3 | 5 | 0.563 | 2.849 | down_dp4a:1, gate_up_dp4a:1, quantize_down_activation:1, quantize_gate_up_activation:1, sum_slots:1, weighted_sum_rows:1 |

The normal resident envelope is the five issued-kernel sequence visible in the representative DAG: activation quantization, fused gate/up DP4A, down-input quantization, down DP4A, and weighted aggregation. `sum_slots` is additional work in the take/window path and is therefore not included in the five-issued average. `offset_to_signed_s4` is a separate layout/rematerialization operation, not part of every resident call.

## Queue latency

- Mean queue->start: **567.799 us**; p95: **1132.708 us**; max: **10235.272 us**. The p95 of per-call mean queue delay is 796.4 us; the p95 of individual issued-kernel waits is 1132.7 us.
- Of issued-kernel queue wait, conservative prior-GPU-work coverage: **16389.286 ms**; exposed/unexplained: **6072.827 ms**.
- CUPTI cannot split the uncovered remainder into host preparation, stream dependency, driver/API, and instrumentation without a lower-perturbation synchronized trace.

## Memcpy/API interpretation

The JSON contains all-trace and resident issue/take/window histograms by direction, size bucket, and API label. Small D2D/D2H transactions are envelope activity, not bandwidth-limited transfers. Pointer/table maintenance and staging can be removed only if the corresponding buffers are made persistent and the routing metadata update is made device-resident.

In the resident window the recurring shape is approximately 11 memcpy records/call: one 8 KiB upstream H2D activation staging copy, two 8 KiB D2D P2P copies (input to island and partial result home), six ≤64 B H2D pointer-table updates, one small H2D weight-vector update, and one 8 KiB D2H result copy. The first H2D is adjacent upstream staging; the six pointer updates are implementation/layout overhead; the P2P/result copies are real island-boundary transfers.

## API boundary classification

Per resident window the trace contains approximately 32.8 CUDA API records: 6 kernel-launch records per window (five issued here plus one `sum_slots` executing in the take window), one stream synchronization, one stream-wait-event, five event records, two event elapsed-time queries, about seven async memcpy calls, two synchronous memcpy calls, and one capability query. The event record/elapsed-time calls are diagnostic-only because this run used `COLI_TIMERS=1`; the stream sync, waits, launches and data movement are production path operations. `cudaGetDeviceProperties` is an avoidable per-call capability query and should be cached, but the trace alone cannot assign C source lines beyond the correlation/API name.

## Optimization candidates (not implemented)

1. **Cache SM capability and persistent device metadata**: remove the per-call `cudaGetDeviceProperties` and repeated pointer/control copies; low-to-medium correctness risk; directly compatible with dynamic routing and future islands.
2. **Fuse/hoist activation quantization**: remove one strict launch and an intermediate buffer for gate/up; separately hoist down-input quantization so it is not repeated per output block; medium risk, compatible with dynamic routing if scratch buffers remain stable.
3. **Fuse weighted aggregation with down or use one batched reduction**: remove a completion-critical micro-launch and intermediate write; medium risk because expert routing/weights must remain exact.
4. **CUDA Graph feasibility test**: capture stable per-layer topology with stable buffers and update parameters/pointers; medium-to-high implementation risk, promising for dispatch/API pressure and future islands.
5. **Earlier asynchronous issue / fewer completion boundaries**: potentially high upside but constrained by GDN dependencies; defer until the present envelope is reduced.

## Current layer critical path

**Structural path:** upstream activation staging → P2P input → pointer/control updates → quantize gate/up → fused gate/up DP4A → quantize down activation → down DP4A → weight update → weighted aggregation → P2P partial result → event wait → `sum_slots` reduction → stream synchronization → D2H result. CPU fallback runs in parallel between issue and take.

**Avoidable:** one capability query/call, six small pointer-table H2D updates/call, at least one quantization/reduction launch boundary, and potentially one result staging boundary. The trace supports the candidate classes but not a trustworthy production-ms total because CUPTI perturbed the run.

**Probably unavoidable today:** one layer completion boundary, dynamic routing metadata, one input/result transfer per island boundary, and mathematically required expert projection/reduction work.

**Highest-value first intervention:** remove the six pointer-table copies and per-call capability query by keeping device metadata persistent, then validate with a no-CUPTI matched run; next test quantization hoisting/fusion.

The arithmetic kernel is not the explanation for the five-launch envelope. The current path pays for multiple strict stream stages plus metadata/staging/API boundaries around very short Pascal kernels. Absolute ms/token savings require a matched non-CUPTI validation run.
