# MiniMax-M3 dedicated-memory ROI campaign — 2026-09-09

Host condition: Chrome closed; Windows 11; i5-12600K; 32 GB RAM; GTX 1080.

Common benchmark settings: persistent mode, MiniMax-M3 int4, `max-new=32`,
one discarded warm-up, two identical warm requests, three rotating prompts,
`COLI_TEMP=0`, `--no-evict`.

## CPU, C: only

Environment: `COLI_CUDA=0`; no `COLI_MODEL_MIRROR`.

- Engine load: 4.75 s
- Expert tiers: VRAM 0 / RAM 399 / disk 6897
- Rotating requests: 109.16 s, 110.79 s, 112.55 s
- Rotating median: 0.44 tok/s
- Rotating p95 request: 112.38 s
- Rotating median cache hit: 41.1%
- RSS: approximately 20.5–21.2 GB
- Profile medians: expert disk 153.341 s; expert wait 53.660 s;
  expert matmul 9.828 s; attention 5.559 s; LM head 0.817 s

## CUDA, C: only

Environment: `COLI_CUDA=1`, `COLI_GPU=0`, `CUDA_DENSE=1`,
`CUDA_EXPERT_GB=4`, `COLI_CUDA_PROFILE=1`.

- CUDA device confirmed: NVIDIA GeForce GTX 1080, sm_61, 8.6 GB VRAM
- Engine load: 4.81 s
- Expert tiers: VRAM 0 / RAM 456 / disk 6840
- Rotating requests: 110.99 s, 114.98 s, 113.15 s
- Rotating median: 0.43 tok/s
- Rotating p95 request: 114.79 s
- Rotating median cache hit: 43.3%
- Profile medians: expert disk 150.222 s; expert wait 53.435 s;
  expert matmul 10.995 s; attention 6.202 s; LM head 0.712 s

This was a real CUDA run, but not yet an expert-island placement run:
`VRAM 0` means no routed experts were resident in VRAM. The automatic pin
policy retained only about 0.4 GB after its LRU reserve, below the loading
threshold. The result therefore evaluates dense GPU offload plus CUDA
overhead, not the intended routed-expert compute-island configuration.

## CUDA routed-expert tier, C: only

Environment: `COLI_CUDA=1`, `COLI_GPU=0`, `CUDA_EXPERT_GB=4`,
`PIN=auto`, `PIN_GB=4`, `CUDA_RELEASE_HOST=1`, `CUDA_DENSE` unset.

- CUDA expert tier confirmed: 125 VRAM experts, 3.98 GB; 126 additional
  usage-ranked pinned RAM experts; 342 LRU experts
- Rotating requests: 106.59 s, 105.52 s, 110.31 s
- Rotating median: 0.45 tok/s
- Rotating p95 request: 109.94 s
- Rotating median cache hit: 43.6%
- GPU critical routed time: approximately 0.44 s per profiled request
- Routed CPU time: approximately 9.4 s per profiled request
- Profile time shares: expert I/O 72%; expert matmul 14%; attention 8%
- Tier straggler: approximately 21.5x

This is a genuine routed-expert compute-island run, but it did not produce a
material end-to-end improvement over the CPU or mirror baselines. The GPU
island is underfed by storage and routing imbalance; DP4A is not yet the
dominant opportunity.

## CPU, C:+E: mirror

Mirror path: `E:\Models\minimax_m3_i4`.

The benchmark is still running. Startup measurements confirmed:

- C: measured at 5.04 GB/s
- E: measured at 0.98 GB/s
- mirror split: 84% / 16%

The full result will be appended after completion.

Completed result:

- Engine load: 4.76 s
- Expert tiers: VRAM 0 / RAM 456 / disk 6840
- Rotating requests: 102.70 s, 107.60 s, 108.80 s
- Rotating median: 0.45 tok/s
- Rotating p95 request: 108.68 s
- Rotating median cache hit: 43.3%
- Profile medians: expert disk 138.503 s; expert wait 51.393 s;
  expert matmul 10.812 s; attention 5.997 s; LM head 0.825 s

The mirror was measured at startup as C: 5.04 GB/s and E: 0.98 GB/s,
with an 84% / 16% read split. It improved the rotating median modestly
relative to C: only, but did not produce a large end-to-end gain.

## Scope

These runs are baseline measurements. The new telemetry-driven planner,
active admission, and explicit layer-affinity compute-island planning are not
enabled in this campaign. The routed-expert CUDA arm did run with explicit
pinning and verified nonzero VRAM expert residency, but the current placement
is a globally popularity-ranked prefix rather than a layer-complete policy.
No result here should be interpreted as evidence that multiple compute islands
are useless.

## CUDA full layer 49, 5 GB candidate

The first attempt was discarded because `CUDA_RELEASE_HOST=1` implicitly filled
the candidate tier and loaded 313 experts instead of the requested full layer.
That run was aborted and is not included below. The valid rerun used
`PIN_FILL=0` and loaded exactly 128/128 layer-49 experts into VRAM (4.08 GB).

- Expert tiers: VRAM 128 / RAM 448 / disk 6720
- Rotating requests: 107.45 s, 107.72 s, 107.74 s
- Rotating median: 0.45 tok/s
- Rotating p95 request: 107.74 s
- Rotating median cache hit: 44.2%
- Profile medians: expert disk 140.770 s; expert wait 50.807 s;
  expert matmul 9.827 s; attention 5.578 s; LM head 0.814 s
- Profile time shares: expert I/O 72%; expert matmul 14%; attention 8%
- GPU routed critical time: approximately 0.10 s per profiled request

This single full-layer result is consistent with the routed-expert baseline,
but does not yet establish a material end-to-end improvement. Storage remains
the dominant limit. The 80%-coverage candidate for contiguous layers 17--22
has been generated separately (142 experts, approximately 4.5 GB) and still
needs its matched A/B run.

## CUDA contiguous layers 17--22, 80% per-layer coverage

This valid run used `PIN_FILL=0` and loaded exactly 142/142 candidate experts
into VRAM (4.52 GB). The candidate contains the smallest observed expert set
per layer reaching 80% cumulative routed-hit coverage for layers 17--22.

- Expert tiers: VRAM 142 / RAM 456 / disk 6698
- Rotating requests: 102.10 s, 105.40 s, 107.73 s
- Rotating median: 0.47 tok/s
- Rotating p95 request: 107.49 s
- Rotating median cache hit: 46.8%
- Profile medians: expert disk 133.955 s; expert wait 48.615 s;
  expert matmul 9.705 s; attention 5.580 s; LM head 0.810 s
- Profile time shares: expert I/O 70%; expert matmul 15%; attention 9%
- GPU routed critical time: approximately 0.31 s per profiled request

Relative to the single-layer-49 full-residency run, this is a suggestive small
improvement, but not a conclusive ROI result: the end-to-end difference is
within the scale of a single short campaign and the unified M3 multi-island
DAG was not enabled. The result does confirm that a contiguous, per-layer
80%-coverage candidate can be loaded and executed as a real GPU-resident
placement.
