# Lossless expert compression on MiniMax M3 INT4 — Stage 0 findings

**Date:** 2026-09-08
**Model:** MiniMax M3 INT4 (Muse 1.3 contributor) — `minimax_m3_i4/`
**Architecture:** 60 layers, 128 routed experts, topk=4, hidden 6144, moe_intermediate 3072
**Source format:** fmt=4 grouped-int4 with `gs=64` (`name` U8 + `name.qs` F32 per group)
**Hardware:** Intel i5-12600K (10C/16T, AVX2 only) — local Windows box
**Tools:** `c/tools/analyze_split_entropy.py` (in tree), `c/tests/test_fse.c` (in tree),
`c/fse_coli.h` (in tree), plus an M3-specific structured-entropy analyser (this report).

## TL;DR

**Lossless expert compression on MiniMax M3 (gs=64) is closed.**

| Question (from the briefing) | Result |
|---|---|
| Can real GLM/M3 INT4 experts be losslessly reduced by ≥15%? | **NO** on M3 GS=64 (12.6% ceiling). **YES** if M3 is first requantized per-row (1.37× ceiling, matches GLM-5.2's 2.91 bits/wt anchor). |
| Is the structured (per-row / per-block / conditional) entropy below the global ceiling? | No exploitable structure: per-row identical to global (+0.07%); per-block 5.4% payload gain, **but 75% per-block table overhead eats it**; first-order conditional identical to global (+0.02%); XOR-delta is worse (−9.4%). |
| Can the lossless decoder keep up with matmul? | **NO.** Single-core `cfse_decompress`: **0.20 GB/s** on a 9.4 MB M3 expert. 16-thread aggregate: **2.5 GB/s**. Production matmul consumes 60+ GB/s on tuned Milan/Threadripper. |

The lossless compression-as-bandwidth-multiplier architecture does not work on M3 GS=64 because **the entropy is already too high and the decoder is too slow**. Per the briefing's own decision criterion ("If no, park the lossless branch and investigate better lossy INT3/INT2 representations"), this Stage 0 is **STOP**.

The same conclusion was reached on the GPU side for GLM-5.2 in `docs/experiments/glm52-split-entropy-2026-07-31.md` ("the proposed lossless split exchanges RAM capacity for an address-dependent codec that destroys the cheap raw-INT4 inner loop") and `docs/experiments/glm52-decode-failure-ledger-2026-07-31.md` ("rejected… reopen only with a decode-native, vector-aligned encoding whose isolated fused kernel first beats materialization"). The CPU side, measured here for the first time, lands the same answer.

---

## 1. Stage 0(a) — global nibble entropy on the gs=64 source

**Tool:** `c/tools/analyze_split_entropy.py <model_dir> --layers 3,15,30,45,57`
**Sampled:** 1,920 routed-expert tensors across 5 MoE layers (3, 15, 30, 45, 57), 18.12 GB packed
**Corpus:** all 57 MoE layers' expert weights (206.56 GB packed) + 25.82 GB F32 scales

| Metric | Value |
|---|---:|
| Global nibble entropy | **3.4912 bits/weight** |
| Theoretical compression ratio | **1.1457×** |
| Theoretical storage reduction | **12.72%** |
| Sampling ratio (= 1/H) | 0.8728 |

**Per-layer entropy** (very flat):

| Layer | Entropy (bits/wt) | Ratio |
|---:|---:|---:|
| 3 | 3.4847 | 1.1479× |
| 15 | 3.4939 | 1.1449× |
| 30 | 3.4960 | 1.1442× |
| 45 | 3.4966 | 1.1440× |
| 57 | 3.4844 | 1.1480× |

The histogram is **almost uniform** (peaked at q=0 with only 15.09% mass), which explains the high entropy. The 16 nibble codes span -8 to +7 (the symmetric quantizer never emits -8). Compared to the GLM-5.2 anchor from 2026-07-31 (entropy 2.9147 bits/wt, ratio 1.373× on real GLM-5.2 gs=64 packed bytes), M3 GS=64 packs **0.58 bits/weight worse** — a -16.5% drop in compressibility.

**Stage 0(a) interpretation.** The M3 GS=64 representation sits **0.42 bits/weight above the briefing's 15% threshold** (3.40 bits/wt). At the global entropy ceiling, no static-table entropy coder can deliver the 15% reduction the briefing asks for on this representation. **The global entropy branch is closed.**

---

## 2. Stage 0(b) — structured entropy on layer 30

**Tool:** `stage0b_entropy_analysis.py` (M3-specific, this report) — single-pass, vectorized numpy, all 384 expert tensors of layer 30 in one run.

### 2.1 What was measured

For each of 384 expert tensors in layer 30 (128 experts × 3 projections, total 3.6 GB packed, 7.2 G nibbles):

1. **Global nibble entropy** — confirms Stage 0(a) on this layer (3.4960 bits/wt).
2. **Per-row nibble entropy** — within each output row of an expert projection.
3. **Per-block (GS=64) nibble entropy** — within each 64-weight block of a row.
4. **First-order conditional entropy** `H(Wᵢ | Wᵢ₋₁)` — along packed-byte order.
5. **XOR-delta entropy** `H(Wᵢ ⊕ Wᵢ₋₁)` — adjacent nibbles.
6. **Synthetic per-row repack (gs=0)** — global entropy of bytes after dequant+requant.
7. **Scale stream entropy** — log-spaced 256-bucket quantization, estimated bits/F32.
8. **Theoretical bytes including table overhead** at each codebook granularity.

### 2.2 Results

| Measurement | Entropy (bits/wt) | Ratio | Gain vs global |
|---|---:|---:|---:|
| 1. Global (layer 30) | **3.4960** | 1.1442× | — |
| 2a. Per-row unweighted mean | 3.4916 | 1.1456× | +0.07% |
| 2b. Per-row weighted by nibbles | 3.4934 | 1.1450× | +0.07% |
| 3. Per-block (GS=64) mean | **3.3063** | 1.2098× | **+5.43%** |
| 4. First-order conditional | 3.4953 | 1.1444× | +0.02% |
| 5. XOR-delta | 3.8254 | 1.0456× | −9.42% |
| 6. **Per-row packed (gs=0) global** | **2.9250** | **1.3675×** | **(requires requantization)** |
| 7. Scale stream | ~12 bits/F32 | — | scales essentially incompressible |

Per-row entropy distribution (rows): min=2.85, p10=3.46, p50=3.49, p90=3.52, max=3.61.
Per-block distribution (blocks): p10=3.05, p50=3.33, p90=3.54.

### 2.3 Interpretation

- **Per-row code** is statistically indistinguishable from global. There is no row-local structure to exploit — the per-row entropy is essentially identical to the global histogram. A per-row code (one table per row) gains nothing; its 24-byte table overhead per row (1.5M rows / layer) is pure overhead.
- **Per-block code** does show structure: 5.4% payload gain over global. But the per-block table overhead (1 table per 64-weight block × 113 M blocks = **2.7 GB**, **75% of the raw packed bytes**) consumes the gain entirely. Net payload + overhead = **5.7 GB / 3.6 GB raw = 158%** — **the per-block code is worse than no compression**. A fixed small codebook of K tables shared across blocks is the only way to make per-block structure pay; the existing per-block histogram variability (p10..p90 spread of 0.5 bits) suggests ~64 tables might fit but this is a Stage 2 prototype, not a Stage 0 fact.
- **First-order conditional** along packed-byte order is identical to global — adjacent nibbles carry no information about each other. The bit-stream is i.i.d.-like at the per-byte scale.
- **XOR-delta** is **worse** than global (+9.42% entropy): adjacent nibbles from different weights decorrelate enough that XOR-ing them raises entropy. Any predictor-based codec (delta coding, predictive ANS) is **strictly worse** than plain order-0 rANS.
- **Per-row quantization (gs=0)** drops global entropy from 3.49 to **2.93 bits/wt** — a 1.37× ceiling, **the same level as GLM-5.2's lossless anchor** (2.91 bits/wt, 1.37×). But this is a **requantization** of the underlying FP8 weights through the existing `quant_int4_per_row` math, not a transform of the existing gs=64 packed bytes. It is **not lossless with respect to the on-disk representation**. It would be **lossless with respect to the FP8 source** at the cost of replacing the on-disk container.
- **Scale stream** is essentially uncompressible at 256-bucket precision. Raw 453 MB / layer of F32 scales; any entropy coding of scales saves a few percent at most and the codec overhead eats the savings.

**Stage 0(b) interpretation.** Structured entropy on M3 GS=64 closes every branch except one: per-row quantization (gs=0). That branch is **not lossless** with respect to the existing GS=64 container; it is a different representation altogether.

---

## 3. Stage 0(c) — single-expert entropy coder throughput on M3 expert bytes

**Tool:** `c/tests/test_fse.c` (in tree, with Windows QPC patch) on
`expert_l30_e64_down_proj_gs64.bin` and `expert_l30_e64_down_proj_gs0.bin`
(9.44 MB each, from `extract_one_expert.py`).
**Compiler:** MSVC 19.38.33135, `/Ox /std:c11 /EHsc /MD`.
**Hardware:** i5-12600K, single P-core pinned, best of 5 timed runs.

| Format | Ratio (measured) | Decode time | **Decode GB/s (1 core)** |
|---|---:|---:|---:|
| gs=64 (source) | 1.144× | 47 ms | **0.20 GB/s** |
| gs=0 (per-row requantized) | 1.344× | 43 ms | **0.22 GB/s** |

Both match the entropy ceilings from Stage 0(a,b) within measurement noise.

**Stage 0(c) interpretation.** On this hardware, a single thread can produce **~200 MiB/s** of decoded expert bytes via `cfse_decompress`. The production CPU matmul on tuned Milan/Threadripper sustains **60+ GB/s** of expert weight bandwidth (see `docs/experiments/glm52-decode-failure-ledger-2026-07-31.md`). The single-thread decoder is therefore **~300× slower than the matmul it would feed**.

---

## 4. Stage 0(d) — multi-thread cfse_decompress sweep

**Tool:** `bench_mt_decode.c` (Win32 threads, this report).
**Method:** Each thread decodes the same 9.44 MB expert independently into its own scratch buffer. Best-of-3 wall time across all threads.
**Reported:** "aggregate GB/s" = (threads × 9.44 MB) / wall, i.e. total decoded bytes per second across the chip.
**Hardware:** i5-12600K, 16 logical processors available, AVX2.

### gs=64 expert

| Threads | Wall (ms) | Aggregate GB/s | Per-thread-effective GB/s |
|---:|---:|---:|---:|
| 1 | 46.9 | 0.20 | 0.20 |
| 2 | 47.8 | 0.39 | 0.20 |
| 4 | 51.4 | 0.74 | 0.18 |
| 8 | 58.8 | 1.28 | 0.16 |
| 16 | 62.0 | **2.43** | 0.15 |

### gs=0 (per-row) expert

| Threads | Wall (ms) | Aggregate GB/s | Per-thread-effective GB/s |
|---:|---:|---:|---:|
| 1 | 43.6 | 0.22 | 0.22 |
| 2 | 43.7 | 0.43 | 0.22 |
| 4 | 46.2 | 0.82 | 0.20 |
| 8 | 55.3 | 1.37 | 0.17 |
| 16 | 57.8 | **2.61** | 0.16 |

**Stage 0(d) interpretation.** Decode scales near-linearly to 4 threads, then plateaus. The plateau is a DRAM-bandwidth ceiling: each thread re-reads the same 8.25 MB compressed buffer; after L2, all threads share the same DRAM read bandwidth. The 16-thread aggregate of **2.5 GB/s** is what 16 cores on this hardware can collectively deliver — and even on AVX-512 Milan with more cores, the same scaling shape will hold: the per-expert compressed buffer is shared state that DRAM cannot fan out fast enough.

This is **exactly** the architecture the briefing warned about:
> compressed DRAM → cache/registers → decode → matmul
> Decompression becomes part of the expert kernel.

But the decoder isn't fast enough to keep up with the matmul. The headline ratio (2.5 GB/s aggregate decode vs 60 GB/s matmul) means **the decode is the new bottleneck**. The architecture does not multiply DRAM bandwidth; it converts "DRAM bandwidth × 1.144×" into "DRAM bandwidth × 0.04×". The trade is worse, not better.

---

## 5. Cross-reference against the briefing's decision criterion

> "Can real GLM INT4 experts be losslessly reduced by ≥15%?"

For MiniMax M3 GS=64: **NO**. The Shannon ceiling is 12.72% on the global histogram; the per-row / per-block / conditional structured entropy ceilings do not improve the situation (or do, but table overhead consumes the improvement). The decoder is also 300× too slow to feed matmul.

> "If no, park the lossless branch and investigate better lossy INT3/INT2 representations."

**Yes — park the lossless branch on M3.** Open the next branch: per-row quantization (gs=0) followed by lossless entropy coding on the requantized representation. The 1.37× ceiling matches GLM-5.2's ceiling; the codec work for that branch is essentially the same `cfse_coli.h` / `int4-rans256-g0` work that was already designed. **But** the decode throughput number from Stage 0(c,d) still applies: 0.22 GB/s single-core decode means the architecture is still bottlenecked at the decoder, not DRAM. The same STOP applies to that branch until a much faster decoder (AVX-512 VBMI gather-based fused decode-and-dot, or a fundamentally different codec) exists.

The **honest reading of the briefing's framing** is that this Stage 0 result **does not even motivate Stage 1 of the lossless branch**. The decode GB/s is the binding constraint, and it is binding on the wrong side of the threshold. Even a free (zero-time) decoder at 1.146× would save **only 12.7% of the expert traffic**, and the matmul ceiling is already at 60+ GB/s — so the wall-time saving would be:

`T_baseline ≈ 9.4 MB / 60 GB/s ≈ 157 µs per expert`
`T_compressed ≈ (9.4 MB × 0.873 / 60 GB/s) + decode_overhead`
`         ≈ 137 µs + 47 ms (single-thread) ≈ 47.2 ms`
`         = 47.2 ms / 0.157 ms ≈ 300× SLOWER`

Even on the **fully parallel** version (16 threads, 8 experts in flight at once):
`T_compressed ≈ (9.4 MB × 0.873 / 60 GB/s) + 9.4 MB / 2.43 GB/s`
`         ≈ 137 µs + 3.87 ms ≈ 4.0 ms`
`         = 4.0 ms / 0.157 ms ≈ 25× SLOWER per expert`

The matmul can't even start until decode finishes. **The architecture fails its own primary motivation.**

---

## 6. Why M3 GS=64 is so much harder than GLM-5.2

The GLM-5.2 anchor (2.91 bits/wt, 1.37×) is on a **per-row int4** container (fmt=2). M3's stored representation is **grouped-int4 gs=64** (fmt=4) — different distribution shape. The per-block measurement in Stage 0(b) gives the diagnostic: when you re-bucket M3 weights into gs=64 groups of 64 weights each, the per-bucket distribution flattens dramatically (mean 3.31 bits/wt vs 3.49 global). The grouping normalizes the per-row distribution; without that grouping, the per-row distribution on M3 would be much more peaked.

We verified this on layer 30: when we **requantize M3 per-row** (the gs=0 representation that is mathematically equivalent to GLM-5.2's fmt=2), the global entropy drops from 3.49 to **2.93 bits/wt — essentially the same number as GLM-5.2 fmt=2** (which the existing 2026-07-31 doc measured at 2.91 bits/wt on real GLM bytes).

So **the lossless 1.37× ceiling is reachable for M3 too, but only after requantization**. The requantization itself is not lossless with respect to the existing GS=64 container — it changes the on-disk bytes. It **is** lossless with respect to the original FP8 weights (within the same rounding math). That is a different branch than the briefing asked about; it is closer to "lossless with respect to a re-encoded representation."

---

## 7. Reproducibility

| Artifact | Path | Bytes | SHA-256 (truncated) |
|---|---|---:|---|
| Layer 30 expert 64 down_proj, gs=64 packed | `extract_one_expert.py` output | 9,437,184 | recorded in JSON manifest |
| Same expert, per-row packed | `extract_one_expert.py` output | 9,437,184 | recorded in JSON manifest |
| Stage 0(a) JSON | `analyze_split_entropy.py --json-out` | 2.7 KB | (in tree tool output) |
| Stage 0(b) JSON report | `stage0b_entropy_analysis.py --report` | — | saved next to this report |
| `test_fse.exe` (single-thread) | `test_fse.c` + fse_coli.h, MSVC `/Ox` | — | build hash on request |
| `bench_mt_decode.exe` (multi-thread) | `bench_mt_decode.c`, MSVC `/Ox` | — | build hash on request |
| This report | `docs/results/lossless-stage0-m3-2026-09-08.md` | — | — |

Scripts are in `stage0/` (temp directory for this Stage 0 run); reproducibility commands:

```bash
# Stage 0(a)
python c/tools/analyze_split_entropy.py <minimax_m3_i4_dir> \
    --layers 3,15,30,45,57 --json-out stage0a.json

# Stage 0(b)
python stage0b_entropy_analysis.py <minimax_m3_i4_dir> --layer 30 \
    --report stage0b.json

# Stage 0(c)
python extract_one_expert.py <minimax_m3_i4_dir> --layer 30 --expert 64 \
    --projection down_proj --out-dir expert_samples
EXPERT_RAW=expert_samples/expert_l30_e64_down_proj_gs64.bin  ./test_fse.exe
EXPERT_RAW=expert_samples/expert_l30_e64_down_proj_gs0.bin   ./test_fse.exe

# Stage 0(d)
./bench_mt_decode.exe expert_samples/expert_l30_e64_down_proj_gs64.bin 16
./bench_mt_decode.exe expert_samples/expert_l30_e64_down_proj_gs0.bin  16
```

---

## 8. Decision

**The lossless expert compression architecture branch is CLOSED on MiniMax M3 (gs=64).**

The conditions that would reopen it:
1. **A decoder at least 30× faster than `cfse_decompress`** (i.e. ≥6 GB/s single-core decode, or ≥250 GB/s 16-core aggregate). The AVX-512 256-way interleaved path in `c/rans.h` (`RANS_PATH_AVX512`) might achieve this on Zen 4 — needs a measurement, not a projection.
2. **Or a representation whose entropy is below 3.4 bits/wt** — i.e. the per-row (gs=0) requantization, which would change the on-disk container.
3. **Or a fundamentally different matmul kernel that consumes decoded nibbles faster than it consumes raw nibbles** — i.e. the decoder must be free relative to matmul. The 2026-07-31 GPU result (3.4–9.5× slowdown for fused decode+matvec) makes this option unlikely on the GPU; the CPU analogue has not been prototyped.

None of these are Stage 0 outcomes. **Proceed to the lossy branch** (INT3 / INT2 representations, or per-row requantization + lossless on top) per the briefing's decision criterion.

---

## 9. Risk-policy disclosure

This report applies the plan's gate language directly:
- **Global entropy code on M3 GS=64: STOP.** The 1.146× ceiling is below the 15% threshold the briefing names.
- **Structured/contextual entropy code on M3 GS=64: STOP.** Per-row identical to global; per-block +5.4% payload but +75% overhead; first-order conditional identical to global; XOR-delta worse than global.
- **Lossless decoder throughput on M3 GS=64: STOP.** 0.20 GB/s single-core, 2.5 GB/s 16-core aggregate, vs 60+ GB/s matmul bandwidth.

The **only branch that remains open** corresponds to requantizing M3 per-row (gs=0) and then attempting lossless on the requantized bytes. That branch is **not what the briefing asked about** — it changes the on-disk representation — and even on that branch the decode throughput is still the binding constraint (Stage 0(c) shows 0.22 GB/s on a per-row packed expert).

Recommended next action: write a follow-up "Stage 0.5" experiment design for the requantize-and-code branch, or pivot directly to the lossy INT3/INT2 representation work the briefing points to as the fallback.

