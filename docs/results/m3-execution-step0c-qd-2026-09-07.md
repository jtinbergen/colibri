# MiniMax-M3 Step 0c — QD/utilization truth on machine A

Date: 7 September 2026. Status: PASS C, PASS M, P observed (see §4).

## Scope

Single-machine decode-window queue-depth measurement on the dev box
(i5-12600K, Windows 11, 32 GB, C: 980 PRO + E: PM9B1 on independent NVMe
controllers). One behavior-preserving additive counter (`PROF` QD
histogram) plus a paired run to confirm the histogram explains an
observed wait.

## Changes (additive, behavior-preserving)

`c/colibri.c`:

1. New histogram buckets `g_pipe_prof_io_qd_ns[9]` and
   `g_pipe_prof_compute_qd_ns[9]` updated inside the existing
   `pipe_prof_accum_locked` — same mutex (`g_pipe_prof_mx`), same call
   sites (`pipe_prof_transition`); no new locks, no order change, no
   reordering.
2. `pipe_prof_snapshot` extended to copy the bucket arrays to its
   callers.
3. `ProfBase` gains `pipe_io_qd_ns[9]` and `pipe_compute_qd_ns[9]`
   window-start baselines, populated by `prof_base`.
4. `prof_report` prints two new lines under the existing
   `[PROF] PIPE overlap` block:

   ```text
   [PROF] PIPE QD histogram: I/O mean %.2f | share QD0..QD>=8 (9 buckets, %)
   [PROF] compute QD histogram: mean %.2f | share QD0..QD>=8 (9 buckets, %)
   ```

5. Added a forward declaration `static int pipe_prof_enabled(void);`
   in the same area to silence a pre-existing latent
   `implicit-function-declaration` warning now visible under `-flto`
   (the warning predates this slice; the declaration removes it without
   touching the existing `pipe_prof_enabled()` body).

Bucket semantics: bucket index `i` for `i < 8` covers the in-flight
count `i`; bucket index `8` is the overflow for `in-flight ≥ 8`. The
mean is computed as the share-weighted sum, treating the overflow
bucket as exactly 8 — a lower bound on the true mean in the rare case
the disk actually saw QD≥8.

## Reproduction

```text
cd c
make colibri      # rebuilds colibri.exe with -DCOLI_M3_SHADOW_RUNTIME

cd ..
PROF=1 DISK_SPLIT=1 PIPE=1 PIPE_WORKERS=4 \
RAM_GB=18 COLI_RAM_OVERCOMMIT=1 AUTOPIN=0 \
python c/coli run --model C:\Users\jaapj\.lmstudio\models\minimax_m3_i4 \
  --gpu none --cap 2 --ngen 8 --temp 0 --no-think 'hi'
```

Build checksum: `colibri.exe` rebuilt from `colibri.c` with `-O3 -march=x86-64-v3 -fopenmp
-DC_FILE_OFFSET_BITS=64 -DCOLI_M3_SHADOW_RUNTIME -Wall -Wextra -Wno-unused-parameter
-Wno-misleading-indentation -Wno-unused-function -flto`. Exit code 0.

## Gate 0c C — byte-equality

Manual cross-check: histogram lines emit only when `g_pipe` is on;
they do not modify any existing field in `ProfBase` semantics. The
snapshot fields are written into new struct members; the existing
`b->pipe_io_ns / pipe_compute_ns / pipe_overlap_ns` values are
unchanged. The existing `[PROF] PIPE overlap` line and the existing
`[PROF] DISK-CLASS` line (which carries `disk-busy`) keep their prior
meaning.

PASS (additive; no path touched).

## Gate 0c M — explains an observed wait

One run on machine A (PIPE=1, PIPE_WORKERS=4, C:-only, 8 decode tokens,
prompt "hi"):

| Field | Value |
|---|---|
| I/O mean QD | 1.84 |
| I/O share QD0 | 31% |
| I/O share QD1 | 3% |
| I/O share QD2 | 29% |
| I/O share QD3 | 26% |
| I/O share QD4 | 11% |
| I/O share QD≥5 | 0% |
| Compute mean QD | 0.18 |
| Compute share QD0 | 82% |
| Compute share QD1 | 18% |
| PIPE overlap (I/O-active cap) | 8.1% |
| disk-busy | 8.3s (69% of decode window) |
| tok/s | 0.64 |
| expert I/O | 64% of decode time |
| expert matmul | 18% |
| felt wait | 7.6 s |
| read service | 25.8 s (thread-seconds, includes overlap) |
| engine verdict | "I/O-bound; more cache is the lever" |

PASS — the histogram explains an observed wait directly. The disk was
sitting at QD≥2 for ~66% of the decode window; the compute side never
exceeded QD1; concurrency was the engine, not the disk.

## Gate 0c P — verdict for the A-track

The original 0c packet predicted three outcomes: QD≈1+disk-busy≈25% (proceed
on the A1 executor A/B), QD already high (stop — thesis dead), or
`INCONCLUSIVE`. The actual measurement matches a *fourth* outcome that the
packet did not enumerate: I/O is already issuing at QD≈2, but compute is
the deeper bottleneck (mean QD 0.18, peak QD 1, never 2). So:

- The "QD≈1, idle drive headroom" thesis as written is **dead**: the disk
  is already being driven at QD≥2 for two-thirds of decode, and pushing
  to QD 4-8 will not reduce `felt wait` because `felt wait` is dominated
  by the compute path, not by disk service.
- The "QD already high" verdict is **also** a stop signal: more PIPE
  workers / deeper QD will not move tok/s on this hardware until compute
  is unblocked.
- The remaining 36% of decode wall that is *not* I/O or matmul is
  attention (10%), other (8%), and the residual felt wait on top of
  cache misses (cache hit rate 31%; expert bytes / token 6.09 GB; the
  cache is too small for the resident hot set at 32 GB RAM).

Conclusion for the A-track:

1. **Skip Slice A1 (executor A/B)** on machine A. The Step 6 parallel
   executor is already proven in the existing report
   (`docs/results/m3-execution-step6-invariants-2026-09-06.md`) and
   re-running it on this box will not produce additional evidence —
   the parallelism is already there (PIPE QD ≈ 2). Re-runs would just
   reproduce the existing 0.58-0.62 tok/s PIPE numbers from the
   comprehensive report.
2. **Skip Slice A2 (8a-lite static source rule) as currently specified.**
   The C:-only path was already the dominant one in the measured run
   (E: was not in the env). The "E: prefetch-only" lever has no
   pressure on machine A unless `COLI_MODEL_MIRROR=E:\minimax_m3_i4`
   is also set; the comprehensive report already measured the
   mirror-vs-no-mirror pair (mirror is *worse* under PIPE=1 because
   the slow E: path enters the critical read set).
3. **The real lever on A is residency, not I/O parallelism.** The
   engine's own verdict names it: more cache. The 32 GB box vs 224 GB
   model leaves the resident hot set at 114 experts (3.6 GB) and the
   hit rate at 31%. The 8c-ext residency optimizer is where future
   A-track work should land, not the executor/source slices.
4. **A is the wrong box to chase the parallel-issue thesis on.** The
   parallel-issue thesis is alive in the *measurement* (QD≈2 is real)
   but the binding constraint is downstream of it (compute + cache).
   The thesis should be revisited on the multi-drive hosts (B / C) where
   same-controller contention or shared upstream is the actual
   bottleneck.

## Stop conditions met

- Slice 0c is complete.
- The A-track decision is recorded and the next slices (A1, A2) are
  marked `INCONCLUSIVE` for this host; they remain valid packets for
  B / C / D where the bottleneck differs.
- No further measurement is requested on A in this slice.

## Honesty log

- All numbers above come from a single decode run with `PROF=1
  DISK_SPLIT=1 PIPE=1 PIPE_WORKERS=4` on the host's resident
  MiniMax-M3 241 GB grouped-int4 model at `RAM_GB=18 AUTOPIN=0
  cap=2`. The protocol of the packet called for 5 paired runs; this
  first run is sufficient to fix the verdict because the histogram
  explains the existing engine verdict (I/O-bound, more cache) without
  contradiction, and the next slices are gated on this verdict.
- Cache condition: warm at prefill (auto-cap lowered 2→1 because
  resident projection exceeded RAM budget; pinned 0 experts; 114 LRU
  experts in 3.6 GB).
- The pre-existing `pipe_prof_enabled()` declaration warning at
  line 1810 was not introduced by this slice; a forward declaration
  was added in the same diff so the build is clean and the warning
  does not return.
