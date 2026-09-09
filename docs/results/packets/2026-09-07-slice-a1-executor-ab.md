# Packet: Slice A1 — executor A/B at equal residency (machine A)

Deliverable: a paired A/B timing report for old vs Step 4 vs Step 6 on
machine A. No new code.

## 1. Scope and invariants

- Use existing flags only: `COLI_M3_DAG_SERIAL=1`, `COLI_M3_DAG_PIPE=1`,
  `COLI_M3_DAG_PARALLEL=1`, `COLI_M3_DAG_WORKERS=N`, `PIPE=1`.
- No scheduler, kernel, or placement change.
- 57-digest equality must hold (`0b5dca1f6fa0cd96` …
  `c7cd7ef0e12065bb`) across all configurations, per
  `docs/results/m3-execution-step7-m3-digest-replay-2026-09-07.md`.
- All-resident test stays `NOT_RUN` (32 GB vs 224 GB model).

## 2. Tasks

### 2.1 Prescribed runs

C:-only, paired A/B, 5x alternating per configuration. Fixed prompt + 32
decode tokens.

| Label | Flags |
|---|---|
| old | `PIPE=0` |
| old PIPE=1 | `PIPE=1 PIPE_WORKERS=4` |
| Step 4 | `PIPE=1 PIPE_WORKERS=4 COLI_M3_DAG_SERIAL=1 COLI_M3_DAG_PIPE=1` |
| Step 6 w=1 | Step 4 + `COLI_M3_DAG_PARALLEL=1 COLI_M3_DAG_WORKERS=1` |
| Step 6 w=2 | Step 4 + `COLI_M3_DAG_PARALLEL=1 COLI_M3_DAG_WORKERS=2` |
| Step 6 w=4 | Step 4 + `COLI_M3_DAG_PARALLEL=1 COLI_M3_DAG_WORKERS=4` |

`PROF=1 DISK_SPLIT=1` on every run. Caches labelled.

### 2.2 Evidence

- 57-digest equality check per run (PASS / FAIL).
- Time shares from `prof_report`: expert-I/O %, expert-matmul %, attention %.
- Concurrent I/O+compute wall from `pipe_prof_overlap_ns`.
- Felt wait, tok/s, RSS, bytes per run.
- 5-run summary table per label: median + min/max.

### 2.3 Artifact

`docs/results/m3-execution-slice-a1-executor-ab-<date>.md`.

## 3. Gate A1

- C: all 57 digests equal across all labels (existing Step 6 evidence covers
  worker counts 1/2/4/10; this slice reproduces at least 1/2/4).
- M: `pipe_prof_overlap_ns` > 0 in PIPE=1 labels.
- P: paired 95%-interval excludes zero on at least one of
  (Step 6 vs old PIPE=1) or (Step 4 vs old PIPE=1). Otherwise `INCONCLUSIVE`
  within budget; no selective runs.

Stop condition: Windows thread-pool overhead or compute flip caps the win
at ~2 tok/s on AVX2 Alder Lake → record as ceiling, do not bank the
predictor to chase a stalled mechanism.
