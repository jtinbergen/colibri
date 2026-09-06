# M3 execution-DAG step 3: contracts and serial reference executor

Date: 5 September 2026.

## Scope

`COLI_M3_DAG_SERIAL=1` enables a bounded, one-thread reference executor for
the MiniMax-M3 CPU path only. It is eligible only for `S=1`, CPU execution,
grouped int4 with group size 64, no PIPE/URING/PILOT_REAL/cluster/draft route,
and no ablation. Every other mode takes the existing path before tasks are
published.

The executor preserves `uniq[]` traversal order and the existing weighted
accumulation expression. It runs the shared expert as its final task using
the existing gate/up/activation/down sequence. It does not add workers,
change placement, alter routing, or select a different kernel.

## Contract

[`m3_dag.h`](../../c/m3_dag.h) defines:

- `M3DagExecutionContext`: model, forward, layer, generation, resident budget,
  cancellation, and failure state.
- `M3DagExpertTask`: immutable identity, exact routed index or shared kind,
  opaque weight handle, input/scratch/output, independent weight and compute
  states, a compute lease, and an exactly-once output-commit bit.
- Monotonic weight states `absent -> queued -> loading -> resident` (or
  failed) and compute states `waiting -> ready -> running -> complete` (or
  failed/cancelled).
- A serial executor that acquires the routed `ESlot` before compute and
  releases it only after reduction commits. Failed contexts cannot start a
  later prepared task; terminal tasks cannot be cancelled or committed again.

The M3 adapter uses the existing `ESlot.in_flight` reference counter for the
temporary routed lease. Promotion occurs only after task retirement. A
partial task failure terminates instead of retrying the old forward after an
already-committed output.

## Evidence

Machine: Windows, i5-12600K-class host; MSYS2 MinGW GCC/OpenMP build.

```text
make colibri.exe tests/test_m3_dag.exe tests/test_pipe_block.exe
./tests/test_m3_dag.exe
test_m3_dag: ok (serial order, shared once, stale, cancel, failure)

./tests/test_pipe_block.exe
test_pipe_block: ok

/c/Python313/python.exe tests/test_m3_tiny.py \
  --binary ./colibri.exe --snap ./m3tiny_i8 --ref ./ref_m3.json
m3-tiny-check: OK (prefill 24/24, decode 20/20)
```

The lifecycle test also asserts that a later ready task cannot execute after a
prior task fails. The tiny oracle uses int8 experts, so it verifies fallback,
not grouped-int4 executor activation.

For the installed grouped-int4 model, matched CPU-only runs used:

```text
SNAP=/c/Users/jaapj/.lmstudio/models/minimax_m3_i4
PROMPT=hi NGEN=1 TEMP=0 PIPE=0 RAM_GB=18 COLI_RAM_OVERCOMMIT=1 AUTOPIN=0
./colibri.exe 2 4 8

# then add:
COLI_M3_DAG_SERIAL=1 COLI_M3_DAG_DIGEST=1
```

The flagged run reported `M3_DAG active`. The baseline and flagged runs
produced the same greedy token (`{tabular`) and identical FNV-64 byte hashes
for the outputs of all 57 sparse layers (layers 3 through 59) in the prefill
forward. The digest is test-only (`COLI_M3_DAG_DIGEST=1`) and hashes the
actual `float` layer output after routed plus shared reduction.

## Gate 3

| Gate | Status | Scope |
|---|---|---|
| C | PASS | Serial contract lifecycle test; tiny-M3 fallback oracle; real grouped-int4 one-token layer-output A/B over 57 sparse layers. |
| M | PASS | Real checkpoint printed the opt-in executor activation; task state/lifetime checks run before cache promotion. |
| P | INCONCLUSIVE | The matched one-token prefill probes measured 5.41 s baseline and 5.52 s with the serial contract (about 2% slower, one pair). This records contract cost only; it is neither a speed claim nor a promotion result. |

Limitations: this is a CPU/S=1/grouped-int4/g64 reference scope. PIPE,
URING, speculative/draft, cluster, device, cancellation of live I/O, and
parallel expert execution remain outside this step and are gated for later
steps. The full-model A/B covers one prefill token; longer replay and
mixed-residency/cache-swap matrices remain useful regression extensions, not
evidence for a speed claim.
