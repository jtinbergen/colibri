# MiniMax-M3 Step 5 — row-kernel invariants

## Supported path

Step 5 adds an internal CPU-only reference path for one M3 expert whose gate,
up and down tensors are non-planar grouped int4 (`fmt==4`, `gs==64`).  It is
first called by the existing serial Step 3/4 executor only.  Other formats,
planar IDOT, GPU, URING, mmap, cluster and unsupported shape combinations stay
on the established path.

## Contract

The new primitives accept immutable matrix views, an explicit input, explicit
activation configuration, caller-owned gate/up/down buffers, and a half-open
output-row interval.  They do not allocate, create an OpenMP team, access
`g_pq`, use TLS quantization scratch, or update model profiling state.

For each matrix, the order inside one output row remains exactly the existing
grouped-int4 order: groups, SIMD body, horizontal sum/fused scale add, then
scalar nibble tail.  The legacy whole-matrix wrapper remains available and
retains its current OpenMP behavior.

Dependencies are explicit:

```
all gate/up row ranges -> all activation ranges -> down row ranges -> commit
```

No down tile may start before the whole activation vector for that expert is
complete.  Each executor task owns distinct scratch; output reduction remains
outside the kernel and in the established router order.

## Evidence required for promotion

- Existing whole-row kernel versus 2-, 3-, and irregular-tile execution,
  including an empty range and final one-row tile.
- Group and packed-nibble boundaries (`I=15/16/17`, `63/64/65`), zero and
  extreme nibbles, clipping boundaries, and distinct expert inputs.
- Independent grouped-int4 reference and M3 oracle, with runtime evidence that
  the g64 path was selected.
- Concurrent calls with separate inputs/weights/scratch and canaries, with a
  sentinel `g_pq` left unchanged.
- Single-kernel overhead only; no executor or scalability claim.

## Current gate record

- C (pass): `test_i4_grouped.exe` passes grouped-int4 reference cases,
  full-versus-irregular-tile bit equality for single and fused gate/up ranges,
  packed/group boundaries, zero/extreme nibbles, and SwiGLU clipping bounds.
  `test_m3_grouped_int4_reference.py` and `test_m3_tiny.py` pass. A real M3
  g64 ready-first prefill produced the established 57 sparse-layer FNV values.
- M (pass): two simultaneous full explicit g64 experts use distinct non-planar
  views, inputs, gate/up scratch, and output buffers; they match sequential
  references, preserve canaries and a `g_pq` sentinel, and use a broadcast start
  barrier so neither worker can be stranded.
- P (pass, narrow witness): `test_i4_grouped.exe` records the explicit g64
  expert kernel at about 3.1 us/call on this machine. This is a single-thread
  overhead witness only, not an end-to-end throughput or scalability claim.
