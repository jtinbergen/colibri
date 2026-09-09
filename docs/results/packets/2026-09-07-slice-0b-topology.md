# Packet: Slice 0b — per-machine topology capture schema

Deliverable: one topology-capture artifact shape, instantiated for machine A
first. B / C / D stay `NOT_RUN` until measured.

## 1. Scope and invariants

- Data-model only. No engine code change.
- One schema, four instantiations (A / B / C / D).
- Output is human-readable markdown + a structured JSON companion for tooling.
- All numeric fields carry explicit units (`/s`, `bytes`, `lanes`, `MT/s`,
  `ns`, `°C`). No implicit unit conversion.
- Cache condition labelled per measurement; `unknown` is allowed but must
  not be promoted to default.

## 2. Topology schema

For each machine, capture:

### Compute

- CPU: model, ISA extensions (AVX1 / AVX2 / FMA / AVX-512 / VNNI / SVE2 /
  i8mm), physical cores, logical cores, socket count, NUMA domains.
- Roofline: per-instruction peaks where measurable; if not measured, mark
  `NOT_RUN` and never extrapolate.

### Memory

- Host RAM: total, per-NUMA-node, channel count, speed class.
- VRAM (if any): per-GPU total, speed class.
- Memory domains: cross-socket penalty measured or `NOT_RUN`.

### Storage

- Per drive: model, capacity, controller, parent PCI id, lane count, gen.
- Per drive: tail-curve class label (`cold sequential`, `warm random`,
  `competing reader`) at 1/2/4+ inflight — measured or `NOT_RUN`.
- Upstream link (if shared across controllers): name and evidence class.

### Interconnect

- PCIe lanes per GPU vs root competition, measured or `NOT_RUN`.
- P2P class per GPU pair, measured or `NOT_RUN`.
- H2D / D2H per GPU, measured or `NOT_RUN`.
- QPI / UPI penalty (only on B), measured or `NOT_RUN`.

### Software

- OS, kernel, CUDA / ROCm / Metal / Vulkan versions if any backend is
  enabled.
- Build flags and compiler that produced the engine binary.

## 3. Inventory of existing evidence

- A: Windows topology PCI ids in
  `docs/results/m3-execution-step7-calibration-2026-09-07.md:32-46`. Cache
  state `unknown`. No same-controller / shared-upstream / per-island
  conversion / memory-bandwidth curves.
- B / C / D: no machine evidence captured. All sections `NOT_RUN`.

## 4. Tasks

1. Fill in section §2 for machine A using the existing calibration doc and
   `docs/minimax-m3-comprehensive-report-2026-09-05.md`. Anything not yet
   measured stays `NOT_RUN`.
2. Save as `docs/results/m3-topology-A-<date>.md` and
   `docs/results/m3-topology-A-<date>.json`.
3. Produce empty-but-typed templates for B / C / D under
   `docs/results/m3-topology-B.template.md` etc., so a future capture is
   mechanical.

## 5. Gate 0b

- C: schema compiles into the validator script (separate packet, not part
  of this slice).
- M: every numeric field has explicit units; every missing measurement is
  labelled `NOT_RUN`.
- P: no perf claim. The schema is the substrate for Slice 7b-ext.
