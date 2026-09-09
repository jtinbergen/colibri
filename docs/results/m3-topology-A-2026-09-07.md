# MiniMax-M3 topology capture — machine A (dev desktop)

Date: 7 September 2026. Status: machine A captured; machines B/C/D `NOT_RUN`.

Companion JSON: `m3-topology-A-2026-09-07.json`.

## Method

This artifact is data-model only. It collects existing evidence from
the workspace plus a live `Get-CimInstance` snapshot of the host. Every
measurement that has not actually been taken is labelled `NOT_RUN`. The
artifact does not claim a default capacity or extrapolate; it is the
substrate for Slice 7a-ext (data-model extension) and Slice 7b-ext
(calibration).

## Compute

- **CPU:** 12th Gen Intel Core i5-12600K (Alder Lake).
  - 10 physical cores (6 P-cores with HT + 4 E-cores), 16 logical threads.
  - ISA: SSE4.2, AVX, AVX2, FMA, BMI2.
  - Missing: AVX-512, AMX, VNNI, SVE/SVE2, i8mm.
  - Single socket, single NUMA node (desktop board).
  - Base 3.7 GHz, boost 4.9 GHz, 125 W PL1.
  - LGA1700, Gigabyte Z690 AORUS ELITE DDR4.
  - **CPU roofline:** theoretical peaks only (`avx2_fma ≈ 421 GFLOP/s`,
    `mem_bw ≈ 51.2 GB/s`); not measured on this host.

- **GPU:** NVIDIA GeForce GTX 1080 (Pascal, sm_61, 8 GB GDDR5X).
  - `eligible_for_expert_compute: false` on this machine.
    - Reason: consumer Pascal; engine's CUDA tier per base plan Rules 4/9
      does not include sm_61; existing Step 6 evidence on this host
      records `routed GPU critical = 0.000 s`.
  - Intel UHD Graphics 770 (iGPU): `eligible_for_expert_compute: false`
    (no expert compute kernel for iGPU in this engine).

## Memory

- **DRAM (mem:drama):** 32 GiB total (Win32 reports 34.1 GB; OS reports
  32 GiB available to user). 2-channel DDR4.
- VRAM: 8 GB on the GTX 1080 (not eligible for expert compute on A).
- NUMA attach: node 0; no NUMA on this board.

## Storage

| ID | Label | Model | Capacity | Controller | Lanes | Calibration |
|---|---|---|---|---|---|---|
| drive:101 | C: | Samsung 980 PRO | 1 TB | ctrl:201 | PCIe 3.0 x4 | QD1 3.41 GB/s, QD4 6.57 GB/s; p99 (QD4, 19 MiB) 20.8 ms |
| drive:102 | E: | SAMSUNG MZAL4512HBLU | 512 GB | ctrl:202 | PCIe 3.0 x4 | QD1 0.81 GB/s, QD4 1.00 GB/s; p99 (QD4, 19 MiB) 134 ms |
| drive:103 | F: | Kingston XS1000 | 1 TB | ctrl:usb | USB 3.2 | NOT_RUN |

The two NVMe drives hang off separate PCH root ports
(`PCI\VEN_8086&DEV_7AB0` and `PCI\VEN_8086&DEV_7AB4`); evidence for two
distinct NVMe controller paths. **Both NVMe controllers share the Z690
PCH as their upstream** — a contention scenario on the PCH link is
physically possible but not measured.

## Interconnects

- `pcie:z690-pch` — PCIe 3.0 PCH root, 28 lanes total. Shared group
  `pch:z690`. Both NVMe controllers attach here; a same-controller
  capture is `NOT_RUN` on this host (no two drives share a controller).
- `usb:3.2` — USB 3.2 Gen 2 for the external SSD.

## Software

- Windows 11 Pro build 26200. NTFS.
- MinGW GCC + OpenMP. CPU-only engine (CUDA/Metal/Vulkan/ROCm disabled).
- `OMP_NUM_THREADS=10` default; PIPE=1 PIPE_WORKERS=4 in the Step 0c
  run.

## Open evidence gaps

| ID | Field | Status |
|---|---|---|
| cal:qd8_drive101 | QD 8/16 random reads | NOT_RUN |
| cal:same_controller_contention | same-controller pair | NOT_RUN (no such pair on A) |
| cal:shared_upstream | PCH shared-upstream contention | NOT_RUN (no injection mechanism on A) |
| cal:cache_warm_cold | warm vs cold reads | NOT_RUN (`cache_condition: unknown` in Step 7 capture) |
| gpu:h2d_d2h | GPU H2D/D2H | NOT_RUN (GPU ineligible) |
| mem:bandwidth | DRAM bandwidth | NOT_RUN (theoretical peak only) |

## Honesty log

- Live CPU/board/drive discovery via `Get-CimInstance Win32_Processor`,
  `Win32_ComputerSystem`, `Win32_DiskDrive`, `Win32_VideoController` on
  the capture host.
- Calibration numbers copied verbatim from
  `docs/results/m3-execution-step7-calibration-2026-09-07.md` and
  `docs/results/m3-execution-step7-calibration-2026-09-07.json` and
  the latency tables in
  `docs/minimax-m3-comprehensive-report-2026-09-05.md`.
- Step 0c PROF counters were active during the C:-only run; QD
  histogram values are real, not extrapolated.
- No `units_per_second` invented; missing classes are `NOT_RUN`.
- Per-machine `eligibility` flags follow base plan Rules 4/9: CPU-only
  execution on machine A.
