# Machine A disjoint overlapped direct-I/O calibration v2

Date: 9 September 2026. This is machine-specific shadow evidence; it is not
an active admission approval.

## Capture

- Backend: Windows `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`
  (`win32-overlapped`).
- Cache condition: `cold` requested through direct/unbuffered reads.
- Expert request: `31,850,496` bytes.
- Scalar levels: 1, 2, 4, 8, 16.
- Shared-upstream matrix levels: 0, 1, 2, 4, 8, 16 per drive, bounded by
  total inflight 16.
- Matrix cells: 26; cells with both drives active: 16.
- Samples: 8; operations per worker: 2.
- Range reuse: **false across the complete capture**. Each drive required
  2,448 virtual read slots and had 2,556 available slots across 18 shards.
- Quantiles: conservative nearest-rank over request completions.

## Observed results

- Controller 201 review candidate: inflight 2 at approximately `5.375 GB/s`.
- Controller 202 review candidate: inflight 4 at approximately `3.112 GB/s`.
- Shared upstream 901 produced no candidate under the observed p95 <= 2x
  baseline plus retention/fairness policy; its rate remains UNKNOWN.
- Review status: `CANDIDATE_ONLY`; active admission remains disabled.
- Direct read-to-ready endpoint: `OBSERVED_NOT_ATTESTED`; the destination is
  an aligned reusable buffer, but hardware DMA is not independently attested.
- Host memory result remains `host_memcpy` evidence, not DMA evidence.
  The measured host-copy rates are approximately 8.298 GB/s p50 and 8.408
  GB/s p95 for the 256 MiB buffer.
- PCIe endpoint observations: controller 201 is negotiated Gen4 x4
  (approximately 7.877 GB/s payload upper bound), controller 202 is negotiated
  Gen3 x4 (approximately 3.938 GB/s payload upper bound), and the GTX 1080
  endpoint is observed separately. These are negotiated-link observations,
  not PCH/DMI traffic measurements.
- Windows device-tree evidence records distinct root ports for controllers 201
  and 202, with a common PCI root-complex ancestor (`ACPI\\PNP0A08\\0`).
  That supports the topology declaration behind upstream 901, but does not
  turn the root-complex/DMI capacity into a measured rate.
- Volume extents map the C: source volume for drive 101 to physical disk 1 and
  the E: source volume for drive 102 to physical disk 0; this is disk-identity
  evidence, not a controller/PCH bandwidth measurement.
- NUMA inventory reports one OS-visible node (`node 0`); no NUMA placement
  penalty is inferred from that inventory result.

## Generated artifacts

- Raw capture: `m3-machine-A-disjoint-calibration-v2.json`
- Planner text: `m3-machine-A-disjoint-calibration-v2.planner.txt`
- Review: `m3-machine-A-disjoint-calibration-v2.review.json`
- Reviewed candidate planner: `m3-machine-A-disjoint-candidate-planner-v1.txt`
- Machine profile: `m3-machine-A-disjoint-profile-v2.json`
- Full model configuration: `m3-machine-A-disjoint-full-model-config-v2.txt`
- Candidate full model configuration: `m3-machine-A-disjoint-candidate-full-model-config-v1.txt`
- Source manifest: `m3-machine-A-disjoint-calibration-manifest-2026-09-08.json`

The C planner parser passed with 10 topology resources and zero raw profile
rows; the full-model generator passed with 7,296 experts and 14,592 copy rows.
The profile retains CPU, memory, PCH-link, and GPU declarations; the PCH/DMI
traffic counter is explicitly `UNAVAILABLE` on this installation, while endpoint PCIe negotiation and
the parent device chain are recorded as separate evidence. Controller service
and shared-upstream observations are evidence-only; shared-controller sharing
is `NOT_APPLICABLE` because each controller has one declared drive. The GTX
1080 and Intel UHD transfer tier are explicitly ineligible, so GPU transfer is
`NOT_APPLICABLE` rather than an unperformed required measurement.
The separate reviewed candidate emitter produced 4 candidate profile rows; its
configuration remains explicitly `candidate_only` and inactive.

## Remaining gates

- The stronger-model review completed and kept the candidate evidence-only;
  held-out planner validation remains open.
- Storage-to-memory hardware-DMA attestation and PCH/DMI traffic measurement
  remain unavailable/unverified; the direct read-to-ready path is only observed.
- The authoritative backend uses Windows OS-overlapped I/O;
  `asynchronous=true` is preserved in the artifacts. This still does not attest
  the hardware DMA engine.
- No measured candidate has been promoted into active planner bounds; raw
  calibration output is intentionally topology-only.
