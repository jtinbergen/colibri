# MiniMax-M3 Step 7 — bounded read calibration capture

## Scope

This is a real Windows capture on the current desktop, not a planner-profile
approval. The host exposes two NVMe devices through two distinct Standard NVM
Express Controller instances, plus a USB SSD. The capture used equivalent
16 MiB ranges from the same M3 shard copied to the two NVMe volumes:

- resource `101`: `C:` / Samsung SSD 980 PRO, controller `201`;
- resource `102`: `E:` / Samsung MZAL4512HBLU, controller `202`.

The raw results are [64 KiB](m3-execution-step7-calibration-64k-2026-09-07.json),
[1 MiB](m3-execution-step7-calibration-1m-2026-09-07.json), and
[16 MiB](m3-execution-step7-calibration-2026-09-07.json).
An additional 16 MiB capture records the same matrix with an explicit p99
summary: [p99 capture](m3-execution-step7-calibration-p99-2026-09-07.json).
That capture has two samples per point, so its p99 is exploratory evidence,
not a statistically adequate production-tail guarantee.
The follow-up [16-sample p99 capture](m3-execution-step7-calibration-p99-16-2026-09-07.json)
reduces that sampling weakness. Its matching [2x sensitivity review](m3-execution-step7-calibration-review-p99-16-2x-2026-09-07.json)
finds the largest observed full-group level within the illustrative bound at
inflight 2 for resource 101 and inflight 4 for resource 102 and the pair.
These remain review outputs, not accepted admission limits.
The manifest format is demonstrated in
[m3-execution-step7-calibration-manifest.example.json](m3-execution-step7-calibration-manifest.example.json).
The follow-up [inflight-8/16 capture](m3-execution-step7-calibration-inflight-8-16-2026-09-07.json)
extends the observed matrix to the plan's next bounded levels with 16 samples
per cell. A combined [p99 2x review](m3-execution-step7-calibration-review-p99-2x-1-16-2026-09-07.json)
reports only observed candidates; it remains sensitivity evidence, not an
accepted queue bound.
The paired [16-sample capture](m3-execution-step7-calibration-pair-16-2026-09-07.json)
repeats the two-drive cross-controller matrix at inflight 1/2/4/8/16. Its
[p99 2x review](m3-execution-step7-calibration-pair-16-review-p99-2x-2026-09-07.json)
selects only the largest observed full-group levels (2, 4, and 4 for the two
drives and pair); these are sensitivity results, not controller/upstream
admission bounds and are not promoted.
The follow-up [three-drive capture](m3-execution-step7-calibration-3drive-2026-09-07.md)
adds the Kingston USB model copy as drive 103 and measures all singles, pairs,
and the triple at inflight 1/2/4/8. It confirms a materially slower USB tail
without inferring a controller or upstream relationship.

## Windows topology evidence

The volume-to-device mapping was checked separately with Windows storage and
PnP properties on the capture host:

- `C:` is disk 1, the Samsung 980 PRO, whose NVMe parent is
  `PCI\VEN_144D&DEV_A80B&SUBSYS_A80B144D&REV_02\4&1e1f98a2&0&00E8`
  and whose PCI parent is
  `PCI\VEN_8086&DEV_7AB0&SUBSYS_50011458&REV_11\3&11583659&0&E8`.
- `E:` is disk 0, the Samsung MZAL4512, whose NVMe parent is
  `PCI\VEN_144D&DEV_A80A&SUBSYS_A801144D&REV_00\4&21705936&0&00EC`
  and whose PCI parent is
  `PCI\VEN_8086&DEV_7AB4&SUBSYS_50011458&REV_11\3&11583659&0&EC`.

The differing Intel PCI parent devices are evidence for two distinct NVMe
controller paths on this host; they are not evidence about an upstream
motherboard link. No shared-controller or shared-upstream contention claim is
made, and those measurements remain `NOT_RUN`.

## Reproduction

```text
$env:COLI_M3_C_EXPERT='C:\Users\jaapj\.lmstudio\models\minimax_m3_i4\out-00002.safetensors'
$env:COLI_M3_E_EXPERT='E:\Models\minimax_m3_i4\out-00002.safetensors'
python c/tools/m3_shadow_calibrate.py \
  --manifest docs/results/m3-execution-step7-calibration-manifest.example.json \
  --bytes 16777216 --samples 2 --inflight 1,2,4 \
  --cache-condition unknown \
  --output docs/results/m3-execution-step7-calibration-2026-09-07.json
```

The runs completed with no short reads. Each measured every single-drive
group and the two-drive group at inflight 1/2/4 for one request-size class.
The JSON preserves every sample, per-worker latency, aggregate bytes/second,
source path/size, topology hash, and manifest hash, plus the active-drive set
for each group measurement. The pair's inflight-1 row is explicitly marked
as partial-group activity; the inflight-2 and inflight-4 rows exercise both
NVMe drives.

The latency distribution is therefore available to the planner calibration
review as p50/p95/p99/max evidence when the capture has enough samples. Once a profile is approved, its selected
`residual_ns` must carry an explicit tail label (for example `950000` for
p95); the planner uses that residual as an uncertainty margin in deadline
scoring and includes the label in its profile hash.

The review helper
[`m3_shadow_calibration_review.py`](../../c/tools/m3_shadow_calibration_review.py)
requires an explicit allowed p95 or p99 multiplier and reports only the largest
observed full-group inflight level within that bound. The illustrative 2x
sensitivity output is
[here](m3-execution-step7-calibration-review-2x-2026-09-07.json); it is not an
accepted queue bound because the 2x threshold was not a pre-registered product
criterion and the capture has unknown cache state.
The corresponding p99 2x sensitivity output is
[here](m3-execution-step7-calibration-review-p99-2x-2026-09-07.json); it has the
same illustrative, non-promotional status.
The paired 16-sample review is
[here](m3-execution-step7-calibration-pair-16-review-p99-2x-2026-09-07.json);
it has the same status and does not establish shared-controller or upstream
capacity.

## Interpretation limits

- Cache state was not flushed or classified; it is recorded as `unknown`.
  The run is therefore not a cold-read latency claim.
- The two NVMe devices are separate controller instances, so this capture
  measures independent-controller concurrency and a cross-controller pair.
  It does not measure two drives contending behind one controller or a shared
  upstream link.
- The tool records measurements but does not synthesize `profile` rows. A
  conservative calibration review must choose supported size/load classes,
  residuals, and admission bounds from a larger repeated matrix.
- In-flight levels are observed concurrency levels, not automatically a safe
  maximum queue depth. `max_inflight` remains an explicit admission bound
  chosen from the latency curve and validated against the measured classes.
- The current machine does not provide the plan's six/seven-drive,
  three-controller target topology. That hardware contention and all-resident
  performance gate remain `NOT_RUN`.

This closes the locally executable 7b instrumentation/evidence slice, not the
full Step-7 Gate-M or Gate-P requirements.
