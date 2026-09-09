# MiniMax-M3 Step 7b — paired 16-sample calibration slice

## Status

The capture and review completed successfully. This is cross-controller
sensitivity evidence only: it is not a promoted planner profile, queue bound,
or Gate-P result.

## Matrix

- source drive `101` on `C:`, controller `201`;
- source drive `102` on `E:`, controller `202`;
- paired group `nvme-pair`, with the two drives on distinct controller paths;
- 16 samples at each inflight level `1,2,4,8,16`;
- cache condition recorded as `unknown`.

The p99/2x review reports the largest observed full-group levels as `2` for
drive `101`, `4` for drive `102`, and `4` for `nvme-pair`. These are
illustrative observed levels, not accepted admission limits.

## Evidence

- [raw capture](m3-execution-step7-calibration-pair-16-2026-09-07.json)
- [p99/2x review](m3-execution-step7-calibration-pair-16-review-p99-2x-2026-09-07.json)
- [manifest](m3-execution-step7-calibration-pair-16-manifest-2026-09-07.json)

No same-controller, shared-upstream, cache-controlled, six/seven-drive, or
all-resident performance claim is made. Those remain `NOT_RUN`.
