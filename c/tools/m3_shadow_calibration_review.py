#!/usr/bin/env python3
"""Review bounded calibration captures for queue and tail-latency evidence.

This tool is deliberately a review aid, not a profile generator.  It accepts
one or more raw JSON captures from m3_shadow_calibrate.py and requires the
reviewer to state an allowed p95 or p99 latency multiplier.  It reports the
largest *observed* inflight level that stays within that bound, along with
p50/tail/max spread.  It never extrapolates beyond an observed level and
never writes an active planner configuration.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path
from typing import Any


REVIEW_SCHEMA = "m3-shadow-calibration-review-v2"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"m3-shadow-calibration-review: {message}")


def capture_hash(capture: dict[str, Any]) -> str:
    """Return the canonical digest of a capture without its self-digest."""
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_capture(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read capture {path}: {exc}")
    if (not isinstance(value, dict)
            or value.get("schema") not in ("m3-shadow-calibration-v1",
                                            "m3-shadow-calibration-v2")):
        fail(f"{path}: unsupported capture schema")
    if not isinstance(value.get("read_bytes"), int) or value["read_bytes"] <= 0:
        fail(f"{path}: invalid read_bytes")
    if not isinstance(value.get("measurements"), list) or not value["measurements"]:
        fail(f"{path}: missing measurements")
    if value.get("schema") == "m3-shadow-calibration-v2":
        actual = value.get("capture_sha256")
        if not isinstance(actual, str) or actual != capture_hash(value):
            fail(f"{path}: capture_sha256 is missing or invalid")
    return value


def as_int(row: dict[str, Any], key: str, path: Path) -> int:
    value = row.get(key)
    if not isinstance(value, int) or value < 0:
        fail(f"{path}: measurement field {key} is invalid")
    return value


def review(captures: list[dict[str, Any]], multiplier: float,
           tail_percentile: int, min_per_drive_retention: float = 0.0,
           min_fairness: float = 0.0, min_samples: int = 1,
           require_physical_evidence: bool = False) -> dict[str, Any]:
    if not 0.0 <= min_per_drive_retention <= 1.0:
        fail("min-per-drive-retention must be in 0..1")
    if not 0.0 <= min_fairness <= 1.0:
        fail("min-fairness must be in 0..1")
    if min_samples < 1:
        fail("min-samples must be positive")
    if not captures:
        fail("at least one capture is required")
    def scope(capture: dict[str, Any]) -> tuple[Any, ...]:
        host = capture.get("host")
        if not isinstance(host, dict):
            host = {}
        return (
            capture.get("read_bytes"),
            capture.get("topology_hash"),
            capture.get("manifest_sha256"),
            host.get("io_mode"),
            host.get("cache_condition"),
            host.get("backend"),
            host.get("asynchronous"),
            host.get("range_reuse"),
        )
    first_scope = scope(captures[0])
    if any(scope(capture) != first_scope for capture in captures[1:]):
        fail("captures have incompatible request/topology scope")
    tail_key = f"latency_ns_p{tail_percentile}"
    isolated_rates: dict[tuple[int, int, int], tuple[int, int]] = {}
    for capture in captures:
        size = int(capture["read_bytes"])
        for measurement in capture["measurements"]:
            if not isinstance(measurement, dict):
                continue
            ids = measurement.get("drive_ids")
            summary = measurement.get("summary")
            if (not isinstance(ids, list) or len(ids) != 1
                    or not isinstance(summary, dict)):
                continue
            if measurement.get("measurement_kind", "scalar") != "scalar":
                continue
            qd = measurement.get("inflight")
            rate = summary.get("aggregate_bytes_per_s_p50")
            if (isinstance(qd, int) and qd > 0
                    and isinstance(rate, int) and rate > 0):
                key = (size, int(ids[0]), qd)
                specificity = (1 if measurement.get("target_kind") == "drive"
                                else 0)
                prior = isolated_rates.get(key)
                if (prior is None or specificity > prior[1]
                        or (specificity == prior[1] and rate < prior[0])):
                    isolated_rates[key] = (rate, specificity)
    groups: dict[tuple[str, tuple[int, ...], int], list[dict[str, Any]]] = {}
    for capture in captures:
        size = int(capture["read_bytes"])
        for measurement in capture["measurements"]:
            if not isinstance(measurement, dict):
                fail("measurement must be an object")
            name = measurement.get("group")
            ids = measurement.get("drive_ids")
            if not isinstance(name, str) or not name:
                fail("measurement group must be a non-empty string")
            if (not isinstance(ids, list) or not ids
                    or any(not isinstance(item, int) or item <= 0 for item in ids)):
                fail(f"{name}: invalid drive_ids")
            summary = measurement.get("summary")
            if not isinstance(summary, dict):
                fail(f"{name}: missing summary")
            p50 = as_int(summary, "latency_ns_p50", Path("capture"))
            tail_latency = as_int(summary, tail_key, Path("capture"))
            maximum = as_int(summary, "latency_ns_max", Path("capture"))
            inflight = measurement.get("inflight")
            if not isinstance(inflight, int) or inflight < 1:
                fail(f"{name}: invalid inflight")
            active_ids = measurement.get("active_drive_ids", ids)
            all_active = measurement.get(
                "all_group_drives_active", len(set(active_ids)) == len(set(ids)))
            if not isinstance(active_ids, list) or not isinstance(all_active, bool):
                fail(f"{name}: invalid active-drive metadata")
            row = {
                "inflight": inflight,
                "sample_count": measurement.get("sample_count", 1),
                "measurement_kind": measurement.get("measurement_kind", "scalar"),
                "target_resource_id": measurement.get("target_resource_id"),
                "target_kind": measurement.get("target_kind"),
                "active_drive_ids": active_ids,
                "all_group_drives_active": all_active,
                "latency_ns_p50": p50,
                tail_key: tail_latency,
                "latency_ns_max": maximum,
                "aggregate_bytes_per_s_p50": as_int(
                    summary, "aggregate_bytes_per_s_p50", Path("capture")),
            }
            if measurement.get("measurement_kind") == "contention_matrix":
                queue_depths = measurement.get("queue_depths")
                if not isinstance(queue_depths, dict):
                    fail(f"{name}: matrix measurement lacks queue_depths")
                row["queue_depths"] = {
                    str(key): int(value) for key, value in queue_depths.items()
                }
                expected_keys = {str(item) for item in ids}
                if set(row["queue_depths"]) != expected_keys:
                    fail(f"{name}: matrix queue_depths do not match drive_ids")
                if sum(row["queue_depths"].values()) != inflight:
                    fail(f"{name}: matrix queue depths do not sum to inflight")
                per_drive = summary.get("per_drive", {})
                if not isinstance(per_drive, dict):
                    fail(f"{name}: matrix summary lacks per_drive data")
                row["per_drive"] = per_drive
                retention: dict[str, float] = {}
                for drive_id in ids:
                    qd = row["queue_depths"][str(drive_id)]
                    if qd == 0:
                        continue
                    drive_summary = per_drive.get(str(drive_id))
                    baseline = isolated_rates.get((size, int(drive_id), qd))
                    baseline_rate = baseline[0] if baseline is not None else None
                    contended_rate = (drive_summary.get(
                        "aggregate_bytes_per_s_p50")
                        if isinstance(drive_summary, dict) else None)
                    if (isinstance(baseline_rate, int) and baseline_rate > 0
                            and isinstance(contended_rate, int)
                            and contended_rate >= 0):
                        retention[str(drive_id)] = contended_rate / baseline_rate
                row["retention"] = retention
                values = list(retention.values())
                row["min_retention"] = min(values) if values else None
                if values:
                    value_sum = sum(values)
                    row["normalized_jain"] = (value_sum * value_sum) / (
                        len(values) * sum(value * value for value in values))
                else:
                    row["normalized_jain"] = None
            key = (name, tuple(int(item) for item in ids), size)
            groups.setdefault(key, []).append(row)

    results: list[dict[str, Any]] = []
    for (name, drive_ids, size), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["inflight"])
        # A group baseline must exercise every declared drive.  This prevents
        # a multi-drive group's inflight=1 partial row from becoming its
        # supposed one-read baseline.
        full = [row for row in rows if row["all_group_drives_active"]]
        if not full:
            continue
        baseline = full[0]
        allowed_tail = baseline[tail_key] * multiplier
        passing = [
            row for row in full
            if row[tail_key] <= allowed_tail
        ]
        def cell_passes(row: dict[str, Any]) -> bool:
            if row[tail_key] > allowed_tail:
                return False
            if row.get("sample_count", min_samples) < min_samples:
                return False
            if row["measurement_kind"] == "contention_matrix":
                if (min_per_drive_retention > 0.0
                        and (row.get("min_retention") is None
                             or row["min_retention"] < min_per_drive_retention)):
                    return False
                if (min_fairness > 0.0
                        and (row.get("normalized_jain") is None
                             or row["normalized_jain"] < min_fairness)):
                    return False
            return True

        by_total: dict[int, list[dict[str, Any]]] = {}
        for row in full:
            by_total.setdefault(row["inflight"], []).append(row)
        envelope = []
        for total, partition_rows in sorted(by_total.items()):
            passing_cells = [row for row in partition_rows if cell_passes(row)]
            envelope.append({
                "total_inflight": total,
                "observed_partitions": len(partition_rows),
                "passing_partitions": len(passing_cells),
                "all_observed_partitions_pass": (
                    len(passing_cells) == len(partition_rows)),
                "partitions": [
                    {"queue_depths": row.get("queue_depths"),
                     "pass": row in passing_cells,
                     "min_retention": row.get("min_retention"),
                     "normalized_jain": row.get("normalized_jain")}
                    for row in partition_rows
                ],
            })
        safe_limit = None
        prefix_pass = True
        for envelope_row in envelope:
            if prefix_pass and envelope_row["all_observed_partitions_pass"]:
                safe_limit = envelope_row["total_inflight"]
            else:
                prefix_pass = False
        candidate_limit = safe_limit
        passing = [row for row in full if cell_passes(row)]
        safe_rows = [row for row in passing
                     if candidate_limit is not None
                     and row["inflight"] == candidate_limit]
        candidate = (max(safe_rows, key=lambda row: (
            row["inflight"], -row[tail_key], row["aggregate_bytes_per_s_p50"]))
                     if safe_rows else None)
        results.append({
            "group": name,
            "drive_ids": list(drive_ids),
            "read_bytes": size,
            "measurement_kind": baseline["measurement_kind"],
            "target_resource_id": baseline.get("target_resource_id"),
            "target_kind": baseline.get("target_kind"),
            "tail_percentile_ppm": 950000 if tail_percentile == 95 else 990000,
            "baseline_inflight": baseline["inflight"],
            "baseline_latency_ns_tail": baseline[tail_key],
            "allowed_latency_ns_tail": int(allowed_tail),
            "max_inflight_candidate": candidate_limit,
            "candidate_is_observed_not_extrapolated": candidate is not None,
            "candidate_latency_ns_p50": (candidate["latency_ns_p50"]
                                          if candidate else None),
            "candidate_latency_ns_tail": (candidate[tail_key]
                                           if candidate else None),
            "candidate_latency_ns_max": (candidate["latency_ns_max"]
                                          if candidate else None),
            "candidate_rate": (candidate["aggregate_bytes_per_s_p50"]
                                if candidate else None),
            "candidate_sample_count": (candidate["sample_count"]
                                        if candidate else None),
            "candidate_tail_spread_ns": (
                candidate[tail_key] - candidate["latency_ns_p50"]
                if candidate else None),
            "candidate_queue_depths": (candidate.get("queue_depths")
                                        if candidate else None),
            "candidate_per_drive": (candidate.get("per_drive")
                                     if candidate else None),
            "candidate_retention": (candidate.get("retention")
                                     if candidate else None),
            "candidate_min_retention": (candidate.get("min_retention")
                                         if candidate else None),
            "candidate_normalized_jain": (candidate.get("normalized_jain")
                                           if candidate else None),
            "review_policy": {
                "min_per_drive_retention": min_per_drive_retention,
                "min_fairness": min_fairness,
                "min_samples": min_samples,
            },
            "observed_downward_envelope": envelope,
            "observations": full,
        })
    hosts = [capture.get("host", {}) for capture in captures]
    io_modes = {host.get("io_mode") for host in hosts
                if isinstance(host, dict)}
    cache_conditions = {host.get("cache_condition") for host in hosts
                        if isinstance(host, dict)}
    dma_verified = all(
        isinstance(host.get("dma_evidence"), dict)
        and host["dma_evidence"].get("status") == "VERIFIED"
        for host in hosts if isinstance(host, dict)) and bool(hosts)
    no_range_reuse = all(
        not bool(host.get("range_reuse", True))
        for host in hosts if isinstance(host, dict)) and bool(hosts)
    physical_evidence = (
        io_modes.issubset({"direct", "overlapped", "win32-overlapped",
                           "f_nocache"})
        and "unknown" not in cache_conditions
        and dma_verified and no_range_reuse)
    return {
        "schema": REVIEW_SCHEMA,
        "tool": {
            "name": "m3_shadow_calibration_review",
            "version": 2,
            "source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
        },
        "tail_percentile_ppm": 950000 if tail_percentile == 95 else 990000,
        "tail_percentile": tail_percentile,
        "allowed_tail_multiplier": multiplier,
        "captures": [
            {
                "topology_hash": capture.get("topology_hash"),
                "manifest_sha256": capture.get("manifest_sha256"),
                "capture_sha256": capture.get("capture_sha256"),
                "tool": capture.get("tool"),
                "range_reuse": bool(capture.get("host", {}).get(
                    "range_reuse", True)) if isinstance(capture.get("host"), dict)
                    else True,
                "range_reuse_allowed": bool(capture.get("host", {}).get(
                    "range_reuse_allowed", False)) if isinstance(
                        capture.get("host"), dict) else False,
            }
            for capture in captures
        ],
        "policy": (f"largest observed full-group inflight within allowed p{tail_percentile}; "
                   "no extrapolation"),
        "promotion": {
            "status": "CANDIDATE_ONLY",
            "physical_io_cache_dma_verified": physical_evidence,
            "required_by_policy": require_physical_evidence,
            "active_admission_allowed": False,
        },
        "results": results,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="append", required=True, type=Path)
    ap.add_argument("--max-tail-multiplier", "--max-p95-multiplier",
                    dest="max_tail_multiplier", required=True, type=float)
    ap.add_argument("--tail-percentile", choices=("p95", "p99"), default="p95")
    ap.add_argument("--min-per-drive-retention", type=float, default=0.0)
    ap.add_argument("--min-fairness", type=float, default=0.0)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--require-physical-evidence", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.max_tail_multiplier < 1.0:
        fail("--max-tail-multiplier must be at least 1.0")
    captures = [load_capture(path) for path in args.capture]
    result = review(captures, args.max_tail_multiplier,
                    int(args.tail_percentile[1:]),
                    args.min_per_drive_retention, args.min_fairness,
                    args.min_samples, args.require_physical_evidence)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    main()
