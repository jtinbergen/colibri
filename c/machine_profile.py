"""Canonical, read-only machine evidence for ``coli profile`` and ``coli plan``.

This module is deliberately an inventory/evidence bundle, not a benchmark.  It
does not scan model shards, flush caches, or infer shared-controller capacity.
It records OS-visible NUMA inventory, while leaving placement locality and
unmeasured capacity conservative.  The Step-7 calibration capture is
referenced by metadata only; the planner must not treat an incomplete capture
as an admission profile.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from resource_plan import (cpu_socket_count, discover_gpus, memory_available,
                           physical_cpu_count, ssd_probe_state)


SCHEMA = "colibri-machine-profile-v1"
PROFILE_VERSION = 1
_CALIBRATION_SCHEMAS = {
    "m3-shadow-calibration-v1",
    "m3-shadow-calibration-v2",
}
_REVIEW_SCHEMA = "m3-shadow-calibration-review-v2"
_RUNTIME_REPORT_SCHEMA = "m3-shadow-runtime-report-v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _capture_hash(raw: dict[str, Any]) -> str:
    """Hash a complete v2 capture without its self-referential digest."""
    unsigned = dict(raw)
    unsigned.pop("capture_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _review_summary(path: Path, capture: dict[str, Any]) -> dict[str, Any]:
    """Load a review as planner evidence without admitting its candidate."""
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration review {path}: {exc}") from exc
    if not isinstance(review, dict) or review.get("schema") != _REVIEW_SCHEMA:
        raise ValueError("calibration review must use m3-shadow-calibration-review-v2")
    promotion = review.get("promotion")
    if (not isinstance(promotion, dict)
            or promotion.get("status") != "CANDIDATE_ONLY"
            or promotion.get("active_admission_allowed") is not False):
        raise ValueError("calibration review must explicitly disallow admission")
    captures = review.get("captures")
    if not isinstance(captures, list) or len(captures) != 1:
        raise ValueError("calibration review has no capture provenance")
    capture_sha = capture.get("capture_sha256")
    matching = [item for item in captures
                if isinstance(item, dict)
                and item.get("capture_sha256") == capture_sha]
    if len(matching) != 1:
        raise ValueError("calibration review is not bound to this capture")
    if (matching[0].get("topology_hash") != capture.get("topology_hash")
            or matching[0].get("manifest_sha256") != capture.get(
                "manifest_sha256")
            or matching[0].get("tool") != capture.get("tool")):
        raise ValueError("calibration review topology provenance does not match capture")
    results = review.get("results")
    if not isinstance(results, list):
        raise ValueError("calibration review has no results list")
    candidate_rows: list[dict[str, Any]] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        candidate_rows.append({
            "group": result.get("group"),
            "target_resource_id": result.get("target_resource_id"),
            "target_kind": result.get("target_kind"),
            "drive_ids": result.get("drive_ids"),
            "candidate_is_observed_not_extrapolated": bool(
                result.get("candidate_is_observed_not_extrapolated", False)),
            "candidate_rate": result.get("candidate_rate"),
            "candidate_inflight": result.get("max_inflight_candidate"),
            "candidate_latency_ns_p50": result.get("candidate_latency_ns_p50"),
            "candidate_latency_ns_tail": result.get("candidate_latency_ns_tail"),
            "candidate_latency_ns_max": result.get("candidate_latency_ns_max"),
            "candidate_tail_spread_ns": result.get("candidate_tail_spread_ns"),
            "candidate_sample_count": result.get("candidate_sample_count"),
            "candidate_queue_depths": result.get("candidate_queue_depths"),
            "candidate_min_retention": result.get("candidate_min_retention"),
            "candidate_normalized_jain": result.get("candidate_normalized_jain"),
            "tail_percentile_ppm": result.get("tail_percentile_ppm",
                                               review.get("tail_percentile_ppm")),
            "review_policy": result.get("review_policy"),
        })
    return {
        "path": str(path),
        "schema": review["schema"],
        "review_sha256": _canonical_hash(review),
        "capture_sha256": capture_sha,
        "capture_binding_verified": True,
        "topology_hash": capture.get("topology_hash"),
        "manifest_sha256": capture.get("manifest_sha256"),
        "tail_percentile_ppm": review.get("tail_percentile_ppm"),
        "allowed_tail_multiplier": review.get("allowed_tail_multiplier"),
        "promotion_status": promotion.get("status"),
        "active_admission_allowed": False,
        "candidate_rows": candidate_rows,
    }


def _capture_summary(path: Path) -> dict[str, Any]:
    """Read only the bounded metadata needed to identify a calibration capture."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration capture {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") not in _CALIBRATION_SCHEMAS:
        raise ValueError("calibration capture must use a supported m3-shadow schema")
    capture_hash_verified = None
    if raw.get("schema") == "m3-shadow-calibration-v2":
        actual_capture_hash = raw.get("capture_sha256")
        if (not isinstance(actual_capture_hash, str)
                or actual_capture_hash != _capture_hash(raw)):
            raise ValueError("v2 calibration capture_sha256 is missing or invalid")
        capture_hash_verified = True
    levels = raw.get("inflight_levels")
    matrix_levels = raw.get("matrix_inflight_levels", [])
    measurements = raw.get("measurements")
    if (not isinstance(levels, list) or not levels or
            any(not isinstance(level, int) or level < 1 for level in levels) or
            not isinstance(measurements, list)):
        raise ValueError("calibration capture has invalid inflight metadata")
    if (matrix_levels and
            (not isinstance(matrix_levels, list)
             or any(not isinstance(level, int) or level < 0
                    for level in matrix_levels))):
        raise ValueError("calibration capture has invalid matrix metadata")
    groups: dict[str, dict[str, Any]] = {}
    for item in measurements:
        if not isinstance(item, dict) or not isinstance(item.get("group"), str):
            continue
        name = item["group"]
        row = groups.setdefault(name, {"name": name, "inflight_levels": [],
                                       "target_kind": item.get("target_kind"),
                                       "drive_ids": item.get("drive_ids", []),
                                       "all_group_drives_active": item.get(
                                           "all_group_drives_active", False),
                                       "measurement_kinds": [],
                                       "matrix_cells": 0,
                                       "full_matrix_cells": 0})
        row["all_group_drives_active"] = (
            bool(row["all_group_drives_active"])
            or bool(item.get("all_group_drives_active", False)))
        level = _int_or_none(item.get("inflight"))
        if level is not None:
            row["inflight_levels"].append(level)
        measurement_kind = item.get("measurement_kind", "scalar")
        if measurement_kind not in row["measurement_kinds"]:
            row["measurement_kinds"].append(measurement_kind)
        if measurement_kind == "contention_matrix":
            row["matrix_cells"] += 1
            if bool(item.get("all_group_drives_active", False)):
                row["full_matrix_cells"] += 1
    for row in groups.values():
        row["inflight_levels"] = sorted(set(row["inflight_levels"]))
        row["max_observed_inflight"] = max(row["inflight_levels"], default=None)
        row["measurement_kinds"].sort()
    host = raw.get("host") if isinstance(raw.get("host"), dict) else {}
    tool = raw.get("tool") if isinstance(raw.get("tool"), dict) else None
    manifest = raw.get("manifest") if isinstance(raw.get("manifest"), dict) else {}
    topology = {
        "topology_hash": raw.get("topology_hash"),
        "topology_hash_algorithm": raw.get("topology_hash_algorithm"),
        "topology_hash_verified": bool(raw.get("topology_hash_verified", False)),
        "manifest_sha256": raw.get("manifest_sha256"),
        "sources": (raw.get("sources")
                    if isinstance(raw.get("sources"), list) else []),
        "groups": (manifest.get("groups")
                   if isinstance(manifest.get("groups"), list) else []),
        "islands": (manifest.get("islands")
                    if isinstance(manifest.get("islands"), list) else []),
        "storage_bounds": (manifest.get("storage_bounds")
                           if isinstance(manifest.get("storage_bounds"), dict)
                           else None),
        "active_admission": bool(manifest.get("active_admission", False)),
        "profile_status": manifest.get("profile_status", "raw_only"),
    }
    tool_provenance = "UNKNOWN"
    if isinstance(tool, dict) and isinstance(tool.get("source_sha256"), str):
        try:
            tool_path = Path(__file__).resolve().parent / "tools" / "m3_shadow_calibrate.py"
            actual_sha = hashlib.sha256(tool_path.read_bytes()).hexdigest()
            tool_provenance = ("VERIFIED" if actual_sha == tool["source_sha256"]
                               else "MISMATCH")
        except OSError:
            tool_provenance = "UNAVAILABLE"
    cache_condition = host.get("cache_condition", "UNKNOWN")
    if not isinstance(cache_condition, str) or not cache_condition:
        cache_condition = "UNKNOWN"
    return {
        "path": str(path),
        "schema": raw["schema"],
        "tool": tool,
        "tool_provenance": tool_provenance,
        "quantiles": (raw.get("quantiles")
                      if isinstance(raw.get("quantiles"), dict) else None),
        "topology_hash": raw.get("topology_hash"),
        "manifest_sha256": raw.get("manifest_sha256"),
        "capture_sha256": raw.get("capture_sha256"),
        "capture_hash_verified": capture_hash_verified,
        "read_bytes": _int_or_none(raw.get("read_bytes")),
        "sample_count": _int_or_none(raw.get("sample_count")),
        "inflight_levels": sorted(set(levels)),
        "matrix_inflight_levels": sorted(set(matrix_levels)),
        "max_total_inflight": _int_or_none(raw.get("max_total_inflight")),
        "operations_per_worker": _int_or_none(
            raw.get("operations_per_worker")),
        "io": {
            "mode": host.get("io_mode", "UNKNOWN"),
            "backend": host.get("backend", "UNKNOWN"),
            "asynchronous": bool(host.get("asynchronous", False)),
            "async_request_protocol": host.get("async_request_protocol"),
            "cache_protocol": host.get("cache_protocol", "UNKNOWN"),
            "dma_evidence": host.get("dma_evidence"),
            "platform_topology": host.get("platform_topology"),
            "pcie_traffic_evidence": host.get("pcie_traffic_evidence"),
            "range_reuse": bool(host.get("range_reuse", True)),
            "range_reuse_allowed": bool(host.get("range_reuse_allowed", False)),
        },
        "range_allocation": (raw.get("range_allocation")
                             if isinstance(raw.get("range_allocation"), dict)
                             else None),
        "cache_condition": cache_condition,
        "memory": raw.get("memory") if isinstance(raw.get("memory"), dict) else None,
        "topology": topology,
        "source_count": len(raw.get("sources", [])) if isinstance(raw.get("sources"), list) else 0,
        "groups": sorted(groups.values(), key=lambda row: row["name"]),
    }


def _runtime_report_summary(
        path: str | os.PathLike[str] | None,
        expected_topology_hash: int | None = None) -> dict[str, Any]:
    """Attach runtime telemetry as evidence without making it admission data."""
    if not path:
        return {"status": "NOT_RUN", "report": None,
                "reason": "no runtime telemetry report supplied"}
    report_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                "reason": f"cannot read runtime telemetry report: {exc}"}
    if not isinstance(raw, dict) or raw.get("schema") != _RUNTIME_REPORT_SCHEMA:
        return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                "reason": f"runtime telemetry report must use {_RUNTIME_REPORT_SCHEMA}"}
    policy = raw.get("candidate_policy")
    if (not isinstance(policy, dict)
            or policy.get("status") != "CANDIDATE_ONLY"
            or policy.get("active_admission_allowed") is not False
            or policy.get("unknown_and_unavailable_are_not_capacity") is not True):
        return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                "reason": "runtime telemetry report does not have the conservative candidate policy"}
    candidates = raw.get("runtime_candidates", [])
    if not isinstance(candidates, list):
        return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                "reason": "runtime telemetry report has no candidate list"}
    safe_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                    "reason": "runtime telemetry report contains an invalid candidate"}
        if (candidate.get("status") != "CANDIDATE_ONLY"
                or candidate.get("active_admission_allowed") is not False
                or candidate.get("evidence_state") != "OBSERVED"):
            return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                    "reason": "runtime candidate is not explicitly inactive observed evidence"}
        scope = candidate.get("scope")
        if not isinstance(scope, dict) or not scope.get("actual_path"):
            return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                    "reason": "runtime candidate has no exact resource path scope"}
        candidate_topology_hash = scope.get("topology_hash")
        if (expected_topology_hash is not None
                and candidate_topology_hash != expected_topology_hash):
            return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                    "reason": "runtime candidate topology hash does not match the machine profile"}
        # Keep this attachment bounded and evidence-only.  Do not copy
        # arbitrary report content into a profile that downstream code could
        # mistake for active planner configuration.
        safe_candidates.append({
            "status": "CANDIDATE_ONLY",
            "active_admission_allowed": False,
            "evidence_state": "OBSERVED",
            "confidence": candidate.get("confidence", "UNKNOWN"),
            "scope": scope,
            "max_inflight_candidate": candidate.get("max_inflight_candidate"),
            "candidate_sample_count": candidate.get("candidate_sample_count"),
            "independent_run_count": candidate.get("independent_run_count"),
            "independent_run_ids": candidate.get("independent_run_ids", []),
            "candidate_latency_ns_p95": candidate.get("candidate_latency_ns_p95"),
            "candidate_latency_ns_p99": candidate.get("candidate_latency_ns_p99"),
            "candidate_latency_ns_tail": candidate.get("candidate_latency_ns_tail"),
            "allowed_latency_ns_tail": candidate.get("allowed_latency_ns_tail"),
            "tail_percentile": candidate.get("tail_percentile"),
            "observed_levels": candidate.get("observed_levels", []),
            "policy": candidate.get("policy"),
        })
    try:
        report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    except OSError as exc:
        return {"status": "UNKNOWN", "report": {"path": str(report_path)},
                "reason": f"cannot hash runtime telemetry report: {exc}"}
    sources = raw.get("sources", [])
    safe_sources = ([item for item in sources if isinstance(item, dict)]
                    if isinstance(sources, list) else [])
    return {
        "status": "EVIDENCE_ONLY",
        "report": {
            "path": str(report_path),
            "schema": raw["schema"],
            "report_sha256": report_sha256,
            "tool_sha256": raw.get("tool_sha256"),
            "sources": safe_sources,
            "rows_total": raw.get("rows_total", 0),
            "rows_completed": raw.get("rows_completed", 0),
            "rows_successful": raw.get("rows_successful", 0),
            "evidence_counts": raw.get("evidence_counts", {}),
            "candidate_policy": {
                "status": "CANDIDATE_ONLY",
                "active_admission_allowed": False,
                "unknown_and_unavailable_are_not_capacity": True,
                "tail_percentile": policy.get("tail_percentile"),
                "max_tail_multiplier": policy.get("max_tail_multiplier"),
                "min_samples": policy.get("min_samples"),
            },
            "topology_binding": ("MATCHED" if expected_topology_hash is not None
                                  else "UNBOUND"),
            "runtime_candidates": safe_candidates,
        },
        "reason": "runtime telemetry is attached as evidence-only candidate data; active admission remains disabled",
    }


def _storage(model_dir: str | os.PathLike[str] | None) -> dict[str, Any]:
    if not model_dir:
        return {"status": "NOT_RUN", "model_path": None,
                "reason": "no model directory supplied"}
    path = Path(model_dir).expanduser()
    result: dict[str, Any] = {"status": "DISCOVERED", "model_path": str(path.resolve())}
    try:
        usage = shutil.disk_usage(path)
        result.update({"total_bytes": usage.total, "free_bytes": usage.free})
    except OSError as exc:
        result.update({"status": "UNKNOWN", "reason": f"disk usage unavailable: {exc}"})
    try:
        state, gbs = ssd_probe_state(path)
        result["ssd_probe"] = {"state": state, "gbs": gbs}
    except (OSError, ValueError):
        result["ssd_probe"] = {"state": "UNKNOWN", "gbs": None}
    return result


def _gpus() -> dict[str, Any]:
    try:
        devices = discover_gpus()
    except Exception as exc:  # discovery is advisory; profile must remain usable
        return {"status": "UNKNOWN", "devices": [], "reason": str(exc)}
    return {"status": "DISCOVERED", "devices": devices}


def _probe_numa_topology() -> dict[str, Any]:
    """Discover NUMA placement without changing process affinity or policy.

    This is intentionally an inventory probe.  It reports the OS-visible node
    topology, but does not claim that a storage endpoint or a particular
    allocation is local to one node.  A one-node result is useful evidence and
    is distinct from a probe that was never attempted.
    """
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            highest = wintypes.USHORT()
            function = kernel32.GetNumaHighestNodeNumber
            function.argtypes = [ctypes.POINTER(wintypes.USHORT)]
            function.restype = wintypes.BOOL
            if function(ctypes.byref(highest)):
                node_count = int(highest.value) + 1
                return {
                    "status": "OBSERVED",
                    "method": "GetNumaHighestNodeNumber",
                    "node_count": node_count,
                    "nodes": list(range(node_count)),
                    "single_node": node_count == 1,
                }
            error = ctypes.get_last_error()
            return {"status": "UNAVAILABLE",
                    "method": "GetNumaHighestNodeNumber",
                    "reason": f"Windows probe failed with error {error}"}
        except (AttributeError, OSError, TypeError, ValueError) as exc:
            return {"status": "UNAVAILABLE",
                    "method": "GetNumaHighestNodeNumber",
                    "reason": str(exc)}

    if sys.platform.startswith("linux"):
        node_root = Path("/sys/devices/system/node")
        try:
            nodes = sorted(
                int(path.name[4:])
                for path in node_root.glob("node[0-9]*")
                if path.name[4:].isdigit())
        except OSError as exc:
            return {"status": "UNAVAILABLE", "method": "sysfs", "reason": str(exc)}
        if nodes:
            return {
                "status": "OBSERVED",
                "method": "linux-sysfs",
                "node_count": len(nodes),
                "nodes": nodes,
                "single_node": len(nodes) == 1,
            }
        return {"status": "UNAVAILABLE", "method": "linux-sysfs",
                "reason": "no online NUMA node directories were reported"}

    if sys.platform == "darwin":
        # macOS does not expose a user-selectable NUMA topology on the
        # supported client/server machines.  This is a qualified one-domain
        # result, not a missing measurement.
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.packages"], capture_output=True,
                text=True, check=True, timeout=5)
            packages = int(result.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            packages = 1
        if packages > 0:
            return {"status": "NOT_APPLICABLE", "method": "macOS-sysctl",
                    "node_count": 1, "nodes": [0], "single_node": True,
                    "reason": "macOS exposes one planner memory domain"}
        return {"status": "UNAVAILABLE", "method": "macOS-sysctl",
                "reason": "could not determine memory-domain count"}

    return {"status": "UNAVAILABLE", "method": sys.platform,
            "reason": "NUMA inventory probe is not implemented for this OS"}


def _baseline_fallback(max_inflight: int | None = None) -> dict[str, Any]:
    """Return the explicit behavior used when no trusted profile is present."""
    if not isinstance(max_inflight, int) or max_inflight < 1:
        max_inflight = 1
    return {
        "mode": "original_colibri_baseline",
        "admission": "disabled",
        "unknown_capacity": "conservative",
        "max_inflight": max_inflight,
        "rate_policy": "UNKNOWN; do not infer unavailable link/DMA capacity",
    }


def baseline_planner_contract(max_inflight: int = 1) -> dict[str, Any]:
    """Return the planner metadata for an absent or unusable profile."""
    return {
        "admission": "DISABLED",
        "cache_condition_required": True,
        "unknown_is_not_optimistic_capacity": True,
        "baseline_fallback": _baseline_fallback(max_inflight),
    }


def _calibration(path: str | os.PathLike[str] | None,
                 numa: dict[str, Any] | None = None,
                 review_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    coverage = {
        "storage": "NOT_RUN",
        "drive_service": "NOT_RUN",
        "controller_service": "NOT_RUN",
        "shared_controller": "NOT_RUN",
        "shared_upstream": "NOT_RUN",
        "host_memory": "NOT_RUN",
        "to_memory_dma": "NOT_RUN",
        "pcie_negotiated_topology": "NOT_RUN",
        "pch_dmi": "NOT_RUN",
        "numa": (numa.get("status", "NOT_RUN")
                 if isinstance(numa, dict) else "NOT_RUN"),
        "gpu_transfer": "NOT_RUN",
    }
    if not path:
        return {"status": "NOT_RUN", "capture": None, "coverage": coverage,
                "reason": "no calibration capture supplied"}
    capture_path = Path(path).expanduser().resolve()
    try:
        capture = _capture_summary(capture_path)
    except ValueError as exc:
        return {"status": "UNKNOWN", "capture": {"path": str(capture_path)},
                "coverage": coverage, "reason": str(exc)}
    review_error = None
    if review_path:
        review_file = Path(review_path).expanduser().resolve()
        try:
            capture["review"] = _review_summary(review_file, capture)
        except ValueError as exc:
            review_error = str(exc)
            capture["review"] = {
                "path": str(review_file),
                "schema": None,
                "capture_binding_verified": False,
                "status": "UNKNOWN",
                "reason": review_error,
            }
    coverage["storage"] = "EVIDENCE_ONLY"
    if isinstance(capture.get("memory"), dict):
        coverage["host_memory"] = "EVIDENCE_ONLY"
        capture["memory_method"] = capture["memory"].get("method", "UNKNOWN")
    io_evidence = capture.get("io", {}).get("dma_evidence")
    if isinstance(io_evidence, dict) and io_evidence.get("status") == "OBSERVED_NOT_ATTESTED":
        coverage["to_memory_dma"] = "OBSERVED_NOT_ATTESTED"
    platform_topology = capture.get("io", {}).get("platform_topology")
    if isinstance(platform_topology, dict):
        status = platform_topology.get("status")
        if status in ("OBSERVED", "UNAVAILABLE"):
            coverage["pcie_negotiated_topology"] = status
    traffic_evidence = capture.get("io", {}).get("pcie_traffic_evidence")
    if isinstance(traffic_evidence, dict):
        traffic_status = traffic_evidence.get("status")
        if traffic_status in ("OBSERVED", "EVIDENCE_ONLY",
                              "OBSERVED_NOT_ATTESTED", "AVAILABLE_NOT_RUN",
                              "UNAVAILABLE"):
            coverage["pch_dmi"] = traffic_status
    targeted = capture.get("groups", [])
    if isinstance(targeted, list):
        controller_group_has_siblings = False
        gpu_islands = []
        for item in targeted:
            if not isinstance(item, dict):
                continue
            kind = item.get("target_kind")
            drives = item.get("drive_ids", [])
            if kind == "drive" and isinstance(drives, list):
                coverage["drive_service"] = "EVIDENCE_ONLY"
            if kind == "controller" and isinstance(drives, list):
                coverage["controller_service"] = "EVIDENCE_ONLY"
                if len(set(drives)) > 1:
                    controller_group_has_siblings = True
            all_active = item.get("all_group_drives_active", False)
            matrix_complete = (
                "contention_matrix" in item.get("measurement_kinds", [])
                and item.get("full_matrix_cells", 0) > 0)
            if (kind == "controller" and isinstance(drives, list)
                    and len(set(drives)) > 1 and all_active and matrix_complete):
                coverage["shared_controller"] = "EVIDENCE_ONLY"
            if (kind == "upstream" and isinstance(drives, list)
                    and len(set(drives)) > 1 and all_active and matrix_complete):
                coverage["shared_upstream"] = "EVIDENCE_ONLY"
        if not controller_group_has_siblings:
            # A topology with one drive per controller has no controller-level
            # sharing experiment to run; the per-controller service rows are
            # still retained above.
            coverage["shared_controller"] = "NOT_APPLICABLE"

    topology = capture.get("topology", {})
    islands = topology.get("islands", []) if isinstance(topology, dict) else []
    if isinstance(islands, list):
        gpu_islands = [item for item in islands
                       if isinstance(item, dict) and item.get("kind") == "gpu_island"]
        eligible_gpus = [item for item in gpu_islands if item.get("eligible") == 1]
        if gpu_islands and not eligible_gpus:
            coverage["gpu_transfer"] = "NOT_APPLICABLE"
        for item in islands:
            if not isinstance(item, dict) or item.get("kind") != "link":
                continue
            if item.get("status") == "MEASURED" and isinstance(
                    item.get("rate_bytes_per_s"), int):
                coverage["pch_dmi"] = "EVIDENCE_ONLY"
            elif item.get("status") in ("NOT_APPLICABLE", "UNAVAILABLE"):
                coverage["pch_dmi"] = item["status"]
    # A raw capture can contain controller-named groups, but that is not proof
    # of shared contention.  Keep those classes NOT_RUN until the dedicated
    # simultaneous multi-drive measurement has been reviewed and admitted.
    reason = ("raw capture is evidence-only, not an admitted planner profile; "
              "unattested or unavailable hardware limits remain conservative UNKNOWN")
    if review_error:
        reason += f"; review attachment is unavailable: {review_error}"
    return {"status": "EVIDENCE_ONLY", "capture": capture, "coverage": coverage,
            "reason": reason}


def build_machine_profile(model_dir: str | os.PathLike[str] | None = None,
                          calibration_path: str | os.PathLike[str] | None = None,
                          review_path: str | os.PathLike[str] | None = None,
                          runtime_report_path: str | os.PathLike[str] | None = None,
                          now: str | None = None) -> dict[str, Any]:
    """Build a portable snapshot without benchmarking or model-shard scans."""
    try:
        logical = os.cpu_count()
    except (OSError, ValueError):
        logical = None
    try:
        physical = physical_cpu_count()
    except (OSError, ValueError):
        physical = None
    try:
        sockets = cpu_socket_count()
    except (OSError, ValueError):
        sockets = None
    try:
        available = memory_available()
    except (OSError, ValueError):
        available = None
    numa = _probe_numa_topology()
    calibration = _calibration(calibration_path, numa, review_path)
    capture = calibration.get("capture")
    expected_topology_hash = (
        capture.get("topology_hash")
        if isinstance(capture, dict)
        and isinstance(capture.get("topology_hash"), int)
        else None)
    runtime_evidence = _runtime_report_summary(
        runtime_report_path, expected_topology_hash)
    coverage = calibration.get("coverage", {})
    storage_action = (
        "review/promote storage calibration under the declared cache and tail policy"
        if coverage.get("storage") == "EVIDENCE_ONLY"
        else "measure representative expert reads with explicit cache condition and tail units")
    contention_action = (
        "review/promote shared-controller/upstream envelopes; DMA and admission gates remain open"
        if (coverage.get("shared_controller") == "EVIDENCE_ONLY"
            or coverage.get("shared_upstream") == "EVIDENCE_ONLY")
        else "measure simultaneous drives within controller and shared-upstream groups")
    bounds = (capture.get("topology", {}).get("storage_bounds")
              if isinstance(capture, dict)
              and isinstance(capture.get("topology"), dict) else None)
    review = (capture.get("review")
              if isinstance(capture, dict)
              and isinstance(capture.get("review"), dict) else None)
    review_bound = bool(review and review.get("capture_binding_verified"))
    baseline_fallback = _baseline_fallback(
        bounds.get("max_inflight") if isinstance(bounds, dict) else None)
    if review_bound:
        storage_action_status = "evidence_only"
        storage_action_reason = (
            "reviewed candidate rows are attached; active admission remains disabled "
            "until physical I/O and held-out gates close")
        contention_action_status = "evidence_only"
        contention_action_reason = (
            "reviewed controller/upstream envelopes are attached; DMA and admission "
            "gates remain open")
    else:
        storage_action_status = "required"
        storage_action_reason = storage_action
        contention_action_status = "required"
        contention_action_reason = contention_action
    return {
        "schema": SCHEMA,
        "version": PROFILE_VERSION,
        "generated_at_utc": now or _utc_now(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "cpu": {
            "logical_cores": logical,
            "physical_cores": physical,
            "sockets": sockets,
            "numa": numa,
        },
        "memory": {"available_bytes": available},
        "gpus": _gpus(),
        "storage": _storage(model_dir),
        "calibration": calibration,
        "runtime_evidence": runtime_evidence,
        "planner_contract": {
            "admission": "DISABLED",
            "cache_condition_required": True,
            "unknown_is_not_optimistic_capacity": True,
            "baseline_fallback": baseline_fallback,
        },
        "calibration_review": review,
        "next_actions": [
            {"id": "calibrate-storage", "status": storage_action_status,
             "reason": storage_action_reason},
            {"id": "calibrate-contention", "status": contention_action_status,
             "reason": contention_action_reason},
            {"id": "calibrate-numa", "status": "optional",
             "reason": "measure CPU/memory/link placement before enabling NUMA-sensitive planning"},
        ],
    }


def load_machine_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    profile_path = Path(path)
    try:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read machine profile {profile_path}: {exc}") from exc
    if (not isinstance(profile, dict) or profile.get("schema") != SCHEMA or
            profile.get("version") != PROFILE_VERSION):
        raise ValueError(f"machine profile must use {SCHEMA} version {PROFILE_VERSION}")
    return profile


def write_machine_profile(path: str | os.PathLike[str], profile: dict[str, Any]) -> None:
    if profile.get("schema") != SCHEMA or profile.get("version") != PROFILE_VERSION:
        raise ValueError("refusing to write an invalid machine profile")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp",
                                     dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(profile, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def profile_summary(profile: dict[str, Any], path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    calibration = profile.get("calibration", {})
    storage = profile.get("storage", {})
    return {
        "schema": profile.get("schema"),
        "version": profile.get("version"),
        "path": str(path) if path is not None else None,
        "generated_at_utc": profile.get("generated_at_utc"),
        "host": profile.get("host", {}),
        "cpu": profile.get("cpu", {}),
        "memory_available_bytes": profile.get("memory", {}).get("available_bytes"),
        "gpu_status": profile.get("gpus", {}).get("status", "UNKNOWN"),
        "gpu_count": len(profile.get("gpus", {}).get("devices", [])),
        "storage_status": storage.get("status", "UNKNOWN"),
        "ssd_probe": storage.get("ssd_probe"),
        "status": calibration.get("status", "UNKNOWN"),
        "calibration_status": calibration.get("status", "UNKNOWN"),
        "calibration_coverage": calibration.get("coverage", {}),
        "calibration_review": profile.get("calibration_review"),
        "runtime_evidence": profile.get("runtime_evidence", {
            "status": "NOT_RUN", "report": None,
            "reason": "no runtime telemetry report supplied",
        }),
        "planner_contract": profile.get("planner_contract", {}),
        "topology": calibration.get("capture", {}).get("topology")
                    if isinstance(calibration.get("capture"), dict) else None,
        "next_actions": profile.get("next_actions", []),
    }


def attach_machine_profile(plan: dict[str, Any], profile: dict[str, Any],
                           path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Attach evidence to a plan without changing any placement decision."""
    plan["machine_profile"] = profile_summary(profile, path)
    if profile.get("calibration", {}).get("status") != "EVIDENCE_ONLY":
        plan.setdefault("warnings", []).append(
            "machine profile has no admitted Step-7 calibration; storage/contention capacity remains UNKNOWN")
    else:
        plan.setdefault("warnings", []).append(
            "machine profile calibration is evidence-only; no storage/controller admission limit was applied")
    runtime = profile.get("runtime_evidence", {})
    if runtime.get("status") == "EVIDENCE_ONLY":
        plan.setdefault("warnings", []).append(
            "runtime telemetry candidates are evidence-only; no runtime admission limit was applied")
    elif runtime.get("status") == "UNKNOWN":
        plan.setdefault("warnings", []).append(
            "runtime telemetry report is unavailable; runtime capacity remains UNKNOWN")
    return plan


def format_machine_profile(profile: dict[str, Any], path: str | os.PathLike[str] | None = None) -> str:
    summary = profile_summary(profile, path)
    cpu = summary["cpu"]
    lines = [
        f"schema  {summary['schema']} v{summary['version']}",
        f"host    {summary['host'].get('system', '?')} {summary['host'].get('release', '?')} · "
        f"{summary['host'].get('machine', '?')}",
        f"cpu     {cpu.get('physical_cores', '?')} physical · {cpu.get('logical_cores', '?')} logical · "
        f"{cpu.get('sockets', '?')} socket(s)",
        f"memory  {summary['memory_available_bytes']} bytes available",
        f"gpu     {summary['gpu_count']} discovered ({summary['gpu_status']})",
        f"storage {summary['storage_status']}",
        f"calib   {summary['calibration_status']} · contention="
        f"controller={summary['calibration_coverage'].get('shared_controller', 'UNKNOWN')} "
        f"upstream={summary['calibration_coverage'].get('shared_upstream', 'UNKNOWN')}",
    ]
    runtime = summary.get("runtime_evidence", {})
    report = runtime.get("report") if isinstance(runtime, dict) else None
    candidates = (report.get("runtime_candidates", [])
                  if isinstance(report, dict) else [])
    lines.append(f"runtime {runtime.get('status', 'UNKNOWN')} Â· "
                 f"candidates={len(candidates) if isinstance(candidates, list) else 0} "
                 "(admission disabled)")
    if summary["path"]:
        lines.append(f"saved   {summary['path']}")
    lines.append("next actions:")
    for action in summary["next_actions"]:
        lines.append(f"  [{action['status']}] {action['id']}: {action['reason']}")
    return "\n".join(lines)
