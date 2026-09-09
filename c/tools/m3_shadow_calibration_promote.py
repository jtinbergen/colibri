#!/usr/bin/env python3
"""Emit an explicitly reviewed, candidate-only planner profile.

``m3_shadow_calibrate.py`` emits topology only.  This tool is the separate
review boundary: it accepts a raw capture plus its review JSON, verifies their
provenance, and emits only observed candidate rows selected by the review.
It never emits an active-admission configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

try:
    from m3_shadow_calibrate import (emit_planner_text, _row_classify_load,
                                     _row_classify_size)
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from m3_shadow_calibrate import (emit_planner_text, _row_classify_load,
                                     _row_classify_size)


PROMOTION_SCHEMA = "m3-shadow-candidate-planner-v1"


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"m3-shadow-calibration-promote: {message}")


def capture_hash(capture: dict[str, Any]) -> str:
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read {label} {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"{label} must be a JSON object")
    return value


def _verify_provenance(capture: dict[str, Any], review: dict[str, Any],
                       manifest_path: Path) -> str:
    if capture.get("schema") not in ("m3-shadow-calibration-v1",
                                      "m3-shadow-calibration-v2"):
        fail("capture schema is unsupported")
    if capture.get("schema") != "m3-shadow-calibration-v2":
        fail("legacy v1 captures are evidence-only; recapture with v2 before promotion")
    if review.get("schema") != "m3-shadow-calibration-review-v2":
        fail("review schema is unsupported")
    promotion = review.get("promotion")
    if (not isinstance(promotion, dict)
            or promotion.get("status") != "CANDIDATE_ONLY"
            or promotion.get("active_admission_allowed") is not False):
        fail("review is not an inactive candidate-only result")
    host = capture.get("host")
    if (not isinstance(host, dict) or bool(host.get("range_reuse", True))):
        fail("candidate emission requires a capture with range_reuse=false")
    capture_digest = None
    if capture.get("schema") == "m3-shadow-calibration-v2":
        capture_digest = capture.get("capture_sha256")
        if not isinstance(capture_digest, str) or capture_digest != capture_hash(capture):
            fail("capture_sha256 does not match complete capture contents")
    try:
        manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except OSError as exc:
        fail(f"cannot read manifest {manifest_path}: {exc}")
    if capture.get("manifest_sha256") != manifest_sha:
        fail("capture manifest_sha256 does not match manifest")
    captures = review.get("captures")
    if not isinstance(captures, list) or len(captures) != 1:
        fail("review must contain exactly one capture provenance record")
    provenance = captures[0]
    if (not isinstance(provenance, dict)
            or provenance.get("topology_hash") != capture.get("topology_hash")
            or provenance.get("manifest_sha256") != capture.get("manifest_sha256")
            or provenance.get("capture_sha256") != capture_digest
            or provenance.get("tool") != capture.get("tool")):
        fail("review provenance does not match capture")
    try:
        review_sha = hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        fail(f"cannot hash review: {exc}")
    return review_sha


def _candidate_profiles(capture: dict[str, Any],
                        review: dict[str, Any]) -> list[str]:
    tail_ppm = review.get("tail_percentile_ppm")
    if tail_ppm not in (950000, 990000):
        fail("review tail_percentile_ppm must be 950000 or 990000")
    size_class = _row_classify_size(int(capture["read_bytes"]))
    selected: dict[tuple[int, int, int], tuple[int, int, int, int, int]] = {}
    for result in review.get("results", []):
        if not isinstance(result, dict):
            fail("review results must be objects")
        if not result.get("candidate_is_observed_not_extrapolated"):
            continue
        if result.get("tail_percentile_ppm") != tail_ppm:
            fail(f"review group {result.get('group')} has inconsistent tail percentile")
        resource_id = result.get("target_resource_id")
        drive_ids = result.get("drive_ids")
        if resource_id is None:
            if not isinstance(drive_ids, list) or len(drive_ids) != 1:
                fail(f"review group {result.get('group')} lacks unambiguous resource owner")
            resource_id = drive_ids[0]
        if (not isinstance(resource_id, int) or resource_id <= 0
                or not isinstance(result.get("max_inflight_candidate"), int)
                or not isinstance(result.get("candidate_rate"), int)
                or result["candidate_rate"] <= 0):
            fail(f"review group {result.get('group')} has invalid candidate row")
        candidate_tail = result.get("candidate_latency_ns_tail")
        candidate_p50 = result.get("candidate_latency_ns_p50")
        samples = result.get("candidate_sample_count")
        if (not isinstance(candidate_tail, int) or not isinstance(candidate_p50, int)
                or not isinstance(samples, int) or samples < 1):
            fail(f"review group {result.get('group')} lacks candidate latency evidence")
        residual = max(0, candidate_tail - candidate_p50)
        key = (resource_id, size_class,
               _row_classify_load(result["max_inflight_candidate"]))
        row = (result["candidate_rate"], 0, residual, samples, tail_ppm)
        prior = selected.get(key)
        if prior is None:
            selected[key] = row
        else:
            selected[key] = (min(prior[0], row[0]), max(prior[1], row[1]),
                             max(prior[2], row[2]), max(prior[3], row[3]),
                             tail_ppm)
    if not selected:
        fail("review contains no observed candidate rows")
    return [
        f"profile {resource_id} {size} {load} {rate} {startup} {residual} "
        f"{samples} {tail_ppm}"
        for (resource_id, size, load),
        (rate, startup, residual, samples, tail_ppm)
        in sorted(selected.items())
    ]


def emit_candidate(capture: dict[str, Any], review: dict[str, Any],
                   review_sha: str) -> str:
    topology = emit_planner_text(capture)
    lines = topology.splitlines()
    profiles = _candidate_profiles(capture, review)
    replaced = []
    for line in lines:
        if line.startswith("# configuration_status = "):
            replaced.append("# configuration_status = candidate_only")
        elif line.startswith("# profile_rows = "):
            replaced.append(f"# profile_rows = {len(profiles)}; source=review")
        else:
            replaced.append(line)
    insert_at = next((index for index, line in enumerate(replaced)
                      if line.startswith(("cpu_island ", "memory_dom ",
                                          "link ", "gpu_island "))),
                     len(replaced))
    replaced[insert_at:insert_at] = [
        "# review_schema = m3-shadow-calibration-review-v2",
        f"# review_sha256 = {review_sha}",
        "# active_admission_allowed = False",
        *profiles,
    ]
    return "\n".join(replaced) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--review", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    capture = _load_json(args.capture, "capture")
    review = _load_json(args.review, "review")
    review_sha = _verify_provenance(capture, review, args.manifest)
    args.output.write_text(emit_candidate(capture, review, review_sha),
                           encoding="utf-8", newline="\n")
    profiles = sum(line.startswith("profile ")
                   for line in args.output.read_text(encoding="utf-8").splitlines())
    print(f"candidate planner: PASS profiles={profiles} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
