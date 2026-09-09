#!/usr/bin/env python3
"""Summarize correlated Step-7 runtime traces without changing planner state.

The report deliberately treats missing prediction, completion, consumer need,
or six-component mapping as UNKNOWN.  It computes prediction error and
soft-deadline misses only from completed rows that contain the relevant fields;
it never fills a missing value from another row or from the observed
completion itself.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED = {
    "request_id", "actual_drive_id", "actual_path", "consumer_need_ns",
    "predicted_ready_ns", "predicted_uncertainty_ns", "ready_ns",
    "component_mask", "load_result", "completed",
}

COMPONENT_MASK_ALL = 0x3F
EVIDENCE_STATES = {
    "OBSERVED", "UNKNOWN", "UNAVAILABLE", "THEORETICAL", "NOT_APPLICABLE",
}


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"m3-shadow-trace-report: {message}")


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def integer(row: dict[str, str], name: str, line: int) -> int:
    value = row.get(name)
    if value is None or value == "":
        fail(f"line {line}: missing {name}")
    try:
        parsed = value if isinstance(value, int) else int(value, 10)
    except ValueError as exc:
        fail(f"line {line}: invalid {name}={value!r}")
        raise AssertionError from exc
    if parsed < 0 and name not in {"load_result"}:
        fail(f"line {line}: negative {name}")
    return parsed


def optional_integer(row: dict[str, str], name: str, line: int) -> int | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    return integer(row, name, line)


def pipe_integers(row: dict[str, str], name: str, line: int) -> list[int] | None:
    value = row.get(name)
    if value is None or value == "":
        return None
    try:
        result = [int(part, 10) for part in value.split("|")]
    except ValueError as exc:
        fail(f"line {line}: invalid {name}={value!r}")
        raise AssertionError from exc
    if any(part < 0 for part in result):
        fail(f"line {line}: negative {name}")
    return result


def evidence_state(row: dict[str, str], line: int) -> str:
    value = row.get("evidence_state")
    if value:
        if value not in EVIDENCE_STATES:
            fail(f"line {line}: invalid evidence_state={value!r}")
        return value
    # v1 traces predate the explicit epistemic column.  Derive only the
    # narrowest honest state and never infer a positive state from prediction.
    done = integer(row, "completed", line)
    result = integer(row, "load_result", line)
    mask = integer(row, "component_mask", line)
    path = row.get("actual_path", "")
    return ("OBSERVED" if done and result == 0
            and mask == COMPONENT_MASK_ALL and path else "UNKNOWN")


def request_type(row: dict[str, str]) -> str:
    value = row.get("request_type")
    return value if value in {"demand", "prefetch"} else "unknown"


def service_latency_ns(row: dict[str, str], line: int) -> int | None:
    explicit = optional_integer(row, "service_latency_ns", line)
    if explicit is not None:
        return explicit
    issue = optional_integer(row, "issue_ns", line)
    ready = optional_integer(row, "ready_ns", line)
    if issue is None or ready is None or not issue or not ready:
        return None
    return max(0, ready - issue)


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * q)))
    return ordered[index]


def metric(errors: list[int], misses: int, predicted_misses: int,
           guarded_misses: int) -> dict[str, Any]:
    return {
        "samples": len(errors),
        "ready_error_ns_p50": percentile(errors, 0.50),
        "ready_error_ns_p95": percentile(errors, 0.95),
        "ready_error_ns_p99": percentile(errors, 0.99),
        "ready_error_ns_max_abs": max((abs(value) for value in errors),
                                       default=0),
        "deadline_misses": misses,
        "predicted_deadline_misses": predicted_misses,
        "guarded_deadline_misses": guarded_misses,
    }


def analyze(path: Path | list[Path], rows: list[dict[str, str]] | None = None,
            max_tail_multiplier: float = 1.5, min_samples: int = 2,
            tail_percentile: float = 0.95) -> dict[str, Any]:
    if rows is None:
        paths = path if isinstance(path, list) else [path]
        loaded: list[dict[str, str]] = []
        for source in paths:
            try:
                with source.open("r", encoding="ascii", newline="") as stream:
                    reader = csv.DictReader(stream)
                    fields = set(reader.fieldnames or [])
                    missing = sorted(REQUIRED - fields)
                    if missing:
                        fail(f"missing trace columns: {', '.join(missing)}")
                    loaded.extend(reader)
            except (OSError, UnicodeDecodeError) as exc:
                fail(f"cannot read {source}: {exc}")
        rows = loaded
    else:
        rows = list(rows)
    source_paths = path if isinstance(path, list) else [path]

    groups: dict[tuple[int, str], list[int]] = {}
    group_deadlines: dict[tuple[int, str], list[int]] = {}
    controller_groups: dict[int, list[int]] = {}
    controller_deadlines: dict[int, list[int]] = {}
    total = completed = successful = failed = incomplete = unknown = 0
    incomplete_mapping = 0
    deadline_misses = predicted_misses = guarded_misses = 0
    all_errors: list[int] = []
    evidence_counts = {state: 0 for state in sorted(EVIDENCE_STATES)}
    candidate_inputs: list[dict[str, Any]] = []

    for line, row in enumerate(rows, start=2):
        total += 1
        state = evidence_state(row, line)
        evidence_counts[state] += 1
        done = integer(row, "completed", line)
        result = integer(row, "load_result", line)
        if not done:
            incomplete += 1
            continue
        completed += 1
        if result == 0:
            successful += 1
        else:
            failed += 1
        if result != 0:
            continue
        component_mask = integer(row, "component_mask", line)
        if component_mask != COMPONENT_MASK_ALL:
            incomplete_mapping += 1
            continue
        drive = integer(row, "actual_drive_id", line)
        path_name = row.get("actual_path", "")
        if not path_name:
            unknown += 1
            continue
        # Non-observed rows remain visible in evidence_counts, but they are
        # not measurements.  Keeping them out of prediction/tail summaries
        # prevents theoretical, unavailable, or otherwise unknown evidence
        # from being mistaken for measured runtime behavior.
        if state != "OBSERVED":
            continue
        need = integer(row, "consumer_need_ns", line)
        predicted = integer(row, "predicted_ready_ns", line)
        uncertainty = integer(row, "predicted_uncertainty_ns", line)
        ready = integer(row, "ready_ns", line)
        if not predicted or not ready:
            unknown += 1
        else:
            error = ready - predicted
            guarded = predicted + uncertainty
            key = (drive, path_name)
            groups.setdefault(key, []).append(error)
            actual_miss = int(bool(need and ready > need))
            predicted_miss = int(bool(need and predicted > need))
            guarded_miss = int(bool(need and guarded > need))
            group_deadlines.setdefault(key, []).append(
                (actual_miss, predicted_miss, guarded_miss))
            path_parts = path_name.split("|")
            if len(path_parts) >= 2:
                try:
                    controller_id = int(path_parts[1], 10)
                except ValueError:
                    controller_id = None
                if controller_id is not None and controller_id >= 0:
                    controller_groups.setdefault(controller_id, []).append(error)
                    controller_deadlines.setdefault(controller_id, []).append(
                        (actual_miss, predicted_miss, guarded_miss))
            deadline_misses += actual_miss
            predicted_misses += predicted_miss
            guarded_misses += guarded_miss
            all_errors.append(error)

        # Candidate extraction is intentionally independent of prediction
        # error.  It uses only completed, successful, fully mapped observed
        # loads and preserves the exact runtime scope.
        latency = service_latency_ns(row, line)
        inflight = pipe_integers(row, "path_inflight", line)
        explicit_concurrency = optional_integer(row, "concurrency", line)
        if (explicit_concurrency is not None and inflight
                and (explicit_concurrency < 1
                     or explicit_concurrency != max(inflight))):
            fail(f"line {line}: concurrency disagrees with path_inflight")
        if (state == "OBSERVED" and done and result == 0 and path_name
                and latency is not None and latency > 0 and inflight
                and all(value > 0 for value in inflight)):
            model_id = optional_integer(row, "model_id", line) or 0
            cache_condition = row.get("cache_condition", "UNKNOWN") or "UNKNOWN"
            topology_hash = optional_integer(row, "topology_hash", line) or 0
            profile_hash = optional_integer(row, "profile_hash", line) or 0
            run_id = row.get("run_id", "") or ""
            candidate_inputs.append({
                "scope_key": (
                    model_id, path_name, integer(row, "bytes", line),
                    request_type(row), integer(row, "component_mask", line),
                    cache_condition, topology_hash, profile_hash),
                "path": path_name,
                "model_id": model_id,
                "bytes": integer(row, "bytes", line),
                "request_type": request_type(row),
                "component_mask": integer(row, "component_mask", line),
                "cache_condition": cache_condition,
                "topology_hash": topology_hash,
                "profile_hash": profile_hash,
                "run_id": run_id,
                "concurrency": (explicit_concurrency
                                 if explicit_concurrency is not None
                                 else max(inflight)),
                "latency_ns": latency,
                "throughput_bytes_per_s": integer(row, "bytes", line)
                    * 1_000_000_000 // latency,
            })

    by_path = []
    for (drive, path_name), errors in sorted(groups.items()):
        key = (drive, path_name)
        vectors = group_deadlines.get(key, [])
        by_path.append({
            "actual_drive_id": drive,
            "actual_path": path_name,
            **metric(
                errors,
                sum(value[0] for value in vectors),
                sum(value[1] for value in vectors),
                sum(value[2] for value in vectors),
            ),
        })

    by_controller = []
    for controller_id, errors in sorted(controller_groups.items()):
        vectors = controller_deadlines.get(controller_id, [])
        by_controller.append({
            "controller_id": controller_id,
            **metric(
                errors,
                sum(value[0] for value in vectors),
                sum(value[1] for value in vectors),
                sum(value[2] for value in vectors),
            ),
        })

    return {
        "schema": "m3-shadow-runtime-report-v2",
        "source": str(source_paths[0]) if len(source_paths) == 1
                  else [str(item) for item in source_paths],
        "sources": [
            {"path": str(item), "sha256": sha256_file(item)}
            for item in source_paths if item.exists()
        ],
        "tool_sha256": sha256_file(Path(__file__)),
        "rows_total": total,
        "rows_completed": completed,
        "rows_successful": successful,
        "rows_failed": failed,
        "rows_incomplete": incomplete,
        "rows_unknown_prediction": unknown,
        "rows_incomplete_mapping": incomplete_mapping,
        "evidence_counts": evidence_counts,
        "overall": metric(all_errors, deadline_misses, predicted_misses,
                           guarded_misses),
        "by_drive_path": by_path,
        "by_controller": by_controller,
        "runtime_candidates": runtime_candidates(
            candidate_inputs, max_tail_multiplier, min_samples, tail_percentile),
        "candidate_policy": {
            "status": "CANDIDATE_ONLY",
            "active_admission_allowed": False,
            "requires_repeated_observed_successes": True,
            "requires_two_independent_run_ids": True,
            "requires_contiguous_observed_levels": True,
            "unknown_and_unavailable_are_not_capacity": True,
            "max_tail_multiplier": max_tail_multiplier,
            "min_samples": min_samples,
            "tail_percentile": int(tail_percentile * 100),
        },
    }


def runtime_candidates(rows: list[dict[str, Any]], max_tail_multiplier: float = 1.5,
                       min_samples: int = 2, tail_percentile: float = 0.95
                       ) -> list[dict[str, Any]]:
    """Derive only observed, downward-prefix candidate capacities.

    A candidate is scoped by the exact path, model, byte size, request type,
    component mapping, and cache label.  The largest observed concurrency is
    eligible only when every lower observed level has enough successful
    samples and its latency tail stays within the baseline multiplier.  No
    missing level or specification value is filled in.
    """
    if max_tail_multiplier < 1.0 or min_samples < 1:
        raise ValueError("invalid runtime candidate policy")

    grouped: dict[tuple[Any, ...], dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["scope_key"], {}).setdefault(
            row["concurrency"], []).append(row)

    candidates: list[dict[str, Any]] = []
    for scope_key, levels in sorted(grouped.items(), key=lambda item: item[0]):
        ordered = sorted(levels)
        baseline_level = ordered[0]
        baseline_rows = levels[baseline_level]
        baseline_runs = {row["run_id"] for row in baseline_rows if row["run_id"]}
        if len(baseline_rows) < min_samples or len(baseline_runs) < 2:
            continue
        baseline_tail = percentile(
            [row["latency_ns"] for row in baseline_rows], tail_percentile)
        allowed = baseline_tail * max_tail_multiplier
        candidate_level: int | None = None
        candidate_rows: list[dict[str, Any]] = []
        for level in ordered:
            samples = levels[level]
            runs = {row["run_id"] for row in samples if row["run_id"]}
            if len(samples) < min_samples or len(runs) < 2:
                break
            if any(missing not in levels
                   for missing in range(baseline_level, level + 1)):
                break
            tail = percentile([row["latency_ns"] for row in samples],
                              tail_percentile)
            if tail > allowed:
                break
            candidate_level = level
            candidate_rows = samples
        if candidate_level is None:
            continue
        representative = candidate_rows
        baseline_p95 = percentile(
            [row["latency_ns"] for row in baseline_rows], 0.95)
        baseline_p99 = percentile(
            [row["latency_ns"] for row in baseline_rows], 0.99)
        candidate_p95 = percentile(
            [row["latency_ns"] for row in representative], 0.95)
        candidate_p99 = percentile(
            [row["latency_ns"] for row in representative], 0.99)
        candidates.append({
            "status": "CANDIDATE_ONLY",
            "active_admission_allowed": False,
            "evidence_state": "OBSERVED",
            "confidence": "REPEATED_INDEPENDENT_RUNS",
            "scope": {
                "model_id": scope_key[0],
                "actual_path": scope_key[1],
                "bytes": scope_key[2],
                "request_type": scope_key[3],
                "component_mask": scope_key[4],
                "cache_condition": scope_key[5],
                "topology_hash": scope_key[6],
                "profile_hash": scope_key[7],
            },
            "max_inflight_candidate": candidate_level,
            "candidate_sample_count": len(representative),
            "independent_run_count": len({row["run_id"] for row in representative}),
            "independent_run_ids": sorted({row["run_id"] for row in representative}),
            "baseline_latency_ns_p95": baseline_p95,
            "baseline_latency_ns_p99": baseline_p99,
            "candidate_latency_ns_p95": candidate_p95,
            "candidate_latency_ns_p99": candidate_p99,
            "baseline_latency_ns_tail": baseline_p95 if tail_percentile == 0.95
                                        else baseline_p99,
            "candidate_latency_ns_tail": candidate_p95 if tail_percentile == 0.95
                                         else candidate_p99,
            "allowed_latency_ns_tail": int(allowed),
            "observed_levels": ordered,
            "tail_percentile": int(tail_percentile * 100),
            "policy": "largest observed contiguous downward-prefix level, "
                      "replicated across runs, within tail bound; no extrapolation",
        })
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, action="append", type=Path,
                        help="runtime trace CSV; repeat for independent runs")
    parser.add_argument("--max-tail-multiplier", type=float, default=1.5)
    parser.add_argument("--min-samples", type=int, default=2)
    parser.add_argument("--tail-percentile", choices=("p95", "p99"),
                        default="p95")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_tail_multiplier < 1.0 or args.min_samples < 1:
        fail("invalid runtime candidate policy")
    encoded = json.dumps(
        analyze(args.trace, max_tail_multiplier=args.max_tail_multiplier,
                min_samples=args.min_samples,
                tail_percentile=int(args.tail_percentile[1:]) / 100.0),
        indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
