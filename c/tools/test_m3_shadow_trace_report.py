#!/usr/bin/env python3
"""Deterministic regression test for runtime trace evidence accounting."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("m3_shadow_trace_report.py")
SPEC = importlib.util.spec_from_file_location("m3_shadow_trace_report",
                                               MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def main() -> None:
    rows = [
            {
                "request_id": 1, "actual_drive_id": 101,
                "actual_path": "101|201|301", "consumer_need_ns": 1000,
                "component_mask": 63,
                "predicted_ready_ns": 1000, "predicted_uncertainty_ns": 50,
                "ready_ns": 900, "load_result": 0, "completed": 1,
            },
            {
                "request_id": 2, "actual_drive_id": 101,
                "actual_path": "101|201|301", "consumer_need_ns": 1000,
                "component_mask": 63,
                "predicted_ready_ns": 900, "predicted_uncertainty_ns": 100,
                "ready_ns": 1100, "load_result": 0, "completed": 1,
            },
            {
                "request_id": 3, "actual_drive_id": 101,
                "actual_path": "101|201|301", "consumer_need_ns": 1000,
                "component_mask": 0,
                "predicted_ready_ns": 0, "predicted_uncertainty_ns": 0,
                "ready_ns": 0, "load_result": -1, "completed": 1,
            },
            {
                "request_id": 4, "actual_drive_id": 101,
                "actual_path": "101|201|301", "consumer_need_ns": 1000,
                "component_mask": 63,
                "predicted_ready_ns": 900, "predicted_uncertainty_ns": 0,
                "ready_ns": 0, "load_result": 0, "completed": 0,
            },
            {
                "request_id": 5, "actual_drive_id": 102,
                "actual_path": "102|202", "consumer_need_ns": 3000,
                "component_mask": 63,
                "predicted_ready_ns": 2000, "predicted_uncertainty_ns": 0,
                "ready_ns": 2100, "load_result": 0, "completed": 1,
            },
            {
                "request_id": 6, "actual_drive_id": 103,
                "actual_path": "103|203", "consumer_need_ns": 3000,
                "component_mask": 7,
                "predicted_ready_ns": 2000, "predicted_uncertainty_ns": 0,
                "ready_ns": 2100, "load_result": 0, "completed": 1,
            },
            {
                "request_id": 7, "actual_drive_id": 104,
                "actual_path": "104|204", "consumer_need_ns": 3000,
                "component_mask": 63, "evidence_state": "THEORETICAL",
                "predicted_ready_ns": 2000, "predicted_uncertainty_ns": 0,
                "ready_ns": 2100, "load_result": 0, "completed": 1,
            },
            {
                "request_id": 8, "actual_drive_id": 105,
                "actual_path": "105|205", "consumer_need_ns": 3000,
                "component_mask": 63, "evidence_state": "UNAVAILABLE",
                "predicted_ready_ns": 2000, "predicted_uncertainty_ns": 0,
                "ready_ns": 2100, "load_result": 0, "completed": 1,
            },
    ]
    report = MODULE.analyze(Path("synthetic-trace.csv"), rows)
    assert report["rows_total"] == 8
    assert report["rows_successful"] == 6
    assert report["rows_failed"] == 1
    assert report["rows_incomplete"] == 1
    assert report["rows_incomplete_mapping"] == 1
    assert report["overall"]["samples"] == 3
    assert report["overall"]["ready_error_ns_max_abs"] == 200
    assert report["overall"]["deadline_misses"] == 1
    assert report["by_drive_path"][0]["actual_path"] == "101|201|301"
    assert report["by_drive_path"][0]["samples"] == 2
    assert report["by_drive_path"][1]["actual_drive_id"] == 102
    assert report["by_drive_path"][1]["samples"] == 1
    assert report["by_controller"] == [
        {
            "controller_id": 201,
            "samples": 2,
            "ready_error_ns_p50": -100,
            "ready_error_ns_p95": 200,
            "ready_error_ns_p99": 200,
            "ready_error_ns_max_abs": 200,
            "deadline_misses": 1,
            "predicted_deadline_misses": 0,
            "guarded_deadline_misses": 1,
        },
        {
            "controller_id": 202,
            "samples": 1,
            "ready_error_ns_p50": 100,
            "ready_error_ns_p95": 100,
            "ready_error_ns_p99": 100,
            "ready_error_ns_max_abs": 100,
            "deadline_misses": 0,
            "predicted_deadline_misses": 0,
            "guarded_deadline_misses": 0,
        },
    ]
    assert report["evidence_counts"] == {
        "NOT_APPLICABLE": 0,
        "OBSERVED": 3,
        "THEORETICAL": 1,
        "UNAVAILABLE": 1,
        "UNKNOWN": 3,
    }

    def runtime_row(request_id: int, concurrency: int, latency: int,
                    run_id: str) -> dict:
        return {
            "request_id": request_id, "model_id": 7,
            "run_id": run_id, "topology_hash": 99, "profile_hash": 123,
            "actual_drive_id": 101, "actual_path": "101|201|301",
            "bytes": 1000, "component_mask": 63,
            "path_inflight": f"{concurrency}|{concurrency}|{concurrency}",
            "concurrency": concurrency,
            "request_type": "demand", "evidence_state": "OBSERVED",
            "consumer_need_ns": 1000000, "predicted_ready_ns": 100,
            "predicted_uncertainty_ns": 0, "issue_ns": 1000,
            "ready_ns": 1000 + latency, "service_latency_ns": latency,
            "load_result": 0, "completed": 1,
        }

    candidate_report = MODULE.analyze(
        Path("runtime-trace.csv"),
        [runtime_row(10, 1, 100, "run-a"),
         runtime_row(11, 1, 110, "run-b"),
         runtime_row(12, 2, 130, "run-a"),
         runtime_row(13, 2, 140, "run-b"),
         runtime_row(14, 4, 300, "run-a"),
         runtime_row(15, 4, 320, "run-b")],
        max_tail_multiplier=1.5, min_samples=2, tail_percentile=0.95)
    candidates = candidate_report["runtime_candidates"]
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["max_inflight_candidate"] == 2
    assert candidate["confidence"] == "REPEATED_INDEPENDENT_RUNS"
    assert candidate["active_admission_allowed"] is False
    assert candidate["scope"]["request_type"] == "demand"
    assert candidate["candidate_latency_ns_tail"] == 140
    assert candidate["independent_run_count"] == 2
    assert candidate["scope"]["topology_hash"] == 99
    inconsistent = runtime_row(16, 1, 100, "run-c")
    inconsistent["concurrency"] = 2
    try:
        MODULE.analyze(Path("runtime-trace.csv"), [inconsistent])
    except SystemExit as exc:
        assert "concurrency disagrees" in str(exc)
    else:
        raise AssertionError("inconsistent explicit concurrency was accepted")
    print("m3 shadow runtime trace report: PASS")


if __name__ == "__main__":
    main()
