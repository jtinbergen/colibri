#!/usr/bin/env python3
"""Small deterministic regression test for p95/p99 calibration review."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("m3_shadow_calibration_review.py")
SPEC = importlib.util.spec_from_file_location("m3_shadow_calibration_review",
                                               MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def capture() -> dict:
    def row(inflight: int, p95: int, p99: int) -> dict:
        return {
            "group": "drive-101",
            "drive_ids": [101],
            "active_drive_ids": [101],
            "all_group_drives_active": True,
            "inflight": inflight,
            "summary": {
                "latency_ns_p50": 100,
                "latency_ns_p95": p95,
                "latency_ns_p99": p99,
                "latency_ns_max": p99,
                "aggregate_bytes_per_s_p50": 1000,
            },
        }

    return {
        "schema": "m3-shadow-calibration-v1",
        "read_bytes": 65536,
        "measurements": [row(1, 100, 100), row(2, 150, 150),
                         row(4, 150, 250)],
    }


def scalar_capture(name: str, rate: int, target_kind: str | None,
                   read_bytes: int = 65536) -> dict:
    return {
        "schema": "m3-shadow-calibration-v1",
        "read_bytes": read_bytes,
        "measurements": [{
            "group": name,
            "drive_ids": [101],
            "active_drive_ids": [101],
            "all_group_drives_active": True,
            "inflight": 1,
            "target_kind": target_kind,
            "summary": {
                "latency_ns_p50": 100,
                "latency_ns_p95": 100,
                "latency_ns_p99": 100,
                "latency_ns_max": 100,
                "aggregate_bytes_per_s_p50": rate,
            },
        }],
    }


def matrix_capture(read_bytes: int = 65536) -> dict:
    return {
        "schema": "m3-shadow-calibration-v1",
        "read_bytes": read_bytes,
        "measurements": [{
            "measurement_kind": "contention_matrix",
            "group": "shared-upstream-901",
            "drive_ids": [101, 102],
            "active_drive_ids": [101, 102],
            "all_group_drives_active": True,
            "queue_depths": {"101": 1, "102": 1},
            "inflight": 2,
            "summary": {
                "latency_ns_p50": 100,
                "latency_ns_p95": 100,
                "latency_ns_p99": 100,
                "latency_ns_max": 100,
                "aggregate_bytes_per_s_p50": 1200,
                "per_drive": {
                    "101": {"aggregate_bytes_per_s_p50": 500},
                    "102": {"aggregate_bytes_per_s_p50": 700},
                },
            },
        }],
    }


def main() -> None:
    value = capture()
    p95 = MODULE.review([value], 1.6, 95)
    assert p95["tail_percentile"] == 95
    assert p95["results"][0]["max_inflight_candidate"] == 4

    p99 = MODULE.review([value], 1.6, 99)
    assert p99["tail_percentile"] == 99
    assert p99["results"][0]["max_inflight_candidate"] == 2

    matrix = {
        "schema": "m3-shadow-calibration-v1",
        "read_bytes": 65536,
        "measurements": [
            {
                "measurement_kind": "contention_matrix",
                "group": "shared-upstream-901",
                "drive_ids": [101, 102],
                "active_drive_ids": [101, 102],
                "all_group_drives_active": True,
                "queue_depths": {"101": 1, "102": 1},
                "inflight": 2,
                "summary": {
                    "latency_ns_p50": 100,
                    "latency_ns_p95": 100,
                    "latency_ns_p99": 100,
                    "latency_ns_max": 100,
                    "aggregate_bytes_per_s_p50": 1000,
                    "per_drive": {},
                    "fairness_jain_p50": 1.0,
                },
            },
            {
                "measurement_kind": "contention_matrix",
                "group": "shared-upstream-901",
                "drive_ids": [101, 102],
                "active_drive_ids": [101, 102],
                "all_group_drives_active": True,
                "queue_depths": {"101": 2, "102": 2},
                "inflight": 4,
                "summary": {
                    "latency_ns_p50": 150,
                    "latency_ns_p95": 150,
                    "latency_ns_p99": 150,
                    "latency_ns_max": 150,
                    "aggregate_bytes_per_s_p50": 1400,
                    "per_drive": {},
                    "fairness_jain_p50": 0.98,
                },
            },
            {
                "measurement_kind": "contention_matrix",
                "group": "shared-upstream-901",
                "drive_ids": [101, 102],
                "active_drive_ids": [101, 102],
                "all_group_drives_active": True,
                "queue_depths": {"101": 1, "102": 3},
                "inflight": 4,
                "summary": {
                    "latency_ns_p50": 300,
                    "latency_ns_p95": 400,
                    "latency_ns_p99": 400,
                    "latency_ns_max": 400,
                    "aggregate_bytes_per_s_p50": 1200,
                    "per_drive": {},
                    "fairness_jain_p50": 0.7,
                },
            },
        ],
    }
    matrix_review = MODULE.review([matrix], 2.0, 95)
    assert matrix_review["results"][0]["measurement_kind"] == "contention_matrix"
    assert matrix_review["results"][0]["max_inflight_candidate"] == 2
    assert matrix_review["results"][0]["candidate_queue_depths"] == {
        "101": 1, "102": 1}
    assert not matrix_review["results"][0]["observed_downward_envelope"][1][
        "all_observed_partitions_pass"]

    low_specific = scalar_capture("baseline-low", 800, "drive")
    high_specific = scalar_capture("baseline-high", 900, "drive")
    less_specific = scalar_capture("baseline-generic", 100, "controller")
    drive_102 = {
        **scalar_capture("baseline-102", 1000, "drive"),
        "measurements": [{
            **scalar_capture("baseline-102", 1000, "drive")["measurements"][0],
            "drive_ids": [102], "active_drive_ids": [102],
        }],
    }
    matrix = matrix_capture()
    forward = MODULE.review(
        [low_specific, high_specific, less_specific, drive_102, matrix],
        2.0, 95)
    reverse = MODULE.review(
        [matrix, drive_102, less_specific, high_specific, low_specific],
        2.0, 95)
    forward_shared = next(item for item in forward["results"]
                          if item["group"] == "shared-upstream-901")
    reverse_shared = next(item for item in reverse["results"]
                          if item["group"] == "shared-upstream-901")
    assert forward_shared["candidate_retention"] == reverse_shared["candidate_retention"]
    assert forward_shared["candidate_retention"] == {"101": 0.625, "102": 0.7}

    incompatible = scalar_capture("incompatible", 800, "drive", 131072)
    try:
        MODULE.review([low_specific, incompatible], 2.0, 95)
    except SystemExit:
        pass
    else:
        raise AssertionError("incompatible request sizes must be rejected")
    print("m3 shadow calibration review p95/p99: PASS")


if __name__ == "__main__":
    main()
