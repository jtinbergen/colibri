#!/usr/bin/env python3
"""Join a Qwen resident-call trace with CUPTI activity timestamps.

The overlap CSV contains the host-side call boundaries and layer/token IDs.
The CUPTI CSV contains absolute host-normalized queued/start/end timestamps.
Kernels are assigned to a call by their queued timestamp inside that call's
host issue interval.  A second window list reports work that executes during
the call, including work queued by an earlier call; that distinction exposes
the device backlog instead of hiding it in qt_take.
"""

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from pathlib import Path


def f(row, key, default=0.0):
    try:
        value = float(row.get(key, default) or default)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def i(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def activity_time(row, preferred, fallback):
    value = f(row, preferred)
    return value if value > 0 else f(row, fallback)


def activity_summary(row):
    kind = row.get("kind", "")
    start = activity_time(row, "host_start_ns", "start_ns")
    end = activity_time(row, "host_end_ns", "end_ns")
    queued = activity_time(row, "host_queued_ns", "queued_ns")
    submitted = activity_time(row, "host_submitted_ns", "submitted_ns")
    return {
        "record": i(row, "record"),
        "kind": kind,
        "name": row.get("name", ""),
        "start_ns": start,
        "end_ns": end,
        "queued_ns": queued,
        "submitted_ns": submitted,
        "device": i(row, "device"),
        "stream": i(row, "stream"),
        "correlation": i(row, "correlation"),
        "cbid": i(row, "cbid"),
        "bytes": i(row, "bytes"),
        "copy_kind": i(row, "copy_kind"),
        "src_kind": i(row, "src_kind"),
        "dst_kind": i(row, "dst_kind"),
    }


def in_interval(value, lo, hi):
    return value > 0 and lo <= value <= hi


def overlaps(row, lo, hi):
    return row["start_ns"] < hi and row["end_ns"] > lo


def compact_activity(row):
    result = dict(row)
    for key in ("start_ns", "end_ns", "queued_ns", "submitted_ns"):
        if result[key]:
            result[key] = int(result[key])
    return result


def interval_slice(records, starts, lo, hi, backtrack=64):
    """Return records intersecting [lo, hi) using a start-time index."""
    if not records or hi <= lo:
        return []
    right = bisect.bisect_left(starts, hi)
    left = max(0, bisect.bisect_left(starts, lo) - backtrack)
    return [row for row in records[left:right] if overlaps(row, lo, hi)]


def join_call(call, indexes):
    issue_lo = f(call, "issue_start_ms") * 1_000_000.0
    issue_hi = f(call, "issue_end_ms") * 1_000_000.0
    take_lo = f(call, "take_start_ms") * 1_000_000.0
    take_hi = f(call, "take_end_ms") * 1_000_000.0
    window_lo = issue_lo
    window_hi = max(take_hi, issue_hi)

    kernels, kernel_starts, kernel_queue = indexes["kernels"]
    memcpys, memcpy_starts = indexes["memcpys"]
    apis, api_starts = indexes["apis"]
    qvalues = [item[0] for item in kernel_queue]
    qright = bisect.bisect_right(qvalues, issue_hi)
    qleft = bisect.bisect_left(qvalues, issue_lo)
    issued_kernels = [kernels[item[1]] for item in kernel_queue[qleft:qright]
                      if in_interval(kernels[item[1]]["queued_ns"] or
                                     kernels[item[1]]["start_ns"], issue_lo, issue_hi)]
    executed_kernels = interval_slice(kernels, kernel_starts, window_lo, window_hi)
    memcpys = interval_slice(memcpys, memcpy_starts, window_lo, window_hi)
    apis = interval_slice(apis, api_starts, window_lo, window_hi)
    take_memcpys = interval_slice(memcpys, [a["start_ns"] for a in memcpys], take_lo, take_hi)
    take_apis = interval_slice(apis, [a["start_ns"] for a in apis], take_lo, take_hi)

    queued = [a["queued_ns"] for a in issued_kernels if a["queued_ns"] > 0]
    starts = [a["start_ns"] for a in issued_kernels if a["start_ns"] > 0]
    ends = [a["end_ns"] for a in issued_kernels if a["end_ns"] > 0]
    d2h = [a for a in memcpys if a["copy_kind"] == 2]
    take_d2h = [a for a in take_memcpys if a["copy_kind"] == 2]
    device_copies = [a for a in memcpys if a["kind"] == "memcpy2" or
                     a["copy_kind"] in (8, 10)]
    kernel_names = Counter(a["name"] for a in issued_kernels)
    first_queued = min(queued, default=0)
    first_start = min(starts, default=0)
    last_end = max(ends, default=0)
    last_issued_end = max((a["end_ns"] for a in issued_kernels), default=0)
    queue_delays = [(a["start_ns"] - a["queued_ns"]) / 1000.0
                    for a in issued_kernels if a["queued_ns"] and a["start_ns"]]
    return {
        "row": i(call, "row"),
        "decode_token": i(call, "decode_token"),
        "layer": i(call, "layer"),
        "layer_call": i(call, "layer_call"),
        "gpu_routes": i(call, "gpu_routes"),
        "cpu_routes": i(call, "cpu_routes"),
        "host": {
            "issue_start_ms": f(call, "issue_start_ms"),
            "issue_end_ms": f(call, "issue_end_ms"),
            "take_start_ms": f(call, "take_start_ms"),
            "take_end_ms": f(call, "take_end_ms"),
            "qt_take_ms": f(call, "qt_take_ms"),
        },
        "issued_kernel_count": len(issued_kernels),
        "executed_kernel_count": len(executed_kernels),
        "executed_kernel_count_not_issued_here": max(0, len(executed_kernels) - len(issued_kernels)),
        "first_queued_ns": int(first_queued),
        "first_kernel_start_ns": int(first_start),
        "last_kernel_end_ns": int(last_end),
        "last_issued_kernel_end_ns": int(last_issued_end),
        "first_queue_to_start_us": ((first_start - first_queued) / 1000.0
                                     if first_queued and first_start else 0.0),
        "mean_queue_to_start_us": (sum(queue_delays) / len(queue_delays)
                                    if queue_delays else 0.0),
        "max_queue_to_start_us": max(queue_delays, default=0.0),
        "issued_gpu_end_to_take_start_ms": ((take_lo - last_issued_end) / 1_000_000.0
                                             if last_issued_end else 0.0),
        "issued_gpu_overlap_take_ms": max(0.0, (last_issued_end - take_lo) / 1_000_000.0)
                                       if last_issued_end else 0.0,
        "issued_kernel_service_ms": sum(max(0, a["end_ns"] - a["start_ns"])
                                         for a in issued_kernels) / 1_000_000.0,
        "executed_kernel_service_ms": sum(max(0, a["end_ns"] - a["start_ns"])
                                           for a in executed_kernels) / 1_000_000.0,
        "memcpy_count": len(memcpys),
        "d2h_count": len(d2h),
        "d2h_bytes": sum(a["bytes"] for a in d2h),
        "take_memcpy_count": len(take_memcpys),
        "take_d2h_count": len(take_d2h),
        "take_d2h_bytes": sum(a["bytes"] for a in take_d2h),
        "take_runtime_api_count": sum(a["kind"] == "runtime" for a in take_apis),
        "take_driver_api_count": sum(a["kind"] == "driver" for a in take_apis),
        "device_copy_count": len(device_copies),
        "device_copy_bytes": sum(a["bytes"] for a in device_copies),
        "runtime_api_count": sum(a["kind"] == "runtime" for a in apis),
        "driver_api_count": sum(a["kind"] == "driver" for a in apis),
        "kernel_names": dict(kernel_names),
        "issued_kernels": [compact_activity(a) for a in issued_kernels],
        "window_memcpys": [compact_activity(a) for a in memcpys],
        "take_memcpys": [compact_activity(a) for a in take_memcpys],
        "take_apis": [compact_activity(a) for a in take_apis],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True, help="QTIER_OVERLAP_FILE CSV")
    ap.add_argument("--cupti", required=True, help="COLI_CUDA_CUPTI_FILE CSV")
    ap.add_argument("--out", required=True, help="joined JSON output")
    args = ap.parse_args()

    calls = read_csv(args.overlap)
    activities = [activity_summary(row) for row in read_csv(args.cupti)]
    kernels = sorted((a for a in activities if a["kind"] == "kernel"),
                     key=lambda a: (a["start_ns"], a["end_ns"]))
    memcpys = sorted((a for a in activities if a["kind"] in ("memcpy", "memcpy2")),
                     key=lambda a: (a["start_ns"], a["end_ns"]))
    apis = sorted((a for a in activities if a["kind"] in ("runtime", "driver")),
                  key=lambda a: (a["start_ns"], a["end_ns"]))
    indexes = {
        "kernels": (kernels, [a["start_ns"] for a in kernels],
                    sorted((a["queued_ns"] or a["start_ns"], n)
                           for n, a in enumerate(kernels))),
        "memcpys": (memcpys, [a["start_ns"] for a in memcpys]),
        "apis": (apis, [a["start_ns"] for a in apis]),
    }
    result_calls = [join_call(call, indexes) for call in calls]
    kinds = Counter(a["kind"] for a in activities)
    result = {
        "inputs": {"overlap": args.overlap, "cupti": args.cupti},
        "clock": {
            "join_domain": "CUPTI host_*_ns columns and qwen36 monotonic milliseconds",
            "cupti_host_offset": "applied by backend collector; raw start_ns remains available",
        },
        "activity_counts": dict(kinds),
        "calls": result_calls,
        "summary": {
            "calls": len(result_calls),
            "calls_with_issued_kernels": sum(c["issued_kernel_count"] > 0 for c in result_calls),
            "calls_with_d2h": sum(c["d2h_count"] > 0 for c in result_calls),
            "issued_kernels": sum(c["issued_kernel_count"] for c in result_calls),
            "executed_kernels_in_windows": sum(c["executed_kernel_count"] for c in result_calls),
            "d2h_bytes": sum(c["d2h_bytes"] for c in result_calls),
            "take_d2h_bytes": sum(c["take_d2h_bytes"] for c in result_calls),
            "device_copy_bytes": sum(c["device_copy_bytes"] for c in result_calls),
            "issued_service_ms": sum(c["issued_kernel_service_ms"] for c in result_calls),
            "executed_service_ms": sum(c["executed_kernel_service_ms"] for c in result_calls),
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
