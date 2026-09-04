#!/usr/bin/env python3
"""Analyze the per-call CPU/GPU overlap trace.

The trace deliberately combines host monotonic timestamps with CUDA-event
durations.  ``gpu_end_inferred_ms`` is therefore a heuristic interval, useful
for finding queue pressure but not a replacement for Nsight or synchronized
cross-clock tracing.

The optional island model is a small list-scheduling projection.  It uses the
observed host issue timestamps as arrivals and the measured GPU event duration
as service time.  It does not claim that the current runtime has multiple
GPU queues, nor does it model PCIe transfers or expert placement.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def number(row, key, default=0.0):
    try:
        value = float(row.get(key, default) or default)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(row, key, default=0):
    try:
        return int(float(row.get(key, default) or default))
    except (TypeError, ValueError):
        return default


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = fraction * (len(values) - 1)
    lo = int(pos)
    hi = min(len(values) - 1, lo + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def mean(values):
    return statistics.fmean(values) if values else 0.0


def load(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def normalize(raw):
    result = []
    for row in raw:
        issue = number(row, "issue_start_ms")
        event = number(row, "gpu_event_ms")
        gpu_end = number(row, "gpu_end_inferred_ms")
        take_start = number(row, "take_start_ms")
        take_end = number(row, "take_end_ms")
        host_gpu_lower = number(row, "gpu_end_host_lower_ms",
                                number(row, "gpu_end_host_ms"))
        host_gpu_upper = number(row, "gpu_end_host_upper_ms",
                                number(row, "gpu_end_host_ms"))
        host_reduce_lower = number(row, "reduce_end_host_lower_ms",
                                   number(row, "reduce_end_host_ms"))
        host_reduce_upper = number(row, "reduce_end_host_upper_ms",
                                   number(row, "reduce_end_host_ms"))
        if gpu_end <= 0 and event > 0:
            gpu_end = issue + event
        row = dict(row)
        row.update({
            "_token": integer(row, "decode_token"),
            "_layer": integer(row, "layer"),
            "_issue": issue,
            "_event": event,
            "_gpu_end": gpu_end,
            "_host_gpu_lower": host_gpu_lower,
            "_host_gpu_upper": host_gpu_upper,
            "_host_reduce_lower": host_reduce_lower,
            "_host_reduce_upper": host_reduce_upper,
            "_take_start": take_start,
            "_take_end": take_end,
            # Positive values mean the inferred GPU interval extends into
            # qt_take.  This is the most direct backlog indicator available
            # without a cross-clock GPU/host trace.
            "_gpu_outstanding_at_take": max(0.0, gpu_end - take_start),
            "_host_gpu_outstanding_lower_at_take": max(0.0, host_gpu_lower - take_start)
                if host_gpu_lower > 0 else 0.0,
            "_host_gpu_outstanding_upper_at_take": max(0.0, host_gpu_upper - take_start)
                if host_gpu_upper > 0 else 0.0,
        })
        result.append(row)
    return sorted(result, key=lambda r: (r["_token"], r["_layer"], r["_issue"]))


def call_summary(row):
    return {
        "token": row["_token"],
        "layer": row["_layer"],
        "gpu_routes": integer(row, "gpu_routes"),
        "cpu_routes": integer(row, "cpu_routes"),
        "issue_start_ms": row["_issue"],
        "issue_end_ms": number(row, "issue_end_ms"),
        "cpu_start_ms": number(row, "cpu_start_ms"),
        "cpu_end_ms": number(row, "cpu_end_ms"),
        "take_start_ms": row["_take_start"],
        "take_end_ms": row["_take_end"],
        "gpu_event_ms": row["_event"],
        "gpu_sync_ms": number(row, "gpu_sync_ms"),
        "cpu_window_ms": number(row, "cpu_window_ms"),
        "issue_to_complete_ms": number(row, "issue_to_complete_ms"),
        "gpu_end_inferred_ms": row["_gpu_end"],
        "gpu_outstanding_at_take_ms": row["_gpu_outstanding_at_take"],
        "gpu_end_host_lower_ms": row["_host_gpu_lower"],
        "gpu_end_host_upper_ms": row["_host_gpu_upper"],
        "reduce_end_host_lower_ms": row["_host_reduce_lower"],
        "reduce_end_host_upper_ms": row["_host_reduce_upper"],
        "gpu_start_host_lower_ms": (row["_host_gpu_lower"] - row["_event"]
                                     if row["_host_gpu_lower"] > 0 else 0.0),
        "gpu_start_host_upper_ms": (row["_host_gpu_upper"] - row["_event"]
                                     if row["_host_gpu_upper"] > 0 else 0.0),
        "gpu_outstanding_at_take_host_lower_ms": row["_host_gpu_outstanding_lower_at_take"],
        "gpu_outstanding_at_take_host_upper_ms": row["_host_gpu_outstanding_upper_at_take"],
        "qt_take_ms": number(row, "qt_take_ms"),
    }


def summarize_group(group):
    values = {key: [number(row, key) for row in group]
              for key in ("gpu_event_ms", "gpu_sync_ms", "cpu_window_ms",
                          "issue_to_complete_ms", "qt_take_ms")}
    outstanding = [row["_gpu_outstanding_at_take"] for row in group]
    host_outstanding_lower = [row["_host_gpu_outstanding_lower_at_take"] for row in group]
    host_outstanding_upper = [row["_host_gpu_outstanding_upper_at_take"] for row in group]
    return {
        "calls": len(group),
        "gpu_routes": sum(integer(row, "gpu_routes") for row in group),
        "cpu_routes": sum(integer(row, "cpu_routes") for row in group),
        "gpu_event_ms_sum": sum(values["gpu_event_ms"]),
        "gpu_event_ms_mean": mean(values["gpu_event_ms"]),
        "gpu_sync_ms_mean": mean(values["gpu_sync_ms"]),
        "cpu_window_ms_sum": sum(values["cpu_window_ms"]),
        "issue_to_complete_ms_mean": mean(values["issue_to_complete_ms"]),
        "qt_take_ms_sum": sum(values["qt_take_ms"]),
        "gpu_outstanding_at_take_ms_sum": sum(outstanding),
        "gpu_outstanding_at_take_ms_max": max(outstanding, default=0.0),
        "gpu_outstanding_calls": sum(value > 0.05 for value in outstanding),
        "gpu_outstanding_at_take_host_lower_ms_sum": sum(host_outstanding_lower),
        "gpu_outstanding_at_take_host_upper_ms_sum": sum(host_outstanding_upper),
        "gpu_outstanding_at_take_host_upper_ms_max": max(host_outstanding_upper, default=0.0),
        "gpu_outstanding_host_calls": sum(value > 0.05 for value in host_outstanding_upper),
        "qt_take_ms_p95": percentile(values["qt_take_ms"], 0.95),
        "issue_to_complete_ms_p95": percentile(values["issue_to_complete_ms"], 0.95),
    }


def list_schedule(rows, gpu_count):
    """Project GPU service onto N identical islands in issue order."""
    available = [float("-inf")] * gpu_count
    assignments = []
    for row in rows:
        arrival = row["_issue"]
        island = min(range(gpu_count), key=lambda i: available[i])
        start = max(arrival, available[island])
        end = start + row["_event"]
        available[island] = end
        assignments.append((row, island, start, end, max(0.0, start - arrival)))
    return assignments


def schedule_summary(rows, gpu_count):
    assignments = list_schedule(rows, gpu_count)
    if not assignments:
        return {"gpus": gpu_count, "calls": 0}
    queue_wait = [item[4] for item in assignments]
    first_issue = min(row["_issue"] for row, *_ in assignments)
    last_end = max(item[3] for item in assignments)
    return {
        "gpus": gpu_count,
        "calls": len(assignments),
        "queue_wait_ms_sum": sum(queue_wait),
        "queue_wait_ms_mean": mean(queue_wait),
        "queue_wait_ms_p95": percentile(queue_wait, 0.95),
        "queued_calls": sum(value > 0.05 for value in queue_wait),
        "gpu_tail_ms": last_end - first_issue,
        "gpu_service_ms": sum(row["_event"] for row, *_ in assignments),
        "parallelism": (sum(row["_event"] for row, *_ in assignments) /
                         max(1e-9, last_end - first_issue)),
        "island_service_ms": [
            sum(row["_event"] for row, island, *_ in assignments if island == i)
            for i in range(gpu_count)
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="QTIER_OVERLAP_FILE CSV")
    ap.add_argument("--out", required=True, help="machine-readable JSON output")
    ap.add_argument("--calls-out", default="", help="normalized call CSV")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    rows = normalize(load(args.input))
    if not rows:
        raise SystemExit("overlap trace has no rows")
    tokens = defaultdict(list)
    layers = defaultdict(list)
    for row in rows:
        tokens[row["_token"]].append(row)
        layers[row["_layer"]].append(row)

    token_report = {}
    for token, group in sorted(tokens.items()):
        first_issue = min(row["_issue"] for row in group)
        last_take = max(row["_take_end"] for row in group)
        first_gpu = min(row["_issue"] for row in group)
        gpu_ends = [row["_gpu_end"] for row in group if row["_gpu_end"] > 0]
        token_report[str(token)] = {
            **summarize_group(group),
            "host_span_ms": last_take - first_issue,
            "inferred_gpu_span_ms": max(gpu_ends, default=first_gpu) - first_gpu,
            "host_span_minus_qt_take_sum_ms":
                (last_take - first_issue) - sum(number(row, "qt_take_ms") for row in group),
            "gpu_schedule": {
                str(count): schedule_summary(group, count)
                for count in args.gpus if count > 0
            },
        }

    layer_report = {
        str(layer): summarize_group(group)
        for layer, group in sorted(layers.items())
    }
    overall = summarize_group(rows)
    overall.update({
        "tokens": len(tokens),
        "layers": len(layers),
        "host_span_ms": max(row["_take_end"] for row in rows) -
                        min(row["_issue"] for row in rows),
        "gpu_schedule": {
            str(count): schedule_summary(rows, count)
            for count in args.gpus if count > 0
        },
    })

    result = {
        "inputs": {"overlap_csv": args.input, "gpus": args.gpus},
        "definitions": {
            "gpu_outstanding_at_take_ms":
                "max(0, gpu_end_inferred_ms - take_start_ms); host/CUDA clock heuristic",
            "gpu_outstanding_at_take_host_ms":
                "lower/upper max(0, gpu_end_host_lower/upper_ms - take_start_ms); same-host-clock query bounds",
            "gpu_schedule":
                "list scheduling by measured issue timestamp and gpu_event_ms; diagnostic only",
            "gpu_tail_ms": "last projected GPU completion - first issue timestamp",
        },
        "overall": overall,
        "tokens": token_report,
        "layers": layer_report,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

    if args.calls_out:
        fields = ["token", "layer", "gpu_routes", "cpu_routes", "issue_start_ms",
                  "issue_end_ms", "cpu_start_ms", "cpu_end_ms", "take_start_ms",
                  "take_end_ms", "gpu_event_ms", "gpu_sync_ms", "cpu_window_ms",
                  "issue_to_complete_ms", "gpu_end_inferred_ms",
                  "gpu_outstanding_at_take_ms", "gpu_end_host_lower_ms",
                  "gpu_end_host_upper_ms", "reduce_end_host_lower_ms",
                  "reduce_end_host_upper_ms", "gpu_start_host_lower_ms",
                  "gpu_start_host_upper_ms",
                  "gpu_outstanding_at_take_host_lower_ms",
                  "gpu_outstanding_at_take_host_upper_ms", "qt_take_ms"]
        with Path(args.calls_out).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(call_summary(row))

    print(f"trace={args.input} calls={len(rows)} tokens={len(tokens)} layers={len(layers)}")
    print(f"host_span={overall['host_span_ms']:.3f} ms "
          f"gpu_event_sum={overall['gpu_event_ms_sum']:.3f} ms "
          f"qt_take_sum={overall['qt_take_ms_sum']:.3f} ms")
    print(f"inferred_gpu_outstanding_at_take="
          f"{overall['gpu_outstanding_at_take_ms_sum']:.3f} ms "
          f"({overall['gpu_outstanding_calls']} calls)")
    if overall["gpu_outstanding_at_take_host_upper_ms_sum"] or overall["gpu_outstanding_host_calls"]:
        print(f"host_clock_gpu_outstanding_at_take="
              f"{overall['gpu_outstanding_at_take_host_lower_ms_sum']:.3f}.."
              f"{overall['gpu_outstanding_at_take_host_upper_ms_sum']:.3f} ms "
              f"({overall['gpu_outstanding_host_calls']} calls)")
    for count in args.gpus:
        if count <= 0:
            continue
        item = overall["gpu_schedule"][str(count)]
        print(f"projected_gpu{count}: tail={item['gpu_tail_ms']:.3f} ms "
              f"queue={item['queue_wait_ms_sum']:.3f} ms "
              f"queued_calls={item['queued_calls']}")
    for layer, item in sorted(layer_report.items(),
                              key=lambda pair: pair[1]["gpu_outstanding_at_take_ms_sum"],
                              reverse=True)[:5]:
        print(f"layer {layer}: cpu={item['cpu_window_ms_sum']:.3f} ms "
              f"gpu={item['gpu_event_ms_sum']:.3f} ms "
              f"qt_take={item['qt_take_ms_sum']:.3f} ms "
              f"outstanding={item['gpu_outstanding_at_take_ms_sum']:.3f} ms")


if __name__ == "__main__":
    main()
