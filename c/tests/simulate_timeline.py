#!/usr/bin/env python3
"""Project resident-placement choices onto a measured host/GPU timeline.

This is deliberately a first-order scheduler model.  It preserves the
non-MoE gaps between measured layer calls, estimates changed CPU/GPU service
from routing and placement, and schedules GPU work on one or more identical
islands.  It is intended to reject bad placement objectives before changing
the runtime; it is not a performance prediction guarantee.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def num(row, key, default=0.0):
    try:
        value = float(row.get(key, default) or default)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def integer(row, key, default=0):
    return int(num(row, key, default))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def placement(path):
    return {(integer(row, "layer"), integer(row, "expert"))
            for row in read_csv(path) if integer(row, "gpu_resident")}


def trace_rows(path):
    result = []
    for row in read_csv(path):
        item = dict(row)
        item.update({
            "token": integer(row, "decode_token"),
            "layer": integer(row, "layer"),
            "issue": num(row, "issue_start_ms"),
            "issue_enqueue": max(0.0, num(row, "issue_end_ms") -
                                  num(row, "issue_start_ms")),
            "take_end": num(row, "take_end_ms"),
            "gpu_end": num(row, "gpu_end_inferred_ms"),
            "gpu_ms": num(row, "gpu_event_ms"),
            "cpu_ms": num(row, "cpu_window_ms"),
            "routes": integer(row, "gpu_routes") + integer(row, "cpu_routes"),
            "gpu_routes": integer(row, "gpu_routes"),
            "cpu_routes": integer(row, "cpu_routes"),
        })
        result.append(item)
    return sorted(result, key=lambda row: (row["token"], row["layer"], row["issue"]))


def model_parameters(trace, expert_rows, layer_rows):
    by_layer = defaultdict(list)
    for row in trace:
        by_layer[row["layer"]].append(row)
    expert_by_layer = defaultdict(list)
    for row in expert_rows:
        expert_by_layer[integer(row, "layer")].append(row)
    layers = {integer(row, "layer"): row for row in layer_rows}
    params = {}
    for layer, calls in by_layer.items():
        route_total = sum(row["routes"] for row in calls)
        gpu_routes = sum(row["gpu_routes"] for row in calls)
        gpu_unit = (sum(row["gpu_ms"] for row in calls) / gpu_routes
                    if gpu_routes else 0.0)
        fallback = [row for row in expert_by_layer[layer]
                    if integer(row, "cpu_fallback")]
        fallback_count = sum(integer(row, "cpu_fallback") for row in fallback)
        fallback_ms = sum(num(row, "cpu_get_ms") + num(row, "cpu_matmul_ms")
                          for row in fallback)
        cpu_unit = fallback_ms / fallback_count if fallback_count else 0.0
        calls_count = max(1, integer(layers.get(layer, {}), "calls", len(calls)))
        shared = num(layers.get(layer, {}), "shared_ms") / calls_count
        if not cpu_unit:
            cpu_unit = (sum(row["cpu_ms"] for row in calls) /
                        max(1, sum(row["cpu_routes"] for row in calls)))
        # Fit the measured CPU window with shared/base work plus fallback
        # routes.  A negative residual means the per-route estimate is too
        # large, so clamp the base rather than inventing negative work.
        base_values = [row["cpu_ms"] - row["cpu_routes"] * cpu_unit
                       for row in calls]
        base = max(0.0, statistics.fmean(base_values) if base_values else 0.0)
        if shared > base:
            base = shared
        total_invocations = sum(integer(row, "invocations")
                                for row in expert_by_layer[layer])
        params[layer] = {
            "route_total": route_total,
            "gpu_unit_ms_per_route": gpu_unit,
            "cpu_unit_ms_per_route": cpu_unit,
            "cpu_base_ms_per_call": base,
            "measured_gpu_ms": sum(row["gpu_ms"] for row in calls),
            "measured_cpu_ms": sum(row["cpu_ms"] for row in calls),
            "total_invocations": total_invocations,
        }
    return params, expert_by_layer


def policy_fraction(selected, expert_by_layer):
    result = {}
    for layer, rows in expert_by_layer.items():
        total = sum(integer(row, "invocations") for row in rows)
        hits = sum(integer(row, "invocations") for row in rows
                   if (layer, integer(row, "expert")) in selected)
        result[layer] = (hits / total if total else 0.0)
    return result


def project(trace, params, fractions, gpu_count):
    available = [0.0] * gpu_count
    host_ready = None
    previous = None
    predicted = []
    for row in trace:
        if host_ready is None:
            host_ready = row["issue"]
        else:
            # Preserve measured non-MoE/recurrent work between qt_take and the
            # next issue.  Only the expert completion portion is rescheduled.
            gap = max(0.0, row["issue"] - previous["take_end"])
            host_ready += gap
        p = params[row["layer"]]
        fraction = fractions.get(row["layer"], 0.0)
        gpu_routes = row["routes"] * fraction
        cpu_routes = row["routes"] - gpu_routes
        gpu_ms = gpu_routes * p["gpu_unit_ms_per_route"]
        cpu_ms = p["cpu_base_ms_per_call"] + cpu_routes * p["cpu_unit_ms_per_route"]
        issue = host_ready
        cpu_end = issue + row["issue_enqueue"] + cpu_ms
        island = min(range(gpu_count), key=lambda index: available[index])
        gpu_start = max(issue, available[island])
        gpu_end = gpu_start + gpu_ms
        # This is measured post-completion overhead (D2H/API/reduction
        # remainder), with any inferred GPU wait already represented by max().
        actual_cpu_end = num(row, "cpu_end_ms")
        actual_gpu_end = row["gpu_end"]
        actual_completion = max(actual_cpu_end, actual_gpu_end)
        post = max(0.0, row["take_end"] - actual_completion)
        completion = max(cpu_end, gpu_end) + post
        predicted.append({
            "token": row["token"], "layer": row["layer"],
            "issue_ms": issue, "cpu_routes": cpu_routes,
            "gpu_routes": gpu_routes, "cpu_ms": cpu_ms, "gpu_ms": gpu_ms,
            "gpu_island": island, "gpu_start_ms": gpu_start,
            "gpu_end_ms": gpu_end,
            "gpu_queue_wait_ms": max(0.0, gpu_start - issue),
            "completion_ms": completion,
        })
        available[island] = gpu_end
        host_ready = completion
        previous = row

    by_token = defaultdict(list)
    by_layer = defaultdict(list)
    for row in predicted:
        by_token[row["token"]].append(row)
        by_layer[row["layer"]].append(row)
    token_report = {}
    for token, rows in sorted(by_token.items()):
        token_report[str(token)] = {
            "completion_ms": max(row["completion_ms"] for row in rows) -
                             min(row["issue_ms"] for row in rows),
            "gpu_queue_wait_ms": sum(row["gpu_queue_wait_ms"] for row in rows),
            "cpu_ms": sum(row["cpu_ms"] for row in rows),
            "gpu_ms": sum(row["gpu_ms"] for row in rows),
        }
    layer_report = {}
    for layer, rows in sorted(by_layer.items()):
        layer_report[str(layer)] = {
            "gpu_routes": sum(row["gpu_routes"] for row in rows),
            "cpu_routes": sum(row["cpu_routes"] for row in rows),
            "gpu_ms": sum(row["gpu_ms"] for row in rows),
            "cpu_ms": sum(row["cpu_ms"] for row in rows),
            "gpu_queue_wait_ms": sum(row["gpu_queue_wait_ms"] for row in rows),
            "max_gpu_queue_wait_ms": max((row["gpu_queue_wait_ms"] for row in rows),
                                          default=0.0),
        }
    return {
        "gpu_count": gpu_count,
        "tokens": token_report,
        "layers": layer_report,
        "calls": predicted,
        "total_gpu_ms": sum(row["gpu_ms"] for row in predicted),
        "total_cpu_ms": sum(row["cpu_ms"] for row in predicted),
        "total_gpu_queue_wait_ms": sum(row["gpu_queue_wait_ms"] for row in predicted),
        "predicted_trace_span_ms": (max(row["completion_ms"] for row in predicted) -
                                     min(row["issue_ms"] for row in predicted)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--expert-csv", required=True)
    ap.add_argument("--layer-csv", required=True)
    ap.add_argument("--candidate", action="append", required=True,
                    help="name=placement.csv; repeat for each policy")
    ap.add_argument("--gpus", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    trace = trace_rows(args.overlap)
    expert_rows = read_csv(args.expert_csv)
    layer_rows = read_csv(args.layer_csv)
    params, expert_by_layer = model_parameters(trace, expert_rows, layer_rows)
    policies = {}
    for spec in args.candidate:
        if "=" not in spec:
            raise SystemExit(f"candidate must be name=path: {spec}")
        name, path = spec.split("=", 1)
        selected = placement(path)
        fractions = policy_fraction(selected, expert_by_layer)
        policies[name] = {
            "placement_csv": path,
            "resident_slots": len(selected),
            "gpu_hit_rate": (sum(params[layer]["total_invocations"] * fractions.get(layer, 0.0)
                                 for layer in params) /
                             max(1, sum(params[layer]["total_invocations"] for layer in params))),
            "slots_per_layer": {str(layer): sum(1 for key in selected if key[0] == layer)
                                 for layer in sorted(params)},
            "gpu_hit_fraction_by_layer": {str(layer): fractions.get(layer, 0.0)
                                           for layer in sorted(params)},
            "simulations": {
                str(gpus): project(trace, params, fractions, gpus)
                for gpus in args.gpus if gpus > 0
            },
        }

    result = {
        "inputs": {"overlap": args.overlap, "expert_csv": args.expert_csv,
                   "layer_csv": args.layer_csv, "gpus": args.gpus},
        "warning": "host/CUDA clocks and alternate-placement service rates are heuristic",
        "model": {
            "gpu_service": "measured layer gpu_event_ms / current gpu routes",
            "cpu_service": "measured CPU fallback cost plus fitted per-call base",
            "arrival": "measured inter-call gap after prior projected completion",
            "completion": "max(projected CPU end, projected GPU end) + measured post-completion overhead",
            "gpu_assignment": "earliest-available identical island",
        },
        "policies": policies,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, report in policies.items():
        print(f"{name}: slots={report['resident_slots']} "
              f"hit={100*report['gpu_hit_rate']:.2f}%")
        for gpus, sim in report["simulations"].items():
            print(f"  gpu{gpus}: span={sim['predicted_trace_span_ms']:.3f} ms "
                  f"cpu={sim['total_cpu_ms']:.3f} ms gpu={sim['total_gpu_ms']:.3f} ms "
                  f"queue={sim['total_gpu_queue_wait_ms']:.3f} ms")
        for layer in (38, 39):
            item = report["simulations"][str(args.gpus[0])]["layers"].get(str(layer))
            if item:
                print(f"  layer {layer}: cpu={item['cpu_ms']:.3f} ms "
                      f"gpu={item['gpu_ms']:.3f} ms queue={item['gpu_queue_wait_ms']:.3f} ms")


if __name__ == "__main__":
    main()
