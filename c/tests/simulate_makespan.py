#!/usr/bin/env python3
"""Overlap-aware offline resident placement simulator.

Uses measured CPU/GPU windows to rank a fixed-capacity placement.  It does not
modify runtime scheduling or placement.
"""

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path


def num(row, key, default=0.0):
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def integer(row, key, default=0):
    return int(num(row, key, default))


def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def median_or(values, fallback):
    values = [v for v in values if math.isfinite(v) and v > 0]
    return statistics.median(values) if values else fallback


def placement(path):
    return {(int(row["layer"]), int(row["expert"]))
            for row in rows(path) if integer(row, "gpu_resident")}


def write_heat(path, selected, layers, experts):
    values = [2 if (layer, expert) in selected else 0
              for layer in range(layers) for expert in range(experts)]
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<III", 0x51544831, layers, experts))
        handle.write(struct.pack("<" + "I" * len(values), *values))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert-csv", required=True)
    ap.add_argument("--layer-csv", required=True)
    ap.add_argument("--placement-csv", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--capacity", type=int, default=3642)
    ap.add_argument("--heat-out", default="",
                    help="write the makespan-balanced placement as QTH1 heat")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    expert_rows = rows(args.expert_csv)
    layer_rows = {int(row["layer"]): row for row in rows(args.layer_csv)}
    current = placement(args.placement_csv)
    keys = [(layer, expert) for layer in range(args.layers)
            for expert in range(args.experts)]
    route = {(int(row["layer"]), int(row["expert"])): integer(row, "invocations")
             for row in expert_rows}
    fallback_costs = {}
    for key in keys:
        row = next((r for r in expert_rows
                    if int(r["layer"]) == key[0] and int(r["expert"]) == key[1]), {})
        count = integer(row, "cpu_fallback")
        fallback_costs[key] = ((num(row, "cpu_get_ms") + num(row, "cpu_matmul_ms")) / count
                               if count else 0.0)
    fallback_default = median_or(list(fallback_costs.values()), 1.0)

    layer = {}
    for number_layer in range(args.layers):
        row = layer_rows.get(number_layer, {})
        calls = max(1, integer(row, "calls"))
        routes = integer(row, "routes")
        shared = num(row, "shared_ms") / calls
        fallback_default_layer = num(row, "cpu_fallback_ms") / max(1, integer(row, "cpu_fallback"))
        fallback_default_layer = fallback_default_layer or fallback_default
        gpu_hits = integer(row, "gpu_hits")
        gpu_cost = num(row, "gpu_event_ms") / gpu_hits if gpu_hits else 0.0
        gpu_cost = gpu_cost or median_or(
            [num(r, "gpu_event_ms") / integer(r, "gpu_hits")
             for r in layer_rows.values() if integer(r, "gpu_hits")], 0.1)
        cpu_window = num(row, "cpu_window_ms") / calls
        gpu_event = num(row, "gpu_event_ms") / calls
        overlap = num(row, "gpu_cpu_overlap_ms") / calls
        overlap_fraction = overlap / min(cpu_window, gpu_event) if min(cpu_window, gpu_event) > 0 else 0.0
        union = cpu_window + gpu_event - overlap
        boundary = max(0.0, num(row, "issue_to_complete_ms") / calls - union)
        layer[number_layer] = {
            "calls": calls, "routes": routes, "shared": shared,
            "fallback_default": fallback_default_layer,
            "gpu_cost": gpu_cost, "overlap_fraction": min(1.0, overlap_fraction),
            "boundary": boundary,
        }

    for key, cost in list(fallback_costs.items()):
        if cost <= 0:
            fallback_costs[key] = layer[key[0]]["fallback_default"]

    def counts(selected):
        hits = [0] * args.layers
        for key in selected:
            hits[key[0]] += route.get(key, 0)
        return hits

    def layer_cost(number_layer, hits):
        spec = layer[number_layer]
        calls = max(1, spec["calls"])
        nonresident = max(0, spec["routes"] - hits) / calls
        # Per-layer average fallback cost is used for the aggregate model;
        # selected candidates still use their measured cost during ranking.
        cpu = spec["shared"] + nonresident * spec["fallback_default"]
        gpu = (hits / calls) * spec["gpu_cost"]
        overlap = spec["overlap_fraction"] * min(cpu, gpu)
        return cpu + gpu - overlap + spec["boundary"]

    def total_cost(selected):
        hit = counts(selected)
        return sum(layer_cost(number_layer, hit[number_layer])
                   for number_layer in range(args.layers))

    def greedy_makespan():
        selected = set()
        hit = [0] * args.layers
        # Allocate one block at a time to the candidate with the largest
        # immediate reduction in sequential layer completion time.  This is
        # intentionally simple and deterministic; it captures the CPU/GPU
        # balancing objective without pretending to be a scheduler.
        for _ in range(args.capacity):
            best = None
            best_gain = -float("inf")
            for key in keys:
                if key in selected:
                    continue
                l = key[0]
                spec = layer[l]
                before = layer_cost(l, hit[l])
                old_default = spec["fallback_default"]
                # Account for this expert's measured fallback cost as a
                # first-order correction to the layer-average CPU model.
                effective = fallback_costs[key]
                after_hits = hit[l] + route.get(key, 0)
                after = layer_cost(l, after_hits)
                after += route.get(key, 0) * (old_default - effective)
                gain = before - after
                if best is None or gain > best_gain or (gain == best_gain and key < best):
                    best, best_gain = key, gain
            selected.add(best)
            hit[best[0]] += route.get(best, 0)
        return selected

    policies = {"current": current, "makespan_balanced": greedy_makespan()}
    reports = {}
    measured_total = total_cost(current)
    for name, selected in policies.items():
        hit = counts(selected)
        per_layer = {}
        for number_layer in range(args.layers):
            spec = layer[number_layer]
            routes = spec["routes"]
            per_layer[str(number_layer)] = {
                "slots": sum(1 for key in selected if key[0] == number_layer),
                "routes": routes,
                "gpu_hits": hit[number_layer],
                "gpu_hit_rate": hit[number_layer] / routes if routes else 0.0,
                "cpu_fallbacks": max(0, routes - hit[number_layer]),
                "predicted_completion_ms_per_token": layer_cost(number_layer, hit[number_layer]),
            }
        total = sum(item["predicted_completion_ms_per_token"]
                    for item in per_layer.values())
        reports[name] = {
            "slots_total": len(selected),
            "slots_per_layer": {str(l): sum(1 for key in selected if key[0] == l)
                                 for l in range(args.layers)},
            "predicted_gpu_hit_rate": sum(hit) / sum(spec["routes"] for spec in layer.values()),
            "predicted_cpu_fallbacks": sum(max(0, spec["routes"] - hit[l])
                                            for l, spec in layer.items()),
            "predicted_completion_ms_per_token": total,
            "predicted_improvement_ms_per_token": measured_total - total,
            "layers": per_layer,
        }

    result = {
        "inputs": {"expert_csv": args.expert_csv, "layer_csv": args.layer_csv,
                   "placement_csv": args.placement_csv, "capacity": args.capacity,
                   "layers": args.layers, "experts": args.experts},
        "formula": "cpu + gpu - overlap_fraction*min(cpu,gpu) + boundary_overhead",
        "definitions": {
            "cpu": "shared_ms_per_token + nonresident_routes * fallback_cost",
            "gpu": "resident_routes * measured_gpu_event_cost_per_route",
            "overlap_fraction": "measured gpu_cpu_overlap / min(cpu_window, gpu_event)",
            "boundary_overhead": "measured issue_to_complete - (cpu_window + gpu_event - overlap)",
        },
        "policies": reports,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.heat_out:
        write_heat(args.heat_out, policies["makespan_balanced"],
                   args.layers, args.experts)
        print(f"wrote {args.heat_out}")
    for name, report in reports.items():
        print(f"{name:18s} slots={report['slots_total']:4d} "
              f"hit={100*report['predicted_gpu_hit_rate']:6.2f}% "
              f"fallbacks={report['predicted_cpu_fallbacks']:6d} "
              f"completion_delta={report['predicted_improvement_ms_per_token']:+8.3f} ms/token")
    for number_layer in (38, 39):
        for name, report in reports.items():
            item = report["layers"][str(number_layer)]
            print(f"layer {number_layer} {name:18s} slots={item['slots']:3d} "
                  f"hit={100*item['gpu_hit_rate']:6.2f}% "
                  f"completion={item['predicted_completion_ms_per_token']:7.3f} ms/token")


if __name__ == "__main__":
    main()
