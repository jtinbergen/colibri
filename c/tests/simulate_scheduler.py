#!/usr/bin/env python3
"""Offline compute-island scheduler simulator.

This is a model, not a runtime scheduler.  It keeps expert ownership and
resident capacity explicit, calibrates CPU/GPU service lanes from an observed
run, and evaluates alternative policies against the same routed work.
"""

import argparse
import csv
import heapq
import json
import math
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path


def value(row, key, default=0.0):
    try:
        result = float(row.get(key, default) or default)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def integer(row, key, default=0):
    return int(value(row, key, default))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mean(values, fallback=0.0):
    return statistics.fmean(values) if values else fallback


def keys_for(layers, experts):
    return [(layer, expert) for layer in range(layers)
            for expert in range(experts)]


def read_placement(path):
    return {(integer(row, "layer"), integer(row, "expert"))
            for row in read_csv(path) if integer(row, "gpu_resident")}


def write_heat(path, selected, layers, experts):
    values = [2 if (layer, expert) in selected else 0
              for layer in range(layers) for expert in range(experts)]
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<III", 0x51544831, layers, experts))
        handle.write(struct.pack("<" + "I" * len(values), *values))


def load_timing(path, layers):
    """Load the lightweight layer boundary trace, grouped by layer."""
    if not path:
        return ({layer: {key: 0.0 for key in ("pre", "dispatch", "merge", "post")}
                 for layer in range(layers)},
                {layer: {"cpu": 0.0, "gpu": 0.0} for layer in range(layers)},
                {"used": False})
    rows = read_csv(path)
    grouped = defaultdict(list)
    for row in rows:
        grouped[integer(row, "layer")].append(row)
    envelope, lanes = {}, {}
    for layer in range(layers):
        group = grouped.get(layer, [])
        if not group:
            envelope[layer] = {key: 0.0 for key in ("pre", "dispatch", "merge", "post")}
            lanes[layer] = {"cpu": 0.0, "gpu": 0.0}
            continue
        def series(key):
            return [value(row, key) for row in group]
        pre = [max(0.0, value(row, "gpu_runnable_ms") - value(row, "layer_begin_ms"))
               for row in group]
        post = [max(0.0, value(row, "layer_complete_ms") - value(row, "merge_begin_ms"))
                for row in group if value(row, "merge_begin_ms") > 0]
        envelope[layer] = {
            "pre": mean(pre),
            "dispatch": mean(series("gpu_dispatch_delay_ms")),
            "merge": mean(series("exposed_merge_ms")),
            "post": mean(post),
        }
        lanes[layer] = {"cpu": mean(series("cpu_lane_ms")),
                        "gpu": mean(series("gpu_lane_ms"))}
    return envelope, lanes, {
        "used": True,
        "records": len(rows),
        "tokens": len({integer(row, "token") for row in rows}),
        "layers": len(grouped),
        "path": str(path),
        "trace_mean_layer_makespan_ms": mean(
            [value(row, "layer_makespan_ms") for row in rows]),
        "clock_note": "GPU completion is host-estimated after resident synchronization; not hardware-clock synchronized",
    }


def build_model(expert_rows, layer_rows, timing_envelope, timing_lanes,
                layers, experts, current=None):
    by_key = {(integer(row, "layer"), integer(row, "expert")): row
              for row in expert_rows}
    layer_by_id = {integer(row, "layer"): row for row in layer_rows}
    all_keys = keys_for(layers, experts)
    route = {key: integer(by_key.get(key, {}), "invocations") for key in all_keys}

    # CPU costs are expert-specific where the export measured fallback work.
    # Missing values receive the layer median, then the global median.
    raw_cpu = {}
    for key in all_keys:
        row = by_key.get(key, {})
        fallback = integer(row, "cpu_fallback")
        raw_cpu[key] = ((value(row, "cpu_get_ms") + value(row, "cpu_matmul_ms")) /
                        fallback if fallback else 0.0)
    positive = [cost for cost in raw_cpu.values() if cost > 0]
    global_cpu = statistics.median(positive) if positive else 1.0
    for layer in range(layers):
        costs = [raw_cpu[(layer, expert)] for expert in range(experts)
                 if raw_cpu[(layer, expert)] > 0]
        layer_default = statistics.median(costs) if costs else global_cpu
        for expert in range(experts):
            if raw_cpu[(layer, expert)] <= 0:
                raw_cpu[(layer, expert)] = layer_default

    model, calibration = {}, {}
    for layer in range(layers):
        row = layer_by_id.get(layer, {})
        timing = timing_lanes.get(layer, {})
        current_layer = {(layer, expert) for expert in range(experts)
                         if current is not None and (layer, expert) in current}
        routes = sum(route[(layer, expert)] for expert in range(experts))
        current_hits = sum(route[key] for key in current_layer)
        measured_cpu = timing.get("cpu", 0.0)
        measured_gpu = timing.get("gpu", 0.0)
        if measured_cpu <= 0:
            measured_cpu = value(row, "cpu_fallback_ms") / max(1, integer(row, "calls"))
        if measured_gpu <= 0:
            measured_gpu = value(row, "qt_take_ms") / max(1, integer(row, "qt_take_calls"))
        raw_cpu_ms = sum(route[key] * raw_cpu[key]
                         for key in all_keys if key[0] == layer and key not in current_layer)
        raw_gpu_units = float(current_hits)
        cpu_scale = measured_cpu / raw_cpu_ms if raw_cpu_ms > 0 else 0.0
        gpu_scale = measured_gpu / raw_gpu_units if raw_gpu_units > 0 else 0.0
        envelope = timing_envelope.get(layer, {})
        model[layer] = {
            "routes": routes,
            "current_hits": current_hits,
            "cpu_cost_ms_per_route": {
                str(expert): raw_cpu[(layer, expert)] * cpu_scale
                for expert in range(experts)},
            "gpu_cost_ms_per_route": gpu_scale,
            "measured_cpu_lane_ms": measured_cpu,
            "measured_gpu_lane_ms": measured_gpu,
            "envelope": envelope,
            "source": "lightweight timing trace" if timing.get("cpu", 0.0) > 0
                       else "layer export fallback",
        }
        calibration[layer] = {
            "routes": routes,
            "current_gpu_hits": current_hits,
            "current_cpu_routes": max(0, routes - current_hits),
            "raw_cpu_ms": raw_cpu_ms,
            "raw_gpu_units": raw_gpu_units,
            "measured_cpu_ms": measured_cpu,
            "measured_gpu_ms": measured_gpu,
            "cpu_anchor_error_ms": raw_cpu_ms * cpu_scale - measured_cpu,
            "gpu_anchor_error_ms": raw_gpu_units * gpu_scale - measured_gpu,
        }
    return all_keys, model, route, by_key, calibration


def top_by_score(all_keys, score, capacity):
    return set(sorted(all_keys, key=lambda key: (-score(key), key))[:capacity])


def placement_policies(all_keys, route, model, current, capacity, layers, experts):
    policies = {"current": set(current)}
    policies["global_hot"] = top_by_score(all_keys, lambda key: route[key], capacity)
    base, extra = divmod(capacity, layers)
    balanced = set()
    for layer in range(layers):
        quota = base + (1 if layer < extra else 0)
        candidates = [(layer, expert) for expert in range(experts)]
        balanced.update(sorted(candidates, key=lambda key: (-route[key], key))[:quota])
    policies["layer_balanced"] = balanced

    def critical_score(key):
        spec = model[key[0]]
        lane = spec["measured_cpu_lane_ms"]
        gpu = spec["measured_gpu_lane_ms"]
        criticality = 1.0 + max(0.0, lane - gpu) / max(lane, gpu, 1e-9)
        return route[key] * spec["cpu_cost_ms_per_route"][str(key[1])] * criticality
    policies["critical_path"] = top_by_score(all_keys, critical_score, capacity)

    def objective(layer, selected):
        spec = model[layer]
        gpu = sum(route[key] * spec["gpu_cost_ms_per_route"]
                   for key in selected if key[0] == layer)
        cpu = sum(route[key] * spec["cpu_cost_ms_per_route"][str(key[1])]
                  for key in all_keys if key[0] == layer and key not in selected)
        env = spec["envelope"]
        return env["pre"] + env["dispatch"] + max(cpu, gpu) + env["merge"] + env["post"]

    prefixes = {}
    for layer in range(layers):
        candidates = [(layer, expert) for expert in range(experts)]
        candidates.sort(key=lambda key: (
            -(route[key] * model[layer]["cpu_cost_ms_per_route"][str(key[1])]), key))
        prefixes[layer] = candidates

    # Greedy marginal allocation over layer prefixes.  This is transparent,
    # deterministic, and fast, but intentionally not advertised as optimal.
    selected, counts, heap = set(), [0] * layers, []
    for layer in range(layers):
        candidate = prefixes[layer][0]
        heapq.heappush(heap, (-(objective(layer, set()) -
                               objective(layer, {candidate})), layer, 1))
    for _ in range(min(capacity, len(all_keys))):
        if not heap:
            break
        _, layer, expected = heapq.heappop(heap)
        if expected != counts[layer] + 1:
            continue
        candidate = prefixes[layer][counts[layer]]
        selected.add(candidate)
        counts[layer] += 1
        if counts[layer] < experts:
            next_candidate = prefixes[layer][counts[layer]]
            heapq.heappush(heap, (-(objective(layer, selected) -
                                   objective(layer, selected | {next_candidate})),
                                  layer, counts[layer] + 1))
    policies["makespan_balanced"] = selected
    return policies


def predict(policy, placement, all_keys, route, model, layers):
    layer_reports, classes, total = {}, Counter(), 0.0
    for layer in range(layers):
        spec = model[layer]
        gpu_keys = {key for key in placement if key[0] == layer}
        hits = sum(route[key] for key in gpu_keys)
        fallbacks = sum(route[key] for key in all_keys
                        if key[0] == layer and key not in gpu_keys)
        cpu = sum(route[key] * spec["cpu_cost_ms_per_route"][str(key[1])]
                  for key in all_keys if key[0] == layer and key not in gpu_keys)
        gpu = hits * spec["gpu_cost_ms_per_route"]
        env = spec["envelope"]
        complete = env["pre"] + env["dispatch"] + max(cpu, gpu) + env["merge"] + env["post"]
        critical = "balanced"
        if cpu > gpu * 1.10:
            critical = "CPU-critical"
        elif gpu > cpu * 1.10:
            critical = "GPU-critical"
        if env["merge"] > 0.10 * max(complete, 1e-9):
            critical = "merge/synchronization-critical"
        classes[critical] += 1
        latest = max(cpu, gpu)
        layer_reports[str(layer)] = {
            "slots": len(gpu_keys),
            "routes": spec["routes"],
            "gpu_hits": hits,
            "gpu_hit_rate": hits / spec["routes"] if spec["routes"] else 0.0,
            "cpu_fallbacks": fallbacks,
            "cpu_lane_ms": cpu,
            "gpu_lane_ms": gpu,
            "imbalance_ms": abs(cpu - gpu),
            "gpu_slack_ms": max(0.0, complete - (env["pre"] + env["dispatch"] + gpu + env["post"])),
            "cpu_slack_ms": max(0.0, complete - (env["pre"] + env["dispatch"] + cpu + env["post"])),
            "predicted_layer_makespan_ms": complete,
            "critical": critical,
            "envelope_ms": env,
            "marginal_cpu_service_ms": sum(route[key] * spec["cpu_cost_ms_per_route"][str(key[1])]
                                            for key in gpu_keys),
            "latest_island_completion_ms": env["pre"] + env["dispatch"] + latest,
        }
        total += complete
    routes = sum(spec["routes"] for spec in model.values())
    hits = sum(item["gpu_hits"] for item in layer_reports.values())
    return {
        "policy": policy,
        "slots_total": len(placement),
        "slots_per_layer": {str(layer): layer_reports[str(layer)]["slots"]
                             for layer in range(layers)},
        "predicted_gpu_hit_rate": hits / routes if routes else 0.0,
        "predicted_gpu_hits": hits,
        "predicted_cpu_fallbacks": sum(item["cpu_fallbacks"] for item in layer_reports.values()),
        "predicted_completion_ms_per_token": total,
        "classification_counts": dict(classes),
        "islands": {"CPU0": "nonresident routed expert work",
                    "GPU0": "resident routed expert work"},
        "layers": layer_reports,
    }


def layer_prediction(layer, placement, all_keys, route, model):
    """Return the same layer model used by predict(), for marginal analysis."""
    spec = model[layer]
    gpu_keys = {key for key in placement if key[0] == layer}
    hits = sum(route[key] for key in gpu_keys)
    fallbacks = sum(route[key] for key in all_keys
                    if key[0] == layer and key not in gpu_keys)
    cpu = sum(route[key] * spec["cpu_cost_ms_per_route"][str(key[1])]
              for key in all_keys if key[0] == layer and key not in gpu_keys)
    gpu = hits * spec["gpu_cost_ms_per_route"]
    env = spec["envelope"]
    complete = env["pre"] + env["dispatch"] + max(cpu, gpu) + env["merge"] + env["post"]
    return {"complete": complete, "cpu": cpu, "gpu": gpu,
            "hits": hits, "fallbacks": fallbacks}


def marginal_gpu_moves(placement, all_keys, route, model, layers, limit=10):
    """Rank currently nonresident experts by predicted makespan reduction."""
    by_layer = defaultdict(list)
    for key in all_keys:
        if key in placement or route[key] <= 0:
            continue
        before = layer_prediction(key[0], placement, all_keys, route, model)
        after = layer_prediction(key[0], placement | {key}, all_keys, route, model)
        by_layer[key[0]].append({
            "layer": key[0], "expert": key[1],
            "invocations": route[key],
            "predicted_layer_gain_ms": before["complete"] - after["complete"],
            "cpu_ms_removed": route[key] * model[key[0]]["cpu_cost_ms_per_route"][str(key[1])],
            "gpu_ms_added": route[key] * model[key[0]]["gpu_cost_ms_per_route"],
            "before_ms": before["complete"], "after_ms": after["complete"],
        })
    for layer in by_layer:
        by_layer[layer].sort(key=lambda item: (-item["predicted_layer_gain_ms"],
                                               item["expert"]))
    all_moves = sorted((item for items in by_layer.values() for item in items),
                       key=lambda item: (-item["predicted_layer_gain_ms"],
                                          item["layer"], item["expert"]))
    return {
        "top_global": all_moves[:limit],
        "top_by_layer": {str(layer): items[:limit]
                          for layer, items in sorted(by_layer.items())},
        "positive_gain_count": sum(item["predicted_layer_gain_ms"] > 0
                                    for item in all_moves),
        "candidate_count": len(all_moves),
    }


def markdown(result):
    policies = result["policies"]
    current = policies["current"]["predicted_completion_ms_per_token"]
    lines = ["# Compute-island scheduler simulation", "",
             "Offline first-order model; it does not change runtime placement or scheduling.", "",
             "## Inputs and model", "",
             f"- Expert rows: {result['inputs']['expert_rows']}; routed invocations: {result['inputs']['observed_routes']}",
             f"- Capacity: **{result['inputs']['capacity']}** resident expert slots",
             f"- Timing calibration: `{result['inputs']['timing_calibration'].get('path', 'none')}`",
             f"- Measured trace mean layer makespan: **{result['inputs']['timing_calibration'].get('trace_mean_layer_makespan_ms', 0.0):.3f} ms/layer-record**",
             "- Layer cost: `measured pre + dispatch + max(CPU0, GPU0) + measured merge + post`.",
             "- Current kernels, quantization, and reduction order are not changed.", "",
             "## Policy predictions", "",
             "| Policy | Slots | GPU hit | CPU fallbacks | Predicted ms/token | vs current |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, report in policies.items():
        delta = current - report["predicted_completion_ms_per_token"]
        lines.append(f"| {name} | {report['slots_total']} | {100*report['predicted_gpu_hit_rate']:.2f}% | {report['predicted_cpu_fallbacks']} | {report['predicted_completion_ms_per_token']:.3f} | {delta:+.3f} |")
    lines += ["", "## Makespan-balanced policy by layer", "",
              "| Layer | Slots | GPU hit | CPU ms | GPU ms | Imbalance | Makespan | GPU slack | Class |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    balanced = policies["makespan_balanced"]
    for layer in range(result["inputs"]["layers"]):
        item = balanced["layers"][str(layer)]
        lines.append(f"| {layer} | {item['slots']} | {100*item['gpu_hit_rate']:.2f}% | {item['cpu_lane_ms']:.3f} | {item['gpu_lane_ms']:.3f} | {item['imbalance_ms']:.3f} | {item['predicted_layer_makespan_ms']:.3f} | {item['gpu_slack_ms']:.3f} | {item['critical']} |")
    lines += ["", "## Problem layers", ""]
    for layer in (38, 39):
        if str(layer) in balanced["layers"]:
            item = balanced["layers"][str(layer)]
            lines.append(f"- Layer {layer}: {item['slots']} slots, {100*item['gpu_hit_rate']:.2f}% GPU hit, {item['predicted_layer_makespan_ms']:.3f} ms, {item['critical']}.")
    lines += ["", "## Top marginal GPU moves from current placement", ""]
    for item in result["marginal_gpu_moves_from_current"]["top_global"][:10]:
        lines.append(f"- ({item['layer']},{item['expert']}): {item['predicted_layer_gain_ms']:+.3f} ms/layer; {item['invocations']} invocations; CPU removed {item['cpu_ms_removed']:.3f} ms, GPU added {item['gpu_ms_added']:.3f} ms.")
    lines += ["", "The policy ranking is predictive only. Compare deltas with matched runtime A/B runs before changing placement or scheduling.", "",
              "The generic CPU0/GPU0 labels are intentional: future NUMA and GPU islands can use the same schema once measured cost profiles exist."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-csv", required=True)
    parser.add_argument("--layer-csv", required=True)
    parser.add_argument("--placement-csv", required=True)
    parser.add_argument("--timing-csv", default="")
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=3642)
    parser.add_argument("--heat-out", default="")
    parser.add_argument("--current-heat-out", default="",
                        help="write the current placement snapshot as QTH1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--md", default="")
    args = parser.parse_args()
    expert_rows = read_csv(args.expert_csv)
    layer_rows = read_csv(args.layer_csv)
    current = read_placement(args.placement_csv)
    envelope, lanes, timing_meta = load_timing(args.timing_csv, args.layers)
    all_keys, model, route, by_key, calibration = build_model(
        expert_rows, layer_rows, envelope, lanes, args.layers, args.experts,
        current)
    policies = placement_policies(all_keys, route, model, current,
                                  args.capacity, args.layers, args.experts)
    reports = {name: predict(name, placement, all_keys, route, model, args.layers)
               for name, placement in policies.items()}
    marginal = marginal_gpu_moves(current, all_keys, route, model, args.layers)
    result = {
        "schema": "colibri.compute_islands.scheduler_simulation.v1",
        "inputs": {
            "expert_csv": str(Path(args.expert_csv)),
            "layer_csv": str(Path(args.layer_csv)),
            "placement_csv": str(Path(args.placement_csv)),
            "timing_calibration": timing_meta,
            "layers": args.layers, "experts": args.experts, "capacity": args.capacity,
            "expert_rows": len(expert_rows),
            "observed_routes": sum(route.values()),
            "observed_gpu_hits": sum(integer(row, "gpu_hits") for row in expert_rows),
            "observed_cpu_fallbacks": sum(integer(row, "cpu_fallback") for row in expert_rows),
            "current_slots": len(current),
        },
        "islands": {
            "CPU0": {"role": "nonresident routed expert work", "profile": "measured CPU lane; expert-scaled"},
            "GPU0": {"role": "resident routed expert work", "profile": "measured GPU lane; per-route first-order scale"},
        },
        "objective": "sum sequential layers of pre + dispatch + max(CPU0, GPU0) + merge + post",
        "assumptions": [
            "Only expert ownership changes between policies.",
            "Measured pre-assignment, dispatch, merge, and post-merge envelopes remain fixed.",
            "CPU expert cost is scaled from measured fallback timing and anchored to the observed CPU lane.",
            "GPU cost is a first-order per-resident-route scale anchored to the observed GPU lane.",
            "All routed invocations in an owned expert go to that island; partial token splitting is not modeled.",
            "Storage congestion, NUMA affinity, and multiple GPU profiles are not modeled yet.",
        ],
        "formula": {
            "layer": "pre + dispatch + max(cpu_service(nonresident), gpu_service(resident)) + merge + post",
            "cpu_service": "sum(invocations[key] * calibrated_cpu_ms_per_route[key])",
            "gpu_service": "resident_invocations * calibrated_gpu_ms_per_route[layer]",
            "slack": "layer completion - island completion; positive slack means non-critical island",
        },
        "calibration": {str(layer): data for layer, data in calibration.items()},
        "marginal_gpu_moves_from_current": marginal,
        "policies": reports,
        "validation": {
            "current_anchor_ms_per_token": reports["current"]["predicted_completion_ms_per_token"],
            "timing_trace_mean_ms_per_layer": timing_meta.get("trace_mean_layer_makespan_ms", 0.0),
            "model_current_ms_per_layer": reports["current"]["predicted_completion_ms_per_token"] / max(1, args.layers),
            "model_vs_trace_relative_error": (
                (reports["current"]["predicted_completion_ms_per_token"] / max(1, args.layers) - timing_meta.get("trace_mean_layer_makespan_ms", 0.0)) /
                timing_meta.get("trace_mean_layer_makespan_ms", 1.0)
                if timing_meta.get("trace_mean_layer_makespan_ms", 0.0) else None),
            "note": "Compare policy deltas, not absolute totals, when expert and timing samples differ.",
        },
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(result), encoding="utf-8")
    if args.heat_out:
        write_heat(args.heat_out, policies["makespan_balanced"], args.layers, args.experts)
    if args.current_heat_out:
        write_heat(args.current_heat_out, current, args.layers, args.experts)
    current_value = reports["current"]["predicted_completion_ms_per_token"]
    for name, report in reports.items():
        delta = current_value - report["predicted_completion_ms_per_token"]
        print(f"{name:18s} slots={report['slots_total']:4d} hit={100*report['predicted_gpu_hit_rate']:6.2f}% "
              f"fallbacks={report['predicted_cpu_fallbacks']:6d} predicted={report['predicted_completion_ms_per_token']:9.3f} ms/token delta={delta:+9.3f}")
    for layer in (38, 39):
        if str(layer) in reports["makespan_balanced"]["layers"]:
            item = reports["makespan_balanced"]["layers"][str(layer)]
            print(f"layer {layer}: slots={item['slots']} hit={100*item['gpu_hit_rate']:.2f}% cpu={item['cpu_lane_ms']:.3f} gpu={item['gpu_lane_ms']:.3f} makespan={item['predicted_layer_makespan_ms']:.3f} {item['critical']}")


if __name__ == "__main__":
    main()
