#!/usr/bin/env python3
"""Offline resident-expert placement simulator.

This is deliberately a prediction tool, not a runtime scheduler.  It uses
one decode export and keeps the resident capacity fixed while changing only
the set of (layer, expert) blocks assigned to VRAM.
"""

import argparse
import csv
import json
import math
import statistics
import struct
from pathlib import Path


def number(row, name, default=0.0):
    try:
        return float(row.get(name, default) or default)
    except (TypeError, ValueError):
        return default


def integer(row, name, default=0):
    return int(number(row, name, default))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def median_or(values, fallback):
    values = [v for v in values if math.isfinite(v) and v > 0]
    return statistics.median(values) if values else fallback


def make_candidates(layers, experts, expert_rows, layer_rows):
    by_key = {(int(r["layer"]), int(r["expert"])): r for r in expert_rows}
    layer_by_id = {int(r["layer"]): r for r in layer_rows}
    all_keys = [(layer, expert) for layer in range(layers)
                for expert in range(experts)]

    layer_fallback = {}
    layer_qtake = {}
    layer_moe = {}
    for layer in range(layers):
        row = layer_by_id.get(layer, {})
        calls = max(1, integer(row, "calls"))
        qtake_calls = max(1, integer(row, "qt_take_calls", calls))
        layer_fallback[layer] = number(row, "cpu_fallback_ms") / calls
        layer_qtake[layer] = number(row, "qt_take_ms") / qtake_calls
        layer_moe[layer] = number(row, "moe_ms") / calls

    global_fallback = median_or(list(layer_fallback.values()), 1.0)
    fallback_by_key = {}
    invocations = {}
    observed_gpu = {}
    for key in all_keys:
        row = by_key.get(key, {})
        invocations[key] = integer(row, "invocations")
        observed_gpu[key] = integer(row, "gpu_hits")
        fallback_count = integer(row, "cpu_fallback")
        measured_cost = ((number(row, "cpu_get_ms") +
                          number(row, "cpu_matmul_ms")) / fallback_count
                         if fallback_count > 0 else 0.0)
        fallback_by_key[key] = (measured_cost if measured_cost > 0
                                else layer_fallback[key[0]] or global_fallback)

    median_qtake = median_or(list(layer_qtake.values()), 1.0)
    penalty = {}
    for layer in range(layers):
        penalty[layer] = max(0.0, layer_qtake[layer] - median_qtake) / median_qtake

    return (all_keys, by_key, layer_by_id, invocations, observed_gpu,
            fallback_by_key, layer_fallback, layer_qtake, layer_moe, penalty,
            global_fallback, median_qtake)


def placement_from_file(path, all_keys):
    if not path or not Path(path).exists():
        return set()
    rows = read_csv(path)
    return {(int(r["layer"]), int(r["expert"]))
            for r in rows if integer(r, "gpu_resident")}


def write_heat_file(path, placement, layers, experts):
    """Encode a fixed placement through the tier's existing QTH1 heat ABI."""
    values = [2 if (layer, expert) in placement else 0
              for layer in range(layers) for expert in range(experts)]
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<III", 0x51544831, layers, experts))
        handle.write(struct.pack("<" + "I" * len(values), *values))


def top_keys(keys, score, capacity):
    return set(sorted(keys, key=lambda key: (-score(key), key[0], key[1]))[:capacity])


def make_policy(name, all_keys, capacity, layers, experts, invocations,
                current):
    if name == "current":
        return set(current)
    if name == "global_hot":
        return top_keys(all_keys, lambda key: invocations[key], capacity)
    if name == "layer_balanced":
        base, extra = divmod(capacity, layers)
        selected = set()
        for layer in range(layers):
            quota = base + (1 if layer < extra else 0)
            layer_keys = [key for key in all_keys if key[0] == layer]
            selected.update(top_keys(layer_keys, lambda key: invocations[key], quota))
        return selected
    raise ValueError(f"unknown policy: {name}")


def summarize_policy(name, placement, all_keys, layer_rows, by_key,
                     invocations, fallback_by_key, layer_fallback,
                     layer_qtake, layer_moe, penalty, current, experts):
    layer_by_id = {int(r["layer"]): r for r in layer_rows}
    routes_total = sum(invocations.values())
    actual_routes = sum(integer(r, "invocations") for r in by_key.values())
    # The expert export contains only routed experts; use its total for the
    # observed request and all candidate keys for deterministic prediction.
    routes_total = actual_routes
    predicted_hits = {layer: 0 for layer in range(len(layer_by_id))}
    predicted_fallbacks = {layer: 0 for layer in range(len(layer_by_id))}
    predicted_cpu = {layer: 0.0 for layer in range(len(layer_by_id))}
    for key in all_keys:
        count = invocations[key]
        if key in placement:
            predicted_hits[key[0]] += count
        else:
            predicted_fallbacks[key[0]] += count
            predicted_cpu[key[0]] += count * fallback_by_key[key]

    slots = {layer: sum(1 for key in placement if key[0] == layer)
             for layer in range(len(layer_by_id))}
    current_fallbacks = sum(invocations[key] for key in all_keys if key not in current)
    current_actual_fallbacks = sum(integer(r, "cpu_fallback") for r in by_key.values())
    layer_result = {}
    predicted_moe = 0.0
    measured_moe = 0.0
    for layer in range(len(layer_by_id)):
        row = layer_by_id.get(layer, {})
        calls = max(1, integer(row, "calls"))
        current_hits = sum(invocations[(layer, e)] for e in range(experts)
                           if (layer, e) in current)
        new_hits = predicted_hits[layer]
        # Heuristic makespan model.  The non-MoE part is held constant.  GPU
        # qt_take scales with resident-route count; CPU fallback scales with
        # measured per-expert fallback cost.  max() models overlap of the two
        # paths.  Layers are sequential, so their predicted costs are summed.
        measured_qtake = layer_qtake[layer]
        gpu_cost = (measured_qtake * new_hits / current_hits
                     if current_hits > 0 else 0.0)
        cpu_cost = predicted_cpu[layer] / calls
        base = max(0.0, layer_moe[layer] - measured_qtake -
                   (number(row, "cpu_fallback_ms") / calls))
        predicted_layer = base + max(gpu_cost, cpu_cost)
        predicted_moe += predicted_layer
        measured_moe += layer_moe[layer]
        observed_routes = integer(row, "routes")
        layer_result[str(layer)] = {
            "slots": slots[layer],
            "routes": observed_routes,
            "predicted_gpu_hits": new_hits,
            "predicted_gpu_hit_rate": new_hits / observed_routes if observed_routes else 0.0,
            "predicted_cpu_fallbacks": predicted_fallbacks[layer],
            "cpu_fallbacks_avoided_vs_current_snapshot":
                sum(invocations[(layer, e)] for e in range(experts)
                    if (layer, e) not in current and (layer, e) in placement),
            "measured_moe_ms_per_token": layer_moe[layer],
            "measured_qt_take_ms_per_token": measured_qtake,
            "predicted_moe_ms_per_token": predicted_layer,
            "predicted_moe_delta_ms_per_token": predicted_layer - layer_moe[layer],
            "critical_path_penalty": penalty[layer],
        }

    policy_hits = sum(predicted_hits.values())
    policy_fallbacks = sum(predicted_fallbacks.values())
    return {
        "policy": name,
        "slots_total": len(placement),
        "slots_per_layer": slots,
        "global_gpu_hit_rate": policy_hits / routes_total if routes_total else 0.0,
        "predicted_gpu_hits": policy_hits,
        "predicted_cpu_fallbacks": policy_fallbacks,
        "cpu_fallbacks_avoided_vs_current_snapshot": current_fallbacks - policy_fallbacks,
        "cpu_fallbacks_avoided_vs_observed_current": current_actual_fallbacks - policy_fallbacks,
        "measured_moe_ms_per_token": measured_moe,
        "predicted_moe_ms_per_token": predicted_moe,
        "predicted_critical_path_improvement_ms_per_token": measured_moe - predicted_moe,
        "predicted_critical_path_improvement_pct":
            100.0 * (measured_moe - predicted_moe) / measured_moe if measured_moe else 0.0,
        "layers": layer_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--capacity", type=int, default=3642)
    parser.add_argument("--expert-csv", required=True)
    parser.add_argument("--layer-csv", required=True)
    parser.add_argument("--placement-csv", default="")
    parser.add_argument("--heat-dir", default="",
                        help="also write QTH1 heat files for each policy")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    expert_rows = read_csv(args.expert_csv)
    layer_rows = read_csv(args.layer_csv)
    data = make_candidates(args.layers, args.experts, expert_rows, layer_rows)
    (all_keys, by_key, _layer_by_id, invocations, observed_gpu,
     fallback_by_key, layer_fallback, layer_qtake, layer_moe, penalty,
     global_fallback, median_qtake) = data
    placement_path = args.placement_csv or args.expert_csv + ".placement.csv"
    current = placement_from_file(placement_path, all_keys)
    if len(current) != args.capacity:
        print(f"warning: current snapshot has {len(current)} slots, expected {args.capacity}")

    policies = {}
    placement_sets = {}
    for name in ("current", "global_hot", "layer_balanced"):
        placement = make_policy(name, all_keys, args.capacity, args.layers,
                                 args.experts, invocations, current)
        placement_sets[name] = placement
        policies[name] = summarize_policy(
            name, placement, all_keys, layer_rows, by_key, invocations,
            fallback_by_key, layer_fallback, layer_qtake, layer_moe, penalty,
            current, args.experts)

    critical_score = lambda key: invocations[key] * fallback_by_key[key] * (1.0 + penalty[key[0]])
    critical = top_keys(all_keys, critical_score, args.capacity)
    placement_sets["critical_path"] = critical
    policies["critical_path"] = summarize_policy(
        "critical_path", critical, all_keys, layer_rows, by_key, invocations,
        fallback_by_key, layer_fallback, layer_qtake, layer_moe, penalty,
        current, args.experts)

    result = {
        "inputs": {
            "expert_csv": str(Path(args.expert_csv)),
            "layer_csv": str(Path(args.layer_csv)),
            "placement_csv": str(Path(placement_path)),
            "layers": args.layers, "experts": args.experts,
            "capacity": args.capacity, "observed_expert_rows": len(expert_rows),
            "observed_routes": sum(integer(r, "invocations") for r in expert_rows),
            "observed_gpu_hits": sum(integer(r, "gpu_hits") for r in expert_rows),
            "observed_cpu_fallbacks": sum(integer(r, "cpu_fallback") for r in expert_rows),
        },
        "policy_formula": {
            "critical_path_score": "invocations * fallback_cost_ms_per_route * (1 + layer_penalty)",
            "fallback_cost_ms_per_route": "(cpu_get_ms + cpu_matmul_ms) / cpu_fallback when observed; otherwise layer median, then global median",
            "layer_penalty": "max(0, layer_qt_take_ms_per_token - median_layer_qt_take_ms) / median_layer_qt_take_ms",
            "makespan_heuristic": "base + max(predicted_gpu_qt_take, predicted_cpu_fallback); sequential layer costs are summed",
            "caveat": "This is an offline ranking heuristic, not a calibrated performance model; it assumes the measured routing distribution repeats and holds non-MoE cost constant.",
        },
        "fallback_cost_median_ms": global_fallback,
        "median_layer_qt_take_ms_per_token": median_qtake,
        "policies": policies,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.heat_dir:
        heat_dir = Path(args.heat_dir)
        heat_dir.mkdir(parents=True, exist_ok=True)
        for name, placement in placement_sets.items():
            write_heat_file(heat_dir / f"{name}.heat", placement,
                            args.layers, args.experts)
        print(f"wrote heat files in {heat_dir}")
    print(f"wrote {args.out}")
    for name, report in policies.items():
        print(f"{name:14s} slots={report['slots_total']:4d} "
              f"hit={100*report['global_gpu_hit_rate']:6.2f}% "
              f"fallbacks={report['predicted_cpu_fallbacks']:6d} "
              f"predicted_moe_delta={report['predicted_critical_path_improvement_ms_per_token']:+8.3f} ms/token "
              f"({report['predicted_critical_path_improvement_pct']:+6.2f}%)")
    for layer in (38, 39):
        print(f"layer {layer}:")
        for name, report in policies.items():
            item = report["layers"].get(str(layer))
            if item:
                print(f"  {name:12s} slots={item['slots']:3d} hit={100*item['predicted_gpu_hit_rate']:6.2f}% "
                      f"moe={item['predicted_moe_ms_per_token']:7.3f} ms/token")


if __name__ == "__main__":
    main()
