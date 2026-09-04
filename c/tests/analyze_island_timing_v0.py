#!/usr/bin/env python3
"""Summarize the lightweight Compute Islands v0 per-layer trace."""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def f(row, key):
    try:
        v = float(row.get(key, 0.0) or 0.0)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def p50(values):
    return statistics.median(values) if values else 0.0


def p95(values):
    if not values:
        return 0.0
    values = sorted(values)
    return values[min(len(values) - 1, int(0.95 * (len(values) - 1)))]


def classify(row):
    cpu, gpu = f(row, "cpu_lane_ms"), f(row, "gpu_lane_ms")
    merge = f(row, "exposed_merge_ms")
    dominant = max(cpu, gpu, 1e-9)
    if merge > 0.5 and merge > 0.10 * max(f(row, "layer_makespan_ms"), 1e-9):
        return "merge/synchronization-critical"
    if cpu <= 0.01 and gpu > 0.01:
        return "GPU-critical"
    if gpu <= 0.01 and cpu > 0.01:
        return "CPU-critical"
    if abs(cpu - gpu) <= 0.10 * dominant:
        return "balanced"
    return "CPU-critical" if cpu > gpu else "GPU-critical"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    with Path(args.input).open(newline="", encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    if not rows:
        raise SystemExit("empty trace")

    by_layer = defaultdict(list)
    for row in rows:
        by_layer[int(float(row["layer"]))].append(row)

    layer_reports = []
    for layer, group in sorted(by_layer.items()):
        def vals(key): return [f(row, key) for row in group]
        series = {key: vals(key) for key in (
            "gpu_dispatch_delay_ms", "gpu_lane_ms", "cpu_lane_ms",
            "imbalance_ms", "exposed_merge_ms", "layer_makespan_ms",
            "gpu_slack_ms")}
        pre_assignment = [max(0.0, f(row, "gpu_runnable_ms") -
                               f(row, "layer_begin_ms"))
                          for row in group]
        post_merge = [max(0.0, f(row, "layer_complete_ms") -
                           f(row, "merge_begin_ms"))
                      for row in group if f(row, "merge_begin_ms") > 0.0]
        series["pre_assignment_ms"] = pre_assignment
        series["post_merge_ms"] = post_merge if post_merge else [0.0]
        means = {key: statistics.fmean(value) for key, value in series.items()}
        classes = Counter(classify(row) for row in group)
        class_name = classes.most_common(1)[0][0]
        layer_reports.append({
            "layer": layer,
            "samples": len(group),
            "token_first": int(float(group[0]["token"])),
            "token_last": int(float(group[-1]["token"])),
            "gpu_samples": sum(f(row, "gpu_lane_ms") > 0.01 for row in group),
            "cpu_samples": sum(f(row, "cpu_lane_ms") > 0.01 for row in group),
            "mean": means,
            "median": {key: p50(value) for key, value in series.items()},
            "p95": {key: p95(value) for key, value in series.items()},
            "classification_counts": dict(classes),
            "classification": class_name,
        })

    def allvals(key): return [f(row, key) for row in rows]
    class_counts = Counter(classify(row) for row in rows)
    pre_assignment_all = [max(0.0, f(row, "gpu_runnable_ms") -
                               f(row, "layer_begin_ms")) for row in rows]
    post_merge_all = [max(0.0, f(row, "layer_complete_ms") -
                           f(row, "merge_begin_ms")) for row in rows
                      if f(row, "merge_begin_ms") > 0.0]
    aggregate = {
        "records": len(rows),
        "layers": len(layer_reports),
        "tokens": len(set(row["token"] for row in rows)),
        "mean": {key: statistics.fmean(allvals(key)) for key in (
            "gpu_dispatch_delay_ms", "gpu_lane_ms", "cpu_lane_ms",
            "imbalance_ms", "exposed_merge_ms", "layer_makespan_ms",
            "gpu_slack_ms")},
        "p50": {key: p50(allvals(key)) for key in (
            "gpu_dispatch_delay_ms", "gpu_lane_ms", "cpu_lane_ms",
            "imbalance_ms", "exposed_merge_ms", "layer_makespan_ms",
            "gpu_slack_ms")},
        "p95": {key: p95(allvals(key)) for key in (
            "gpu_dispatch_delay_ms", "gpu_lane_ms", "cpu_lane_ms",
            "imbalance_ms", "exposed_merge_ms", "layer_makespan_ms",
            "gpu_slack_ms")},
        "classification_counts": dict(class_counts),
    }
    aggregate["mean"]["pre_assignment_ms"] = statistics.fmean(pre_assignment_all)
    aggregate["mean"]["post_merge_ms"] = statistics.fmean(post_merge_all) if post_merge_all else 0.0
    aggregate["p50"]["pre_assignment_ms"] = p50(pre_assignment_all)
    aggregate["p50"]["post_merge_ms"] = p50(post_merge_all)
    aggregate["p95"]["pre_assignment_ms"] = p95(pre_assignment_all)
    aggregate["p95"]["post_merge_ms"] = p95(post_merge_all)
    report = {
        "schema": "colibri.compute_islands_v0.timing.v1",
        "inputs": {"trace_csv": args.input, "cross_clock": "single host clock plus host-estimated GPU completion"},
        "definitions": {
            "gpu_dispatch_delay_ms": "gpu_submit - gpu_runnable",
            "gpu_lane_ms": "gpu_complete - gpu_submit",
            "cpu_lane_ms": "cpu_complete - cpu_begin",
            "imbalance_ms": "absolute island completion difference when both lanes are active",
            "exposed_merge_ms": "layer_complete - latest active island completion",
            "layer_makespan_ms": "layer_complete - layer_begin",
            "gpu_slack_ms": "layer_complete - gpu_complete",
            "classification_note": "thresholded diagnostic classification; not a scheduler decision",
        },
        "aggregate": aggregate,
        "layers": layer_reports,
        "gates": {
            "A_boundary_understood": "PASS",
            "B_explicit_executor": "PASS",
            "C_overlap": "PASS-observed",
            "D_makespan_model": "PASS-provisional",
            "E_performance": "OPEN",
            "F_scheduler_readiness": "HOLD",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.md:
        ranked = sorted(layer_reports, key=lambda x: x["mean"]["layer_makespan_ms"], reverse=True)
        lines = ["# Compute Islands v0 timing", "", f"Input: `{args.input}`; {len(rows)} records.", "",
                 "## Aggregate", "",
                 f"- Mean makespan: **{aggregate['mean']['layer_makespan_ms']:.3f} ms/layer**",
                 f"- Median makespan: **{aggregate['p50']['layer_makespan_ms']:.3f} ms/layer**",
                 f"- p95 makespan: **{aggregate['p95']['layer_makespan_ms']:.3f} ms/layer**",
                 f"- Mean dispatch delay: **{aggregate['mean']['gpu_dispatch_delay_ms']:.3f} ms**",
                 f"- Mean pre-assignment work: **{aggregate['mean']['pre_assignment_ms']:.3f} ms**",
                 f"- Mean GPU lane: **{aggregate['mean']['gpu_lane_ms']:.3f} ms**",
                 f"- Mean CPU lane: **{aggregate['mean']['cpu_lane_ms']:.3f} ms**",
                 f"- Mean exposed merge: **{aggregate['mean']['exposed_merge_ms']:.3f} ms**",
                 f"- Mean post-merge tail: **{aggregate['mean']['post_merge_ms']:.3f} ms**",
                 f"- Row classifications: `{json.dumps(aggregate['classification_counts'], sort_keys=True)}`", "",
                 "## Per-layer distribution (mean over tokens)", "",
                 "| Layer | Pre ms | CPU ms | GPU ms | Dispatch ms | Imbalance ms | Merge ms | Makespan ms | GPU slack ms | Class |",
                 "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for x in sorted(layer_reports, key=lambda x: x["layer"]):
            m = x["mean"]
            lines.append(f"| {x['layer']} | {m['pre_assignment_ms']:.3f} | {m['cpu_lane_ms']:.3f} | {m['gpu_lane_ms']:.3f} | {m['gpu_dispatch_delay_ms']:.3f} | {m['imbalance_ms']:.3f} | {m['exposed_merge_ms']:.3f} | {m['layer_makespan_ms']:.3f} | {m['gpu_slack_ms']:.3f} | {x['classification']} |")
        lines += ["", "## Largest makespans", ""]
        for x in ranked[:5]:
            lines.append(f"- Layer {x['layer']}: {x['mean']['layer_makespan_ms']:.3f} ms, {x['classification']}")
        lines += ["", "The GPU completion timestamp is host-clocked immediately after the resident synchronization; this is suitable for the lightweight v0 comparison, but not equivalent to a hardware-synchronized CUPTI timeline."]
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
