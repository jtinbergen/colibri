#!/usr/bin/env python3
"""Classify the existing Qwen layer trace as CPU/GPU/island-boundary critical.

This is deliberately an analysis tool, not a scheduler.  It consumes the
lightweight per-layer CSV emitted by the existing overlap instrumentation and
does not change execution.  ``cpu_window_ms`` and ``gpu_event_ms`` are the
observed local service windows; ``issue_to_complete_ms`` is the measured
resident issue-to-take envelope.  The clocks are not treated as a single
cross-clock timeline.
"""

import argparse
import csv
import json
import math
from pathlib import Path


def f(row, key):
    try:
        value = float(row.get(key, 0.0) or 0.0)
        return value if math.isfinite(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def i(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def classify(cpu, gpu, boundary):
    dominant = max(cpu, gpu, 1e-9)
    # A large unexplained envelope after both local windows is a boundary
    # signal.  Keep the threshold conservative because these are aggregate
    # host/event measurements rather than synchronized clocks.
    if boundary > 0.10 * dominant and boundary > 0.50:
        return "merge/synchronization-critical"
    if abs(cpu - gpu) <= 0.10 * dominant:
        return "balanced"
    return "CPU-critical" if cpu > gpu else "GPU-critical"


def load(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-layer overlap CSV")
    ap.add_argument("--out", required=True, help="machine-readable JSON")
    ap.add_argument("--md", default="", help="optional Markdown report")
    args = ap.parse_args()

    rows = load(args.input)
    if not rows:
        raise SystemExit("no layer rows")

    layers = []
    totals = {
        "routes": 0, "gpu_hits": 0, "cpu_fallback": 0,
        "cpu_window_ms": 0.0, "gpu_event_ms": 0.0,
        "issue_to_complete_ms": 0.0, "qt_issue_ms": 0.0,
        "qt_take_ms": 0.0, "shared_ms": 0.0,
        "gpu_cpu_overlap_ms": 0.0,
    }
    counts = {}
    for row in rows:
        cpu = f(row, "cpu_window_ms")
        gpu = f(row, "gpu_event_ms")
        complete = f(row, "issue_to_complete_ms")
        # The max() term models the parallel local work.  This is not a claim
        # that the two source clocks are synchronized; it is a conservative
        # residual decomposition of the measured envelope.
        boundary = max(0.0, complete - max(cpu, gpu))
        overlap = f(row, "gpu_cpu_overlap_ms")
        denom = min(cpu, gpu)
        item = {
            "layer": i(row, "layer"),
            "calls": i(row, "calls"),
            "routes": i(row, "routes"),
            "gpu_hits": i(row, "gpu_hits"),
            "cpu_fallback": i(row, "cpu_fallback"),
            "gpu_hit_rate": (i(row, "gpu_hits") / i(row, "routes")
                              if i(row, "routes") else 0.0),
            "cpu_window_ms": cpu,
            "gpu_event_ms": gpu,
            "issue_to_complete_ms": complete,
            "boundary_residual_ms": boundary,
            "gpu_cpu_overlap_ms": overlap,
            "overlap_fraction_of_shorter_window": overlap / denom if denom > 0 else 0.0,
            "imbalance_abs_ms": abs(cpu - gpu),
            "critical_path_class": classify(cpu, gpu, boundary),
            "cpu_fallback_cost_ms_per_route":
                f(row, "cpu_fallback_ms") / i(row, "cpu_fallback")
                if i(row, "cpu_fallback") else 0.0,
            "gpu_event_cost_ms_per_hit":
                gpu / i(row, "gpu_hits") if i(row, "gpu_hits") else 0.0,
            "qt_issue_ms": f(row, "qt_issue_ms"),
            "qt_take_ms": f(row, "qt_take_ms"),
            "shared_ms": f(row, "shared_ms"),
        }
        layers.append(item)
        counts[item["critical_path_class"]] = counts.get(item["critical_path_class"], 0) + 1
        for key in totals:
            if key in row:
                totals[key] += f(row, key) if key not in ("routes", "gpu_hits", "cpu_fallback") else i(row, key)

    report = {
        "schema": "colibri.compute_islands_v0.layer_classification.v1",
        "inputs": {
            "layer_csv": args.input,
            "layers": len(layers),
            "measurement_kind": "existing lightweight overlap/timer aggregate",
            "cross_clock_warning": True,
        },
        "definitions": {
            "cpu_window_ms": "host CPU/fallback window recorded between issue and take",
            "gpu_event_ms": "sum of resident CUDA event spans; overlapped work is not added to wall time",
            "boundary_residual_ms": "max(0, issue_to_complete - max(cpu_window, gpu_event))",
            "classification": "boundary if residual > 0.5 ms and >10% dominant window; otherwise CPU/GPU dominant within 10%, or balanced",
            "limitation": "classification is aggregate evidence, not a synchronized host/GPU timeline",
        },
        "aggregate": {
            **totals,
            "mean_issue_to_complete_ms": totals["issue_to_complete_ms"] / len(layers),
            "mean_cpu_window_ms": totals["cpu_window_ms"] / len(layers),
            "mean_gpu_event_ms": totals["gpu_event_ms"] / len(layers),
            "mean_boundary_residual_ms": sum(x["boundary_residual_ms"] for x in layers) / len(layers),
            "mean_imbalance_abs_ms": sum(x["imbalance_abs_ms"] for x in layers) / len(layers),
            "classification_counts": counts,
        },
        "layers": layers,
        "gate_status": {
            "A_boundary_understood": "PASS-provisional",
            "B_explicit_executor": "PASS-structural-existing-QtIslandWork",
            "C_overlap": "PASS-observed-but-not-cross-clock-synchronized",
            "D_makespan_model": "PASS-provisional",
            "E_performance": "NOT_TESTED-by-this-analysis",
            "F_scheduler_readiness": "NOT_READY; cost variance requires matched controlled executor data",
        },
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.md:
        ordered = sorted(layers, key=lambda x: x["issue_to_complete_ms"], reverse=True)
        lines = [
            "# Compute Islands v0 — layer classification",
            "",
            f"Input: `{args.input}`; layers: {len(layers)}.",
            "This is aggregate lightweight instrumentation. Host and CUDA clocks are not treated as synchronized.",
            "",
            "## Aggregate",
            "",
            f"- Mean layer envelope: **{report['aggregate']['mean_issue_to_complete_ms']:.3f} ms**",
            f"- Mean CPU window: **{report['aggregate']['mean_cpu_window_ms']:.3f} ms**",
            f"- Mean GPU event work: **{report['aggregate']['mean_gpu_event_ms']:.3f} ms**",
            f"- Mean residual boundary: **{report['aggregate']['mean_boundary_residual_ms']:.3f} ms**",
            f"- Mean CPU/GPU imbalance: **{report['aggregate']['mean_imbalance_abs_ms']:.3f} ms**",
            f"- Class counts: `{json.dumps(counts, sort_keys=True)}`",
            "",
            "## Largest layer envelopes",
            "",
            "| Layer | CPU ms | GPU ms | Envelope ms | Residual ms | Hit rate | Class |",
            "|---:|---:|---:|---:|---:|---:|---|",
        ]
        for x in ordered[:10]:
            lines.append("| {layer} | {cpu_window_ms:.3f} | {gpu_event_ms:.3f} | {issue_to_complete_ms:.3f} | {boundary_residual_ms:.3f} | {gpu_hit_rate:.1%} | {critical_path_class} |".format(**x))
        lines += [
            "",
            "## Interpretation",
            "",
            "This report is a makespan diagnostic, not a scheduler. It supports the island model only provisionally: the observed layer envelope is compared with the maximum of the CPU and GPU local windows, while the residual is retained as exposed boundary/measurement territory.",
        ]
        Path(args.md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
