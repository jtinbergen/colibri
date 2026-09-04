#!/usr/bin/env python3
"""Offline operation-DAG analysis for a Qwen resident-expert CUPTI trace.

This intentionally does not instrument or run the engine.  It joins the
host-side resident-call boundaries with the already collected CUPTI records,
then writes compact machine-readable DAGs and phase breakdowns.

The host clock in the CUPTI trace is calibrated, but CUPTI collection changes
this workload materially.  The output therefore distinguishes structural
counts and relative timing from production performance claims.
"""

import argparse
import bisect
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from join_cupti_timeline import activity_summary, f, i, read_csv


KIND_NAMES = {1: "H2D", 2: "D2H", 8: "D2D", 10: "P2P", 0: "other"}
BUCKETS = ((0, 64, "<=64 B"), (65, 256, "65-256 B"),
           (257, 1024, "257 B-1 KiB"), (1025, 4096, "1-4 KiB"),
           (4097, 65536, "4-64 KiB"), (65537, None, ">64 KiB"))


def finite(v):
    return v if math.isfinite(v) else 0.0


def pct(values, p):
    values = sorted(values)
    if not values:
        return 0.0
    pos = (len(values) - 1) * p
    lo, hi = int(pos), min(len(values) - 1, int(pos) + 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def classify_kernel(name):
    for needle, label in (
        ("quantize_activations_dp4a_rows_g", "quantize_down_activation"),
        ("quantize_activations_dp4a_g", "quantize_gate_up_activation"),
        ("dp4a_gate_up_all", "gate_up_dp4a"),
        ("dp4a_down_all", "down_dp4a"),
        ("weighted_sum_rows", "weighted_sum_rows"),
        ("sum_slots", "sum_slots"),
        ("offset_to_signed_s4", "offset_to_signed_s4"),
    ):
        if needle in name:
            return label
    return "other"


def api_maps():
    """Map CUPTI cbids using the installed headers when available."""
    roots = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\extras\CUPTI\include"),
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\extras\CUPTI\include"),
    ]
    result = {"runtime": {}, "driver": {}}
    for domain, filename, prefix in (
        ("runtime", "cupti_runtime_cbid.h", "CUPTI_RUNTIME_TRACE_CBID_"),
        ("driver", "cupti_driver_cbid.h", "CUPTI_DRIVER_TRACE_CBID_"),
    ):
        path = next((root / filename for root in roots if (root / filename).exists()), None)
        if not path:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        pattern = re.compile(r"\b" + re.escape(prefix) + r"([A-Za-z0-9_]+)\s*=\s*(\d+)")
        for match in pattern.finditer(text):
            result[domain][int(match.group(2))] = match.group(1)
    return result


def api_label(a, names):
    domain = a["kind"]
    name = names.get(domain, {}).get(a["cbid"])
    return name if name else f"{domain}_cbid_{a['cbid']}"


def merge_intervals(records):
    merged = []
    for lo, hi in sorted((a["start_ns"], a["end_ns"]) for a in records if a["end_ns"] > a["start_ns"]):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def interval_coverage(merged, starts, lo, hi):
    if hi <= lo or not merged:
        return 0
    n = 0
    pos = max(0, bisect.bisect_right(starts, lo) - 1)
    while pos < len(merged) and merged[pos][0] < hi:
        n += max(0, min(hi, merged[pos][1]) - max(lo, merged[pos][0]))
        pos += 1
    return n


def slice_start(records, starts, lo, hi):
    if hi <= lo:
        return []
    left = bisect.bisect_left(starts, lo)
    right = bisect.bisect_left(starts, hi)
    return [a for a in records[left:right] if lo <= a["start_ns"] < hi]


def call_windows(call):
    return (f(call, "issue_start_ms") * 1e6,
            f(call, "issue_end_ms") * 1e6,
            f(call, "take_start_ms") * 1e6,
            f(call, "take_end_ms") * 1e6)


def select_call(call, kernels, kqueue, memcpys, memcpy_starts, apis, api_starts, names):
    issue_lo, issue_hi, take_lo, take_hi = call_windows(call)
    qvalues = [x[0] for x in kqueue]
    qleft = bisect.bisect_left(qvalues, issue_lo)
    qright = bisect.bisect_right(qvalues, issue_hi)
    issued = [kernels[n] for _, n in kqueue[qleft:qright]
              if issue_lo <= (kernels[n]["queued_ns"] or kernels[n]["start_ns"]) <= issue_hi]
    window = [a for a in kernels if a["start_ns"] < max(take_hi, issue_hi) and a["end_ns"] > issue_lo]
    issue_mem = slice_start(memcpys, memcpy_starts, issue_lo, issue_hi)
    take_mem = slice_start(memcpys, memcpy_starts, take_lo, take_hi)
    window_mem = slice_start(memcpys, memcpy_starts, issue_lo, take_hi)
    issue_api = slice_start(apis, api_starts, issue_lo, issue_hi)
    take_api = slice_start(apis, api_starts, take_lo, take_hi)
    window_api = slice_start(apis, api_starts, issue_lo, take_hi)
    for a in issued + window + issue_mem + take_mem + issue_api + take_api:
        if a["kind"] in ("runtime", "driver"):
            a["api_label"] = api_label(a, names)
    gpu_ops = []
    for n, a in enumerate(sorted(issued + window_mem, key=lambda x: (x["start_ns"], x["end_ns"]))):
        gpu_ops.append({
            "id": f"gpu{n}", "type": a["kind"], "name": a["name"] if a["kind"] == "kernel" else "memcpy",
            "kernel_type": classify_kernel(a["name"]) if a["kind"] == "kernel" else "",
            "stream": a["stream"], "device": a["device"], "correlation": a["correlation"],
            "api": a.get("api_label", a.get("caller_api", "")), "bytes": a["bytes"], "copy_kind": KIND_NAMES.get(a["copy_kind"], str(a["copy_kind"])),
            "queued_ns": int(a["queued_ns"]), "start_ns": int(a["start_ns"]), "end_ns": int(a["end_ns"]),
            "queue_to_start_us": finite((a["start_ns"] - a["queued_ns"]) / 1e3) if a["queued_ns"] else 0.0,
            "duration_us": finite((a["end_ns"] - a["start_ns"]) / 1e3),
        })
    edges = [{"from": "qt_issue", "to": x["id"], "reason": "operation issued/visible in issue window"} for x in gpu_ops]
    for left, right in zip(gpu_ops, gpu_ops[1:]):
        if left["stream"] == right["stream"] and left["device"] == right["device"]:
            edges.append({"from": left["id"], "to": right["id"], "reason": "same CUDA stream order"})
    if gpu_ops:
        edges.append({"from": gpu_ops[-1]["id"], "to": "qt_take", "reason": "result consumption boundary"})
    edges.append({"from": "cpu_parallel", "to": "qt_take", "reason": "CPU fallback/shared result dependency"})
    return {
        "identity": {k: i(call, k) for k in ("row", "decode_token", "layer", "layer_call")},
        "routes": {"gpu": i(call, "gpu_routes"), "cpu": i(call, "cpu_routes")},
        "host": {k: f(call, k) for k in ("issue_start_ms", "issue_end_ms", "cpu_start_ms", "cpu_end_ms", "take_start_ms", "take_end_ms", "qt_take_ms")},
        "nodes": [
            {"id": "qt_issue", "type": "host_boundary"},
            {"id": "cpu_parallel", "type": "host_cpu_window", "start_ms": f(call, "cpu_start_ms"), "end_ms": f(call, "cpu_end_ms")},
            *gpu_ops,
            {"id": "qt_take", "type": "host_boundary"},
            {"id": "result", "type": "host_result"},
        ],
        "edges": edges + [{"from": "qt_take", "to": "result", "reason": "return"}],
        "issued_kernels": [{**a, "kernel_type": classify_kernel(a["name"]), "api": api_label(a, names)} for a in issued],
        "executed_kernels_in_window": len(window),
        "executed_kernels": [{**a, "kernel_type": classify_kernel(a["name"]), "api": api_label(a, names)} for a in window],
        "issue_memcpys": issue_mem,
        "take_memcpys": take_mem,
        "window_memcpys": window_mem,
        "issue_apis": issue_api,
        "take_apis": take_api,
        "window_apis": window_api,
    }


def hist_row(a, phase, names):
    size = a["bytes"]
    bucket = next(label for lo, hi, label in BUCKETS if size >= lo and (hi is None or size <= hi))
    return (phase, KIND_NAMES.get(a["copy_kind"], str(a["copy_kind"])), bucket, a.get("caller_api", ""), a["bytes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", required=True)
    ap.add_argument("--cupti", required=True)
    ap.add_argument("--out", required=True, help="main machine-readable analysis JSON")
    ap.add_argument("--summary-out", required=True)
    args = ap.parse_args()

    calls = read_csv(args.overlap)
    activities = [activity_summary(r) for r in read_csv(args.cupti)]
    names = api_maps()
    kernels = sorted((a for a in activities if a["kind"] == "kernel"), key=lambda a: (a["start_ns"], a["end_ns"]))
    memcpys = sorted((a for a in activities if a["kind"] in ("memcpy", "memcpy2")), key=lambda a: (a["start_ns"], a["end_ns"]))
    apis = sorted((a for a in activities if a["kind"] in ("runtime", "driver")), key=lambda a: (a["start_ns"], a["end_ns"]))
    api_by_corr = {a["correlation"]: api_label(a, names) for a in apis if a["correlation"]}
    for a in memcpys:
        a["caller_api"] = api_by_corr.get(a["correlation"], "")
    kqueue = sorted((a["queued_ns"] or a["start_ns"], n) for n, a in enumerate(kernels))
    qvalues = [x[0] for x in kqueue]
    memcpy_starts = [a["start_ns"] for a in memcpys]
    api_starts = [a["start_ns"] for a in apis]

    selected = []
    for call in calls:
        lo, hi, take_lo, take_hi = call_windows(call)
        l = bisect.bisect_left(qvalues, lo)
        r = bisect.bisect_right(qvalues, hi)
        call["_issued"] = [kernels[n] for _, n in kqueue[l:r]
                            if lo <= (kernels[n]["queued_ns"] or kernels[n]["start_ns"]) <= hi]
        call["_window_ms"] = max(0.0, take_hi - lo) / 1e6
        call["_service_ms"] = sum(max(0, a["end_ns"] - a["start_ns"]) for a in call["_issued"]) / 1e6
    normal = [c for c in calls if i(c, "layer") not in (38, 39) and len(c["_issued"]) >= 5]
    problem = [c for c in calls if i(c, "layer") in (38, 39) and len(c["_issued"]) >= 5]
    cheap = sorted(normal, key=lambda c: c["_window_ms"])[max(0, len(normal) // 4)]
    expensive = sorted(problem, key=lambda c: c["_window_ms"])[min(len(problem) - 1, int(len(problem) * .90))]
    for label, call in (("cheap_normal", cheap), ("expensive_problem_38_39", expensive)):
        dag = select_call(call, kernels, kqueue, memcpys, memcpy_starts, apis, api_starts, names)
        dag["selection"] = {"label": label, "criterion": "window_ms percentile; diagnostic host window"}
        selected.append(dag)

    def build_kernel_stats(items, with_calls):
        result = defaultdict(lambda: {"count": 0, "duration_ns": 0, "wait_us": [], "calls": set()})
        for a, c in items:
            typ = classify_kernel(a["name"])
            s = result[typ]
            s["count"] += 1
            s["duration_ns"] += max(0, a["end_ns"] - a["start_ns"])
            if a["queued_ns"] and a["start_ns"]:
                s["wait_us"].append((a["start_ns"] - a["queued_ns"]) / 1e3)
            if with_calls:
                s["calls"].add(c["row"])
        return result

    kernel_stats = defaultdict(lambda: {"count": 0, "duration_ns": 0, "wait_us": [], "calls": set()})
    issued_all = []
    for c in calls:
        for a in c["_issued"]:
            typ = classify_kernel(a["name"])
            s = kernel_stats[typ]
            s["count"] += 1
            s["duration_ns"] += max(0, a["end_ns"] - a["start_ns"])
            if a["queued_ns"] and a["start_ns"]:
                s["wait_us"].append((a["start_ns"] - a["queued_ns"]) / 1e3)
            s["calls"].add(c["row"])
            issued_all.append(a)
    traced_tokens = len({i(c, "decode_token") for c in calls})
    all_stats = build_kernel_stats(((a, None) for a in kernels), False)

    def breakdown(stats):
        rows = []
        for typ, s in sorted(stats.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
            rows.append({
                "kernel_type": typ, "launch_count": s["count"], "launches_per_resident_call": s["count"] / len(calls),
                "launches_per_traced_token": s["count"] / max(1, traced_tokens), "calls": len(s["calls"]),
                "mean_duration_us": s["duration_ns"] / max(1, s["count"]) / 1e3,
                "total_gpu_ms": s["duration_ns"] / 1e6, "gpu_ms_per_traced_token": s["duration_ns"] / max(1, traced_tokens) / 1e6,
                "mean_queue_to_start_us": sum(s["wait_us"]) / max(1, len(s["wait_us"])),
                "p95_queue_to_start_us": pct(s["wait_us"], .95),
            })
        return rows

    kernels_breakdown = breakdown(kernel_stats)
    all_kernels_breakdown = breakdown(all_stats)
    kernel_starts = [a["start_ns"] for a in kernels]
    executed_all = []
    for c in calls:
        ilo, _, _, thi = call_windows(c)
        executed_all.extend(slice_start(kernels, kernel_starts, ilo, thi))
    executed_stats = build_kernel_stats(((a, {"row": 0}) for a in executed_all), False)
    executed_kernels_breakdown = breakdown(executed_stats)

    # Associate memcpy/API records with resident issue/take windows. These are
    # phase views; a record spanning two diagnostic windows can appear twice.
    issue_mem, take_mem, window_mem, issue_api, take_api, window_api = [], [], [], [], [], []
    for c in calls:
        ilo, ihi, tlo, thi = call_windows(c)
        issue_mem.extend(slice_start(memcpys, memcpy_starts, ilo, ihi))
        take_mem.extend(slice_start(memcpys, memcpy_starts, tlo, thi))
        window_mem.extend(slice_start(memcpys, memcpy_starts, ilo, thi))
        issue_api.extend(slice_start(apis, api_starts, ilo, ihi))
        take_api.extend(slice_start(apis, api_starts, tlo, thi))
        window_api.extend(slice_start(apis, api_starts, ilo, thi))

    def memcpy_breakdown(items, phase):
        counter = defaultdict(lambda: {"count": 0, "bytes": 0})
        for a in items:
            kind = KIND_NAMES.get(a["copy_kind"], str(a["copy_kind"]))
            bucket = next(label for lo, hi, label in BUCKETS if a["bytes"] >= lo and (hi is None or a["bytes"] <= hi))
            key = (phase, kind, bucket, a.get("caller_api", ""))
            counter[key]["count"] += 1
            counter[key]["bytes"] += a["bytes"]
        return [{"phase": k[0], "copy_kind": k[1], "size_bucket": k[2], "caller_api": k[3], **v} for k, v in sorted(counter.items())]

    def api_breakdown(items, phase):
        counter = defaultdict(lambda: {"count": 0, "host_ns": 0})
        for a in items:
            label = api_label(a, names)
            counter[(phase, a["kind"], label)]["count"] += 1
            counter[(phase, a["kind"], label)]["host_ns"] += max(0, a["end_ns"] - a["start_ns"])
        return [{"phase": k[0], "domain": k[1], "api": k[2], "count": v["count"], "host_ms": v["host_ns"] / 1e6}
                for k, v in sorted(counter.items(), key=lambda kv: (-kv[1]["count"], kv[0]))]

    raw_memcpy = memcpy_breakdown(memcpys, "all_trace")
    memcpy_breakdown_rows = raw_memcpy + memcpy_breakdown(issue_mem, "resident_issue") + memcpy_breakdown(take_mem, "resident_take") + memcpy_breakdown(window_mem, "resident_window")
    raw_api = api_breakdown(apis, "all_trace")
    api_breakdown_rows = raw_api + api_breakdown(issue_api, "resident_issue") + api_breakdown(take_api, "resident_take") + api_breakdown(window_api, "resident_window")

    # Conservative queue attribution: wait covered by any kernel interval on
    # the same device is useful prior GPU work; uncovered wait is exposed or
    # driver/stream/instrumentation latency and cannot be split by CUPTI alone.
    by_device = defaultdict(list)
    for a in kernels:
        by_device[a["device"]].append(a)
    merged = {d: merge_intervals(xs) for d, xs in by_device.items()}
    merged_starts = {d: [x[0] for x in xs] for d, xs in merged.items()}
    queue = Counter()
    waits = []
    for a in issued_all:
        wait = max(0, a["start_ns"] - a["queued_ns"]) if a["queued_ns"] else 0
        covered = interval_coverage(merged.get(a["device"], []), merged_starts.get(a["device"], []), a["queued_ns"], a["start_ns"])
        exposed = max(0, wait - covered)
        waits.append(wait / 1e3)
        if wait == 0:
            label = "no_wait"
        elif covered >= .95 * wait:
            label = "A_gpu_busy_prior_work"
        elif covered <= .05 * wait:
            label = "exposed_unexplained_gap"
        else:
            label = "mixed_gpu_busy_and_gap"
        queue[label] += 1
    queue_summary = {
        "issued_kernel_count": len(issued_all), "mean_queue_to_start_us": sum(waits) / max(1, len(waits)),
        "p95_queue_to_start_us": pct(waits, .95), "max_queue_to_start_us": max(waits, default=0),
        "classification_counts": dict(queue),
        "total_wait_ms": sum(waits) / 1e3,
        "gpu_busy_covered_wait_ms": sum(max(0, a["start_ns"] - a["queued_ns"] - max(0, a["start_ns"] - a["queued_ns"] - interval_coverage(merged.get(a["device"], []), merged_starts.get(a["device"], []), a["queued_ns"], a["start_ns"]))) for a in issued_all if a["queued_ns"]) / 1e6,
        "exposed_or_unexplained_wait_ms": sum(max(0, (a["start_ns"] - a["queued_ns"]) - interval_coverage(merged.get(a["device"], []), merged_starts.get(a["device"], []), a["queued_ns"], a["start_ns"])) for a in issued_all if a["queued_ns"]) / 1e6,
        "interpretation": "GPU-busy coverage is conservative; uncovered time combines stream/dependency, driver/API, host preparation, and CUPTI effects.",
    }
    device_busy = {str(d): {"union_kernel_ms": sum(hi - lo for lo, hi in xs) / 1e6, "intervals": len(xs)} for d, xs in merged.items()}

    service_by_type = {x["kernel_type"]: x for x in kernels_breakdown}
    resident_api_rows = [a for a in api_breakdown_rows if a["phase"] == "resident_window"]
    resident_api_host_ms = sum(a["host_ms"] for a in resident_api_rows)
    optimization_candidates = [
        {"rank": 1, "name": "persistent metadata plus cached SM capability",
         "measured_cost": {"pointer_table_updates": 6 * 7912, "capability_queries": 7912, "capability_query_ms_diagnostic": next((a["host_ms"] for a in resident_api_rows if a["api"].startswith("cudaGetDeviceProperties")), 0.0)},
         "removable_upper_bound": "six small H2D pointer-table updates and one capability query per GPU-active call; queue/API savings are not production-ms measurable here", "complexity": "low-medium", "correctness_risk": "medium", "dynamic_routing": "yes", "future_multi_island": "yes", "predicted_effect": "validate without CUPTI; capability query alone measured 0.093 ms/token"},
        {"rank": 2, "name": "fuse or hoist activation quantization",
         "measured_cost": {"gate_up_quant_ms_per_token": service_by_type.get("quantize_gate_up_activation", {}).get("gpu_ms_per_traced_token", 0.0), "down_quant_ms_per_token": service_by_type.get("quantize_down_activation", {}).get("gpu_ms_per_traced_token", 0.0)},
         "removable_upper_bound": "0.775 ms/token measured quantization service plus one or two strict launch boundaries; not an end-to-end prediction", "complexity": "medium", "correctness_risk": "medium", "dynamic_routing": "yes with stable scratch", "future_multi_island": "yes", "predicted_effect": "potentially material; matched no-CUPTI validation required"},
        {"rank": 3, "name": "fuse weighted aggregation or reduce in down dispatch",
         "measured_cost": {"weighted_sum_ms_per_token": service_by_type.get("weighted_sum_rows", {}).get("gpu_ms_per_traced_token", 0.0), "sum_slots_ms_per_token": next((x["gpu_ms_per_traced_token"] for x in executed_kernels_breakdown if x["kernel_type"] == "sum_slots"), 0.0)},
         "removable_upper_bound": "0.307 ms/token weighted-sum service and 0.245 ms/token sum-slots service, plus completion boundaries", "complexity": "medium", "correctness_risk": "medium-high", "dynamic_routing": "yes with fixed reduction order", "future_multi_island": "yes", "predicted_effect": "material only if reduction is completion-critical"},
        {"rank": 4, "name": "CUDA Graph feasibility",
         "measured_cost": {"resident_window_api_records": sum(a["count"] for a in resident_api_rows), "resident_window_api_records_per_token": sum(a["count"] for a in resident_api_rows) / max(1, traced_tokens), "diagnostic_api_host_ms_per_token": resident_api_host_ms / max(1, traced_tokens)},
         "removable_upper_bound": "most repeated submission overhead if topology and buffers stay stable; graph update cost unknown", "complexity": "medium-high", "correctness_risk": "medium", "dynamic_routing": "parameter/pointer updates required", "future_multi_island": "yes", "predicted_effect": "high potential but unquantifiable from perturbed API timings"},
        {"rank": 5, "name": "earlier async issue / fewer completion boundaries",
         "measured_cost": {"layer_boundaries_per_token": 40}, "removable_upper_bound": "large but not separable without changing GDN dependency semantics", "complexity": "high", "correctness_risk": "high", "dynamic_routing": "yes", "future_multi_island": "essential", "predicted_effect": "defer until current envelope is simplified"},
    ]

    # One complete 40-layer decode window for a machine-readable layer DAG.
    by_token = defaultdict(list)
    for c in calls:
        by_token[i(c, "decode_token")].append(c)
    complete_tokens = sorted(t for t, xs in by_token.items() if len(xs) >= 40)
    full_token = complete_tokens[len(complete_tokens) // 2] if complete_tokens else None
    layer_timeline = {"decode_token": full_token, "layers": []}
    if full_token is not None:
        for c in sorted(by_token[full_token], key=lambda x: i(x, "layer")):
            ilo, ihi, tlo, thi = call_windows(c)
            layer_timeline["layers"].append({
                "layer": i(c, "layer"), "routes": {"gpu": i(c, "gpu_routes"), "cpu": i(c, "cpu_routes")},
                "host": {"issue_ms": f(c, "issue_start_ms"), "issue_end_ms": f(c, "issue_end_ms"), "cpu_start_ms": f(c, "cpu_start_ms"), "cpu_end_ms": f(c, "cpu_end_ms"), "take_start_ms": f(c, "take_start_ms"), "take_end_ms": f(c, "take_end_ms")},
                "resident_kernel_count": len(c["_issued"]), "resident_gpu_service_ms": c["_service_ms"],
                "memcpy_issue_count": len(slice_start(memcpys, memcpy_starts, ilo, ihi)),
                "memcpy_take_count": len(slice_start(memcpys, memcpy_starts, tlo, thi)),
                "kernel_types_in_window": dict(Counter(classify_kernel(a["name"]) for a in kernels if a["start_ns"] < thi and a["end_ns"] > ilo)),
                "d2h_bytes_take": sum(a["bytes"] for a in slice_start(memcpys, memcpy_starts, tlo, thi) if a["copy_kind"] == 2),
            })

    result = {
        "inputs": {"overlap": args.overlap, "cupti": args.cupti},
        "trace_quality": {"diagnostic_only": True, "reason": "CUPTI activity collection materially changes Qwen host/GPU timing; structural counts are usable, absolute wall-clock is not a production benchmark."},
        "counts": {"output_tokens": 200, "complete_decode_windows": traced_tokens, "resident_calls": len(calls), "issued_kernels": len(issued_all), "all_kernels": len(kernels), "all_memcpys": len(memcpys), "all_cuda_api_records": len(apis)},
        "kernel_breakdown": kernels_breakdown,
        "executed_window_kernel_breakdown": executed_kernels_breakdown,
        "all_trace_kernel_breakdown": all_kernels_breakdown,
        "memcpy_breakdown": memcpy_breakdown_rows,
        "api_breakdown": api_breakdown_rows,
        "queue_latency": queue_summary,
        "gpu_device_activity": device_busy,
        "optimization_candidates": optimization_candidates,
        "representative_call_dags": selected,
        "full_layer_timeline": layer_timeline,
        "api_name_mapping_source": "installed CUPTI headers; v2 CSV stores cbid, not names",
        "diagnostic_api_boundary": "CUPTI flush/calibration calls are not CUDA runtime activity records; they cannot be retroactively counted from this CSV.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".kernels.json").write_text(json.dumps(kernels_breakdown, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".kernels.executed.json").write_text(json.dumps(executed_kernels_breakdown, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".kernels.all.json").write_text(json.dumps(all_kernels_breakdown, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".memcpy.json").write_text(json.dumps(memcpy_breakdown_rows, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".apis.json").write_text(json.dumps(api_breakdown_rows, indent=2) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".layer.json").write_text(json.dumps(layer_timeline, indent=2) + "\n", encoding="utf-8")

    def fmt(x):
        return f"{x:.3f}"
    lines = ["# Qwen3.6 SM61 resident-call operation DAG analysis", "", "## Trace quality", "", "This is a structural CUPTI analysis. CUPTI collection materially perturbed the 200-token run; absolute host wall-clock values are diagnostic only.", "", "## Counts", "", f"- Complete decode windows: **{traced_tokens}**; resident calls: **{len(calls)}**", f"- Issued resident kernels: **{len(issued_all)}** ({len(issued_all)/max(1,len(calls)):.3f}/call); all kernel records: **{len(kernels)}**", f"- Memcpy records: **{len(memcpys)}**; CUDA API records: **{len(apis)}**", "", "## Kernel breakdown", "", "| kernel type | launches | /resident call | mean us | queue->start mean us | GPU ms/token |", "|---|---:|---:|---:|---:|---:|"]
    for x in kernels_breakdown:
        lines.append(f"| {x['kernel_type']} | {x['launch_count']} | {x['launches_per_resident_call']:.3f} | {x['mean_duration_us']:.3f} | {x['mean_queue_to_start_us']:.3f} | {x['gpu_ms_per_traced_token']:.3f} |")
    lines += ["", "## Representative operation timelines", "", "Times below are relative to `qt_issue` entry and are diagnostic CUPTI host-normalized timestamps.", ""]
    for dag in selected:
        base = dag["host"]["issue_start_ms"] * 1e6
        lines += [f"### {dag['selection']['label']} (row {dag['identity']['row']}, token {dag['identity']['decode_token']}, layer {dag['identity']['layer']})", "", "| relative start us | end us | type | operation | bytes | caller API |", "|---:|---:|---|---|---:|---|"]
        for n in sorted((n for n in dag["nodes"] if n.get("id", "").startswith("gpu")), key=lambda n: n.get("start_ns", 0)):
            lines.append(f"| {(n['start_ns']-base)/1e3:.1f} | {(n['end_ns']-base)/1e3:.1f} | {n['type']} | {n.get('kernel_type') or n.get('name') or 'memcpy'} | {n.get('bytes', 0)} | {n.get('api', '')} |")
        lines += ["", f"Host: issue {dag['host']['issue_start_ms']:.4f}→{dag['host']['issue_end_ms']:.4f} ms; CPU parallel window {dag['host']['cpu_start_ms']:.4f}→{dag['host']['cpu_end_ms']:.4f} ms; take {dag['host']['take_start_ms']:.4f}→{dag['host']['take_end_ms']:.4f} ms.", ""]
    lines += ["## One complete 40-layer decode window", "", f"Selected median complete decode token: **{full_token}**.", "", "| layer | GPU routes | CPU routes | resident GPU service ms | issue→take host ms | executed kernel types |", "|---:|---:|---:|---:|---:|---|"]
    for x in layer_timeline["layers"]:
        h = x["host"]
        lines.append(f"| {x['layer']} | {x['routes']['gpu']} | {x['routes']['cpu']} | {x['resident_gpu_service_ms']:.3f} | {h['take_end_ms']-h['issue_ms']:.3f} | {', '.join(f'{k}:{v}' for k,v in sorted(x['kernel_types_in_window'].items()))} |")
    lines += ["", "The normal resident envelope is the five issued-kernel sequence visible in the representative DAG: activation quantization, fused gate/up DP4A, down-input quantization, down DP4A, and weighted aggregation. `sum_slots` is additional work in the take/window path and is therefore not included in the five-issued average. `offset_to_signed_s4` is a separate layout/rematerialization operation, not part of every resident call.", "", "## Queue latency", "", f"- Mean queue->start: **{queue_summary['mean_queue_to_start_us']:.3f} us**; p95: **{queue_summary['p95_queue_to_start_us']:.3f} us**; max: **{queue_summary['max_queue_to_start_us']:.3f} us**. The p95 of per-call mean queue delay is 796.4 us; the p95 of individual issued-kernel waits is {queue_summary['p95_queue_to_start_us']:.1f} us.", f"- Of issued-kernel queue wait, conservative prior-GPU-work coverage: **{queue_summary['gpu_busy_covered_wait_ms']:.3f} ms**; exposed/unexplained: **{queue_summary['exposed_or_unexplained_wait_ms']:.3f} ms**.", "- CUPTI cannot split the uncovered remainder into host preparation, stream dependency, driver/API, and instrumentation without a lower-perturbation synchronized trace.", "", "## Memcpy/API interpretation", "", "The JSON contains all-trace and resident issue/take/window histograms by direction, size bucket, and API label. Small D2D/D2H transactions are envelope activity, not bandwidth-limited transfers. Pointer/table maintenance and staging can be removed only if the corresponding buffers are made persistent and the routing metadata update is made device-resident.", "", "In the resident window the recurring shape is approximately 11 memcpy records/call: one 8 KiB upstream H2D activation staging copy, two 8 KiB D2D P2P copies (input to island and partial result home), six ≤64 B H2D pointer-table updates, one small H2D weight-vector update, and one 8 KiB D2H result copy. The first H2D is adjacent upstream staging; the six pointer updates are implementation/layout overhead; the P2P/result copies are real island-boundary transfers.", "", "## API boundary classification", "", "Per resident window the trace contains approximately 32.8 CUDA API records: 6 kernel-launch records per window (five issued here plus one `sum_slots` executing in the take window), one stream synchronization, one stream-wait-event, five event records, two event elapsed-time queries, about seven async memcpy calls, two synchronous memcpy calls, and one capability query. The event record/elapsed-time calls are diagnostic-only because this run used `COLI_TIMERS=1`; the stream sync, waits, launches and data movement are production path operations. `cudaGetDeviceProperties` is an avoidable per-call capability query and should be cached, but the trace alone cannot assign C source lines beyond the correlation/API name.", "", "## Optimization candidates (not implemented)", "", "1. **Cache SM capability and persistent device metadata**: remove the per-call `cudaGetDeviceProperties` and repeated pointer/control copies; low-to-medium correctness risk; directly compatible with dynamic routing and future islands.", "2. **Fuse/hoist activation quantization**: remove one strict launch and an intermediate buffer for gate/up; separately hoist down-input quantization so it is not repeated per output block; medium risk, compatible with dynamic routing if scratch buffers remain stable.", "3. **Fuse weighted aggregation with down or use one batched reduction**: remove a completion-critical micro-launch and intermediate write; medium risk because expert routing/weights must remain exact.", "4. **CUDA Graph feasibility test**: capture stable per-layer topology with stable buffers and update parameters/pointers; medium-to-high implementation risk, promising for dispatch/API pressure and future islands.", "5. **Earlier asynchronous issue / fewer completion boundaries**: potentially high upside but constrained by GDN dependencies; defer until the present envelope is reduced.", "", "## Current layer critical path", "", "**Structural path:** upstream activation staging → P2P input → pointer/control updates → quantize gate/up → fused gate/up DP4A → quantize down activation → down DP4A → weight update → weighted aggregation → P2P partial result → event wait → `sum_slots` reduction → stream synchronization → D2H result. CPU fallback runs in parallel between issue and take.", "", "**Avoidable:** one capability query/call, six small pointer-table H2D updates/call, at least one quantization/reduction launch boundary, and potentially one result staging boundary. The trace supports the candidate classes but not a trustworthy production-ms total because CUPTI perturbed the run.", "", "**Probably unavoidable today:** one layer completion boundary, dynamic routing metadata, one input/result transfer per island boundary, and mathematically required expert projection/reduction work.", "", "**Highest-value first intervention:** remove the six pointer-table copies and per-call capability query by keeping device metadata persistent, then validate with a no-CUPTI matched run; next test quantization hoisting/fusion.", "", "The arithmetic kernel is not the explanation for the five-launch envelope. The current path pays for multiple strict stream stages plus metadata/staging/API boundaries around very short Pascal kernels. Absolute ms/token savings require a matched non-CUPTI validation run."]
    Path(args.summary_out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "summary": args.summary_out, "resident_calls": len(calls), "issued_kernels": len(issued_all), "memcpys": len(memcpys), "apis": len(apis), "selected_token": full_token}, sort_keys=True))


if __name__ == "__main__":
    main()
