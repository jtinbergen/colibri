#!/usr/bin/env python3
"""Run the Colibri side of the resident-memory pressure sweep."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_scheduler_ab import aggregate, run_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-timeout", type=int, default=120)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    common = SimpleNamespace(
        engine=args.engine, model=args.model, prompt=(
            "Explain, in a technically precise but readable way, how a modern mixture-of-experts "
            "language model performs inference. Cover tokenization, attention, router top-k selection, "
            "expert matrix multiplication, quantization, hot expert residency, CPU and GPU memory "
            "movement, PCIe transfers, and why throughput can be limited by memory bandwidth or scheduling. "
            "Use headings and concrete examples. Compare a single Pascal GPU island with a CPU-only fallback "
            "and identify the most important measurements for diagnosing performance."),
        warmup_prompt="Give a brief technical warmup about expert routing.",
        tokens=args.tokens, warmup_tokens=args.warmup_tokens, port=args.port,
        start_timeout=args.start_timeout, request_timeout=args.request_timeout,
    )
    results = []
    for point in manifest["points"]:
        common.expert_gb = 6.0 * point["target_fraction"]
        result = run_one(f"capacity_{point['label']}", Path(point["heat_file"]), common)
        result["target_fraction"] = point["target_fraction"]
        result["target_slots"] = point["target_slots"]
        result["target_expert_gb"] = common.expert_gb
        result["metric_summary"] = aggregate(result["samples"])
        results.append(result)
        print(f"completed {result['label']}: {result['tok_per_s']:.4f} tok/s")
    output = {
        "schema": "colibri.pressure_sweep.v1",
        "configuration": {
            "tokens": args.tokens, "warmup_tokens": args.warmup_tokens,
            "engine": str(args.engine), "model": str(args.model),
            "timers": False, "cupti": False, "lfru_frozen": True,
            "pressure_axis": "explicit CUDA_EXPERT_GB; hot-prefix QTH1 snapshot",
        },
        "manifest": str(args.manifest), "runs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
