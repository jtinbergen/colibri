#!/usr/bin/env python3
"""Combine the resident-memory pressure sweep into JSON, CSV and Markdown."""

import csv
import hashlib
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COLIBRI_FILES = [ROOT / "colibri_pressure_v1" / "results.json",
                 ROOT / "colibri_pressure_v1" / "results_rep4.json"]
LLAMA_FILES = [ROOT / "llama_pressure_v1" / "results_layers.json",
               ROOT / "llama_pressure_v1" / "boundary14.json",
               ROOT / "llama_pressure_v1" / "results.json"]


def sample_stat(run, field):
    values = [s[field] for s in run.get("samples", [])
              if s.get(field) is not None]
    return statistics.fmean(values) if values else None


def resident_count(run):
    for line in run.get("engine_lines", []):
        match = re.search(r"warmstart .*?all (\d+) experts in RAM, (\d+) in VRAM",
                          line)
        if match:
            return int(match.group(2))
    return None


def colibri_observation(run, manifest_by_label, source):
    label = run["label"].removeprefix("capacity_")
    point = manifest_by_label[label]
    summary = run.get("metric_summary", {})

    def metric(field):
        return (summary.get(field, {}).get("mean")
                if summary else sample_stat(run, field))

    return {
        "runtime": "colibri",
        "source": source,
        "label": label,
        "capacity_fraction": point["target_fraction"],
        "target_slots": point["target_slots"],
        "actual_resident_experts": resident_count(run),
        "routed_hit_rate_observed": point["observed_routed_hit_rate"],
        "tok_per_s": run.get("tok_per_s"),
        "ms_per_token": run.get("ms_per_token"),
        "gpu_util_pct": metric("gpu_util_pct"),
        "power_w": metric("power_w"),
        "vram_used_mib": metric("vram_used_mib"),
        "engine_cpu_pct": metric("engine_cpu_pct"),
        "engine_rss_gib": metric("engine_rss_gib"),
        "output_sha256": run.get("output_sha256"),
        "exit_ok": True,
    }


def llama_observation(run, source):
    requested = str(run["requested_gpu_layers"])
    layers = None if requested == "auto" else int(requested)
    generation = run.get("llama_generation_tok_per_s")
    valid = run.get("exit_code") in (0, 130) and generation is not None
    return {
        "runtime": "llama.cpp",
        "source": source,
        "label": run["label"],
        "gpu_layers": layers,
        "capacity_fraction_proxy": None if layers is None else layers / 40.0,
        "tok_per_s": generation if valid else None,
        "wall_tok_per_s": run.get("measured_wall_tok_per_s") if valid else None,
        "gpu_util_pct": run.get("resource_summary", {}).get("gpu_util_pct", {}).get("mean"),
        "power_w": run.get("resource_summary", {}).get("power_w", {}).get("mean"),
        "vram_used_mib": run.get("resource_summary", {}).get("vram_used_mib", {}).get("mean"),
        "engine_cpu_pct": run.get("resource_summary", {}).get("engine_cpu_pct", {}).get("mean"),
        "engine_rss_gib": run.get("resource_summary", {}).get("engine_rss_gib", {}).get("mean"),
        "output_sha256": run.get("output_sha256"),
        "exit_code": run.get("exit_code"),
        "exit_ok": valid,
        "infeasible": not valid,
    }


def load_colibri(manifest):
    observations = []
    manifest_by_label = {p["label"]: p for p in manifest["points"]}
    for path in COLIBRI_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        for run in data["runs"]:
            observations.append(colibri_observation(run, manifest_by_label, path.name))
    return observations


def load_llama():
    observations = []
    seen = set()
    for path in LLAMA_FILES:
        data = json.loads(path.read_text(encoding="utf-8"))
        for run in data["runs"]:
            key = str(run["requested_gpu_layers"])
            # Prefer the refined layer sweep and the explicit 14-layer check.
            if key in seen:
                continue
            seen.add(key)
            observations.append(llama_observation(run, path.name))
    return observations


def main():
    out_dir = ROOT / "pressure_sweep_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "colibri_pressure_v1" / "manifest.json").read_text(encoding="utf-8"))
    colibri = load_colibri(manifest)
    llama = load_llama()

    by_fraction = {}
    for row in colibri:
        by_fraction.setdefault(row["capacity_fraction"], []).append(row["tok_per_s"])
    colibri_summary = []
    for fraction in sorted(by_fraction, reverse=True):
        values = by_fraction[fraction]
        rows = [r for r in colibri if r["capacity_fraction"] == fraction]
        colibri_summary.append({
            "capacity_fraction": fraction,
            "target_slots": rows[0]["target_slots"],
            "observations": len(values),
            "mean_tok_per_s": statistics.fmean(values),
            "median_tok_per_s": statistics.median(values),
            "min_tok_per_s": min(values),
            "max_tok_per_s": max(values),
            "mean_ms_per_token": statistics.fmean(r["ms_per_token"] for r in rows),
            "routed_hit_rate": rows[0]["routed_hit_rate_observed"],
            "actual_resident_experts": rows[0]["actual_resident_experts"],
        })

    valid_llama = [r for r in llama if r["exit_ok"]]
    max_colibri = max(r["tok_per_s"] for r in colibri)
    min_llama = min(r["tok_per_s"] for r in valid_llama)
    report = {
        "schema": "colibri.pressure_sweep.report.v1",
        "configuration": {
            "colibri_prompt_tokens": 256,
            "colibri_warmup_tokens": 32,
            "llama_prompt_tokens": 256,
            "same_seed_temperature": "llama seed=1,temp=0; Colibri temperature=0",
            "colibri_pressure_axis": "CUDA_EXPERT_GB with fixed hot-prefix QTH1 snapshot",
            "llama_pressure_axis": "--gpu-layers on Vulkan0; proxy, not expert-byte equivalence",
            "gpu": "NVIDIA GeForce GTX 1080, PCIe Gen3 x16, SM61",
            "colibri_model": "ornith15_i4_gs64_bf16",
            "llama_model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "diagnostics": "COLI_TIMERS=0, no CUPTI, no timing dump",
            "commands": {
                "colibri": "python c/tests/run_colibri_pressure.py --manifest c/tests/colibri_pressure_v1/manifest.json --engine D:/src/colibri/c/coli --model C:/Users/jaapj/.lmstudio/models/ornith15_i4_gs64_bf16 --tokens 256 --warmup-tokens 32 --port 8000",
                "llama_layers": "python c/tests/run_llama_pressure.py --exe C:/Users/jaapj/src/llama.cpp-vulkan/build/bin/llama-cli.exe --model C:/Users/jaapj/.lmstudio/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --tokens 256 --points 2,4,6,8,10,12",
                "llama_boundary": "python c/tests/run_llama_pressure.py --exe C:/Users/jaapj/src/llama.cpp-vulkan/build/bin/llama-cli.exe --model C:/Users/jaapj/.lmstudio/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --tokens 256 --points 14",
            },
        },
        "classification": {
            "result": "NO_CROSSOVER_OBSERVED_ON_TESTED_RANGE",
            "comparability": "NON_EQUIVALENT_PRESSURE_AXES",
            "reason": "llama.cpp uses layer offload as its available pressure control; Colibri uses routed expert capacity",
            "colibri_max_tok_per_s": max_colibri,
            "llama_min_valid_generation_tok_per_s": min_llama,
            "llama_first_infeasible_layers": 16,
            "llama_last_valid_layers": 14,
        },
        "colibri_observations": colibri,
        "colibri_summary": colibri_summary,
        "llama_observations": llama,
        "notes": [
            "A generation statistic is retained for llama exit code 130 when the normal Generation line exists; load failures are infeasible, not zero throughput.",
            "The two Colibri repetitions have identical per-point output hashes but materially different rates, so the curve is variance-limited.",
            "Cross-runtime output hashes are not expected to match because the model containers/quantization differ.",
        ],
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with (out_dir / "observations.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = colibri + llama
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = ["# Resident-memory pressure sweep", "",
          "Result: **no crossover observed**; pressure axes are explicitly non-equivalent.", "",
          "## Colibri", "", "| Capacity | Slots | Routed hit | Run 1 tok/s | Run 2 tok/s | Median tok/s |", "|---:|---:|---:|---:|---:|---:|"]
    for row in colibri_summary:
        values = [r["tok_per_s"] for r in colibri if r["capacity_fraction"] == row["capacity_fraction"]]
        md.append("| {0:.1%} | {1} | {2:.1%} | {3:.3f} | {4:.3f} | {5:.3f} |".format(
            row["capacity_fraction"], row["target_slots"], row["routed_hit_rate"],
            values[0], values[1], row["median_tok_per_s"]))
    md += ["", "## llama.cpp", "", "| GPU layers | Proxy fraction | Generation tok/s | GPU util | Power | Status |", "|---:|---:|---:|---:|---:|---|"]
    for row in sorted(llama, key=lambda r: (r["gpu_layers"] is None,
                                            999 if r["gpu_layers"] is None else r["gpu_layers"])):
        md.append("| {0} | {1} | {2} | {3} | {4} | {5} |".format(
            "auto" if row["gpu_layers"] is None else row["gpu_layers"],
            "n/a" if row["capacity_fraction_proxy"] is None else f"{row['capacity_fraction_proxy']:.1%}",
            "n/a" if row["tok_per_s"] is None else f"{row['tok_per_s']:.3f}",
            "n/a" if row["gpu_util_pct"] is None else f"{row['gpu_util_pct']:.1f}%",
            "n/a" if row["power_w"] is None else f"{row['power_w']:.1f} W",
            "valid" if row["exit_ok"] else "infeasible"))
    md += ["", "## Interpretation", "",
           "Colibri's measured maximum is {:.3f} tok/s; llama.cpp's slowest valid generation point is {:.3f} tok/s (CPU-only).".format(max_colibri, min_llama),
           "llama.cpp remains executable through 14 offloaded layers and fails at 16 because Vulkan cannot allocate the required device buffer.",
           "The tested Qwen3.6/model-size regime therefore does not expose a Colibri crossover. The result does not answer the oversized-model regime because the two pressure controls are not physically equivalent.",
           "", "Recommended next experiment: repeat this protocol on a model/working set that exceeds llama.cpp's feasible offload boundary while keeping Colibri's storage-backed expert pool operational."]
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out_dir / "report.json")
    print(out_dir / "observations.csv")
    print(out_dir / "report.md")


if __name__ == "__main__":
    main()
