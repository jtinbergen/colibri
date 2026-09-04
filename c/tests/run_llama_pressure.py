#!/usr/bin/env python3
"""Run a llama.cpp layer-offload pressure sweep with resource sampling."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from run_scheduler_ab import gpu_sample


PROMPT = (
    "Explain, in a technically precise but readable way, how a modern mixture-of-experts "
    "language model performs inference. Cover tokenization, attention, router top-k selection, "
    "expert matrix multiplication, quantization, hot expert residency, CPU and GPU memory "
    "movement, PCIe transfers, and why throughput can be limited by memory bandwidth or scheduling. "
    "Use headings and concrete examples. Compare a single Pascal GPU island with a CPU-only fallback "
    "and identify the most important measurements for diagnosing performance.")


def run_one(label, layers, args):
    if layers == "auto":
        device, gpu_layers = "Vulkan0", "auto"
        fraction = None
    elif int(layers) == 0:
        device, gpu_layers = "none", "0"
        fraction = 0.0
    else:
        device, gpu_layers = "Vulkan0", str(layers)
        fraction = int(layers) / args.layer_count
    command = [str(args.exe), "-m", str(args.model), "-p", args.prompt,
               "-n", str(args.tokens), "-t", "10", "-c", "16384",
               "-b", "512", "-ub", "512", "--seed", "1", "--temp", "0",
               "--reasoning", "off", "--simple-io", "--no-display-prompt",
               "--no-conversation", "--device", device, "--gpu-layers", gpu_layers,
               "--fit", "on", "--fit-target", "1024"]
    env = dict(os.environ)
    env["PATH"] = (str(args.exe.parent) + os.pathsep +
                   r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.1\bin" +
                   os.pathsep + env.get("PATH", ""))
    popen_kwargs = {
        "cwd": str(args.exe.parent), "env": env,
        "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
        "text": True, "bufsize": 1,
    }
    if os.name == "nt":
        # Keep llama-cli from inheriting/creating a visible console window.
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(command, **popen_kwargs)
    output = []
    samples = []
    process_handle = None
    cpu_primed = False

    def reader():
        if proc.stdout:
            for line in proc.stdout:
                output.append(line.rstrip())
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    started = time.perf_counter()
    while proc.poll() is None:
        time.sleep(0.5)
        sample = gpu_sample()
        if psutil:
            try:
                parent = psutil.Process(proc.pid)
                if process_handle is None:
                    process_handle = parent
                if not cpu_primed:
                    process_handle.cpu_percent(None)
                    cpu_primed = True
                    cpu = 0.0
                else:
                    cpu = process_handle.cpu_percent(None)
                # On Windows, memory_full_info() can block in the psutil
                # wrapper long enough to receive an external console
                # interrupt. RSS is sufficient for this sweep; keep private
                # memory explicitly unavailable rather than destabilizing a
                # long benchmark.
                sample.update({
                    "engine_pid": process_handle.pid,
                    "engine_cpu_pct": cpu,
                    "engine_rss_gib": process_handle.memory_info().rss / (1 << 30),
                    "engine_private_gib": None,
                    "system_ram_used_gib": (
                        psutil.virtual_memory().total -
                        psutil.virtual_memory().available) / (1 << 30),
                })
            except (psutil.Error, OSError):
                pass
        if sample:
            sample["elapsed_s"] = time.perf_counter() - started
            samples.append(sample)
    exit_code = proc.wait()
    thread.join(timeout=5)
    text = "\n".join(output)
    match = re.findall(r"Generation:\s*([0-9.]+)", text)
    offload = re.findall(r"offloaded\s+(\d+)\/(\d+)\s+layers", text, re.I)
    generation = float(match[-1]) if match else None
    result = {
        "label": label, "requested_gpu_layers": layers,
        "requested_fraction": fraction, "tokens": args.tokens,
        "exit_code": exit_code, "wall_s": time.perf_counter() - started,
        "llama_generation_tok_per_s": generation,
        "measured_wall_tok_per_s": args.tokens / (time.perf_counter() - started),
        "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "offloaded_layers_match": offload[-1] if offload else None,
        "samples": samples, "stdout": text,
    }
    return result


def mean_metric(samples, field):
    values = [sample[field] for sample in samples
              if field in sample and sample[field] is not None]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points", default="0,8,16,24,32,auto")
    parser.add_argument("--layer-count", type=int, default=40)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()
    runs = []
    for point in args.points.split(","):
        point = point.strip()
        result = run_one(f"gpu_layers_{point}", point, args)
        def values_for(field):
            return [sample[field] for sample in result["samples"]
                    if field in sample and sample[field] is not None]
        result["resource_summary"] = {
            field: {"mean": mean_metric(result["samples"], field),
                    "max": max(values_for(field), default=None)}
            for field in ("gpu_util_pct", "power_w", "vram_used_mib",
                          "engine_cpu_pct", "engine_rss_gib",
                          "engine_private_gib", "system_ram_used_gib")}
        runs.append(result)
        print(f"{result['label']}: wall={result['wall_s']:.2f}s "
              f"wall_tok/s={result['measured_wall_tok_per_s']:.4f} "
              f"generation_tok/s={result['llama_generation_tok_per_s']}")
    output = {
        "schema": "llama_cpp.pressure_sweep.v1",
        "configuration": {
            "exe": str(args.exe), "model": str(args.model),
            "tokens": args.tokens, "layer_count_assumed": args.layer_count,
            "threads": 10, "context": 16384, "batch": 512,
            "seed": 1, "temperature": 0, "device_for_nonzero": "Vulkan0",
            "pressure_axis": "--gpu-layers; layer-offload proxy, not expert-byte equivalence",
        },
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
