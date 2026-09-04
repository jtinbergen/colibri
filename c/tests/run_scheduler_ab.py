#!/usr/bin/env python3
"""Matched runtime A/B for an offline scheduler heat snapshot.

The only independent variable is HEAT_FILE.  Both runs use the same model,
binary, prompt, warmup, token count, resident DP4A path, and frozen LFRU
placement.  No Colibri timers, timing dump, or CUPTI variables are enabled.
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - the target box has psutil
    psutil = None


def post(port, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return time.perf_counter() - started, json.loads(raw), raw


def gpu_sample():
    query = ("utilization.gpu,utilization.memory,memory.used,memory.free,"
             "power.draw,temperature.gpu,pcie.link.gen.current,"
             "pcie.link.width.current")
    try:
        run_kwargs = {
            "capture_output": True, "text": True, "timeout": 5, "check": True,
        }
        if os.name == "nt":
            # nvidia-smi is sampled frequently; without this flag Windows
            # briefly creates a visible console window for every sample.
            run_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            **run_kwargs)
        fields = [field.strip() for field in result.stdout.strip().split(",")]
        names = ["gpu_util_pct", "memory_util_pct", "vram_used_mib",
                 "vram_free_mib", "power_w", "temperature_c", "pcie_gen",
                 "pcie_width"]
        sample = {name: float(fields[index]) if index < 6 else int(float(fields[index]))
                  for index, name in enumerate(names) if index < len(fields)}
        return sample
    except (OSError, subprocess.SubprocessError, ValueError):
        return {}


def run_one(label, heat_file, args):
    env = dict(os.environ)
    env.update({
        "COLI_MODEL": str(args.model), "COLI_CUDA": "1", "COLI_GPUS": "0",
        "CUDA_EXPERT_GB": (str(args.expert_gb) if args.expert_gb else "auto"),
        "HEAT_FILE": str(heat_file),
        "QTIER_HEAT_READONLY": "1", "QTIER_LFRU": "0",
        "COLI_CUDA_RESIDENT": "1", "COLI_CUDA_DP4A": "1",
        "COLI_CUDA_DP4A_QUIET": "1", "OMP_NUM_THREADS": "10",
        "Q36_MAXT": "16384", "PILOT": "1", "PIPE": "1",
    })
    # Explicitly remove diagnostic knobs inherited from the calling shell.
    for name in ("COLI_TIMERS", "COLI_ISLAND_TIMING", "QTIER_TIMING_DUMP",
                 "COLI_CUPTI", "COLI_CUDA_CUPTI", "COLI_CUDA_GRAPH"):
        env.pop(name, None)
    popen_kwargs = {
        "cwd": str(args.engine.parent), "env": env,
        "stdout": subprocess.PIPE, "stderr": subprocess.STDOUT,
        "text": True, "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        ["python", str(args.engine), "serve", "--cap", "256", "--auto-tier"],
        **popen_kwargs)
    lines = []

    def reader():
        if proc.stdout:
            for line in proc.stdout:
                lines.append(line.rstrip())

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    samples = []
    cpu_baselines = {}
    process_handles = {}
    started = time.time()
    try:
        while time.time() - started < args.start_timeout:
            if any("OpenAI-compatible API listening" in line for line in lines):
                break
            if proc.poll() is not None:
                raise RuntimeError(f"{label}: server exited with {proc.returncode}")
            time.sleep(0.5)
        else:
            raise RuntimeError(f"{label}: server did not become ready")

        warmup_elapsed, warmup, _ = post(args.port, {
            "model": "qwen3.6-colibri",
            "messages": [{"role": "user", "content": args.warmup_prompt}],
            "max_tokens": args.warmup_tokens, "temperature": 0,
        }, args.request_timeout)
        timed_payload = {
            "model": "qwen3.6-colibri",
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.tokens, "temperature": 0,
        }
        request_result = {}
        done = threading.Event()

        def request_thread():
            try:
                elapsed, response, raw = post(args.port, timed_payload,
                                              args.request_timeout)
                request_result.update(elapsed=elapsed, response=response,
                                      raw=raw)
            except Exception as error:  # surfaced below
                request_result["error"] = repr(error)
            finally:
                done.set()

        worker = threading.Thread(target=request_thread, daemon=True)
        worker.start()
        while not done.wait(1.0):
            sample = gpu_sample()
            if psutil:
                try:
                    parent = psutil.Process(proc.pid)
                    candidates = [parent] + parent.children(recursive=True)
                    # `coli serve` is a Python parent; qwen36 is normally a
                    # child.  Follow the largest resident child so CPU/RSS
                    # describe the actual inference process, not the server.
                    process = max(candidates,
                                  key=lambda item: item.memory_info().rss)
                    if process.pid not in cpu_baselines:
                        process_handles[process.pid] = process
                        process.cpu_percent(None)
                        cpu_baselines[process.pid] = True
                        cpu_pct = 0.0
                    else:
                        cpu_pct = process_handles[process.pid].cpu_percent(None)
                    # RSS is stable and sufficient for the sweep.  On this
                    # Windows host memory_full_info() can block and trigger
                    # a console KeyboardInterrupt in long detached runs.
                    sample.update({
                        "engine_pid": process.pid,
                        "engine_cpu_pct": cpu_pct,
                        "engine_rss_gib": process.memory_info().rss / (1 << 30),
                        "engine_private_gib": None,
                        "system_ram_used_gib": (
                            psutil.virtual_memory().total -
                            psutil.virtual_memory().available) / (1 << 30),
                    })
                except (psutil.Error, OSError):
                    pass
            if sample:
                sample["elapsed_s"] = time.time() - started
                samples.append(sample)
        worker.join()
        if "error" in request_result:
            raise RuntimeError(f"{label}: request failed: {request_result['error']}")
        response = request_result["response"]
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", args.tokens))
        elapsed = request_result["elapsed"]
        result = {
            "label": label, "heat_file": str(heat_file),
            "warmup_s": warmup_elapsed,
            "timed_s": elapsed,
            "completion_tokens": completion_tokens,
            "tok_per_s": completion_tokens / elapsed if elapsed else 0.0,
            "ms_per_token": 1000.0 * elapsed / completion_tokens if completion_tokens else 0.0,
            "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "output_chars": len(content), "output_text": content,
            "samples": samples,
            "engine_lines": lines,
            "diagnostics_disabled": True,
        }
        return result
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def aggregate(samples):
    if not samples:
        return {}
    fields = ("gpu_util_pct", "memory_util_pct", "vram_used_mib", "power_w",
              "temperature_c", "engine_cpu_pct", "engine_rss_gib",
              "engine_private_gib", "system_ram_used_gib")
    output = {}
    for field in fields:
        values = [sample[field] for sample in samples
                  if field in sample and sample[field] is not None]
        if values:
            output[field] = {"mean": sum(values) / len(values),
                             "min": min(values), "max": max(values)}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-heat", type=Path, required=True)
    parser.add_argument("--candidate-heat", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--engine", type=Path, default=Path("coli"))
    parser.add_argument("--model", type=Path,
                        default=Path(r"C:\Users\jaapj\.lmstudio\models\ornith15_i4_gs64_bf16"))
    parser.add_argument("--prompt", default=(
        "Explain, in a technically precise but readable way, how a modern mixture-of-experts "
        "language model performs inference. Cover tokenization, attention, router top-k selection, "
        "expert matrix multiplication, quantization, hot expert residency, CPU and GPU memory "
        "movement, PCIe transfers, and why throughput can be limited by memory bandwidth or scheduling. "
        "Use headings and concrete examples. Compare a single Pascal GPU island with a CPU-only fallback "
        "and identify the most important measurements for diagnosing performance."))
    parser.add_argument("--warmup-prompt", default="Give a brief technical warmup about expert routing.")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--warmup-tokens", type=int, default=32)
    parser.add_argument("--expert-gb", type=float, default=0.0,
                        help="explicit Colibri expert budget in GiB")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--start-timeout", type=int, default=120)
    parser.add_argument("--request-timeout", type=int, default=900)
    args = parser.parse_args()
    results = [run_one("current", args.current_heat, args),
               run_one("makespan_balanced", args.candidate_heat, args)]
    for result in results:
        result["metric_summary"] = aggregate(result["samples"])
    output = {"schema": "colibri.compute_islands.scheduler_ab.v1",
              "configuration": {"tokens": args.tokens, "warmup_tokens": args.warmup_tokens,
                                "engine": str(args.engine), "model": str(args.model),
                                "timers": False, "cupti": False, "lfru_frozen": True},
              "runs": results}
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"{result['label']}: {result['tok_per_s']:.4f} tok/s, "
              f"{result['ms_per_token']:.3f} ms/token, warmup {result['warmup_s']:.2f}s, "
              f"samples={len(result['samples'])}, sha256={result['output_sha256']}")
        print("  metrics:", json.dumps(result["metric_summary"], sort_keys=True))


if __name__ == "__main__":
    main()
