"""Direct Qwen engine A/B for the coarse dense-island prototype.

This uses qwen36's own binary serve protocol, avoiding the Python gateway's
diagnostic-stream parser during the CUDA experiment. It performs an 8-token
warmup followed by a timed run and samples nvidia-smi every 500 ms.
"""
import os
import hashlib
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"C:\Users\jaapj\.lmstudio\models\ornith15_i4_gs64_bf16")
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
WARMUP = int(os.environ.get("BENCH_WARMUP", "8"))


def kill_tree(p):
    if p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait(timeout=5)


def read_request(p, request_id):
    done = None
    data = bytearray()
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"engine exited while reading request rc={p.poll()}")
        if line.startswith(b"DATA "):
            parts = line.split()
            n = int(parts[2])
            data.extend(p.stdout.read(n))
            p.stdout.read(1)  # protocol newline after the data payload
        elif line.startswith((b"DONE ", b"ERROR ")):
            done = line.decode("utf-8", errors="replace").rstrip()
            if line.startswith(b"ERROR "):
                raise RuntimeError(done)
            if request_id.encode() in line:
                return done, bytes(data)
        elif (b"[timers]" in line or b"dense-shared-check" in line or
              b"dense-attn-check" in line or b"qtier-dense-check" in line or
              b"cpu-batch" in line):
            print(line.decode("utf-8", errors="replace").rstrip(), flush=True)
        elif (b"[qtier-exec]" in line or b"qtier-stage-debug" in line or
              b"VRAM hit rate" in line or b"[qtier] resident " in line or
              b"[islands-v0]" in line):
            print(line.decode("utf-8", errors="replace").rstrip(), flush=True)


def submit(p, request_id, prompt, tokens):
    payload = prompt.encode()
    p.stdin.write(f"SUBMIT {request_id} 0 {len(payload)} {tokens} 0.0 1.0\n".encode())
    p.stdin.write(payload + b"\n")
    p.stdin.flush()
    return read_request(p, request_id)


def main():
    env = dict(os.environ)
    dense_requested = env.get("COLI_DENSE_GPU") == "1"
    resident_enabled = env.get("DENSE_EXEC_RESIDENT", "1") != "0"
    dense_reserve = env.get("COLI_CUDA_DENSE_GB", "1.0")
    expert_budget = env.get("CUDA_EXPERT_GB", "auto")
    cpu_batch = env.get("COLI_CPU_EXPERT_BATCH", "default")
    omp_threads = env.get("OMP_NUM_THREADS", "10")
    decode_threads = env.get("COLI_QWEN_DECODE_THREADS", "auto")
    timer_mode = env.get("COLI_TIMERS", "0")
    heat_file = env.get("HEAT_FILE", "")
    lfru_mode = env.get("QTIER_LFRU", "0")
    placement_mode = env.get("QTIER_PLACEMENT", "default")
    dense_attn = env.get("COLI_DENSE_ATTN", "0")
    dense_shared = env.get("COLI_DENSE_SHARED", "0")
    env.update({"SNAP": str(MODEL), "SERVE": "1", "COLI_CUDA": "1",
                "COLI_GPUS": "0", "CUDA_EXPERT_GB": expert_budget,
                # A/B must not inherit heat/LFRU state from the previous
                # process; both sides therefore use the same natural placement.
                "HEAT_FILE": heat_file, "QTIER_LFRU": lfru_mode,
                "COLI_CUDA_DENSE_GB": dense_reserve,
                "COLI_CUDA_RESIDENT": "1" if resident_enabled else "0",
                "COLI_CUDA_DP4A": "1",
                "COLI_CUDA_DP4A_QUIET": "1", "COLI_TIMERS": timer_mode,
                "QTIER_TIMING_DUMP": "0", "OMP_NUM_THREADS": "10",
                # Freeze host prefetch for this correctness/execution A/B.
                "PILOT": "0", "PIPE": "1", "Q36_MAXT": "16384"})
    env["OMP_NUM_THREADS"] = omp_threads
    if dense_requested: env["COLI_DENSE_GPU"] = "1"
    else: env.pop("COLI_DENSE_GPU", None)
    mode = ("dense" if dense_requested else "baseline") + f"+cpu-batch={cpu_batch}"
    p = subprocess.Popen([str(ROOT / "qwen36.exe"), "256", "4"], cwd=str(ROOT),
                         env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT)
    gpu_samples = []
    host_samples = []
    smi = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,memory.used",
         "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    proc = psutil.Process(p.pid)
    proc.cpu_percent(None)
    def sample_reader():
        for line in smi.stdout:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                try:
                    try:
                        cpu_pct = proc.cpu_percent(None)
                        rss_gb = proc.memory_info().rss / (1024**3)
                        ram = psutil.virtual_memory()
                        host_samples.append({"cpu_pct": cpu_pct,
                                             "rss_gb": rss_gb,
                                             "ram_used_gb": ram.used / (1024**3),
                                             "ram_total_gb": ram.total / (1024**3)})
                    except (psutil.Error, OSError):
                        pass
                    gpu_samples.append({"gpu_pct": float(parts[0]),
                                        "power_w": float(parts[1]),
                                        "vram_mib": float(parts[2])})
                except ValueError: pass
    st = threading.Thread(target=sample_reader, daemon=True); st.start()
    diagnostics = []
    try:
        while True:
            line = p.stdout.readline()
            if not line: raise RuntimeError(f"engine exited before READY rc={p.poll()}")
            diagnostics.append(line.decode("utf-8", errors="replace").rstrip())
            if b"\x01\x01READY\x01\x01" in line: break
        prompt = os.environ.get("BENCH_PROMPT",
                               "Give a concise, factual answer about compute islands.")
        if WARMUP > 0:
            submit(p, "warm", prompt, WARMUP)
        t0 = time.perf_counter()
        done, output = submit(p, "timed", prompt, TOKENS)
        wall = time.perf_counter() - t0
        output_file = env.get("BENCH_OUTPUT_FILE")
        if output_file:
            Path(output_file).write_bytes(output)
        m = re.search(r"DONE\s+timed\s+STAT\s+(\d+)\s+([0-9.]+)", done)
        gen = int(m.group(1)) if m else 0
        tps = float(m.group(2)) if m else 0.0
        print({"mode": mode, "cpu_expert_batch": cpu_batch,
               "omp_threads": omp_threads,
               "decode_threads": decode_threads,
               "heat_file": heat_file, "heat_readonly": env.get("QTIER_HEAT_READONLY", "0"),
               "placement": placement_mode, "lfru": lfru_mode,
               "dense_attn": dense_attn, "dense_shared": dense_shared,
               "cuda_expert_gb": expert_budget, "dense_reserve_gb": dense_reserve,
               "warmup_tokens": WARMUP, "tokens": TOKENS, "generated": gen,
               "engine_decode_tok_s": tps, "request_wall_s": wall,
               "request_wall_tok_s": gen / wall if wall else 0.0,
               "output_bytes": len(output),
               "output_sha256": hashlib.sha256(output).hexdigest(),
               "gpu_avg_pct": sum(x["gpu_pct"] for x in gpu_samples) / len(gpu_samples) if gpu_samples else None,
               "gpu_peak_pct": max((x["gpu_pct"] for x in gpu_samples), default=None),
               "power_avg_w": sum(x["power_w"] for x in gpu_samples) / len(gpu_samples) if gpu_samples else None,
               "power_peak_w": max((x["power_w"] for x in gpu_samples), default=None),
               "vram_avg_mib": sum(x["vram_mib"] for x in gpu_samples) / len(gpu_samples) if gpu_samples else None,
               "vram_peak_mib": max((x["vram_mib"] for x in gpu_samples), default=None),
               "cpu_avg_pct": sum(x["cpu_pct"] for x in host_samples) / len(host_samples) if host_samples else None,
               "cpu_peak_pct": max((x["cpu_pct"] for x in host_samples), default=None),
               "rss_avg_gb": sum(x["rss_gb"] for x in host_samples) / len(host_samples) if host_samples else None,
               "rss_peak_gb": max((x["rss_gb"] for x in host_samples), default=None),
               "ram_used_avg_gb": sum(x["ram_used_gb"] for x in host_samples) / len(host_samples) if host_samples else None,
               "ram_total_gb": max((x["ram_total_gb"] for x in host_samples), default=None),
               "diagnostics": [x for x in diagnostics if "dense" in x.lower() or "budget" in x.lower()]})
    except Exception:
        print({"mode": mode, "cpu_expert_batch": cpu_batch,
               "omp_threads": omp_threads,
               "decode_threads": decode_threads,
               "heat_file": heat_file, "heat_readonly": env.get("QTIER_HEAT_READONLY", "0"),
               "placement": placement_mode, "lfru": lfru_mode,
               "dense_attn": dense_attn, "dense_shared": dense_shared,
               "cuda_expert_gb": expert_budget, "dense_reserve_gb": dense_reserve,
               "warmup_tokens": WARMUP, "error": "engine request failed",
               "diagnostics": diagnostics[-40:]}, flush=True)
        raise
    finally:
        kill_tree(p)
        kill_tree(smi)
        st.join(timeout=2)


if __name__ == "__main__":
    main()
