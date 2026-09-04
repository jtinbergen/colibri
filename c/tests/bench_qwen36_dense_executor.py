"""Matched production-style A/B for the first coarse dense-island executor.

Usage from c/:  python tests/bench_qwen36_dense_executor.py [tokens]
Set COLI_DENSE_GPU=1 for the prototype; unset/0 is the persistent-metadata
baseline. The child is stopped with taskkill, never by writing Ctrl-C to the
console, so a run cannot manufacture a visible ^C.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"C:\Users\jaapj\.lmstudio\models\ornith15_i4_gs64_bf16")
COLI = ROOT / "coli"
TOKENS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
PORT = 8000


def post_chat(prompt, max_tokens):
    body = json.dumps({
        "model": "qwen3.6-colibri",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=max(180, TOKENS * 2)) as r:
        return json.loads(r.read().decode())


def main():
    env = dict(os.environ)
    env.update({
        "COLI_MODEL": str(MODEL),
        "COLI_CUDA": "1",
        "COLI_GPUS": "0",
        "CUDA_EXPERT_GB": "auto",
        # Keep A/B on the same fixed VRAM partition. The baseline leaves this
        # slice unused; the prototype fills it with persistent dense weights.
        "COLI_CUDA_DENSE_GB": "1.0",
        "HEAT_FILE": str(MODEL / ".coli_usage"),
        "OMP_NUM_THREADS": "10",
        "Q36_MAXT": "16384",
        "PILOT": "1",
        "PIPE": "1",
        "COLI_CUDA_RESIDENT": "1",
        "COLI_CUDA_DP4A": "1",
        "COLI_CUDA_DP4A_QUIET": "1",
        "COLI_TIMERS": "0",
        "QTIER_TIMING_DUMP": "0",
    })
    mode = "dense" if env.get("COLI_DENSE_GPU") == "1" else "baseline"
    log_path = ROOT / "tests" / f"dense_executor_{mode}.log"
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        p = subprocess.Popen(
            [sys.executable, str(COLI), "serve", "--cap", "256", "--auto-tier"],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        lines = []
        def read_output():
            for line in p.stdout:
                lines.append(line)
                log.write(line)
                log.flush()
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        ready = False
        deadline = time.time() + 120
        while time.time() < deadline:
            if any("OpenAI-compatible API listening" in line for line in lines):
                ready = True; break
            if p.poll() is not None: break
            time.sleep(0.05)
        try:
            if not ready:
                raise RuntimeError("server did not become ready")
            t0 = time.perf_counter()
            response = post_chat("Give a concise, factual answer about compute islands.", TOKENS)
            wall = time.perf_counter() - t0
            usage = response.get("usage", {})
            timing = response.get("timings", {})
            print(json.dumps({
                "mode": mode, "requested_tokens": TOKENS,
                "completion_tokens": usage.get("completion_tokens"),
                "response_tok_s": timing.get("tokens_per_sec"),
                "wall_s": wall,
                "wall_tok_s": (usage.get("completion_tokens", 0) / wall if wall else 0),
                "log": str(log_path),
            }, sort_keys=True))
        finally:
            if p.poll() is None:
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False)
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            for line in lines:
                if any(tag in line for tag in ("qtier-dense", "dense DeltaNet",
                                                "groups calls", "VRAM hit rate",
                                                "GPU utilization", "power")):
                    print(line.rstrip())
            reader.join(timeout=2)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
