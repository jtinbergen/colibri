"""Direct child probe for COLI_DENSE_GPU; keeps engine diagnostics visible."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(r"C:\Users\jaapj\.lmstudio\models\ornith15_i4_gs64_bf16")
env = dict(os.environ)
env.update({"SNAP": str(MODEL), "SERVE": "1", "COLI_CUDA": "1",
            "COLI_GPUS": "0", "CUDA_EXPERT_GB": "auto",
            "COLI_CUDA_RESIDENT": "1", "COLI_CUDA_DP4A": "1",
            "COLI_CUDA_DP4A_QUIET": "1", "COLI_CUDA_DENSE_GB": "1.0",
            "COLI_DENSE_GPU": "1", "OMP_NUM_THREADS": "10", "PILOT": "1",
            "PIPE": "1", "Q36_MAXT": "16384"})
p = subprocess.Popen([str(ROOT / "qwen36.exe"), "256", "4"], cwd=str(ROOT),
                     env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.STDOUT)
try:
    while True:
        line = p.stdout.readline()
        if not line:
            raise RuntimeError(f"child exited rc={p.poll()}")
        sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
        if b"\x01\x01READY\x01\x01" in line:
            break
    payload = b"Give a short answer about GPU compute islands."
    p.stdin.write(f"SUBMIT probe 0 {len(payload)} 2 0.0 1.0\n".encode())
    p.stdin.write(payload + b"\n"); p.stdin.flush()
    while True:
        line = p.stdout.readline()
        if not line: break
        sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
        if line.startswith(b"DONE "): break
finally:
    if p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       check=False)
    p.wait(timeout=15)
