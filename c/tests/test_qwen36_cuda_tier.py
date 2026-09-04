"""End-to-end test for the qwen36 CUDA tier on the local Windows box.

Spawns `coli serve --cap 256 --auto-tier` with the CUDA tier env vars,
parses stderr for engine markers, and hits the OpenAI-compatible API.

T1: warmstart line says `all 10240 experts in RAM` WITHOUT the
    `int8 only for non-residents` qualifier. Today the line reads
    `all 10240 experts in RAM (int8 only for non-residents), 3642 in VRAM`
    — that suffix is the ~20 GB of redundant int8 copies the patch removes.
T2: stderr contains `[qtier] CUDA VRAM expert tier active`.
T3: stderr contains `OpenAI-compatible API listening`.
T4: HTTP 200 on a chat completion; response decodes as JSON with a
    non-empty `choices[0].message.content`.
T5: warmstart elapsed < 30 s (today: 42.9 s).
T6: `RSS after load: < 12 GB` (today: 9.23 GB — actually OK; this is
    here to catch any regression that bloats the dense load).

Tests run only on this machine (no CI). Skipped if the engine binary is
missing or the model directory isn't present.
"""
import os
import sys
import re
import time
import shutil
import threading
import subprocess
from pathlib import Path

try:
    import requests  # noqa: F401  -- only used if Python's stdlib can't connect
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

import urllib.request
import urllib.error
import json as jsonlib

ENGINE_DIR = Path(r"D:\src\colibri\c")
COLI = ENGINE_DIR / "coli"
MODEL_DIR = Path(r"C:\Users\jaapj\.lmstudio\models\ornith15_i4_gs64_bf16")
HEAT_FILE = MODEL_DIR / ".coli_usage"
COLI_CUDA_DLL = ENGINE_DIR / "coli_cuda.dll"
Qwen36_EXE = ENGINE_DIR / "qwen36.exe"

LAUNCH_ENV_OVERRIDES = {
    "COLI_MODEL": str(MODEL_DIR),
    "COLI_CUDA": "1",
    "COLI_GPUS": "0",
    "CUDA_EXPERT_GB": "auto",
    "HEAT_FILE": str(HEAT_FILE),
    "OMP_NUM_THREADS": "10",
    "Q36_MAXT": "16384",
    "PILOT": "1",
    "PIPE": "1",
    "PATH": r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin;"
            + os.environ.get("PATH", ""),
}


def _ready():
    for p in (COLI, Qwen36_EXE, COLI_CUDA_DLL, MODEL_DIR, HEAT_FILE):
        if not p.exists():
            return False, f"missing: {p}"
    return True, None


def _post_chat(port, prompt, max_tokens=8, temperature=0.0):
    body = jsonlib.dumps({
        "model": "qwen3.6-colibri",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None, repr(e)


def _health(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status, r.read().decode("utf-8")
    except Exception as e:
        return None, repr(e)


class TestFail(Exception):
    pass


def run():
    ready, why = _ready()
    if not ready:
        print(f"SKIP: {why}", file=sys.stderr)
        return 0

    env = dict(os.environ)
    env.update(LAUNCH_ENV_OVERRIDES)

    proc = subprocess.Popen(
        ["python", str(COLI), "serve", "--cap", "256", "--auto-tier"],
        cwd=str(ENGINE_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []

    def reader():
        try:
            for line in proc.stdout:
                output_lines.append(line)
        except Exception:
            pass
    t = threading.Thread(target=reader, daemon=True)
    t.start()

    server_up = False
    deadline = time.time() + 90
    port = 8000
    failures = []
    rss_gb = None
    warmstart_secs = None

    try:
        while time.time() < deadline:
            if any("OpenAI-compatible API listening" in l for l in output_lines):
                server_up = True
                break
            time.sleep(1)

        if not server_up:
            proc.kill()
            failures.append("server never came up within 90s")

        if server_up:
            status, body = _health(port)
            if status != 200:
                failures.append(f"/health returned {status} body={body[:200]}")
            else:
                cs, cb = _post_chat(port, "hi", max_tokens=8)
                if cs != 200:
                    failures.append(f"chat completion returned {cs} body={cb[:300]}")
                else:
                    try:
                        j = jsonlib.loads(cb)
                        content = j["choices"][0]["message"]["content"]
                        if not content.strip():
                            failures.append("chat returned empty content")
                        else:
                            print(f"OK  chat: {content!r}", file=sys.stderr)
                    except Exception as e:
                        failures.append(f"bad chat response: {e} body={cb[:300]}")

        # When the caller explicitly requests both optimizations, require
        # evidence from the backend that the resident path actually selected
        # DP4A; resident grouped-W4 alone is not this test's target.
        if (os.environ.get("COLI_CUDA_RESIDENT") == "1" and
                os.environ.get("COLI_CUDA_DP4A") == "1"):
            resident_dp4a = [l.strip() for l in output_lines
                             if ("[dp4a] resident dispatch:" in l or
                                 "[cuda-graph] resident graph dispatch:" in l)]
            quiet_dp4a = os.environ.get("COLI_CUDA_DP4A_QUIET") == "1"
            if not resident_dp4a and not quiet_dp4a:
                failures.append("resident DP4A requested but no resident dispatch marker was emitted")
            elif resident_dp4a:
                print(f"OK  resident DP4A: {resident_dp4a[-1].strip()}", file=sys.stderr)
            else:
                print("OK  resident DP4A: dispatch marker suppressed by COLI_CUDA_DP4A_QUIET=1", file=sys.stderr)

        # T1: warmstart line must NOT contain the "int8 only for non-residents" qualifier
        warmstart_line = None
        for line in output_lines:
            if "warmstart (parallel)" in line:
                warmstart_line = line.strip()
                break
        if warmstart_line is None:
            failures.append("T1: no 'warmstart (parallel)' line in stderr")
        elif "int8 only for non-residents" in warmstart_line:
            failures.append(
                f"T1: warmstart line still says '(int8 only for non-residents)' "
                f"-- redundant int8 copies kept: {warmstart_line}"
            )
        else:
            print(f"OK  T1: warmstart no fallback int8: {warmstart_line}", file=sys.stderr)

        # T2: tier-active marker
        if not any("[qtier] CUDA VRAM expert tier active" in l for l in output_lines):
            failures.append("T2: stderr missing '[qtier] CUDA VRAM expert tier active'")
        else:
            print("OK  T2: CUDA VRAM expert tier active", file=sys.stderr)

        # T5: warmstart time < 30 s
        if warmstart_line:
            m = re.search(r"-- ([0-9.]+) s", warmstart_line)
            if m:
                warmstart_secs = float(m.group(1))
                if warmstart_secs >= 30.0:
                    failures.append(f"T5: warmstart took {warmstart_secs}s (>= 30 s budget)")
                else:
                    print(f"OK  T5: warmstart {warmstart_secs}s", file=sys.stderr)

        # T6: RSS after load budget (pre-warmstart; after-load RSS shouldn't change much with the patch)
        for line in output_lines:
            m = re.search(r"RSS after load: ([0-9.]+) GB", line)
            if m:
                rss_gb = float(m.group(1))
                break
        if rss_gb is None:
            failures.append("T6: 'RSS after load: X GB' line not found in stderr")
        elif rss_gb >= 12.0:
            failures.append(f"T6: RSS after load = {rss_gb} GB (>= 12 GB budget)")
        else:
            print(f"OK  T6: RSS after load {rss_gb} GB", file=sys.stderr)

    finally:
        try:
            if proc.poll() is None:
                if sys.platform == "win32":
                    # CTRL_BREAK_EVENT is visible as "^C" in an attached
                    # console and is easy to mistake for the agent stopping
                    # itself.  This test owns the exact launcher process, so
                    # terminate that process tree directly and keep the
                    # console quiet.
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                else:
                    proc.terminate()
                proc.wait(timeout=10)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    stderr_dump = "".join(output_lines)
    if failures:
        print("=== FAIL ===", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        print("--- stderr (last 60 lines) ---", file=sys.stderr)
        for line in stderr_dump.splitlines()[-60:]:
            print(f"  {line}", file=sys.stderr)
        return 1
    print(f"=== PASS (RSS={rss_gb} GB, warmstart={warmstart_secs}s) ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(run())
