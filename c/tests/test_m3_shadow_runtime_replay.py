#!/usr/bin/env python3
"""Compare the same tiny M3 run with the opt-in shadow observer off and on.

The observer is allowed to emit diagnostics on stderr and to write its own
decision log, but it must not alter semantic stdout, exit status, or the
normalized engine lifecycle trace.  This is a behavioral smoke test for the
real ``expert_load()`` integration seam; it is not a planner-quality or
hardware-performance gate.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


_SHADOW_ENV = (
    "COLI_M3_SHADOW_CONFIG",
    "COLI_M3_SHADOW_TOPOLOGY_HASH",
    "COLI_M3_SHADOW_REPLICA_DRIVES",
    "COLI_M3_SHADOW_MODEL_ID",
    "COLI_M3_SHADOW_LOG",
    "COLI_M3_SHADOW_TRACE",
)


def run_engine(binary: str, cwd: Path, snap: str, ref: str,
               shadow: dict[str, str] | None,
               synthetic_need_delta_ns: str,
               event_trace: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, SNAP=snap, REF=ref, TF="1", TF_DECODE="1", IDOT="0")
    # The live counterfactual assertion requires both observed loads to be
    # concurrently in flight.  Pin the tiny replay to the bounded parallel
    # contract so host scheduling cannot turn this into a serial, vacuous
    # occupancy check.  The same settings are used for shadow-off and -on.
    env["OMP_NUM_THREADS"] = "16"
    env["COLI_M3_DAG_SERIAL"] = "1"
    env["COLI_M3_DAG_PARALLEL"] = "1"
    env["COLI_M3_DAG_PIPE"] = "1"
    env["COLI_M3_DAG_WORKERS"] = "8"
    for name in _SHADOW_ENV:
        env.pop(name, None)
    env["COLI_TRACE"] = event_trace
    if shadow:
        env.update(shadow)
        env["COLI_M3_SHADOW_SYNTHETIC_NEED_DELTA_NS"] = synthetic_need_delta_ns
    return subprocess.run(
        [binary, "8"], cwd=cwd, env=env, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, timeout=600,
    )


def semantic_stdout(text: str) -> list[str]:
    """Remove intentionally variable performance/timing fields."""
    stable: list[str] = []
    for line in text.splitlines():
        if line.startswith(("PROFILE:", "ATTENTION:")):
            continue
        line = re.sub(r"\| [0-9]+(?:\.[0-9]+)? pos/s", "| RATE pos/s", line)
        line = re.sub(r"\+?[0-9]+(?:\.[0-9]+)?s", "+TIME", line)
        stable.append(line)
    return stable


def canonical_engine_trace(path: Path) -> list[tuple[str, ...]]:
    """Drop wall-clock/thread identity, retain semantic lifecycle facts."""
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"cannot read COLI_TRACE {path}: {exc}") from exc
    rows: list[tuple[str, ...]] = []
    for line in lines:
        if not line or line.startswith("#") or line.startswith("ns,"):
            continue
        fields = line.split(",")
        if len(fields) != 8:
            raise RuntimeError(f"malformed COLI_TRACE row: {line!r}")
        # ns and thread are intentionally excluded: neither is a routing or
        # placement decision, and both vary between identical runs.
        rows.append((fields[1], fields[3], fields[4], fields[5],
                     fields[6], fields[7]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="./colibri")
    ap.add_argument("--snap", default="m3tiny_i8")
    ap.add_argument("--ref", default="ref_m3.json")
    ap.add_argument("--config", required=True)
    ap.add_argument("--topology-hash", default="99")
    ap.add_argument("--replica-drives", default="0:101,1:102")
    ap.add_argument("--model-id", default="7")
    ap.add_argument("--expected-counterfactual-drive", type=int)
    ap.add_argument("--expected-actual-drive", type=int)
    ap.add_argument("--synthetic-need-delta-ns", default="1000000000")
    args = ap.parse_args()

    binary = str(Path(args.binary).resolve())
    cwd = Path(binary).parent
    config = str(Path(args.config).resolve())
    with tempfile.TemporaryDirectory(prefix="coli-m3-shadow-") as temp:
        log_path = str(Path(temp) / "shadow.log")
        trace_path = str(Path(temp) / "shadow-trace.csv")
        off_event_trace = Path(temp) / "engine-off.trace.csv"
        on_event_trace = Path(temp) / "engine-on.trace.csv"
        shadow = {
            "COLI_M3_SHADOW_CONFIG": config,
            "COLI_M3_SHADOW_TOPOLOGY_HASH": args.topology_hash,
            "COLI_M3_SHADOW_REPLICA_DRIVES": args.replica_drives,
            "COLI_M3_SHADOW_MODEL_ID": args.model_id,
            "COLI_M3_SHADOW_LOG": log_path,
            "COLI_M3_SHADOW_TRACE": trace_path,
        }
        off = run_engine(binary, cwd, args.snap, args.ref, None,
                         args.synthetic_need_delta_ns, str(off_event_trace))
        on = run_engine(binary, cwd, args.snap, args.ref, shadow,
                        args.synthetic_need_delta_ns, str(on_event_trace))
        sys.stdout.write(off.stdout)
        sys.stderr.write(off.stderr)
        sys.stdout.write(on.stdout)
        sys.stderr.write(on.stderr)

        failures: list[str] = []
        if off.returncode != 0:
            failures.append(f"shadow-off exit {off.returncode}")
        if on.returncode != 0:
            failures.append(f"shadow-on exit {on.returncode}")
        if semantic_stdout(off.stdout) != semantic_stdout(on.stdout):
            failures.append("shadow-on changed semantic engine stdout")
        try:
            off_trace = canonical_engine_trace(off_event_trace)
            on_trace = canonical_engine_trace(on_event_trace)
            if not off_trace or not on_trace:
                failures.append("COLI_TRACE produced no semantic rows")
            elif off_trace != on_trace:
                failures.append("shadow-on changed normalized COLI_TRACE events")
        except RuntimeError as exc:
            failures.append(str(exc))
        log = Path(log_path)
        expected_decision_ids: set[str] = set()
        if not log.is_file() or not log.read_text(encoding="ascii").strip():
            failures.append("shadow-on produced no decision log")
        elif args.expected_counterfactual_drive is not None:
            decision_rows = [
                row.split(",") for row in log.read_text(encoding="ascii").splitlines()
                if row
            ]
            matching = [
                row for row in decision_rows
                if len(row) >= 12
                and row[8] == str(args.expected_counterfactual_drive)
                and row[10] == "1" and row[11] == "2"
            ]
            if not matching:
                failures.append(
                    "decision log lacks expected occupied-to-alternate row "
                    f"for drive {args.expected_counterfactual_drive}")
            else:
                expected_decision_ids = {row[0] for row in matching}
        trace = Path(trace_path)
        if not trace.is_file():
            failures.append("shadow-on produced no runtime trace")
        else:
            trace_text = trace.read_text(encoding="ascii")
            if ("consumer_need_ns,predicted_ready_ns,predicted_uncertainty_ns"
                    not in trace_text
                    or "component_mask" not in trace_text
                    or ",63," not in trace_text
                    or "actual_path" not in trace_text
                    or trace_text.count("\n") < 2):
                failures.append("shadow-on runtime trace lacks correlated rows")
            if args.expected_counterfactual_drive is not None:
                try:
                    rows = list(csv.DictReader(trace_text.splitlines()))
                    chosen = {
                        int(row["predicted_resource_id"])
                        for row in rows
                        if row.get("predicted_resource_id")
                    }
                except (ValueError, KeyError) as exc:
                    failures.append(f"malformed runtime decision trace: {exc}")
                else:
                    if args.expected_counterfactual_drive not in chosen:
                        failures.append(
                            "runtime replay did not choose expected counterfactual "
                            f"drive {args.expected_counterfactual_drive}: {sorted(chosen)}")
                    if args.expected_actual_drive is not None:
                        matching = [
                            row for row in rows
                            if int(row["predicted_resource_id"])
                            == args.expected_counterfactual_drive
                            and int(row["actual_drive_id"])
                            == args.expected_actual_drive
                            and int(row["component_mask"]) == 63
                            and row["request_id"] in expected_decision_ids
                            and row["actual_path"]
                        ]
                        if not matching:
                            failures.append(
                                "counterfactual row lacks complete actual path "
                                f"on drive {args.expected_actual_drive}")
        if failures:
            print(f"m3-shadow-runtime-replay: FAIL ({'; '.join(failures)})")
            return 1
        print("m3-shadow-runtime-replay: OK (stdout/exit/COLI_TRACE unchanged; live log and runtime trace non-empty)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
