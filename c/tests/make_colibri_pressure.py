#!/usr/bin/env python3
"""Create fixed-capacity Colibri QTH1 pressure snapshots.

The runtime budget is supplied separately as CUDA_EXPERT_GB.  Each snapshot
selects the hottest members of the current 3,642-slot placement, so the sweep
tests capacity pressure without changing the placement ranking or scheduler.
"""

import argparse
import csv
import json
import struct
from pathlib import Path


def num(row, key):
    try:
        return int(float(row.get(key, 0) or 0))
    except (TypeError, ValueError):
        return 0


def read(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_heat(path, selected, layers, experts):
    values = [2 if (layer, expert) in selected else 0
              for layer in range(layers) for expert in range(experts)]
    with Path(path).open("wb") as handle:
        handle.write(struct.pack("<III", 0x51544831, layers, experts))
        handle.write(struct.pack("<" + "I" * len(values), *values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-csv", required=True)
    parser.add_argument("--placement-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--full-capacity", type=int, default=3642)
    parser.add_argument("--fractions", default="1,.75,.5,.375,.25,.125")
    args = parser.parse_args()
    expert_rows = read(args.expert_csv)
    placement_rows = read(args.placement_csv)
    routes = {(num(row, "layer"), num(row, "expert")): num(row, "invocations")
              for row in expert_rows}
    current = {(num(row, "layer"), num(row, "expert"))
               for row in placement_rows if num(row, "gpu_resident")}
    ranked = sorted(current, key=lambda key: (-routes.get(key, 0), key))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for fraction_text in args.fractions.split(","):
        fraction = float(fraction_text)
        capacity = max(1, min(args.full_capacity,
                              int(round(args.full_capacity * fraction))))
        selected = set(ranked[:capacity])
        label = f"{fraction:g}".replace(".", "p")
        heat = out_dir / f"capacity_{label}.heat"
        write_heat(heat, selected, args.layers, args.experts)
        routed_total = sum(routes.values())
        routed_hits = sum(routes.get(key, 0) for key in selected)
        manifest.append({
            "label": label, "target_fraction": fraction,
            "target_slots": capacity, "heat_file": str(heat),
            "selected_slots": len(selected),
            "observed_routed_invocations": routed_total,
            "observed_routed_hits": routed_hits,
            "observed_routed_hit_rate": routed_hits / routed_total if routed_total else 0.0,
            "selection": "hottest experts within current placement by invocations",
        })
    result = {
        "schema": "colibri.pressure_heat_manifest.v1",
        "expert_csv": str(Path(args.expert_csv)),
        "placement_csv": str(Path(args.placement_csv)),
        "full_capacity_slots": args.full_capacity,
        "expert_budget_gib_per_slot": 6.0 / args.full_capacity,
        "current_slots": len(current), "points": manifest,
        "caveat": "Capacity pressure is explicit CUDA_EXPERT_GB; snapshots preserve current ranking but select a hot prefix at each reduced capacity.",
    }
    (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
