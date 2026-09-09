#!/usr/bin/env python3
"""Build an evidence-only M3 PIN candidate from .coli_usage history.

The runtime still owns placement and admission.  This tool only creates a
bounded candidate file for an explicit benchmark.  Full-layer candidates add
zero-count experts with count 1 so the candidate really represents the whole
layer instead of only the experts seen in history.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def read_history(path: Path):
    headers: list[str] = []
    counts: dict[tuple[int, int], int] = {}
    n_layers = n_experts = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) != 3:
            continue
        try:
            left, middle, right = (int(item) for item in parts)
        except ValueError:
            continue
        if left == -1:
            n_layers, n_experts = middle, right
            headers.append(raw)
        elif left == -2:
            headers.append(raw)
        elif left >= 0:
            counts[(left, middle)] = right
    if n_layers is None or n_experts is None:
        raise ValueError("history is missing the -1 dimensions header")
    if not headers:
        raise ValueError("history is missing required headers")
    return headers, counts, n_layers, n_experts


def select(records, counts, layers, coverage, n_experts):
    selected: list[tuple[int, int, int]] = []
    for layer in layers:
        if not 0 <= layer < 10_000:
            raise ValueError(f"invalid layer: {layer}")
        ranked = sorted(
            ((expert, counts.get((layer, expert), 0)) for expert in range(n_experts)),
            key=lambda item: (-item[1], item[0]),
        )
        if coverage >= 1.0:
            chosen = ranked
        else:
            total = sum(count for _, count in ranked)
            if total <= 0:
                chosen = []
            else:
                chosen = []
                covered = 0
                for expert, count in ranked:
                    if covered >= coverage * total:
                        break
                    if count > 0:
                        chosen.append((expert, count))
                        covered += count
        for expert, count in chosen:
            # Count 1 keeps an unseen expert in the candidate while preserving
            # the runtime's popularity ordering for observed experts.
            selected.append((layer, expert, max(1, count)))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--coverage", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 < args.coverage <= 1:
        parser.error("--coverage must be in (0, 1]")

    headers, counts, _n_layers, n_experts = read_history(args.history)
    selected = select(args.history, counts, args.layers, args.coverage, n_experts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(headers + [f"{l} {e} {c}" for l, e, c in selected]) + "\n"
    args.output.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {len(selected)} candidate experts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
