#!/usr/bin/env python3
"""Validate the fixed grouped-int4 fixture used by M3 execution-DAG work.

This is deliberately a small independent numerical reference, not a call into
Colibri. It pins grouped-int4 packing, per-group scaling, and the final short
group before reentrant M3 expert kernels are introduced.
"""

from __future__ import annotations

import json
from pathlib import Path


def grouped_matvec(fixture: dict) -> list[float]:
    input_size = fixture["input_size"]
    group_size = fixture["group_size"]
    vector = fixture["input"]
    groups = (input_size + group_size - 1) // group_size
    result: list[float] = []

    for packed_hex, scales in zip(fixture["packed_weights_hex"], fixture["scales"]):
        packed = bytes.fromhex(packed_hex)
        if len(scales) != groups:
            raise AssertionError(f"expected {groups} scales, got {len(scales)}")
        total = 0.0
        for group in range(groups):
            start = group * group_size
            stop = min(start + group_size, input_size)
            dot = 0.0
            for column in range(start, stop):
                byte = packed[column // 2]
                nibble = (byte >> (4 * (column % 2))) & 0x0F
                dot += (nibble - 8) * vector[column]
            total += dot * scales[group]
        result.append(total)
    return result


def main() -> int:
    path = Path(__file__).with_name("fixtures") / "m3_grouped_int4_reference.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    actual = grouped_matvec(fixture)
    expected = fixture["expected_output"]
    if len(actual) != len(expected):
        raise AssertionError("output length mismatch")
    for index, (got, want) in enumerate(zip(actual, expected)):
        if abs(got - want) > 1e-12:
            raise AssertionError(f"output[{index}] = {got!r}, expected {want!r}")
    print(f"m3-grouped-int4-reference: ok ({len(actual)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
