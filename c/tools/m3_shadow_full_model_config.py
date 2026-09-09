#!/usr/bin/env python3
"""Emit a bounded M3 shadow topology for a complete two-replica model.

The tool derives copy identity and exact expert byte counts from safetensors
headers in both replicas.  A planner-text capture from
``m3_shadow_calibrate.py`` may be supplied to use measured resource/profile
rows.  Without that capture, ``--synthetic-rate`` remains available only for
route/lease/read-equality integration fixtures and the output is marked
synthetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    from doctor import _safetensors_header, deep_container_report
except ImportError:  # direct execution from c/tools
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from doctor import _safetensors_header, deep_container_report

try:
    from m3_shadow_calibrate import _format_island_row, topology_hash64
except ImportError:
    from c.tools.m3_shadow_calibrate import _format_island_row, topology_hash64


_EXPERT = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(\.qs)?$"
)
_EXPECTED_PROJECTIONS = {
    (projection, suffix)
    for projection in ("gate_proj", "up_proj", "down_proj")
    for suffix in ("", ".qs")
}
_EXPECTED_COPY_KEYS = 7296
_MAX_COPIES = 16384


def _expected_expert_keys(model: Path) -> set[tuple[int, int]]:
    config_path = model / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read model config {config_path}: {exc}") from exc
    try:
        layer_count = config["num_hidden_layers"]
        expert_count = config["num_local_experts"]
        layer_frequency = config["moe_layer_freq"]
    except KeyError as exc:
        raise ValueError(f"model config lacks M3 expert field {exc.args[0]}") from exc
    if (isinstance(layer_count, bool) or not isinstance(layer_count, int) or
            layer_count <= 0 or isinstance(expert_count, bool) or
            not isinstance(expert_count, int) or expert_count <= 0 or
            not isinstance(layer_frequency, list) or
            len(layer_frequency) != layer_count):
        raise ValueError("model config has invalid M3 expert topology")
    expected = {
        (layer, expert)
        for layer, enabled in enumerate(layer_frequency)
        if enabled
        for expert in range(expert_count)
    }
    if len(expected) != _EXPECTED_COPY_KEYS:
        raise ValueError(
            f"unexpected M3 expert topology: expected {_EXPECTED_COPY_KEYS} keys, "
            f"got {len(expected)}"
        )
    return expected


def _validate_containers(primary: Path, mirror: Path) -> None:
    report = deep_container_report(primary, mirror_dir=mirror)
    if report["sequence"]["status"] != "pass":
        raise ValueError(f"primary safetensors sequence invalid: {report['sequence']}")
    if report["required"]["status"] != "pass":
        raise ValueError(f"primary required tensors invalid: {report['required']}")
    if report["mirror"]["status"] != "pass":
        raise ValueError(f"mirror safetensors admission invalid: {report['mirror']}")


def _expert_bytes(model: Path) -> dict[tuple[int, int], int]:
    tensors: dict[tuple[int, int, str, str], int] = {}
    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no safetensors shards in {model}")
    for shard in shards:
        _, _, header = _safetensors_header(shard)
        for name, metadata in header.items():
            match = _EXPERT.fullmatch(name)
            if match is None:
                continue
            layer, expert, projection, suffix = match.groups()
            key = (int(layer), int(expert), projection, suffix or "")
            if key in tensors:
                raise ValueError(f"duplicate expert tensor {name}")
            begin, end = metadata["data_offsets"]
            if end <= begin:
                raise ValueError(f"empty expert tensor {name}")
            tensors[key] = int(end - begin)
    grouped: dict[tuple[int, int], dict[tuple[str, str], int]] = defaultdict(dict)
    for (layer, expert, projection, suffix), size in tensors.items():
        grouped[(layer, expert)][(projection, suffix)] = size
    if not grouped:
        raise ValueError(f"no M3 expert tensors in {model}")
    result: dict[tuple[int, int], int] = {}
    for key, parts in grouped.items():
        if set(parts) != _EXPECTED_PROJECTIONS:
            missing = sorted(_EXPECTED_PROJECTIONS - set(parts))
            extra = sorted(set(parts) - _EXPECTED_PROJECTIONS)
            raise ValueError(f"incomplete expert {key}: missing={missing} extra={extra}")
        result[key] = sum(parts.values())
    expected = _expected_expert_keys(model)
    actual = set(result)
    if actual != expected:
        raise ValueError(
            "expert inventory does not match config: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    return result


def _expected_topology_rows(manifest: dict) -> set[str]:
    bounds = manifest.get("storage_bounds")
    if not isinstance(bounds, dict):
        raise ValueError("manifest lacks storage_bounds")
    max_inflight = bounds.get("max_inflight")
    max_bytes = bounds.get("max_bytes")
    if (not isinstance(max_inflight, int) or not isinstance(max_bytes, int)):
        raise ValueError("manifest storage_bounds are malformed")
    expected: set[str] = set()
    for source in manifest.get("sources", []):
        drive = int(source["drive_id"])
        controller = int(source.get("controller_id") or 0)
        upstream = int(source.get("upstream_id") or 0)
        for resource_id, kind, parent in (
                (drive, 0, controller), (controller, 1, upstream),
                (upstream, 2, 0)):
            if resource_id:
                expected.add(f"resource {resource_id} {kind} {parent} "
                             f"{max_inflight} {max_bytes}")
    for island in manifest.get("islands", []):
        row = _format_island_row(island)
        if row is not None:
            expected.add(row.rstrip("\n"))
    return expected


def _measured_rows(path: Path, topology_hash: int,
                   required_drives: tuple[int, int],
                   manifest_path: Path | None = None,
                   allow_topology_only: bool = False,
                   review_path: Path | None = None
                   ) -> tuple[list[str], str, list[str]]:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read planner capture {path}: {exc}") from exc
    if f"# topology_hash = {topology_hash}" not in raw_lines:
        raise ValueError("planner capture topology_hash does not match model config")
    manifest_lines = [line for line in raw_lines
                      if line.startswith("# manifest_sha256 = ")]
    if len(manifest_lines) != 1:
        raise ValueError("planner capture must contain one manifest_sha256")
    manifest_sha = manifest_lines[0].split("=", 1)[1].strip()
    if not re.fullmatch(r"[0-9a-fA-F]{64}", manifest_sha):
        raise ValueError("planner capture manifest_sha256 is malformed")
    if manifest_path is None:
        raise ValueError("measured planner text requires --manifest verification")
    if manifest_path is not None:
        try:
            expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot read calibration manifest {manifest_path}: {exc}") from exc
        if manifest_sha != expected_sha:
            raise ValueError("planner capture manifest_sha256 does not match manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse calibration manifest {manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ValueError("calibration manifest must be an object")
        if manifest.get("topology_hash") != topology_hash:
            raise ValueError("planner topology_hash does not match manifest")
        if manifest.get("topology_hash_algorithm") != "sha256-trunc64-v1":
            raise ValueError("manifest topology_hash_algorithm is not canonical")
        if topology_hash64(manifest) != topology_hash:
            raise ValueError("manifest topology_hash is not canonical")
    rows = [line.strip() for line in raw_lines
            if line.strip() and not line.lstrip().startswith("#")]
    if manifest_path is not None:
        actual_topology = {
            line for line in rows
            if line.startswith(("resource ", "cpu_island ", "gpu_island ",
                                "memory_dom ", "link "))
        }
        expected_topology = _expected_topology_rows(manifest)
        if actual_topology != expected_topology:
            raise ValueError(
                "planner topology rows do not match manifest: "
                f"missing={sorted(expected_topology - actual_topology)} "
                f"extra={sorted(actual_topology - expected_topology)}")
    if any(line.startswith("copy ") for line in rows):
        raise ValueError("planner capture must not contain copy rows")
    resources = {int(line.split()[1]) for line in rows
                 if line.startswith("resource ")}
    missing = [str(drive) for drive in required_drives if drive not in resources]
    if missing:
        raise ValueError(f"planner capture lacks model drive resource(s): {', '.join(missing)}")
    if (not allow_topology_only
            and not any(line.startswith("profile ") for line in rows)):
        raise ValueError("planner capture has no measured profile rows")
    has_profiles = any(line.startswith("profile ") for line in rows)
    review_schema_lines = [line for line in raw_lines
                           if line.startswith("# review_schema = ")]
    review_sha_lines = [line for line in raw_lines
                        if line.startswith("# review_sha256 = ")]
    active_lines = [line for line in raw_lines
                    if line.startswith("# active_admission_allowed = ")]
    if has_profiles:
        if (review_schema_lines !=
                ["# review_schema = m3-shadow-calibration-review-v2"]):
            raise ValueError("profile rows require the reviewed v2 schema")
        if len(review_sha_lines) != 1 or not re.fullmatch(
                r"# review_sha256 = [0-9a-fA-F]{64}", review_sha_lines[0]):
            raise ValueError("profile rows require a valid review_sha256")
        if active_lines != ["# active_admission_allowed = False"]:
            raise ValueError("profile rows require inactive admission")
        if review_path is None:
            raise ValueError("profile rows require --review verification")
        try:
            review = json.loads(review_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read review {review_path}: {exc}") from exc
        if (not isinstance(review, dict)
                or review.get("schema") != "m3-shadow-calibration-review-v2"):
            raise ValueError("review file has an unsupported schema")
        review_sha = hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if review_sha_lines[0].split("=", 1)[1].strip() != review_sha:
            raise ValueError("planner review_sha256 does not match --review")
    prefixes = (
        "# topology_hash_algorithm = ", "# topology_hash_verified = ",
        "# cache_condition = ", "# io_mode = ",
        "# io_backend = ", "# asynchronous = ",
        "# async_request_protocol = ",
        "# capture_sha256 = ",
        "# platform_topology_status = ", "# platform_topology_method = ",
        "# pcie_negotiated_link = ",
        "# pcie_parent_chain = ",
        "# pcie_common_ancestor = ",
        "# pcie_traffic_status = ",
        "# pcie_traffic_policy = ",
        "# unavailable_resource_policy = ",
        "# unavailable_resource_fallback = ",
        "# capture_schema = ", "# calibrator_tool = ",
        "# calibrator_source_sha256 = ", "# quantiles = ",
        "# physical_traffic_note: ", "# range_reuse = ",
        "# range_reuse_scope = ", "# range_slots_required = ",
        "# range_reuse_allowed = ", "# range_slots_available = ",
        "# range_slots_reused = ",
        "# topology_not_run ",
        "# profile_rows = ",
        "# review_schema = ", "# review_sha256 = ",
        "# active_admission_allowed = ",
        "# configuration_status = ", "# host_memory_method = ",
        "# host_memory_evidence = ",
        "# limitations = ",
    )
    qualifiers = [line for line in raw_lines
                  if line.startswith(prefixes)]
    return rows, manifest_sha, qualifiers


def emit_config(primary: Path, mirror: Path, model_id: int, drive: int,
                mirror_drive: int, synthetic_rate: int | None,
                topology_hash: int, planner_text: Path | None = None,
                manifest_path: Path | None = None,
                allow_topology_only: bool = False,
                review_path: Path | None = None) -> str:
    if (drive, mirror_drive) != (101, 102):
        raise ValueError("this fixture emits topology resources 101/102; use those drive IDs")
    _validate_containers(primary, mirror)
    primary_bytes = _expert_bytes(primary)
    mirror_bytes = _expert_bytes(mirror)
    if primary_bytes != mirror_bytes:
        only_primary = sorted(set(primary_bytes) - set(mirror_bytes))
        only_mirror = sorted(set(mirror_bytes) - set(primary_bytes))
        different = sorted(k for k in set(primary_bytes) & set(mirror_bytes)
                           if primary_bytes[k] != mirror_bytes[k])
        raise ValueError(
            "replica expert inventories differ: "
            f"only_primary={only_primary[:3]} only_mirror={only_mirror[:3]} "
            f"different={different[:3]}"
        )
    if model_id <= 0 or drive <= 0 or mirror_drive <= 0:
        raise ValueError("model and drive IDs must be positive")
    if topology_hash <= 0:
        raise ValueError("topology hash must be positive")
    if planner_text is None and (synthetic_rate is None or synthetic_rate <= 0):
        raise ValueError("provide --planner-text or a positive --synthetic-rate")
    if drive == mirror_drive:
        raise ValueError("replica drives must be distinct")
    if len(primary_bytes) * 2 > _MAX_COPIES:
        raise ValueError(
            f"copy inventory exceeds planner bound {_MAX_COPIES}: "
            f"{len(primary_bytes) * 2}"
        )

    lines = [
        "# M3 full-model route/read configuration",
        f"# topology_hash = {topology_hash}",
    ]
    planner_has_profiles = False
    if planner_text is not None:
        measured_rows, manifest_sha, qualifiers = _measured_rows(
            planner_text, topology_hash, (drive, mirror_drive), manifest_path,
            allow_topology_only, review_path)
        lines.append(f"# manifest_sha256 = {manifest_sha}")
        lines.extend(qualifiers)
        if not any(line.startswith("# limitations = ") for line in qualifiers):
            lines.append("# limitations = cache/direct-I/O and host-memory evidence remain qualified")
        planner_has_profiles = any(line.startswith("profile ")
                                   for line in measured_rows)
        lines.append("# profiles are reviewed capture rows; missing paths remain UNKNOWN"
                     if planner_has_profiles
                     else "# profiles are absent; topology-only configuration")
        lines.extend(measured_rows)
    else:
        lines.append("# profiles are synthetic; this file is NOT hardware calibration")
        lines.extend([
            f"resource {drive} 0 201 64 1099511627776",
            "resource 201 1 301 64 1099511627776",
            "resource 301 2 0 64 1099511627776",
            f"resource {mirror_drive} 0 202 64 1099511627776",
            "resource 202 1 302 64 1099511627776",
            "resource 302 2 0 64 1099511627776",
        ])
        for resource_id in (drive, 201, 301, mirror_drive, 202, 302):
            for size_class in range(4):
                for load_class in range(5):
                    lines.append(
                        f"profile {resource_id} {size_class} {load_class} "
                        f"{synthetic_rate} 0 0 1"
                    )
    copy_id = 1
    for layer, expert in sorted(primary_bytes):
        tensor_id = (layer << 32) | expert
        size = primary_bytes[(layer, expert)]
        lines.append(f"copy {copy_id} {model_id} {tensor_id} {size} {drive} 1")
        copy_id += 1
        lines.append(
            f"copy {copy_id} {model_id} {tensor_id} {size} {mirror_drive} 1"
        )
        copy_id += 1
    if planner_text is not None:
        artifact_label = ("reviewed-profile-rows" if planner_has_profiles
                          else "topology-only")
    else:
        artifact_label = "synthetic-profile-only"
    lines.append(f"# experts={len(primary_bytes)} copies={copy_id - 1} "
                 f"{artifact_label}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--mirror", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", required=True, type=int)
    parser.add_argument("--drive", type=int, default=101)
    parser.add_argument("--mirror-drive", type=int, default=102)
    parser.add_argument("--synthetic-rate", type=int)
    parser.add_argument("--planner-text", type=Path,
                        help="measured planner text emitted by m3_shadow_calibrate.py")
    parser.add_argument("--manifest", type=Path,
                        help="optional manifest to verify against planner-text provenance")
    parser.add_argument("--review", type=Path,
                        help="review JSON to verify profile-row provenance")
    parser.add_argument("--allow-topology-only", action="store_true",
                        help="accept a raw topology-only planner capture with no profile rows")
    parser.add_argument("--topology-hash", type=int, default=20260907)
    args = parser.parse_args()
    try:
        text = emit_config(args.model, args.mirror, args.model_id,
                           args.drive, args.mirror_drive,
                           args.synthetic_rate, args.topology_hash,
                           args.planner_text, args.manifest,
                           args.allow_topology_only, args.review)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output.write_text(text, encoding="utf-8", newline="\n")
    copies = sum(line.startswith("copy ") for line in text.splitlines())
    experts = copies // 2
    print(f"full-model config: PASS experts={experts} copies={copies} "
          f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
