#!/usr/bin/env python3
"""Bounded read calibration capture for the Step-7 shadow planner.

The manifest names existing expert-container files and the physical resource
IDs they represent.  The tool only reads; it never creates, truncates, or
flushes model files.  It records raw per-sample latency and aggregate
throughput for each drive/group and in-flight level.  It deliberately does
not turn a measurement into a planner profile: cache condition, request
class, and controller interpretation remain explicit review inputs.

Manifest shape::

    {
      "topology_hash": 20260907,
      "sources": [
        {"drive_id": 101, "controller_id": 201,
         "upstream_id": 901, "path": "C:/models/out-00.safetensors"}
      ],
      "groups": [
        {"name": "drive-101", "drive_ids": [101]},
        {"name": "controller-201", "drive_ids": [101]}
      ]
    }

Groups are explicit.  A group with one drive measures queued reads on that
drive.  A multi-drive group defaults to a bounded contention matrix with an
independent queue depth for every drive; ``"sweep": "scalar"`` opts into the
legacy round-robin aggregate sweep.  A controller/upstream target must be
declared by the caller, not inferred from drive count.

Slice 7b-ext: the planner can also consume the new resource kinds
(cpu_island, gpu_island, memory_dom, link).  The capture side of this
tool remains a measurement instrument: it does not invent defaults or
extrapolate.  When the manifest declares compute-island entries (with
their own bandwidth or latency numbers, measured externally), the tool
emits a sibling planner-readable text dump alongside the raw capture JSON
via ``--emit-planner-text <path>``.  Each new resource kind is independent:
a missing measurement becomes a ``NOT_RUN`` field on the planner row,
not a fabricated capacity.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import hashlib
import itertools
import json
import mmap
import math
import ntpath
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import struct
import sys
import threading
import time
from typing import Any, NoReturn


CALIBRATION_SCHEMA = "m3-shadow-calibration-v2"
CALIBRATION_TOOL_VERSION = 2


def fail(message: str) -> "NoReturn":
    raise SystemExit(f"m3-shadow-calibrate: {message}")


_TOPOLOGY_HASH_ALGORITHM = "sha256-trunc64-v1"


def canonical_topology_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only the topology fields visible to the planner."""
    sources = []
    for source in manifest.get("sources", []):
        sources.append({key: source.get(key) for key in
                        ("drive_id", "controller_id", "upstream_id")})
    groups = []
    for group in manifest.get("groups", []):
        groups.append({key: group.get(key) for key in
                       ("name", "drive_ids", "resource_id", "kind", "sweep")})
    return {
        "sources": sorted(sources, key=lambda item: int(item["drive_id"])),
        "groups": sorted(groups, key=lambda item: str(item["name"])),
        # Platform endpoint bindings are evidence metadata, not planner
        # topology.  Keeping them out of this hash lets a machine-specific
        # probe enrich the capture without changing resource identity.
        "islands": sorted([
            {key: value for key, value in island.items()
             if key != "platform"}
            for island in manifest.get("islands", [])
        ], key=lambda item: int(item["resource_id"])),
        "storage_bounds": manifest.get("storage_bounds", {}),
        "active_admission": bool(manifest.get("active_admission", False)),
        "profile_status": manifest.get("profile_status", "raw_only"),
    }


def topology_hash64(manifest: dict[str, Any]) -> int:
    encoded = json.dumps(canonical_topology_payload(manifest),
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return value or 1


def capture_hash(capture: dict[str, Any]) -> str:
    """Hash the complete capture excluding its self-referential digest."""
    unsigned = dict(capture)
    unsigned.pop("capture_sha256", None)
    encoded = json.dumps(unsigned, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    if not 0.0 <= q <= 1.0:
        fail("percentile q must be in 0..1")
    ordered = sorted(values)
    # Nearest-rank, conservatively rounded upward.  This is explicit so a
    # small p95/p99 population cannot silently use interpolation or bankers'
    # rounding.
    rank = max(1, int(math.ceil(q * len(ordered))))
    index = min(len(ordered) - 1, rank - 1)
    return ordered[index]


def load_manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read manifest {path}: {exc}")
    if not isinstance(manifest, dict):
        fail("manifest root must be an object")
    if not isinstance(manifest.get("topology_hash"), int) or not manifest["topology_hash"]:
        fail("topology_hash must be a nonzero integer")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("sources must be a non-empty list")
    seen: set[int] = set()
    resource_kinds: dict[int, str] = {}
    controller_upstream: dict[int, int | None] = {}
    for source in sources:
        if not isinstance(source, dict):
            fail("each source must be an object")
        drive_id = source.get("drive_id")
        if not isinstance(drive_id, int) or not drive_id or drive_id in seen:
            fail("source drive_id values must be unique nonzero integers")
        seen.add(drive_id)
        prior_drive_kind = resource_kinds.get(drive_id)
        if prior_drive_kind is not None:
            fail(f"resource {drive_id} is used as both {prior_drive_kind} and drive")
        resource_kinds[drive_id] = "drive"
        for field in ("controller_id", "upstream_id"):
            if field in source and source[field] is not None and (
                not isinstance(source[field], int) or not source[field]
            ):
                fail(f"source {field} must be a nonzero integer when present")
            value = source.get(field)
            if value:
                kind = "controller" if field == "controller_id" else "upstream"
                prior = resource_kinds.get(value)
                if prior is not None and prior != kind:
                    fail(f"resource {value} is used as both {prior} and {kind}")
                resource_kinds[value] = kind
        controller_id = source.get("controller_id")
        if controller_id:
            upstream_id = source.get("upstream_id")
            if (controller_id in controller_upstream
                    and controller_upstream[controller_id] != upstream_id):
                fail(f"controller {controller_id} has contradictory upstream parents")
            controller_upstream[controller_id] = upstream_id
        path = source.get("path")
        paths = source.get("paths")
        if paths is not None:
            if path is not None:
                fail("source must use either path or paths, not both")
            if (not isinstance(paths, list) or not paths
                    or any(not isinstance(item, str) or not item for item in paths)):
                fail("source paths must be a non-empty list of strings")
        elif not isinstance(path, str) or not path:
            fail("every source needs a path or paths")
    groups = manifest.get("groups")
    if not isinstance(groups, list) or not groups:
        fail("groups must be a non-empty list")
    known_resource_ids = set(resource_kinds)
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get("name"), str):
            fail("each group needs a name")
        ids = group.get("drive_ids")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(i, int) or i <= 0 for i in ids)
                or len(set(ids)) != len(ids)
                or any(i not in seen for i in ids)):
            fail(f"group {group.get('name', '<unnamed>')} has invalid drive_ids")
        target_id = group.get("resource_id")
        target_kind = group.get("kind")
        if target_id is not None:
            if (not isinstance(target_id, int) or not target_id
                    or target_id not in known_resource_ids):
                fail(f"group {group['name']} has an unknown resource_id")
            if target_kind not in ("drive", "controller", "upstream"):
                fail(f"group {group['name']} target kind must be drive/controller/upstream")
            if resource_kinds.get(target_id) != target_kind:
                fail(f"group {group['name']} target kind does not match resource topology")
            actual_ids = set(ids)
            if target_kind == "drive":
                expected_ids = {target_id}
            elif target_kind == "controller":
                expected_ids = {
                    int(source["drive_id"]) for source in sources
                    if source.get("controller_id") == target_id
                }
            else:
                expected_ids = {
                    int(source["drive_id"]) for source in sources
                    if source.get("upstream_id") == target_id
                }
            if not expected_ids or actual_ids != expected_ids:
                fail(f"group {group['name']} does not name the complete {target_kind} membership")
        elif target_kind is not None:
            fail(f"group {group['name']} has kind without resource_id")
        sweep = group.get("sweep", "matrix" if len(ids) > 1 else "scalar")
        if sweep not in ("scalar", "matrix"):
            fail(f"group {group['name']} sweep must be scalar or matrix")
        if sweep == "matrix" and len(ids) < 2:
            fail(f"group {group['name']} matrix sweep needs at least two drives")
    bounds = manifest.get("storage_bounds", {})
    if not isinstance(bounds, dict):
        fail("storage_bounds must be an object when present")
    if bounds:
        max_inflight = bounds.get("max_inflight")
        max_bytes = bounds.get("max_bytes")
        if (not isinstance(max_inflight, int) or not 1 <= max_inflight <= 16
                or not isinstance(max_bytes, int) or max_bytes <= 0):
            fail("storage_bounds requires max_inflight 1..16 and positive max_bytes")
    islands = manifest.get("islands") or []
    if not isinstance(islands, list):
        fail("islands must be a list when present")
    island_kinds: dict[int, str] = {}
    for island in islands:
        if not isinstance(island, dict):
            fail("each island entry must be an object")
        kind = island.get("kind")
        if kind not in ("cpu_island", "gpu_island", "memory_dom", "link"):
            fail(f"island kind must be one of cpu_island/gpu_island/memory_dom/link "
                 f"(got {kind!r})")
        resource_id = island.get("resource_id")
        if not isinstance(resource_id, int) or not resource_id:
            fail("each island entry needs a nonzero integer resource_id")
        if resource_id in resource_kinds or resource_id in island_kinds:
            fail(f"resource id {resource_id} is declared more than once")
        island_kinds[resource_id] = str(kind)
        if kind in ("cpu_island", "gpu_island"):
            eligible = island.get("eligible")
            if not isinstance(eligible, int) or eligible not in (0, 1):
                fail(f"{kind} {resource_id} requires explicit eligible 0 or 1")
        capacity_status = island.get("capacity_status", "NOT_RUN")
        if capacity_status not in ("MEASURED", "NOT_RUN", "UNKNOWN"):
            fail(f"island {resource_id} has invalid capacity_status")
        rate = island.get("rate_bytes_per_s")
        if (capacity_status in ("NOT_RUN", "UNKNOWN")
                and isinstance(rate, int) and rate > 0):
            fail(f"island {resource_id} has a positive rate with unknown capacity")
        if (capacity_status == "MEASURED"
                and kind in ("memory_dom", "link")
                and (not isinstance(rate, int) or rate <= 0)):
            fail(f"measured island {resource_id} requires a positive rate")
        required_fields = {
            "cpu_island": ("parent_mem_id", "isa_bits", "cores", "eligible"),
            "gpu_island": ("parent_link_id", "vram_bytes", "eligible"),
            "memory_dom": ("parent_id",),
            "link": ("parent_cpu_id", "pcie_parent_id"),
        }[kind]
        missing = [field for field in required_fields if field not in island]
        if missing:
            fail(f"island {resource_id} is missing required field(s): "
                 + ", ".join(missing))
        for field in ("parent_id", "parent_mem_id", "parent_cpu_id",
                      "parent_link_id", "pcie_parent_id"):
            if field in island and island[field] is not None and (
                    not isinstance(island[field], int) or island[field] < 0):
                fail(f"island {resource_id} {field} must be a nonnegative integer")
        for field in ("rate_bytes_per_s", "bandwidth_class", "cores",
                      "isa_bits", "vram_bytes"):
            if field in island and island[field] is not None and (
                    not isinstance(island[field], int) or island[field] < 0):
                fail(f"island {resource_id} {field} must be a nonnegative integer")
    all_kinds = dict(resource_kinds)
    all_kinds.update(island_kinds)
    parent_edges: dict[int, list[int]] = {}
    for island in islands:
        resource_id = int(island["resource_id"])
        kind = island["kind"]
        parent_field = {
            "cpu_island": "parent_mem_id",
            "memory_dom": "parent_id",
            "link": "parent_cpu_id",
            "gpu_island": "parent_link_id",
        }[kind]
        parent = island.get(parent_field, 0)
        if parent:
            expected_kind = {
                "cpu_island": "memory_dom",
                "memory_dom": None,
                "link": "cpu_island",
                "gpu_island": "link",
            }[kind]
            if parent not in all_kinds:
                fail(f"island {resource_id} has undeclared parent {parent}")
            if expected_kind is not None and all_kinds[parent] != expected_kind:
                fail(f"island {resource_id} parent {parent} has wrong kind")
            parent_edges.setdefault(resource_id, []).append(parent)
        pcie_parent = island.get("pcie_parent_id", 0)
        if pcie_parent and pcie_parent not in all_kinds:
            fail(f"island {resource_id} has undeclared pcie parent {pcie_parent}")
        if pcie_parent:
            parent_edges.setdefault(resource_id, []).append(pcie_parent)
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit_topology(node: int) -> None:
        if node in visiting:
            fail(f"topology parent cycle includes resource {node}")
        if node in visited:
            return
        visiting.add(node)
        for parent in parent_edges.get(node, []):
            visit_topology(parent)
        visiting.remove(node)
        visited.add(node)

    for start in parent_edges:
        visit_topology(start)
    if manifest.get("active_admission", False) is not False:
        fail("machine calibration manifest cannot enable active admission")
    profile_status = manifest.get("profile_status", "raw_only")
    if profile_status not in ("raw_only", "candidate_only"):
        fail("profile_status must be raw_only or candidate_only")
    hash_algorithm = manifest.get("topology_hash_algorithm")
    if hash_algorithm != _TOPOLOGY_HASH_ALGORITHM:
        fail("topology_hash_algorithm is required for canonical calibration")
    if manifest["topology_hash"] != topology_hash64(manifest):
        fail("topology_hash does not match canonical planner topology")
    return manifest, raw


def _parse_windows_volume_disk_extents(raw: bytes) -> dict[str, Any]:
    """Parse VOLUME_DISK_EXTENTS without trusting a path-to-disk guess."""
    if len(raw) < 8:
        return {"status": "UNAVAILABLE", "reason": "short extents response"}
    count = struct.unpack_from("<I", raw, 0)[0]
    extent_size = struct.calcsize("<I4xqq")
    extents = []
    for index in range(count):
        offset = 8 + index * extent_size
        if offset + extent_size > len(raw):
            return {"status": "UNAVAILABLE",
                    "reason": "truncated extents response",
                    "reported_extent_count": count}
        disk_number, start, length = struct.unpack_from(
            "<I4xqq", raw, offset)
        extents.append({"disk_number": disk_number,
                        "starting_offset": start,
                        "extent_length": length})
    return {"status": "OBSERVED", "disk_numbers": sorted({
        int(item["disk_number"]) for item in extents}),
            "extents": extents,
            "method": "windows-ioctl-volume-disk-extents"}


def probe_windows_volume_disk_extents(source: dict[str, Any],
                                      path: str) -> dict[str, Any]:
    """Map a Windows volume to physical disk numbers using a read-only IOCTL."""
    if os.name != "nt":
        return {"status": "UNAVAILABLE", "reason": "Windows-only probe"}
    platform_info = source.get("platform")
    if not isinstance(platform_info, dict):
        platform_info = {}
    drive = str(platform_info.get("windows_volume") or
                ntpath.splitdrive(path)[0]).rstrip("\\/")
    if not re.fullmatch(r"[A-Za-z]:", drive):
        return {"status": "UNAVAILABLE", "reason": "volume binding is missing"}
    volume_path = "\\\\.\\" + drive
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle_t = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, handle_t]
        kernel32.CreateFileW.restype = handle_t
        kernel32.DeviceIoControl.argtypes = [
            handle_t, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p]
        kernel32.DeviceIoControl.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [handle_t]
        kernel32.CloseHandle.restype = wintypes.BOOL
        share = 0x00000001 | 0x00000002
        handle = kernel32.CreateFileW(volume_path, 0, share, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value
        if handle == invalid:
            return {"status": "UNAVAILABLE", "volume": drive,
                    "reason": f"CreateFileW failed with error {ctypes.get_last_error()}"}
        try:
            ioctl_volume_disk_extents = 0x00560000
            output = ctypes.create_string_buffer(4096)
            returned = wintypes.DWORD()
            ok = kernel32.DeviceIoControl(
                handle, ioctl_volume_disk_extents, None, 0,
                output, len(output), ctypes.byref(returned), None)
            if not ok:
                return {"status": "UNAVAILABLE", "volume": drive,
                        "reason": ("DeviceIoControl failed with error "
                                   f"{ctypes.get_last_error()}"),
                        "method": "windows-ioctl-volume-disk-extents"}
            evidence = _parse_windows_volume_disk_extents(
                output.raw[:int(returned.value)])
            evidence["volume"] = drive
            return evidence
        finally:
            kernel32.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError) as exc:
        return {"status": "UNAVAILABLE", "volume": drive,
                "method": "windows-ioctl-volume-disk-extents",
                "reason": str(exc)}


def resolve_sources(manifest: dict[str, Any], read_bytes: int) -> dict[int, dict[str, Any]]:
    resolved: dict[int, dict[str, Any]] = {}
    physical_files: dict[tuple[Any, ...], tuple[int, str]] = {}
    for source in manifest["sources"]:
        drive_id = int(source["drive_id"])
        raw_paths = source.get("paths")
        if raw_paths is None:
            raw_paths = [source["path"]]
        files: list[dict[str, Any]] = []
        total_size = 0
        slot_base = 0
        for raw_path in raw_paths:
            path = Path(os.path.expandvars(os.path.expanduser(raw_path)))
            try:
                stat_result = path.stat()
                size = stat_result.st_size
            except OSError as exc:
                fail(f"cannot stat drive {drive_id} path {path}: {exc}")
            canonical = str(path.resolve()).casefold()
            if getattr(stat_result, "st_ino", 0):
                identity = ("stat", int(stat_result.st_dev),
                            int(stat_result.st_ino))
            else:
                identity = ("path", canonical)
            prior = physical_files.get(identity)
            if prior is not None:
                fail(f"drive {drive_id} path {path} aliases drive {prior[0]} "
                     f"path {prior[1]}; disjoint capacity would be overstated")
            physical_files[identity] = (drive_id, str(path))
            # Only complete read-sized slots are addressable. This prevents a
            # virtual concatenation from issuing a request across shards.
            slot_count = size // read_bytes
            if slot_count < 1:
                fail(f"drive {drive_id} file is only {size} bytes; need {read_bytes}")
            files.append({"path": str(path), "size_bytes": size,
                          "device_id": int(stat_result.st_dev),
                          "file_id": str(getattr(stat_result, "st_ino", 0)),
                          "slot_base": slot_base, "slot_count": slot_count})
            total_size += size
            slot_base += slot_count
        resolved[drive_id] = {
            "drive_id": drive_id,
            "controller_id": source.get("controller_id"),
            "upstream_id": source.get("upstream_id"),
            "path": files[0]["path"],
            "paths": [item["path"] for item in files],
            "files": files,
            "size_bytes": total_size,
            "available_slots": slot_base,
        }
        if os.name == "nt":
            resolved[drive_id]["physical_volume"] = (
                probe_windows_volume_disk_extents(
                    source, files[0]["path"]))
    return resolved


def _source_available_slots(source: dict[str, Any], read_bytes: int) -> int:
    slots = source.get("available_slots")
    if isinstance(slots, int) and slots > 0:
        return slots
    size = source.get("size_bytes")
    if not isinstance(size, int) or size < read_bytes:
        fail("source does not have one complete read-sized slot")
    return size // read_bytes


def _source_target(source: dict[str, Any], slot_index: int,
                   read_bytes: int) -> tuple[str, int]:
    """Map a virtual source slot to one physical file and local offset."""
    if slot_index < 0 or slot_index >= _source_available_slots(source, read_bytes):
        fail(f"source slot {slot_index} is outside the available range")
    files = source.get("files")
    if isinstance(files, list):
        for item in files:
            base = item.get("slot_base")
            count = item.get("slot_count")
            if (isinstance(base, int) and isinstance(count, int)
                    and base <= slot_index < base + count):
                return (str(item["path"]), (slot_index - base) * read_bytes)
        fail(f"source slot {slot_index} has no physical file mapping")
    return (str(source["path"]), slot_index * read_bytes)


def read_range(path: str, offset: int, length: int) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    fd = os.open(path, flags)
    try:
        if hasattr(os, "pread"):
            data = os.pread(fd, length, offset)
        else:
            os.lseek(fd, offset, os.SEEK_SET)
            data = os.read(fd, length)
    finally:
        os.close(fd)
    if len(data) != length:
        raise OSError(f"short read: got {len(data)} of {length}")
    return len(data)


class _BufferedReadHandle:
    """Persistent buffered descriptor used by sustained contention cells."""

    def __init__(self, path: str):
        flags = os.O_RDONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        self._fd = os.open(path, flags)

    def read(self, offset: int, length: int) -> int:
        if hasattr(os, "pread"):
            data = os.pread(self._fd, length, offset)
        else:
            os.lseek(self._fd, offset, os.SEEK_SET)
            data = os.read(self._fd, length)
        if len(data) != length:
            raise OSError(f"short read: got {len(data)} of {length}")
        return len(data)

    def close(self) -> None:
        os.close(self._fd)


class _DirectReadHandle:
    """Platform direct/unbuffered reader with one handle per worker."""

    def __init__(self, path: str, length: int):
        if length <= 0 or length % 4096:
            raise OSError("direct-I/O length must be a positive 4096-byte multiple")
        self._kind = ""
        self._handle = None
        self._fd = None
        self._buffer = None
        if os.name == "nt":
            self._open_windows(path, length)
        elif sys.platform == "darwin":
            self._open_posix(path, length, f_nocache=True)
        elif sys.platform.startswith("linux"):
            self._open_posix(path, length, f_nocache=False)
        else:
            raise OSError(f"direct-I/O backend unsupported on {sys.platform}")

    def _open_windows(self, path: str, length: int) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle_t = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
            wintypes.DWORD, handle_t]
        kernel32.CreateFileW.restype = handle_t
        kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          wintypes.DWORD,
                                          wintypes.DWORD]
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [handle_t]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         wintypes.DWORD]
        kernel32.VirtualFree.restype = wintypes.BOOL
        kernel32.SetFilePointerEx.argtypes = [handle_t, ctypes.c_longlong,
                                              ctypes.POINTER(ctypes.c_longlong),
                                              wintypes.DWORD]
        kernel32.SetFilePointerEx.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [handle_t, ctypes.c_void_p,
                                      wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD),
                                      ctypes.c_void_p]
        kernel32.ReadFile.restype = wintypes.BOOL
        share = (0x00000001 | 0x00000002 | 0x00000004)
        no_buffering = 0x20000000
        open_existing = 3
        invalid = wintypes.HANDLE(-1).value
        handle = kernel32.CreateFileW(
            path, 0x80000000, share, None, open_existing, no_buffering, None)
        if handle == invalid:
            raise OSError(ctypes.get_last_error(), "CreateFileW direct failed")
        buffer = kernel32.VirtualAlloc(None, length, 0x1000 | 0x2000, 0x04)
        if not buffer:
            kernel32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(), "VirtualAlloc direct buffer failed")
        self._kind = "win32-no-buffering"
        self._kernel32 = kernel32
        self._handle = handle
        self._buffer = buffer
        self._length = length

    def _open_posix(self, path: str, length: int, f_nocache: bool) -> None:
        if not hasattr(os, "preadv"):
            raise OSError("preadv is required for aligned direct-I/O reads")
        flags = os.O_RDONLY
        if not f_nocache:
            direct = getattr(os, "O_DIRECT", 0)
            if not direct:
                raise OSError("O_DIRECT is unavailable")
            flags |= direct
        fd = os.open(path, flags)
        try:
            if f_nocache:
                import fcntl
                command = getattr(fcntl, "F_NOCACHE", None)
                if command is None:
                    raise OSError("F_NOCACHE is unavailable")
                fcntl.fcntl(fd, command, 1)
            buffer = mmap.mmap(-1, length, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(fd)
            raise
        self._kind = "macos-f_nocache" if f_nocache else "linux-o_direct"
        self._fd = fd
        self._buffer = buffer
        self._length = length

    def read(self, offset: int, length: int) -> int:
        if length != self._length:
            raise OSError("direct reader length changed")
        if self._fd is not None:
            count = os.preadv(self._fd, [self._buffer], offset)
        else:
            new_position = ctypes.c_longlong()
            if not self._kernel32.SetFilePointerEx(
                    self._handle, ctypes.c_longlong(offset),
                    ctypes.byref(new_position), 0):
                raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
            completed = wintypes.DWORD()
            if not self._kernel32.ReadFile(
                    self._handle, self._buffer, length,
                    ctypes.byref(completed), None):
                raise OSError(ctypes.get_last_error(), "ReadFile direct failed")
            count = int(completed.value)
        if count != length:
            raise OSError(f"short direct read: got {count} of {length}")
        return count

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        elif self._handle is not None:
            self._kernel32.VirtualFree(self._buffer, 0, 0x8000)
            self._kernel32.CloseHandle(self._handle)
            self._handle = None
        if self._buffer is not None and hasattr(self._buffer, "close"):
            self._buffer.close()
        self._buffer = None


class _WindowsOverlappedReadHandle:
    """Windows unbuffered reader using an OVERLAPPED completion event.

    This is deliberately a separate backend from synchronous ``ReadFile``.
    The handle is opened with ``FILE_FLAG_OVERLAPPED`` and every request is
    completed through ``GetOverlappedResult``.  The completion path proves
    asynchronous OS I/O was requested and observed; it does not attest which
    hardware engine moved the bytes into the destination buffer.
    """

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    def __init__(self, path: str, length: int):
        if os.name != "nt":
            raise OSError("win32-overlapped backend requires Windows")
        if length <= 0 or length % 4096:
            raise OSError("overlapped direct-I/O length must be a positive 4096-byte multiple")
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._handle = None
        self._event = None
        self._buffer = None
        self._length = length
        self._pending: list[_WindowsOverlappedReadHandle._OVERLAPPED] = []
        self._configure_api()
        self._open(path, length)

    def _configure_api(self) -> None:
        kernel32 = self._kernel32
        handle_t = wintypes.HANDLE
        kernel32.CreateFileW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, handle_t]
        kernel32.CreateFileW.restype = handle_t
        kernel32.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL,
                                          wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.CreateEventW.restype = handle_t
        kernel32.ResetEvent.argtypes = [handle_t]
        kernel32.ResetEvent.restype = wintypes.BOOL
        kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                          wintypes.DWORD, wintypes.DWORD]
        kernel32.VirtualAlloc.restype = ctypes.c_void_p
        kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                         wintypes.DWORD]
        kernel32.VirtualFree.restype = wintypes.BOOL
        kernel32.ReadFile.argtypes = [handle_t, ctypes.c_void_p,
                                      wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD),
                                      ctypes.POINTER(self._OVERLAPPED)]
        kernel32.ReadFile.restype = wintypes.BOOL
        kernel32.GetOverlappedResult.argtypes = [
            handle_t, ctypes.POINTER(self._OVERLAPPED),
            ctypes.POINTER(wintypes.DWORD), wintypes.BOOL]
        kernel32.GetOverlappedResult.restype = wintypes.BOOL
        kernel32.CancelIoEx.argtypes = [
            handle_t, ctypes.POINTER(self._OVERLAPPED)]
        kernel32.CancelIoEx.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [handle_t]
        kernel32.CloseHandle.restype = wintypes.BOOL

    def _open(self, path: str, length: int) -> None:
        kernel32 = self._kernel32
        share = 0x00000001 | 0x00000002 | 0x00000004
        no_buffering = 0x20000000
        overlapped = 0x40000000
        open_existing = 3
        invalid = wintypes.HANDLE(-1).value
        handle = kernel32.CreateFileW(
            path, 0x80000000, share, None, open_existing,
            no_buffering | overlapped, None)
        if handle == invalid:
            raise OSError(ctypes.get_last_error(),
                          "CreateFileW overlapped direct failed")
        buffer = kernel32.VirtualAlloc(None, length, 0x1000 | 0x2000, 0x04)
        if not buffer:
            kernel32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(),
                          "VirtualAlloc overlapped direct buffer failed")
        event = kernel32.CreateEventW(None, True, False, None)
        if not event:
            kernel32.VirtualFree(buffer, 0, 0x8000)
            kernel32.CloseHandle(handle)
            raise OSError(ctypes.get_last_error(),
                          "CreateEventW overlapped direct failed")
        self._handle = handle
        self._buffer = buffer
        self._event = event

    def begin_read(self, offset: int, length: int) -> tuple[_OVERLAPPED, bool]:
        if self._handle is None or self._event is None:
            raise OSError("overlapped reader is closed")
        if length != self._length:
            raise OSError("overlapped reader length changed")
        if offset < 0:
            raise OSError("overlapped reader offset is negative")
        if not self._kernel32.ResetEvent(self._event):
            raise OSError(ctypes.get_last_error(),
                          "ResetEvent overlapped direct failed")
        overlapped = self._OVERLAPPED()
        overlapped.Offset = offset & 0xffffffff
        overlapped.OffsetHigh = (offset >> 32) & 0xffffffff
        overlapped.hEvent = self._event
        completed = wintypes.DWORD()
        ok = self._kernel32.ReadFile(
            self._handle, self._buffer, length,
            ctypes.byref(completed), ctypes.byref(overlapped))
        self._pending.append(overlapped)
        if not ok:
            error = ctypes.get_last_error()
            if error != 997:  # ERROR_IO_PENDING
                self._pending.remove(overlapped)
                raise OSError(error, "ReadFile overlapped failed")
            return overlapped, True
        return overlapped, False

    def complete_read(self, operation: _OVERLAPPED, length: int) -> int:
        if self._handle is None:
            raise OSError("overlapped reader is closed")
        completed = wintypes.DWORD()
        if not self._kernel32.GetOverlappedResult(
                self._handle, ctypes.byref(operation),
                ctypes.byref(completed), True):
            raise OSError(ctypes.get_last_error(),
                          "GetOverlappedResult failed")
        if int(completed.value) != length:
            raise OSError(f"short overlapped read: got {completed.value} of {length}")
        self._pending.remove(operation)
        return int(completed.value)

    def read(self, offset: int, length: int) -> int:
        operation, _ = self.begin_read(offset, length)
        return self.complete_read(operation, length)

    def cancel_and_drain(self) -> None:
        if self._handle is None:
            return
        for operation in list(self._pending):
            cancelled = self._kernel32.CancelIoEx(
                self._handle, ctypes.byref(operation))
            cancel_error = ctypes.get_last_error() if not cancelled else 0
            completed = wintypes.DWORD()
            drained = self._kernel32.GetOverlappedResult(
                self._handle, ctypes.byref(operation),
                ctypes.byref(completed), True)
            drain_error = ctypes.get_last_error() if not drained else 0
            # ERROR_OPERATION_ABORTED is the expected terminal result after
            # CancelIoEx.  A successful result is also safe: the request
            # completed before cancellation won the race.  Any other result
            # leaves ownership unresolved, so close() must not free the
            # buffer/event/handle.
            if not drained and drain_error != 995:  # ERROR_OPERATION_ABORTED
                raise OSError(drain_error or cancel_error,
                              "GetOverlappedResult did not drain request")
            self._pending = [item for item in self._pending
                             if item is not operation]

    def close(self) -> None:
        self.cancel_and_drain()
        if self._event is not None:
            self._kernel32.CloseHandle(self._event)
            self._event = None
        if self._handle is not None:
            if self._buffer is not None:
                self._kernel32.VirtualFree(self._buffer, 0, 0x8000)
                self._buffer = None
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def open_read_handle(path: str, mode: str = "buffered",
                     length: int = 0) -> (_BufferedReadHandle
                                          | _DirectReadHandle
                                          | _WindowsOverlappedReadHandle):
    if mode == "buffered":
        return _BufferedReadHandle(path)
    if mode == "direct":
        return _DirectReadHandle(path, length)
    if mode == "overlapped":
        return _WindowsOverlappedReadHandle(path, length)
    raise OSError(f"unsupported I/O mode {mode}")


def direct_backend_name() -> str:
    if os.name == "nt":
        return "win32-no-buffering"
    if sys.platform == "darwin":
        return "macos-f_nocache"
    if sys.platform.startswith("linux"):
        return "linux-o_direct"
    return "unsupported"


def io_backend_name(io_mode: str) -> str:
    if io_mode == "overlapped":
        return "win32-overlapped"
    if io_mode == "direct":
        return direct_backend_name()
    return "python-pread"


def _pcie_payload_bytes_per_s(speed_code: int, width: int) -> int | None:
    """Convert Windows' negotiated PCIe speed code to payload B/s.

    This is link-negotiation evidence, not a traffic benchmark.  The result
    must never be promoted as an observed controller or PCH throughput rate.
    """
    speed_gts = {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0,
                 5: 32.0, 6: 64.0}.get(speed_code)
    return _pcie_payload_bytes_per_s_gts(speed_gts, width, speed_code)


def _pcie_payload_bytes_per_s_gts(speed_gts: float | None, width: int,
                                  speed_code: int | None = None) -> int | None:
    """Convert a negotiated PCIe rate reported as GT/s to payload B/s."""
    if speed_gts is None or width <= 0:
        return None
    efficiency = (0.8 if ((speed_code is not None and speed_code <= 2)
                           or speed_gts <= 5.0)
                  else 128.0 / 130.0)
    return int(speed_gts * 1_000_000_000 * width * efficiency / 8.0)


def probe_platform_topology(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect optional negotiated PCIe endpoint evidence.

    A manifest may bind a planner resource to a platform endpoint using
    ``platform.windows_pci_instance_id``.  The probe is read-only and best
    effort.  It reports negotiated endpoint properties and parent identity,
    but deliberately does not call them a measured PCH/DMI capacity.
    """
    targets: list[dict[str, Any]] = []
    for source in manifest.get("sources", []):
        metadata = source.get("platform")
        if not isinstance(metadata, dict):
            continue
        instance_id = (metadata.get("windows_pci_instance_id")
                       or metadata.get("linux_pci_address")
                       or metadata.get("macos_pci_name"))
        if isinstance(instance_id, str) and instance_id:
            targets.append({
                "resource_id": source.get("controller_id"),
                "resource_kind": "controller",
                "drive_id": source.get("drive_id"),
                "instance_id": instance_id,
            })
    for island in manifest.get("islands", []):
        metadata = island.get("platform")
        if not isinstance(metadata, dict):
            continue
        instance_id = (metadata.get("windows_pci_instance_id")
                       or metadata.get("linux_pci_address")
                       or metadata.get("macos_pci_name"))
        if isinstance(instance_id, str) and instance_id:
            targets.append({
                "resource_id": island.get("resource_id"),
                "resource_kind": island.get("kind"),
                "instance_id": instance_id,
            })
    if not targets:
        return {"status": "UNAVAILABLE", "method": None, "links": [],
                "reason": "manifest has no platform endpoint bindings"}

    if os.name == "nt":
        links = [_probe_windows_pci_target(target) for target in targets]
        observed = any(isinstance(row, dict)
                       and row.get("evidence_status") == "OBSERVED_NEGOTIATED"
                       for row in links)
        ancestor_sets = []
        for row in links:
            if not isinstance(row, dict):
                continue
            chain = row.get("parent_chain", [])
            ancestors = {item.get("instance_id") for item in chain
                         if isinstance(item, dict)
                         and isinstance(item.get("instance_id"), str)}
            parent = row.get("parent_instance_id")
            if isinstance(parent, str) and parent:
                ancestors.add(parent)
            if ancestors:
                ancestor_sets.append(ancestors)
        common_ancestors = (sorted(set.intersection(*ancestor_sets))
                            if ancestor_sets else [])
        return {
            "status": "OBSERVED" if observed else "UNAVAILABLE",
            "method": "windows-pnp",
            "links": links,
            "common_ancestor_instance_ids": common_ancestors,
            "limitations": [
                "negotiated endpoint properties are not a PCH/DMI traffic benchmark",
                "a common OS parent identifies topology only; it does not prove a shared bandwidth limit",
                "power-management state may lower the current link speed",
                "hardware DMA completion is not independently attested",
            ],
        }
    if sys.platform.startswith("linux"):
        links = [_probe_linux_pci_target(target) for target in targets]
        observed = any(isinstance(row, dict)
                       and row.get("evidence_status") == "OBSERVED_NEGOTIATED"
                       for row in links)
        return {
            "status": "OBSERVED" if observed else "UNAVAILABLE",
            "method": "linux-lspci",
            "links": links,
            "limitations": [
                "lspci negotiated endpoint properties are not a PCH/DMI traffic benchmark",
                "power-management state may lower the current link speed",
                "hardware DMA completion is not independently attested",
            ],
        }
    if sys.platform == "darwin":
        links = [_probe_macos_pci_target(target) for target in targets]
        observed = any(isinstance(row, dict)
                       and row.get("evidence_status") == "OBSERVED_NEGOTIATED"
                       for row in links)
        return {
            "status": "OBSERVED" if observed else "UNAVAILABLE",
            "method": "macos-system-profiler",
            "links": links,
            "limitations": [
                "macOS PCI endpoint properties are not a PCH/DMI traffic benchmark",
                "system_profiler may omit negotiated link state for some Apple buses",
                "hardware DMA completion is not independently attested",
            ],
        }
    return {"status": "UNAVAILABLE", "method": sys.platform,
            "links": [], "reason": "unsupported platform"}


def probe_pcie_traffic_telemetry() -> dict[str, Any]:
    """Report whether a supported PCH/DMI traffic counter is available.

    Negotiated PCIe link properties and end-to-end read throughput are useful
    evidence, but neither is a byte counter for the chipset interconnect.  Do
    not silently turn a missing privileged/vendor counter into ``NOT_RUN`` or
    a fabricated capacity.
    """
    candidates = []
    names = ("pcm-pcie", "pcm-pcie.exe", "perf", "powermetrics")
    for name in names:
        path = shutil.which(name)
        if path:
            candidates.append(path)
    if candidates:
        return {
            "status": "AVAILABLE_NOT_RUN",
            "method": "command-capability-scan",
            "candidates": candidates,
            "reason": "a counter tool exists but was not invoked by the bounded profiler",
            "limitations": [
                "a vendor counter still requires a separate permission and unit-validation gate",
                "end-to-end storage throughput is not substituted for PCH/DMI bytes",
            ],
        }
    return {
        "status": "UNAVAILABLE",
        "method": "command-capability-scan",
        "candidates": [],
        "reason": "no supported unprivileged PCH/DMI traffic counter was found",
        "limitations": [
            "endpoint negotiation and shared-upstream throughput remain available as separate evidence",
            "no PCH/DMI capacity is inferred",
        ],
    }


def read_to_ready_evidence(io_mode: str, backend: str) -> dict[str, Any]:
    """Describe the endpoint semantics of the selected read backend."""
    if io_mode not in ("direct", "overlapped"):
        return {
            "status": "NOT_RUN",
            "method": None,
            "reason": "buffered reads do not isolate the storage-to-ready path",
        }
    evidence = {
        "status": "OBSERVED_NOT_ATTESTED",
        "method": ("overlapped_read_into_reusable_aligned_user_buffer"
                    if io_mode == "overlapped"
                    else "direct_read_into_reusable_aligned_user_buffer"),
        "backend": backend,
        "completion": "read operation returned after the destination buffer was populated",
        "limitations": [
            "the profiler does not independently attest the device DMA engine",
            "this is not an active-admission approval",
        ],
    }
    if io_mode == "overlapped":
        evidence["completion"] = (
            "OVERLAPPED ReadFile completed through GetOverlappedResult after "
            "the destination buffer was populated")
    return evidence


def _pnputil_property(text: str, key: str) -> str | None:
    match = re.search(
        rf"(?m)^\s*{re.escape(key)}\s+\[[^\]]+\]:\s*\r?\n\s*(.+?)\s*$",
        text)
    return match.group(1).strip() if match else None


def _pnputil_int(text: str, key: str) -> int | None:
    value = _pnputil_property(text, key)
    if value is None:
        return None
    match = re.search(r"\((\d+)\)", value)
    if match:
        return int(match.group(1))
    match = re.search(r"0x([0-9a-fA-F]+)", value)
    if match:
        return int(match.group(1), 16)
    match = re.search(r"\b(\d+)\b", value)
    return int(match.group(1)) if match else None


def _run_pnputil_properties(instance_id: str) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["pnputil.exe", "/enum-devices", "/instanceid", instance_id,
             "/properties"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    if completed.returncode != 0:
        return None, completed.stderr.strip() or f"pnputil exited {completed.returncode}"
    return completed.stdout, None


def _probe_windows_pci_target(target: dict[str, Any]) -> dict[str, Any]:
    instance_id = str(target["instance_id"])
    output, error = _run_pnputil_properties(instance_id)
    if output is None:
        return {**target, "status": "UNAVAILABLE", "error": error}
    speed_code = _pnputil_int(
        output, "DEVPKEY_PciDevice_CurrentLinkSpeed")
    width = _pnputil_int(
        output, "DEVPKEY_PciDevice_CurrentLinkWidth")
    if speed_code is None or width is None:
        return {**target, "status": "UNAVAILABLE",
                "error": "negotiated PCIe link properties were not reported"}
    payload = _pcie_payload_bytes_per_s(speed_code, width)
    result = {
        **target,
        "friendly_name": _pnputil_property(
            output, "DEVPKEY_Device_DeviceDesc"),
        "status": "PRESENT",
        "speed_code": speed_code,
        "width": width,
        "parent_instance_id": _pnputil_property(
            output, "DEVPKEY_Device_Parent"),
        "location": _pnputil_property(
            output, "DEVPKEY_Device_LocationInfo"),
        "location_path": _pnputil_property(
            output, "DEVPKEY_Device_LocationPaths"),
        "speed_gts": {1: 2.5, 2: 5.0, 3: 8.0, 4: 16.0,
                       5: 32.0, 6: 64.0}.get(speed_code),
        "negotiated_payload_bytes_per_s": payload,
        "evidence_status": ("OBSERVED_NEGOTIATED"
                             if payload is not None else "UNKNOWN"),
    }
    # Walk the OS device tree far enough to expose whether independently
    # negotiated endpoints have a common parent.  This is topology evidence,
    # not a claim that the common parent has a measured capacity.
    chain: list[dict[str, Any]] = []
    current = result.get("parent_instance_id")
    seen = {instance_id}
    for _ in range(8):
        if not isinstance(current, str) or not current or current in seen:
            break
        seen.add(current)
        parent_output, parent_error = _run_pnputil_properties(current)
        if parent_output is None:
            chain.append({"instance_id": current, "status": "UNAVAILABLE",
                          "error": parent_error})
            break
        parent = _pnputil_property(parent_output, "DEVPKEY_Device_Parent")
        chain.append({
            "instance_id": current,
            "status": "PRESENT",
            "friendly_name": _pnputil_property(
                parent_output, "DEVPKEY_Device_DeviceDesc"),
            "parent_instance_id": parent,
            "location": _pnputil_property(
                parent_output, "DEVPKEY_Device_LocationInfo"),
            "location_path": _pnputil_property(
                parent_output, "DEVPKEY_Device_LocationPaths"),
        })
        current = parent
    result["parent_chain"] = chain
    result["root_instance_id"] = (
        chain[-1].get("instance_id") if chain else result.get("parent_instance_id"))
    return result


def _probe_linux_pci_target(target: dict[str, Any]) -> dict[str, Any]:
    """Read negotiated PCIe endpoint state through unprivileged lspci."""
    address = str(target["instance_id"])
    try:
        completed = subprocess.run(
            ["lspci", "-s", address, "-vv"], capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=20,
            check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {**target, "status": "UNAVAILABLE", "error": str(exc)}
    if completed.returncode != 0:
        return {**target, "status": "UNAVAILABLE",
                "error": completed.stderr.strip() or
                f"lspci exited {completed.returncode}"}
    # lspci prints both LnkCap (capability) and LnkSta (negotiated state).
    # Only parse LnkSta; using the first generic Speed/Width match can
    # silently report a theoretical maximum instead of the live endpoint.
    status_lines = re.findall(r"(?im)^\s*LnkSta:\s*(.+)$",
                              completed.stdout)
    negotiated = " ".join(status_lines)
    speed_match = re.search(
        r"\bSpeed\s+([0-9]+(?:\.[0-9]+)?)GT/s", negotiated,
        re.IGNORECASE)
    width_match = re.search(r"\bWidth\s+x([0-9]+)", negotiated,
                            re.IGNORECASE)
    if speed_match is None or width_match is None:
        return {**target, "status": "UNAVAILABLE",
                "error": "negotiated PCIe link properties were not reported"}
    speed_gts = float(speed_match.group(1))
    width = int(width_match.group(1))
    payload = _pcie_payload_bytes_per_s_gts(speed_gts, width)
    result = {
        **target,
        "status": "PRESENT",
        "speed_gts": speed_gts,
        "width": width,
        "negotiated_payload_bytes_per_s": payload,
        "evidence_status": ("OBSERVED_NEGOTIATED"
                             if payload is not None else "UNKNOWN"),
    }
    return result


def _probe_macos_pci_target(target: dict[str, Any]) -> dict[str, Any]:
    """Best-effort macOS endpoint probe using system_profiler JSON.

    Apple system-profiler schemas differ between Intel and Apple Silicon.  A
    binding may provide ``macos_pci_name``; if the platform omits link state,
    the result remains explicitly unavailable instead of guessing from a bus
    generation or device name.
    """
    name = str(target["instance_id"])
    try:
        completed = subprocess.run(
            ["system_profiler", "SPPCIDataType", "-json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {**target, "status": "UNAVAILABLE", "error": str(exc)}
    if completed.returncode != 0:
        return {**target, "status": "UNAVAILABLE",
                "error": completed.stderr.strip() or
                f"system_profiler exited {completed.returncode}"}
    try:
        tree = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {**target, "status": "UNAVAILABLE",
                "error": f"invalid system_profiler JSON: {exc}"}

    def walk(value: Any) -> dict[str, Any] | None:
        if isinstance(value, dict):
            values = " ".join(str(value.get(key, ""))
                               for key in ("_name", "name", "slot_name",
                                           "pci_device", "device_name"))
            if name.casefold() in values.casefold():
                return value
            for child in value.values():
                found = walk(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = walk(child)
                if found is not None:
                    return found
        return None

    row = walk(tree)
    if row is None:
        return {**target, "status": "UNAVAILABLE",
                "error": "bound PCI device was not found in system_profiler"}
    speed_value = next((row.get(key) for key in
                        ("link_speed", "link_speed_gts", "current_link_speed")
                        if row.get(key) is not None), None)
    width_value = next((row.get(key) for key in
                        ("link_width", "current_link_width")
                        if row.get(key) is not None), None)
    speed_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*GT/s",
                            str(speed_value), re.IGNORECASE)
    width_match = re.search(r"x\s*([0-9]+)", str(width_value), re.IGNORECASE)
    if speed_match is None or width_match is None:
        return {**target, "status": "UNAVAILABLE",
                "error": "system_profiler omitted negotiated link state"}
    speed_gts = float(speed_match.group(1))
    width = int(width_match.group(1))
    payload = _pcie_payload_bytes_per_s_gts(speed_gts, width)
    return {
        **target,
        "status": "PRESENT",
        "speed_gts": speed_gts,
        "width": width,
        "negotiated_payload_bytes_per_s": payload,
        "evidence_status": ("OBSERVED_NEGOTIATED"
                             if payload is not None else "UNKNOWN"),
    }


def measure(group: dict[str, Any], sources: dict[int, dict[str, Any]],
            read_bytes: int, inflight: int, samples: int,
            allow_range_reuse: bool = False,
            cache_condition: str = "unknown",
            io_mode: str = "buffered",
            range_base_slots: dict[int, int] | None = None,
            operations_per_worker: int = 1) -> dict[str, Any]:
    drive_ids = [int(value) for value in group["drive_ids"]]
    for drive_id in drive_ids:
        if drive_id not in sources:
            fail(f"group {group['name']} refers to unknown drive {drive_id}")
    if not 1 <= operations_per_worker <= 32:
        fail("measure operations-per-worker must be in 1..32")
    slot_rank: list[int] = []
    slots_per_drive = {drive_id: 0 for drive_id in drive_ids}
    for slot in range(inflight):
        drive_id = drive_ids[slot % len(drive_ids)]
        slot_rank.append(slots_per_drive[drive_id])
        slots_per_drive[drive_id] += 1
    range_reuse_observed = False
    sample_rows: list[dict[str, Any]] = []
    for sample in range(samples):
        offsets: list[list[tuple[str, int]]] = []
        for slot in range(inflight):
            drive_id = drive_ids[slot % len(drive_ids)]
            source = sources[drive_id]
            slot_offsets: list[tuple[str, int]] = []
            for operation in range(operations_per_worker):
                slot_index = ((range_base_slots or {}).get(drive_id, 0)
                              + (sample * slots_per_drive[drive_id]
                                 + slot_rank[slot]) * operations_per_worker
                              + operation)
                available_slots = _source_available_slots(source, read_bytes)
                if slot_index >= available_slots:
                    if not allow_range_reuse:
                        fail(f"{group['name']} sample {sample}: drive {drive_id} "
                             "lacks a distinct range; pass --allow-range-reuse "
                             "only when cache condition remains explicitly unknown")
                    range_reuse_observed = True
                    slot_index %= available_slots
                slot_offsets.append(_source_target(source, slot_index, read_bytes))
            offsets.append(slot_offsets)
        handles: list[dict[str, _BufferedReadHandle | _DirectReadHandle
                           | _WindowsOverlappedReadHandle]] = []
        try:
            for slot_offsets in offsets:
                by_path: dict[str, _BufferedReadHandle | _DirectReadHandle
                              | _WindowsOverlappedReadHandle] = {}
                for path, _ in slot_offsets:
                    if path not in by_path:
                        by_path[path] = open_read_handle(
                            path, io_mode, read_bytes)
                handles.append(by_path)
        except OSError as exc:
            for by_path in handles:
                for handle in by_path.values():
                    handle.close()
            fail(f"{group['name']} could not open read handles: {exc}")
        start_ns = [0]

        def mark_start() -> None:
            start_ns[0] = time.perf_counter_ns()

        barrier = threading.Barrier(inflight, action=mark_start)
        submit_barrier = (threading.Barrier(inflight)
                          if io_mode == "overlapped" else None)
        errors: list[str] = []
        durations: list[list[int]] = [[] for _ in range(inflight)]
        worker_elapsed = [0] * inflight
        pending = [False] * inflight
        submitted_ns = [0] * inflight

        def worker(slot: int) -> None:
            drive_id = drive_ids[slot % len(drive_ids)]
            try:
                if cache_condition == "warm":
                    for path, offset in offsets[slot]:
                        handles[slot][path].read(offset, read_bytes)
                barrier.wait()
                for operation_index, (path, offset) in enumerate(offsets[slot]):
                    request_start = time.perf_counter_ns()
                    handle = handles[slot][path]
                    if operation_index == 0 and io_mode == "overlapped":
                        async_operation, pending[slot] = handle.begin_read(
                            offset, read_bytes)
                        submitted_ns[slot] = time.perf_counter_ns()
                        assert submit_barrier is not None
                        submit_barrier.wait()
                        handle.complete_read(async_operation, read_bytes)
                    elif operation_index == 0:
                        # Preserve the synchronous fan-out barrier for the
                        # legacy backends before the first request.
                        barrier.wait()
                        handle.read(offset, read_bytes)
                    else:
                        handle.read(offset, read_bytes)
                    durations[slot].append(
                        time.perf_counter_ns() - request_start)
                worker_elapsed[slot] = time.perf_counter_ns() - start_ns[0]
            except (OSError, threading.BrokenBarrierError) as exc:
                errors.append(f"drive {drive_id}: {exc}")
                barrier.abort()
                if submit_barrier is not None:
                    submit_barrier.abort()

        threads = [threading.Thread(target=worker, args=(slot,))
                   for slot in range(inflight)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for by_path in handles:
            for handle in by_path.values():
                handle.close()
        if errors:
            fail(f"{group['name']} sample {sample}: {'; '.join(errors)}")
        if any(not worker_durations for worker_durations in durations):
            fail(f"{group['name']} sample {sample}: missing worker timing")
        wall_ns = max(worker_elapsed)
        request_count = inflight * operations_per_worker
        latencies = [duration for worker_durations in durations
                     for duration in worker_durations]
        sample_row = {
            "sample": sample,
            "bytes": read_bytes * request_count,
            "wall_ns": wall_ns,
            "aggregate_bytes_per_s": (read_bytes * request_count * 1_000_000_000)
            // wall_ns,
            "latency_ns_p50": percentile(latencies, 0.50),
            "latency_ns_p95": percentile(latencies, 0.95),
            "latency_ns_p99": percentile(latencies, 0.99),
            "latency_ns_max": max(latencies),
            "worker_latency_ns": latencies,
        }
        if io_mode == "overlapped":
            sample_row["overlap_evidence"] = {
                "protocol": "submit_first_wave_before_reap",
                "submitted_requests": inflight,
                "total_requests": request_count,
                "pending_submissions": sum(pending),
                "submission_skew_ns": max(submitted_ns) - min(submitted_ns),
                "replenishment_operations": max(0, operations_per_worker - 1),
                "completion_reap_gate": True,
                "hardware_queue_depth_attested": False,
            }
        sample_rows.append(sample_row)
    aggregate = [int(row["aggregate_bytes_per_s"]) for row in sample_rows]
    latencies = [int(value) for row in sample_rows for value in row["worker_latency_ns"]]
    active_drive_ids = sorted({
        drive_ids[slot % len(drive_ids)] for slot in range(inflight)
    })
    return {
        "group": group["name"],
        "drive_ids": drive_ids,
        "target_resource_id": group.get("resource_id"),
        "target_kind": group.get("kind"),
        "active_drive_ids": active_drive_ids,
        "all_group_drives_active": len(active_drive_ids) == len(set(drive_ids)),
        "inflight": inflight,
        "operations_per_worker": operations_per_worker,
        "sample_count": samples,
        "range_reuse": range_reuse_observed,
        "range_reuse_allowed": bool(allow_range_reuse),
        "samples": sample_rows,
        "summary": {
            "aggregate_bytes_per_s_p50": percentile(aggregate, 0.50),
            "aggregate_bytes_per_s_p95": percentile(aggregate, 0.95),
            "latency_ns_p50": percentile(latencies, 0.50),
            "latency_ns_p95": percentile(latencies, 0.95),
            "latency_ns_p99": percentile(latencies, 0.99),
            "latency_ns_max": max(latencies),
        },
    }


def matrix_cells(drive_ids: list[int], levels: list[int],
                 max_total_inflight: int) -> list[dict[int, int]]:
    """Return a bounded Cartesian product of per-drive queue depths.

    Zero is allowed for individual drives so that the matrix contains useful
    single-drive controls.  The all-zero cell is omitted, and the total is
    bounded separately from each drive's queue depth.
    """
    if not drive_ids or len(set(drive_ids)) != len(drive_ids):
        fail("matrix drive_ids must be non-empty and unique")
    if (not levels or any(not isinstance(level, int) or level < 0 or level > 16
                          for level in levels)
            or len(set(levels)) != len(levels)):
        fail("matrix queue-depth levels must be unique integers in 0..16")
    if not isinstance(max_total_inflight, int) or not 1 <= max_total_inflight <= 16:
        fail("matrix max-total-inflight must be in 1..16")
    if len(levels) ** len(drive_ids) > 4096:
        fail("matrix has more than 4096 possible cells")
    cells = []
    for values in itertools.product(levels, repeat=len(drive_ids)):
        total = sum(values)
        if total == 0 or total > max_total_inflight:
            continue
        cells.append({drive_id: int(qd)
                      for drive_id, qd in zip(drive_ids, values)})
    if not cells:
        fail("matrix levels have no nonzero cell within max-total-inflight")
    cells.sort(key=lambda cell: (sum(cell.values()),
                                 tuple(cell[drive_id] for drive_id in drive_ids)))
    return cells


def plan_range_slots(manifest: dict[str, Any], inflight: list[int],
                     matrix_levels: list[int], samples: int,
                     max_total_inflight: int,
                     operations_per_worker: int) -> dict[int, int]:
    """Calculate capture-wide per-drive slots before any I/O starts."""
    required: dict[int, int] = {
        int(source["drive_id"]): 0 for source in manifest["sources"]}
    for group in manifest["groups"]:
        drive_ids = [int(value) for value in group["drive_ids"]]
        sweep = group.get("sweep", "matrix" if len(drive_ids) > 1 else "scalar")
        if sweep == "matrix":
            cells = matrix_cells(drive_ids, matrix_levels, max_total_inflight)
            for drive_id in drive_ids:
                required[drive_id] += (
                    sum(cell[drive_id] for cell in cells)
                    * samples * operations_per_worker)
        else:
            for level in inflight:
                for slot in range(level):
                    drive_id = drive_ids[slot % len(drive_ids)]
                    required[drive_id] += samples * operations_per_worker
    return required


def measure_matrix(group: dict[str, Any], sources: dict[int, dict[str, Any]],
                   read_bytes: int, levels: list[int], samples: int,
                   max_total_inflight: int,
                   allow_range_reuse: bool = False,
                   operations_per_worker: int = 1,
                   cache_condition: str = "unknown",
                   io_mode: str = "buffered",
                   range_base_slots: dict[int, int] | None = None) -> list[dict[str, Any]]:
    """Measure a shared group over independent per-drive queue depths.

    Each cell starts all of its workers together and times through the last
    completion.  The result remains raw evidence; no admission limit is
    inferred here.
    """
    drive_ids = [int(value) for value in group["drive_ids"]]
    for drive_id in drive_ids:
        if drive_id not in sources:
            fail(f"group {group['name']} refers to unknown drive {drive_id}")
    if not 1 <= operations_per_worker <= 32:
        fail("matrix operations-per-worker must be in 1..32")
    cells = matrix_cells(drive_ids, levels, max_total_inflight)
    # Allocate a compact, disjoint virtual slot range per drive.  Reserving
    # max_qd for every cell leaves gaps for cells with smaller queue depth and
    # can falsely force range reuse even when the source pool has enough bytes
    # for all actual requests.
    cell_slot_bases: dict[int, list[int]] = {drive_id: [] for drive_id in drive_ids}
    next_slot = {drive_id: (range_base_slots or {}).get(drive_id, 0)
                 for drive_id in drive_ids}
    for cell in cells:
        for drive_id in drive_ids:
            cell_slot_bases[drive_id].append(next_slot[drive_id])
            next_slot[drive_id] += (cell[drive_id] * samples
                                     * operations_per_worker)
    results: list[dict[str, Any]] = []
    for cell_index, queue_depths in enumerate(cells):
        total_inflight = sum(queue_depths.values())
        sample_rows: list[dict[str, Any]] = []
        range_reuse_observed = False
        for sample in range(samples):
            slots: list[tuple[int, int]] = []
            for drive_id in drive_ids:
                slots.extend((drive_id, local_slot)
                             for local_slot in range(queue_depths[drive_id]))
            if len(slots) != total_inflight or not slots:
                fail(f"{group['name']} matrix cell has invalid worker count")
            offsets: list[list[tuple[str, int]]] = []
            for drive_id, local_slot in slots:
                source = sources[drive_id]
                available_slots = _source_available_slots(source, read_bytes)
                slot_offsets: list[tuple[str, int]] = []
                for operation in range(operations_per_worker):
                    slot_index = (cell_slot_bases[drive_id][cell_index]
                                  + (sample * queue_depths[drive_id]
                                     + local_slot) * operations_per_worker
                                  + operation)
                    if slot_index >= available_slots:
                        if not allow_range_reuse:
                            fail(f"{group['name']} cell {queue_depths} "
                                 f"sample {sample}: drive {drive_id} lacks "
                                 "a distinct matrix range")
                        range_reuse_observed = True
                        slot_index %= available_slots
                    slot_offsets.append(_source_target(
                        source, slot_index, read_bytes))
                offsets.append(slot_offsets)
            handles: list[dict[str, _BufferedReadHandle | _DirectReadHandle
                                | _WindowsOverlappedReadHandle]] = []
            try:
                for slot_offsets in offsets:
                    by_path: dict[str, _BufferedReadHandle | _DirectReadHandle
                                  | _WindowsOverlappedReadHandle] = {}
                    for path, _ in slot_offsets:
                        if path not in by_path:
                            by_path[path] = open_read_handle(
                                path, io_mode, read_bytes)
                    handles.append(by_path)
            except OSError as exc:
                for by_path in handles:
                    for handle in by_path.values():
                        handle.close()
                fail(f"{group['name']} could not open sustained read handles: {exc}")
            start_ns = [0]

            def mark_start() -> None:
                start_ns[0] = time.perf_counter_ns()

            barrier = threading.Barrier(total_inflight, action=mark_start)
            submit_barrier = (threading.Barrier(total_inflight)
                              if io_mode == "overlapped" else None)
            errors: list[str] = []
            durations = [[] for _ in range(total_inflight)]
            worker_elapsed = [0] * total_inflight
            pending = [False] * total_inflight
            submitted_ns = [0] * total_inflight

            def worker(slot: int) -> None:
                drive_id, _ = slots[slot]
                try:
                    if cache_condition == "warm":
                        for path, offset in offsets[slot]:
                            handles[slot][path].read(offset, read_bytes)
                    barrier.wait()
                    for operation in range(operations_per_worker):
                        request_start = time.perf_counter_ns()
                        path, offset = offsets[slot][operation]
                        handle = handles[slot][path]
                        if operation == 0 and io_mode == "overlapped":
                            async_operation, pending[slot] = handle.begin_read(
                                offset, read_bytes)
                            submitted_ns[slot] = time.perf_counter_ns()
                            assert submit_barrier is not None
                            submit_barrier.wait()
                            handle.complete_read(async_operation, read_bytes)
                        else:
                            handle.read(offset, read_bytes)
                        durations[slot].append(
                            time.perf_counter_ns() - request_start)
                    worker_elapsed[slot] = time.perf_counter_ns() - start_ns[0]
                except (OSError, threading.BrokenBarrierError) as exc:
                    errors.append(f"drive {drive_id}: {exc}")
                    barrier.abort()
                    if submit_barrier is not None:
                        submit_barrier.abort()

            threads = [threading.Thread(target=worker, args=(slot,))
                       for slot in range(total_inflight)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for by_path in handles:
                for handle in by_path.values():
                    handle.close()
            if errors:
                fail(f"{group['name']} cell {queue_depths} sample {sample}: "
                     f"{'; '.join(errors)}")
            if any(not worker_durations for worker_durations in durations):
                fail(f"{group['name']} cell {queue_depths} sample {sample}: "
                     "missing worker timing")
            wall_ns = max(worker_elapsed)
            request_count = total_inflight * operations_per_worker
            aggregate_rate = (read_bytes * request_count * 1_000_000_000) // wall_ns
            per_drive: dict[str, dict[str, Any]] = {}
            drive_rates: list[int] = []
            all_latencies: list[int] = []
            for drive_id in drive_ids:
                drive_durations = [duration for (slot, (owner, _))
                                   in enumerate(slots) if owner == drive_id
                                   for duration in durations[slot]]
                if not drive_durations:
                    continue
                drive_rate = (read_bytes * len(drive_durations)
                              * 1_000_000_000) // wall_ns
                drive_rates.append(drive_rate)
                all_latencies.extend(drive_durations)
                per_drive[str(drive_id)] = {
                    "inflight": queue_depths[drive_id],
                    "request_count": len(drive_durations),
                    "bytes": read_bytes * len(drive_durations),
                    "aggregate_bytes_per_s": drive_rate,
                    "latency_ns_p50": percentile(drive_durations, 0.50),
                    "latency_ns_p95": percentile(drive_durations, 0.95),
                    "latency_ns_p99": percentile(drive_durations, 0.99),
                    "latency_ns_max": max(drive_durations),
                    "latency_ns": drive_durations,
                }
            throughput_share_jain = 0.0
            if drive_rates:
                rate_sum = sum(drive_rates)
                throughput_share_jain = (rate_sum * rate_sum) / (
                    len(drive_rates) * sum(rate * rate for rate in drive_rates))
            sample_row = {
                "sample": sample,
                "bytes": read_bytes * request_count,
                "wall_ns": wall_ns,
                "aggregate_bytes_per_s": aggregate_rate,
                "latency_ns_p50": percentile(all_latencies, 0.50),
                "latency_ns_p95": percentile(all_latencies, 0.95),
                "latency_ns_p99": percentile(all_latencies, 0.99),
                "latency_ns_max": max(all_latencies),
                "throughput_share_jain": throughput_share_jain,
                "per_drive": per_drive,
                "worker_latency_ns": [duration for worker_durations in durations
                                      for duration in worker_durations],
            }
            if io_mode == "overlapped":
                sample_row["overlap_evidence"] = {
                    "protocol": "submit_first_wave_before_reap",
                    "submitted_requests": total_inflight,
                    "total_requests": request_count,
                    "pending_submissions": sum(pending),
                    "submission_skew_ns": max(submitted_ns) - min(submitted_ns),
                    "replenishment_operations": max(0, operations_per_worker - 1),
                    "completion_reap_gate": True,
                    "hardware_queue_depth_attested": False,
                }
            sample_rows.append(sample_row)
        aggregate = [int(row["aggregate_bytes_per_s"]) for row in sample_rows]
        latencies = [int(value) for row in sample_rows
                     for value in row["worker_latency_ns"]]
        per_drive_summary: dict[str, dict[str, Any]] = {}
        for drive_id in drive_ids:
            rows = [row["per_drive"][str(drive_id)] for row in sample_rows
                    if str(drive_id) in row["per_drive"]]
            if not rows:
                continue
            raw_latencies = [int(value)
                             for row in sample_rows
                             for value in row["per_drive"][str(drive_id)]
                             .get("latency_ns", [])]
            per_drive_summary[str(drive_id)] = {
                "inflight": queue_depths[drive_id],
                "request_count": sum(int(row.get("request_count", 0))
                                     for row in rows),
                "aggregate_bytes_per_s_p50": percentile(
                    [int(row["aggregate_bytes_per_s"]) for row in rows], 0.50),
                "latency_ns_p50": percentile(raw_latencies, 0.50),
                "latency_ns_p95": percentile(raw_latencies, 0.95),
                "latency_ns_p99": percentile(raw_latencies, 0.99),
                "latency_ns_max": max(raw_latencies),
            }
        results.append({
            "measurement_kind": "contention_matrix",
            "group": group["name"],
            "drive_ids": drive_ids,
            "target_resource_id": group.get("resource_id"),
            "target_kind": group.get("kind"),
            "queue_depths": {str(drive_id): queue_depths[drive_id]
                             for drive_id in drive_ids},
            "active_drive_ids": sorted(drive_id for drive_id in drive_ids
                                        if queue_depths[drive_id] > 0),
            "all_group_drives_active": all(queue_depths[drive_id] > 0
                                            for drive_id in drive_ids),
            "inflight": total_inflight,
            "operations_per_worker": operations_per_worker,
            "sample_count": samples,
            "range_reuse": range_reuse_observed,
            "range_reuse_allowed": bool(allow_range_reuse),
            "samples": sample_rows,
            "summary": {
                "aggregate_bytes_per_s_p50": percentile(aggregate, 0.50),
                "aggregate_bytes_per_s_p95": percentile(aggregate, 0.95),
                "latency_ns_p50": percentile(latencies, 0.50),
                "latency_ns_p95": percentile(latencies, 0.95),
                "latency_ns_p99": percentile(latencies, 0.99),
                "latency_ns_max": max(latencies),
                "throughput_share_jain_p50": percentile(
                    [int(row["throughput_share_jain"] * 1_000_000)
                     for row in sample_rows],
                    0.50) / 1_000_000,
                "per_drive": per_drive_summary,
            },
        })
    return results


# m3_shd_size_class_t index for the planner profile row's size-class field.
# The plan defines TINY<=64 KiB, SMALL<=1 MiB, MEDIUM<=16 MiB, LARGE>16 MiB.
_SIZE_CLASS_LARGE = 3  # M3_SHD_SIZE_LARGE; M3 expert tensors fall here


def _row_classify_size(read_bytes: int) -> int:
    if read_bytes <= 64 * 1024:
        return 0  # TINY
    if read_bytes <= 1024 * 1024:
        return 1  # SMALL
    if read_bytes <= 16 * 1024 * 1024:
        return 2  # MEDIUM
    return 3      # LARGE


def _row_classify_load(inflight: int) -> int:
    """Map an observed level to the planner's bounded load classes."""
    if inflight <= 1:
        return 0  # FREE
    if inflight == 2:
        return 1  # LOW
    if inflight <= 4:
        return 2  # MID
    if inflight <= 7:
        return 4  # SAT
    return 3      # HIGH


def measure_memory_copy(read_bytes: int, samples: int) -> dict[str, Any]:
    """Measure host DRAM copy bandwidth without involving the filesystem.

    This is intentionally labelled ``host_memcpy``: it is evidence for the
    host memory domain, not proof of a particular storage controller's DMA
    path.  The buffers are touched before timing so page fault cost is not
    silently reported as steady-state bandwidth.
    """
    if read_bytes <= 0 or read_bytes % 4096:
        fail("--memory-copy-bytes must be a positive 4096-byte multiple")
    if samples < 1 or samples > 64:
        fail("--memory-samples must be in 1..64")
    try:
        source = bytearray(read_bytes)
        target = bytearray(read_bytes)
    except MemoryError as exc:
        fail(f"cannot allocate memory-copy buffers: {exc}")
    # Prefault every page before timing.  Endpoint touches alone would make
    # the first copy measure page faults rather than DRAM service bandwidth.
    for index in range(0, read_bytes, 4096):
        source[index] = 1
        target[index] = 3
    source[-1] = 2
    target[-1] = 4
    durations: list[int] = []
    for _ in range(samples):
        t0 = time.perf_counter_ns()
        target[:] = source
        elapsed = time.perf_counter_ns() - t0
        if elapsed <= 0:
            fail("memory-copy timer returned a non-positive duration")
        if target[0] != 1 or target[-1] != 2:
            fail("memory-copy verification failed")
        durations.append(elapsed)
    rates = [(read_bytes * 1_000_000_000) // elapsed
             for elapsed in durations]
    return {
        "kind": "memory_dom",
        "method": "host_memcpy",
        "bytes": read_bytes,
        "samples": samples,
        "duration_ns_p50": percentile(durations, 0.50),
        "duration_ns_p95": percentile(durations, 0.95),
        "bytes_per_s_p50": percentile(rates, 0.50),
        "bytes_per_s_p95": percentile(rates, 0.95),
        "units": {"bytes": "B", "duration": "ns", "rate": "B/s"},
    }


def _format_island_row(island: dict[str, Any]) -> str | None:
    """Format a single planner text row for a non-storage island.

    Returns None when the island is not a recognized kind, or when a required
    topology field is missing.  Unknown capacity is represented by a zero
    rate; zero is the planner's explicit UNKNOWN-rate sentinel.
    """
    kind = island.get("kind")
    rid = int(island["resource_id"])
    if kind == "cpu_island":
        parent_mem = island.get("parent_mem_id")
        isa_bits = island.get("isa_bits")
        cores = island.get("cores")
        eligible = island.get("eligible")
        if not (isinstance(parent_mem, int) and parent_mem
                and isinstance(isa_bits, int)
                and isinstance(cores, int) and cores
                and isinstance(eligible, int) and eligible in (0, 1)):
            return None
        return (f"cpu_island {rid} {parent_mem} {isa_bits} {cores} {eligible}\n")
    if kind == "gpu_island":
        parent_link = island.get("parent_link_id")
        vram = island.get("vram_bytes")
        p2p = island.get("p2p_class", 0)
        pcie_parent = island.get("pcie_parent_id", 0)
        eligible = island.get("eligible")
        if not (isinstance(parent_link, int) and parent_link
                and isinstance(vram, int) and vram
                and isinstance(eligible, int) and eligible in (0, 1)):
            return None
        return (f"gpu_island {rid} {parent_link} {vram} {p2p} {pcie_parent} "
                f"{eligible}\n")
    if kind == "memory_dom":
        parent_id = island.get("parent_id", 0)
        rate = island.get("rate_bytes_per_s", 0)
        bw_class = island.get("bandwidth_class", 0)
        if not isinstance(rate, int) or rate < 0:
            return None
        return (f"memory_dom {rid} {parent_id} {rate} {bw_class}\n")
    if kind == "link":
        parent_cpu = island.get("parent_cpu_id", 0)
        pcie_parent = island.get("pcie_parent_id", 0)
        rate = island.get("rate_bytes_per_s", 0)
        if not isinstance(rate, int) or rate < 0:
            return None
        return (f"link {rid} {parent_cpu} {pcie_parent} {rate}\n")
    return None


def emit_planner_text(capture: dict[str, Any]) -> str:
    """Render a planner text dump alongside the JSON capture.

    Emit topology only. Raw observations remain JSON evidence and are never
    converted into planner profile rows here; reviewed promotion is a separate
    operation.  Islands declared in the manifest remain parser-visible even
    when their rate is zero (UNKNOWN).
    """
    lines: list[str] = []
    lines.append("# planner text dump -- m3-shadow-calibration-v2 emitter")
    lines.append(f"# topology_hash = {capture['topology_hash']}")
    lines.append(f"# topology_hash_algorithm = "
                 f"{capture.get('topology_hash_algorithm', 'legacy')}")
    lines.append(f"# topology_hash_verified = "
                 f"{capture.get('topology_hash_verified', False)}")
    lines.append(f"# manifest_sha256 = {capture.get('manifest_sha256', 'UNKNOWN')}")
    lines.append(f"# capture_sha256 = {capture.get('capture_sha256', 'UNKNOWN')}")
    lines.append(f"# capture_schema = {capture.get('schema', 'UNKNOWN')}")
    tool = capture.get("tool", {})
    if isinstance(tool, dict):
        lines.append(f"# calibrator_tool = {tool.get('name', 'UNKNOWN')}"
                     f"/{tool.get('version', 'UNKNOWN')}")
        lines.append(f"# calibrator_source_sha256 = "
                     f"{tool.get('source_sha256', 'UNKNOWN')}")
    quantiles = capture.get("quantiles", {})
    if isinstance(quantiles, dict):
        lines.append(f"# quantiles = {quantiles.get('method', 'UNKNOWN')}"
                     f"/{quantiles.get('population', 'UNKNOWN')}")
    lines.append(f"# cache_condition = {capture['host']['cache_condition']}")
    lines.append(f"# io_mode = {capture['host'].get('io_mode', 'buffered')}")
    lines.append(f"# io_backend = {capture['host'].get('backend', 'UNKNOWN')}")
    lines.append(f"# asynchronous = {capture['host'].get('asynchronous', False)}")
    if capture['host'].get('asynchronous'):
        lines.append("# async_request_protocol = "
                     f"{capture['host'].get('async_request_protocol', 'UNKNOWN')}")
    traffic_evidence = capture['host'].get('pcie_traffic_evidence', {})
    if isinstance(traffic_evidence, dict):
        lines.append("# pcie_traffic_status = "
                     f"{traffic_evidence.get('status', 'UNKNOWN')}")
        lines.append("# pcie_traffic_policy = endpoint_and_shared_path_only")
    platform_topology = capture['host'].get('platform_topology', {})
    if isinstance(platform_topology, dict):
        lines.append(f"# platform_topology_status = {platform_topology.get('status', 'UNKNOWN')}")
        lines.append(f"# platform_topology_method = {platform_topology.get('method', 'UNKNOWN')}")
        links = platform_topology.get('links', [])
        if isinstance(links, list) and links:
            for link in links:
                if not isinstance(link, dict):
                    continue
                lines.append(
                    "# pcie_negotiated_link = "
                    f"resource={link.get('resource_id', 'UNKNOWN')} "
                    f"drive={link.get('drive_id', 'UNKNOWN')} "
                    f"speed_gts={link.get('speed_gts', 'UNKNOWN')} "
                    f"width={link.get('width', 'UNKNOWN')} "
                    f"payload_bps={link.get('negotiated_payload_bytes_per_s', 'UNKNOWN')} "
                    f"status={link.get('evidence_status', 'UNKNOWN')}")
                chain = link.get("parent_chain", [])
                if isinstance(chain, list) and chain:
                    lines.append(
                        "# pcie_parent_chain = "
                        f"resource={link.get('resource_id', 'UNKNOWN')} "
                        f"depth={len(chain)} "
                        f"root={link.get('root_instance_id', 'UNKNOWN')}")
        common_ancestors = platform_topology.get(
            "common_ancestor_instance_ids", [])
        if isinstance(common_ancestors, list) and common_ancestors:
            lines.append(
                "# pcie_common_ancestor = "
                f"{common_ancestors[0]}")
        lines.append("# unavailable_resource_policy = conservative_unknown")
        lines.append("# unavailable_resource_fallback = storage_bounds.max_inflight; rate=UNKNOWN")
    lines.append(f"# range_reuse = {capture['host'].get('range_reuse', False)}")
    lines.append(f"# range_reuse_allowed = "
                 f"{capture['host'].get('range_reuse_allowed', False)}")
    lines.append(f"# range_reuse_scope = {capture['host'].get('range_reuse_scope', 'UNKNOWN')}")
    allocation = capture.get("range_allocation")
    if isinstance(allocation, dict):
        lines.append("# range_slots_required = " + json.dumps(
            allocation.get("slots_required", {}), sort_keys=True,
            separators=(",", ":")))
        lines.append("# range_slots_available = " + json.dumps(
            allocation.get("slots_available", {}), sort_keys=True,
            separators=(",", ":")))
        lines.append("# range_slots_reused = " + json.dumps(
            allocation.get("slots_reused", {}), sort_keys=True,
            separators=(",", ":")))
    lines.append("# configuration_status = topology_only")
    lines.append("# profile_rows = 0; raw measurements require separate review")
    memory_capture = capture.get("memory")
    memory_method = (memory_capture.get("method", "UNKNOWN")
                     if isinstance(memory_capture, dict) else "NOT_RUN")
    lines.append(f"# host_memory_method = {memory_method}")
    if isinstance(memory_capture, dict):
        lines.append(
            "# host_memory_evidence = "
            f"method={memory_method} "
            f"p50_bps={memory_capture.get('bytes_per_s_p50', 'UNKNOWN')} "
            f"p95_bps={memory_capture.get('bytes_per_s_p95', 'UNKNOWN')} "
            "status=EVIDENCE_ONLY")
    lines.append(f"# physical_traffic_note: {capture['host']['physical_traffic_note']}")
    # Storage-kind declarations: one resource row per drive, controller,
    # upstream.  These are topology declarations only.  A resource receives
    # a profile row only when its own measurement group produced one; missing
    # controller/upstream measurements must remain UNKNOWN in the planner.

    manifest = capture.get("manifest", {})
    islands = manifest.get("islands", [])
    bounds = manifest.get("storage_bounds")
    if not isinstance(bounds, dict):
        fail("planner emission requires explicit manifest storage_bounds")
    max_inflight = bounds.get("max_inflight")
    max_bytes = bounds.get("max_bytes")
    if (not isinstance(max_inflight, int) or not 1 <= max_inflight <= 16
            or not isinstance(max_bytes, int) or max_bytes <= 0):
        fail("manifest storage_bounds must contain positive bounded values")
    seen_resource_ids: set[int] = set()
    def emit_resource(rid: int, kind: int, parent: int) -> None:
        if not rid or rid in seen_resource_ids:
            return
        seen_resource_ids.add(rid)
        lines.append(f"resource {rid} {kind} {parent} {max_inflight} {max_bytes}")
    for source in capture.get("sources", []):
        drive_id = int(source["drive_id"])
        controller_id = source.get("controller_id")
        upstream_id = source.get("upstream_id")
        emit_resource(drive_id, 0, controller_id or 0)
        emit_resource(controller_id or 0, 1, upstream_id or 0)
        emit_resource(upstream_id or 0, 2, 0)
    # Island rows, including zero-rate topology-only nodes.
    for island in islands:
        row = _format_island_row(island)
        if row is not None:
            lines.append(row.rstrip("\n"))
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--emit-planner-text", type=Path, default=None,
                    help="Also write a planner-text dump to this path")
    ap.add_argument("--bytes", type=int, default=16 * 1024 * 1024)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--inflight", default="1,2,4,8,16")
    ap.add_argument("--matrix-inflight", default=None,
                    help="per-drive queue-depth levels for multi-drive groups; "
                         "defaults to 0 plus --inflight levels")
    ap.add_argument("--max-total-inflight", type=int, default=16,
                    help="maximum sum of per-drive queue depths in a matrix cell")
    ap.add_argument("--operations-per-worker", type=int, default=2,
                    help="sustained reads per worker in each matrix sample")
    ap.add_argument("--cache-condition", default="unknown",
                    choices=("unknown", "warm", "cold"))
    ap.add_argument("--io-mode", default="buffered",
                    choices=("buffered", "direct", "overlapped"))
    ap.add_argument("--memory-copy-bytes", type=int, default=0,
                    help="also measure host DRAM copy bandwidth with this buffer size")
    ap.add_argument("--memory-samples", type=int, default=8)
    ap.add_argument("--allow-range-reuse", action="store_true",
                    help="reuse file ranges when the requested matrix exceeds a source; "
                         "only valid with --cache-condition unknown")
    args = ap.parse_args()
    if args.bytes <= 0 or args.bytes % 4096:
        fail("--bytes must be a positive 4096-byte multiple")
    if args.samples < 1 or args.samples > 64:
        fail("--samples must be in 1..64")
    try:
        inflight = [int(item) for item in args.inflight.split(",")]
    except ValueError:
        fail("--inflight must be comma-separated integers")
    if not inflight or any(value < 1 or value > 16 for value in inflight):
        fail("--inflight values must be in 1..16")
    if args.matrix_inflight is None:
        matrix_inflight = sorted({0, *inflight})
    else:
        try:
            matrix_inflight = [int(item) for item in args.matrix_inflight.split(",")]
        except ValueError:
            fail("--matrix-inflight must be comma-separated integers")
    if (not matrix_inflight
            or any(value < 0 or value > 16 for value in matrix_inflight)
            or len(set(matrix_inflight)) != len(matrix_inflight)):
        fail("--matrix-inflight values must be unique integers in 0..16")
    if not 1 <= args.max_total_inflight <= 16:
        fail("--max-total-inflight must be in 1..16")
    if not 1 <= args.operations_per_worker <= 32:
        fail("--operations-per-worker must be in 1..32")
    if (args.allow_range_reuse and args.cache_condition != "unknown"
            and not (args.io_mode in ("direct", "overlapped")
                     and args.cache_condition == "cold")):
        fail("--allow-range-reuse requires unknown cache state, except for "
             "explicit direct/cold reads")
    if args.cache_condition == "cold" and args.io_mode not in ("direct", "overlapped"):
        fail("cold cache calibration requires a direct-I/O backend; "
             "the Python backend is buffered-only")
    if args.io_mode == "overlapped" and os.name != "nt":
        fail("--io-mode overlapped is currently implemented only on Windows")
    if args.cache_condition == "warm" and args.io_mode in ("direct", "overlapped"):
        fail("warm cache calibration is incompatible with direct-I/O mode")
    manifest, raw = load_manifest(args.manifest)
    platform_topology = probe_platform_topology(manifest)
    sources = resolve_sources(manifest, args.bytes)
    range_slots_required = plan_range_slots(
        manifest, inflight, matrix_inflight, args.samples,
        args.max_total_inflight, args.operations_per_worker)
    range_reuse_observed = any(
        required > int(sources[drive_id].get("available_slots", 0))
        for drive_id, required in range_slots_required.items())
    if not args.allow_range_reuse:
        for drive_id, required in range_slots_required.items():
            available = int(sources[drive_id].get("available_slots", 0))
            if required > available:
                fail(f"drive {drive_id} needs {required} disjoint read slots, "
                     f"but the manifest provides {available}; add source paths "
                     "or pass --allow-range-reuse only for qualified evidence")
    measurements = []
    # Reserve virtual read slots across the entire capture, not just within
    # one group.  Independent drive/controller/upstream observations must not
    # silently reread the same physical ranges when range reuse is disabled.
    range_cursor = {drive_id: 0 for drive_id in sources}
    for group in manifest["groups"]:
        sweep = group.get("sweep", "matrix"
                          if len(group["drive_ids"]) > 1 else "scalar")
        if sweep == "matrix":
            cells = matrix_cells([int(value) for value in group["drive_ids"]],
                                 matrix_inflight, args.max_total_inflight)
            range_base = {int(value): range_cursor[int(value)]
                          for value in group["drive_ids"]}
            measurements.extend(measure_matrix(
                group, sources, args.bytes, matrix_inflight, args.samples,
                args.max_total_inflight, args.allow_range_reuse,
                args.operations_per_worker, args.cache_condition,
                args.io_mode, range_base))
            for drive_id in group["drive_ids"]:
                drive_id = int(drive_id)
                required = (sum(cell[drive_id] for cell in cells)
                            * args.samples * args.operations_per_worker)
                range_cursor[drive_id] += required
        else:
            for level in inflight:
                range_base = {int(value): range_cursor[int(value)]
                              for value in group["drive_ids"]}
                measurements.append(measure(
                    group, sources, args.bytes, level, args.samples,
                    args.allow_range_reuse, args.cache_condition,
                    args.io_mode, range_base, args.operations_per_worker))
                for slot in range(level):
                    drive_id = int(group["drive_ids"][slot
                                                       % len(group["drive_ids"])])
                    range_cursor[drive_id] += (
                        args.samples * args.operations_per_worker)
    try:
        source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError as exc:
        fail(f"cannot hash calibration tool source: {exc}")
    result = {
        "schema": CALIBRATION_SCHEMA,
        "tool": {
            "name": "m3_shadow_calibrate",
            "version": CALIBRATION_TOOL_VERSION,
            "source_sha256": source_sha256,
        },
        "topology_hash": manifest["topology_hash"],
        "topology_hash_algorithm": manifest.get("topology_hash_algorithm"),
        "topology_hash_verified": (
            manifest.get("topology_hash_algorithm") == _TOPOLOGY_HASH_ALGORITHM),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cache_condition": args.cache_condition,
            "io_mode": args.io_mode,
            "backend": io_backend_name(args.io_mode),
            "asynchronous": args.io_mode == "overlapped",
            "async_request_protocol": (
                "submit_first_wave_before_reap" if args.io_mode == "overlapped"
                else None),
            "cache_protocol": ("precondition_same_ranges"
                                if args.cache_condition == "warm"
                                else "uncontrolled"),
            "physical_traffic_note": (
                ("direct/unbuffered reads; OS page cache bypass requested"
                 if args.io_mode in ("direct", "overlapped")
                 else "buffered reads; OS cache state and controller traffic "
                      "were not flushed or inferred")
                + ("; range reuse observed" if range_reuse_observed else "")
                + ("; reuse permission was enabled" if args.allow_range_reuse
                   else "")),
            "dma_evidence": read_to_ready_evidence(
                args.io_mode,
                io_backend_name(args.io_mode)),
            "platform_topology": platform_topology,
            "pcie_traffic_evidence": probe_pcie_traffic_telemetry(),
            "range_reuse": range_reuse_observed,
            "range_reuse_allowed": bool(args.allow_range_reuse),
            "range_reuse_scope": ("capture" if not range_reuse_observed
                                   else "capture_observed"),
        },
        "quantiles": {
            "method": "nearest-rank-ceil",
            "population": "request-completions",
        },
        "read_bytes": args.bytes,
        "sample_count": args.samples,
        "inflight_levels": inflight,
        "matrix_inflight_levels": matrix_inflight,
        "max_total_inflight": args.max_total_inflight,
        "operations_per_worker": args.operations_per_worker,
        "sources": list(sources.values()),
        "range_allocation": {
            "scope": "capture",
            "slots_required": {str(k): v for k, v in range_slots_required.items()},
            "slots_available": {str(k): int(v.get("available_slots", 0))
                                 for k, v in sources.items()},
            "slots_unique": {str(k): min(v, int(sources[k].get(
                "available_slots", 0)))
                             for k, v in range_slots_required.items()},
            "slots_reused": {str(k): max(0, v - int(sources[k].get(
                "available_slots", 0)))
                             for k, v in range_slots_required.items()},
        },
        "measurements": measurements,
    }
    if args.memory_copy_bytes:
        result["memory"] = measure_memory_copy(args.memory_copy_bytes,
                                                args.memory_samples)
    if args.emit_planner_text is not None:
        result["manifest"] = {
            "topology_hash": manifest["topology_hash"],
            "storage_bounds": manifest.get("storage_bounds", {}),
            "groups": manifest["groups"],
            "islands": manifest.get("islands", []),
        }
    # Finalize the self-digest only after every optional capture field,
    # including the planner-emission manifest, has been added.
    result["capture_sha256"] = capture_hash(result)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    else:
        print(encoded, end="")
    if args.emit_planner_text is not None:
        try:
            text = emit_planner_text(result)
        except (KeyError, TypeError, ValueError) as exc:
            fail(f"could not render planner text: {exc}")
        args.emit_planner_text.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    main()
