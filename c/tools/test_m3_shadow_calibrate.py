import importlib.util
import json
import struct
import unittest
from pathlib import Path
from unittest import mock
from types import SimpleNamespace


_MODULE_PATH = Path(__file__).with_name("m3_shadow_calibrate.py")
_SPEC = importlib.util.spec_from_file_location("m3_shadow_calibrate", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


def _measurement(group, drive_ids, rate, target_resource_id=None,
                  target_kind=None):
    return {
        "group": group,
        "drive_ids": list(drive_ids),
        "target_resource_id": target_resource_id,
        "target_kind": target_kind,
        "active_drive_ids": list(drive_ids),
        "all_group_drives_active": True,
        "inflight": 1,
        "summary": {
            "aggregate_bytes_per_s_p50": rate,
            "latency_ns_p50": 100,
            "latency_ns_p95": 120,
        },
    }


def _capture(measurements):
    return {
        "topology_hash": 7,
        "host": {"cache_condition": "unknown", "physical_traffic_note": "test"},
        "read_bytes": 16 * 1024 * 1024,
        "sample_count": 4,
        "sources": [
            {"drive_id": 101, "controller_id": 201, "upstream_id": 901, "path": "a"},
            {"drive_id": 102, "controller_id": 202, "upstream_id": 901, "path": "b"},
        ],
        "measurements": measurements,
        "manifest": {"storage_bounds": {
            "max_inflight": 1, "max_bytes": 16 * 1024 * 1024,
        }},
    }


class CalibrateEmitterTests(unittest.TestCase):
    def test_percentile_uses_conservative_nearest_rank(self):
        self.assertEqual(_MOD.percentile([1, 2, 3, 4], 0.50), 2)
        self.assertEqual(_MOD.percentile([1, 2, 3, 4], 0.95), 4)
        self.assertEqual(_MOD.percentile([1, 2, 3, 4], 0.99), 4)

    def test_pcie_negotiated_payload_is_endpoint_evidence(self):
        self.assertEqual(_MOD._pcie_payload_bytes_per_s(3, 4),
                         3_938_461_538)
        self.assertEqual(_MOD._pcie_payload_bytes_per_s(4, 4),
                         7_876_923_076)
        self.assertEqual(_MOD._pcie_payload_bytes_per_s_gts(2.5, 16),
                         4_000_000_000)
        self.assertIsNone(_MOD._pcie_payload_bytes_per_s(99, 4))

    def test_windows_volume_extent_parser_records_physical_disk_identity(self):
        raw = struct.pack("<I4x", 2)
        raw += struct.pack("<I4xqq", 3, 0, 1024 * 1024)
        raw += struct.pack("<I4xqq", 4, 1024 * 1024, 2048 * 1024)
        result = _MOD._parse_windows_volume_disk_extents(raw)
        self.assertEqual(result["status"], "OBSERVED")
        self.assertEqual(result["disk_numbers"], [3, 4])
        self.assertEqual(result["extents"][1]["starting_offset"], 1024 * 1024)

    def test_direct_read_to_ready_evidence_is_not_dma_attestation(self):
        evidence = _MOD.read_to_ready_evidence(
            "direct", "win32-no-buffering")
        self.assertEqual(evidence["status"], "OBSERVED_NOT_ATTESTED")
        self.assertIn("DMA", " ".join(evidence["limitations"]).upper())
        self.assertEqual(
            _MOD.read_to_ready_evidence("buffered", "python-pread")["status"],
            "NOT_RUN")
        overlapped = _MOD.read_to_ready_evidence(
            "overlapped", "win32-overlapped")
        self.assertEqual(overlapped["status"], "OBSERVED_NOT_ATTESTED")
        self.assertIn("OVERLAPPED", overlapped["completion"])
        self.assertEqual(_MOD.io_backend_name("overlapped"),
                         "win32-overlapped")

    def test_pcie_traffic_probe_is_explicit_when_no_counter_is_available(self):
        with mock.patch.object(_MOD.shutil, "which", return_value=None):
            result = _MOD.probe_pcie_traffic_telemetry()
        self.assertEqual(result["status"], "UNAVAILABLE")
        self.assertIn("PCH/DMI", result["reason"])

    def test_pcie_traffic_probe_does_not_invoke_a_counter(self):
        with mock.patch.object(_MOD.shutil, "which",
                               side_effect=lambda name: "tool" if name == "perf" else None):
            result = _MOD.probe_pcie_traffic_telemetry()
        self.assertEqual(result["status"], "AVAILABLE_NOT_RUN")
        self.assertEqual(result["candidates"], ["tool"])

    def test_windows_probe_records_negotiated_link_without_claiming_capacity(self):
        manifest = {
            "sources": [{
                "drive_id": 101, "controller_id": 201,
                "platform": {"windows_pci_instance_id": "PCI\\TEST"},
            }],
            "islands": [],
        }
        response = SimpleNamespace(
            returncode=0,
            stdout=("DEVPKEY_Device_DeviceDesc [String]:\n"
                    "    Standard NVM Express Controller\n"
                    "DEVPKEY_Device_Parent [String]:\n"
                    "    PCI\\\\ROOT\n"
                    "DEVPKEY_Device_LocationInfo [String]:\n"
                    "    PCI bus 6, device 0, function 0\n"
                    "DEVPKEY_PciDevice_CurrentLinkSpeed [UINT32]:\n"
                    "    0x00000004 (4)\n"
                    "DEVPKEY_PciDevice_CurrentLinkWidth [UINT32]:\n"
                    "    0x00000004 (4)\n"),
            stderr="",
        )
        with mock.patch.object(_MOD.os, "name", "nt"), \
                mock.patch.object(_MOD.subprocess, "run", return_value=response):
            result = _MOD.probe_platform_topology(manifest)
        self.assertEqual(result["status"], "OBSERVED")
        self.assertEqual(result["links"][0]["evidence_status"],
                         "OBSERVED_NEGOTIATED")
        self.assertNotIn("rate_bytes_per_s", result["links"][0])
        self.assertEqual(result["links"][0]["parent_chain"][0]["instance_id"],
                         "PCI\\\\ROOT")
        self.assertIn("PCI\\\\ROOT", result["common_ancestor_instance_ids"])

    def test_linux_probe_records_negotiated_link_without_claiming_capacity(self):
        manifest = {
            "sources": [{
                "drive_id": 101, "controller_id": 201,
                "platform": {"linux_pci_address": "0000:06:00.0"},
            }],
            "islands": [],
        }
        response = SimpleNamespace(
            returncode=0,
            stdout=("06:00.0 Non-Volatile memory controller: Test\n"
                    "\tLnkCap: Speed 32GT/s, Width x4\n"
                    "\tLnkSta: Speed 16GT/s (ok), Width x4 (ok)\n"),
            stderr="",
        )
        with mock.patch.object(_MOD.os, "name", "posix"), \
                mock.patch.object(_MOD.sys, "platform", "linux"), \
                mock.patch.object(_MOD.subprocess, "run", return_value=response):
            result = _MOD.probe_platform_topology(manifest)
        self.assertEqual(result["status"], "OBSERVED")
        self.assertEqual(result["links"][0]["speed_gts"], 16.0)
        self.assertEqual(result["links"][0]["width"], 4)
        self.assertEqual(result["links"][0]["negotiated_payload_bytes_per_s"],
                         7_876_923_076)
        self.assertNotIn("rate_bytes_per_s", result["links"][0])

    def test_virtual_source_slots_map_to_shards_without_crossing_boundaries(self):
        source = {
            "path": "first.safetensors",
            "available_slots": 5,
            "files": [
                {"path": "first.safetensors", "slot_base": 0, "slot_count": 2},
                {"path": "second.safetensors", "slot_base": 2, "slot_count": 3},
            ],
        }
        self.assertEqual(_MOD._source_target(source, 0, 4096),
                         ("first.safetensors", 0))
        self.assertEqual(_MOD._source_target(source, 1, 4096),
                         ("first.safetensors", 4096))
        self.assertEqual(_MOD._source_target(source, 2, 4096),
                         ("second.safetensors", 0))
        self.assertEqual(_MOD._source_target(source, 4, 4096),
                         ("second.safetensors", 8192))

    def test_matrix_cells_are_independent_and_total_bounded(self):
        cells = _MOD.matrix_cells([101, 102], [0, 1, 2, 4], 4)
        self.assertIn({101: 0, 102: 1}, cells)
        self.assertIn({101: 1, 102: 1}, cells)
        self.assertIn({101: 2, 102: 2}, cells)
        self.assertNotIn({101: 4, 102: 1}, cells)
        self.assertNotIn({101: 0, 102: 0}, cells)
        self.assertTrue(all(sum(cell.values()) <= 4 for cell in cells))

    def test_raw_planner_emit_keeps_shared_resource_topology_without_profiles(self):
        measurement = _measurement(
            "shared-upstream-901", (101, 102), 500, 901, "upstream")
        measurement.update({
            "measurement_kind": "contention_matrix",
            "queue_depths": {"101": 1, "102": 1},
        })
        measurement["summary"]["per_drive"] = {
            "101": {"aggregate_bytes_per_s_p50": 250},
            "102": {"aggregate_bytes_per_s_p50": 250},
        }
        text = _MOD.emit_planner_text(_capture([measurement]))
        self.assertIn("resource 901 ", text)
        self.assertNotIn("profile ", text)

    def test_planner_emits_explicit_unknown_fallback_policy(self):
        capture = _capture([])
        capture["host"].update({
            "io_mode": "direct",
            "platform_topology": {
                "status": "UNAVAILABLE", "method": "windows-pnp",
                "links": [],
            },
        })
        text = _MOD.emit_planner_text(capture)
        self.assertIn("# unavailable_resource_policy = conservative_unknown", text)
        self.assertIn("# unavailable_resource_fallback = storage_bounds.max_inflight; rate=UNKNOWN", text)

    def test_overlapped_protocol_is_preserved_in_planner_metadata(self):
        capture = _capture([_measurement("drive-101", (101,), 100)])
        capture["host"].update({
            "asynchronous": True,
            "io_mode": "overlapped",
            "backend": "win32-overlapped",
            "async_request_protocol": "submit_first_wave_before_reap",
            "pcie_traffic_evidence": {"status": "UNAVAILABLE"},
        })
        text = _MOD.emit_planner_text(capture)
        self.assertIn("# asynchronous = True", text)
        self.assertIn("# async_request_protocol = submit_first_wave_before_reap", text)

    def test_matrix_measurement_records_per_drive_and_fairness(self):
        read_bytes = 4096
        sources = {
            101: {"drive_id": 101, "size_bytes": read_bytes * 32,
                  "path": "a"},
            102: {"drive_id": 102, "size_bytes": read_bytes * 32,
                  "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}
        class FakeHandle:
            def read(self, offset, length):
                return length

            def close(self):
                return None

        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FakeHandle()):
            rows = _MOD.measure_matrix(group, sources, read_bytes, [0, 1], 2, 2)
        self.assertEqual(len(rows), 3)
        cell = next(row for row in rows
                    if row["queue_depths"] == {"101": 1, "102": 1})
        self.assertEqual(cell["inflight"], 2)
        self.assertEqual(set(cell["summary"]["per_drive"]), {"101", "102"})
        self.assertGreaterEqual(cell["summary"]["throughput_share_jain_p50"], 0.0)
        self.assertLessEqual(cell["summary"]["throughput_share_jain_p50"], 1.0)

    def test_matrix_range_allocation_is_compact_and_disjoint(self):
        read_bytes = 4096
        sources = {
            101: {"drive_id": 101, "size_bytes": read_bytes * 32,
                  "path": "a"},
            102: {"drive_id": 102, "size_bytes": read_bytes * 32,
                  "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}

        class FakeHandle:
            def read(self, offset, length):
                return length

            def close(self):
                return None

        # The compact allocator needs 26 slots per drive for this matrix;
        # the old max-QD-per-cell formula incorrectly needed 80.
        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FakeHandle()):
            rows = _MOD.measure_matrix(group, sources, read_bytes,
                                       [0, 1, 2, 4], 2, 4,
                                       operations_per_worker=1)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(not row["range_reuse"] for row in rows))

    def test_matrix_operations_use_slot_bases_in_slot_units(self):
        read_bytes = 4096
        sources = {
            101: {"drive_id": 101, "size_bytes": read_bytes * 256,
                  "path": "a"},
            102: {"drive_id": 102, "size_bytes": read_bytes * 256,
                  "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}
        seen: dict[str, list[int]] = {"a": [], "b": []}

        class FakeHandle:
            def __init__(self, path):
                self.path = path

            def read(self, offset, length):
                seen[self.path].append(offset // read_bytes)
                return length

            def close(self):
                return None

        with mock.patch.object(
                _MOD, "open_read_handle",
                side_effect=lambda path, *args: FakeHandle(path)):
            rows = _MOD.measure_matrix(
                group, sources, read_bytes, [0, 1, 2], 2, 2,
                operations_per_worker=2,
                range_base_slots={101: 10, 102: 20})
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(seen["a"]), len(set(seen["a"])))
        self.assertEqual(len(seen["b"]), len(set(seen["b"])))
        self.assertGreaterEqual(min(seen["a"]), 10)
        self.assertGreaterEqual(min(seen["b"]), 20)

    def test_matrix_reports_queue_depth_separately_from_operation_count(self):
        read_bytes = 4096
        sources = {
            101: {"drive_id": 101, "size_bytes": read_bytes * 64,
                  "path": "a"},
            102: {"drive_id": 102, "size_bytes": read_bytes * 64,
                  "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}

        class FakeHandle:
            def read(self, offset, length):
                return length

            def close(self):
                return None

        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FakeHandle()):
            rows = _MOD.measure_matrix(group, sources, read_bytes, [0, 1],
                                       1, 2, operations_per_worker=2)
        cell = next(row for row in rows
                    if row["queue_depths"] == {"101": 1, "102": 1})
        drive = cell["summary"]["per_drive"]["101"]
        self.assertEqual(drive["inflight"], 1)
        self.assertEqual(drive["request_count"], 2)

    def test_overlapped_matrix_submits_all_first_requests_before_reap(self):
        read_bytes = 4096
        sources = {
            101: {"drive_id": 101, "size_bytes": read_bytes * 16,
                  "path": "a"},
            102: {"drive_id": 102, "size_bytes": read_bytes * 16,
                  "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}
        state = {"submitted": 0, "reaped_before_all": False}

        class FakeOverlapped:
            def begin_read(self, offset, length):
                state["submitted"] += 1
                return object(), True

            def complete_read(self, operation, length):
                if state["submitted"] != 2:
                    state["reaped_before_all"] = True
                return length

            def read(self, offset, length):
                return length

            def close(self):
                return None

        with mock.patch.object(
                _MOD, "open_read_handle",
                side_effect=lambda *args: FakeOverlapped()):
            rows = _MOD.measure_matrix(
                group, sources, read_bytes, [1], 1, 2,
                operations_per_worker=1, io_mode="overlapped")
        cell = next(row for row in rows
                    if row["queue_depths"] == {"101": 1, "102": 1})
        self.assertEqual(state["submitted"], 2)
        self.assertFalse(state["reaped_before_all"])
        self.assertEqual(cell["samples"][0]["overlap_evidence"]["protocol"],
                         "submit_first_wave_before_reap")

    def test_scalar_operations_are_disjoint_and_counted(self):
        read_bytes = 4096
        sources = {101: {"drive_id": 101, "size_bytes": read_bytes * 8,
                         "path": "a"}}
        group = {"name": "drive", "drive_ids": [101],
                 "resource_id": 101, "kind": "drive"}
        seen = []

        class FakeHandle:
            def read(self, offset, length):
                seen.append(offset // read_bytes)
                return length

            def close(self):
                return None

        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FakeHandle()):
            result = _MOD.measure(
                group, sources, read_bytes, 1, 2,
                io_mode="buffered", operations_per_worker=2)
        self.assertEqual(seen, [0, 1, 2, 3])
        self.assertEqual(result["operations_per_worker"], 2)
        self.assertEqual(result["samples"][0]["bytes"], read_bytes * 2)

    def test_scalar_warmup_failure_aborts_peer_barrier(self):
        sources = {101: {"drive_id": 101, "size_bytes": 4096 * 8,
                         "path": "a"}}
        group = {"name": "drive", "drive_ids": [101]}

        class FailingHandle:
            def read(self, offset, length):
                raise OSError("synthetic warmup failure")

            def close(self):
                return None

        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FailingHandle()):
            with self.assertRaises(SystemExit):
                _MOD.measure(group, sources, 4096, 2, 1,
                             cache_condition="warm")

    def test_matrix_warmup_failure_aborts_peer_barrier(self):
        sources = {
            101: {"drive_id": 101, "size_bytes": 4096 * 8, "path": "a"},
            102: {"drive_id": 102, "size_bytes": 4096 * 8, "path": "b"},
        }
        group = {"name": "shared", "drive_ids": [101, 102],
                 "resource_id": 901, "kind": "upstream"}

        class FailingHandle:
            def read(self, offset, length):
                raise OSError("synthetic warmup failure")

            def close(self):
                return None

        with mock.patch.object(_MOD, "open_read_handle",
                               side_effect=lambda *args: FailingHandle()):
            with self.assertRaises(SystemExit):
                _MOD.measure_matrix(group, sources, 4096, [0, 1], 1, 2,
                                    cache_condition="warm")

    def test_aggregate_is_not_copied_and_singletons_are_order_independent(self):
        measurements = [
            _measurement("pair", (101, 102), 900),
            _measurement("drive-101", (101,), 100),
            _measurement("drive-101-repeat", (101,), 50),
            _measurement("drive-102", (102,), 200),
        ]
        first = _MOD.emit_planner_text(_capture(measurements))
        second = _MOD.emit_planner_text(_capture(list(reversed(measurements))))
        self.assertEqual(first, second)
        profiles = [line for line in first.splitlines() if line.startswith("profile ")]
        self.assertEqual(profiles, [])
        self.assertNotIn(" 900 ", first)

    def test_explicit_controller_and_upstream_targets_are_not_inferred(self):
        capture = _capture([
            _measurement("drive-101", (101,), 100),
            _measurement("controller-201", (101,), 90, 201, "controller"),
            _measurement("upstream-901", (101, 102), 80, 901, "upstream"),
        ])
        text = _MOD.emit_planner_text(capture)
        self.assertIn("resource 101 ", text)
        self.assertIn("resource 201 ", text)
        self.assertIn("resource 901 ", text)
        self.assertNotIn("profile ", text)

    def test_memory_copy_measurement_is_emitted_as_memory_domain(self):
        capture = _capture([_measurement("drive-101", (101,), 100)])
        capture["manifest"] = {
            "storage_bounds": {"max_inflight": 1, "max_bytes": 16 * 1024 * 1024},
            "islands": [{
                "kind": "memory_dom", "resource_id": 500,
                "parent_id": 0, "bandwidth_class": 2,
            }],
        }
        capture["memory"] = {
            "method": "host_memcpy", "bytes": 16 * 1024 * 1024,
            "bytes_per_s_p50": 1234, "duration_ns_p50": 10,
            "duration_ns_p95": 12, "samples": 4,
        }
        text = _MOD.emit_planner_text(capture)
        self.assertIn("memory_dom 500 0 0 2", text)
        self.assertIn("host_memory_evidence = method=host_memcpy p50_bps=1234",
                      text)
        self.assertNotIn("profile 500 ", text)

    def test_unmeasured_islands_remain_parser_visible_with_unknown_rates(self):
        capture = _capture([_measurement("drive-101", (101,), 100)])
        capture["manifest"]["islands"] = [
            {"kind": "link", "resource_id": 600, "parent_cpu_id": 400,
             "status": "NOT_RUN", "reason": "not measured"},
            {"kind": "gpu_island", "resource_id": 700,
             "parent_link_id": 600, "vram_bytes": 8,
             "eligible": 0, "status": "NOT_RUN"},
        ]
        text = _MOD.emit_planner_text(capture)
        self.assertIn("link 600 400 0 0", text)
        self.assertIn("gpu_island 700 600 8 0 0 0", text)

    def test_manifest_rejects_conflicting_controller_parents(self):
        manifest = {
            "topology_hash": 7,
            "sources": [
                {"drive_id": 1, "controller_id": 10, "upstream_id": 20,
                 "path": "a"},
                {"drive_id": 2, "controller_id": 10, "upstream_id": 21,
                 "path": "b"},
            ],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
        }
        path = Path("synthetic-conflicting-manifest.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_resolver_rejects_duplicate_physical_paths(self):
        manifest = {
            "sources": [
                {"drive_id": 1, "path": str(_MODULE_PATH)},
                {"drive_id": 2, "path": str(_MODULE_PATH)},
            ]
        }
        with self.assertRaises(SystemExit):
            _MOD.resolve_sources(manifest, 1)

    def test_manifest_rejects_topology_hash_mismatch(self):
        manifest = {
            "topology_hash": 99,
            "topology_hash_algorithm": "sha256-trunc64-v1",
            "sources": [{"drive_id": 1, "path": "a"}],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
        }
        path = Path("synthetic-bad-topology-hash.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_manifest_requires_canonical_topology_hash_algorithm(self):
        manifest = {
            "topology_hash": 7,
            "sources": [{"drive_id": 1, "path": "a"}],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
        }
        path = Path("synthetic-legacy-hash-manifest.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_manifest_rejects_unknown_positive_island_rate(self):
        manifest = {
            "topology_hash": 7,
            "sources": [{"drive_id": 1, "path": "a"}],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
            "islands": [{"kind": "memory_dom", "resource_id": 500,
                          "parent_id": 0, "capacity_status": "UNKNOWN",
                          "rate_bytes_per_s": 1}],
        }
        path = Path("synthetic-unknown-rate-manifest.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_manifest_rejects_missing_typed_island_parent(self):
        manifest = {
            "topology_hash": 7,
            "sources": [{"drive_id": 1, "path": "a"}],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
            "islands": [{"kind": "memory_dom", "resource_id": 500}],
        }
        path = Path("synthetic-missing-island-parent.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_manifest_rejects_cross_kind_resource_id_collision(self):
        manifest = {
            "topology_hash": 7,
            "sources": [
                {"drive_id": 1, "controller_id": 10, "path": "a"},
                {"drive_id": 10, "path": "b"},
            ],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
        }
        path = Path("synthetic-cross-kind-id.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_manifest_rejects_pcie_parent_cycle(self):
        manifest = {
            "topology_hash": 7,
            "sources": [{"drive_id": 1, "path": "a"}],
            "groups": [{"name": "drive-1", "drive_ids": [1]}],
            "islands": [{"kind": "link", "resource_id": 600,
                          "parent_cpu_id": 0, "pcie_parent_id": 600,
                          "rate_bytes_per_s": 1}],
        }
        path = Path("synthetic-pcie-cycle.json")
        with mock.patch.object(_MOD.Path, "read_bytes",
                               return_value=json.dumps(manifest).encode()):
            with self.assertRaises(SystemExit):
                _MOD.load_manifest(path)

    def test_planner_emit_requires_explicit_storage_policy(self):
        capture = _capture([_measurement("drive-101", (101,), 100)])
        del capture["manifest"]["storage_bounds"]
        with self.assertRaises(SystemExit):
            _MOD.emit_planner_text(capture)


if __name__ == "__main__":
    unittest.main()
