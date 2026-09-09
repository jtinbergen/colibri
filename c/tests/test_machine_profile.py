import hashlib
import json
import uuid
import unittest
from pathlib import Path
from unittest import mock

from machine_profile import (SCHEMA, PROFILE_VERSION, attach_machine_profile,
                             build_machine_profile, format_machine_profile,
                             load_machine_profile, write_machine_profile)


class MachineProfileTest(unittest.TestCase):
    def setUp(self):
        # Managed Windows ACLs deny access to children of TemporaryDirectory.
        # Use unique files directly in the already-writable repository root.
        self.root = Path(__file__).resolve().parents[2]
        self.token = uuid.uuid4().hex
        self.model = self.root

    def tearDown(self):
        for path in self.root.glob(f".machine-profile-{self.token}-*"):
            try:
                path.unlink()
            except OSError:
                pass

    def artifact(self, name):
        return self.root / f".machine-profile-{self.token}-{name}"

    def profile(self, calibration=None, review=None, runtime_report=None):
        with mock.patch("machine_profile.os.cpu_count", return_value=32), \
             mock.patch("machine_profile.physical_cpu_count", return_value=24), \
             mock.patch("machine_profile.cpu_socket_count", return_value=2), \
             mock.patch("machine_profile.memory_available", return_value=64 * 1_000_000_000), \
             mock.patch("machine_profile.discover_gpus", return_value=[]):
            return build_machine_profile(self.model,
                                         calibration_path=calibration,
                                         review_path=review,
                                         runtime_report_path=runtime_report,
                                         now="2026-09-07T12:00:00Z")

    def test_profile_is_inventory_only_and_has_explicit_unknowns(self):
        profile = self.profile()
        self.assertEqual(profile["schema"], SCHEMA)
        self.assertEqual(profile["version"], PROFILE_VERSION)
        self.assertEqual(profile["cpu"]["physical_cores"], 24)
        self.assertEqual(profile["memory"]["available_bytes"], 64 * 1_000_000_000)
        self.assertEqual(profile["calibration"]["status"], "NOT_RUN")
        self.assertEqual(profile["calibration"]["coverage"]["shared_controller"],
                         "NOT_RUN")
        self.assertEqual(profile["calibration"]["coverage"]["controller_service"],
                         "NOT_RUN")
        self.assertEqual(profile["planner_contract"]["admission"], "DISABLED")
        self.assertEqual(profile["planner_contract"]["baseline_fallback"]["mode"],
                         "original_colibri_baseline")
        self.assertEqual(profile["planner_contract"]["baseline_fallback"]["rate_policy"],
                         "UNKNOWN; do not infer unavailable link/DMA capacity")

    def test_raw_capture_is_referenced_as_evidence_only(self):
        capture = self.artifact("capture.json")
        capture.write_text(json.dumps({
            "schema": "m3-shadow-calibration-v1",
            "topology_hash": 7,
            "manifest_sha256": "abc",
            "read_bytes": 16_777_216,
            "sample_count": 4,
            "inflight_levels": [1, 2, 4],
            "matrix_inflight_levels": [0, 1, 2],
            "host": {"cache_condition": "unknown",
                     "pcie_traffic_evidence": {"status": "UNAVAILABLE"}},
            "memory": {"method": "host_memcpy", "bytes_per_s_p50": 1234},
            "sources": [{"drive_id": 1}],
            "manifest": {
                "groups": [{"name": "upstream-1", "drive_ids": [1, 2],
                             "resource_id": 9, "kind": "upstream",
                             "sweep": "matrix"}],
                "islands": [{"kind": "cpu_island", "resource_id": 400,
                             "parent_mem_id": 500, "isa_bits": 6, "cores": 10}],
                "storage_bounds": {"max_inflight": 1, "max_bytes": 16},
            },
            "measurements": [
                {"group": "drive-1", "target_kind": "drive", "inflight": 1},
                {"group": "controller-1", "target_kind": "controller", "inflight": 4},
                {"group": "upstream-1", "target_kind": "upstream",
                 "drive_ids": [1, 2], "all_group_drives_active": True,
                 "measurement_kind": "contention_matrix",
                 "queue_depths": {"1": 1, "2": 1}},
            ],
        }), encoding="utf-8")
        profile = self.profile(capture)
        self.assertEqual(profile["calibration"]["status"], "EVIDENCE_ONLY")
        self.assertEqual(profile["calibration"]["capture"]["inflight_levels"], [1, 2, 4])
        self.assertEqual(profile["calibration"]["capture"]["groups"][0]["name"], "controller-1")
        self.assertEqual(profile["calibration"]["coverage"]["shared_controller"],
                         "NOT_APPLICABLE")
        self.assertEqual(profile["calibration"]["coverage"]["controller_service"],
                         "EVIDENCE_ONLY")
        self.assertEqual(profile["calibration"]["coverage"]["shared_upstream"], "EVIDENCE_ONLY")
        self.assertEqual(profile["calibration"]["coverage"]["host_memory"], "EVIDENCE_ONLY")
        self.assertEqual(profile["calibration"]["coverage"]["pch_dmi"],
                         "UNAVAILABLE")
        topology = profile["calibration"]["capture"]["topology"]
        self.assertEqual(topology["groups"][0]["resource_id"], 9)
        self.assertEqual(topology["islands"][0]["kind"], "cpu_island")
        self.assertEqual(topology["storage_bounds"]["max_inflight"], 1)
        self.assertEqual(profile["calibration"]["capture"]["source_count"], 1)
        self.assertIn("evidence-only", profile["calibration"]["reason"])

    def test_invalid_capture_is_visible_without_becoming_capacity(self):
        profile = self.profile(self.artifact("missing.json"))
        self.assertEqual(profile["calibration"]["status"], "UNKNOWN")
        self.assertEqual(profile["calibration"]["coverage"]["storage"], "NOT_RUN")

    def test_v2_capture_digest_is_verified_before_ingestion(self):
        capture = {
            "schema": "m3-shadow-calibration-v2",
            "topology_hash": 7,
            "read_bytes": 4096,
            "sample_count": 1,
            "inflight_levels": [1],
            "matrix_inflight_levels": [],
            "measurements": [{"group": "drive-1", "inflight": 1,
                              "target_kind": "drive", "drive_ids": [1]}],
            "sources": [{"drive_id": 1}],
            "manifest": {"groups": [], "islands": [],
                          "storage_bounds": {"max_inflight": 1,
                                             "max_bytes": 4096}},
        }
        encoded = json.dumps(capture, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        capture["capture_sha256"] = hashlib.sha256(encoded).hexdigest()
        path = self.artifact("v2-capture.json")
        path.write_text(json.dumps(capture), encoding="utf-8")
        profile = self.profile(path)
        self.assertEqual(profile["calibration"]["status"], "EVIDENCE_ONLY")
        self.assertTrue(profile["calibration"]["capture"]["capture_hash_verified"])
        capture["read_bytes"] += 4096
        path.write_text(json.dumps(capture), encoding="utf-8")
        tampered = self.profile(path)
        self.assertEqual(tampered["calibration"]["status"], "UNKNOWN")

    def test_review_is_attached_as_hash_bound_candidate_evidence(self):
        capture = {
            "schema": "m3-shadow-calibration-v2",
            "topology_hash": 7,
            "manifest_sha256": "manifest",
            "tool": {"name": "m3_shadow_calibrate", "version": 2,
                      "source_sha256": "tool"},
            "read_bytes": 4096,
            "sample_count": 1,
            "inflight_levels": [1],
            "matrix_inflight_levels": [],
            "measurements": [{"group": "drive-1", "inflight": 1,
                              "target_kind": "drive", "drive_ids": [1]}],
            "sources": [{"drive_id": 1}],
            "manifest": {"groups": [], "islands": [],
                          "storage_bounds": {"max_inflight": 1,
                                             "max_bytes": 4096}},
        }
        capture["capture_sha256"] = hashlib.sha256(
            json.dumps(capture, sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
        capture_path = self.artifact("review-capture.json")
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        review = {
            "schema": "m3-shadow-calibration-review-v2",
            "tail_percentile_ppm": 950000,
            "allowed_tail_multiplier": 2.0,
            "promotion": {"status": "CANDIDATE_ONLY",
                           "active_admission_allowed": False},
            "captures": [{"topology_hash": 7,
                           "manifest_sha256": "manifest",
                           "capture_sha256": capture["capture_sha256"],
                           "tool": capture["tool"]}],
            "results": [{"group": "drive-1", "target_resource_id": 1,
                         "target_kind": "drive", "drive_ids": [1],
                         "candidate_is_observed_not_extrapolated": True,
                         "candidate_rate": 1000,
                         "max_inflight_candidate": 2,
                         "candidate_latency_ns_tail": 200,
                         "candidate_min_retention": 0.9,
                         "candidate_normalized_jain": 1.0}],
        }
        review_path = self.artifact("review.json")
        review_path.write_text(json.dumps(review), encoding="utf-8")
        profile = self.profile(capture_path, review_path)
        evidence = profile["calibration"]["capture"]["review"]
        self.assertTrue(evidence["capture_binding_verified"])
        self.assertEqual(evidence["promotion_status"], "CANDIDATE_ONLY")
        self.assertEqual(evidence["candidate_rows"][0]["candidate_rate"], 1000)
        self.assertFalse(evidence["active_admission_allowed"])

    def test_mismatched_review_stays_unknown_without_invalidating_capture(self):
        capture = {
            "schema": "m3-shadow-calibration-v1",
            "topology_hash": 7,
            "manifest_sha256": "manifest",
            "read_bytes": 4096,
            "sample_count": 1,
            "inflight_levels": [1],
            "matrix_inflight_levels": [],
            "measurements": [],
            "sources": [],
            "manifest": {"groups": [], "islands": [],
                          "storage_bounds": {"max_inflight": 1,
                                             "max_bytes": 4096}},
        }
        capture_path = self.artifact("mismatch-capture.json")
        capture_path.write_text(json.dumps(capture), encoding="utf-8")
        review_path = self.artifact("mismatch-review.json")
        review_path.write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
        profile = self.profile(capture_path, review_path)
        self.assertEqual(profile["calibration"]["status"], "EVIDENCE_ONLY")
        self.assertFalse(profile["calibration"]["capture"]["review"]
                         ["capture_binding_verified"])

    def test_runtime_report_is_attached_as_inactive_evidence(self):
        report_path = self.artifact("runtime-report.json")
        report_path.write_text(json.dumps({
            "schema": "m3-shadow-runtime-report-v2",
            "sources": [{"path": "run-1.csv", "sha256": "abc"}],
            "tool_sha256": "tool",
            "rows_total": 8,
            "rows_completed": 8,
            "rows_successful": 8,
            "evidence_counts": {"OBSERVED": 8},
            "candidate_policy": {
                "status": "CANDIDATE_ONLY",
                "active_admission_allowed": False,
                "unknown_and_unavailable_are_not_capacity": True,
                "tail_percentile": 95,
                "max_tail_multiplier": 1.5,
                "min_samples": 2,
            },
            "runtime_candidates": [{
                "status": "CANDIDATE_ONLY",
                "active_admission_allowed": False,
                "evidence_state": "OBSERVED",
                "confidence": "REPEATED_INDEPENDENT_RUNS",
                "scope": {"actual_path": "101|201|301", "bytes": 512},
                "max_inflight_candidate": 2,
                "candidate_latency_ns_p95": 100,
                "candidate_latency_ns_p99": 120,
                "observed_levels": [1, 2],
            }],
        }), encoding="utf-8")
        profile = self.profile(runtime_report=report_path)
        evidence = profile["runtime_evidence"]
        self.assertEqual(evidence["status"], "EVIDENCE_ONLY")
        self.assertEqual(evidence["report"]["runtime_candidates"][0]
                         ["max_inflight_candidate"], 2)
        self.assertFalse(evidence["report"]["candidate_policy"]
                         ["active_admission_allowed"])
        plan = {"tiers": {"ram": {"budget_bytes": 123}}, "warnings": []}
        attach_machine_profile(plan, profile, "machine.json")
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 123)
        self.assertTrue(any("runtime telemetry candidates are evidence-only" in warning
                            for warning in plan["warnings"]))

    def test_nonconservative_runtime_report_stays_unknown(self):
        report_path = self.artifact("unsafe-runtime-report.json")
        report_path.write_text(json.dumps({
            "schema": "m3-shadow-runtime-report-v2",
            "candidate_policy": {"status": "CANDIDATE_ONLY",
                                  "active_admission_allowed": True},
            "runtime_candidates": [],
        }), encoding="utf-8")
        profile = self.profile(runtime_report=report_path)
        self.assertEqual(profile["runtime_evidence"]["status"], "UNKNOWN")
        self.assertIn("conservative candidate policy",
                      profile["runtime_evidence"]["reason"])

    def test_runtime_report_must_match_calibration_topology(self):
        capture_path = self.artifact("topology-capture.json")
        capture_path.write_text(json.dumps({
            "schema": "m3-shadow-calibration-v1",
            "topology_hash": 700,
            "read_bytes": 4096,
            "sample_count": 1,
            "inflight_levels": [1],
            "matrix_inflight_levels": [],
            "measurements": [],
            "sources": [],
            "manifest": {"groups": [], "islands": [],
                          "storage_bounds": {"max_inflight": 1,
                                             "max_bytes": 4096}},
        }), encoding="utf-8")
        report_path = self.artifact("mismatched-runtime-report.json")
        report_path.write_text(json.dumps({
            "schema": "m3-shadow-runtime-report-v2",
            "candidate_policy": {
                "status": "CANDIDATE_ONLY",
                "active_admission_allowed": False,
                "unknown_and_unavailable_are_not_capacity": True,
            },
            "runtime_candidates": [{
                "status": "CANDIDATE_ONLY",
                "active_admission_allowed": False,
                "evidence_state": "OBSERVED",
                "scope": {"actual_path": "101|201|301",
                          "topology_hash": 701},
            }],
        }), encoding="utf-8")
        profile = self.profile(capture_path, runtime_report=report_path)
        self.assertEqual(profile["runtime_evidence"]["status"], "UNKNOWN")
        self.assertIn("topology hash", profile["runtime_evidence"]["reason"])

    def test_atomic_write_and_schema_validation(self):
        profile = self.profile()
        destination = self.artifact("machine.json")
        write_machine_profile(destination, profile)
        self.assertEqual(load_machine_profile(destination), profile)
        bad = dict(profile, schema="wrong")
        with self.assertRaises(ValueError):
            write_machine_profile(self.artifact("bad.json"), bad)
        destination.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_machine_profile(destination)

    def test_attachment_does_not_change_placement_data(self):
        profile = self.profile()
        plan = {"tiers": {"ram": {"budget_bytes": 123}}, "warnings": []}
        attach_machine_profile(plan, profile, "machine.json")
        self.assertEqual(plan["tiers"]["ram"]["budget_bytes"], 123)
        self.assertEqual(plan["machine_profile"]["status"], "NOT_RUN")
        self.assertIn("NOT_RUN", format_machine_profile(profile))


if __name__ == "__main__":
    unittest.main()
