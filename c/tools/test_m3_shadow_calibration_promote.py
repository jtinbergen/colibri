import hashlib
import importlib.util
import json
import unittest
from pathlib import Path


_MODULE_PATH = Path(__file__).with_name("m3_shadow_calibration_promote.py")
_SPEC = importlib.util.spec_from_file_location("m3_shadow_calibration_promote", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)


class CandidatePlannerTests(unittest.TestCase):
    def setUp(self):
        self.manifest = _MODULE_PATH
        manifest_sha = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.capture = {
            "schema": "m3-shadow-calibration-v2",
            "topology_hash": 7,
            "manifest_sha256": manifest_sha,
            "tool": {"name": "m3_shadow_calibrate", "version": 2,
                      "source_sha256": "tool-hash"},
            "host": {"range_reuse": False, "range_reuse_allowed": False,
                     "cache_condition": "cold", "io_mode": "direct",
                     "backend": "test", "asynchronous": False,
                     "physical_traffic_note": "test"},
            "read_bytes": 16 * 1024 * 1024,
            "sample_count": 16,
            "sources": [{"drive_id": 101, "controller_id": 201,
                         "upstream_id": 901}],
            "manifest": {"storage_bounds": {"max_inflight": 1,
                                               "max_bytes": 16 * 1024 * 1024},
                          "islands": []},
        }
        self.capture["capture_sha256"] = _MOD.capture_hash(self.capture)
        self.review = {
            "schema": "m3-shadow-calibration-review-v2",
            "tail_percentile_ppm": 950000,
            "promotion": {"status": "CANDIDATE_ONLY",
                           "active_admission_allowed": False},
            "captures": [{"topology_hash": 7,
                           "manifest_sha256": manifest_sha,
                           "capture_sha256": self.capture["capture_sha256"],
                           "tool": self.capture["tool"]}],
            "results": [{
                "group": "drive-101",
                "drive_ids": [101],
                "target_resource_id": 101,
                "target_kind": "drive",
                "candidate_is_observed_not_extrapolated": True,
                "max_inflight_candidate": 2,
                "candidate_rate": 1234,
                 "candidate_sample_count": 16,
                 "candidate_latency_ns_tail": 200,
                 "candidate_latency_ns_p50": 100,
                 "tail_percentile_ppm": 950000,
             }],
        }

    def test_only_reviewed_candidate_rows_are_emitted(self):
        review_sha = _MOD._verify_provenance(
            self.capture, self.review, self.manifest)
        text = _MOD.emit_candidate(self.capture, self.review, review_sha)
        self.assertIn("# configuration_status = candidate_only", text)
        self.assertIn("# active_admission_allowed = False", text)
        self.assertIn("profile 101 2 1 1234 0 100 16 950000", text)

    def test_reused_capture_cannot_be_promoted(self):
        self.capture["host"]["range_reuse"] = True
        with self.assertRaises(SystemExit):
            _MOD._verify_provenance(self.capture, self.review, self.manifest)

    def test_modified_v2_capture_cannot_be_promoted(self):
        self.capture["read_bytes"] += 4096
        with self.assertRaises(SystemExit):
            _MOD._verify_provenance(self.capture, self.review, self.manifest)

    def test_legacy_v1_capture_cannot_be_promoted(self):
        legacy = dict(self.capture)
        legacy["schema"] = "m3-shadow-calibration-v1"
        legacy.pop("capture_sha256")
        with self.assertRaises(SystemExit):
            _MOD._verify_provenance(legacy, self.review, self.manifest)

    def test_reviewed_percentile_is_preserved(self):
        self.review["tail_percentile_ppm"] = 990000
        self.review["captures"][0]["capture_sha256"] = self.capture["capture_sha256"]
        self.review["results"][0]["tail_percentile_ppm"] = 990000
        text = _MOD.emit_candidate(self.capture, self.review, "review-hash")
        self.assertIn("profile 101 2 1 1234 0 100 16 990000", text)


if __name__ == "__main__":
    unittest.main()
