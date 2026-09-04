import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("simulate_scheduler.py")
SPEC = importlib.util.spec_from_file_location("simulate_scheduler", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SchedulerSimulationTest(unittest.TestCase):
    def write_csv(self, path, fieldnames, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root):
        expert_fields = ["layer", "expert", "invocations", "gpu_hits",
                         "cpu_fallback", "gpu_resident", "cpu_get_ms",
                         "cpu_matmul_ms"]
        expert_rows = []
        for layer in range(2):
            for expert in range(4):
                count = [10, 8, 3, 1][expert] if layer == 0 else [1, 3, 8, 10][expert]
                expert_rows.append({"layer": layer, "expert": expert,
                                    "invocations": count, "gpu_hits": 0,
                                    "cpu_fallback": count, "gpu_resident": 0,
                                    "cpu_get_ms": count * 0.1,
                                    "cpu_matmul_ms": count * 0.2})
        layer_fields = ["layer", "calls", "routes", "gpu_hits", "cpu_fallback",
                        "cpu_fallback_ms", "qt_take_ms", "qt_take_calls"]
        layer_rows = [{"layer": layer, "calls": 1,
                       "routes": sum(row["invocations"] for row in expert_rows
                                      if row["layer"] == layer),
                       "gpu_hits": 0, "cpu_fallback": 0,
                       "cpu_fallback_ms": 1.0, "qt_take_ms": 1.0,
                       "qt_take_calls": 1} for layer in range(2)]
        placement_rows = [{"layer": layer, "expert": expert,
                           "gpu_resident": int((layer, expert) == (0, 0))}
                          for layer in range(2) for expert in range(4)]
        timing_fields = ["token", "layer", "layer_begin_ms", "gpu_runnable_ms",
                         "gpu_submit_ms", "gpu_complete_ms", "cpu_begin_ms",
                         "cpu_complete_ms", "merge_begin_ms", "layer_complete_ms",
                         "gpu_dispatch_delay_ms", "gpu_lane_ms", "cpu_lane_ms",
                         "imbalance_ms", "exposed_merge_ms", "layer_makespan_ms",
                         "gpu_slack_ms"]
        timing_rows = []
        for layer in range(2):
            timing_rows.append({"token": 0, "layer": layer,
                                "layer_begin_ms": 0, "gpu_runnable_ms": 1,
                                "gpu_submit_ms": 1, "gpu_complete_ms": 3,
                                "cpu_begin_ms": 1, "cpu_complete_ms": 4,
                                "merge_begin_ms": 4, "layer_complete_ms": 4,
                                "gpu_dispatch_delay_ms": 0, "gpu_lane_ms": 2,
                                "cpu_lane_ms": 3, "imbalance_ms": 1,
                                "exposed_merge_ms": 0, "layer_makespan_ms": 4,
                                "gpu_slack_ms": 1})
        expert = root / "expert.csv"
        layer = root / "layer.csv"
        placement = root / "placement.csv"
        timing = root / "timing.csv"
        self.write_csv(expert, expert_fields, expert_rows)
        self.write_csv(layer, layer_fields, layer_rows)
        self.write_csv(placement, ["layer", "expert", "gpu_resident"], placement_rows)
        self.write_csv(timing, timing_fields, timing_rows)
        return expert, layer, placement, timing

    def test_current_is_anchored_and_slack_is_reported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as directory:
            expert, layer, placement, timing = self.fixture(Path(directory))
            envelope, lanes, _ = MODULE.load_timing(timing, 2)
            all_keys, model, route, _, _ = MODULE.build_model(
                MODULE.read_csv(expert), MODULE.read_csv(layer), envelope,
                lanes, 2, 4, MODULE.read_placement(placement))
            current = MODULE.read_placement(placement)
            report = MODULE.predict("current", current, all_keys, route, model, 2)
            self.assertAlmostEqual(report["layers"]["0"]["cpu_lane_ms"], 3.0)
            self.assertAlmostEqual(report["layers"]["0"]["gpu_lane_ms"], 2.0)
            self.assertIn("gpu_slack_ms", report["layers"]["0"])

    def test_fixed_capacity_policies_and_balancing(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[2]) as directory:
            expert, layer, placement, timing = self.fixture(Path(directory))
            envelope, lanes, _ = MODULE.load_timing(timing, 2)
            all_keys, model, route, _, _ = MODULE.build_model(
                MODULE.read_csv(expert), MODULE.read_csv(layer), envelope,
                lanes, 2, 4, MODULE.read_placement(placement))
            current = MODULE.read_placement(placement)
            policies = MODULE.placement_policies(
                all_keys, route, model, current, 2, 2, 4)
            self.assertEqual(len(policies["global_hot"]), 2)
            self.assertEqual(len(policies["layer_balanced"]), 2)
            self.assertEqual(len(policies["makespan_balanced"]), 2)
            reports = {name: MODULE.predict(name, value, all_keys, route, model, 2)
                       for name, value in policies.items()}
            self.assertTrue(all("classification_counts" in report
                                for report in reports.values()))


if __name__ == "__main__":
    unittest.main()
