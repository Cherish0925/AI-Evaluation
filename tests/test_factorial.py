import json
import math
from pathlib import Path
import tempfile
import unittest

from ai_eval.contracts import system_id
from ai_eval.factorial import CELLS, factorial


class FactorialTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def runs(self, values, *, trials=1, groups=None, weights=None, slices=None,
             direction="maximize", same_system=False):
        identifiers = sorted(values[0])
        paths = []
        for cell, scores in zip(CELLS, values):
            path = self.root / cell
            path.mkdir(exist_ok=True)
            system = {"name": "model", "revision": "1" if same_system else cell,
                      "config": {} if same_system else {"A": int(cell[0]), "B": int(cell[1])}}
            samples = [{"id": name, "group": (groups or {}).get(name, name),
                        "weight": (weights or {}).get(name, 1),
                        "slices": (slices or {}).get(name, {})} for name in identifiers]
            manifest = {
                "schema_version": "1.0", "run_id": f"run-{cell}", "system": system,
                "system_id": system_id(system), "samples": samples,
                "planned_trials": len(samples) * trials,
                "benchmark": {"id": "benchmark", "name": "factorial-example", "revision": "1",
                              "sample_sha256": "same-cases",
                              "protocol": {"trials": trials, "seed": 3},
                              "metrics": [{"name": "quality", "scorer": "numeric", "params": {},
                                           "direction": direction}]},
            }
            rows = []
            for sample in samples:
                for trial, value in enumerate(scores[sample["id"]]):
                    rows.append({
                        "schema_version": "1.0", "benchmark_id": "benchmark",
                        "system_id": manifest["system_id"], "sample_id": sample["id"],
                        "group": sample["group"], "weight": sample["weight"], "slices": sample["slices"],
                        "trial": trial, "seed": trial, "output": value,
                        "scores": {"quality": value}, "status": "ok" if value is not None else "target_error",
                        "errors": {} if value is not None else {"target": "unavailable"},
                        "elapsed_seconds": 0.1, "usage": {}, "metadata": {},
                    })
            (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths.append(path)
        return paths

    def analyze(self, paths, **kwargs):
        return factorial(*paths, "quality", resamples=400, **kwargs)

    def test_main_simple_and_interaction_have_declared_scale(self):
        values = [{"x": [offset], "y": [10 + offset], "z": [100 + offset]}
                  for offset in (0, 2, 3, 4)]
        result = self.analyze(self.runs(values), factor_a="retrieval", factor_b="memory")
        expected = {"main_a": 1.5, "main_b": 2.5, "interaction": -1,
                    "a_at_b0": 2, "a_at_b1": 1, "b_at_a0": 3, "b_at_a1": 2}
        for name, value in expected.items():
            with self.subTest(effect=name):
                effect = result["effects"][name]
                self.assertAlmostEqual(effect["estimate"], value)
                self.assertAlmostEqual(effect["improvement"], value)
                self.assertAlmostEqual(effect["interval"]["lower"], value)
                self.assertAlmostEqual(effect["interval"]["upper"], value)
        self.assertEqual(result["kind"], "factorial")
        self.assertEqual(result["factors"], {"a": "retrieval", "b": "memory", "levels": [0, 1]})
        self.assertEqual([row["cell"] for row in result["conditions"]], list(CELLS))
        self.assertEqual(result["effects"]["interaction"]["coefficients"],
                         {"00": 1, "10": -1, "01": -1, "11": 1})

    def test_averages_attempts_then_weights_samples(self):
        values = [{"a": [0, 2], "b": [0, 0]}, {"a": [2, 4], "b": [10, 10]},
                  {"a": [3, 5], "b": [0, 0]}, {"a": [4, 6], "b": [14, 14]}]
        result = self.analyze(self.runs(values, trials=2, weights={"a": 3, "b": 1}))
        self.assertAlmostEqual(result["effects"]["main_a"]["estimate"], 4.125)
        self.assertAlmostEqual(result["effects"]["main_b"]["estimate"], 2.375)
        self.assertAlmostEqual(result["effects"]["interaction"]["estimate"], 0.25)
        self.assertEqual(result["conditions"][0]["mean"], 0.75)
        self.assertEqual(result["samples"][0]["scores"]["00"], 1)
        self.assertEqual(result["coverage"]["conditions"]["00"]["planned_trials"], 4)

    def test_resamples_paired_groups_not_independent_cells_or_cases(self):
        zero = {name: [0] for name in ("a0", "a1", "b0", "b1")}
        changed = {"a0": [0], "a1": [2], "b0": [0], "b1": [2]}
        groups = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
        result = self.analyze(self.runs([zero, changed, zero, changed], groups=groups))
        interval = result["effects"]["main_a"]["interval"]
        self.assertEqual((interval["lower"], interval["upper"]), (1, 1))
        self.assertEqual(interval["independent_groups"], 2)
        independent = self.analyze(self.runs([zero, changed, zero, changed]))
        self.assertLess(independent["effects"]["main_a"]["interval"]["lower"], 1)
        self.assertGreater(independent["effects"]["main_a"]["interval"]["upper"], 1)

    def test_one_common_population_and_wholly_excluded_slice(self):
        values = [{"a": [0], "b": [100], "c": [0]}, {"a": [2], "b": [], "c": [100]},
                  {"a": [3], "b": [100], "c": [None]}, {"a": [4], "b": [100], "c": [100]}]
        paths = self.runs(values, weights={"a": 1, "b": 99, "c": 2},
                          slices={"a": {"domain": "x"}, "b": {"domain": "y"}, "c": {"domain": "y"}})
        with self.assertRaisesRegex(ValueError, "2 samples lack complete"):
            self.analyze(paths)
        result = self.analyze(paths, missing="complete_cases", slice_by="domain")
        self.assertEqual([row["mean"] for row in result["conditions"]], [0, 2, 3, 4])
        self.assertEqual([row["sample_id"] for row in result["exclusions"]], ["b", "c"])
        self.assertEqual(result["coverage"]["retained_samples"], 1)
        self.assertAlmostEqual(result["coverage"]["retained_weight_fraction"], 1 / 102)
        self.assertIsNone(result["effects"]["interaction"]["interval"])
        self.assertEqual(result["effects"]["interaction"]["interval_reason"], "fewer_than_two_independent_groups")
        excluded = result["slices"][1]
        self.assertEqual(excluded["value"], "y")
        self.assertIsNone(excluded["effects"]["main_a"]["estimate"])
        self.assertEqual(excluded["effects"]["main_a"]["interval_reason"], "no_scored_samples")
        self.assertEqual(excluded["excluded_sample_ids"], ["b", "c"])
        self.assertTrue(all(row["mean"] is None for row in excluded["conditions"]))

    def test_minimize_reverses_estimates_and_interval_endpoints(self):
        values = [{"a": [10], "b": [10]}, {"a": [8], "b": [6]},
                  {"a": [10], "b": [10]}, {"a": [8], "b": [6]}]
        result = self.analyze(self.runs(values, direction="minimize"))
        effect = result["effects"]["main_a"]
        self.assertAlmostEqual(effect["estimate"], -3)
        self.assertAlmostEqual(effect["improvement"], 3)
        self.assertEqual(effect["improvement_interval"]["lower"], -effect["interval"]["upper"])
        self.assertEqual(effect["improvement_interval"]["upper"], -effect["interval"]["lower"])

    def test_duplicate_runs_and_incompatible_measurement_are_rejected(self):
        values = [{"a": [value]} for value in (0, 1, 2, 3)]
        paths = self.runs(values)
        with self.assertRaisesRegex(ValueError, "four distinct runs"):
            self.analyze([paths[0], paths[0], paths[2], paths[3]])
        manifest = json.loads((paths[3] / "manifest.json").read_text())
        manifest["benchmark"]["protocol"]["seed"] = 100
        (paths[3] / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "same execution protocol"):
            self.analyze(paths)

    def test_four_recordings_can_share_a_model_description(self):
        values = [{"a": [value]} for value in (0, 1, 2, 3)]
        result = self.analyze(self.runs(values, same_system=True))
        self.assertEqual(len({cell["system_id"] for cell in result["conditions"]}), 1)
        self.assertEqual(len({cell["run_id"] for cell in result["conditions"]}), 4)

    def test_no_complete_population_has_no_factorial_estimate(self):
        paths = self.runs([{"a": [0]}, {"a": [1]}, {"a": []}, {"a": [2]}])
        with self.assertRaisesRegex(ValueError, "no complete factorial"):
            self.analyze(paths, missing="complete_cases")

    def test_large_common_offsets_cancel_without_intermediate_overflow(self):
        paths = self.runs([{name: [1e308] for name in ("a", "b")} for _ in CELLS],
                          weights={"a": 1e308, "b": 1e308})
        result = self.analyze(paths)
        self.assertTrue(all(row["estimate"] == 0 for row in result["effects"].values()))
        self.assertTrue(all(math.isfinite(row["mean"]) for row in result["conditions"]))

    def test_settings_require_named_distinct_factors_and_known_missing_policy(self):
        for kwargs in ({"factor_a": ""}, {"factor_b": "A"}, {"factor_a": 1},
                       {"missing": "zero"}, {"slice_by": ""}, {"confidence": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.analyze(["unread"] * 4, **kwargs)


if __name__ == "__main__":
    unittest.main()
