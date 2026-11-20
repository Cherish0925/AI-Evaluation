import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_eval.engine import run
from ai_eval.studies import compare_systems


def fixture(values, system, *, trials=1, groups=None, weights=None, slices=None,
            costs=None, direction="maximize"):
    samples = [{"id": name, "group": (groups or {}).get(name, name),
                "weight": (weights or {}).get(name, 1), "slices": (slices or {}).get(name, {})}
               for name in sorted(values)]
    rows = []
    for sample in samples:
        for trial, value in enumerate(values[sample["id"]]):
            resource = (costs or {}).get(sample["id"], [10] * trials)[trial]
            rows.append({"sample_id": sample["id"], "group": sample["group"], "weight": sample["weight"],
                         "slices": sample["slices"], "trial": trial, "scores": {"quality": value},
                         "status": "ok" if value is not None else "target_error", "elapsed_seconds": 0.5,
                         "usage": {"tokens": resource} if resource is not None else {}})
    return {"manifest": {"schema_version": "1.0", "run_id": f"run-{system}", "system_id": system,
                          "system": {"name": "authored-system", "revision": system, "config": {}},
                          "benchmark": {"id": "benchmark", "name": "authored-study", "revision": "1",
                                        "sample_sha256": "bytes", "protocol": {"trials": trials, "seed": 17},
                                        "metrics": [{"name": "quality", "scorer": "numeric", "direction": direction}]},
                          "samples": samples, "planned_trials": len(samples) * trials}, "results": rows}


def study(runs, **kwargs):
    with patch("ai_eval.studies._load", side_effect=runs):
        return compare_systems([f"study-{i}" for i in range(len(runs))], "quality", resamples=200, **kwargs)


class StudyTests(unittest.TestCase):
    def test_three_systems_share_one_complete_population(self):
        runs = [fixture({"a": [0], "b": [10], "c": [100], "d": [1000]}, "a"),
                fixture({"a": [1], "b": [None], "c": [101], "d": [1001]}, "b"),
                fixture({"a": [2], "b": [12], "c": [], "d": [1002]}, "c")]
        with self.assertRaisesRegex(ValueError, "2 samples lack complete"):
            study(runs)
        result = study(runs, missing="complete_cases")
        self.assertEqual([entry["mean"] for entry in result["systems"]], [500, 501, 502])
        self.assertEqual([entry["delta"] for entry in result["systems"]], [0, 1, 2])
        self.assertEqual([item["sample_id"] for item in result["samples"]], ["a", "d"])
        self.assertEqual(result["coverage"]["retained_weight_fraction"], 0.5)
        self.assertEqual(result["coverage"]["systems"][2]["unrecorded_trials"], 1)
        self.assertEqual(result["exclusions"][0]["systems"][1]["missing_scores"], 1)
        self.assertEqual(result["exclusions"][1]["systems"][2]["observed_trials"], 0)
        self.assertIsNone(result["pareto_frontier"])

    def test_weighted_trial_means_and_resources_use_the_same_cohort(self):
        left = fixture({"a": [0, 1], "b": [1, 1]}, "a", trials=2,
                       weights={"a": 3}, costs={"a": [10, 30], "b": [100, 100]})
        right = fixture({"a": [1, 1], "b": [1, 1]}, "b", trials=2,
                        weights={"a": 3}, costs={"a": [30, 50], "b": [200, 200]})
        result = study([left, right], resource="usage.tokens")
        self.assertEqual(result["systems"][0]["mean"], .625)
        self.assertEqual(result["systems"][1]["mean"], 1)
        self.assertEqual(result["systems"][0]["resource_mean"], 40)
        self.assertEqual(result["systems"][1]["resource_mean"], 80)
        self.assertEqual(result["systems"][1]["improvement"], .375)
        self.assertEqual(result["pareto_frontier"], [0, 1])
        self.assertEqual(result["resource_aggregation"], "sample_weighted_mean_per_attempt")
        right["results"][-1]["usage"].clear()
        with self.assertRaisesRegex(ValueError, "resource observations"):
            study([left, right], resource="usage.tokens")
        partial = study([left, right], resource="usage.tokens", missing="complete_cases")
        self.assertEqual([entry["mean"] for entry in partial["systems"]], [.5, 1])
        self.assertEqual([entry["resource_mean"] for entry in partial["systems"]], [20, 40])
        self.assertEqual(partial["coverage"]["retained_weight_fraction"], .75)
        self.assertEqual(partial["coverage"]["systems"][1]["resource_missing_trials"], 1)

    def test_group_bootstrap_preserves_related_samples_and_paired_effects(self):
        groups = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
        left = fixture({"a0": [0], "a1": [1], "b0": [0], "b1": [1]}, "a", groups=groups)
        right = fixture({"a0": [2], "a1": [3], "b0": [2], "b1": [3]}, "b", groups=groups)
        result = study([left, right])
        self.assertEqual(result["systems"][0]["interval"]["lower"], .5)
        self.assertEqual(result["systems"][0]["interval"]["upper"], .5)
        self.assertEqual(result["systems"][1]["delta_interval"]["lower"], 2)
        self.assertEqual(result["systems"][1]["delta_interval"]["upper"], 2)
        self.assertEqual(result["coverage"]["retained_groups"], 2)
        reordered = copy.deepcopy([left, right])
        for item in reordered:
            item["manifest"]["samples"].reverse()
            item["results"].reverse()
        self.assertEqual(result, study(reordered))

    def test_pareto_frontier_keeps_ties_and_quality_resource_tradeoffs(self):
        qualities = [1, 2, 2, 2, 0]
        costs = [10, 20, 20, 30, 10]
        runs = [fixture({"a": [q], "b": [q]}, str(i), costs={"a": [c], "b": [c]})
                for i, (q, c) in enumerate(zip(qualities, costs))]
        result = study(runs, resource="usage.tokens")
        self.assertEqual(result["pareto_frontier"], [0, 1, 2])
        self.assertEqual(result["systems"][1]["dominated_by"], [])
        self.assertEqual(result["systems"][2]["dominated_by"], [])
        self.assertEqual(result["systems"][3]["dominated_by"], [1, 2])
        self.assertEqual(result["systems"][4]["dominated_by"], [0])
        self.assertEqual(result["frontier_basis"], "observed_point_estimates")

    def test_minimize_direction_and_selected_baseline(self):
        left = fixture({"a": [3], "b": [5]}, "a", direction="minimize")
        right = fixture({"a": [2], "b": [2]}, "b", direction="minimize")
        result = study([left, right], baseline=1, resource="elapsed_seconds")
        self.assertEqual(result["baseline"]["index"], 1)
        self.assertEqual(result["systems"][0]["delta"], 2)
        self.assertEqual(result["systems"][0]["improvement"], -2)
        self.assertEqual(result["systems"][0]["improvement_interval"]["lower"],
                         -result["systems"][0]["delta_interval"]["upper"])
        self.assertEqual(result["pareto_frontier"], [1])
        self.assertEqual(result["resource_unit"], "seconds")

    def test_empty_slices_and_distinct_scalar_labels_remain_visible(self):
        labels = {"a": {"domain": False}, "b": {"domain": 0}, "c": {"domain": None}, "d": {}}
        left = fixture({key: [1] for key in labels}, "a", slices=labels)
        right = fixture({key: [1] if key != "c" else [] for key in labels}, "b", slices=labels)
        result = study([left, right], missing="complete_cases", slice_by="domain", resource="usage.tokens")
        self.assertEqual(len(result["slices"]), 4)
        empty = next(item for item in result["slices"] if item["value"] is None and not item["missing"])
        self.assertEqual(empty["coverage"]["retained_samples"], 0)
        self.assertEqual(empty["coverage"]["systems"][1]["resource_missing_trials"], 1)
        self.assertEqual(empty["excluded_sample_ids"], ["c"])
        self.assertTrue(all(entry["mean"] is None and entry["interval"] is None for entry in empty["systems"]))
        self.assertEqual(empty["pareto_frontier"], [])

    def test_changed_protocol_measurement_or_population_cannot_be_pooled(self):
        left = fixture({"a": [1], "b": [0]}, "a")
        changes = [lambda r: r["manifest"]["benchmark"].update(id="different"),
                   lambda r: r["manifest"]["benchmark"]["protocol"].update(seed=99),
                   lambda r: r["manifest"]["benchmark"]["metrics"][0].update(direction="minimize"),
                   lambda r: r["manifest"]["samples"][0].update(weight=20)]
        for change in changes:
            right = fixture({"a": [1], "b": [0]}, "b")
            change(right)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "requires"):
                study([left, right])

    def test_repeated_run_identity_and_empty_cohort_are_not_new_comparisons(self):
        left = fixture({"a": [1]}, "a")
        with self.assertRaisesRegex(ValueError, "unique run identities"):
            study([left, copy.deepcopy(left)])
        with self.assertRaisesRegex(ValueError, "no complete samples"):
            study([left, fixture({"a": []}, "b")], missing="complete_cases")
        with self.assertRaisesRegex(ValueError, "unique run directories"):
            compare_systems(["same", "./same"], "quality")

    def test_invalid_resource_is_distinct_from_an_unobserved_resource(self):
        left = fixture({"a": [1]}, "a")
        for value in (-1, True, "100", float("inf"), 10 ** 400):
            right = fixture({"a": [1]}, "b", costs={"a": [value]})
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                study([left, right], resource="usage.tokens")
        for resource in ("tokens", "usage.", "usage. ", 42):
            with self.subTest(resource=resource), self.assertRaisesRegex(ValueError, "resource must"):
                compare_systems(["a", "b"], "quality", resource=resource)

    def test_single_group_large_weights_and_system_labels(self):
        runs = [fixture({"a": [0], "b": [1]}, "a", weights={"a": 1e308, "b": 1e308}, groups={"a": "g", "b": "g"}),
                fixture({"a": [1], "b": [1]}, "b", weights={"a": 1e308, "b": 1e308}, groups={"a": "g", "b": "g"})]
        runs[1]["manifest"]["system"]["revision"] = "a"
        result = study(runs)
        self.assertEqual(result["systems"][0]["mean"], .5)
        self.assertIsNone(result["systems"][0]["interval"])
        self.assertEqual(result["coverage"]["retained_weight_fraction"], 1)
        self.assertNotEqual(result["systems"][0]["label"], result["systems"][1]["label"])
        for baseline in (-1, 2, True):
            with self.subTest(baseline=baseline), self.assertRaisesRegex(ValueError, "baseline must"):
                study(runs, baseline=baseline)

    def test_native_run_records_produce_a_complete_study(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = [{"id": str(i), "input": i, "target": i, "group": f"g{i // 2}"} for i in range(4)]
            (root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
            benchmark = {"name": "authored-arithmetic", "revision": "1", "samples": "samples.jsonl",
                         "protocol": {"trials": 2, "seed": 3},
                         "metrics": [{"name": "accuracy", "scorer": "exact_match", "direction": "maximize"}]}
            source = root / "benchmark.json"
            source.write_text(json.dumps(benchmark), encoding="utf-8")
            paths = [root / "baseline", root / "candidate"]
            run(source, paths[0], system={"name": "arithmetic", "revision": "baseline"},
                target=lambda request: {"output": request["input"] if request["input"] % 2 else -1, "usage": {"tokens": 2}})
            run(source, paths[1], system={"name": "arithmetic", "revision": "candidate"},
                target=lambda request: {"output": request["input"], "usage": {"tokens": 3}})
            result = compare_systems(paths, "accuracy", resource="usage.tokens", resamples=40)
            self.assertEqual([system["mean"] for system in result["systems"]], [.5, 1])
            self.assertEqual(result["systems"][1]["improvement_interval"]["lower"], .5)
            self.assertEqual(result["systems"][1]["improvement_interval"]["upper"], .5)
            self.assertEqual(result["coverage"]["retained_trials_per_system"], 8)
            json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
