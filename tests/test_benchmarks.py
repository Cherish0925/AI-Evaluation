import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_eval.benchmarks import analyze_benchmark
from ai_eval.engine import run


def fixture(values, *, system, trials=1, groups=None, weights=None, slices=None,
            direction="maximize", bounds=None, scorer="authored"):
    samples = [{"id": name, "group": (groups or {}).get(name, name),
                "weight": (weights or {}).get(name, 1), "slices": (slices or {}).get(name, {})}
               for name in sorted(values)]
    metric = {"name": "quality", "scorer": scorer, "direction": direction}
    if bounds is not None:
        metric["bounds"] = bounds
    rows = [{"sample_id": sample["id"], "trial": trial, "scores": {"quality": value}}
            for sample in samples for trial, value in enumerate(values[sample["id"]])]
    return {"manifest": {"schema_version": "1.0", "run_id": f"run-{system}", "system_id": system,
                         "system": {"name": system, "revision": "1"},
                         "benchmark": {"id": "authored", "name": "authored", "revision": "1",
                                       "sample_sha256": "same", "protocol": {"trials": trials, "seed": 0},
                                       "metrics": [metric]},
                         "samples": samples}, "results": rows}


def analyze(runs, **kwargs):
    with patch("ai_eval.benchmarks._load", side_effect=runs):
        return analyze_benchmark([f"run-{index}" for index in range(len(runs))], "quality", **kwargs)


def sample(result, identifier):
    return next(item for item in result["samples"] if item["sample_id"] == identifier)


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_corrected_discrimination_excludes_the_whole_group(self):
        groups = {"focal": "shared", "sibling": "shared", "rest": "other"}
        runs = [fixture({"focal": [value], "sibling": [100 * value], "rest": [1 - value]},
                        system=str(index), groups=groups)
                for index, value in enumerate([0, .5, 1])]
        result = analyze(runs)
        focal = sample(result, "focal")
        self.assertEqual(focal["system_means"], [0, .5, 1])
        self.assertEqual(focal["rest_system_means"], [1, .5, 0])
        self.assertAlmostEqual(focal["discrimination"], -1)
        self.assertIsNone(focal["discrimination_reason"])
        # Including the related sibling would reverse the sign; leave-item-out is insufficient.
        self.assertAlmostEqual(sample(result, "sibling")["discrimination"], -1)

    def test_weighted_rest_retains_small_groups_after_dominant_group_exclusion(self):
        weights = {"focal": 1e308, "rest_a": 1, "rest_b": 3}
        runs = [fixture({"focal": [value], "rest_a": [1 - value], "rest_b": [value]},
                        system=str(index), weights=weights)
                for index, value in enumerate([0, .5, 1])]
        focal = sample(analyze(runs), "focal")
        for actual, expected in zip(focal["rest_system_means"], [.25, .5, .75]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(focal["discrimination"], 1)

    def test_trials_average_within_sample_and_systems_receive_equal_weight(self):
        runs = [fixture({"a": a, "b": b}, system=str(index), trials=2, weights={"a": 3, "b": 1}, bounds=[0, 1])
                for index, (a, b) in enumerate([([0, 0], [1, 1]), ([0, 1], [1, 1]), ([1, 1], [1, 1])])]
        result = analyze(runs)
        self.assertEqual(sample(result, "a")["system_means"], [0, .5, 1])
        self.assertEqual(sample(result, "a")["facility"], .5)
        self.assertAlmostEqual(sample(result, "a")["panel_stddev"], math.sqrt(1 / 6))
        self.assertEqual([item["mean"] for item in result["panel"]], [.25, .625, 1])
        self.assertEqual(result["summary"]["panel_mean"], .625)
        self.assertTrue(result["binary_scores"])
        self.assertEqual(result["score_bounds"]["basis"], "declared_metric_bounds")
        self.assertEqual(result["summary"]["ceiling_samples"], 1)
        self.assertEqual(result["summary"]["floor_samples"], 0)

    def test_three_distinct_nonconstant_systems_and_other_groups_are_needed(self):
        two = [fixture({"a": [value], "b": [value]}, system=str(value)) for value in (0, 1)]
        self.assertEqual(sample(analyze(two), "a")["discrimination_reason"], "fewer_than_three_systems")
        constant_item = [fixture({"a": [1], "b": [value]}, system=str(value)) for value in (0, .5, 1)]
        self.assertEqual(sample(analyze(constant_item), "a")["discrimination_reason"], "constant_item_scores")
        self.assertEqual(sample(analyze(constant_item), "b")["discrimination_reason"], "constant_rest_scores")
        one_group = [fixture({"a": [value], "b": [value]}, system=str(value), groups={"a": "g", "b": "g"})
                     for value in (0, .5, 1)]
        self.assertEqual(sample(analyze(one_group), "a")["discrimination_reason"], "no_other_independent_groups")
        constant = [fixture({"a": [1], "b": [1]}, system=str(value)) for value in (0, 1, 2)]
        result = analyze(constant)
        self.assertEqual(result["summary"]["discrimination_defined"], 0)
        self.assertEqual(result["summary"]["constant_samples"], 2)
        self.assertTrue(all(item["discrimination"] is None for item in result["samples"]))

    def test_common_complete_cohort_reports_each_incomplete_system_cell(self):
        common = {"a": [0, 1], "b": [1, 1], "c": [0, 0], "d": [1, 1]}
        weights = {"a": 1, "b": 4, "c": 3, "d": 2}
        slices = {name: {"domain": "retained" if name in ("a", "d") else "excluded"} for name in common}
        runs = [fixture(common, system=str(index), trials=2, weights=weights, slices=slices) for index in range(3)]
        runs[1] = fixture({**common, "b": [1]}, system="1", trials=2, weights=weights, slices=slices)
        runs[2] = fixture({**common, "c": [0, None]}, system="2", trials=2, weights=weights, slices=slices)
        with self.assertRaisesRegex(ValueError, "2 samples lack complete panel scores"):
            analyze(runs)
        result = analyze(runs, missing="complete_cases", slice_by="domain")
        self.assertEqual([item["sample_id"] for item in result["samples"]], ["a", "d"])
        self.assertEqual(result["coverage"]["retained_samples"], 2)
        self.assertAlmostEqual(result["coverage"]["retained_weight_fraction"], .3)
        self.assertEqual(result["panel"][1]["coverage"]["unrecorded_trials"], 1)
        self.assertEqual(result["panel"][2]["coverage"]["unscored_trials"], 1)
        self.assertTrue(all(abs(item["mean"] - 5 / 6) < 1e-12 for item in result["panel"]))
        self.assertEqual(result["exclusions"][0]["systems"][1], {
            "index": 1, "planned_trials": 2, "observed_trials": 1, "scored_trials": 1})
        self.assertEqual(result["exclusions"][1]["systems"][2]["scored_trials"], 1)
        excluded = next(item for item in result["slices"] if item["value"] == "excluded")
        self.assertIsNone(excluded["panel_mean"])
        self.assertEqual(excluded["excluded_sample_ids"], ["b", "c"])
        self.assertEqual(excluded["coverage"]["retained_samples"], 0)
        self.assertEqual(len(result["groups"]), 4)
        self.assertEqual(next(item for item in result["groups"] if item["group"] == "b")["samples"], 0)

    def test_empty_cohort_and_duplicate_system_identity_are_rejected(self):
        repeated = [fixture({"a": [0]}, system="same"), fixture({"a": [1]}, system="same")]
        repeated[1]["manifest"]["run_id"] = "different-attempt"
        with self.assertRaisesRegex(ValueError, "distinct system identities"):
            analyze(repeated)
        with self.assertRaisesRegex(ValueError, "no complete panel"):
            analyze([fixture({"a": [0]}, system="one"), fixture({"a": []}, system="two")], missing="complete_cases")

    def test_continuous_scores_do_not_imply_a_bounded_scale(self):
        runs = [fixture({"a": [value], "b": [value + 2]}, system=str(index))
                for index, value in enumerate([.2, .5, .8])]
        result = analyze(runs)
        self.assertIsNone(result["score_bounds"])
        self.assertIsNone(result["summary"]["floor_samples"])
        self.assertIsNone(result["summary"]["ceiling_samples"])
        self.assertTrue(all(item["facility"] is None for item in result["samples"]))
        self.assertAlmostEqual(sample(result, "a")["discrimination"], 1)

    def test_binary_observations_do_not_establish_metric_support(self):
        for scorer in ("authored", "latency", "brier", "log_loss", "imported_inspect_score", "plugin_score"):
            with self.subTest(scorer=scorer):
                runs = [fixture({"all_zero": [0], "all_one": [1], "varying": [index % 2]},
                                system=str(index), scorer=scorer) for index in range(3)]
                result = analyze(runs)
                self.assertTrue(result["binary_scores"])
                self.assertIsNone(result["score_bounds"])
                self.assertIsNone(result["summary"]["floor_samples"])
                self.assertIsNone(result["summary"]["ceiling_samples"])
                self.assertTrue(all(item["floor"] is None and item["ceiling"] is None and item["facility"] is None
                                    for item in result["samples"]))

    def test_known_native_scorer_contracts_supply_bounds_not_binary_success(self):
        for scorer in ("exact_match", "numeric", "set_f1", "token_f1", "recall_at_k", "ndcg_at_k"):
            with self.subTest(scorer=scorer):
                runs = [fixture({"floor": [0], "ceiling": [1]}, system=str(index), scorer=scorer) for index in range(3)]
                result = analyze(runs)
                self.assertEqual(result["score_bounds"], {"lower": 0, "upper": 1, "basis": "native_scorer_bounds"})
                self.assertEqual(result["summary"]["floor_samples"], 1)
                self.assertEqual(result["summary"]["ceiling_samples"], 1)
                self.assertEqual(sample(result, "ceiling")["facility"], 1)
        fractional = [fixture({"a": [.25 * index]}, system=str(index), scorer="token_f1") for index in range(3)]
        result = analyze(fractional)
        self.assertEqual(result["score_bounds"]["basis"], "native_scorer_bounds")
        self.assertFalse(result["binary_scores"])
        self.assertIsNone(sample(result, "a")["facility"])

    def test_declared_bounds_and_metric_direction_define_performance_boundaries(self):
        runs = [fixture({"best": [0], "worst": [10], "middle": [index + 1]},
                        system=str(index), direction="minimize", bounds=[0, 10]) for index in range(3)]
        result = analyze(runs)
        self.assertTrue(sample(result, "best")["ceiling"])
        self.assertTrue(sample(result, "worst")["floor"])
        self.assertFalse(sample(result, "middle")["floor"])
        self.assertIsNone(sample(result, "best")["facility"])
        self.assertEqual(result["score_bounds"]["basis"], "declared_metric_bounds")
        # Binary observations on a wider declared scale do not establish success rates.
        narrow_observations = [fixture({"a": [index % 2]}, system=str(index), bounds=[0, 10]) for index in range(3)]
        self.assertIsNone(sample(analyze(narrow_observations), "a")["facility"])

    def test_invalid_bounds_and_scores_outside_them_are_rejected(self):
        for bounds in ([1, 0], [0, 0], [False, 1], [0, math.inf], [0, 10 ** 1000], "0,1"):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "metric bounds"):
                analyze([fixture({"a": [0]}, system=str(index), bounds=bounds) for index in range(2)])
        with self.assertRaisesRegex(ValueError, "outside the declared"):
            analyze([fixture({"a": [index]}, system=str(index), bounds=[0, .5]) for index in range(2)])

    def test_slices_preserve_missing_null_false_and_zero(self):
        slices = {"a": {}, "b": {"domain": None}, "c": {"domain": False}, "d": {"domain": 0}}
        runs = [fixture({name: [index] for name in slices}, system=str(index), slices=slices) for index in range(3)]
        result = analyze(runs, slice_by="domain")
        self.assertEqual(len(result["slices"]), 4)
        self.assertEqual(sum(item["missing"] for item in result["slices"]), 1)
        self.assertTrue(all(item["samples"] == 1 for item in result["slices"]))

    def test_protocol_metric_and_sample_population_must_match(self):
        first = fixture({"a": [0], "b": [1]}, system="one")
        changes = [lambda run: run["manifest"]["benchmark"]["protocol"].update(seed=1),
                   lambda run: run["manifest"]["benchmark"]["metrics"][0].update(bounds=[0, 1]),
                   lambda run: run["manifest"]["samples"][0].update(group="different")]
        for change in changes:
            second = fixture({"a": [1], "b": [1]}, system="two")
            change(second)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "requires"):
                analyze([first, second])

    def test_panel_permutation_preserves_descriptive_quantities(self):
        runs = [fixture({"a": [value], "b": [value ** 2], "c": [1 - value]}, system=str(index))
                for index, value in enumerate([0, .25, .75, 1])]
        original = analyze(runs)
        reversed_runs = copy.deepcopy(list(reversed(runs)))
        for run_data in reversed_runs:
            run_data["manifest"]["samples"].reverse()
            run_data["results"].reverse()
        reordered = analyze(reversed_runs)
        for before, after in zip(original["samples"], reordered["samples"]):
            for key in ("panel_mean", "panel_stddev", "panel_range", "discrimination"):
                self.assertAlmostEqual(before[key], after[key])
        json.dumps(reordered, allow_nan=False)

    def test_native_run_directories_work_without_a_special_import_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "samples.jsonl").write_text("".join(json.dumps({"id": str(index), "input": index, "target": 1}) + "\n"
                                                         for index in range(3)), encoding="utf-8")
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps({"schema_version": "1.0", "name": "authored-panel", "revision": "1",
                                             "samples": "samples.jsonl", "protocol": {"trials": 1, "seed": 0},
                                             "metrics": [{"name": "quality", "scorer": "exact_match", "direction": "maximize"}]}),
                                 encoding="utf-8")
            directories = []
            for ability in range(3):
                directory = root / str(ability)
                run(benchmark, directory, system={"name": "authored-reference", "revision": str(ability)},
                    target=lambda request, ability=ability: {"output": int(request["input"] <= ability)})
                directories.append(directory)
            result = analyze_benchmark(directories, "quality")
            self.assertEqual([item["mean"] for item in result["panel"]], [1 / 3, 2 / 3, 1])
            self.assertEqual(result["benchmark"]["name"], "authored-panel")
            self.assertEqual(result["coverage"]["retained_samples"], 3)
            self.assertEqual(result["score_bounds"]["basis"], "native_scorer_bounds")


if __name__ == "__main__":
    unittest.main()
