import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ai_eval.analysis import compare, pass_at_k, summarize
from ai_eval.engine import run


def fixture(values, *, trials=1, groups=None, weights=None, slices=None, direction="maximize", system="a"):
    ids = sorted(values)
    samples = [{"id": name, "group": (groups or {}).get(name, name),
                "weight": (weights or {}).get(name, 1), "slices": (slices or {}).get(name, {})} for name in ids]
    rows = []
    for sample in samples:
        for trial, value in enumerate(values[sample["id"]]):
            rows.append({"sample_id": sample["id"], "group": sample["group"], "weight": sample["weight"],
                         "slices": sample["slices"], "trial": trial, "scores": {"quality": value},
                         "status": "ok" if value is not None else "target_error", "elapsed_seconds": 0.5,
                         "usage": {"tokens": 10} if trial == 0 else {}})
    return {"manifest": {"schema_version": "1.0", "run_id": f"run-{system}", "system_id": system,
                          "benchmark": {"id": "benchmark", "sample_sha256": "bytes",
                                        "protocol": {"trials": trials, "seed": 17},
                                        "metrics": [{"name": "quality", "scorer": "exact_match", "direction": direction}]},
                          "samples": samples, "planned_trials": len(samples) * trials}, "results": rows}


def summary(run, **kwargs):
    with patch("ai_eval.analysis._load", return_value=run):
        return summarize("run", "quality", resamples=500, **kwargs)


def comparison(left, right, **kwargs):
    with patch("ai_eval.analysis._load", side_effect=[left, right]):
        return compare("left", "right", "quality", resamples=500, **kwargs)


class AnalysisTests(unittest.TestCase):
    def test_trials_reduce_within_sample_before_weighting(self):
        run = fixture({"a": [0, 1], "b": [1]}, trials=2, weights={"a": 3, "b": 1})
        result = summary(run)
        self.assertEqual(result["mean"], 0.625)
        self.assertEqual(result["coverage"]["planned_trials"], 4)
        self.assertEqual(result["coverage"]["scored_trials"], 3)
        self.assertEqual(result["coverage"]["complete_samples"], 1)
        self.assertEqual(result["coverage"]["complete_weight_fraction"], 0.75)
        self.assertEqual(result["resources"]["elapsed_seconds"], {"sum": 1.5, "observations": 3})
        self.assertEqual(result["resources"]["usage"]["tokens"], {"sum": 20, "observations": 2})

    def test_missing_and_unscored_are_separate_and_not_zero_scores(self):
        run = fixture({"a": [1, None], "b": []}, trials=2)
        result = summary(run)
        self.assertEqual(result["mean"], 1)
        self.assertEqual(result["coverage"]["unrecorded_trials"], 2)
        self.assertEqual(result["coverage"]["unscored_trials"], 1)
        self.assertEqual(result["status_counts"], {"ok": 1, "target_error": 1})
        self.assertIsNone(result["interval"])
        self.assertEqual(result["interval_reason"], "fewer_than_two_independent_groups")

    def test_no_scores_returns_absent_estimate(self):
        result = summary(fixture({"a": [None], "b": []}))
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["interval"])
        self.assertEqual(result["interval_reason"], "no_scored_samples")

    def test_group_resampling_preserves_complementary_samples(self):
        values = {"a0": [0], "a1": [1], "b0": [0], "b1": [1]}
        groups = {"a0": "a", "a1": "a", "b0": "b", "b1": "b"}
        result = summary(fixture(values, groups=groups))
        self.assertEqual(result["interval"]["independent_groups"], 2)
        self.assertEqual(result["interval"]["lower"], 0.5)
        self.assertEqual(result["interval"]["upper"], 0.5)
        independent = summary(fixture(values))
        self.assertLess(independent["interval"]["lower"], 0.5)
        self.assertGreater(independent["interval"]["upper"], 0.5)

    def test_bootstrap_is_order_independent_and_reproducible(self):
        run = fixture({"a": [0, 1], "b": [0, 0], "c": [1, 1]}, trials=2)
        original = summary(run, seed=11)
        reversed_run = copy.deepcopy(run)
        reversed_run["manifest"]["samples"].reverse()
        reversed_run["results"].reverse()
        self.assertEqual(original, summary(reversed_run, seed=11))

    def test_slices_distinguish_null_false_zero_and_missing(self):
        slices = {"a": {"domain": None}, "b": {"domain": False}, "c": {"domain": 0}, "d": {}}
        result = summary(fixture({name: [1] for name in slices}, slices=slices), slice_by="domain")
        self.assertEqual(len(result["slices"]), 4)
        self.assertEqual(sum(row["missing"] for row in result["slices"]), 1)
        self.assertTrue(all(row["coverage"]["total_samples"] == 1 for row in result["slices"]))

    def test_paired_interval_uses_case_differences(self):
        left = fixture({"a": [0], "b": [10], "c": [100]})
        right = fixture({"a": [2], "b": [12], "c": [102]}, system="b")
        result = comparison(left, right)
        self.assertEqual(result["delta"], 2)
        self.assertEqual(result["interval"]["lower"], 2)
        self.assertEqual(result["interval"]["upper"], 2)
        self.assertEqual(result["improvement"], 2)

    def test_lower_is_better_reverses_interval_and_improvement(self):
        left = fixture({"a": [3], "b": [5]}, direction="minimize")
        right = fixture({"a": [2], "b": [2]}, direction="minimize", system="b")
        result = comparison(left, right)
        self.assertEqual(result["delta"], -2)
        self.assertEqual(result["improvement"], 2)
        self.assertEqual(result["improvement_interval"]["lower"], -result["interval"]["upper"])
        self.assertEqual(result["improvement_interval"]["upper"], -result["interval"]["lower"])

    def test_pairing_rejects_changed_protocol_metric_or_population(self):
        left = fixture({"a": [1], "b": [0]})
        changes = [lambda r: r["manifest"]["benchmark"].update(id="other"),
                   lambda r: r["manifest"]["benchmark"]["protocol"].update(seed=18),
                   lambda r: r["manifest"]["benchmark"]["metrics"][0].update(params={"new": 1}),
                   lambda r: r["manifest"]["samples"][0].update(group="different"),
                   lambda r: r["manifest"]["samples"][0].update(weight=2),
                   lambda r: r["manifest"]["benchmark"].update(sample_sha256="other")]
        for change in changes:
            right = copy.deepcopy(left)
            change(right)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "requires"):
                comparison(left, right)

    def test_native_runs_remain_comparable_after_analysis_package_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "samples.jsonl").write_text("".join(
                json.dumps({"id": str(index), "input": index, "target": 1}) + "\n"
                for index in range(2)), encoding="utf-8")
            benchmark = root / "benchmark.json"
            benchmark.write_text(json.dumps({
                "schema_version": "1.0", "name": "release-comparison", "revision": "1",
                "samples": "samples.jsonl", "protocol": {"trials": 1, "seed": 17},
                "metrics": [{"name": "quality", "scorer": "exact_match", "direction": "maximize"}],
            }), encoding="utf-8")
            left, right = root / "left", root / "right"
            with patch("ai_eval.__version__", "0.1.0"):
                run(benchmark, left, system={"name": "model", "revision": "old"},
                    target=lambda request: {"output": request["input"]})
            with patch("ai_eval.__version__", "0.2.0"):
                run(benchmark, right, system={"name": "model", "revision": "new"},
                    target=lambda request: {"output": 1})
            originals = [(path / "manifest.json").read_bytes() for path in (left, right)]
            descriptors = [json.loads(raw)["benchmark"]["metrics"][0] for raw in originals]
            self.assertEqual([item["implementation"]["version"] for item in descriptors], ["0.1.0", "0.2.0"])
            self.assertEqual(descriptors[0]["implementation"]["module_sha256"],
                             descriptors[1]["implementation"]["module_sha256"])
            result = compare(left, right, "quality", resamples=100)
            self.assertEqual(result["left_mean"], .5)
            self.assertEqual(result["right_mean"], 1)
            self.assertEqual(result["delta"], .5)
            self.assertEqual(originals, [(path / "manifest.json").read_bytes() for path in (left, right)])

    def test_native_package_version_exception_preserves_measurement_definition(self):
        left = fixture({"a": [0], "b": [1]})
        left["manifest"]["benchmark"]["metrics"][0].update(
            bounds=[0, 1], implementation={"kind": "native", "package": "ai-evaluation",
                "version": "0.1.0", "scorer": "exact_match", "module_sha256": "a" * 64})
        changes = [
            ("code", lambda metric: metric["implementation"].update(module_sha256="b" * 64)),
            ("implementation scorer", lambda metric: metric["implementation"].update(scorer="numeric")),
            ("kind", lambda metric: metric["implementation"].update(kind="plugin")),
            ("package", lambda metric: metric["implementation"].update(package="another-package")),
            ("bounds", lambda metric: metric.update(bounds=[0, 2])),
            ("parameters", lambda metric: metric.update(params={"casefold": True})),
            ("direction", lambda metric: metric.update(direction="minimize")),
        ]
        for name, change in changes:
            right = copy.deepcopy(left)
            descriptor = right["manifest"]["benchmark"]["metrics"][0]
            descriptor["implementation"]["version"] = "0.2.0"
            change(descriptor)
            with self.subTest(change=name), self.assertRaisesRegex(ValueError, "same metric definition"):
                comparison(left, right)

    def test_native_package_versions_require_a_valid_recorded_code_identity(self):
        for code in (None, "", "a" * 63, "g" * 64, True):
            left = fixture({"a": [0], "b": [1]})
            implementation = {"kind": "native", "package": "ai-evaluation", "version": "0.1.0",
                              "scorer": "exact_match"}
            if code is not None:
                implementation["module_sha256"] = code
            left["manifest"]["benchmark"]["metrics"][0]["implementation"] = implementation
            right = copy.deepcopy(left)
            right["manifest"]["benchmark"]["metrics"][0]["implementation"]["version"] = "0.2.0"
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "same metric definition"):
                comparison(left, right)

    def test_plugin_version_changes_remain_incompatible(self):
        left = fixture({"a": [0], "b": [1]})
        left["manifest"]["benchmark"]["metrics"][0].update(
            scorer="research_score", implementation={"kind": "plugin", "name": "research_score",
                "entry_point": "research.scorers:score", "distribution": "research-scorers", "version": "1.0"})
        right = copy.deepcopy(left)
        right["manifest"]["benchmark"]["metrics"][0]["implementation"]["version"] = "2.0"
        with self.assertRaisesRegex(ValueError, "same metric definition"):
            comparison(left, right)

    def test_complete_cases_reports_every_exclusion(self):
        left = fixture({"a": [0, 1], "b": [1, 1], "c": [1, 1]}, trials=2)
        right = fixture({"a": [1, 1], "b": [1], "c": [1, None]}, trials=2, system="b")
        with self.assertRaisesRegex(ValueError, "2 samples lack complete"):
            comparison(left, right)
        result = comparison(left, right, missing="complete_cases")
        self.assertEqual(result["delta"], 0.5)
        self.assertEqual(result["coverage"]["excluded_samples"], 2)
        self.assertEqual([row["sample_id"] for row in result["exclusions"]], ["b", "c"])
        self.assertIsNone(result["interval"])

    def test_comparison_refuses_empty_complete_case_population(self):
        with self.assertRaisesRegex(ValueError, "no complete paired"):
            comparison(fixture({"a": [1]}), fixture({"a": []}), missing="complete_cases")

    def test_comparison_slices_retain_wholly_excluded_slices(self):
        left = fixture({"a": [0], "b": [0], "c": [0]}, slices={"a": {"domain": "x"}, "b": {"domain": "x"}, "c": {"domain": "y"}})
        right = copy.deepcopy(left)
        for row in right["results"]:
            row["scores"]["quality"] = 1 if row["sample_id"] != "c" else None
        result = comparison(left, right, missing="complete_cases", slice_by="domain")
        self.assertEqual(len(result["slices"]), 2)
        self.assertEqual(result["slices"][0]["value"], "x")
        self.assertEqual(result["slices"][0]["delta"], 1)
        self.assertEqual(result["slices"][1]["value"], "y")
        self.assertIsNone(result["slices"][1]["delta"])
        self.assertIsNone(result["slices"][1]["interval"])
        self.assertEqual(result["slices"][1]["coverage"]["excluded_samples"], 1)
        self.assertEqual(result["slices"][1]["excluded_sample_ids"], ["c"])

    def test_score_for_otherwise_failed_row_is_retained(self):
        run = fixture({"a": [1], "b": [0]})
        run["results"][0]["status"] = "score_error"
        self.assertEqual(summary(run)["mean"], 0.5)

    def test_pass_at_k_matches_exact_combinatorial_estimator(self):
        run = fixture({"a": [0, 0, 1, 1], "b": [0, 0, 0, 0]}, trials=4, weights={"a": 3, "b": 1})
        with patch("ai_eval.analysis._load", return_value=run):
            result = pass_at_k("run", "quality", 2)
        expected = 1 - math.comb(2, 2) / math.comb(4, 2)
        self.assertAlmostEqual(result["samples"][0]["estimate"], expected)
        self.assertAlmostEqual(result["estimate"], 3 * expected / 4)

    def test_pass_at_k_requires_complete_binary_success_trials(self):
        cases = [(fixture({"a": [1]}), 2), (fixture({"a": [1, None]}, trials=2), 1),
                 (fixture({"a": [0.5]}), 1), (fixture({"a": [1]}, direction="minimize"), 1)]
        for run, k in cases:
            with self.subTest(run=run, k=k), patch("ai_eval.analysis._load", return_value=run), self.assertRaises(ValueError):
                pass_at_k("run", "quality", k)

    def test_single_group_and_large_weights_remain_well_defined(self):
        run = fixture({"a": [1], "b": [0]}, groups={"a": "same", "b": "same"}, weights={"a": 1e308, "b": 1e308})
        result = summary(run)
        self.assertEqual(result["mean"], 0.5)
        self.assertIsNone(result["interval"])

    def test_bootstrap_duplicate_groups_do_not_overflow_weights(self):
        run = fixture({"a": [1], "b": [0]}, weights={"a": 1e308, "b": 1e308})
        result = summary(run)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result["interval"]["lower"], 0)
        self.assertEqual(result["interval"]["upper"], 1)
        self.assertEqual(result["interval"]["seed"], 0)

    def test_coverage_reveals_missing_high_weight_samples(self):
        run = fixture({"a": [None], "b": [1]}, weights={"a": 99, "b": 1})
        result = summary(run)
        self.assertEqual(result["coverage"]["scored_fraction"], 0.5)
        self.assertAlmostEqual(result["coverage"]["scored_weight_fraction"], 0.01)

    def test_imported_population_scope_survives_all_analysis(self):
        run = fixture({"a": [1], "b": [0]})
        provenance = {"scope": "observed sample IDs", "unobserved_tasks_known": False}
        run["manifest"]["metadata"] = {"population": provenance}
        self.assertEqual(summary(run)["population"], provenance)
        self.assertEqual(comparison(run, run)["population"], {"left": provenance, "right": provenance})
        with patch("ai_eval.analysis._load", return_value=run):
            self.assertEqual(pass_at_k("run", "quality", 1)["population"], provenance)

    def test_invalid_analysis_parameters(self):
        for params in ({"confidence": 1}, {"confidence": float("nan")}, {"resamples": 0}, {"seed": True}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                summarize("unused", "quality", **params)


if __name__ == "__main__":
    unittest.main()
