import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ai_eval.contracts import canonical_json, digest, load_benchmark, load_run, system_id


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)

    def benchmark(self, *, spec=None, samples=None, name="benchmark.json"):
        specification = {"name": "fixture", "revision": "1", "samples": "samples.jsonl",
                         "metrics": [{"name": "accuracy", "scorer": "exact_match", "direction": "maximize"}]}
        if spec:
            specification.update(spec)
        values = samples if samples is not None else [{"id": "case", "input": {"question": "x"}, "target": ["y"]}]
        (self.root / specification["samples"]).write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        path = self.root / name
        path.write_text(json.dumps(specification), encoding="utf-8")
        return path

    def run_data(self, *, imported=False):
        system = {"name": "model", "revision": "v1", "config": {"temperature": 0}}
        protocol = {"trials": 2, "seed": None if imported else 17}
        if imported:
            protocol["import_format"] = "fixture-v1"
        manifest = {
            "schema_version": "1.0", "run_id": "run-1",
            "benchmark": {"id": "benchmark-1", "name": "fixture", "revision": "1",
                          "sample_sha256": "sample-content", "protocol": protocol,
                          "metrics": [{"name": "accuracy", "scorer": "exact_match", "direction": "maximize", "params": {}}]},
            "system": system, "system_id": system_id(system), "planned_trials": 2,
            "samples": [{"id": "case", "group": "source", "weight": 2, "slices": {"domain": "math"},
                         "metadata": {"origin": "fixture"}}],
            "metadata": {"population": {"scope": "fixture samples"}},
        }
        row = {"schema_version": "1.0", "benchmark_id": "benchmark-1", "system_id": manifest["system_id"],
               "sample_id": "case", "group": "source", "weight": 2, "slices": {"domain": "math"},
               "trial": 0, "seed": None if imported else 17, "status": "ok", "output": "answer",
               "scores": {"accuracy": 1}, "errors": {}, "elapsed_seconds": None if imported else 0.2,
               "usage": {"tokens": 10}, "metadata": {"origin": {"sample": 1}}}
        return manifest, [row]

    def write_run(self, manifest, rows):
        path = self.root / "run"
        path.mkdir(exist_ok=True)
        (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (path / "results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_benchmark_defaults_and_structured_values(self):
        value = load_benchmark(self.benchmark())
        self.assertEqual(value["protocol"], {"trials": 1, "seed": 0})
        self.assertEqual(value["metrics"][0]["params"], {})
        self.assertEqual(value["samples"][0], {"id": "case", "input": {"question": "x"}, "target": ["y"],
                                                "group": "case", "weight": 1, "slices": {}, "metadata": {}})
        self.assertEqual(value["sample_sha256"], hashlib.sha256((self.root / "samples.jsonl").read_bytes()).hexdigest())
        self.assertEqual(len(value["benchmark_id"]), 64)

    def test_benchmark_identity_survives_physical_relocation_and_key_order(self):
        first = load_benchmark(self.benchmark())
        original = json.loads((self.root / "benchmark.json").read_text())
        original["samples"] = "renamed.jsonl"
        (self.root / "renamed.jsonl").write_bytes((self.root / "samples.jsonl").read_bytes())
        (self.root / "second.json").write_text(json.dumps(dict(reversed(list(original.items())))))
        second = load_benchmark(self.root / "second.json")
        self.assertEqual(first["benchmark_id"], second["benchmark_id"])
        (self.root / "renamed.jsonl").write_text('{"id":"case","input":"changed","target":"y"}\n')
        self.assertNotEqual(second["benchmark_id"], load_benchmark(self.root / "second.json")["benchmark_id"])

    def test_defaults_are_canonical_for_benchmark_and_system(self):
        first = load_benchmark(self.benchmark())
        second = load_benchmark(self.benchmark(spec={"schema_version": "1.0", "metadata": {},
                                                    "protocol": {"seed": 0, "trials": 1}}))
        self.assertEqual(first["benchmark_id"], second["benchmark_id"])
        self.assertEqual(system_id({"name": "m", "revision": "1"}),
                         system_id({"config": {}, "revision": "1", "name": "m"}))
        self.assertNotEqual(system_id({"name": "m", "revision": "1", "config": {"seed": 1}}),
                            system_id({"name": "m", "revision": "1", "config": {"seed": 2}}))

    def test_benchmark_rejects_ambiguous_or_invalid_populations(self):
        cases = [[{"id": "a", "input": 1, "target": 1}, {"id": "a", "input": 2, "target": 2}],
                 [{"id": "a", "input": 1}], [{"id": "a", "input": 1, "target": 1, "weight": 0}],
                 [{"id": "a", "input": 1, "target": 1, "weight": True}],
                 [{"id": "a", "input": 1, "target": 1, "slices": {"label": []}}], []]
        for samples in cases:
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                load_benchmark(self.benchmark(samples=samples))

    def test_metrics_and_protocol_are_explicitly_validated(self):
        changes = [{"metrics": []}, {"metrics": [{"name": "a", "scorer": "exact_match", "direction": "higher"}]},
                   {"protocol": {"trials": 0}}, {"protocol": {"trials": True}}, {"protocol": {"seed": None}},
                   {"schema_version": "2.0"}]
        for spec in changes:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                load_benchmark(self.benchmark(spec=spec))

    def test_json_rejects_duplicate_keys_and_nonfinite_values(self):
        path = self.benchmark()
        for raw in ['{"name":"a","name":"b"}', '{"name":NaN}', '{"name":1e999}']:
            path.write_text(raw)
            with self.subTest(raw=raw), self.assertRaisesRegex(ValueError, "invalid JSON"):
                load_benchmark(path)
        with self.assertRaises(ValueError):
            system_id({"name": "m", "revision": "1", "config": {"temperature": float("inf")}})
        with self.assertRaises(ValueError):
            canonical_json({1: "ambiguous key"})

    def test_partial_run_retains_metadata_without_inventing_results(self):
        manifest, rows = self.run_data()
        result = load_run(self.write_run(manifest, rows))
        self.assertEqual(result["manifest"], manifest)
        self.assertEqual(result["results"], rows)
        self.assertEqual(len(result["results"]), 1)
        empty = load_run(self.write_run(manifest, []))
        self.assertEqual(empty["results"], [])

    def test_imported_unknown_seed_and_elapsed_are_retained_as_null(self):
        manifest, rows = self.run_data(imported=True)
        result = load_run(self.write_run(manifest, rows))
        self.assertIsNone(result["results"][0]["seed"])
        self.assertIsNone(result["results"][0]["elapsed_seconds"])
        self.assertEqual(result["manifest"]["metadata"]["population"]["scope"], "fixture samples")

    def test_native_seed_and_elapsed_cannot_be_unknown(self):
        for key in ("seed", "elapsed_seconds"):
            manifest, rows = self.run_data()
            rows[0][key] = None
            with self.subTest(key=key), self.assertRaises(ValueError):
                load_run(self.write_run(manifest, rows))

    def test_run_rejects_identity_population_and_trial_mismatch(self):
        mutations = [lambda m, r: m.update(system_id="different"),
                     lambda m, r: m.update(planned_trials=3),
                     lambda m, r: r[0].update(benchmark_id="different"),
                     lambda m, r: r[0].update(system_id="different"),
                     lambda m, r: r[0].update(sample_id="absent"),
                     lambda m, r: r[0].update(group="other"),
                     lambda m, r: r[0].update(weight=3),
                     lambda m, r: r[0].update(slices={"domain": "other"}),
                     lambda m, r: r[0].update(trial=2),
                     lambda m, r: r[0].update(trial=True)]
        for mutation in mutations:
            manifest, rows = self.run_data()
            mutation(manifest, rows)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                load_run(self.write_run(manifest, rows))

    def test_run_rejects_duplicate_outcomes(self):
        manifest, rows = self.run_data()
        with self.assertRaisesRegex(ValueError, "duplicate sample/trial"):
            load_run(self.write_run(manifest, rows + copy.deepcopy(rows)))

    def test_scores_require_declared_finite_numbers(self):
        for scores in ({"other": 1}, {}, {"accuracy": True}, {"accuracy": float("nan")}, {"accuracy": float("inf")}):
            manifest, rows = self.run_data()
            rows[0]["scores"] = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                load_run(self.write_run(manifest, rows))

    def test_execution_failure_and_wrong_answer_remain_distinct(self):
        manifest, rows = self.run_data()
        wrong = copy.deepcopy(rows[0])
        wrong["scores"]["accuracy"] = 0
        failure = copy.deepcopy(rows[0])
        failure.update(trial=1, status="target_error", output=None,
                       scores={"accuracy": None}, errors={"target": {"message": "timeout"}})
        result = load_run(self.write_run(manifest, [wrong, failure]))
        self.assertEqual(result["results"][0]["scores"]["accuracy"], 0)
        self.assertIsNone(result["results"][1]["scores"]["accuracy"])
        failure["scores"]["accuracy"] = 0
        with self.assertRaisesRegex(ValueError, "target_error"):
            load_run(self.write_run(manifest, [failure]))

    def test_one_failed_scorer_does_not_erase_other_scores(self):
        manifest, rows = self.run_data()
        manifest["benchmark"]["metrics"].append({"name": "other", "scorer": "custom", "direction": "minimize"})
        rows[0].update(status="score_error", scores={"accuracy": 1, "other": None},
                       errors={"other": {"message": "undefined"}})
        self.assertEqual(load_run(self.write_run(manifest, rows))["results"][0]["scores"]["accuracy"], 1)
        rows[0]["errors"] = {}
        with self.assertRaisesRegex(ValueError, "unavailable metrics"):
            load_run(self.write_run(manifest, rows))

    def test_hash_is_canonical_and_distinguishes_scalar_types(self):
        self.assertEqual(digest({"a": 1, "b": 2}), digest({"b": 2, "a": 1}))
        self.assertNotEqual(digest({"a": False}), digest({"a": 0}))


if __name__ == "__main__":
    unittest.main()
