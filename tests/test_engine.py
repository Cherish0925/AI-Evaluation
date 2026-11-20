import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai_eval.analysis import compare
from ai_eval.contracts import load_run
from ai_eval.engine import _RunLease, run
from ai_eval.targets import CommandTarget, PredictionTarget, resolve_plugin


SYSTEM = {"name": "authored-test-target", "revision": "fixture-1"}


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.benchmark = self.root / "benchmark.json"

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, *, count=3, trials=1, metrics=None):
        samples = [{"id": f"sample-{i}", "input": {"value": i}, "target": i,
                    "metadata": {"note": "authored arithmetic fixture"}, "group": f"group-{i // 2}",
                    "slices": {"parity": i % 2}} for i in range(count)]
        (self.root / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8")
        spec = {"schema_version": "1.0", "name": "authored-engine-fixture", "revision": "1",
                "samples": "samples.jsonl", "protocol": {"trials": trials, "seed": 19},
                "metrics": metrics or [{"name": "accuracy", "scorer": "numeric", "direction": "maximize"}]}
        self.benchmark.write_text(json.dumps(spec), encoding="utf-8")
        return samples

    def test_concurrent_execution_and_seed_stability_without_reference_leak(self):
        self.fixture(count=4, trials=2)
        requests = []
        lock = threading.Lock()
        active, peak = 0, 0

        def target(request):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                requests.append(request)
            time.sleep(.01)
            with lock:
                active -= 1
            return {"output": request["input"]["value"], "usage": {"tokens": 3}}

        first = run(self.benchmark, self.root / "parallel", system=SYSTEM, target=target, workers=2)
        second = run(self.benchmark, self.root / "serial", system=SYSTEM, target=target)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 2)
        self.assertEqual(len(first["results"]), 8)
        self.assertTrue(all(row["scores"]["accuracy"] == 1 for row in first["results"]))
        self.assertTrue(all(set(request) == {"sample_id", "input", "trial", "seed", "metadata"} for request in requests))
        seeds = lambda result: {(row["sample_id"], row["trial"]): row["seed"] for row in result["results"]}
        self.assertEqual(seeds(first), seeds(second))
        self.assertEqual(len(set(seeds(first).values())), 8)

    def test_target_and_scorer_failures_are_distinct_and_partial_scores_survive(self):
        self.fixture(metrics=[{"name": "numeric", "scorer": "numeric", "direction": "maximize"},
                              {"name": "exact", "scorer": "exact_match", "direction": "maximize"}])

        def target(request):
            if request["sample_id"] == "sample-0":
                raise RuntimeError("authored target failure")
            return {"output": "1" if request["sample_id"] == "sample-1" else 2}

        result = run(self.benchmark, self.root / "run", system=SYSTEM, target=target)
        rows = {row["sample_id"]: row for row in result["results"]}
        self.assertEqual(rows["sample-0"]["status"], "target_error")
        self.assertEqual(rows["sample-0"]["scores"], {"numeric": None, "exact": None})
        self.assertEqual(rows["sample-0"]["errors"]["target"]["type"], "RuntimeError")
        self.assertEqual(rows["sample-1"]["status"], "score_error")
        self.assertEqual(rows["sample-1"]["scores"], {"numeric": None, "exact": 0})
        self.assertEqual(rows["sample-2"]["status"], "ok")

    def test_resume_recovers_only_incomplete_tail_and_retains_failed_outcomes(self):
        self.fixture(count=4)
        calls = []

        def target(request):
            calls.append(request["sample_id"])
            if request["sample_id"] == "sample-0":
                raise ValueError("durable failure")
            return {"output": request["input"]["value"]}

        directory = self.root / "run"
        first = run(self.benchmark, directory, system=SYSTEM, target=target)
        results_path = directory / "results.jsonl"
        lines = results_path.read_bytes().splitlines(keepends=True)
        results_path.write_bytes(b"".join(lines[:2]) + b'{"schema_version":')
        calls.clear()
        resumed = run(self.benchmark, directory, system=SYSTEM, target=target, resume=True)
        self.assertEqual(calls, ["sample-2", "sample-3"])
        self.assertEqual(resumed["manifest"]["run_id"], first["manifest"]["run_id"])
        self.assertEqual(resumed["results"][0], first["results"][0])
        self.assertEqual(len(resumed["results"]), 4)

    def test_resume_retains_complete_record_without_newline_and_handles_initial_empty_log(self):
        self.fixture(count=1)
        calls = []

        def target(request):
            calls.append(request["sample_id"])
            return {"output": 0}

        directory = self.root / "run"
        run(self.benchmark, directory, system=SYSTEM, target=target)
        path = directory / "results.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        run(self.benchmark, directory, system=SYSTEM, target=target, resume=True)
        self.assertEqual(len(calls), 1)
        path.unlink()
        run(self.benchmark, directory, system=SYSTEM, target=target, resume=True)
        self.assertEqual(len(calls), 2)

    def test_resume_does_not_discard_corruption_in_the_middle(self):
        self.fixture(count=2)
        target = lambda request: {"output": request["input"]["value"]}
        directory = self.root / "run"
        run(self.benchmark, directory, system=SYSTEM, target=target)
        path = directory / "results.jsonl"
        malformed = b"corrupt complete record\n" + b'{"incomplete":'
        path.write_bytes(malformed)
        with self.assertRaises(ValueError):
            run(self.benchmark, directory, system=SYSTEM, target=target, resume=True)
        self.assertEqual(path.read_bytes(), malformed)

    def test_prediction_identity_blocks_changed_data_and_system_on_resume(self):
        self.fixture(count=1)
        path = self.root / "predictions.jsonl"
        path.write_text('{"sample_id":"sample-0","output":0}\n', encoding="utf-8")
        target = PredictionTarget(path)
        directory = self.root / "run"
        run(self.benchmark, directory, system=SYSTEM, target=target)
        with self.assertRaises(ValueError):
            run(self.benchmark, directory, system={**SYSTEM, "revision": "fixture-2"}, target=target, resume=True)
        resumed = run(self.benchmark, directory, system=SYSTEM, target=target, workers=2, resume=True)
        self.assertEqual(len(resumed["results"]), 1)
        path.write_text('{"sample_id":"sample-0","output":1}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            run(self.benchmark, directory, system=SYSTEM, target=PredictionTarget(path), resume=True)

    def test_prediction_trial_override_missing_result_and_duplicate_detection(self):
        self.fixture(count=2, trials=2)
        path = self.root / "predictions.jsonl"
        rows = [{"sample_id": "sample-0", "output": 0}, {"sample_id": "sample-0", "trial": 1, "output": 7}]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        result = run(self.benchmark, self.root / "run", system=SYSTEM, target=PredictionTarget(path))
        scores = {(row["sample_id"], row["trial"]): row["scores"]["accuracy"] for row in result["results"]}
        self.assertEqual(scores, {("sample-0", 0): 1, ("sample-0", 1): 0, ("sample-1", 0): None, ("sample-1", 1): None})
        with path.open("a", encoding="utf-8") as source:
            source.write(json.dumps(rows[0]) + "\n")
        with self.assertRaises(ValueError):
            PredictionTarget(path)

    def test_command_process_protocol_timeout_and_failure_diagnostics(self):
        self.fixture(count=1)
        code = "import json,sys; r=json.load(sys.stdin); print(json.dumps({'output':r['input']['value']}))"
        result = run(self.benchmark, self.root / "success", system=SYSTEM, target=CommandTarget([sys.executable, "-c", code]))
        self.assertEqual(result["results"][0]["scores"]["accuracy"], 1)
        empty_arg = CommandTarget([sys.executable, "-c", "import json,sys; print(json.dumps({'output':sys.argv[1]}))", ""])
        self.assertEqual(empty_arg({"input": None})["output"], "")
        for name, code, kind in (("timeout", "import time; time.sleep(3)", "TimeoutError"),
                                 ("exit", "import sys; print('fixture diagnostic',file=sys.stderr); sys.exit(7)", "RuntimeError"),
                                 ("json", "print('not JSON')", "JSONDecodeError")):
            with self.subTest(name=name):
                target = CommandTarget([sys.executable, "-c", code], timeout=.1)
                result = run(self.benchmark, self.root / name, system=SYSTEM, target=target)
                self.assertEqual(result["results"][0]["status"], "target_error")
                self.assertEqual(result["results"][0]["errors"]["target"]["type"], kind)
                if name == "exit":
                    self.assertIn("fixture diagnostic", result["results"][0]["errors"]["target"]["message"])

    def test_command_script_bytes_participate_in_resume_identity(self):
        self.fixture(count=1)
        script = self.root / "target.py"
        script.write_text('print(\'{"output":0}\')\n', encoding="utf-8")
        directory = self.root / "run"
        target = CommandTarget([sys.executable, str(script)])
        run(self.benchmark, directory, system=SYSTEM, target=target)
        script.write_text('print(\'{"output":1}\')\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            run(self.benchmark, directory, system=SYSTEM, target=target, resume=True)

    def test_changed_scorer_implementation_blocks_resume_and_comparison(self):
        self.fixture(count=2, metrics=[{"name": "quality", "scorer": "fixture-score", "direction": "maximize"}])
        target = lambda request: {"output": request["input"]["value"]}
        first_point = SimpleNamespace(value="fixture.scorer:score", dist=SimpleNamespace(name="fixture-scoring", version="1"),
                                      load=lambda: lambda output, reference: 1)
        second_point = SimpleNamespace(value="fixture.scorer:score", dist=SimpleNamespace(name="fixture-scoring", version="2"),
                                       load=lambda: lambda output, reference: 0)
        left, right = self.root / "left", self.root / "right"
        with patch("ai_eval.metrics.entry_points", return_value=[first_point]):
            run(self.benchmark, left, system=SYSTEM, target=target)
        with patch("ai_eval.metrics.entry_points", return_value=[second_point]):
            with self.assertRaises(ValueError):
                run(self.benchmark, left, system=SYSTEM, target=target, resume=True)
            run(self.benchmark, right, system=SYSTEM, target=target)
        with self.assertRaises(ValueError):
            compare(left, right, "quality", resamples=20)

    def test_plugin_factory_configuration_is_bound_to_execution(self):
        self.fixture(count=1)
        point = SimpleNamespace(value="fixture.plugin:factory", dist=SimpleNamespace(version="1.2.3"),
                                load=lambda: lambda config: lambda request: {"output": config["value"]})
        with patch("ai_eval.targets.entry_points", return_value=[point]):
            target = resolve_plugin("fixture", {"value": 0})
            result = run(self.benchmark, self.root / "run", system=SYSTEM, target=target)
            self.assertEqual(result["results"][0]["scores"]["accuracy"], 1)
            self.assertEqual(result["manifest"]["execution"]["target"]["distribution_version"], "1.2.3")
            with self.assertRaises(ValueError):
                run(self.benchmark, self.root / "run", system=SYSTEM, target=resolve_plugin("fixture", {"value": 1}), resume=True)

    def test_os_lease_prevents_simultaneous_writers_and_new_run_overwrite(self):
        self.fixture(count=1)
        directory = self.root / "run"
        directory.mkdir()
        target = lambda request: {"output": 0}
        with _RunLease(directory), self.assertRaises(ValueError):
            run(self.benchmark, directory, system=SYSTEM, target=target)
        run(self.benchmark, directory, system=SYSTEM, target=target)
        with self.assertRaises(ValueError):
            run(self.benchmark, directory, system=SYSTEM, target=target)
        self.assertEqual(len(load_run(directory)["results"]), 1)

    def test_malformed_target_usage_is_not_a_valid_observation(self):
        self.fixture(count=1)
        result = run(self.benchmark, self.root / "run", system=SYSTEM,
                     target=lambda request: {"output": 0, "usage": {"tokens": "unknown"}})
        self.assertEqual(result["results"][0]["status"], "target_error")
        self.assertIsNone(result["results"][0]["scores"]["accuracy"])


if __name__ == "__main__":
    unittest.main()
