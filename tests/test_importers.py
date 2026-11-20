"""Authored format fixtures, not records of real model or Gym executions.

Shapes follow Gym export.py/rollout.py and Inspect log/_log.py. Source URLs and
the checked source revisions are recorded in docs/scoring.md.
"""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from ai_eval.analysis import compare, summarize
from ai_eval.contracts import load_run
from ai_eval.importers import import_gym, import_inspect, import_plugin


def gym_row(task_seed=7, attempt=0, *, policy_revision="fixture-1", status="passed", reward=1, seconds=30):
    task = {"family": "fixture-task", "version": 1, "seed": task_seed, "split": "test", "difficulty": 1}
    return {"schema_version": 1, "episode_id": f"fixture-episode-{task_seed}-{attempt}", "task": task,
            "agent": "fixture-policy", "status": status, "reward": reward,
            "verification": {"passed": status == "passed", "reward": reward,
                             "reason": "authored fixture verdict", "metrics": {}} if reward is not None else None,
            "trajectory": [{"time": "2026-01-01T00:00:00+00:00", "step": 0, "kind": "observation",
                            "content": {"type": "start", "task": task, "prompt": "authored fixture", "tools": []}}],
            "rollout": {"suite_id": "fixture-suite", "task": task, "task_group_id": f"fixture-task-group-{task_seed}",
                        "group_id": f"fixture-policy-group-{task_seed}-{policy_revision}",
                        "rollout_id": f"fixture-rollout-{task_seed}-{attempt}", "attempt": attempt,
                        "epoch": 0, "sample_seed": task_seed + attempt,
                        "episode_id": f"fixture-episode-{task_seed}-{attempt}", "path": "fixture-path",
                        "policy": {"name": "fixture-policy", "revision": policy_revision, "generation": {"temperature": .5}},
                        "data_reference": {"revision": "fixture-data-v1", "sha256": "f" * 64},
                        "limits": {"actions": 8, "seconds": seconds}}, "policy_steps": []}


def inspect_log(model="fixture/model-1"):
    return {"version": 2, "status": "success",
            "eval": {"task": "fixture-inspect-task", "task_version": 1, "created": "2026-01-01T00:00:00Z",
                     "model": model, "model_args": {}, "config": {"epochs": 2, "token_limit": 1000},
                     "dataset": {"name": "authored-fixture", "samples": 2, "sample_ids": [1, 2]},
                     "packages": {"inspect_ai": "fixture-version"},
                     "scorers": [{"name": "accuracy", "options": {}}]},
            "plan": {"name": "plan", "steps": [{"solver": "generate", "params": {}}], "config": {"temperature": 0}},
            "samples": [{"id": sample_id, "epoch": epoch, "input": f"authored input {sample_id}",
                         "target": "authored target", "output": {"model": model, "choices": []},
                         "scores": {"accuracy": {"value": "C" if sample_id == 1 else "I"}},
                         "metadata": {}, "model_usage": {model: {"input_tokens": 8, "output_tokens": 2}},
                         "total_time": .5}
                        for sample_id in (1, 2) for epoch in (1, 2)]}


class ImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def gym(self, rows, name="gym"):
        source = self.root / f"{name}.jsonl"
        source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        output = self.root / name
        manifest = import_gym(source, output, system={"name": "authored policy", "revision": "fixture"},
                              benchmark_name="authored-gym-fixture")
        return output, manifest, source

    def inspect(self, log, name="inspect", **kwargs):
        source = self.root / f"{name}.json"
        source.write_text(json.dumps(log), encoding="utf-8")
        output = self.root / name
        manifest = import_inspect(source, output, scorer=kwargs.pop("scorer", "accuracy"), **kwargs)
        return output, manifest, source

    def test_gym_trials_are_clustered_and_source_bytes_preserved(self):
        rows = [gym_row(seed, attempt) for seed in (7, 8) for attempt in (0, 1)]
        output, manifest, source = self.gym(rows)
        run = load_run(output)
        self.assertEqual(manifest["planned_trials"], 4)
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual(len({row["group"] for row in run["results"]}), 2)
        self.assertEqual((output / "source.jsonl").read_bytes(), source.read_bytes())
        self.assertEqual(manifest["metadata"]["source"]["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(run["results"][0]["metadata"]["original"], rows[0])
        self.assertIsNone(run["results"][0]["elapsed_seconds"])
        self.assertEqual(summarize(output, "reward", resamples=20)["interval"]["independent_groups"], 2)

    def test_gym_unscored_is_missing_not_zero(self):
        output, _, _ = self.gym([gym_row(), gym_row(attempt=1, status="truncated", reward=None)])
        summary = summarize(output, "reward", resamples=20)
        self.assertEqual(summary["mean"], 1)
        self.assertEqual(summary["coverage"]["unscored_trials"], 1)
        self.assertEqual(summary["status_counts"]["target_error"], 1)
        self.assertIsNone(summary["interval"])

    def test_gym_missing_attempt_retains_slot_and_policy_does_not_define_benchmark(self):
        left, left_manifest, _ = self.gym([gym_row(), gym_row(attempt=2)], "left")
        right, right_manifest, _ = self.gym([gym_row(policy_revision="fixture-2"),
                                           gym_row(attempt=2, policy_revision="fixture-2")], "right")
        self.assertEqual(left_manifest["benchmark"]["id"], right_manifest["benchmark"]["id"])
        self.assertNotEqual(left_manifest["system_id"], right_manifest["system_id"])
        self.assertEqual(left_manifest["planned_trials"], 3)
        with self.assertRaises(ValueError):
            compare(left, right, "reward", resamples=20)

    def test_gym_policy_and_per_task_budget_mixtures_rejected(self):
        for other in (gym_row(attempt=1, policy_revision="changed"), gym_row(attempt=1, seconds=60)):
            with self.subTest(other=other), self.assertRaises(ValueError):
                self.gym([gym_row(), other])

    def test_gym_changed_budget_prevents_paired_comparison(self):
        left, _, _ = self.gym([gym_row()], "left")
        right, _, _ = self.gym([gym_row(seconds=60)], "right")
        with self.assertRaises(ValueError):
            compare(left, right, "reward", resamples=20)

    def test_gym_ordinary_export_uses_task_not_episode_as_independent_unit(self):
        rows = [gym_row(attempt=attempt) for attempt in (0, 1)]
        for row in rows:
            row.pop("rollout")
            row.pop("policy_steps")
        output, manifest, _ = self.gym(rows)
        self.assertEqual(len(manifest["samples"]), 1)
        self.assertEqual(manifest["planned_trials"], 2)
        self.assertFalse(manifest["benchmark"]["protocol"]["budget_recorded"])
        self.assertIsNone(summarize(output, "reward", resamples=20)["interval"])

    def test_inspect_epochs_model_identity_and_usage(self):
        output, manifest, source = self.inspect(inspect_log())
        run = load_run(output)
        self.assertEqual(len(manifest["samples"]), 2)
        self.assertEqual(manifest["planned_trials"], 4)
        self.assertEqual([row["trial"] for row in run["results"]], [0, 1, 0, 1])
        self.assertEqual(run["results"][0]["usage"], {"input_tokens": 8, "output_tokens": 2})
        self.assertEqual((output / "source.json").read_bytes(), source.read_bytes())
        self.assertEqual(summarize(output, "accuracy", resamples=20)["mean"], .5)
        other, other_manifest, _ = self.inspect(inspect_log("fixture/model-2"), "other")
        self.assertNotEqual(manifest["system_id"], other_manifest["system_id"])
        self.assertEqual(compare(output, other, "accuracy", resamples=20)["delta"], 0)

    def test_inspect_dictionary_score_selection_direction_and_groups(self):
        log = inspect_log()
        for sample in log["samples"]:
            sample["scores"] = {"accuracy": {"value": {"loss": .25, "quality": .75}}}
            sample["metadata"] = {"ai_eval": {"group": "one-document", "slices": {"domain": "fixture"}, "weight": 2}}
        output, manifest, _ = self.inspect(log, value_key="loss", direction="minimize")
        self.assertEqual(manifest["benchmark"]["metrics"][0]["direction"], "minimize")
        summary = summarize(output, "accuracy.loss", resamples=20)
        self.assertEqual(summary["mean"], .25)
        self.assertIsNone(summary["interval"])

    def test_inspect_errors_missing_epochs_and_invalidated_scores(self):
        log = inspect_log()
        log["samples"].pop()
        log["samples"][0]["error"] = {"message": "authored fixture failure", "traceback": "fixture"}
        log["samples"][1]["invalidation"] = {"reason": "authored invalidation fixture"}
        output, _, _ = self.inspect(log)
        run = load_run(output)
        summary = summarize(output, "accuracy", resamples=20)
        self.assertEqual(summary["coverage"]["unrecorded_trials"], 1)
        self.assertEqual(summary["coverage"]["unscored_trials"], 2)
        self.assertEqual(run["results"][0]["errors"]["target"], log["samples"][0]["error"])
        self.assertIsNone(run["results"][1]["scores"]["accuracy"])

    def test_inspect_no_guess_for_unsupported_score_or_aggregate_only(self):
        log = inspect_log()
        log["samples"][0]["scores"]["accuracy"]["value"] = "P"
        output, _, _ = self.inspect(log)
        self.assertIsNone(load_run(output)["results"][0]["scores"]["accuracy"])
        log.pop("samples")
        with self.assertRaises(ValueError):
            self.inspect(log, "aggregate-only")

    def test_inspect_nan_unscored_sentinel_and_boolean_scores(self):
        log = inspect_log()
        log["samples"][0]["scores"]["accuracy"] = {"value": float("nan"), "reason": "grader_failed"}
        log["samples"][1]["scores"]["accuracy"]["value"] = True
        output, _, source = self.inspect(log)
        self.assertEqual((output / "source.json").read_bytes(), source.read_bytes())
        rows = load_run(output)["results"]
        self.assertIsNone(rows[0]["scores"]["accuracy"])
        self.assertEqual(rows[0]["errors"]["accuracy"]["source_score"]["reason"], "grader_failed")
        self.assertEqual(rows[0]["metadata"]["original"]["scores"]["accuracy"]["value"], {"$source_nonfinite": "NaN"})
        self.assertEqual(rows[1]["scores"]["accuracy"], 1)

    def test_inspect_explicit_score_on_error_policy_preserves_failure_score(self):
        log = inspect_log()
        log["eval"]["config"]["score_on_error"] = True
        log["samples"][0]["error"] = {"message": "authored failed attempt"}
        log["samples"][0]["scores"]["accuracy"]["value"] = "I"
        output, _, _ = self.inspect(log)
        row = load_run(output)["results"][0]
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["scores"]["accuracy"], 0)
        self.assertEqual(row["metadata"]["source_error"], log["samples"][0]["error"])
        self.assertTrue(row["metadata"]["scored_on_error"])
        summary = summarize(output, "accuracy", resamples=20)
        self.assertEqual(summary["coverage"]["scored_trials"], 4)
        self.assertEqual(summary["mean"], .25)

    def test_inspect_condition_changes_prevent_pooling_or_comparison(self):
        log = inspect_log()
        log["samples"][0]["token_limit"] = 2
        with self.assertRaises(ValueError):
            self.inspect(log)
        log = inspect_log()
        left, _, _ = self.inspect(log, "left")
        log["eval"]["config"]["token_limit"] = 2000
        right, _, _ = self.inspect(log, "right")
        with self.assertRaises(ValueError):
            compare(left, right, "accuracy", resamples=20)
        log["config_updates"] = [{"changed": "token_limit"}]
        with self.assertRaises(ValueError):
            self.inspect(log, "changed")

    def test_inspect_changed_judge_role_prevents_paired_comparison(self):
        log = inspect_log()
        log["eval"]["model_roles"] = {"grader": {"model": "fixture/judge-1"}}
        left, _, _ = self.inspect(log, "left")
        log["eval"]["model_roles"]["grader"]["model"] = "fixture/judge-2"
        right, _, _ = self.inspect(log, "right")
        with self.assertRaises(ValueError):
            compare(left, right, "accuracy", resamples=20)

    def test_inspect_task_metadata_changes_measurement_identity(self):
        log = inspect_log()
        for sample in log["samples"]:
            sample["metadata"]["rubric"] = "authored original rubric"
        left, _, _ = self.inspect(log, "left")
        for sample in log["samples"]:
            sample["metadata"]["rubric"] = "authored changed rubric"
        right, _, _ = self.inspect(log, "right")
        with self.assertRaises(ValueError):
            compare(left, right, "accuracy", resamples=20)

    def test_inspect_default_judge_tracks_candidate_but_explicit_judge_allows_comparison(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                log = inspect_log("fixture/candidate-a")
                options = {"model": "fixture/fixed-judge"} if explicit else {}
                log["eval"]["scorers"] = [{"name": "model_graded_qa", "options": options}]
                for sample in log["samples"]:
                    sample["scores"]["model_graded_qa"] = sample["scores"].pop("accuracy")
                left, _, _ = self.inspect(log, f"left-{explicit}", scorer="model_graded_qa")
                log["eval"]["model"] = "fixture/candidate-b"
                right, _, _ = self.inspect(log, f"right-{explicit}", scorer="model_graded_qa")
                if explicit:
                    self.assertEqual(compare(left, right, "model_graded_qa", resamples=20)["delta"], 0)
                else:
                    with self.assertRaises(ValueError):
                        compare(left, right, "model_graded_qa", resamples=20)

    def test_import_plugin_invokes_adapter_and_validates_its_output(self):
        source = self.root / "plugin-source.json"
        source.write_text(json.dumps(inspect_log()), encoding="utf-8")
        calls = []

        def adapter(source, output, *, selected):
            calls.append(selected)
            import_inspect(source, output, scorer=selected)

        point = SimpleNamespace(load=lambda: adapter)
        with patch("ai_eval.importers.entry_points", return_value=[point]):
            result = import_plugin("fixture", source, self.root / "plugin-output", selected="accuracy")
        self.assertEqual(calls, ["accuracy"])
        self.assertEqual(len(result["results"]), 4)

        def malformed(source, output, **params):
            Path(output).mkdir()
            (Path(output) / "manifest.json").write_text("{}", encoding="utf-8")
            (Path(output) / "results.jsonl").write_text("", encoding="utf-8")

        point = SimpleNamespace(load=lambda: malformed)
        with patch("ai_eval.importers.entry_points", return_value=[point]), self.assertRaises(ValueError):
            import_plugin("fixture", source, self.root / "malformed")

    def test_duplicates_and_invalid_usage_do_not_write_run(self):
        row = gym_row()
        with self.assertRaises(ValueError):
            self.gym([row, copy.deepcopy(row)])
        self.assertFalse((self.root / "gym" / "manifest.json").exists())
        log = inspect_log()
        log["samples"][0]["model_usage"]["fixture/model-1"]["input_tokens"] = "unknown"
        with self.assertRaises(ValueError):
            self.inspect(log)
        self.assertFalse((self.root / "inspect" / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
