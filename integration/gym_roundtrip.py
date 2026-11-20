"""Real Gym episodes -> evaluation comparison -> optional data releases.

The two policies differ only in a cutoff-frequency unit conversion at submission.
This is an executable integration regression, not a measured LLM capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile


GYM_REVISION = "82b7100d88be2d1158d355802e6a864cde1e5684"
DATA_REVISION = "c6be7171b6adf643ea3418307d7ad4d168acf033"
TASKS, ATTEMPTS = 6, 3


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def source_info(module, reference_revision: str) -> dict:
    root = Path(module.__file__).parent
    hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(root.rglob("*.py"))}
    return {"package_version": module.__version__, "reference_revision": reference_revision,
            "loaded_source_sha256": hashes}


def run_policy(output: Path, suite, *, wrong_units: bool) -> dict:
    from agent_gym.reference import ReferenceAgent
    from agent_gym.rollout import PolicyVersion, RolloutBatch, export_rollouts

    name = "rc-cutoff-unit-error" if wrong_units else "rc-observation-fit"
    sampler = suite.sampler(seed=17)
    policies, observations = {}, {}
    with RolloutBatch(output, slots=3, workers=3, episode_seconds=30,
                      policy=PolicyVersion(name, "1")) as batch:
        while len(sampler) or batch.active_slots:
            free = [slot for slot in range(3) if slot not in batch.active_slots]
            opened = batch.reset(dict(zip(free, sampler.take(len(free)))))
            if any(row["done"] for row in opened.values()):
                raise AssertionError("Gym task setup failed")
            observations.update(opened)
            policies.update({slot: ReferenceAgent() for slot in opened})
            actions = {}
            for slot, row in observations.items():
                action = policies[slot].act(row["observation"])
                if wrong_units and "submit" in action:
                    action["submit"]["cutoff_hz"] *= 1e-3
                actions[slot] = action
            for slot, row in batch.step(actions).items():
                if row["done"]:
                    policies.pop(slot).close()
                    observations.pop(slot)
                else:
                    observations[slot] = row
        summary = batch.summary()
    exported = export_rollouts(output, output / "episodes.jsonl", include_unscored=True)
    assert exported == TASKS * ATTEMPTS == summary["completed"]
    assert summary["counts"]["failed" if wrong_units else "passed"] == exported
    return {"policy": name, "exported": exported, "outcomes": summary}


def factual_observations(export: Path) -> dict:
    observations = {}
    for row in map(json.loads, export.read_text(encoding="utf-8").splitlines()):
        key = row["rollout"]["task_group_id"], row["rollout"]["attempt"]
        observations[key] = [event["content"] for event in row["trajectory"]
                             if event["kind"] == "observation" and event["content"]["type"] == "tool_result"]
    return observations


def data_release(output: Path, episodes: Path, *, suite_id: str) -> dict:
    import ai_data
    from ai_data.pipeline import build, verify_release

    recipe = {"schema_version": "1.0", "name": "rc-evaluation-observations",
              "sources": [{"connector": "gym-jsonl", "path": str(episodes.relative_to(output)),
                           "uri": "example://rc-unit-regression", "revision": suite_id,
                           "allowed_uses": ["evaluate", "analyze"]}],
              "use": "evaluate", "seed": "17",
              "splits": {"train": 0, "validation": 0, "test": 1}}
    recipe_path = output / (episodes.parent.name + ".recipe.json")
    write_json(recipe_path, recipe)
    destination = output / (episodes.parent.name + "-release")
    manifest = build(recipe_path, destination)
    verified = verify_release(destination)
    rows = [json.loads(line) for line in (destination / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    source = [json.loads(line) for line in episodes.read_text(encoding="utf-8").splitlines()]
    assert verified["records"] == len(rows) == TASKS * ATTEMPTS
    assert {row["id"]: row["payload"] for row in rows} == {"gym:" + row["episode_id"]: row for row in source}
    assert manifest["counts"]["splits"] == {"train": 0, "validation": 0, "test": len(rows)}
    assert all(row["source"]["revision"] == suite_id for row in rows)
    return {"release_id": manifest["release_id"], "records": len(rows),
            "split": "test", "implementation": source_info(ai_data, DATA_REVISION)}


def run(output: Path, *, with_data: bool = False) -> dict:
    import agent_gym
    from agent_gym.contracts import TaskSpec
    from agent_gym.suite import TaskSuite
    from ai_eval.analysis import compare, summarize
    from ai_eval.contracts import load_run
    from ai_eval.importers import import_gym
    from ai_eval.report import write_report

    output.mkdir(parents=True, exist_ok=False)
    suite = TaskSuite([TaskSpec("rc-identification", seed=i, split="test", difficulty=1)
                       for i in range(TASKS)], attempts_per_task=ATTEMPTS,
                      name="rc-cutoff-unit-regression", metadata={"origin": "authored integration regression"})
    suite.save(output / "suite.json")
    results = {}
    for name, wrong_units in (("baseline", True), ("candidate", False), ("candidate-repeat", False)):
        gym_run = output / name
        result = run_policy(gym_run, suite, wrong_units=wrong_units)
        source = gym_run / "episodes.jsonl"
        imported = output / (name + "-evaluation")
        manifest = import_gym(source, imported,
                              system={"name": result["policy"], "revision": "1",
                                      "config": {"origin": "scripted integration policy"}},
                              benchmark_name="rc-cutoff-unit-regression")
        loaded = load_run(imported)
        assert len(loaded["results"]) == TASKS * ATTEMPTS
        assert (imported / "source.jsonl").read_bytes() == source.read_bytes()
        assert manifest["metadata"]["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert len({row["group"] for row in loaded["results"]}) == TASKS
        assert {row["trial"] for row in loaded["results"]} == set(range(ATTEMPTS))
        assert all(row["elapsed_seconds"] is None for row in loaded["results"])
        summary = summarize(imported, "reward", resamples=200, seed=17)
        assert summary["mean"] == (0.0 if wrong_units else 1.0)
        assert summary["coverage"]["scored_trials"] == TASKS * ATTEMPTS
        assert summary["coverage"]["independent_groups"] == TASKS
        result.update(run_id=manifest["run_id"], benchmark_id=manifest["benchmark"]["id"],
                      system_id=manifest["system_id"], summary=summary)
        if with_data and name != "candidate-repeat":
            result["data_release"] = data_release(output, source, suite_id=suite.suite_id)
        results[name] = result
    assert results["baseline"]["benchmark_id"] == results["candidate"]["benchmark_id"]
    assert results["baseline"]["system_id"] != results["candidate"]["system_id"]
    assert factual_observations(output / "baseline/episodes.jsonl") == factual_observations(output / "candidate/episodes.jsonl")
    assert factual_observations(output / "candidate/episodes.jsonl") == factual_observations(output / "candidate-repeat/episodes.jsonl")
    comparison = compare(output / "baseline-evaluation", output / "candidate-evaluation",
                         "reward", resamples=200, seed=17, slice_by="difficulty")
    repeated = compare(output / "candidate-evaluation", output / "candidate-repeat-evaluation",
                       "reward", resamples=200, seed=17)
    assert comparison["delta"] == 1.0 and repeated["delta"] == 0.0
    write_report(comparison, output / "comparison.html", title="RC cutoff-unit regression")
    report = {"origin": "scripted integration regression; no LLM calls",
              "gym": source_info(agent_gym, GYM_REVISION), "suite_id": suite.suite_id,
              "tasks": TASKS, "attempts_per_task": ATTEMPTS,
              "runs": results, "comparison": comparison, "repeat_delta": repeated["delta"],
              "same_measurements": True, "source_bytes_preserved": True}
    write_json(output / "result.json", report)
    return {"output": str(output), "tasks": TASKS, "attempts_per_task": ATTEMPTS,
            "episodes_per_run": TASKS * ATTEMPTS, "total_episodes": TASKS * ATTEMPTS * 3,
            "baseline_reward": 0.0, "candidate_reward": 1.0,
            "paired_delta": comparison["delta"], "repeat_delta": repeated["delta"],
            "same_measurements": True, "data_releases": 2 if with_data else 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gym-root", type=Path, help="Optional source checkout; otherwise use installed Gym")
    parser.add_argument("--data-root", type=Path, help="Optional Data Engineering source checkout; enables data release step")
    parser.add_argument("--with-data", action="store_true", help="Also build releases using installed Data Engineering")
    parser.add_argument("--output", type=Path, help="A new directory; default uses and removes a temporary directory")
    args = parser.parse_args()
    for root in (args.gym_root, args.data_root):
        if root is not None:
            sys.path.insert(0, str(root.resolve() / "src"))
    if args.output is not None:
        result = run(args.output.resolve(), with_data=args.with_data or args.data_root is not None)
    else:
        with tempfile.TemporaryDirectory(prefix="ai-eval-gym-") as temp:
            result = run(Path(temp) / "run", with_data=args.with_data or args.data_root is not None)
            result["output"] = "temporary directory removed"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
