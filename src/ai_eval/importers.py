"""Lossless source retention with explicit, comparable imported populations."""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
from typing import Any

from .contracts import load_run, system_id


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"source contains non-finite JSON constant {value}")


def _parse(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_constant)


def _retained(value: Any) -> Any:
    """Represent Inspect's nonfinite sentinels in standards-compliant metadata.

    The original source file remains byte-identical, including NaN constants.
    """
    if isinstance(value, dict):
        return {key: _retained(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_retained(item) for item in value]
    if type(value) is float and not math.isfinite(value):
        label = "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
        return {"$source_nonfinite": label}
    return value


def _number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be finite and numeric")
    return float(value)


def _integer(value: Any, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _object(value: Any, field: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _save(source: Path, raw: bytes, output: str | Path, *, source_format: str,
          source_version: int, system: dict, name: str, revision: str, metrics: list,
          protocol: dict, samples: list, records: list, provenance: dict) -> dict:
    _text(name, "benchmark name")
    _text(revision, "benchmark revision")
    system_digest = system_id(system)
    sample_sha = _digest(samples)
    benchmark = {"name": name, "revision": revision, "sample_sha256": sample_sha,
                 "metrics": metrics, "protocol": protocol}
    benchmark["id"] = _digest(benchmark)
    source_sha = hashlib.sha256(raw).hexdigest()
    manifest = {
        "schema_version": "1.0", "run_id": _digest({"benchmark": benchmark["id"],
            "system": system_digest, "source": source_sha}),
        "benchmark": benchmark, "system": system, "system_id": system_digest,
        "planned_trials": len(samples) * protocol["trials"], "samples": samples,
        "engine": {"name": "ai-evaluation-import", "version": "0.1.0"},
        "metadata": {"source": {"format": source_format, "version": source_version,
            "filename": source.name, "sha256": source_sha,
            "retained_file": "source.jsonl" if source_format == "gym-export" else "source.json"},
            **provenance},
    }
    seen = set()
    for row in records:
        key = row["sample_id"], row["trial"]
        if key in seen:
            raise ValueError(f"duplicate imported sample/trial: {key}")
        seen.add(key)
        row.update(schema_version="1.0", benchmark_id=benchmark["id"], system_id=system_digest)
    # Encode before creating the output directory, so malformed metadata produces no partial run.
    manifest_json = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    results_json = "".join(_json(row) + "\n" for row in records)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("import output must be a new or empty directory")
    output.mkdir(parents=True, exist_ok=True)
    (output / manifest["metadata"]["source"]["retained_file"]).write_bytes(raw)
    (output / "results.jsonl").write_text(results_json, encoding="utf-8")
    (output / "manifest.json").write_text(manifest_json, encoding="utf-8")
    return manifest


def import_gym(source: str | Path, output: str | Path, *, system: dict,
               benchmark_name: str, revision: str = "1") -> dict:
    """Import Verifiable-Agent-Gym export-v1, including rollout exports.

    Only the exported task population is known. The exporter can omit unscored
    episodes and always omits setup failures; the manifest records this scope.
    """
    source = Path(source)
    raw = source.read_bytes()
    rows = [_parse(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Gym export has no episode records")
    rollout_mode = isinstance(_object(rows[0], "Gym row").get("rollout"), dict)
    policies, episodes, definitions, records = set(), set(), {}, []
    encounter = Counter()
    conditions = {}
    max_trial = 0
    for line, row in enumerate(rows, 1):
        _object(row, "Gym row")
        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
            raise ValueError("only Gym export schema_version 1 is supported")
        episode = _text(row.get("episode_id"), "episode_id")
        if episode in episodes:
            raise ValueError(f"duplicate episode_id: {episode}")
        episodes.add(episode)
        task = _object(row.get("task"), "task")
        _text(row.get("agent"), "agent")
        if not isinstance(row.get("trajectory"), list):
            raise ValueError("Gym trajectory must be a list")
        rollout = row.get("rollout")
        if isinstance(rollout, dict) != rollout_mode:
            raise ValueError("do not mix rollout exports and ordinary episode exports")
        if rollout_mode:
            policy = _object(rollout.get("policy"), "rollout.policy")
            _text(policy.get("name"), "policy.name")
            _text(policy.get("revision"), "policy.revision")
            _object(policy.get("generation", {}), "policy.generation")
            if row["agent"] != policy["name"] or rollout.get("task") != task:
                raise ValueError("rollout policy/task disagree with episode fields")
            limits = _object(rollout.get("limits"), "rollout.limits")
            _integer(limits.get("actions"), "action limit", 1)
            if _number(limits.get("seconds"), "seconds limit") <= 0:
                raise ValueError("seconds limit must be positive")
            task_group = _text(rollout.get("task_group_id"), "rollout.task_group_id")
            identity = {"task": task, "data_reference": rollout.get("data_reference"), "limits": limits,
                        "task_group_id": task_group, "suite_id": rollout.get("suite_id"),
                        "epoch": _integer(rollout.get("epoch"), "rollout.epoch")}
            condition = _digest(identity)
            if task_group in conditions and conditions[task_group] != condition:
                raise ValueError("a rollout task group mixes task/data/budget conditions")
            conditions[task_group] = condition
            trial = _integer(rollout.get("attempt"), "rollout.attempt")
            seed = _integer(rollout.get("sample_seed"), "rollout.sample_seed")
        else:
            policy = {"agent": row["agent"]}
            identity = {"task": task, "data_reference": None, "limits": None}
            seed = None
        policies.add(_json(policy))
        if len(policies) > 1:
            raise ValueError("Gym export mixes policies; split into one import per policy/version/generation")
        sample_id = "gym:" + _digest(identity)
        group = "gym-task:" + _digest({"task": task, "data_reference": identity["data_reference"]})
        if not rollout_mode:
            trial = encounter[sample_id]
            encounter[sample_id] += 1
        max_trial = max(max_trial, trial)
        slices = {key: value for key, value in task.items() if key in ("family", "split", "difficulty")
                  and type(value) in (str, int, float, bool)}
        descriptor = {"id": sample_id, "group": group, "slices": slices, "weight": 1,
                      "metadata": {"gym_condition": identity}}
        definitions[sample_id] = descriptor
        source_status = row.get("status")
        if source_status not in ("passed", "failed", "truncated", "error", "cancelled"):
            raise ValueError(f"unsupported Gym status {source_status!r}")
        reward = row.get("reward")
        errors = {}
        if source_status in ("passed", "failed") and reward is not None:
            reward = _number(reward, "Gym reward")
            status = "ok"
        else:
            reward = None
            status = "target_error" if source_status in ("truncated", "error", "cancelled") else "score_error"
            errors["target" if status == "target_error" else "reward"] = {
                "type": "ImportedGymOutcome", "message": f"Gym status={source_status}; no usable reward",
                "source_status": source_status,
            }
        records.append({"sample_id": sample_id, "group": group, "slices": slices, "weight": 1,
            "trial": trial, "seed": seed, "status": status,
            "output": {"trajectory": row["trajectory"], "verification": row.get("verification")},
            "scores": {"reward": reward}, "errors": errors, "elapsed_seconds": None, "usage": {},
            "metadata": {"source_line": line, "original": row}})
    samples = [definitions[key] for key in sorted(definitions)]
    system = _parse(_json(system))
    config = _object(system.setdefault("config", {}), "system.config")
    if "gym_policy" in config and config["gym_policy"] != _parse(next(iter(policies))):
        raise ValueError("supplied system.config.gym_policy differs from exported policy")
    config["gym_policy"] = _parse(next(iter(policies)))
    protocol = {"trials": max_trial + 1, "seed": None, "import_format": "gym-export-v1",
                "trial_mapping": "rollout.attempt" if rollout_mode else "per-task export encounter order",
                "population_scope": "exported task groups", "budget_recorded": rollout_mode}
    return _save(source, raw, output, source_format="gym-export", source_version=1,
        system=system, name=benchmark_name, revision=revision,
        metrics=[{"name": "reward", "scorer": "imported_gym_reward", "direction": "maximize", "params": {}}],
        protocol=protocol, samples=samples, records=records,
        provenance={"population": {"scope": "exported task groups",
            "unobserved_tasks_known": False, "planned_trial_count_inferred_from_max_attempt": True,
            "note": "Export may exclude unscored episodes; rollout exports omit setup failures. "
                    "Coverage refers to exported groups and inferred attempt slots, not the original full run."}})


def _inspect_id(value: Any) -> str:
    if type(value) not in (str, int) or value == "":
        raise ValueError("Inspect sample id must be a nonempty string or integer")
    return "inspect:" + _json(value)


def _inspect_usage(sample: dict) -> dict:
    totals = Counter()
    for model, usage in _object(sample.get("model_usage", {}), "model_usage").items():
        for key, value in _object(usage, f"model_usage.{model}").items():
            if value is not None:
                amount = _number(value, f"model_usage.{model}.{key}")
                if amount < 0:
                    raise ValueError("model usage must be nonnegative")
                totals[key] += amount
    return dict(totals)


def _inspect_value(sample: dict, scorer: str, value_key: str | None) -> float:
    selected = _object(sample.get("scores") or {}, "scores").get(scorer)
    if not isinstance(selected, dict) or "value" not in selected:
        raise ValueError(f"sample has no value for scorer {scorer!r}")
    value = selected["value"]
    if value_key is not None:
        if not isinstance(value, dict) or value_key not in value:
            raise ValueError(f"score has no value key {value_key!r}")
        value = value[value_key]
    elif isinstance(value, (dict, list)):
        raise ValueError("non-scalar score requires an explicit value_key selecting a numeric value")
    if value == "C":
        return 1.0
    if value == "I":
        return 0.0
    if type(value) is bool:
        return float(value)
    return _number(value, "Inspect score (numeric or C/I)")


def _inspect_judges(evaluation: dict, definitions: list, plan: dict) -> list:
    """Resolve the documented model/role/default precedence of Inspect graders."""
    result = []
    for definition in definitions:
        name = definition.get("scorer") or definition.get("name", "")
        if name.rsplit("/", 1)[-1] not in ("model_graded_qa", "model_graded_fact", "_model_graded_qa_single"):
            continue
        options = definition.get("options") or definition.get("params") or {}
        model = options.get("model")
        role = options.get("model_role", None if name.rsplit("/", 1)[-1] == "_model_graded_qa_single" else "grader")
        required = isinstance(role, dict) and role.get("required") is True
        role_name = role.get("name") if isinstance(role, dict) else role
        roles = evaluation.get("model_roles") or {}
        if model is not None:
            binding = {"kind": "explicit-model", "model": model}
        elif role_name is not None and roles.get(role_name) is not None:
            binding = {"kind": "model-role", "role": role_name, "model": roles[role_name]}
        elif required:
            binding = {"kind": "unbound-required-role", "role": role_name}
        else:
            binding = {"kind": "evaluated-model-default", "model": evaluation.get("model"),
                       "model_args": evaluation.get("model_args", {}),
                       "model_base_url": evaluation.get("model_base_url"),
                       "model_generate_config": evaluation.get("model_generate_config", {}),
                       "plan_generate_config": plan.get("config", {})}
        result.append({"scorer": definition.get("name", name), "binding": binding})
    return result


def import_inspect(source: str | Path, output: str | Path, *, scorer: str,
                   value_key: str | None = None, direction: str = "maximize") -> dict:
    """Import Inspect JSON log v2 with explicit scorer/value selection."""
    _text(scorer, "scorer")
    if value_key is not None:
        _text(value_key, "value_key")
    if direction not in ("maximize", "minimize"):
        raise ValueError("direction must be maximize or minimize")
    source = Path(source)
    raw = source.read_bytes()
    try:
        # Inspect deliberately serializes unscored values as the NaN constant.
        log = _object(json.loads(raw.decode("utf-8")), "Inspect log")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("expected Inspect JSON; convert binary .eval logs with inspect log convert") from exc
    if type(log.get("version")) is not int or log["version"] != 2:
        raise ValueError("only Inspect JSON log version 2 is supported")
    samples = log.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Inspect log needs per-sample records; aggregate-only results cannot be imported")
    if log.get("config_updates"):
        raise ValueError("Inspect log contains mid-run configuration changes; split stable conditions before importing")
    evaluation = _object(log.get("eval"), "eval")
    config = _object(evaluation.get("config", {}), "eval.config")
    plan = _object(log.get("plan", {}), "plan")
    name = _text(evaluation.get("task"), "eval.task")
    revision = str(evaluation.get("task_version", 0))
    epochs = config.get("epochs")
    if epochs is None:
        epochs = max(_integer(sample.get("epoch"), "sample.epoch", 1) for sample in samples)
    _integer(epochs, "eval.config.epochs", 1)
    metric_name = scorer if value_key is None else f"{scorer}.{value_key}"
    definitions, records = {}, []
    for sample in samples:
        _object(sample, "Inspect sample")
        sample_id = _inspect_id(sample.get("id"))
        trial = _integer(sample.get("epoch"), "sample.epoch", 1) - 1
        if trial >= epochs:
            raise ValueError("Inspect epoch exceeds configured epochs")
        metadata = _object(sample.get("metadata", {}), "sample.metadata")
        mapping = _object(metadata.get("ai_eval", {}), "sample.metadata.ai_eval")
        group = _text(mapping.get("group", sample_id), "independent group")
        slices = _object(mapping.get("slices", {}), "sample slices")
        if any(type(value) not in (str, int, float, bool) and value is not None for value in slices.values()):
            raise ValueError("slice labels must be JSON scalars")
        weight = _number(mapping.get("weight", 1), "sample weight")
        if weight <= 0:
            raise ValueError("sample weight must be positive")
        identity = {key: sample.get(key) for key in
                    ("input", "target", "choices", "sandbox", "files", "setup", "token_limit", "token_limit_type",
                     "message_limit", "time_limit")}
        identity["metadata"] = metadata
        if "input" not in sample or "target" not in sample:
            raise ValueError("Inspect samples require input and target")
        descriptor = {"id": sample_id, "group": group, "slices": slices, "weight": weight,
                      "metadata": {"inspect_condition": identity}}
        if sample_id in definitions and definitions[sample_id] != descriptor:
            raise ValueError("an Inspect sample changes input/target/group/budget across epochs")
        definitions[sample_id] = descriptor
        errors = {}
        source_error = sample.get("error")
        invalidation = sample.get("invalidation")
        value = None
        if invalidation:
            status = "score_error"
            errors[metric_name] = {"type": "InvalidatedInspectSample", "message": "source sample was invalidated",
                                   "invalidation": invalidation}
        elif source_error:
            status = "target_error"
            errors["target"] = source_error
            if config.get("score_on_error") is True:
                try:
                    value = _inspect_value(sample, scorer, value_key)
                except ValueError:
                    pass
                else:
                    status, errors = "ok", {}
        else:
            try:
                value = _inspect_value(sample, scorer, value_key)
                status = "ok"
            except ValueError as exc:
                status = "score_error"
                errors[metric_name] = {"type": "ImportedScoreError", "message": str(exc)}
                original_score = (sample.get("scores") or {}).get(scorer)
                if isinstance(original_score, dict):
                    errors[metric_name]["source_score"] = _retained(original_score)
        elapsed = sample.get("total_time")
        if elapsed is not None:
            elapsed = _number(elapsed, "sample.total_time")
            if elapsed < 0:
                raise ValueError("sample.total_time must be nonnegative")
        retained_metadata = {"original": _retained(sample)}
        if source_error:
            retained_metadata.update(source_error=_retained(source_error), scored_on_error=status == "ok")
        records.append({"sample_id": sample_id, "group": group, "slices": slices, "weight": weight,
            "trial": trial, "seed": None, "status": status, "output": sample.get("output"),
            "scores": {metric_name: value}, "errors": errors, "elapsed_seconds": elapsed,
            "usage": _inspect_usage(sample), "metadata": retained_metadata})
    selected_scorers = evaluation.get("scorers")
    if selected_scorers is None:
        selected_scorers = [{key: item.get(key) for key in ("name", "scorer", "params", "reducer")}
                            for item in (log.get("results") or {}).get("scores", [])]
    protocol = {"trials": epochs, "seed": None, "import_format": "inspect-json-v2",
        "trial_mapping": "sample.epoch minus one", "population_scope": "observed sample IDs",
        "evaluation_config": config, "scorers": selected_scorers,
        "resolved_builtin_judges": _inspect_judges(evaluation, selected_scorers, plan),
        "dataset": evaluation.get("dataset"), "task_attribs": evaluation.get("task_attribs", {}),
        "model_roles": evaluation.get("model_roles"),
        "task_args": evaluation.get("task_args", {}), "sandbox": evaluation.get("sandbox"),
        "source_revision": evaluation.get("revision"), "packages": evaluation.get("packages", {})}
    system = {"name": _text(evaluation.get("model"), "eval.model"),
        "revision": "unrecorded", "kind": "inspect",
        "config": {"model": evaluation["model"], "model_args": evaluation.get("model_args", {}),
                   "model_base_url": evaluation.get("model_base_url"),
                   "model_roles": evaluation.get("model_roles"),
                   "model_generate_config": evaluation.get("model_generate_config", {}), "plan": plan,
                   "solver": evaluation.get("solver"), "solver_args": evaluation.get("solver_args")}}
    return _save(source, raw, output, source_format="inspect-json", source_version=2,
        system=system, name=name, revision=revision,
        metrics=[{"name": metric_name, "scorer": "imported_inspect_score", "direction": direction,
                  "params": {"scorer": scorer, "value_key": value_key, "C": 1, "I": 0, "true": 1, "false": 0}}],
        protocol=protocol, samples=[definitions[key] for key in sorted(definitions)], records=records,
        provenance={"original_header": _retained({key: value for key, value in log.items() if key != "samples"}),
            "population": {"scope": "observed sample IDs crossed with configured epochs",
                "unobserved_tasks_known": False, "planned_trial_count_inferred": config.get("epochs") is None,
                "source_total_samples": log.get("total_samples", (log.get("results") or {}).get("total_samples")),
                "source_dataset": evaluation.get("dataset"),
                "note": "Entirely unobserved sample IDs are not reconstructed. Epochs are repeated attempts, "
                        "not independent tasks. Model identifier is retained; no unrecorded model snapshot is inferred."}})


def import_plugin(name: str, source: str | Path, output: str | Path, **params) -> dict:
    """Execute an installed importer, then validate and return its native run."""
    _text(name, "importer name")
    matches = list(entry_points(group="ai_eval.importers", name=name))
    if len(matches) != 1:
        raise ValueError(f"expected one importer named {name!r}, found {len(matches)}")
    matches[0].load()(source, output, **params)
    return load_run(output)
