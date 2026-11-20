"""Versioned JSON contracts and content identities for evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


def _object(value: Any, field: str) -> dict:
    if type(value) is not dict:
        raise ValueError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _integer(value: Any, field: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = f" >= {minimum}" if minimum is not None else ""
        raise ValueError(f"{field} must be an integer{suffix}")
    return value


def _number(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be finite and numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (positive and value <= 0) or (nonnegative and value < 0):
        constraint = "positive " if positive else "nonnegative " if nonnegative else ""
        raise ValueError(f"{field} must be {constraint}finite and numeric")
    return value


def _json_tree(value: Any, field: str = "value") -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if type(value) is list:
        for item in value:
            _json_tree(item, field)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"{field} object keys must be strings")
            _json_tree(item, field)
        return
    raise ValueError(f"{field} must contain only JSON values")


def canonical_json(value: Any) -> str:
    """Canonical JSON used for stable experiment and system identities."""
    _json_tree(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _parse(text: str, source: str) -> Any:
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
        _json_tree(value)
        return value
    except (ValueError, RecursionError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc


def _schema(value: dict, field: str) -> None:
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{field}.schema_version must be {SCHEMA_VERSION!r}")


def _metrics(value: Any) -> list[dict]:
    if type(value) is not list or not value:
        raise ValueError("metrics must be a nonempty list")
    result, names = [], set()
    for item in value:
        metric = dict(_object(item, "metric"))
        name = _text(metric.get("name"), "metric.name")
        if name == "target":
            raise ValueError("metric name 'target' is reserved for execution errors")
        if name in names:
            raise ValueError(f"duplicate metric name: {name}")
        names.add(name)
        _text(metric.get("scorer"), "metric.scorer")
        if metric.get("direction") not in ("maximize", "minimize"):
            raise ValueError("metric.direction must be maximize or minimize")
        metric.setdefault("params", {})
        _object(metric["params"], "metric.params")
        result.append(metric)
    return result


def _protocol(value: Any, *, imported: bool = False, defaults: bool = False) -> dict:
    protocol = dict(_object(value, "protocol"))
    if defaults:
        protocol.setdefault("trials", 1)
        protocol.setdefault("seed", 0)
    _integer(protocol.get("trials"), "protocol.trials", 1)
    if "seed" not in protocol or protocol["seed"] is not None or not imported:
        _integer(protocol.get("seed"), "protocol.seed")
    if "import_format" in protocol:
        _text(protocol["import_format"], "protocol.import_format")
    return protocol


def _sample(value: Any, *, content: bool) -> dict:
    sample = dict(_object(value, "sample"))
    identifier = _text(sample.get("id"), "sample.id")
    if content and ("input" not in sample or "target" not in sample):
        raise ValueError(f"sample {identifier} must contain input and target")
    sample.setdefault("group", identifier)
    sample.setdefault("slices", {})
    sample.setdefault("weight", 1)
    if content:
        sample.setdefault("metadata", {})
    _text(sample["group"], "sample.group")
    _number(sample["weight"], "sample.weight", positive=True)
    for label in _object(sample["slices"], "sample.slices").values():
        if label is not None and type(label) not in (str, int, float, bool):
            raise ValueError("sample slice labels must be JSON scalars")
    if "metadata" in sample:
        _object(sample["metadata"], "sample.metadata")
    return sample


def _population(values: Any, *, content: bool) -> list[dict]:
    if type(values) is not list or not values:
        raise ValueError("samples must be a nonempty list")
    samples, identifiers = [], set()
    for value in values:
        sample = _sample(value, content=content)
        if sample["id"] in identifiers:
            raise ValueError(f"duplicate sample id: {sample['id']}")
        identifiers.add(sample["id"])
        samples.append(sample)
    return samples


def load_benchmark(path: str | Path) -> dict:
    """Load a native specification and content-addressed JSONL task population."""
    path = Path(path)
    spec = dict(_object(_parse(path.read_text(encoding="utf-8"), str(path)), "benchmark"))
    spec.setdefault("schema_version", SCHEMA_VERSION)
    _schema(spec, "benchmark")
    _text(spec.get("name"), "benchmark.name")
    _text(spec.get("revision"), "benchmark.revision")
    if "description" in spec and not isinstance(spec["description"], str):
        raise ValueError("benchmark.description must be a string")
    source = _text(spec.get("samples"), "benchmark.samples")
    sample_path = path.parent / source
    raw = sample_path.read_bytes()
    samples = _population([
        _parse(line, f"{sample_path}:{number}")
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1) if line.strip()
    ], content=True)
    spec["metrics"] = _metrics(spec.get("metrics"))
    spec["protocol"] = _protocol(spec.get("protocol", {}), defaults=True)
    spec.setdefault("metadata", {})
    _object(spec["metadata"], "benchmark.metadata")
    sample_sha = hashlib.sha256(raw).hexdigest()
    identity = dict(spec, samples=sample_sha)
    return {**spec, "samples": samples, "spec": spec, "sample_path": str(sample_path),
            "sample_sha256": sample_sha, "benchmark_id": digest(identity)}


def system_id(system: dict) -> str:
    """Hash the declared system, including harness and generation configuration."""
    normalized = dict(_object(system, "system"))
    _text(normalized.get("name"), "system.name")
    _text(normalized.get("revision"), "system.revision")
    if "kind" in normalized:
        _text(normalized["kind"], "system.kind")
    normalized.setdefault("config", {})
    _object(normalized["config"], "system.config")
    return digest(normalized)


def _manifest(value: Any) -> dict:
    manifest = dict(_object(value, "manifest"))
    _schema(manifest, "manifest")
    _text(manifest.get("run_id"), "manifest.run_id")
    benchmark = dict(_object(manifest.get("benchmark"), "manifest.benchmark"))
    for key in ("id", "name", "revision", "sample_sha256"):
        _text(benchmark.get(key), f"manifest.benchmark.{key}")
    benchmark["protocol"] = _protocol(benchmark.get("protocol"),
                                      imported="import_format" in _object(benchmark.get("protocol"), "protocol"))
    benchmark["metrics"] = _metrics(benchmark.get("metrics"))
    if manifest.get("system_id") != system_id(manifest.get("system")):
        raise ValueError("manifest.system_id does not match the declared system")
    samples = _population(manifest.get("samples"), content=False)
    planned = _integer(manifest.get("planned_trials"), "manifest.planned_trials", 1)
    if planned != len(samples) * benchmark["protocol"]["trials"]:
        raise ValueError("manifest.planned_trials does not match samples × trials")
    for key in ("engine", "runtime", "execution", "metadata"):
        if key in manifest:
            _object(manifest[key], f"manifest.{key}")
    manifest.update(benchmark=benchmark, samples=samples)
    return manifest


def _result(value: Any, manifest: dict, population: dict, metric_names: set[str]) -> dict:
    row = _object(value, "result")
    _schema(row, "result")
    for key, expected in (("benchmark_id", manifest["benchmark"]["id"]), ("system_id", manifest["system_id"])):
        if row.get(key) != expected:
            raise ValueError(f"result.{key} does not match manifest")
    identifier = _text(row.get("sample_id"), "result.sample_id")
    if identifier not in population:
        raise ValueError(f"result sample is not in manifest: {identifier}")
    sample = population[identifier]
    for key in ("group", "slices"):
        if key not in row or canonical_json(row[key]) != canonical_json(sample[key]):
            raise ValueError(f"result.{key} does not match sample {identifier}")
    if _number(row.get("weight"), "result.weight", positive=True) != sample["weight"]:
        raise ValueError(f"result.weight does not match sample {identifier}")
    trials = manifest["benchmark"]["protocol"]["trials"]
    if _integer(row.get("trial"), "result.trial", 0) >= trials:
        raise ValueError("result.trial is outside the planned population")
    imported = "import_format" in manifest["benchmark"]["protocol"]
    if "seed" not in row or row["seed"] is not None or not imported:
        _integer(row.get("seed"), "result.seed")
    scores = _object(row.get("scores"), "result.scores")
    if set(scores) != metric_names:
        raise ValueError("result.scores must contain exactly the declared metric names")
    missing = {name for name, value in scores.items() if value is None}
    for name, value in scores.items():
        if value is not None:
            _number(value, f"score {name}")
    errors = _object(row.get("errors"), "result.errors")
    if not set(errors).issubset(metric_names | {"target"}):
        raise ValueError("result.errors contains an undeclared metric")
    status = row.get("status")
    if status == "ok":
        if missing or errors:
            raise ValueError("ok result must have all scores and no errors")
    elif status == "target_error":
        if missing != metric_names or "target" not in errors:
            raise ValueError("target_error result must have no scores and a target error")
    elif status == "score_error":
        if not missing or not missing.issubset(errors) or "target" in errors:
            raise ValueError("score_error result must identify unavailable metrics")
    else:
        raise ValueError("result.status must be ok, target_error or score_error")
    if "output" not in row:
        raise ValueError("result must contain output, including null for failed executions")
    if "elapsed_seconds" not in row or row["elapsed_seconds"] is not None or not imported:
        _number(row.get("elapsed_seconds"), "result.elapsed_seconds", nonnegative=True)
    _object(row.get("usage"), "result.usage")
    _object(row.get("metadata"), "result.metadata")
    return row


def load_run(path: str | Path) -> dict:
    """Load any valid partial run without converting missing outcomes to scores."""
    path = Path(path)
    manifest_path, results_path = path / "manifest.json", path / "results.jsonl"
    manifest = _manifest(_parse(manifest_path.read_text(encoding="utf-8"), str(manifest_path)))
    population = {sample["id"]: sample for sample in manifest["samples"]}
    metric_names = {metric["name"] for metric in manifest["benchmark"]["metrics"]}
    results, seen = [], set()
    with results_path.open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            row = _result(_parse(line, f"{results_path}:{number}"), manifest, population, metric_names)
            key = row["sample_id"], row["trial"]
            if key in seen:
                raise ValueError(f"duplicate sample/trial result: {key}")
            seen.add(key)
            results.append(row)
    return {"manifest": manifest, "results": results}
