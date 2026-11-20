"""Explicit per-observation scoring; aggregation belongs to analysis."""

from __future__ import annotations

from collections import Counter
import hashlib
from importlib.metadata import entry_points
import json
import math
from pathlib import Path
from typing import Any, Callable


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def exact_match(output: Any, target: Any, *, strip: bool = False, casefold: bool = False) -> float:
    """Structural JSON equality; text normalization is opt-in."""
    _boolean(strip, "strip")
    _boolean(casefold, "casefold")
    if strip or casefold:
        if not isinstance(output, str) or not isinstance(target, str):
            raise ValueError("text normalization requires string output and target")
        if strip:
            output, target = output.strip(), target.strip()
        if casefold:
            output, target = output.casefold(), target.casefold()
    return float(_key(output) == _key(target))


def numeric(output: Any, target: Any, *, abs_tol: float = 1e-8, rel_tol: float = 0.0) -> float:
    """Accuracy under |output-target| <= max(abs_tol, rel_tol*|target|)."""
    output, target = _number(output, "output"), _number(target, "target")
    abs_tol, rel_tol = _number(abs_tol, "abs_tol"), _number(rel_tol, "rel_tol")
    if abs_tol < 0 or rel_tol < 0:
        raise ValueError("tolerances must be nonnegative")
    difference = abs(output - target)
    if math.isfinite(difference):
        return float(difference <= max(abs_tol, rel_tol * abs(target)))
    # Opposite finite extremes can overflow subtraction; compare on a common scale.
    scale = max(abs(output), abs(target))
    return float(abs(output / scale - target / scale) <= max(abs_tol / scale, rel_tol * abs(target / scale)))


def _ids(value: Any, name: str, *, unique: bool = False) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON list")
    if any(type(item) not in (str, int, float, bool) and item is not None for item in value):
        raise ValueError(f"{name} identifiers must be JSON scalars")
    keys = [_key(item) for item in value]
    if unique and len(keys) != len(set(keys)):
        raise ValueError(f"{name} must not contain duplicate identifiers")
    return keys


def _f1(overlap: int, predicted: int, reference: int) -> float:
    return 2.0 * overlap / (predicted + reference) if predicted + reference else 1.0


def set_f1(output: Any, target: Any) -> float:
    predicted, reference = set(_ids(output, "output")), set(_ids(target, "target"))
    return _f1(len(predicted & reference), len(predicted), len(reference))


def token_f1(output: Any, target: Any, *, casefold: bool = False) -> float:
    """Multiset F1 over whitespace-delimited tokens; no implicit QA normalization."""
    _boolean(casefold, "casefold")
    if not isinstance(output, str) or not isinstance(target, str):
        raise ValueError("token_f1 requires string output and target")
    if casefold:
        output, target = output.casefold(), target.casefold()
    predicted, reference = Counter(output.split()), Counter(target.split())
    return _f1(sum((predicted & reference).values()), sum(predicted.values()), sum(reference.values()))


def _cutoff(k: Any) -> int:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    return k


def recall_at_k(output: Any, target: Any, *, k: int) -> float:
    ranked = _ids(output, "output", unique=True)
    reference = set(_ids(target, "target", unique=True))
    if not reference:
        raise ValueError("recall is undefined for an empty relevant set")
    return len(set(ranked[:_cutoff(k)]) & reference) / len(reference)


def ndcg_at_k(output: Any, target: Any, *, k: int, gain: str = "exponential") -> float:
    _cutoff(k)
    _ids(output, "output", unique=True)
    if any(not isinstance(item, str) for item in output):
        raise ValueError("ndcg output identifiers must be strings")
    if not isinstance(target, dict) or any(not isinstance(key, str) for key in target):
        raise ValueError("ndcg target must map string identifiers to relevance grades")
    if gain not in ("linear", "exponential"):
        raise ValueError("gain must be linear or exponential")
    gains = {}
    for key, value in target.items():
        grade = _number(value, "relevance")
        if grade < 0:
            raise ValueError("relevance must be nonnegative")
        try:
            gains[key] = grade if gain == "linear" else math.expm1(grade * math.log(2))
        except OverflowError as exc:
            raise ValueError("relevance gain overflowed") from exc
    # Normalize before summing: this avoids overflow without changing the ratio.
    scale = max(gains.values(), default=0.0)
    if not scale:
        raise ValueError("ndcg is undefined when ideal discounted gain is zero")
    ranked = [gains.get(item, 0.0) / scale for item in output[:k]]
    ideal = sorted((value / scale for value in gains.values()), reverse=True)[:k]
    dcg = math.fsum(value / math.log2(rank + 2) for rank, value in enumerate(ranked))
    idcg = math.fsum(value / math.log2(rank + 2) for rank, value in enumerate(ideal))
    return dcg / idcg


def _probabilities(output: Any, target: Any) -> tuple[list[float], list[float], float]:
    if isinstance(output, dict):
        if not output or any(not isinstance(key, str) for key in output):
            raise ValueError("class probabilities must map string labels to probabilities")
        if not isinstance(target, str) or target not in output:
            raise ValueError("target must name a supplied class")
        probabilities = [_number(value, "probability") for value in output.values()]
        if any(value < 0 or value > 1 for value in probabilities):
            raise ValueError("probabilities must lie in [0, 1]")
        if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("class probabilities must sum to one; they are not renormalized")
        outcomes = [float(label == target) for label in output]
        return probabilities, outcomes, float(output[target])
    probability = _number(output, "P(y=1)")
    outcome = _number(target, "binary target")
    if not 0 <= probability <= 1 or outcome not in (0.0, 1.0):
        raise ValueError("binary scoring requires P(y=1) in [0, 1] and target 0 or 1")
    return [probability], [outcome], probability if outcome else 1.0 - probability


def brier(output: Any, target: Any) -> float:
    probabilities, outcomes, _ = _probabilities(output, target)
    return math.fsum((p - y) ** 2 for p, y in zip(probabilities, outcomes))


def log_loss(output: Any, target: Any, *, epsilon: float | None = None) -> float:
    _, _, probability = _probabilities(output, target)
    if epsilon is not None:
        epsilon = _number(epsilon, "epsilon")
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must lie in (0, 1)")
        probability = max(probability, epsilon)
    if probability == 0:
        raise ValueError("log loss is infinite at zero target probability; declare epsilon to cap the loss")
    return -math.log(probability)


SCORERS: dict[str, Callable[..., float]] = {
    name: value for name, value in {
        "exact_match": exact_match, "numeric": numeric, "set_f1": set_f1,
        "token_f1": token_f1, "recall_at_k": recall_at_k, "ndcg_at_k": ndcg_at_k,
        "brier": brier, "log_loss": log_loss,
    }.items()
}


def implementation(spec: dict[str, Any]) -> dict:
    """Identify the installed measuring code before a run starts or resumes."""
    name = spec.get("scorer")
    if name in SCORERS:
        from . import __version__
        return {"kind": "native", "package": "ai-evaluation", "version": __version__,
                "scorer": name, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    matches = list(entry_points(group="ai_eval.scorers", name=name))
    if len(matches) != 1:
        raise ValueError(f"expected one scorer named {name!r}, found {len(matches)}")
    point = matches[0]
    return {"kind": "plugin", "name": name, "entry_point": point.value,
            "distribution": point.dist.name if point.dist else None,
            "version": point.dist.version if point.dist else None}


def score(spec: dict[str, Any], output: Any, target: Any) -> float:
    """Resolve a native/installed scorer and require a finite scalar result.

    Entry points in ``ai_eval.scorers`` implement ``fn(output, target, **params)``.
    Native names cannot be overridden. Duplicate plugin names are errors.
    """
    name = spec.get("scorer")
    params = spec.get("params", {})
    if not isinstance(name, str) or not isinstance(params, dict):
        raise ValueError("metric requires a scorer string and params object")
    function = SCORERS.get(name)
    if function is None:
        matches = list(entry_points(group="ai_eval.scorers", name=name))
        if len(matches) != 1:
            raise ValueError(f"expected one scorer named {name!r}, found {len(matches)}")
        function = matches[0].load()
    result = function(output, target, **params)
    return _number(result, f"score from {name}")
