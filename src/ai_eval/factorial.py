"""Paired two-factor contrasts on one common experimental population."""

from __future__ import annotations

import math
from pathlib import Path

from .analysis import (
    _compatible, _coverage, _interval, _load, _mean, _population, _settings, _sliced, _views,
)


CELLS = ("00", "10", "01", "11")
_CONTRASTS = {
    "main_a": (-0.5, 0.5, -0.5, 0.5),
    "main_b": (-0.5, -0.5, 0.5, 0.5),
    "interaction": (1, -1, -1, 1),
    "a_at_b0": (-1, 1, 0, 0),
    "a_at_b1": (0, 0, -1, 1),
    "b_at_a0": (-1, 0, 1, 0),
    "b_at_a1": (0, -1, 0, 1),
}


def _contrast(values: list[float], coefficients: tuple[float, ...]) -> float:
    scale = max(abs(value) for value in values)
    if not scale:
        return 0.0
    value = math.fsum(coefficient * (score / scale)
                      for score, coefficient in zip(values, coefficients)) * scale
    if not math.isfinite(value):
        raise ValueError("factorial contrast exceeds finite numerical range")
    return value


def _estimate(samples: list[dict], key: str, confidence: float, resamples: int,
              seed: int, direction: str, *, contrast: bool) -> dict:
    values = [dict(sample, value=sample["contrasts" if contrast else "scores"][key])
              for sample in samples]
    estimate = _mean(values, "value")
    interval, reason = _interval(values, confidence, resamples, seed, "value", paired=contrast)
    result = {"estimate" if contrast else "mean": estimate,
              "interval": interval, "interval_reason": reason}
    if contrast:
        sign = 1 if direction == "maximize" else -1
        oriented = None if interval is None else dict(interval)
        if oriented is not None and sign == -1:
            oriented["lower"], oriented["upper"] = -interval["upper"], -interval["lower"]
        result.update(improvement=sign * estimate if estimate is not None else None,
                      improvement_interval=oriented,
                      coefficients=dict(zip(CELLS, _CONTRASTS[key])))
    return result


def _cohort_coverage(views: list[list[dict]], retained: list[dict]) -> dict:
    population = views[0]
    largest = max((sample["weight"] for sample in population), default=1)
    total = math.fsum(sample["weight"] / largest for sample in population)
    weight = math.fsum(sample["weight"] / largest for sample in retained)
    return {
        "conditions": {cell: _coverage(samples) for cell, samples in zip(CELLS, views)},
        "total_samples": len(population), "retained_samples": len(retained),
        "excluded_samples": len(population) - len(retained),
        "independent_groups": len({sample["group"] for sample in population}),
        "retained_groups": len({sample["group"] for sample in retained}),
        "retained_weight_fraction": weight / total if total else None,
    }


def _estimates(samples: list[dict], conditions: list[dict], confidence: float,
               resamples: int, seed: int, direction: str) -> dict:
    return {
        "conditions": [dict(condition, **_estimate(samples, condition["cell"], confidence,
                                                   resamples, seed, direction, contrast=False))
                       for condition in conditions],
        "effects": {key: _estimate(samples, key, confidence, resamples, seed,
                                    direction, contrast=True) for key in _CONTRASTS},
    }


def factorial(run_00: str | Path, run_10: str | Path, run_01: str | Path,
              run_11: str | Path, metric: str, *, factor_a: str = "A", factor_b: str = "B",
              confidence: float = 0.95, resamples: int = 2000, seed: int = 0,
              missing: str = "error", slice_by: str | None = None) -> dict:
    """Estimate 2×2 main, simple and interaction contrasts on matched cases.

    The positional order is 00, 10, 01, 11: the first digit is factor A.
    Interaction is the full difference of differences, not the half-scaled
    interaction effect sometimes reported for {-1,+1}-coded factorial designs.
    Returned intervals are marginal group-bootstrap intervals. Condition labels
    describe the supplied design; this operation does not establish causality.
    """
    _settings(confidence, resamples, seed)
    if missing not in ("error", "complete_cases"):
        raise ValueError("missing must be 'error' or 'complete_cases'")
    if any(not isinstance(label, str) or not label.strip() for label in (factor_a, factor_b)):
        raise ValueError("factor names must be nonempty strings")
    if factor_a.strip() == factor_b.strip():
        raise ValueError("factor names must be distinct")
    if slice_by is not None and (not isinstance(slice_by, str) or not slice_by):
        raise ValueError("slice_by must be a nonempty slice field name")

    runs = [_load(path) for path in (run_00, run_10, run_01, run_11)]
    if len({run["manifest"]["run_id"] for run in runs}) != 4:
        raise ValueError("factorial analysis requires four distinct runs")
    descriptor = None
    for run in runs[1:]:
        descriptor = _compatible(runs[0], run, metric)
    direction = descriptor["direction"]
    views = [_views(run, metric)[0] for run in runs]
    samples, exclusions = [], []
    for matched in zip(*views):
        first = matched[0]
        if not all(sample["complete"] for sample in matched):
            exclusions.append({
                "sample_id": first["sample_id"], "group": first["group"],
                "planned_trials": first["planned_trials"],
                "conditions": {cell: {"observed_trials": sample["observed_trials"],
                                      "scored_trials": sample["scored_trials"]}
                               for cell, sample in zip(CELLS, matched)},
            })
            continue
        values = [sample["mean"] for sample in matched]
        samples.append({
            "sample_id": first["sample_id"], "group": first["group"],
            "weight": first["weight"], "slices": first["slices"],
            "trials": first["planned_trials"], "scores": dict(zip(CELLS, values)),
            "contrasts": {key: _contrast(values, coefficients)
                          for key, coefficients in _CONTRASTS.items()},
        })
    if exclusions and missing == "error":
        ids = ", ".join(sample["sample_id"] for sample in exclusions[:5])
        raise ValueError(f"{len(exclusions)} samples lack complete scores across all four conditions ({ids}); "
                         "choose missing='complete_cases' to explicitly exclude them")
    if not samples:
        raise ValueError("no complete factorial samples remain")

    conditions = []
    for cell, run in zip(CELLS, runs):
        manifest = run["manifest"]
        system = manifest["system"]
        conditions.append({"cell": cell, "factor_a_level": int(cell[0]),
                           "factor_b_level": int(cell[1]), "run_id": manifest["run_id"],
                           "system_id": manifest["system_id"], "system": system,
                           "name": system["name"], "revision": system["revision"]})
    manifest = runs[0]["manifest"]
    result = {
        "schema_version": "1.0", "kind": "factorial", "metric": metric, "direction": direction,
        "benchmark_id": manifest["benchmark"]["id"],
        "benchmark": {key: manifest["benchmark"][key] for key in ("name", "revision")},
        "factors": {"a": factor_a, "b": factor_b, "levels": [0, 1]},
        "missing": missing,
        "population": {cell: _population(run["manifest"]) for cell, run in zip(CELLS, runs)},
        "estimand": "sample_weighted_contrasts_of_complete_four_condition_trial_means",
        "interval_scope": "marginal; not simultaneous across contrasts or slices",
        **_estimates(samples, conditions, confidence, resamples, seed, direction),
        "coverage": _cohort_coverage(views, samples), "samples": samples, "exclusions": exclusions,
    }
    if slice_by is not None:
        result["slice_by"], result["slices"] = slice_by, []
        for bucket in _sliced(views[0], slice_by):
            ids = {sample["sample_id"] for sample in bucket["samples"]}
            retained = [sample for sample in samples if sample["sample_id"] in ids]
            sliced_views = [[sample for sample in view if sample["sample_id"] in ids] for view in views]
            result["slices"].append({
                "value": bucket["value"], "missing": bucket["missing"],
                **_estimates(retained, conditions, confidence, resamples, seed, direction),
                "coverage": _cohort_coverage(sliced_views, retained),
                "excluded_sample_ids": [sample["sample_id"] for sample in exclusions
                                        if sample["sample_id"] in ids],
            })
    return result
