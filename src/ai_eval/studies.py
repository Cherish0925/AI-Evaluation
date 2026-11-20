"""Compare several systems on one shared population and observed resource scale."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from pathlib import Path
from typing import Iterable

from .analysis import (
    _compatible, _coverage, _interval, _load, _mean, _paired_summary,
    _population, _settings, _sliced, _views,
)


def _resource_field(resource: str | None) -> str | None:
    if resource is None or resource == "elapsed_seconds":
        return resource
    if not isinstance(resource, str) or not resource.startswith("usage.") or not resource[6:].strip():
        raise ValueError("resource must be 'elapsed_seconds' or 'usage.KEY'")
    return resource


def _resource_views(run: dict, resource: str | None) -> dict:
    """Retain observed counts independently of metric completeness."""
    values = defaultdict(list)
    if resource is not None:
        for row in run["results"]:
            value = row.get("elapsed_seconds") if resource == "elapsed_seconds" else row.get("usage", {}).get(resource[6:])
            if value is None:
                continue
            try:
                valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
            except OverflowError:
                valid = False
            if not valid:
                raise ValueError(f"{resource} must be finite and nonnegative: sample {row['sample_id']}, trial {row['trial']}")
            values[row["sample_id"]].append(value)
    return {
        sample["id"]: {
            "trials": len(values[sample["id"]]),
            "mean": math.fsum(value / len(values[sample["id"]]) for value in values[sample["id"]])
            if values[sample["id"]] else None,
        }
        for sample in run["manifest"]["samples"]
    }


def _identities(runs: list[dict]) -> list[dict]:
    identities = []
    for index, run in enumerate(runs):
        manifest = run["manifest"]
        system = manifest["system"]
        name, revision = system["name"], system["revision"]
        identities.append({"index": index, "label": f"{name} · {revision}",
                           "name": name, "revision": revision,
                           "run_id": manifest["run_id"], "system_id": manifest["system_id"],
                           "config": system.get("config", {})})
    counts = Counter(item["label"] for item in identities)
    for item in identities:
        if counts[item["label"]] > 1:
            item["label"] += f" [{item['index'] + 1}]"
    return identities


def _cohort_coverage(planned: list[dict], retained: list[dict], views: list[list[dict]],
                     resources: list[dict], resource: str | None) -> dict:
    identifiers = {sample["sample_id"] for sample in planned}
    included = {sample["sample_id"] for sample in retained}
    planned_trials = sum(sample["planned_trials"] for sample in planned)
    result = {
        "total_samples": len(planned), "retained_samples": len(retained),
        "excluded_samples": len(planned) - len(retained),
        "retained_weight_fraction": _mean([
            dict(sample, retained=float(sample["sample_id"] in included)) for sample in planned
        ], "retained"),
        "independent_groups": len({sample["group"] for sample in planned}),
        "retained_groups": len({sample["group"] for sample in retained}),
        "planned_trials_per_system": planned_trials,
        "retained_trials_per_system": sum(sample["trials"] for sample in retained),
        "systems": [],
    }
    for index, system_views in enumerate(views):
        selected = [sample for sample in system_views if sample["sample_id"] in identifiers]
        available = sum(resources[index][sample["sample_id"]]["trials"] for sample in selected)
        result["systems"].append({
            "index": index, **_coverage(selected),
            "resource_observed_trials": available if resource is not None else None,
            "resource_missing_trials": planned_trials - available if resource is not None else None,
            "resource_complete_samples": sum(
                resources[index][sample["sample_id"]]["trials"] == sample["planned_trials"]
                for sample in selected
            ) if resource is not None else None,
        })
    return result


def _frontier(systems: list[dict], direction: str) -> list[int]:
    sign = 1 if direction == "maximize" else -1
    frontier = []
    for system in systems:
        dominated_by = []
        if system["mean"] is not None and system["resource_mean"] is not None:
            quality, cost = sign * system["mean"], system["resource_mean"]
            for other in systems:
                if other["mean"] is None or other["resource_mean"] is None:
                    continue
                other_quality, other_cost = sign * other["mean"], other["resource_mean"]
                if other_quality >= quality and other_cost <= cost and (other_quality > quality or other_cost < cost):
                    dominated_by.append(other["index"])
            if not dominated_by:
                frontier.append(system["index"])
        system["dominated_by"] = dominated_by
    return frontier


def _estimates(samples: list[dict], identities: list[dict], baseline: int, direction: str,
               resource: str | None, confidence: float, resamples: int, seed: int) -> tuple[list[dict], list[int] | None]:
    systems = []
    for identity in identities:
        index = identity["index"]
        entries = []
        for sample in samples:
            left, right = sample["means"][baseline], sample["means"][index]
            delta = right - left
            if not math.isfinite(delta):
                raise ValueError("paired difference exceeds finite numerical range")
            entries.append({"group": sample["group"], "weight": sample["weight"],
                            "mean": right, "resource_mean": sample["resources"][index],
                            "left_mean": left, "right_mean": right, "delta": delta})
        interval, reason = _interval(entries, confidence, resamples, seed)
        pair = _paired_summary(entries, confidence, resamples, seed, direction)
        resource_interval, resource_reason = (
            _interval(entries, confidence, resamples, seed, "resource_mean")
            if resource is not None else (None, None)
        )
        systems.append({**identity, "mean": _mean(entries), "interval": interval, "interval_reason": reason,
                        "resource_mean": _mean(entries, "resource_mean"),
                        "resource_interval": resource_interval, "resource_interval_reason": resource_reason,
                        "delta": pair["delta"], "delta_interval": pair["interval"],
                        "improvement": pair["improvement"], "improvement_interval": pair["improvement_interval"],
                        "paired_interval_reason": pair["interval_reason"]})
    return systems, _frontier(systems, direction) if resource is not None else None


def compare_systems(run_paths: Iterable[str | Path], metric: str, *, baseline: int = 0,
                    resource: str | None = None, confidence: float = 0.95,
                    resamples: int = 2000, seed: int = 0, missing: str = "error",
                    slice_by: str | None = None) -> dict:
    """Estimate systems and baseline contrasts on the same complete samples.

    A sample contributes only when every planned trial has the selected score
    in every run, and also the selected resource when one is requested. Trials
    are averaged within a sample before sample weighting. Group bootstrap
    draws are shared across systems and baseline contrasts. Intervals are
    pointwise, not simultaneous. A Pareto frontier describes observed means;
    membership does not establish statistical superiority.

    ``usage.KEY`` must use the same physical or accounting unit across runs.
    ``elapsed_seconds`` measures per-call duration, not total job wall time.
    """
    _settings(confidence, resamples, seed)
    _resource_field(resource)
    if missing not in ("error", "complete_cases"):
        raise ValueError("missing must be 'error' or 'complete_cases'")
    if isinstance(run_paths, (str, bytes, Path)):
        raise ValueError("run_paths must contain at least two run directories")
    paths = [Path(path).resolve() for path in run_paths]
    if len(paths) < 2 or len(set(paths)) != len(paths):
        raise ValueError("a study requires at least two unique run directories")
    if type(baseline) is not int or not 0 <= baseline < len(paths):
        raise ValueError("baseline must be a valid zero-based system index")
    runs = [_load(path) for path in paths]
    if len({run["manifest"]["run_id"] for run in runs}) != len(runs):
        raise ValueError("a study requires unique run identities; copied runs are not separate observations")
    descriptor = None
    for run in runs[1:]:
        descriptor = _compatible(runs[0], run, metric)
    direction = descriptor["direction"]
    views = [_views(run, metric)[0] for run in runs]
    resources = [_resource_views(run, resource) for run in runs]
    planned = views[0]
    # Construct slice groups before execution of the estimator so invalid
    # selectors and completely excluded categories have the same semantics.
    buckets = _sliced(planned, slice_by) if slice_by is not None else None
    samples, exclusions = [], []
    for positions in zip(*views):
        first = positions[0]
        identifier, trials = first["sample_id"], first["planned_trials"]
        available = [resource_view[identifier] for resource_view in resources]
        complete = all(sample["complete"] for sample in positions)
        complete = complete and (resource is None or all(item["trials"] == trials for item in available))
        if not complete:
            exclusions.append({
                "sample_id": identifier, "group": first["group"], "slices": first["slices"],
                "weight": first["weight"], "planned_trials": trials,
                "systems": [
                    {"index": index, "observed_trials": sample["observed_trials"],
                     "scored_trials": sample["scored_trials"], "missing_scores": trials - sample["scored_trials"],
                     "resource_trials": available[index]["trials"] if resource is not None else None,
                     "missing_resource": trials - available[index]["trials"] if resource is not None else None}
                    for index, sample in enumerate(positions)
                ],
            })
            continue
        samples.append({"sample_id": identifier, "group": first["group"], "slices": first["slices"],
                        "weight": first["weight"], "trials": trials,
                        "means": [sample["mean"] for sample in positions],
                        "resources": [item["mean"] if resource is not None else None for item in available]})
    if exclusions and missing == "error":
        identifiers = ", ".join(item["sample_id"] for item in exclusions[:5])
        fields = "scores and resource observations" if resource is not None else "scores"
        raise ValueError(f"{len(exclusions)} samples lack complete {fields} across systems ({identifiers}); "
                         "choose missing='complete_cases' to explicitly exclude them")
    if not samples:
        raise ValueError("no complete samples remain across all systems")
    identities = _identities(runs)
    systems, frontier = _estimates(samples, identities, baseline, direction, resource, confidence, resamples, seed)
    benchmark = runs[0]["manifest"]["benchmark"]
    result = {
        "schema_version": "1.0", "kind": "study", "metric": metric, "direction": direction,
        "benchmark_id": benchmark["id"],
        "benchmark": {key: benchmark[key] for key in ("name", "revision", "protocol")},
        "baseline": {key: identities[baseline][key] for key in ("index", "label", "run_id")},
        "missing": missing,
        "estimand": "sample_weighted_trial_means_on_common_complete_cases",
        "population": [_population(run["manifest"]) for run in runs],
        "interval_scope": "pointwise; shared independent-group resampling",
        "systems": systems, "resource": resource,
        "resource_aggregation": "sample_weighted_mean_per_attempt" if resource is not None else None,
        "resource_unit": "seconds" if resource == "elapsed_seconds" else None,
        "pareto_frontier": frontier,
        "frontier_basis": "observed_point_estimates" if resource is not None else None,
        "coverage": _cohort_coverage(planned, samples, views, resources, resource),
        "samples": samples, "exclusions": exclusions,
    }
    if buckets is not None:
        result["slice_by"] = slice_by
        result["slices"] = []
        for bucket in buckets:
            identifiers = {sample["sample_id"] for sample in bucket["samples"]}
            retained = [sample for sample in samples if sample["sample_id"] in identifiers]
            sliced_systems, sliced_frontier = _estimates(
                retained, identities, baseline, direction, resource, confidence, resamples, seed
            )
            result["slices"].append({
                "value": bucket["value"], "missing": bucket["missing"],
                "systems": sliced_systems, "pareto_frontier": sliced_frontier,
                "coverage": _cohort_coverage(bucket["samples"], retained, views, resources, resource),
                "excluded_sample_ids": [sample["sample_id"] for sample in exclusions if sample["sample_id"] in identifiers],
            })
    return result
