"""Describe benchmark items using a declared panel of evaluated systems."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path

from .analysis import _compatible, _coverage, _load, _mean, _population, _sliced, _views


def _correlation(left: list[float], right: list[float]) -> tuple[float | None, str | None]:
    if len(left) < 3:
        return None, "fewer_than_three_systems"

    def centered(values):
        scale = max(abs(value) for value in values) or 1
        scaled = [value / scale for value in values]
        mean = math.fsum(value / len(scaled) for value in scaled)
        return [value - mean for value in scaled]

    a, b = centered(left), centered(right)
    a_norm, b_norm = math.sqrt(math.fsum(value * value for value in a)), math.sqrt(math.fsum(value * value for value in b))
    if not a_norm:
        return None, "constant_item_scores"
    if not b_norm:
        return None, "constant_rest_scores"
    value = math.fsum((x / a_norm) * (y / b_norm) for x, y in zip(a, b))
    return max(-1.0, min(1.0, value)), None


def _merge_aggregate(left, right):
    """Combine weighted means without subtracting a dominant group later."""
    if left is None:
        return right
    if right is None:
        return left
    if left[0] < right[0]:
        left, right = right, left
    ratio = math.exp(right[0] - left[0])
    share = ratio / (1 + ratio)
    means = [math.fsum((a * (1 - share), b * share)) for a, b in zip(left[1], right[1])]
    return left[0] + math.log1p(ratio), means


def _rest_means(samples: list[dict], systems: int) -> dict[str, list[float] | None]:
    """Compute leave-group-out means in O(samples × systems) time."""
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group"]].append(sample)
    names, aggregates = sorted(groups), []
    for name in names:
        members = groups[name]
        largest = max(item["weight"] for item in members)
        log_weight = math.log(largest) + math.log(math.fsum(item["weight"] / largest for item in members))
        means = [_mean([dict(item, mean=item["system_means"][index]) for item in members]) for index in range(systems)]
        aggregates.append((log_weight, means))
    prefix, suffix = [None], [None] * (len(names) + 1)
    for aggregate in aggregates:
        prefix.append(_merge_aggregate(prefix[-1], aggregate))
    for index in range(len(names) - 1, -1, -1):
        suffix[index] = _merge_aggregate(aggregates[index], suffix[index + 1])
    result = {}
    for index, name in enumerate(names):
        combined = _merge_aggregate(prefix[index], suffix[index + 1])
        result[name] = combined[1] if combined is not None else None
    return result


def _bounds(descriptor: dict, values: list[float]) -> dict | None:
    declared = descriptor.get("bounds")
    if declared is not None:
        try:
            valid = (isinstance(declared, list) and len(declared) == 2 and
                     all(type(value) in (int, float) and math.isfinite(value) for value in declared) and
                     declared[0] < declared[1])
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError("metric bounds must be [lower, upper] with finite lower < upper")
        lower, upper = declared
        basis = "declared_metric_bounds"
    elif descriptor.get("scorer") in {"exact_match", "numeric", "set_f1", "token_f1", "recall_at_k", "ndcg_at_k"}:
        lower, upper, basis = 0, 1, "native_scorer_bounds"
    else:
        return None
    if any(not lower <= value <= upper for value in values):
        source = "declared metric" if basis == "declared_metric_bounds" else "native scorer"
        raise ValueError(f"observed scores lie outside the {source} bounds")
    return {"lower": lower, "upper": upper, "basis": basis}


def _description(values: list[float]) -> dict:
    mean = math.fsum(value / len(values) for value in values)
    lower, upper = min(values), max(values)
    spread = upper - lower
    if not math.isfinite(spread):
        raise ValueError("panel score range exceeds finite numerical range")
    scale = max(abs(value) for value in values) or 1
    scaled_mean = mean / scale
    variance = math.fsum((value / scale - scaled_mean) ** 2 / len(values) for value in values)
    return {"panel_mean": mean, "panel_min": lower, "panel_max": upper,
            "panel_range": spread, "panel_stddev": scale * math.sqrt(variance)}


def _summary(samples: list[dict], bounds: dict | None) -> dict:
    return {
        "samples": len(samples), "groups": len({sample["group"] for sample in samples}),
        "panel_mean": _mean(samples, "panel_mean"),
        "mean_item_range": _mean(samples, "panel_range"),
        "mean_item_stddev": _mean(samples, "panel_stddev"),
        "constant_samples": sum(sample["panel_range"] == 0 for sample in samples),
        "floor_samples": sum(sample["floor"] is True for sample in samples) if bounds is not None else None,
        "ceiling_samples": sum(sample["ceiling"] is True for sample in samples) if bounds is not None else None,
        "discrimination_defined": sum(sample["discrimination"] is not None for sample in samples),
        "negative_discrimination_samples": sum(sample["discrimination"] is not None and sample["discrimination"] < 0 for sample in samples),
    }


def _cohort_coverage(planned: list[dict], retained: list[dict]) -> dict:
    identifiers = {sample["sample_id"] for sample in retained}
    return {
        "planned_samples": len(planned), "retained_samples": len(retained),
        "excluded_samples": len(planned) - len(retained),
        "planned_groups": len({sample["group"] for sample in planned}),
        "retained_groups": len({sample["group"] for sample in retained}),
        "retained_weight_fraction": _mean([dict(sample, mean=int(sample["sample_id"] in identifiers)) for sample in planned]),
    }


def analyze_benchmark(run_paths, metric: str, *, missing: str = "error", slice_by: str | None = None) -> dict:
    """Describe item behavior on a common cohort of distinct reference systems.

    Item discrimination is a descriptive Pearson correlation over the panel.
    Its rest score excludes every item in the focal item's independent group.
    """
    if isinstance(run_paths, (str, bytes, Path)):
        raise ValueError("run_paths must contain at least two run directories")
    paths = list(run_paths)
    if len(paths) < 2:
        raise ValueError("benchmark analysis requires at least two distinct systems")
    if missing not in ("error", "complete_cases"):
        raise ValueError("missing must be 'error' or 'complete_cases'")
    if slice_by is not None and (not isinstance(slice_by, str) or not slice_by):
        raise ValueError("slice_by must be a nonempty slice field name")
    runs = [_load(path) for path in paths]
    system_ids = [run["manifest"]["system_id"] for run in runs]
    if len(set(system_ids)) != len(system_ids):
        raise ValueError("the reference panel must contain distinct system identities; combine repeated attempts within a run")
    descriptor = _compatible(runs[0], runs[1], metric)
    for run in runs[2:]:
        _compatible(runs[0], run, metric)
    views = [_views(run, metric)[0] for run in runs]
    samples, exclusions = [], []
    for aligned in zip(*views):
        first = aligned[0]
        if not all(item["complete"] for item in aligned):
            exclusions.append({key: first[key] for key in ("sample_id", "group", "slices", "weight")} | {
                "systems": [{"index": index, **{key: item[key] for key in ("planned_trials", "observed_trials", "scored_trials")}}
                            for index, item in enumerate(aligned)]})
            continue
        samples.append({key: first[key] for key in ("sample_id", "group", "slices", "weight")} | {
            "trials": first["planned_trials"], "system_means": [item["mean"] for item in aligned]})
    if exclusions and missing == "error":
        names = ", ".join(item["sample_id"] for item in exclusions[:5])
        raise ValueError(f"{len(exclusions)} samples lack complete panel scores ({names}); "
                         "choose missing='complete_cases' to explicitly exclude them")
    if not samples:
        raise ValueError("no complete panel samples remain")

    included = {sample["sample_id"] for sample in samples}
    raw_scores = [row["scores"][metric] for run in runs for row in run["results"] if row["sample_id"] in included]
    bounds = _bounds(descriptor, raw_scores)
    binary = all(value in (0, 1) for value in raw_scores)
    sign = 1 if descriptor["direction"] == "maximize" else -1
    rest = _rest_means(samples, len(runs))
    for sample in samples:
        values = sample["system_means"]
        sample.update(_description(values))
        sample["binary_scores"] = binary
        sample["facility"] = (sample["panel_mean"] if binary and sign == 1 and bounds is not None and
                              bounds["lower"] == 0 and bounds["upper"] == 1 else None)
        sample["floor"], sample["ceiling"] = None, None
        if bounds is not None:
            lower = sample["panel_max"] == bounds["lower"]
            upper = sample["panel_min"] == bounds["upper"]
            sample["floor"], sample["ceiling"] = (lower, upper) if sign == 1 else (upper, lower)
        rest_values = rest[sample["group"]]
        sample["rest_system_means"] = rest_values
        if rest_values is None:
            correlation, reason = None, "no_other_independent_groups"
        else:
            correlation, reason = _correlation([sign * value for value in values], [sign * value for value in rest_values])
        sample["discrimination"], sample["discrimination_reason"] = correlation, reason

    panel = []
    for index, (run, view) in enumerate(zip(runs, views)):
        manifest = run["manifest"]
        system = manifest.get("system", {})
        name, revision = system.get("name", f"system-{index + 1}"), system.get("revision", "")
        panel.append({"index": index, "label": f"{name} / {revision}" if revision else name,
                      "name": name, "revision": revision, "run_id": manifest["run_id"],
                      "system_id": manifest["system_id"], "population": _population(manifest),
                      "mean": _mean([dict(sample, mean=sample["system_means"][index]) for sample in samples]),
                      "coverage": _coverage(view)})
    benchmark = runs[0]["manifest"]["benchmark"]
    result = {
        "schema_version": "1.0", "kind": "benchmark_analysis", "metric": metric,
        "direction": descriptor["direction"], "missing": missing,
        "benchmark": {key: benchmark[key] for key in ("id", "name", "revision") if key in benchmark},
        "panel": panel, "panel_weighting": "equal_weight_per_distinct_system",
        "estimand": "item_behavior_conditional_on_the_reference_panel_and_common_complete_cohort",
        "discrimination_method": "pearson_item_vs_weighted_rest_excluding_its_group",
        "score_bounds": bounds, "binary_scores": binary,
        "coverage": _cohort_coverage(views[0], samples) | {"panel": [_coverage(view) for view in views]},
        "summary": _summary(samples, bounds), "samples": samples, "exclusions": exclusions,
        "groups": [],
    }
    planned_groups, retained_groups = defaultdict(list), defaultdict(list)
    for item in views[0]:
        planned_groups[item["group"]].append(item)
    for item in samples:
        retained_groups[item["group"]].append(item)
    for group, planned in sorted(planned_groups.items()):
        retained = retained_groups[group]
        result["groups"].append({"group": group, **_summary(retained, bounds),
                                 "coverage": _cohort_coverage(planned, retained)})
    if slice_by is not None:
        result["slice_by"], result["slices"] = slice_by, []
        retained_by_id = {item["sample_id"]: item for item in samples}
        for bucket in _sliced(views[0], slice_by):
            identifiers = [item["sample_id"] for item in bucket["samples"]]
            retained = [retained_by_id[identifier] for identifier in identifiers if identifier in retained_by_id]
            result["slices"].append({"value": bucket["value"], "missing": bucket["missing"],
                                      **_summary(retained, bounds),
                                      "coverage": _cohort_coverage(bucket["samples"], retained),
                                      "excluded_sample_ids": [identifier for identifier in identifiers if identifier not in retained_by_id]})
    return result
