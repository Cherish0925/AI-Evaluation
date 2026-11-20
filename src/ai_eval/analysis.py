"""Sample-weighted estimates with independent-group and paired uncertainty."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random


def _load(path: str | Path) -> dict:
    from .contracts import load_run

    return load_run(path)


def _settings(confidence: float, resamples: int, seed: int) -> None:
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and between zero and one")
    if type(resamples) is not int or resamples < 2:
        raise ValueError("resamples must be an integer of at least two")
    if type(seed) is not int:
        raise ValueError("bootstrap seed must be an integer")


def _metric(manifest: dict, name: str) -> dict:
    for metric in manifest["benchmark"]["metrics"]:
        if metric["name"] == name:
            return metric
    raise ValueError(f"unknown metric: {name}")


def _population(manifest: dict) -> dict:
    return manifest.get("metadata", {}).get("population", {"scope": "declared benchmark samples"})


def _mean(entries: list[dict], values: str = "mean", multiplicities: dict | None = None) -> float | None:
    weighted = [(item["weight"],
                 multiplicities.get(item["group"], 0) if multiplicities is not None else 1,
                 item[values]) for item in entries if item[values] is not None]
    weighted = [(weight, count, value) for weight, count, value in weighted if count > 0]
    if not weighted:
        return None
    # Scale before summing: finite weights may themselves add to more than float's range.
    largest = max(weight for weight, _, _ in weighted)
    scaled = [(weight / largest * count, value) for weight, count, value in weighted]
    total = math.fsum(weight for weight, _ in scaled)
    return math.fsum(value * (weight / total) for weight, value in scaled)


def _quantile(ordered: list[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    left = math.floor(position)
    fraction = position - left
    if left == len(ordered) - 1:
        return ordered[left]
    return ordered[left] * (1 - fraction) + ordered[left + 1] * fraction


def _interval(entries: list[dict], confidence: float, resamples: int, seed: int,
              values: str = "mean", paired: bool = False) -> tuple[dict | None, str | None]:
    eligible = [item for item in entries if item[values] is not None]
    groups = sorted({item["group"] for item in eligible})
    if len(groups) < 2:
        reason = "no_scored_samples" if not groups else "fewer_than_two_independent_groups"
        return None, reason
    generator = random.Random(seed)
    distribution = []
    for _ in range(resamples):
        multiplicities = Counter(generator.choice(groups) for _ in groups)
        distribution.append(_mean(eligible, values, multiplicities))
    distribution.sort()
    tail = (1 - confidence) / 2
    return {
        "lower": _quantile(distribution, tail),
        "upper": _quantile(distribution, 1 - tail),
        "confidence": confidence,
        "method": "paired_group_percentile_bootstrap" if paired else "group_percentile_bootstrap",
        "resamples": resamples, "seed": seed,
        "independent_groups": len(groups),
    }, None


def _views(run: dict, metric: str) -> tuple[list[dict], dict]:
    manifest, results = run["manifest"], run["results"]
    descriptor = _metric(manifest, metric)
    trials = manifest["benchmark"]["protocol"]["trials"]
    rows = defaultdict(list)
    for result in results:
        rows[result["sample_id"]].append(result)
    samples = []
    for sample in sorted(manifest["samples"], key=lambda item: item["id"]):
        observed = sorted(rows[sample["id"]], key=lambda item: item["trial"])
        scored = [row["scores"][metric] for row in observed if row["scores"].get(metric) is not None]
        samples.append({
            "sample_id": sample["id"], "group": sample.get("group", sample["id"]),
            "slices": sample.get("slices", {}), "weight": sample.get("weight", 1),
            "mean": math.fsum(value / len(scored) for value in scored) if scored else None,
            "planned_trials": trials, "observed_trials": len(observed), "scored_trials": len(scored),
            "complete": len(scored) == trials,
        })
    return samples, descriptor


def _coverage(samples: list[dict]) -> dict:
    planned = sum(sample["planned_trials"] for sample in samples)
    observed = sum(sample["observed_trials"] for sample in samples)
    scored = sum(sample["scored_trials"] for sample in samples)
    largest_weight = max((sample["weight"] for sample in samples), default=1)
    weights = [sample["weight"] / largest_weight for sample in samples]
    total_weight = math.fsum(weights)

    def weight_fraction(predicate):
        numerator = math.fsum(weight for sample, weight in zip(samples, weights) if predicate(sample))
        return numerator / total_weight if total_weight else None

    return {
        "total_samples": len(samples),
        "scored_samples": sum(sample["mean"] is not None for sample in samples),
        "complete_samples": sum(sample["complete"] for sample in samples),
        "planned_trials": planned,
        "observed_trials": observed,
        "scored_trials": scored,
        "unrecorded_trials": planned - observed,
        "unscored_trials": observed - scored,
        "scored_fraction": scored / planned if planned else None,
        "scored_weight_fraction": weight_fraction(lambda sample: sample["mean"] is not None),
        "complete_weight_fraction": weight_fraction(lambda sample: sample["complete"]),
        "independent_groups": len({sample["group"] for sample in samples}),
        "scored_groups": len({sample["group"] for sample in samples if sample["mean"] is not None}),
    }


def _resources(rows: list[dict]) -> dict:
    elapsed = [row["elapsed_seconds"] for row in rows if row.get("elapsed_seconds") is not None]
    usage = defaultdict(list)
    for row in rows:
        for name, value in row.get("usage", {}).items():
            if type(value) in (int, float) and math.isfinite(value):
                usage[name].append(value)

    def totals(values):
        try:
            total = math.fsum(values)
        except OverflowError:
            total = None
        return {"sum": total, "observations": len(values)}

    return {"elapsed_seconds": totals(elapsed),
            "usage": {name: totals(values) for name, values in sorted(usage.items())}}


def _sliced(samples: list[dict], field: str):
    if not isinstance(field, str) or not field:
        raise ValueError("slice_by must be a nonempty slice field name")
    buckets = {}
    for sample in samples:
        labels = sample.get("slices", {})
        present, value = field in labels, labels.get(field)
        key = json.dumps([present, value], sort_keys=True, ensure_ascii=False, allow_nan=False)
        if key not in buckets:
            buckets[key] = {"value": value, "missing": not present, "samples": []}
        buckets[key]["samples"].append(sample)
    return [buckets[key] for key in sorted(buckets)]


def _summary(samples: list[dict], rows: list[dict], confidence: float, resamples: int, seed: int) -> dict:
    interval, reason = _interval(samples, confidence, resamples, seed)
    return {
        "mean": _mean(samples), "interval": interval, "interval_reason": reason,
        "coverage": _coverage(samples),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "resources": _resources(rows),
    }


def summarize(run_path: str | Path, metric: str, *, confidence: float = 0.95,
              resamples: int = 2000, seed: int = 0, slice_by: str | None = None) -> dict:
    """Average trials within each sample, then weight samples and resample groups.

    Incomplete samples contribute their observed-trial means. Coverage is part of
    the result because this estimand is conditional on the observed completions.
    """
    _settings(confidence, resamples, seed)
    run = _load(run_path)
    samples, descriptor = _views(run, metric)
    manifest = run["manifest"]
    result = {
        "schema_version": "1.0", "run_id": manifest["run_id"],
        "benchmark_id": manifest["benchmark"]["id"], "system_id": manifest["system_id"],
        "metric": metric, "direction": descriptor["direction"],
        "population": _population(manifest),
        "estimand": "sample_weighted_mean_of_observed_trial_means",
        **_summary(samples, run["results"], confidence, resamples, seed), "samples": samples,
    }
    if slice_by is not None:
        result["slice_by"] = slice_by
        result["slices"] = []
        for bucket in _sliced(samples, slice_by):
            identifiers = {sample["sample_id"] for sample in bucket["samples"]}
            rows = [row for row in run["results"] if row["sample_id"] in identifiers]
            result["slices"].append({"value": bucket["value"], "missing": bucket["missing"],
                                     **_summary(bucket["samples"], rows, confidence, resamples, seed)})
    return result


def _comparison_metric(metric: dict) -> dict:
    """Compare native measurement code independently of the package release."""
    implementation = metric.get("implementation")
    if not isinstance(implementation, dict) or implementation.get("kind") != "native":
        return metric
    code = implementation.get("module_sha256")
    if (implementation.get("scorer") != metric.get("scorer") or
            not isinstance(code, str) or len(code) != 64 or
            any(character not in "0123456789abcdefABCDEF" for character in code)):
        return metric
    return {**metric, "implementation": {key: value for key, value in implementation.items()
                                         if key != "version"}}


def _compatible(left: dict, right: dict, metric: str) -> dict:
    left_manifest, right_manifest = left["manifest"], right["manifest"]
    left_benchmark, right_benchmark = left_manifest["benchmark"], right_manifest["benchmark"]
    if left_benchmark["id"] != right_benchmark["id"]:
        raise ValueError("paired comparison requires the same benchmark ID")
    if left_benchmark["protocol"] != right_benchmark["protocol"]:
        raise ValueError("paired comparison requires the same execution protocol")
    descriptor = _metric(left_manifest, metric)
    if _comparison_metric(descriptor) != _comparison_metric(_metric(right_manifest, metric)):
        raise ValueError("paired comparison requires the same metric definition")
    if left_benchmark.get("sample_sha256") != right_benchmark.get("sample_sha256"):
        raise ValueError("paired comparison requires the same sample content")
    fields = lambda manifest: sorted([
        {"id": sample["id"], "group": sample.get("group", sample["id"]),
         "slices": sample.get("slices", {}), "weight": sample.get("weight", 1)}
        for sample in manifest["samples"]], key=lambda sample: sample["id"])
    if json.dumps(fields(left_manifest), sort_keys=True) != json.dumps(fields(right_manifest), sort_keys=True):
        raise ValueError("paired comparison requires the same sample population, groups and weights")
    return descriptor


def _paired_summary(samples: list[dict], confidence: float, resamples: int, seed: int,
                    direction: str) -> dict:
    interval, reason = _interval(samples, confidence, resamples, seed, "delta", paired=True)
    delta = _mean(samples, "delta")
    sign = 1 if direction == "maximize" else -1
    improvement_interval = None
    if interval is not None:
        improvement_interval = dict(interval)
        if sign == -1:
            improvement_interval["lower"], improvement_interval["upper"] = -interval["upper"], -interval["lower"]
    return {
        "left_mean": _mean(samples, "left_mean"), "right_mean": _mean(samples, "right_mean"),
        "delta": delta, "improvement": sign * delta if delta is not None else None,
        "interval": interval, "improvement_interval": improvement_interval, "interval_reason": reason,
        "paired_samples": len(samples), "independent_groups": len({sample["group"] for sample in samples}),
    }


def compare(left_path: str | Path, right_path: str | Path, metric: str, *, confidence: float = 0.95,
            resamples: int = 2000, seed: int = 0, missing: str = "error", slice_by: str | None = None) -> dict:
    """Estimate right-minus-left using complete paired cases and paired groups."""
    _settings(confidence, resamples, seed)
    if missing not in ("error", "complete_cases"):
        raise ValueError("missing must be 'error' or 'complete_cases'")
    left, right = _load(left_path), _load(right_path)
    descriptor = _compatible(left, right, metric)
    left_samples, _ = _views(left, metric)
    right_samples, _ = _views(right, metric)
    paired, exclusions = [], []
    for a, b in zip(left_samples, right_samples):
        if not a["complete"] or not b["complete"]:
            exclusions.append({"sample_id": a["sample_id"], "group": a["group"],
                               "planned_trials": a["planned_trials"],
                               "left_scored_trials": a["scored_trials"],
                               "right_scored_trials": b["scored_trials"],
                               "left_observed_trials": a["observed_trials"],
                               "right_observed_trials": b["observed_trials"]})
            continue
        delta = b["mean"] - a["mean"]
        if not math.isfinite(delta):
            raise ValueError("paired difference exceeds finite numerical range")
        paired.append({"sample_id": a["sample_id"], "group": a["group"], "weight": a["weight"],
                       "slices": a["slices"], "left_mean": a["mean"], "right_mean": b["mean"],
                       "delta": delta, "trials": a["planned_trials"]})
    if exclusions and missing == "error":
        ids = ", ".join(item["sample_id"] for item in exclusions[:5])
        raise ValueError(f"{len(exclusions)} samples lack complete paired scores ({ids}); "
                         "choose missing='complete_cases' to explicitly exclude them")
    if not paired:
        raise ValueError("no complete paired samples remain")
    result = {
        "schema_version": "1.0", "benchmark_id": left["manifest"]["benchmark"]["id"],
        "left_run_id": left["manifest"]["run_id"], "right_run_id": right["manifest"]["run_id"],
        "left_system_id": left["manifest"]["system_id"], "right_system_id": right["manifest"]["system_id"],
        "metric": metric, "direction": descriptor["direction"], "missing": missing,
        "population": {"left": _population(left["manifest"]), "right": _population(right["manifest"])},
        "estimand": "sample_weighted_mean_of_complete_paired_trial_mean_differences",
        **_paired_summary(paired, confidence, resamples, seed, descriptor["direction"]),
        "coverage": {"left": _coverage(left_samples), "right": _coverage(right_samples),
                     "retained_samples": len(paired), "excluded_samples": len(exclusions)},
        "exclusions": exclusions, "samples": paired,
    }
    if slice_by is not None:
        result["slice_by"] = slice_by
        result["slices"] = []
        for bucket in _sliced(left_samples, slice_by):
            identifiers = {sample["sample_id"] for sample in bucket["samples"]}
            retained = [sample for sample in paired if sample["sample_id"] in identifiers]
            excluded = [sample for sample in exclusions if sample["sample_id"] in identifiers]
            other = [sample for sample in right_samples if sample["sample_id"] in identifiers]
            result["slices"].append({
                "value": bucket["value"], "missing": bucket["missing"],
                **_paired_summary(retained, confidence, resamples, seed, descriptor["direction"]),
                "coverage": {"left": _coverage(bucket["samples"]), "right": _coverage(other),
                             "retained_samples": len(retained), "excluded_samples": len(excluded)},
                "excluded_sample_ids": [sample["sample_id"] for sample in excluded],
            })
    return result


def pass_at_k(run_path: str | Path, metric: str, k: int) -> dict:
    """Estimate success within k independent attempts from the same trial policy.

    The combinatorial estimator is unbiased for the corresponding iid sampling
    experiment; complete binary records alone cannot establish that assumption.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    run = _load(run_path)
    samples, descriptor = _views(run, metric)
    if descriptor["direction"] != "maximize":
        raise ValueError("pass_at_k requires a binary success metric with direction 'maximize'")
    by_sample = defaultdict(list)
    for row in run["results"]:
        by_sample[row["sample_id"]].append(row["scores"].get(metric))
    estimates = []
    for sample in samples:
        values = by_sample[sample["sample_id"]]
        if not sample["complete"]:
            raise ValueError(f"pass_at_k requires every planned trial scored: {sample['sample_id']}")
        if any(type(value) not in (int, float) or value not in (0, 1) for value in values):
            raise ValueError("pass_at_k requires binary 0/1 success scores")
        n, successes = len(values), int(sum(values))
        if n < k:
            raise ValueError(f"sample {sample['sample_id']} has {n} trials, fewer than k={k}")
        if n - successes < k:
            estimate = 1.0
        else:
            estimate = -math.expm1(math.fsum(math.log1p(-successes / (n - index)) for index in range(k)))
        estimates.append({"sample_id": sample["sample_id"], "group": sample["group"],
                          "weight": sample["weight"], "trials": n, "successes": successes,
                          "estimate": estimate})
    return {
        "schema_version": "1.0", "run_id": run["manifest"]["run_id"],
        "benchmark_id": run["manifest"]["benchmark"]["id"], "system_id": run["manifest"]["system_id"],
        "metric": metric, "k": k, "estimate": _mean(estimates, "estimate"),
        "population": _population(run["manifest"]),
        "estimand": "sample_weighted_probability_of_success_in_k_attempts",
        "coverage": _coverage(samples), "samples": estimates,
    }
