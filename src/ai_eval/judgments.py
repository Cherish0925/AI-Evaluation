"""Categorical judge agreement against explicit reference annotations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .analysis import _interval, _settings


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def _label(value, field, line):
    if value is not None and (not isinstance(value, str) or not value):
        raise ValueError(f"line {line}: {field} must be a nonempty string or null")
    return value


def calibrate(path: str | Path, *, labels: list[str] | None = None,
              group_key: str = "group", confidence: float = 0.95,
              resamples: int = 2000, seed: int = 0) -> dict:
    """Measure nominal-label agreement on one row per annotated case.

    Missing reference or judge labels are unavailable observations, not negative
    labels. Agreement and kappa condition on paired labels. The agreement
    interval resamples whole independent groups with equal weight per case;
    it is not a confidence interval for kappa or for objective correctness.
    """
    _settings(confidence, resamples, seed)
    if not isinstance(group_key, str) or not group_key:
        raise ValueError("group_key must be a nonempty field name")
    if labels is not None:
        if not isinstance(labels, list) or not labels or any(
                not isinstance(label, str) or not label for label in labels):
            raise ValueError("labels must be a nonempty list of nonempty strings")
        if len(labels) != len(set(labels)):
            raise ValueError("labels must be unique")
        labels = list(labels)

    raw = Path(path).read_bytes()
    samples, seen, observed_labels = [], set(), set()
    for line_number, text in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            row = json.loads(text, object_pairs_hook=_object, parse_constant=_constant)
            json.dumps(row, allow_nan=False)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected an object")
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"line {line_number}: id must be a nonempty string")
        if identifier in seen:
            raise ValueError(f"duplicate judgment id: {identifier}")
        seen.add(identifier)
        group = row.get(group_key, identifier)
        if not isinstance(group, str) or not group:
            raise ValueError(f"line {line_number}: {group_key} must be a nonempty string")
        reference = _label(row.get("reference"), "reference", line_number)
        judge = _label(row.get("judge"), "judge", line_number)
        observed_labels.update(value for value in (reference, judge) if value is not None)
        slices = row.get("slices", {})
        if not isinstance(slices, dict) or any(
                value is not None and type(value) not in (str, int, float, bool)
                for value in slices.values()):
            raise ValueError(f"line {line_number}: slices must contain JSON scalar labels")
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"line {line_number}: metadata must be an object")
        if reference is None:
            status = "missing_both" if judge is None else "missing_reference"
        else:
            status = "missing_judge" if judge is None else "paired"
        samples.append({"sample_id": identifier, "group": group, "reference": reference,
                        "judge": judge, "status": status, "slices": slices,
                        "metadata": metadata,
                        "agreement": float(reference == judge) if status == "paired" else None})
    if not samples:
        raise ValueError("judgment study contains no records")
    if labels is None:
        labels = sorted(observed_labels)
    elif observed_labels - set(labels):
        raise ValueError(f"observed labels absent from declared labels: {sorted(observed_labels - set(labels))}")

    # Sort identities so bootstrap draws are independent of source row order.
    samples.sort(key=lambda item: item["sample_id"])
    paired = [sample for sample in samples if sample["status"] == "paired"]
    total, n = len(samples), len(paired)
    statuses = Counter(sample["status"] for sample in samples)
    indices = {label: index for index, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for sample in paired:
        matrix[indices[sample["reference"]]][indices[sample["judge"]]] += 1
    reference_counts = [sum(row) for row in matrix]
    judge_counts = [sum(row[index] for row in matrix) for index in range(len(labels))]
    matches = sum(matrix[index][index] for index in range(len(labels)))
    agreement = matches / n if n else None
    expected_numerator = sum(left * right for left, right in zip(reference_counts, judge_counts))
    # Integer counts avoid cancellation when marginal agreement is nearly one.
    denominator = n * n - expected_numerator
    if not n:
        kappa, kappa_reason = None, "no_paired_labels"
    elif denominator == 0:
        kappa, kappa_reason = None, "degenerate_paired_marginals"
    else:
        kappa = (n * matches - expected_numerator) / denominator
        kappa_reason = None
    entries = [{"group": sample["group"], "weight": 1,
                "mean": sample["agreement"]} for sample in paired]
    interval, interval_reason = _interval(entries, confidence, resamples, seed)
    if not paired:
        interval_reason = "no_paired_labels"
    per_class = []
    for index, label in enumerate(labels):
        tp, support, predicted = matrix[index][index], reference_counts[index], judge_counts[index]
        per_class.append({"label": label, "reference_count": support, "judge_count": predicted,
                          "matches": tp, "precision": tp / predicted if predicted else None,
                          "recall": tp / support if support else None,
                          "f1": 2 * tp / (support + predicted) if support + predicted else None})
    return {
        "schema_version": "1.0", "kind": "judge_calibration",
        "source": {"format": "judge-labels-jsonl", "sha256": hashlib.sha256(raw).hexdigest()},
        "estimand": "equal_case_exact_agreement_conditional_on_paired_labels",
        "group_key": group_key, "agreement": agreement, "interval": interval,
        "interval_reason": interval_reason, "kappa": kappa, "kappa_reason": kappa_reason,
        "expected_agreement": expected_numerator / (n * n) if n else None,
        "confusion": {"labels": labels, "rows": matrix, "orientation": "reference_rows_judge_columns"},
        "per_class": per_class,
        "coverage": {"total_samples": total, "paired_samples": n,
                     "paired_fraction": n / total,
                     "reference_observed": total - statuses["missing_reference"] - statuses["missing_both"],
                     "judge_observed": total - statuses["missing_judge"] - statuses["missing_both"],
                     "missing_reference": statuses["missing_reference"] + statuses["missing_both"],
                     "missing_judge": statuses["missing_judge"] + statuses["missing_both"],
                     "missing_both": statuses["missing_both"],
                     "independent_groups": len({sample["group"] for sample in samples}),
                     "paired_groups": len({sample["group"] for sample in paired})},
        "status_counts": {status: statuses[status] for status in
                          ("paired", "missing_reference", "missing_judge", "missing_both")},
        "samples": samples,
    }
