"""A command target: infer physical parameters from linearized observations.

This deterministic baseline receives only the evaluation request, never the
reference answer. It fits an intercept to absorb a constant calibration offset.
"""

from __future__ import annotations

import json
import math
import sys


def solve(request: dict) -> dict:
    observation = request["input"]
    x = [float(value) for value in observation["x"]]
    y = [float(value) for value in observation["y"]]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("linear fitting needs at least two paired observations")
    if not all(math.isfinite(value) for value in x + y):
        raise ValueError("observations must be finite")
    x_mean, y_mean = math.fsum(x) / len(x), math.fsum(y) / len(y)
    variance = math.fsum((value - x_mean) ** 2 for value in x)
    if variance == 0:
        raise ValueError("a slope cannot be identified from constant x")
    slope = math.fsum((a - x_mean) * (b - y_mean) for a, b in zip(x, y)) / variance
    intercept = y_mean - slope * x_mean
    quantity = observation["quantity"]
    if quantity in ("relative_refractive_index", "spring_stiffness"):
        estimate = slope
    elif quantity == "focal_length":
        estimate = 1.0 / intercept
    elif quantity == "decay_time":
        estimate = -1.0 / slope
    else:
        raise ValueError(f"unknown physical quantity: {quantity}")
    residual_sum_squares = math.fsum((b - slope * a - intercept) ** 2 for a, b in zip(x, y))
    return {
        "output": estimate,
        "usage": {"observations": len(x)},
        "metadata": {"method": "centered-least-squares", "quantity": quantity,
                     "slope": slope, "intercept": intercept,
                     "residual_sum_squares": residual_sum_squares},
    }


if __name__ == "__main__":
    json.dump(solve(json.load(sys.stdin)), sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
