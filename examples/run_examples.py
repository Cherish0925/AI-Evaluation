"""Run the bundled studies through the installed CLI and verify known outcomes.

Usage: python examples/run_examples.py [--output runs/examples]
Without --output all generated observations and reports live in a temporary
directory and are removed after successful validation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


EXAMPLES = Path(__file__).resolve().parent


def cli(*arguments: str | Path) -> dict:
    process = subprocess.run(
        [sys.executable, "-m", "ai_eval", *(str(value) for value in arguments)],
        check=False, capture_output=True, text=True,
    )
    if process.returncode:
        raise RuntimeError(f"ai-eval {arguments[0]} failed: {process.stderr.strip()}")
    return json.loads(process.stdout)


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"expected {expected}, received {actual}")


def run_examples(output: Path) -> dict:
    retrieval = EXAMPLES / "retrieval"
    baseline, candidate = output / "baseline", output / "candidate"
    for revision, predictions, destination in (
        ("baseline", retrieval / "baseline.jsonl", baseline),
        ("candidate", retrieval / "candidate.jsonl", candidate),
    ):
        cli("run", retrieval / "benchmark.json", "--predictions", predictions,
            "--system-name", "authored-retrieval", "--system-revision", revision,
            "--output", destination)
    summary = cli("summarize", baseline, "--metric", "recall")
    close(summary["mean"], 1 / 3)
    comparison = cli("compare", baseline, candidate, "--metric", "recall",
                     "--slice-by", "variant", "--json-output", output / "comparison.json")
    close(comparison["left_mean"], 1 / 3)
    close(comparison["right_mean"], 11 / 12)
    close(comparison["delta"], 7 / 12)
    if comparison["paired_samples"] != 6 or comparison["independent_groups"] != 3:
        raise AssertionError("retrieval population or independent groups changed")
    cli("report", baseline, "--against", candidate, "--metric", "recall",
        "--slice-by", "variant", "--output", output / "retrieval.html")

    numeric = EXAMPLES / "numeric"
    recovered = output / "numeric"
    command = json.dumps([sys.executable, str(numeric / "solver.py")])
    cli("run", numeric / "benchmark.json", "--command-json", command,
        "--system-name", "least-squares-physics", "--system-revision", "1",
        "--system-config", '{"method":"centered-least-squares"}',
        "--workers", "2", "--timeout", "20", "--output", recovered)
    numeric_summary = cli("summarize", recovered, "--metric", "accuracy", "--slice-by", "domain")
    close(numeric_summary["mean"], 1.0)
    if numeric_summary["status_counts"] != {"ok": 8}:
        raise AssertionError("expected all eight parameter-recovery tasks to complete")
    cli("report", recovered, "--metric", "accuracy", "--slice-by", "domain",
        "--output", output / "numeric.html")
    for report in (output / "retrieval.html", output / "numeric.html"):
        if '<script id="analysis" type="application/json">' not in report.read_text(encoding="utf-8"):
            raise AssertionError(f"report has no embedded analysis: {report}")
    return {"status": "passed", "origin": "authored-examples",
            "retrieval": {key: comparison[key] for key in
                          ("left_mean", "right_mean", "delta", "paired_samples", "independent_groups")},
            "numeric": {"accuracy": numeric_summary["mean"], "completed_tasks": 8}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory to retain run artifacts and HTML reports")
    arguments = parser.parse_args()
    if arguments.output is None:
        with tempfile.TemporaryDirectory(prefix="ai-evaluation-examples-") as temporary:
            result = run_examples(Path(temporary))
        result["artifacts"] = "temporary; removed after validation"
    else:
        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = run_examples(output)
        result["artifacts"] = str(output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
