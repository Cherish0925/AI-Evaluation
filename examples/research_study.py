"""Study retrieval components, resource tradeoffs and benchmark composition.

Usage: python examples/research_study.py [--output runs/research-study]

The corpus is authored; all four retrieval algorithms actually execute. The
example needs no model service or credentials. Without --output, generated
observations and reports are removed when the example finishes.
"""

from __future__ import annotations

import argparse
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


TOPICS = (
    ("transport", "fast vehicle", "quick automobile"),
    ("energy", "energy storage", "power reservoir"),
    ("signals", "clear signal", "clean waveform"),
    ("water", "water purification", "liquid filtration"),
    ("memory", "memory retrieval", "recollection lookup"),
    ("thermal", "thermal control", "temperature regulation"),
)
SYNONYMS = {word: synonym for _, query, paraphrase in TOPICS
            for word, synonym in zip(query.split(), paraphrase.split())}
CONDITIONS = (("00", False, False), ("10", True, False),
              ("01", False, True), ("11", True, True))


def retrieve(request: dict, *, expansion: bool, phrases: bool) -> dict:
    """Rank documents using term coverage and an optional adjacent-phrase bonus.

    Count the string equality operations in the scoring loops. This is a small
    algorithmic work measure, not model tokens, elapsed time or hardware FLOPs.
    """
    query = request["input"]["query"].split()
    terms = [[word, SYNONYMS[word]] if expansion else [word] for word in query]
    phrase_variants = list(product(*terms)) if phrases else []
    comparisons = 0
    ranked = []
    for document in request["input"]["documents"]:
        tokens = document["text"].split()
        score = 0
        for alternatives in terms:
            matches = []
            for token in tokens:
                for alternative in alternatives:
                    matches.append(token == alternative)
                    comparisons += 1
            score += any(matches)
        phrase_match = False
        for left, right in zip(tokens, tokens[1:]):
            for first, second in phrase_variants:
                same_first, same_second = left == first, right == second
                comparisons += 2
                phrase_match |= same_first and same_second
        score += phrase_match
        ranked.append((score, document["id"]))
    # max preserves candidate order when scores tie.
    selected = max(ranked, key=lambda item: item[0])[1]
    return {"output": [selected], "usage": {"token_comparisons": comparisons},
            "metadata": {"query_expansion": expansion, "phrase_reranking": phrases}}


def write_benchmark(directory: Path) -> Path:
    directory.mkdir()
    samples = []
    for topic, query, paraphrase in TOPICS:
        first, second = query.split()
        for variant, distractor, relevant in (
            ("direct", f"{first} background", query),
            ("word-order", f"{first} related {second}", query),
            ("paraphrase", f"{first} related {second}", paraphrase),
        ):
            samples.append({
                "id": f"{topic}-{variant}", "group": topic,
                "slices": {"variant": variant, "topic": topic},
                "input": {"query": query, "documents": [
                    {"id": "distractor", "text": distractor},
                    {"id": "relevant", "text": relevant},
                    {"id": "neutral", "text": "unrelated material"},
                ]},
                "target": ["relevant"],
            })
    (directory / "samples.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples), encoding="utf-8")
    benchmark = {
        "schema_version": "1.0", "name": "retrieval-components", "revision": "1",
        "description": "Authored lexical retrieval cases: expansion × phrase reranking.",
        "samples": "samples.jsonl",
        "metrics": [{"name": "recall", "scorer": "recall_at_k",
                     "direction": "maximize", "params": {"k": 1}}],
        "protocol": {"trials": 1, "seed": 17},
        "metadata": {"origin": "authored-corpus; executed lexical algorithms",
                     "question": "Does query expansion help independently of phrase reranking?",
                     "independent_unit": "topic", "resource": "string equality operations"},
    }
    path = directory / "benchmark.json"
    path.write_text(json.dumps(benchmark, indent=2) + "\n", encoding="utf-8")
    return path


def cli(*arguments: str | Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "-m", "ai_eval", *(str(value) for value in arguments)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"ai-eval {arguments[0]}: {completed.stderr.strip()}")
    return json.loads(completed.stdout)


def close(actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f"expected {expected}, received {actual}")


def run_study(output: Path) -> dict:
    benchmark = write_benchmark(output / "cases")
    runs = []
    for cell, expansion, phrases in CONDITIONS:
        destination = output / cell
        command = [sys.executable, str(Path(__file__).resolve()), "--target",
                   "--expansion", str(int(expansion)), "--phrases", str(int(phrases))]
        cli("run", benchmark, "--command-json", json.dumps(command),
            "--system-name", f"lexical-{cell}", "--system-revision", "1",
            "--system-config", json.dumps({"query_expansion": expansion, "phrase_reranking": phrases}),
            "--workers", "2", "--timeout", "20", "--output", destination)
        runs.append(destination)
    study = cli("study", *runs, "--metric", "recall", "--resource", "usage.token_comparisons",
                "--slice-by", "variant", "--json-output", output / "systems.json",
                "--html-output", output / "systems.html")
    effects = cli("factorial", *runs, "--metric", "recall",
                  "--factor-a", "query_expansion", "--factor-b", "phrase_reranking",
                  "--slice-by", "variant", "--json-output", output / "factorial.json",
                  "--html-output", output / "factorial.html")
    panel = cli("analyze-benchmark", *runs, "--metric", "recall", "--slice-by", "variant",
                "--json-output", output / "benchmark-analysis.json",
                "--html-output", output / "benchmark-analysis.html")

    expected_means = (1 / 3, 1 / 3, 2 / 3, 1.0)
    expected_work = (40 / 3, 80 / 3, 62 / 3, 56.0)
    for system, mean, work in zip(study["systems"], expected_means, expected_work):
        close(system["mean"], mean)
        close(system["resource_mean"], work)
    for name, expected in (("main_a", 1 / 6), ("main_b", 1 / 2), ("interaction", 1 / 3)):
        close(effects["effects"][name]["estimate"], expected)
    if study["pareto_frontier"] != [0, 2, 3]:
        raise AssertionError("expected lexical, phrase and combined conditions on the frontier")
    if panel["summary"]["ceiling_samples"] != 6:
        raise AssertionError("expected the six direct cases to be at this panel's ceiling")

    return {
        "origin": "authored-corpus; executed lexical algorithms",
        "question": "Does query expansion help independently of phrase reranking?",
        "population": {"cases": 18, "topic_groups": 6, "trials_per_case": 1},
        "conditions": [
            {"cell": cell, "recall_at_1": system["mean"],
             "mean_token_comparisons": system["resource_mean"]}
            for (cell, _, _), system in zip(CONDITIONS, study["systems"])
        ],
        "effects": {name: effects["effects"][name]["estimate"]
                    for name in ("main_a", "main_b", "interaction")},
        "observations": [
            "Expansion adds no recall without phrase reranking; it adds 1/3 when phrase reranking is active.",
            "Expansion alone uses more comparisons at unchanged recall and lies off the observed frontier.",
            "Six direct cases are solved by every reference system; the other variants separate the systems.",
            "All topics share the same authored pattern, so group intervals do not describe unseen language diversity.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory to retain cases, runs and reports")
    parser.add_argument("--target", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expansion", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    parser.add_argument("--phrases", type=int, choices=(0, 1), default=0, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.target:
        result = retrieve(json.load(sys.stdin), expansion=bool(arguments.expansion),
                          phrases=bool(arguments.phrases))
    elif arguments.output is None:
        with tempfile.TemporaryDirectory(prefix="ai-evaluation-research-") as temporary:
            result = run_study(Path(temporary))
        result["artifacts"] = "temporary; use --output to retain cases, runs and reports"
    else:
        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        result = run_study(output)
        result["artifacts"] = str(output)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
