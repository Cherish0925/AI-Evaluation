# Workflows

Run commands from the repository root after `python -m pip install -e .`. Use a fresh output directory for a new experiment. `ai-eval capabilities` lists the installed operations and extensions; `ai-eval benchmark PATH` describes a benchmark without running it.

`python examples/run_examples.py` runs both bundled studies through the CLI, checks their known outcomes and removes temporary outputs afterward. Add `--output runs/examples` to retain the run records and HTML reports in a new directory.

## Compare two systems on the same cases

The [retrieval example](../examples/retrieval/benchmark.json) contains six synthetic cases with three labeled groups and authored predictions. It demonstrates the data flow and statistics; it is not a measurement of any real model or retriever.

```bash
ai-eval benchmark examples/retrieval/benchmark.json
ai-eval run examples/retrieval/benchmark.json --predictions examples/retrieval/baseline.jsonl \
  --system-name retriever --system-revision baseline --output runs/tutorial-baseline
ai-eval run examples/retrieval/benchmark.json --predictions examples/retrieval/candidate.jsonl \
  --system-name retriever --system-revision candidate --output runs/tutorial-candidate
ai-eval summarize runs/tutorial-baseline --metric recall --slice-by variant
ai-eval compare runs/tutorial-baseline runs/tutorial-candidate \
  --metric recall --slice-by variant --json-output runs/tutorial-comparison.json
ai-eval report runs/tutorial-baseline --against runs/tutorial-candidate \
  --metric recall --slice-by variant --output runs/tutorial-report.html
```

The authored mean recalls are `1/3` and `11/12`, with candidate-minus-baseline difference `7/12`. The interval resamples the three groups, not six independent queries. These tiny fixtures are useful for understanding the estimator, not for establishing generalization. The report presents the same comparison returned by the analysis API.

For a new study, create a benchmark beside its sample file using the same structure, then collect actual system predictions. Keep that benchmark fixed across systems. Specify the complete model/harness/retrieval configuration using `--system-config` as a JSON object. A revision should identify a concrete implementation or checkpoint, not a floating label such as `latest`.

## Execute a physical-parameter solver

The [numeric example](../examples/numeric/benchmark.json) executes a [least-squares solver](../examples/numeric/solver.py) on eight authored calibration series. It fits $y=ax+b$ with a free intercept and converts the fitted coefficient to the requested physical quantity:

| Observation relation | Recovered parameter |
| --- | --- |
| Sines of transmitted and incident angles | Relative refractive index $a$ |
| Spring extension and force | Stiffness $a$ |
| Inverse object and image distances | Thin-lens focal length $1/b$ |
| Time and logarithm of normalized amplitude | Exponential-decay time constant $-1/a$ |

Selected cases contain constructed residuals orthogonal to the linear fit, so the expected fitted parameters remain known exactly. These are controlled numerical examples, not instrument data. The solver receives observations and task metadata, computes the parameter, and returns fit diagnostics; it never receives the reference answer.

```bash
ai-eval run examples/numeric/benchmark.json \
  --command-json '["python", "examples/numeric/solver.py"]' \
  --system-name least-squares-physics --system-revision 1 \
  --system-config '{"method":"centered-least-squares"}' \
  --workers 2 --output runs/physics-parameters
ai-eval summarize runs/physics-parameters --metric accuracy --slice-by domain
ai-eval report runs/physics-parameters --metric accuracy --slice-by domain \
  --output runs/physics-parameters.html
```

All eight parameters should meet the declared numerical tolerance. The sample files preserve physical units; the target metadata records fitted slope, intercept and residual sum of squares. For an experimental study, replace the controlled observations with calibrated measurements and choose tolerances and uncertainty methods appropriate to the physical question.

## Inspect incomplete runs and repeated attempts

```bash
ai-eval samples runs/tutorial-baseline --limit 10 --include-output
ai-eval samples runs/tutorial-baseline --status target_error --limit 10
```

`summarize` exposes scored coverage, missing observations and statuses. `compare` requires complete planned scores on both sides by default. If a scientific question justifies analysis of complete samples only, use `--missing complete_cases` and examine the reported exclusions. This option changes the analyzed population; it does not repair the missing outcomes.

After interruption, repeat the original `run` command with `--resume` and the same inputs and configuration. Recorded results, including failures, are retained; only work without a complete durable record is issued again. To study a corrected implementation, use a new revision and run directory.

For stochastic systems, declare `protocol.trials` before collecting attempts and include `trial` in every prediction row. Omitting `trial` reuses one prediction across attempts; it does not create independent generations. A binary success metric with enough completed attempts can be analyzed with `ai-eval pass-at-k RUN --metric success --k 3`. See [research methods](research.md) for the difference between success within a search budget and repeated-run reliability.

## Connect a real model or harness

A command target receives one JSON request on standard input and writes one JSON response on standard output. The request includes `sample_id`, `input`, `trial`, `seed` and sample metadata. Return `{"output": ..., "usage": {...}, "metadata": {...}}`. Keep logging on standard error. The target receives the task input; scoring receives its reference answer.

Once an actual `my_harness.py` implements that protocol, invoke it as follows:

```bash
ai-eval run examples/retrieval/benchmark.json \
  --command-json '["python", "my_harness.py"]' \
  --system-name research-harness --system-revision experiment-1 \
  --system-config '{"model_revision":"your-checkpoint","tool_budget":20}' \
  --workers 4 --timeout 300 --output runs/harness-experiment-1
```

The command adapter enforces a per-call timeout. The harness owns its internal model, tool, memory and environment behavior. Report available token/tool usage and artifact references in the response. For persistent sessions or in-process execution, use a Python target callable with `ai_eval.engine.run`; that callable owns its external operation deadlines. Installed target factories can also be selected with `--target-plugin` and `--target-config`; [connection guidance](connect.md) describes registration.

For an ablation, freeze the benchmark and change one system component. If two components may interact, run all four combinations and analyze the interactions explicitly; pairwise CLI results alone do not fit a factorial model. To compare budgets, record each budget as part of system configuration and retain a separate run for each condition.

## Bring observations from an existing runner

These commands require real exported logs at the indicated paths:

```bash
ai-eval import-gym exports/rollouts.jsonl --output runs/gym-study \
  --system-name research-agent --system-revision experiment-1 \
  --benchmark-name project-tasks --revision 1

ai-eval import-inspect logs/study.json --scorer accuracy \
  --output runs/inspect-study
```

Gym input must match its export-v1 schema. Inspect input is a version-2 JSON evaluation log; select the score name actually present in that log, use `--value-key` for one field of a structured score, and set `--direction minimize` for a loss metric. The import retains observed outcomes and provenance. It does not execute the upstream task again. See [contracts](contracts.md) and [ecosystem](ecosystem.md) for the distinction between importing scores and evaluating predictions.

Before comparing imported runs, establish matching task populations, trial definitions and measurement semantics. Shared display names are insufficient. Aggregate-only published results remain aggregate observations; they cannot support this sample-paired comparison.

## Work through an agent interface

Install the optional MCP dependency with `python -m pip install -e '.[mcp]'`. Then write the configuration for the installed interpreter and project directory:

```bash
ai-eval connect --root . --output ai-evaluation.mcp.json
```

Use that configuration in an MCP-compatible host; [connection guidance](connect.md) describes the available tools. Start with capabilities and benchmark metadata, then run or import the experiment, request a bounded summary, and inspect relevant samples in pages. Full results remain in the run directory. Python, CLI and MCP call the same execution and analysis functions.

A useful research instruction supplies a decision and an experimental boundary:

> Determine whether memory revision B improves project completion at a fixed tool budget. Use the same held-out project population and verifier for both systems, preserve project-level groups, and distinguish incomplete measurement from task failure. Report the paired effect, coverage, uncertainty, resource use and the cases that could change the interpretation. Propose the next experiment that would distinguish a memory benefit from task familiarity.

For model judging, multimodal population statistics, adaptive benchmarks or long-horizon scientific outputs, first define the missing measurement or protocol described in [research methods](research.md). Integrate the corresponding implementation through an existing runner or extension point before claiming that the experiment has been performed.
