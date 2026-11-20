# Workflows

Run commands from the repository root after `python -m pip install -e .`. Choose a new output directory for a new experiment. `ai-eval capabilities` lists installed operations; `ai-eval benchmark PATH` describes the cases, groups, metrics and protocol.

## Study mechanisms, resources and benchmark cases

```bash
python examples/research_study.py --output runs/research-study
```

This standalone example constructs a corpus and executes four actual lexical retrieval algorithms through the command target. It needs no model service, credentials or other repository. The two factors are query expansion with a small synonym dictionary and an adjacent-phrase reranking bonus. Cell order is `00`, `10`, `01`, `11`: neither component, expansion only, phrases only, both.

Each of six topic groups contains three cases. The relevant document is identified by a reference used only during scoring; targets receive the query and candidate documents. Tie-breaking preserves candidate order.

| Case variant | Construction | Conditions that retrieve the relevant document |
| --- | --- | --- |
| Direct | Relevant document covers both query terms; distractor covers one | All four |
| Word order | Relevant document contains the query phrase; distractor separates the same terms | `01`, `11` |
| Paraphrase | Relevant document uses adjacent synonyms; distractor separates the original terms | `11` |

The resulting recall@1 values are `1/3`, `1/3`, `2/3`, `1`. The script counts the string equality operations in each algorithm's scoring loops. This resource measures algorithmic work in the example; it is not model token consumption, latency or monetary cost.

The output directory contains `cases/`, four run directories and three JSON/HTML analyses. The CLI calls below show how to perform those analyses directly, using separate report paths.

### Compare systems and resource use

```bash
ai-eval study runs/research-study/00 runs/research-study/10 \
  runs/research-study/01 runs/research-study/11 \
  --metric recall --resource usage.token_comparisons --slice-by variant \
  --json-output reports/retrieval-systems.json \
  --html-output reports/retrieval-systems.html
```

The first run is the default baseline. Every comparison uses the same retained population. Here expansion alone costs more than the baseline without improving recall, so it lies off the observed frontier. Baseline, phrase reranking and the combined system occupy different quality–work tradeoffs.

For a real model study, select a consistently recorded resource such as `usage.output_tokens`, `usage.tool_calls` or `elapsed_seconds`. Keep its units and accounting boundary fixed. Each budget or configuration is a separate condition; the frontier describes observed points.

### Estimate the interaction

```bash
ai-eval factorial runs/research-study/00 runs/research-study/10 \
  runs/research-study/01 runs/research-study/11 \
  --metric recall --factor-a query_expansion --factor-b phrase_reranking \
  --slice-by variant --json-output reports/retrieval-factorial.json \
  --html-output reports/retrieval-factorial.html
```

The expansion main effect is `1/6`, the phrase main effect is `1/2`, and the interaction is `1/3`. Expansion contributes zero when phrases are off and `1/3` when they are on. A comparison of expansion-only against baseline would miss the combined mechanism. [Factorial experiments](factorial-experiments.md) explains the cell order, contrasts and group intervals.

All six topics deliberately share the same mechanism and case proportions, so their group contrasts coincide. Narrow or degenerate intervals in this example do not establish how the algorithms generalize to new language, topics or distractor patterns. A substantive retrieval study would sample that diversity.

### Examine the benchmark

```bash
ai-eval analyze-benchmark runs/research-study/00 runs/research-study/10 \
  runs/research-study/01 runs/research-study/11 \
  --metric recall --slice-by variant \
  --json-output reports/retrieval-benchmark.json \
  --html-output reports/retrieval-benchmark.html
```

The six direct cases are at the ceiling for this panel. The other variants distinguish component behavior. Item–rest association excludes the item's whole topic group. Use these observations to design a next corpus with varied ambiguity, synonym coverage and phrase structure; the present panel does not determine which tasks matter for a broader population. See [benchmark development](benchmark-development.md).

A saved analysis can be rendered without recomputation:

```bash
ai-eval report --analysis reports/retrieval-factorial.json \
  --output reports/retrieval-factorial-copy.html
```

## Compare recorded predictions

The [prediction replay example](../examples/retrieval/benchmark.json) contains six authored retrieval cases in three groups. Its predictions are constructed records, rather than outputs of a running retriever.

```bash
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/baseline.jsonl \
  --system-name retriever --system-revision baseline --output runs/replay-baseline
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/candidate.jsonl \
  --system-name retriever --system-revision candidate --output runs/replay-candidate
ai-eval compare runs/replay-baseline runs/replay-candidate \
  --metric recall --slice-by variant --json-output reports/replay.json
```

Mean recall is `1/3` versus `11/12`, a candidate-minus-baseline difference of `7/12`. The interval resamples the three related-case groups. For your own study, replace the cases and predictions while keeping one benchmark fixed across systems. Record the actual checkpoint, prompts and harness configuration in `--system-config`.

## Execute a physical-parameter solver

The [numeric example](../examples/numeric/benchmark.json) executes a [least-squares solver](../examples/numeric/solver.py) on eight authored calibration series. It fits $y=ax+b$ with a free intercept and converts the fitted coefficient to a physical quantity:

| Observation relation | Recovered parameter |
| --- | --- |
| Sines of transmitted and incident angles | Relative refractive index $a$ |
| Spring extension and force | Stiffness $a$ |
| Inverse object and image distances | Thin-lens focal length $1/b$ |
| Time and logarithm of normalized amplitude | Exponential-decay time constant $-1/a$ |

Selected cases contain residuals constructed to be orthogonal to the fit, preserving known fitted parameters. These are numerical examples, not instrument observations. The target receives the series and task metadata, and computes the requested quantity without the reference answer.

```bash
ai-eval run examples/numeric/benchmark.json \
  --command-json '["python", "examples/numeric/solver.py"]' \
  --system-name least-squares-physics --system-revision 1 \
  --system-config '{"method":"centered-least-squares"}' \
  --workers 2 --output runs/physics-parameters
ai-eval summarize runs/physics-parameters --metric accuracy --slice-by domain
ai-eval report runs/physics-parameters --metric accuracy --slice-by domain \
  --output reports/physics-parameters.html
```

All eight parameters meet the declared numerical tolerance. The cases preserve physical units; output metadata contains slope, intercept and residual sum of squares. To study actual measurements, supply calibrated observations and choose tolerances and uncertainty methods appropriate to the physical question.

`python examples/run_examples.py --output runs/basic-examples` executes the replay and numerical examples together. Both example scripts use temporary directories when `--output` is omitted.

## Work with incomplete runs and repeated attempts

```bash
ai-eval samples runs/replay-baseline --limit 10 --include-output
ai-eval samples runs/replay-baseline --status target_error --limit 10
```

`summarize` reports scored coverage, missing observations and statuses. Comparison, study and factorial operations require all planned scores by default. `--missing complete_cases` explicitly selects samples complete in every participating run and reports exclusions. The resulting estimate concerns that retained population.

After an interruption, repeat the original `run` command with `--resume` and the same inputs and configuration. Recorded outcomes, including failures, remain; only unrecorded trials are scheduled. A changed implementation receives a new revision and run directory.

For stochastic systems, declare `protocol.trials` and record each attempt's `trial`. Omitting it in replay data reuses one prediction across attempts. A binary success metric with enough completed attempts supports `ai-eval pass-at-k RUN --metric success --k 3`; [research methods](research.md) distinguishes success within a search budget from repeated-run reliability.

## Bring your model or harness

A command target receives one JSON request on standard input and writes one JSON response on standard output. The request contains `sample_id`, `input`, `trial`, `seed` and sample metadata. Return `{"output": ..., "usage": {...}, "metadata": {...}}`, with logs on standard error. Scoring handles the reference answer separately.

Once your `my_harness.py` implements this protocol:

```bash
ai-eval run benchmark.json \
  --command-json '["python", "my_harness.py"]' \
  --system-name research-harness --system-revision experiment-1 \
  --system-config '{"model_revision":"your-checkpoint","tool_budget":20}' \
  --workers 4 --timeout 300 --output runs/harness-experiment-1
```

The harness owns generation, tools, memory and environment behavior. Supply available resource use and artifact references with each response. Python callables support persistent execution; target plugins expose installed factories. [Connection guidance](connect.md) describes these interfaces, and [ecosystem](ecosystem.md) covers importing observations from existing runners.

## Work through an agent interface

```bash
python -m pip install -e '.[mcp]'
ai-eval connect --root . --output ai-evaluation.mcp.json
```

Load the configuration in an MCP host. Start with capabilities and the relevant method resource or research prompt. Run the experiment, save its analysis and request a summary; then read selected sample records or sections of the saved analysis. The same data are available through Python, CLI and portable reports.

A useful instruction names a question, interventions and population:

> Determine whether query expansion and reranking improve retrieval independently or through an interaction. Run all four conditions on the same held-out queries, preserve source-level groups, and record the same resource measure throughout. Estimate the conditional effects and interaction, identify the observed quality–resource frontier, and examine which cases distinguish the systems. Use those observations to design the next experiment.

For a model judge, use [judge calibration](judges.md). For a new measurement or protocol, begin with its quantity, sampling unit and observable outcome in [research methods](research.md), then implement the execution or scoring operation it requires.
