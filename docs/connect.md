# Connect a harness or research workspace

The Python API, command line and MCP server call the same runner, scorers and analysis functions. Model clients and agent harnesses remain responsible for generation and tool execution; AI Evaluation supplies experiment identity, trial records, scoring and comparisons.

## Install and discover

From a checkout:

```bash
python -m pip install -e ".[mcp]"
ai-eval capabilities
ai-eval benchmark examples/retrieval/benchmark.json
```

Core execution and reports use the Python standard library. The `mcp` extra adds the optional MCP SDK. `capabilities` reports installed versions, native operations and discovered entry-point names. It does not connect to a model provider.

## Predictions and command targets

Prediction JSONL supplies recorded outputs, keyed by sample ID and optionally trial:

```json
{"sample_id":"q0","output":["doc-0","context-0"],"usage":{"input_tokens":120,"output_tokens":14}}
```

Without `trial`, the output is reused for every scheduled trial. This is useful for importing deterministic predictions; repeating an identical recorded answer does not create independent generation attempts. With `trial`, numbering starts at zero.

```bash
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/baseline.jsonl \
  --system-name baseline --system-revision 1 \
  --system-config '{"retrieval":{"k":2}}' \
  --output runs/baseline

ai-eval summarize runs/baseline --metric recall --slice-by variant \
  --json-output reports/baseline.json
```

For a live model, tool-using agent, retriever or simulator, use the command target. `--command-json` is an executable and argument array, without shell interpretation:

```bash
ai-eval run examples/numeric/benchmark.json \
  --command-json '["python","examples/numeric/solver.py"]' \
  --system-name physical-parameter-solver --system-revision 1 \
  --workers 4 --timeout 30 --output runs/physics
```

Each trial starts one process. It receives one UTF-8 JSON request on stdin:

```json
{"sample_id":"lens-a","input":{"quantity":"focal_length","x":[2,3,4,5,6],"y":[8,7,6,5,4],"x_unit":"1/m","y_unit":"1/m"},"trial":0,"seed":17,"metadata":{}}
```

The process writes one JSON response to stdout and sends diagnostics to stderr:

```json
{"output":0.1,"usage":{"observations":5},"metadata":{"method":"centered-least-squares","quantity":"focal_length"}}
```

The request omits the reference answer. `input` may contain text, structured observations or artifact references. Pass the trial seed to the generation policy when supported, and record the model snapshot, sampling settings, harness revision and tool configuration in `--system-config`. The command inherits the caller's working directory and environment; use absolute script paths when a host launches from another directory. The timeout applies to each command invocation. Native Python targets own their own external-operation deadlines.

`--resume` preserves every complete recorded outcome, including failures, and schedules only unrecorded trials. It requires a matching benchmark, system and target contract. Use a new output directory for a changed experiment.

## Compare and inspect

```bash
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/candidate.jsonl \
  --system-name candidate --system-revision 1 \
  --system-config '{"retrieval":{"k":2}}' --output runs/candidate

ai-eval compare runs/baseline runs/candidate --metric recall \
  --slice-by variant --resamples 2000 --seed 17 \
  --json-output reports/comparison.json

ai-eval samples runs/candidate --limit 20 --offset 0 \
  --slice-key variant --slice-value '"paraphrase"' --include-output

ai-eval report runs/baseline --against runs/candidate \
  --metric recall --slice-by variant --output reports/comparison.html
```

The paired difference is candidate minus baseline. `improvement` also accounts for a metric whose direction is `minimize`. The default comparison requires every planned trial to be scored in both runs. `--missing complete_cases` explicitly excludes incomplete samples and records their IDs and coverage; it changes the population being estimated.

The HTML file includes the complete analysis JSON, sortable sample rows, a text filter, pagination and a JSON download. It opens without a server or network connection. Report rendering does not recalculate scores or uncertainty. `--json-output` and report output paths create new files.

Use `ai-eval pass-at-k RUN --metric success --k 4` for completed binary success trials from a common sampling policy. The estimator and its assumptions are described in [contracts](contracts.md).

## MCP

Generate configuration using the interpreter with MCP support installed:

```bash
ai-eval connect --root "$PWD" --output ai-evaluation.mcp.json
```

The resulting `mcpServers.ai-evaluation` entry launches the absolute Python executable with `-m ai_eval serve --root <absolute workspace>`. Copy that entry into your host's MCP server configuration. Hosts that configure servers through a UI can use the same command and arguments. The generated file is configuration; loading it is the host's responsibility.

The stdio server resolves data and result paths relative to `--root`, independently of the host's working directory. Commands passed to `eval_run` are explicit executable/argument arrays; their script paths follow the command's working-directory behavior described above.

| Tool | Purpose |
|---|---|
| `eval_capabilities` | Discover operations, installed extensions and response limits. |
| `eval_benchmark` | Read benchmark identity, protocol, metric definitions and population size. |
| `eval_run` | Execute predictions, a command or an installed target and return run identity and outcome counts. |
| `eval_summarize` | Return a metric estimate, group interval, coverage and slice overview. |
| `eval_compare` | Return a paired comparison with explicit missing-data treatment. |
| `eval_import` | Import Gym export-v1 or Inspect JSON observations into the run contract. |
| `eval_samples` | Read filtered, paginated trial records. |
| `eval_report` | Write a complete interactive summary or comparison HTML file. |
| `eval_pass_at_k` | Estimate success within k attempts from complete binary trials. |
| `eval_calibrate_judge` | Compare categorical judge labels with reference judgments and inspect agreement. |

MCP summaries omit the per-sample array and retain up to 20 slices and exclusions, with total counts. Supply `output` to `eval_summarize`, `eval_compare` or `eval_pass_at_k` to save the complete JSON. `eval_samples` defaults to 20 rows and accepts up to 100; optional output previews stop at 2,000 characters. Full outputs remain in `results.jsonl`. An agent can inspect a few relevant records without loading the entire experiment into context.

Judge calibration uses JSONL records such as `{"id":"case-1","group":"source-1","reference":"pass","judge":"fail"}`. Run `ai-eval calibrate-judge judgments.jsonl --labels '["pass","fail"]' --json-output reports/calibration.json`, or use `eval_calibrate_judge` with `source` and optional `output`. It reports label agreement, a group-bootstrap interval, Cohen's kappa and a confusion matrix. Reference judgments are a chosen comparison standard; agreement does not establish correctness. MCP returns up to 20 matrix labels and per-class records with explicit truncation counts. Saved JSON retains all labels and records.

A useful harness sequence is `eval_benchmark` → `eval_run` → `eval_summarize`, then `eval_compare` for a stated candidate change. Inspect coverage and selected trial records before interpreting a difference. Use `eval_report` when the result needs human review.

## Import observed experiments

```bash
ai-eval import-gym gym-export.jsonl --output runs/gym \
  --system-name tool-agent --system-revision experiment-1 \
  --benchmark-name gym-study --revision 1

ai-eval import-inspect inspect-log.json --output runs/inspect \
  --scorer latency --direction minimize
```

Inspect accepts version-2 JSON logs. Select a numeric component of a structured score with `--value-key`. Convert binary `.eval` files using Inspect before importing. Original source data and provenance remain with the imported run; aggregate-only tables do not supply sample observations. See [scoring and import semantics](scoring.md) for field mappings and comparison requirements.

## Python and extensions

```python
from ai_eval.analysis import compare
from ai_eval.engine import run
from ai_eval.targets import PredictionTarget

run("benchmark.json", "runs/system-a",
    system={"name": "system-a", "revision": "1", "config": {}},
    target=PredictionTarget("predictions.jsonl"), workers=4)
result = compare("runs/system-a", "runs/system-b", "accuracy")
```

A Python target is a callable accepting the same request and returning the same response as the command protocol. Existing framework integrations can produce predictions, implement that callable, or export sample observations for an importer. Installed extensions use the `ai_eval.scorers`, `ai_eval.targets` and `ai_eval.importers` entry-point groups; `capabilities` lists the names actually installed. Scorer selection is part of the benchmark contract.

To expose a target factory from an installed package, register its import location in that package's `pyproject.toml`:

```toml
[project.entry-points."ai_eval.targets"]
my-harness = "my_package.evaluation:create_target"
```

`create_target(config)` receives the target configuration dictionary and returns the request callable. Select it with:

```bash
ai-eval run benchmark.json --target-plugin my-harness \
  --target-config '{"endpoint":"local","batch_size":8}' \
  --system-name research-agent --system-revision experiment-1 \
  --output runs/research-agent
```

MCP uses `target_plugin` and `target_config` on `eval_run`. Choose exactly one of predictions, command or plugin. Factory configuration belongs to the target execution identity; the system description should also identify the scientific condition, including externally managed model and harness state. Plugin targets are Python callables and own their external-operation deadlines.

For experiment design and complete executable examples, use [workflows](workflows.md). For the complete record and analysis contract, use [contracts](contracts.md).
