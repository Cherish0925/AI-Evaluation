# AI Evaluation

**Benchmark design, reproducible evaluation, and statistical comparison for AI models, agents, and research systems.**

[中文](README.zh-CN.md) · [Start a study](docs/workflows.md) · [Research methods](docs/research.md) · [Connect an agent](docs/connect.md)

An evaluation connects a question, a task population, a versioned system, an execution protocol and observed outcomes. This environment makes those connections explicit so that a change in model, harness, data, tools or inference budget can be studied and understood.

Its scope includes language and multimodal models, retrieval, code generation, interactive agents, scientific predictions and autonomous research. The shared runtime handles structured cases and observations; specialized engines supply domain execution and measurements. A benchmark is a versioned experimental instrument, with a defined population and scoring rule.

## Run a complete example

Python 3.11 or newer. The core uses the standard library; MCP is optional.

```bash
git clone https://github.com/EnvFoundry/AI-Evaluation.git
cd AI-Evaluation
python -m pip install -e '.[mcp]'
python examples/run_examples.py --output runs/example
```

The example compares two authored retrieval systems and evaluates a numerical target through a subprocess. It produces run records, statistical comparisons and portable HTML reports. These small fixtures demonstrate the workflow; they are not measurements of a trained model.

For your own structured predictions:

```bash
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/baseline.jsonl \
  --system-name baseline --system-revision 1 --output runs/baseline
ai-eval run examples/retrieval/benchmark.json \
  --predictions examples/retrieval/candidate.jsonl \
  --system-name candidate --system-revision 1 --output runs/candidate
ai-eval compare runs/baseline runs/candidate --metric recall --slice-by variant
ai-eval report runs/baseline --against runs/candidate \
  --metric recall --output runs/comparison.html
```

## What is implemented

| Layer | Working capabilities |
| --- | --- |
| Study contract | Versioned benchmark and system identities; sample, group, slice and trial definitions; explicit metric direction and weights |
| Execution | Python callables, JSON subprocess targets, prediction replay and installed target plugins; concurrent local workers; resumable durable outcomes |
| Measurement | Exact match, numerical tolerance, set/token F1, recall@k, nDCG@k, Brier score and log loss; installed scorer plugins |
| Comparison | Weighted sample means; paired group bootstrap; complete-case policy; slice analysis; binary pass@k; coverage and resource observations |
| Judge calibration | Reference-label agreement, confusion matrix, per-class precision/recall, Cohen's kappa and grouped uncertainty with missing-label coverage |
| Existing runs | Gym export-v1 and Inspect JSON-v2 imports, with original source bytes, scorer identity and unavailable outcomes preserved |
| Research interfaces | Python, CLI, MCP, repository skills and agent plugin manifests; bounded discovery and paginated observations |
| Reports | Self-contained HTML with filtering, sorting and the exact analysis JSON; no separate scoring path |

The [contracts](docs/contracts.md) specify data semantics. The [scoring guide](docs/scoring.md) defines each metric. The [ecosystem map](docs/ecosystem.md) identifies reusable upstream engines and clearly distinguishes existing integrations from extension work.

Run `ai-eval calibrate-judge examples/judgments.jsonl` to inspect an authored label-comparison example. The [judge guide](docs/judges.md) explains interpretation and reference-label selection. The [Gym integration](docs/gym-integration.md) executes actual episodes, imports outcomes, compares policies and optionally builds Data Engineering releases.

## Design a study that answers the question

A meaningful comparison needs more than a score. Specify what changes and what stays fixed: model checkpoint, prompts, retrieval, memory, tools, environment, sampling policy and resource budget. Identify the independent unit before collecting repeated trials. Preserve execution failures and partial coverage so they cannot disappear inside an average.

The [research guide](docs/research.md) covers construct and ecological validity, grouped uncertainty, judge calibration, contamination, evolving task distributions, adaptive benchmark reuse, transfer, cost-quality curves and long-horizon scientific outcomes. The [design](docs/design.md) explains how to represent these studies without requiring every workload to become an agent episode.

For a new domain, write down the scientific claim, observable outcome, unit of analysis and comparison protocol first. Use an existing execution engine when it already supplies the right world and measurements. Add an adapter only when it preserves those meanings.

## Connect models and agents

```bash
ai-eval capabilities
ai-eval connect --root /absolute/path/to/studies --output mcp.json
```

The generated configuration uses the installed interpreter and can be loaded by an MCP host. A callable or command target can wrap a local model, hosted API, retrieval pipeline or complete agent harness. It receives the task input and trial seed; reference answers remain on the scoring side. Provider credentials and model serving stay in that target's normal runtime.

Read [connection instructions](docs/connect.md) for command JSON, plugin entry points, MCP and source checkout launchers. The repository's [evaluation skill](skills/ai-evaluation/SKILL.md) directs an agent to the relevant method and executable operation without loading the entire documentation set.

## Within EnvFoundry

| Environment | Responsibility |
| --- | --- |
| [Data Engineering](https://github.com/EnvFoundry/Data-Engineering) | Dataset construction, transformations, lineage and sampling |
| [Verifiable Agent Gym](https://github.com/EnvFoundry/Verifiable-Agent-Gym) | Executable tasks, interaction environments, trajectories and verifier outcomes |
| **AI Evaluation** | Benchmark design, measurement, experiment comparison and interpretation |
| [IOP](https://github.com/EnvFoundry/IOP) / [Optics-Photonics-AI-Lab](https://github.com/EnvFoundry/Optics-Photonics-AI-Lab) | Physical instrument operations and optics research workflows |

## Develop

```bash
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
python examples/run_examples.py
python -m build
```

Contributions should add a useful measurement, executable integration, research method or clearer experiment. See [contributing](CONTRIBUTING.md). Distributed scheduling, GPU serving, domain simulators and benchmark datasets remain with their respective systems; the current execution engine is local.

MIT license.
