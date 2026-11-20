# Evaluation contracts

Schema version is `1.0`; the initial package version is `0.1.0`. Python package: `ai_eval`; CLI: `ai-eval`.

## Benchmark

A JSON specification references UTF-8 JSONL samples relative to its directory:

```json
{
  "schema_version": "1.0",
  "name": "retrieval-study",
  "revision": "1",
  "description": "Compare retrieval on independent topics.",
  "samples": "samples.jsonl",
  "metrics": [{"name": "recall", "scorer": "recall_at_k", "direction": "maximize", "params": {"k": 5}}],
  "protocol": {"trials": 1, "seed": 17},
  "metadata": {}
}
```

Each sample has `id`, `input`, `target`, optional `group` (defaults to its ID), `slices` (scalar labels), positive finite `weight` (defaults to 1), and `metadata`. Inputs/targets retain JSON structure; modality artifacts can be referenced there. Group identifies the independent unit for statistical resampling, such as source document, task family, scene or experimental specimen. Trial numbering starts at zero. Metrics have unique names and a declared optimization direction. A benchmark ID hashes the specification and sample bytes, with the physical sample path replaced by its content digest.

`load_benchmark(path)` returns a dict containing the specification, loaded `samples`, `benchmark_id` and `sample_sha256`. `system_id(system)` hashes a system description containing `name`, `revision`, optional `kind`, and `config`. Model, harness, retrieval and generation configurations belong in that description.

## Execute

`run(benchmark_path, output, *, system, target, workers=1, resume=False)` calls `target(request)` for each sample/trial. A request contains `sample_id`, `input`, `trial`, `seed` and sample `metadata`; the target answer is supplied only to the scorer. Target response: `{"output": <JSON>, "usage": {}, "metadata": {}}`. A supplied target is responsible for its own external operation deadlines. The command adapter enforces a per-call deadline.

A run directory contains `manifest.json` and `results.jsonl`. The manifest records `schema_version`, `run_id`, `benchmark` (`id`, `name`, `revision`, `sample_sha256`, `protocol`, `metrics`), `system`, `system_id`, `planned_trials`, sample descriptors (`id`, `group`, `slices`, `weight`), and engine/runtime metadata. Each result records:

```json
{
  "schema_version": "1.0",
  "benchmark_id": "<id>",
  "system_id": "<id>",
  "sample_id": "topic-1",
  "group": "source-1",
  "slices": {"domain": "science"},
  "weight": 1,
  "trial": 0,
  "seed": 17,
  "status": "ok",
  "output": ["doc-1", "doc-3"],
  "scores": {"recall": 0.5},
  "errors": {},
  "elapsed_seconds": 0.02,
  "usage": {},
  "metadata": {}
}
```

Statuses: `ok`, `target_error`, `score_error`. Unavailable metric values are null. Target errors appear under `errors.target`; scorer errors under their metric names. Recorded outcomes are retained on resume, including failures. Resuming requires the same benchmark, system and execution contract; interrupted tasks without a complete durable record are issued again. A new experiment gets a new output directory.

`load_run(path)` returns `{"manifest": <dict>, "results": <list>}`. It validates benchmark/system IDs, unique `(sample_id, trial)` keys, declared metrics, finite values and membership in the manifest's sample/trial population. It can load a partial run; analysis reports coverage.

## Scoring

`ai_eval.metrics.score(metric_spec, output, target) -> float`. Native scorers: `exact_match`, `numeric`, `set_f1`, `token_f1`, `recall_at_k`, `ndcg_at_k`, `brier`, `log_loss`. `numeric` yields tolerance-based accuracy. Predictions and references are structured according to each scorer's guide. Installed scorer functions register in `ai_eval.scorers`; there is no universal score across unrelated metrics.

## Analysis

`summarize(run_path, metric, *, confidence=0.95, resamples=2000, seed=0, slice_by=None)` reports the weighted mean of per-sample means, sample/trial coverage, status counts, resource observations and optional slice results. For a partially scored sample its observed-trial mean is conditional on those trials completing; coverage remains explicit. Confidence intervals use percentile bootstrap of independent groups, keeping samples and attempts in each group together. A single independent group has no estimable interval.

`compare(left_path, right_path, metric, *, confidence=0.95, resamples=2000, seed=0, missing="error", slice_by=None)` reports right minus left on paired sample means. Native runs must share the same benchmark/protocol/metric contract. Default missing policy requires every planned trial scored for both runs. `missing="complete_cases"` retains only samples with complete scores for both systems and reports every exclusion. Bootstrap resamples paired independent groups. Direction-adjusted improvement is reported separately from raw score difference.

`pass_at_k(run_path, metric, k)` uses the unbiased estimator `1 - C(n-c,k)/C(n,k)` per sample for binary completed trials, requires all scheduled trials scored and `n >= k`, and averages with sample weights. It measures success among k attempts, not single-attempt accuracy or repeated-run reliability.

`report` writes JSON-backed HTML with summary/comparison tables, per-sample deltas and slices. It is a presentation of the analysis, not another scoring implementation.

## Interchange

Native prediction JSONL uses `sample_id`, optional `trial` (absence reuses that output for each trial), `output`, optional `usage` and `metadata`. Missing requested predictions are target errors.

Result importers produce the same run directory contract, retaining source format/revision, source bytes hash and original metadata. Gym imports use export-v1 rows and preserve unscored statuses, task/data identity and rollout metadata. Inspect imports use JSON logs and explicit scorer/value selection. Aggregate-only result tables do not become invented sample observations. Imported populations must be aligned explicitly before paired comparisons.

Python, CLI and MCP share these functions. Installed targets and importers use `ai_eval.targets` and `ai_eval.importers` entry points. MCP returns bounded summaries and paginated samples; result payloads stay in files.
