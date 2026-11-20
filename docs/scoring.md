# Scoring and imported observations

The scoring unit is one sample attempt. `score(spec, output, target)` returns one finite number; execution records scorer exceptions as unavailable scores. Metric direction belongs in the benchmark definition. The metric name `target` is reserved for execution errors. Aggregation, uncertainty, repeated attempts and paired comparisons use the run contract in [contracts.md](contracts.md).

## Native score semantics

| Scorer | Output / target | Definition and parameters | Direction |
|---|---|---|---|
| `exact_match` | JSON / JSON | Canonical JSON equality. Object key order is irrelevant; array order and JSON types matter. Strings optionally use `strip=true`, `casefold=true`; neither is enabled by default. | maximize |
| `numeric` | number / number | Accuracy: `abs(output-target) <= max(abs_tol, rel_tol*abs(target))`. Defaults: `abs_tol=1e-8`, `rel_tol=0`. No extraction from prose. | maximize |
| `set_f1` | list of scalar IDs / list of scalar IDs | `2*intersection_size/(predicted_size+reference_size)` after removing duplicates. | maximize |
| `token_f1` | string / string | Multiset F1 over whitespace-separated tokens. Optional `casefold=true`; punctuation and articles remain. | maximize |
| `recall_at_k` | ranked IDs / relevant IDs | Unique relevant IDs in the first `k` results divided by all relevant IDs. Required positive integer `k`. | maximize |
| `ndcg_at_k` | ranked string IDs / object mapping IDs to nonnegative relevance | Discount is `1/log2(rank+1)`, with ranks starting at one. Default gain is `2**relevance-1`; `gain="linear"` uses relevance. Divide by the ideal top-`k` discounted gain. | maximize |
| `brier` | probability payload / observed outcome | Binary: `(p-y)**2`. Multiclass: sum of squared errors against the one-hot outcome. | minimize |
| `log_loss` | probability payload / observed outcome | Natural logarithm: `-log(probability of observed outcome)`. Optional `epsilon` explicitly caps the loss at `-log(epsilon)`. | minimize |

Set/token F1 is one when both sides are empty and zero when only one is empty. Retrieval ranks must contain unique IDs; duplicate results cannot earn additional credit. Recall with no relevant items and NDCG with zero ideal gain are undefined and produce a scoring error. IDs are type-sensitive JSON scalars; NDCG uses string IDs because its references are JSON object keys.

Probability scores accept either a number `p=P(y=1)` with numeric target `0` or `1`, or a class-probability object with a string target:

```python
from ai_eval.metrics import score

score({"scorer": "brier"}, 0.8, 1)                          # 0.04
score({"scorer": "brier"}, {"yes": 0.8, "no": 0.2}, "yes")  # 0.08
score({"scorer": "log_loss"}, {"yes": 0.8, "no": 0.2}, "yes")
score({"scorer": "recall_at_k", "params": {"k": 2}},
      ["doc-a", "doc-c", "doc-b"], ["doc-a", "doc-b"])    # 0.5
```

Binary Brier is in `[0,1]`; this multiclass convention is in `[0,2]`. They must not be mixed into the same reported metric. Probabilities must be finite, within `[0,1]`, and multiclass probabilities must sum to one within absolute tolerance `1e-8`. No normalization, confidence extraction from prose or conversion of logits happens implicitly. Uncapped log loss at zero target probability is infinite; the finite-scalar contract records that as a scorer error. To use a capped estimand, declare `params: {"epsilon": 1e-15}` in the metric specification. The cap changes the scoring rule and remains in benchmark identity.

Installed scorer entry points use group `ai_eval.scorers` and implement `fn(output, target, **params) -> float`. Native names cannot be overridden; ambiguous plugin registrations are errors. A model judge is an ordinary scorer plugin with an explicit rubric, judge/version and generation settings in `params`; the core does not invent a judge or its measurements.

Native runs bind the scorer implementation in each manifest metric: package version/module SHA-256 for native scorers, distribution name/version/entry point for plugins. The benchmark ID identifies the declared specification and sample bytes. Comparison ignores only the native implementation's package version when a valid recorded module identity and all other metric fields match: an analysis-only package upgrade preserves comparability when measurement code and definition are unchanged. Scorer code, parameters, declared bounds and plugin versions remain part of comparison. Resume still requires the same full metric descriptor. Command targets fingerprint the executable and file arguments when each run or resume begins; command files and dependencies must remain fixed during that invocation. Arbitrary Python target state belongs in the declared system configuration or its explicit target identity.

## Verifiable Agent Gym

```python
from ai_eval.importers import import_gym

import_gym("episodes.jsonl", "runs/gym-policy-a",
           system={"name": "policy-a", "revision": "checkpoint-24000"},
           benchmark_name="tool-use-suite", revision="2026-11")
```

The adapter reads export-v1 JSONL produced by `agent_gym.export.export_run` and `agent_gym.rollout.export_rollouts`. A record contains `schema_version=1`, `episode_id`, `task`, `agent`, `status`, `reward`, `verification`, and `trajectory`; rollout exports additionally contain `rollout` and `policy_steps`.

For rollout exports, `rollout.attempt` becomes the zero-based trial. Task, suite/group, epoch, data reference and action/time limits identify a sample. Repeated attempts share that sample; repeated source epochs retain distinct sample IDs but remain clustered by underlying task and data reference. Exported `group_id` includes policy identity, so it is deliberately not the statistical grouping key. One import contains one policy name/revision/generation configuration, retained in `system.config.gym_policy`. A task group that changes its data or execution budget is rejected.

Ordinary exports lack rollout budgets, data references and attempt numbers. Their task specification identifies the sample, and encounter order within each task defines trials. This absence is recorded. Both formats retain the full original row and a byte-identical source file. `passed`/`failed` with a finite reward become scored observations. Truncation, cancellation and errors remain unscored target errors; a terminal result without reward remains a scoring error. No elapsed time or usage is fabricated.

The imported population is the exported task groups crossed with attempt slots from zero through the largest observed attempt. This exposes internal gaps but cannot discover entirely absent tasks or trailing omitted attempts. The ordinary exporter can omit unscored episodes, and the rollout exporter excludes setup failures even when including other unscored episodes. Imported coverage therefore describes this declared exported population, not the full original rollout run. Preserve the original run/suite artifacts when that distinction matters.

## Inspect

```python
from ai_eval.importers import import_inspect

import_inspect("inspect-log.json", "runs/inspect-model-a", scorer="match")
import_inspect("inspect-log.json", "runs/inspect-loss",
               scorer="grader", value_key="loss", direction="minimize")
```

The adapter reads Inspect JSON log version `2`, using `eval`, `plan` and per-sample `samples` records. Convert binary `.eval` logs with Inspect's `inspect log convert` command before import. Aggregate results alone do not identify sample outcomes and are rejected.

Choose the scorer by its exact `sample.scores` key. `value_key` selects one direct key from an object-valued score. Finite numeric values are preserved; Inspect's `C`/`I` and boolean `true`/`false` map explicitly to one/zero. Other text values, including `P`, receive no guessed conversion. Non-scalar or missing values become unavailable scores with an explanation. Inspect's `NaN` unscored sentinel becomes an unavailable score, retaining its reason. Source bytes remain exact; nonfinite source values appear as `{"$source_nonfinite": "NaN"}` (or `Infinity`/`-Infinity`) in JSON metadata. Direction defaults to `maximize`; supply `minimize` for losses. This declared selection and direction are stored in the metric specification.

Inspect epochs start at one and map to zero-based trials. The population is the observed sample IDs crossed with configured epochs; if the epoch count is absent, the largest observed epoch supplies the count. Entirely unobserved sample IDs are not reconstructed. Numeric and string IDs remain distinct. The model identifier, generation configuration, solver plan and model roles define the imported system; no immutable model snapshot is inferred from an alias. Evaluation configuration, source revision/packages, dataset/scorer definitions and per-sample task/budget conditions define comparability. Sample metadata participates in task identity because grading templates can use it. Named model roles also participate in protocol identity because they can bind the judge; imports do not infer which arbitrary roles belong to the candidate versus the measurement instrument. For Inspect's `model_graded_qa`/`model_graded_fact`, the effective judge follows documented explicit-model, bound-role, then evaluated-model precedence; changing a default judge cannot masquerade as a candidate-only comparison. Mid-run configuration changes and a sample that changes task/budget/metadata across epochs require separate stable-condition imports.

The default independent unit is the sample ID. A dataset author can declare larger dependency groups, slices and weights in sample metadata:

```json
{"ai_eval": {"group": "source-document-17", "slices": {"domain": "physics"}, "weight": 1}}
```

Original errors normally remain unscored observations. If the source explicitly declares `eval.config.score_on_error=true` and the selected score is usable, that score remains a scored observation: replacing a declared failure score with missingness would change the estimand. Its original execution error is retained as `metadata.source_error`, with `metadata.scored_on_error=true`. Invalidated samples remain unscored. Raw values remain available in the preserved original sample. Per-model numeric usage fields are summed by field name for resource analysis and preserved per model in source metadata. Attachment references, messages, events, model fallbacks and all other source fields are retained without resolving remote artifacts.

For both importers, `source.json`/`source.jsonl` preserves source bytes; the manifest records source SHA-256, format/version, population scope and original metadata. Output directories must be new or empty. Paired comparison requires identical normalized sample, metric and protocol identities; matching display names alone is insufficient.

Additional importers register `fn(source, output, **params)` in `ai_eval.importers`. Call them from Python with `ai_eval.importers.import_plugin(name, source, output, **params)`. The adapter writes a native run directory; the dispatcher validates it with `load_run` and returns `{"manifest": ..., "results": ...}`. A plugin's return value cannot substitute for valid persisted observations.

## Format references

Implementation was checked against these primary sources. Fixture records in `tests/test_importers.py` are authored schema examples, not measured model runs.

- [Gym episode export](https://github.com/EnvFoundry/Verifiable-Agent-Gym/blob/main/src/agent_gym/export.py), checked blob `85ccc40f9fff6b130fefe89fcde66a2879f49e5b`.
- [Gym rollout export](https://github.com/EnvFoundry/Verifiable-Agent-Gym/blob/main/src/agent_gym/rollout.py), checked blob `721fc62a8dd0e678f544e67092dd4feb9ba962af`.
- [Gym suite/attempt identity](https://github.com/EnvFoundry/Verifiable-Agent-Gym/blob/main/src/agent_gym/suite.py), checked blob `1e901c2ca943664299416de556814f9cef945260`.
- [Inspect log format and conversion](https://inspect.aisi.org.uk/eval-logs.html).
- [Inspect JSON data models](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/log/_log.py), checked blob `c0faa59400cf28113a60dd309b2b9816e411ded6`.
- [Inspect score values and unscored sentinel](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_metric.py), checked blob `9affc618805e29d1ff5747775227d51fe5666ce9`.
- [Inspect model-graded scorer resolution](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_model.py), checked blob `85861220145de4067674237579ab7cefea4376ae`.
