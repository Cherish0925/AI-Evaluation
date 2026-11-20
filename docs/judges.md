# Judge calibration

A judge is a measurement instrument: its labels need an empirical relationship to the distinctions the study intends to measure. Compare its decisions with explicit reference annotations on cases representative of the intended use. References can be adjudicated human judgments or another declared labeling procedure; agreement does not make either source objective truth.

`calibrate` analyzes nominal categories already collected. It computes exact agreement, a confusion matrix, per-class precision/recall/F1, Cohen's kappa and an independent-group bootstrap interval for agreement. It does not call a model or fit a judge. Probability calibration, ordinal distance and multiple-rater reliability require other measurements.

## Data and execution

Use one UTF-8 JSONL row per case:

```json
{"id":"case-1","group":"source-document-1","reference":"pass","judge":"fail","slices":{"domain":"science"},"metadata":{"rubric_revision":"3","judge_revision":"snapshot-id","reference_method":"two annotators, adjudicated"}}
```

Labels are nonempty strings. Missing or null `reference` and `judge` values mean unavailable observations. A genuine category such as `"abstain"` is a string; it participates in the confusion matrix. `group` identifies related cases that must remain together during resampling. It defaults to `id`. An alternative group field is supported through `group_key`. IDs must be unique; repeats of a case must first be reduced according to the study's declared annotation protocol. Cases have equal weight.

```bash
ai-eval calibrate-judge examples/judgments.jsonl --labels '["pass","fail"]' --resamples 2000 --seed 17 --json-output runs/judge-calibration.json
```

```python
from ai_eval.judgments import calibrate

result = calibrate("examples/judgments.jsonl", labels=["pass", "fail"], seed=17)
print(result["agreement"], result["kappa"], result["coverage"])
```

The bundled file is authored demonstration data, not an experiment with human annotators or a model. It contains eight cases, six paired labels and four agreements: agreement is 2/3, kappa is 1/3 and paired coverage is 3/4. Three source groups contribute to the agreement interval; the fourth contains only unpaired observations.

## Read the measurements together

Confusion-matrix rows are reference labels; columns are judge labels. Explicit `labels` fixes their order and rejects unexpected categories. Without it, the category list is the sorted union of every observed reference or judge label, including labels from unpaired cases. Unobserved class precision or recall is null when its denominator is zero.

Agreement is the fraction of paired labels that match. Kappa compares that agreement with the agreement implied by the two paired marginal label distributions:

$$\kappa = \frac{p_o-p_e}{1-p_e}.$$

Kappa is null when there are no paired labels or when the paired marginals make its denominator zero. The group bootstrap interval applies to agreement only. With fewer than two contributing independent groups, the interval is null. Cases remain equally weighted even when group sizes differ; entire groups are resampled. The result retains the bootstrap seed, source-file digest and each case's metadata.

Coverage counts missing reference labels, missing judge labels and missing-both separately. Agreement and kappa are conditional on paired observations. If difficult cases disproportionately lack labels, their absence can change the apparent judge quality; a high conditional agreement does not resolve that missingness. Inspect disagreement cases and class-specific confusion before replacing a reference-labeling process with the judge.

## Design a useful calibration study

Keep the judge revision, prompt, rubric and reference-labeling procedure fixed within one study. Separate rubric development cases from the cases used to estimate performance. Sample the kinds of outputs the judge will actually score, including near-boundary answers, unsupported but fluent answers, alternative valid solutions and partial completions.

For pairwise preference judging, normalize the displayed winner back to stable candidate identities before recording labels. Evaluate reversed presentation orders as a separately declared paired study, retaining a common case group; otherwise position preference can be mistaken for system preference. For open-ended scientific work, separate dimensions such as correctness, novelty, reproducibility and usefulness instead of collapsing them into an unexplained overall label.

Human disagreement can reveal an ambiguous rubric or a genuinely plural judgment. Preserve individual annotations upstream, state the adjudication rule, and inspect where judge/reference disagreement changes a downstream system ranking. The appropriate goal is a measurement useful for that decision, with an explicit scope and reproducible labeling conditions.
