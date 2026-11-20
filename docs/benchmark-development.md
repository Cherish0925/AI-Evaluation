# Developing a benchmark

A benchmark defines a population of tasks, an interaction protocol, and a measurement rule. Its usefulness depends on the claim those measurements support: selecting a system for a workload, measuring a capability, locating a failure mechanism, or studying a training intervention. These purposes may require different samples and different scores.

Start with the intended claim and an explicit coverage map. For a research agent, for example, finding a source, choosing an experiment, interpreting a result, and sustaining a multistep investigation are distinct behaviors. Topic diversity alone does not establish coverage of those behaviors. Record the conditions under which each behavior matters: tools, available evidence, time or token budget, language, modality, and required precision. The evaluation protocol makes those conditions executable.

## Use a reference panel to understand items

`analyze_benchmark` describes a benchmark using runs from distinct systems on the same samples, protocol, and metric. It accepts native runs and imported runs that retain those contracts.

```python
import json
from ai_eval.benchmarks import analyze_benchmark

result = analyze_benchmark(
    ["runs/reference-a", "runs/reference-b", "runs/reference-c"],
    "accuracy",
    slice_by="capability",
)
with open("benchmark-analysis.json", "w", encoding="utf-8") as destination:
    json.dump(result, destination, indent=2, ensure_ascii=False, allow_nan=False)
```

The panel should span capabilities relevant to the intended use: simple baselines, plausible deployment candidates, stronger reference systems, and meaningful changes in architecture or harness. Several revisions of one model provide useful development evidence, but make the description conditional on that family. Merely increasing the number of similar systems does not make the panel representative. Each distinct system receives equal weight; sample weights retain their declared meaning. Repeated attempts belong in trials within a run, not as additional panel members. Duplicate system identities are rejected.

Two systems support item comparisons. Corrected discrimination requires at least three systems and variation in both compared vectors. Three systems are a computational minimum, not a sufficient basis for a stable scientific conclusion. This command describes the chosen panel; it does not estimate a population of future systems or fit an item response model.

## What the quantities mean

Let \(y_{ist}\) be the score on item \(i\), system \(s\), and trial \(t\). First average trials within each item and system:

\[
x_{is}=\frac{1}{T}\sum_{t=1}^{T}y_{ist}.
\]

The item panel mean is \(S^{-1}\sum_s x_{is}\). Range and population standard deviation describe how these system means differ. They do not describe trial variability. When all retained raw trial scores are binary, the supported range is [0, 1], and one is the favorable outcome, the mean is reported as `facility`: the observed success fraction under this panel and protocol. It is not an intrinsic or universal difficulty parameter. General normalized scores do not receive this success-fraction interpretation.

For corrected discrimination, orient scores so larger is better using \(q=1\) for maximize and \(q=-1\) for minimize. If \(g(i)\) is an item's independent group and \(w_j\) its declared weight, the rest score is

\[
r_{s,-g(i)}=
\frac{\sum_{j:g(j)\ne g(i)}w_j qx_{js}}
     {\sum_{j:g(j)\ne g(i)}w_j}.
\]

The item statistic is the Pearson correlation across systems,

\[
d_i=\operatorname{corr}_s(qx_{is},r_{s,-g(i)}).
\]

Every item sharing the focal item's group is excluded from its rest score. For example, variants of the same problem or questions based on one document should not contribute to the focal item's comparator. Group labels express dependence; they do not automatically give each group equal weight. The implementation uses weighted group aggregates without subtracting a potentially dominant group from a rounded total.

A positive value indicates alignment with the rest of this benchmark in this panel. A negative value can reflect a scoring error, an alternative capability, a tradeoff, or a narrow panel whose systems improve in different ways. It is a reason to investigate the construct and examples, not a rule for deleting an item. An undefined value has an explicit reason: too few systems, constant item scores, constant rest scores, or no other independent group. Pearson correlation assumes the numeric scale supports this comparison; a nonlinear score transformation can change it.

`floor` and `ceiling` mean all panel systems reached the worst or best boundary, respectively. Bounds can be declared in the metric descriptor:

```json
{"name": "quality", "scorer": "research_scorers:quality", "direction": "maximize", "bounds": [0, 5]}
```

Without declared bounds, the native scorer contracts for `exact_match`, `numeric`, `set_f1`, `token_f1`, `recall_at_k`, and `ndcg_at_k` establish [0, 1], recorded as `native_scorer_bounds`. No bounds are inferred for other scorers, including imported scores, plugins, `brier`, and `log_loss`. Observing only zeros and ones does not establish metric support: an unbounded latency metric whose observed values happen to equal one is not saturated. `binary_scores` describes the observed values separately from bounds. Fractional means from binary trials remain binary-outcome evidence. For a minimized metric, the lower bound is the performance ceiling. Binary observations on a wider declared scale do not receive a success-fraction interpretation.

## Preserve the population being described

Every retained item must have all planned scores for every panel system. By default an incomplete panel raises an error. `missing="complete_cases"` explicitly selects the common fully scored cohort. The result retains every excluded item, the per-system planned/observed/scored counts, original coverage, retained weight fraction, and wholly excluded groups and slices. A reported score is never silently replaced with zero.

Complete-case selection can remove the exact tasks that one system struggles to execute. Interpret the reduced cohort as such; recover the missing observations when the intended population matters. The command does not infer how unobserved tasks would score. Population metadata from imported runs remains attached to each panel member.

Group and slice summaries aggregate the item descriptions using declared sample weights. Discrimination is always computed against the full retained benchmark excluding the item's group; slicing summarizes those values, rather than silently redefining the comparator. Counts such as `constant_samples` and `negative_discrimination_samples` are unweighted counts. Keep item profiles alongside summaries so that averaging does not hide coverage gaps.

## Make benchmark changes answer a research question

| Observation | Useful next step |
| --- | --- |
| A relevant capability has little coverage | Add tasks that exercise its mechanism and realistic boundary conditions; increasing the number of similar prompts does not fill the gap. |
| Strong reference systems reach the ceiling | Add genuinely demanding conditions or longer dependencies where they matter. Retain easy items when they represent an important workload or baseline competence. |
| Every system reaches the floor | Examine whether the task is feasible under the declared resources and whether scores expose partial progress. A hard item can remain valuable when its behavior is understood. |
| Discrimination is negative | Inspect scoring and solutions, then compare a broader or deliberately different panel. Preserve valid capability tradeoffs instead of optimizing every item for one overall ranking. |
| Scores separate systems only within one slice | Check whether that slice's weight matches the intended population; report the distinction rather than treating the overall mean as universal. |
| Item changes improve apparent separation | Measure the change on systems and tasks held out from item selection before claiming a better benchmark. |

Keep development examples available for prompt, scorer, and task iteration. Reserve separate tasks or task families for assessing the resulting benchmark, and where feasible reserve systems that did not influence item selection. Choosing items to maximize discrimination on one panel can make the benchmark encode that panel's ranking. Shared problem generators, source documents, or templates belong together when separating development and held-out material.

Version changes to task distributions, interaction budgets, scoring rules, or weights. When continuity matters, run a reference panel on both versions and report changes in coverage and item behavior. Treat changed conditions as a changed measurement. Refreshing a saturated benchmark should expand useful measurement range while preserving the claims users still need; novelty and difficulty alone do not establish value.

## Research context

These design choices are consistent with published evidence that data composition and dependence affect evaluation conclusions. They are not a reproduction of the following papers' methods:

- Kovatchev and Lease (2024), [Benchmark Transparency: Measuring the Impact of Data on Evaluation](https://aclanthology.org/2024.naacl-long.86/), study how data properties and stratified sampling affect absolute and relative performance across SQuAD and MNLI evaluations. This motivates explicit coverage and population descriptions.
- Siska et al. (2024), [Examining the robustness of LLM evaluation to the distributional assumptions of benchmarks](https://aclanthology.org/2024.acl-long.560/), examine correlated prompt performance and target-distribution assumptions that can change rankings. This motivates keeping task dependence and use-case distribution visible.

For system comparisons and uncertainty on benchmark scores, use [the comparison workflow](workflows.md#compare-systems-and-resource-use). Item-panel descriptions complement that analysis; they do not replace a study of the scientific claim.
