"""Short, installed research guides shared by the CLI and MCP resources."""

from __future__ import annotations


GUIDES = {
    "experimental-design": (
        "Design an evaluation",
        """State the research question as a comparison on a specified population.
Choose an observable outcome that addresses that question, its direction, and the
independent unit: for example a source document, task family or physical specimen.
Represent related cases by one group and repeated attempts by trials within a case.
Keep the same task, scoring and execution protocol across conditions; put model,
harness, retrieval, tools and sampling settings in each system configuration.

Use eval_benchmark to inspect an existing specification and eval_run to execute a
prediction file, command or installed target. The target receives input and seed;
reference answers stay with the scorer. Record resource measurements with units.
One run supports a descriptive summary. Multiple runs support paired comparisons.
A factorial design supports effects of explicitly crossed component changes.

Choose the primary outcome, intended slices and sample population before looking
at results. Repeated attempts do not add independent tasks. Group intervals reflect
variation across the recorded groups; they do not establish generalization to a
different population. Incomplete observations remain visible in coverage. Decide
whether a completed-case estimate answers the original question before using it.

Deliver an estimate that answers the question, its uncertainty, relevant resource
use and a few informative observations. Separate the measured result from a causal
explanation and identify the experiment that would distinguish that explanation.""",
    ),
    "system-comparison": (
        "Compare systems and resource use",
        """Use eval_study (CLI: study) for two or more runs on one benchmark, metric
and protocol. baseline is a zero-based index in the ordered run list. All systems
use one common complete sample cohort. With missing='complete_cases', exclusions
are shared by every comparison; report the changed population and retained weight.

Trial scores are averaged within each sample and then combined by sample weights.
Paired group resampling keeps all systems together. delta is system minus baseline;
improvement reverses the sign for a minimizing metric. Intervals are pointwise,
not simultaneous guarantees over all comparisons or a rule for selecting a winner.

Set resource='elapsed_seconds' or 'usage.FIELD' to compare quality against measured
resource per attempt on that same cohort. Name the unit and scope: elapsed_seconds
includes target invocation; a reported token count is not total experiment cost.
Missing resource measurements are missing observations, never zero-cost attempts.
The Pareto frontier uses observed means, retaining ties; it does not establish
statistical dominance. A frontier trades off the chosen metric and resource only.

Save full JSON with output, request relevant rows with eval_analysis, and produce
an offline report with eval_report(analysis=..., output=...). Use eval_samples for
the original attempts. Compare held-out outcomes when adapting systems using these
results; selecting the best configuration and estimating its advantage are distinct
statistical questions.""",
    ),
    "factorial-experiments": (
        "Measure component effects and interactions",
        """Use eval_factorial (CLI: factorial) with four runs in 00, 10, 01, 11
order. Name factors A and B and describe levels 0/1 in system configurations. Hold
the benchmark, scoring, trial policy and other system conditions fixed. All four
runs use one common complete cohort with paired independent-group resampling.

For condition means y00, y10, y01, y11:
  A = ((y10 - y00) + (y11 - y01)) / 2
  B = ((y01 - y00) + (y11 - y10)) / 2
  interaction = y11 - y10 - y01 + y00
The interaction is the full difference of differences. Report simple effects too:
A at B=0 is y10-y00; A at B=1 is y11-y01. A small average effect can conceal a
large effect in one condition. Interpret raw estimates on the metric's scale;
direction-adjusted improvement accounts for minimizing metrics.

This analysis estimates the supplied contrasts. A causal account additionally
requires the changed conditions to implement the intended factors without other
confounded changes. Pairing tasks alone does not remove time/order or model-version
confounding. Counterbalance execution when those effects matter.

Use output to save the complete analysis, eval_analysis for samples or conditions,
and eval_report for effects and interaction plots. Read simple effects alongside
the interaction before claiming independent or synergistic component benefits.""",
    ),
    "benchmark-development": (
        "Study the benchmark itself",
        """Use eval_analyze_benchmark (CLI: analyze-benchmark) with runs from at
least two distinct systems and one shared benchmark, metric and protocol. Select a
reference panel relevant to the intended use. Statistics describe this panel; they
are not an intrinsic or model-independent measure of benchmark quality.

Inspect per-item means and ranges to locate saturation and variation. A declared
metric bounds=[lower, upper] defines floor/ceiling endpoints. Native accuracy, F1,
recall and nDCG supply their known [0,1] bounds. Observed zeros and ones alone do
not establish an unknown metric's support. Facility is the panel success fraction
for binary outcomes on a maximizing [0,1] scale, not normalized arbitrary scores.

With at least three systems, discrimination correlates an item's system scores with
system performance on the remaining independent groups. The item's entire group
is excluded from that remainder. Constant scores, no remaining groups and too few
systems have no defined correlation. Correlation reflects panel composition and
can be unstable in a small panel; no latent ability model is fitted.

Use groups and planned slices to understand which populations a benchmark covers
and which it fails to distinguish. Read relevant original records with eval_samples.
An easy, constant or negatively correlated item can still measure an important
capability; investigate its content before revising it. Do not optimize a benchmark
solely to separate the panel used to develop it. Evaluate revisions on additional
systems or held-out groups, preserve revisions, and describe the population change.

Save full output and use eval_analysis(section='samples', search=...) for bounded
item profiles; eval_report renders an offline development report.""",
    ),
}


def get_guide(topic: str = "experimental-design") -> dict:
    """Return a task-specific guide available without a source checkout."""
    if topic not in GUIDES:
        raise ValueError(f"unknown guide; choose one of: {', '.join(GUIDES)}")
    title, content = GUIDES[topic]
    return {"topic": topic, "title": title, "content": content}
