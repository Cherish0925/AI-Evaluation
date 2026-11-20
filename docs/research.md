# Research methods

An evaluation result is a statement about a system, population, protocol and measurement. A difference matters when those four objects correspond to the research question. This guide connects the implemented analyses to the experimental choices that give their results meaning.

## Define the quantity being estimated

Let sample $i$ have weight $w_i$, trial score $s_{it}$, and $T_i$ scored trials. Over samples with at least one scored attempt, the native summary computes

$$
\bar{s}_i = \frac{1}{T_i}\sum_t s_{it},\qquad
\hat{\mu} = \frac{\sum_i w_i\bar{s}_i}{\sum_i w_i}.
$$

The weighting unit is the sample, after averaging its attempts. More attempts on one case do not make that case more important. Weights describe the intended population or utility; choose them before examining system differences. A macro-average over task families is a different estimand and requires corresponding sample weights.

If attempts fail to produce a score, the displayed mean is conditional on observed attempts. Report scored and planned populations together. Decide whether operational failure is itself the target outcome: if the study defines success as finishing within a deadline, deadline failure can receive a declared task-failure score through the measurement protocol. A broken scorer remains a missing measurement. Do not silently substitute zero for either category inside analysis.

For paired systems, the primary difference is

$$
\hat{\Delta}=\frac{\sum_i w_i(\bar{s}_{i,B}-\bar{s}_{i,A})}{\sum_i w_i}.
$$

The raw difference is always candidate minus baseline; improvement additionally respects the metric's direction. Compare the same benchmark, measurement and protocol. Complete-case analysis changes the analyzed population; it is explicitly requested and reports exclusions. Missingness associated with task difficulty can bias that comparison.

For more than two systems, `study` uses one common cohort rather than a separate intersection for each pair. Every retained sample has all planned trial scores for every system. When a resource is selected, its values must also be present. All system means, baseline differences, slices and resource estimates therefore refer to the same retained cases. `factorial` uses the same principle for its four experimental conditions.

## Match uncertainty to the experiment

Cases from the same document, repository, patient, scene or task generator family can share latent difficulty. Set `group` to the independent sampling unit justified by the study, not merely to a convenient category name. The native percentile bootstrap resamples these groups, keeping their samples and attempts together. Comparisons resample the paired differences jointly. A single group supports a descriptive result but no group-based interval.

This interval describes variation across the sampled groups under that design. It does not include unknown label errors, future distribution shift or the full uncertainty of choosing the benchmark. With few groups or a heavily weighted group, examine sensitivity to group removal and collect more independent cases. Repeated trials characterize execution randomness; they cannot replace independent tasks. A hierarchical model or nested resampling is a separate method when both sources of variation are inferential targets.

Use effect sizes and a meaningful decision threshold. An interval crossing zero is not proof of equivalence. A fixed-sample interval repeatedly inspected until it becomes favorable is not an anytime-valid stopping rule. A predeclared sample size, sequential method or fresh confirmation set is needed for that decision. Test selection must fit both the metric and design; [Dror et al.](https://aclanthology.org/P18-1128/) discuss this dependency for NLP experiments.

For binary attempts, `pass@k` estimates the probability of finding at least one success among $k$ attempts, using $1-\binom{n-c}{k}/\binom{n}{k}$. It assumes the sampled attempts represent the proposed attempt procedure. It does not measure reliable repeated success or whether a system can identify the successful candidate. Evaluate a selection mechanism separately when deployment must choose one answer.

## Separate component effects from interactions

Let $\mu_{ab}$ be mean quality with factors $A=a$ and $B=b$, where each factor has levels zero and one. The implemented two-factor analysis estimates

$$
\Delta_A=\tfrac12[(\mu_{10}-\mu_{00})+(\mu_{11}-\mu_{01})],\qquad
\Delta_B=\tfrac12[(\mu_{01}-\mu_{00})+(\mu_{11}-\mu_{10})],
$$

$$
I=\mu_{11}-\mu_{10}-\mu_{01}+\mu_{00}.
$$

The main effects average over the two levels of the other factor. The interaction $I$ asks how much the effect of $A$ changes when $B$ is enabled; it is symmetric in the two factors. Conditional effects retain the answer at each level. A zero average effect can hide positive and negative conditional effects.

All contrasts are computed within each retained sample and then weighted across samples. The group bootstrap keeps all four conditions together. Its intervals are marginal intervals for individual effects, not a simultaneous claim about every reported contrast.

Use actual interventions in system configuration: a retriever revision, a memory policy, a context budget or a tool-selection mechanism. Hold other components fixed. Randomize or counterbalance execution order when shared service load, changing data or carried-over state can influence outcomes. The four-cell arithmetic alone does not establish that a component caused the difference. See [factorial experiments](factorial-experiments.md) for operation details and interpretations.

## Measure the intended capability

Construct validity asks whether the observable score captures the capability of interest. A test of scientific reasoning that rewards familiar wording may primarily measure recognition. Use contrasts that separate plausible mechanisms: changed surface form with fixed underlying problem, changed causal structure with similar wording, and controlled removal of relevant information. Study scorer behavior as well as system behavior.

Ecological validity asks whether the measured capability transfers to its intended setting. An agent with a pristine workspace, complete instructions and unlimited retries faces a different problem from one operating across partial state and scarce time. Preserve the conditions material to the claim. Controlled tasks isolate mechanisms; representative tasks estimate practical value. Both serve distinct purposes.

Component evaluation and end-to-end evaluation answer different questions. A retrieval improvement may disappear because the answer generator ignores the new evidence. Crossing retriever and generator revisions can reveal this interaction, while end-to-end outcomes establish whether the combined change helps the intended task.

## Study the benchmark as an instrument

`analyze-benchmark` compares cases across a panel of distinct reference systems. For each case it reports the panel mean, range and standard deviation. Binary success scores on a known maximizing `[0, 1]` scale support facility, the panel's observed success fraction. Declared score bounds or a native scorer's known bounds support floor/ceiling indicators. Observing only zeros and ones does not establish the bounds of an arbitrary scorer. A case all panel members solve gives little resolution within that panel, but it may still represent a necessary capability or a useful anchor.

The item–rest statistic asks whether systems doing better on one case also do better on the remaining population. The rest score excludes the case's entire independent group, so paraphrases or tasks from the same source do not create a mechanical association. The statistic is undefined when either score vector is constant, when no independent rest remains, or when fewer than three distinct reference systems are available.

These are descriptions of the chosen panel. Four closely related checkpoints do not represent the diversity of possible architectures and strategies; adding a different system can change which cases discriminate. A negative association invites examination of task meaning, scoring and specialized capabilities. It is not an automatic instruction to remove the case. [Benchmark development](benchmark-development.md) connects these observations to case construction and revision.

## Calibrate judgments

A model judge is a measurement instrument with a model revision, rubric, prompt, decoding configuration and artifact inputs. Establish a human-adjudicated calibration set separate from judge prompt tuning. Measure task-relevant errors, agreement by slice, abstentions and sensitivity to response order, length and presentation. For pairwise judgments, counterbalance candidate order and retain ties. Compare against independently executable outcomes where available.

Human disagreement is data. Preserve individual judgments, rubric ambiguity and adjudication outcomes before selecting an aggregation rule. Consensus can conceal multiple legitimate interpretations. For probabilistic judgments, assess calibration using outcomes and proper scoring rules; agreement alone does not establish correctness. The core's Brier and log-loss scorers evaluate supplied probabilities. They do not supply a calibrated judge or adjudication workflow.

For categorical labels, `calibrate-judge` computes agreement, a confusion matrix, per-class precision and recall, Cohen's kappa, grouped agreement uncertainty and missing-label coverage. The [judge guide](judges.md) defines its inputs and interpretation. Tune the judge on a separate set before measuring agreement on held-out references.

## Evolve benchmarks without losing their meaning

Version task definitions, references, generators, scorers and sampling rules. A reference correction changes the measurement; retain the prior result and rerun the relevant cases. Replacing solved tasks with harder ones changes the population, so a falling score can coexist with an improving system. Use overlapping anchor cases and explicit linking assumptions when comparing revisions; measure whether anchors themselves have become saturated.

Temporal evaluation should distinguish event time, collection time and model or system revision. A rolling benchmark can test current competence, while a frozen cohort supports longitudinal comparison. Report both views when the research question requires them. Adaptive item selection additionally changes which systems see which cases: common-item pairing is no longer automatic, and selection probabilities or an explicit ability model may be necessary. The native fixed-population comparator does not perform that inference.

Repeatedly selecting systems against the same holdout can overfit the evaluation process itself. [Blum and Hardt's Ladder work](https://arxiv.org/abs/1502.04585) formalizes this adaptive leaderboard problem. Track evaluation-driven revisions and use fresh confirmation data for the final generalization claim.

Contamination concerns the relationship between training exposure and measured generalization. Record dataset lineage, release dates and known training or tuning use; examine exact and near-duplicate overlap where source data are available. Disjoint identifiers alone do not establish independence. Compare performance on related but independently constructed cases and temporal cohorts. Unknown training data leave exposure uncertain; a high score or overlap heuristic cannot resolve that uncertainty by itself.

## Evaluate transfer, budgets and research progress

A useful transfer study varies task family, language, modality, environment or time while keeping the hypothesized mechanism explicit. Build a matrix of source conditions and target conditions rather than relying on one aggregate. Use predeclared slices for confirmatory claims; new slices discovered after a run generate hypotheses for the next study. Many slice comparisons require a multiple-comparison or hierarchical analysis when used to make formal claims.

Compare quality as a function of budget: tokens, model calls, tool calls, wall time and monetary or physical resources. State the prices, hardware, concurrency and accounting boundary. A fast cached result and a cold execution are different conditions. The runtime records supplied usage and elapsed time; missing billing, accelerator energy or deployment latency require additional measurement.

`study --resource elapsed_seconds` or `--resource usage.KEY` relates sample-mean quality to sample-mean resource use on the common cohort. A system lies on the observed frontier when no other system is at least as good and no more costly, with one strict advantage. Metric direction determines which quality is better; a resource is minimized. The frontier uses point estimates. It does not establish statistically significant dominance, supply a simultaneous confidence region or predict how one system would behave at an unobserved budget.

Long-horizon scientific evaluation needs outcomes beyond completed subtasks. Define the research problem and starting knowledge, then assess the resulting prediction, proof, model, dataset or experimental construction with an independent method. Separate intermediate milestones from final scientific value. Include cost of failed investigations and human interventions. Reproduction, out-of-distribution prediction or a discriminating physical experiment can test a claimed advance; a persuasive report alone cannot.

For an evolving research agent, freeze one version while comparing the next on a later or independently selected problem set. Evaluate improvement in the agent and improvement in its evaluator separately. A system that optimizes its own scoring rule needs an external measurement to establish progress beyond that rule. These are research protocols to implement with the environment's records and extension points, not capabilities established by the bundled example results.
