# Design

AI Evaluation studies what a system can do, under which conditions, at what cost, and how strongly the observations support a claim. Its durable product is an interpretable comparison that another researcher can rerun or challenge. A benchmark is one instrument in that process.

The environment spans predictive models, language models, multimodal systems, agents and scientific workflows. It shares data and execution with other environments while owning measurement design, analysis and benchmark evolution. Scope comes from a common experimental structure, rather than making every task look like a chat completion.

## Six concepts

| Concept | Meaning | Representation in the initial implementation |
| --- | --- | --- |
| Benchmark | A versioned population of cases, references, weights and related-case groups | JSON specification and JSONL samples; content-derived identity |
| System | The complete object being compared: model, harness, tools, memory, retrieval and configuration | Versioned system description and identity in the run manifest |
| Protocol | How observations are produced: trials, randomization, budget and stopping conditions | Trial count and seed in the benchmark; execution configuration in the manifest |
| Observation | What actually happened on one sample and trial | Output, status, usage, timing, errors and artifact references in a result row |
| Measurement | A declared interpretation of observations | Named scorers, directions and parameters; per-trial scores distinct from execution |
| Claim | An estimate about a specified population or intervention, with its conditions | Summary or paired comparison, coverage, uncertainty, slices and an accompanying research interpretation |

These concepts do not require six services or a hierarchy of base classes. The local core uses files, ordinary Python functions and installed extension points. [Contracts](contracts.md) define exact fields and behavior. Research questions and claims remain researcher-authored; the runtime does not infer them from a score.

The system boundary matters. Changing a retrieval index, reasoning budget or tool response changes the tested system even if model weights are constant. A fixed benchmark and protocol permit a direct paired comparison. A changed benchmark asks a different question and requires a linking study before scores can be treated as one time series.

## Execution shapes

| Workload | Observation and measurement unit | Execution owner |
| --- | --- | --- |
| Classification, prediction, language tasks | A labeled case; probability, text or structured output | Native target callable, command adapter or an external model runner |
| Retrieval and RAG | Query, retrieved set, answer and source references; component and end-to-end measurements | Retrieval/harness implementation; evaluation supplies fixed queries and scoring |
| Code, browser, tool and embodied agents | An episode with actions, changing state, terminal outcome and resource consumption | Verifiable Agent Gym or another specialized interaction runtime |
| Image, audio, video and spatial systems | Artifact-linked case or a population of generated artifacts | Modality-specific inference and measurement libraries |
| Scientific and long-horizon research | A research problem, investigation trajectory and independently assessed scientific artifact | Research harness, simulator or instrument environment |

An interactive environment supplies transitions and task observations; evaluation determines the experimental population and meaning of outcomes. A transcript alone cannot establish that the final state meets a task objective. Conversely, static question evaluation does not need an environment transition API.

The initial aggregation core supports scalar, per-case measurements. Some multimodal measures compare entire populations or require joint fitting; a dataset-level distance cannot be converted into artificial per-image scores and averaged. Such measurements need a dedicated population-statistic adapter with its own sampling semantics. The current runtime does not implement that adapter. Structured artifact references enable transport, not automatic understanding of every modality.

## Research lifecycle

Start with a decision: for example, whether a new memory mechanism improves completion of previously unseen project tasks within the same budget. Define the target population, independent unit, outcome, relevant effect size and resource limits before running the comparison. Write these beside the benchmark rather than burying them in a report template.

Construct cases from a versioned data release or task generator. Preserve source families, temporal boundaries and slice labels. Pilot the tasks and scorers on a small development set to find ambiguous references and broken measurements. Then freeze the comparison population and protocol.

Execute both systems, retain observations, and analyze paired cases. The result should expose effect magnitude, uncertainty, missing outcomes, meaningful subpopulations and resource tradeoffs. Inspect the cases that change the scientific interpretation; a large transcript dump is not an analysis.

The final step is an experiment that could overturn the apparent improvement: a different task family, later time window, new interaction setting or independently assessed research artifact. A discovered weakness may yield a new benchmark revision or training data. It does not silently rewrite the experiment that produced the discovery.

## Interfaces and ownership

| Boundary | Responsibility |
| --- | --- |
| Data Engineering → Evaluation | Immutable releases, lineage, sampling frames, splits and mixture metadata |
| Gym → Evaluation | Task and environment revisions, episode outcomes, verifier observations and rollouts |
| Harness → Evaluation | Complete system identity, target execution, budgets and trajectory artifacts |
| Evaluation → research or training | Comparable measurements, explicit uncertainty, failure cases and candidate curricula with provenance |
| Evaluation → human or agent | The same analysis through Python, CLI, reports and bounded MCP tools |

IOP and Optics-Photonics-AI-Lab can supply instrument or simulation observations without changing these roles. Evaluation should retain physical units and specimen identities rather than reducing scientific measurements to an arbitrary success label.

Connectors translate real formats. Plugins implement a needed scorer, target or importer. Skills and prompts guide experimental reasoning and tool use. None of these should contain a second, inconsistent implementation of aggregation. Read the short task context first, then the specific protocol, sample or source artifact needed for the next decision.

## Current engineering boundary

The first implementation is a local, resumable execution and analysis core: structured datasets, deterministic native scorers, offline predictions, extensible targets, grouped uncertainty, paired comparisons, run interchange and reports. Durable sample/trial keys allow interruption recovery while preserving recorded failures. Benchmark and system identities describe reproducibility inputs; replay still depends on external services, dependencies and artifacts remaining available.

The current loaders hold sample descriptors and result rows in memory; large modality artifacts should be referenced rather than embedded. Local workers schedule independent target calls. Targets must supply their own thread-safe interaction with model servers and external resources.

Larger deployments can retain the observation contract while replacing scheduling and artifact storage. Distributed scheduling, adaptive test selection, population-level multimodal statistics, judge training and human adjudication workflows are extension work. Categorical judge–reference agreement analysis is already available through `calibrate-judge`; it does not train or modify the judge.
