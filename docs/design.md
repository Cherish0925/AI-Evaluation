# Design

AI Evaluation studies what a system can do, under which conditions, at what cost, and what the observations allow a researcher to conclude. Its working unit is an experiment: a defined population, specified conditions, observations and an analysis that answers a question.

The environment runs independently. A researcher can begin with JSONL cases and recorded predictions, execute a Python function or command, or bring observations from an existing runner. The same structure covers predictive models, language and multimodal systems, retrieval, interactive agents and scientific workflows.

## Experimental objects

| Object | Meaning | Representation |
| --- | --- | --- |
| Benchmark | Cases, references, weights and related-case groups drawn from a defined population | Versioned JSON specification and JSONL samples |
| System | The complete condition being studied: model, prompts, harness, tools, retrieval, memory and configuration | System description and identity in the run manifest |
| Protocol | How observations are collected | Trial count and seed in the benchmark; execution configuration in the manifest |
| Observation | What happened on one sample and trial | Output, status, usage, timing and artifact references in a result row |
| Measurement | A declared interpretation of an observation | Named scorer, direction and parameters; per-trial scores |
| Analysis | Estimates about systems, interventions or benchmark cases | Common cohort, effects, uncertainty, coverage, slices and sample detail |

These are data and functions, not separate services. [Contracts](contracts.md) define their fields. The research question, sampling rationale and interpretation belong beside the experiment; a score alone does not specify a scientific claim.

Changing a retrieval index, reasoning budget or tool response changes the studied system even when model weights remain fixed. Changing the benchmark changes the population or measurement. Direct paired comparisons hold the benchmark and protocol fixed; comparisons across benchmark revisions require a linking design.

## Three directions of analysis

A system comparison holds the cases fixed and varies the system. `study` analyzes all conditions on one common cohort, compares each with a baseline and relates quality to a chosen resource. This keeps a missing difficult case from quietly giving one pair of systems an easier population than another pair.

A component experiment varies specified parts of the system. `factorial` estimates the four cell means, two main effects, conditional effects and interaction of a two-factor design. Run configuration records what each intervention actually changes. Paired observations improve precision; control of other changes determines whether an effect has a causal interpretation.

Benchmark research holds the reference panel in view and examines the cases. `analyze-benchmark` describes item difficulty for that panel, variation across systems, floor and ceiling, and association with performance on other independent groups. These observations guide new case construction. They do not choose which capabilities the benchmark ought to represent.

All three analyses share the sample/trial contract and independent-group resampling where intervals are computed. HTML and MCP consume those same results. [Research methods](research.md) connects the estimates to their assumptions; [workflows](workflows.md) makes the operations concrete.

## Execution and measurement

| Workload | Case and outcome | Target or measurement provider |
| --- | --- | --- |
| Classification, prediction and language tasks | Labeled input; probability, text or structured output | Model callable, command or external runner |
| Retrieval and RAG | Query, retrieved set, answer and source references | Retrieval or generation pipeline; component and end-to-end scorers |
| Code, browser, tool and embodied agents | Episode, changing state, terminal outcome and resource use | Interaction runtime with task-state measurements |
| Image, audio, video and spatial systems | Artifact-linked case or population of generated artifacts | Modality-specific inference and measurement library |
| Scientific and long-horizon research | Research problem, investigation and resulting scientific artifact | Research harness, simulator, instrument or independent assessor |

The target produces an outcome. The scorer gives that outcome a declared interpretation. Analysis then relates scored outcomes across cases and systems. Keeping these steps distinct allows recorded predictions to be rescored and existing measurements to be analyzed without rerunning the target.

An interaction transcript can explain behavior; the terminal state establishes whether an action task was completed. For scientific measurements, preserve physical units, specimen identities and the uncertainty relevant to the quantity. For artifacts too large for a result row, retain references and the metadata needed to interpret them.

The current aggregation core uses scalar per-case measurements. A metric defined on an entire generated population, such as a fitted distribution distance, needs its own population-level estimator and sampling method. Artifact transport and scorer plugins alone do not supply that method.

## Responsibilities at the interfaces

| Participant | Supplies or owns |
| --- | --- |
| Researcher or research agent | Question, target population, interventions, meaningful effect size and interpretation |
| Case source | Task inputs, references, sampling weights, related-case groups and domain metadata |
| Target | Model or harness execution, internal state, external calls and available resource accounting |
| Scorer | Measurement definition, parameters and outcome-to-score mapping |
| Analysis | Comparable cohorts, weighting, effects, coverage, uncertainty and saved result |
| Human or agent interface | Discovery, focused access to records, research guidance and result presentation |

A single Python project can supply all of these. Existing datasets and runners can supply particular parts. Plugins implement a real scorer, target or importer; connectors translate observed formats. Skills and prompts guide experimental choices while invoking the shared implementation.

The agent interface starts with capabilities and method resources. Large results remain in saved analyses and trial records; pagination brings only the relevant cases into the next reasoning step. A human can use the same study through the command line and an HTML report.

## From one experiment to the next

Start with a decision, such as whether a memory mechanism improves unseen project completion within the same tool budget. Specify the population, independent unit, outcome and interventions. Use a small development set to resolve ambiguous tasks and scoring, then fix the population and protocol for the comparison.

Collect observations for every condition. Interpret the effect together with coverage, slices, resource use and examples that distinguish plausible explanations. A gain confined to familiar task families suggests a different next experiment from a gain across later and independently sampled tasks.

Revise a benchmark when its scientific purpose requires new coverage or measurement. Preserve earlier observations as results of their original design. Anchor cases can help relate revisions, provided their role and linking assumptions are explicit.

## Runtime scope

The runtime executes local, resumable studies. It keeps structured case descriptors and result rows in memory, schedules independent target calls with local workers, and writes durable trial records. Targets manage their own model servers, sessions and external resources.

Multi-system comparison, two-factor experiments, benchmark item analysis, categorical judge calibration and portable reports are implemented. Distributed scheduling, adaptive item selection, population-level multimodal estimators and judge training remain distinct extensions. Larger deployments can keep the observation contract while supplying their own scheduling and artifact storage.
