# Ecosystem

Use existing engines where they already own a difficult execution problem. AI Evaluation provides a stable study and analysis layer around observations that retain their original meaning. This map selects complementary projects rather than treating a long benchmark catalog as an architecture.

Primary sources checked on 2026-11-20. Framework APIs and formats should be pinned for each actual integration.

| Project or source | Capability to learn from or reuse | Relationship to this repository |
| --- | --- | --- |
| [Inspect AI](https://inspect.aisi.org.uk/eval-logs.html) | Evaluation logs preserve sample-level observations and scores; tools support log conversion and targeted sample reads. | The Inspect JSON v2 importer selects an explicit scorer and optional value. Other Inspect execution capabilities remain upstream. |
| [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) | Established language-model task execution, model interfaces and optional sample logging. | Reuse its runner where appropriate. A format-specific importer is not included; preserve task settings and sample identities when building one. |
| [Lighteval](https://github.com/huggingface/lighteval) | Multiple inference backends, custom tasks and metrics, and detailed per-sample results. | Another execution backend candidate. It is not a mandatory dependency or a currently implemented native importer. |
| [HELM](https://github.com/stanford-crfm/helm) and [HEIM](https://github.com/stanford-crfm/helm/blob/main/docs/heim.md) | Broad scenario coverage and multiple measurements; HEIM extends evaluation to several aspects of text-to-image generation. | Methodological references for coverage and multimodal measurement. HELM's repository states that it entered maintenance mode on June 1, 2026; use pinned artifacts when reusing it. |
| [EvalPlus](https://evalplus.github.io/) | Stronger executable tests for generated code and separate evaluation of execution efficiency. | Reuse specialized code measurement instead of treating textual similarity as correctness. Native execution/import integration is not included. |
| [BrowserGym](https://github.com/ServiceNow/BrowserGym) | Browser tasks exposed through an interaction environment, with specialized benchmark setup and agent experimentation. | An interaction backend, parallel in role to Verifiable Agent Gym. It should emit episodes for analysis rather than dictate the statistical population. No native BrowserGym importer is included. |
| [Anthropic: agent evaluation](https://www.anthropic.com/research/demystifying-evals-for-ai-agents) | Separates tasks, repeated trials, graders, traces and outcomes; emphasizes evaluation of the complete agent system. | A design reference. Model and harness belong in system identity; transcripts and final-world outcomes remain distinct observations. |

The statistical design is informed by the experimental question rather than any framework default; see [research methods](research.md) for estimands, paired groups, missing outcomes and benchmark evolution.

## EnvFoundry interfaces

| Environment | What evaluation consumes | What evaluation contributes |
| --- | --- | --- |
| [Data Engineering](https://github.com/EnvFoundry/Data-Engineering) | Versioned datasets, lineage, sampling and split definitions | Measurement requirements, failure cohorts and data-quality findings |
| [Verifiable Agent Gym](https://github.com/EnvFoundry/Verifiable-Agent-Gym) | Tasks, environment revisions, rollout export-v1 observations and verifier results | Comparative analysis across systems and research protocols for task populations |
| IOP | Instrument outputs, experimental conditions and specimen identities | Measurement comparisons with physical units and uncertainty appropriate to the experiment |
| Optics-Photonics-AI-Lab | Simulation outputs, model assumptions and parameter studies | Held-out prediction tests, cross-solver comparisons and evaluation of scientific claims |

These roles are composable. A research agent can use Data Engineering to construct a dataset, Gym to execute an interactive investigation and AI Evaluation to compare the resulting scientific outputs. None of the three needs to own the others' runtime.

## A useful adapter preserves meaning

An adapter must retain the task population and revision, system configuration, sample and trial identities, scorer definition, unavailable outcomes and source provenance. Paired analysis additionally needs the same population and measurement contract on both sides. A table containing only an aggregate score is useful as a published result, but cannot supply invented samples for uncertainty or pairing.

Use native prediction JSONL when an external runner can export unscored structured predictions. Use an importer when retaining already-scored observations and their execution history. Use a target extension when this runtime should schedule execution, and a scorer extension when a new measurement is needed. The choice follows ownership of execution and measurement, not the provider's brand.

OpenAI, Google, Anthropic, DeepSeek and locally served models fit these boundaries as concrete system configurations. Provider-specific integrations belong in target plugins, with each installed adapter declaring its capabilities and revision.
