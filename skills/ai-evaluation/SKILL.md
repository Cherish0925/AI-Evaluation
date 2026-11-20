---
name: ai-evaluation
description: Design benchmarks, execute AI evaluations, compare matched runs, calibrate judges, and investigate sample-level outcomes. Use when evaluating models, retrieval, agents, harnesses, scientific predictions, or benchmark revisions in the AI Evaluation environment.
---

# AI Evaluation

Start with the scientific question and the decision the result should inform. Identify the system being changed, the target population, the observable outcome and the independent unit. Include model, harness, prompts, retrieval, tools and inference configuration when they affect the comparison.

Run `ai-eval capabilities` to discover implemented operations and installed extensions. If working from a source checkout, use `python server.py capabilities`. Use the CLI or MCP operations for execution and analysis; both call the same core.

Read only the relevant guide:

| Task | Guide |
| --- | --- |
| Design a benchmark, choose population or uncertainty method | `docs/research.md` |
| First evaluation, prediction replay, command target or comparison | `docs/workflows.md` |
| Extend schema, scoring or execution | `docs/contracts.md`, then `docs/scoring.md` |
| Connect a model, harness, plugin or MCP host | `docs/connect.md` |
| Use Gym outcomes or an existing framework | `docs/gym-integration.md` or `docs/ecosystem.md` |
| Assess automatic judgments against reference labels | `docs/judges.md` |

For a comparison, name the baseline and candidate and state which conditions change. Use matching benchmark, scoring and task identities. Keep repeated trials inside their sample and related samples inside their independent group. State whether an estimate describes the full planned experiment or completed observations.

Inspect a bounded page of samples when a result needs explanation. Use slices to test a specific mechanism or population difference. Preserve wrong answers, execution failures and missing outcomes as distinct observations. A judge-reference disagreement calls for understanding the rubric and reference quality; it does not establish which party is correct.

Return the result that answers the question: effect size and direction, uncertainty when estimable, coverage, relevant resource observations, concrete examples and artifact paths. Distinguish measured behavior from a hypothesis about its cause. Use the original engine for domain-specific execution; extend this environment when a new measurement or comparison is required.
