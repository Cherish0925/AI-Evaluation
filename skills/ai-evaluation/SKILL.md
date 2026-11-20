---
name: ai-evaluation
description: Design benchmarks, compare AI systems and resource use, measure component interactions, study item behavior, and calibrate judges. Use for evaluation research on models, retrieval, agents, harnesses and scientific predictions.
---

# AI Evaluation

Start with the scientific question and the decision the result should inform. Identify the system being changed, the target population, the observable outcome and the independent unit. Include model, harness, prompts, retrieval, tools and inference configuration when they affect the comparison.

Run `ai-eval capabilities` to discover implemented operations and installed extensions. If working from a source checkout, use `python server.py capabilities`. Use the CLI or MCP operations for execution and analysis; both call the same core.

Installed guidance is available through `ai-eval guide TOPIC` or MCP resources under `evaluation://guides/TOPIC`. Choose `experimental-design`, `system-comparison`, `factorial-experiments` or `benchmark-development`. For a source checkout, read only the relevant document:

| Task | Guide |
| --- | --- |
| Design a benchmark, choose population or uncertainty method | `docs/research.md` |
| First evaluation, prediction replay, command target or comparison | `docs/workflows.md` |
| Cross two component changes and explain their interaction | `docs/factorial-experiments.md` |
| Study item variation, saturation and discrimination across systems | `docs/benchmark-development.md` |
| Extend schema, scoring or execution | `docs/contracts.md`, then `docs/scoring.md` |
| Connect a model, harness, plugin or MCP host | `docs/connect.md` |
| Use Gym outcomes or an existing framework | `docs/gym-integration.md` or `docs/ecosystem.md` |
| Assess automatic judgments against reference labels | `docs/judges.md` |

For a comparison, name the baseline and candidate and state which conditions change. Use matching benchmark, scoring and task identities. Keep repeated trials inside their sample and related samples inside their independent group. State whether an estimate describes the full planned experiment or completed observations.

Choose the analysis by the question. Use `study` / `eval_study` for multiple systems on one common cohort, optionally adding measured resource use. Use `factorial` / `eval_factorial` for two crossed factors in `00, 10, 01, 11` order; read simple effects alongside the interaction. Use `analyze-benchmark` / `eval_analyze_benchmark` for a reference panel's item behavior. Pointwise intervals, observed Pareto frontiers and panel-dependent correlations answer different questions; explain the quantity actually estimated.

Save the full analysis. Use `analysis --section summary` or `eval_analysis(section="summary")` for its overview, then selected sections and samples when a result needs explanation. Use slices to investigate a specific mechanism or population difference. Preserve wrong answers, execution failures and missing outcomes as distinct observations. A judge-reference disagreement calls for understanding the rubric and reference quality; it does not establish which party is correct.

Return the result that answers the question: effect size and direction, uncertainty when estimable, coverage, relevant resource observations, concrete examples and artifact paths. Distinguish measured behavior from a hypothesis about its cause. Use the original engine for domain-specific execution; extend this environment when a new measurement or comparison is required.
