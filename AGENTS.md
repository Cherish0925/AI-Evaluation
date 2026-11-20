# AI Evaluation

Build a standalone research environment for benchmark design, measurement and experimental comparison. Develop capabilities around concrete research questions and working operations. Read `docs/contracts.md` before changing interfaces.

- Keep target execution, scoring and statistical analysis distinct. Python, CLI, MCP and reports share the same implementation.
- Preserve sample, independent-group, trial, system and protocol identities. Record the whole experimental condition, including model, harness and budget changes.
- State the estimand, weighting and missing-data treatment. Multi-system and factorial analyses use one common cohort; resampling keeps related cases and paired conditions together.
- Distinguish an execution or measurement failure from a scored task failure. Keep coverage visible when incomplete observations change the analyzed population.
- Interpret item statistics relative to their reference panel and resource frontiers as observed tradeoffs. Factorial contrasts describe interventions; causal conclusions depend on the design.
- Keep agent discovery compact and link to the method needed for the current question. Implement a connector or plugin when a real format or execution path needs one.
- Make examples executable and report their actual origin: authored corpus, recorded predictions or a model experiment. Keep English and Chinese entry pages aligned.
- Work on branches, preserve unrelated changes and use pull requests for publication. Optional integrations should preserve this environment's independent operation.

Run the affected examples and tests when behavior changes. Exercise installed CLI/MCP interfaces when changing them, and build and relocate the package when changing distribution behavior.
