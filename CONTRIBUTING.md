# Contributing

A useful contribution improves an experiment: a measurement that captures the intended capability, an adapter that preserves existing observations, an executable workload, or an analysis that changes how a result is understood.

Read [the design](docs/design.md) and [contracts](docs/contracts.md). Open a branch and pull request. Explain the concrete scientific or engineering problem, resulting behavior, and the example or test that establishes it. Keep the English and Chinese READMEs aligned when public capabilities change.

For an importer, retain source format and version, sample and trial identity, scorer definition, unavailable outcomes and original source bytes. Test against a documented upstream representation. Aggregated leaderboard values cannot be expanded into invented per-sample observations.

For a scorer, specify its domain, units, orientation, undefined cases and parameters. A score should distinguish the behavior it claims to measure. For a statistical method, specify the estimand and independent sampling unit, then test cases that would reveal a wrong weighting, pairing or missingness assumption.

For a target, declare its configuration and revision, preserve errors as outcomes, and support explicit resource or timeout settings where the upstream interface offers them. Reuse a serving engine or domain framework instead of duplicating its runtime.

Run affected tests and examples. Changes to packaging, CLI or MCP also need an installed invocation from another directory. Keep source/plugin entry points relocatable. Avoid empty adapter classes, unimplemented commands, benchmark-name catalogs without operational value, and claims unsupported by the accompanying study.
