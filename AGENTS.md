# AI Evaluation

Build methods and tools for measuring AI systems and making useful comparisons. An evaluation connects a scientific question, a task population, a versioned system, an execution protocol and observed results.

Read `docs/contracts.md` before changing interfaces. Keep task execution, scoring and statistical analysis separate. Use existing domain engines and evaluation frameworks through adapters.

- Preserve sample, independent-group, trial, system and protocol identities. Keep execution failures distinct from scored wrong answers.
- Make the estimand, weighting and missing-data treatment explicit. Resample independent groups, pair the same cases across systems, and keep repeated attempts within their sample.
- Describe supported operations precisely. Broad research scope does not require empty adapters or speculative interfaces.
- Give agents a small discovery surface and task-specific documentation. Share one implementation across Python, CLI and MCP.
- Keep examples executable and data compact. Report authored examples, imported results and model experiments by their actual origin.
- Use branches and pull requests. Preserve unrelated work and update English and Chinese entry pages together.

Run the affected tests and examples. For interface changes, exercise installed CLI/MCP behavior; for release changes, build and relocate the package.
