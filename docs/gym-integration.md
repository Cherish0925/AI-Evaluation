# Gym, evaluation and data releases

The [integration script](../integration/gym_roundtrip.py) executes actual Verifiable Agent Gym episodes, imports their measured rewards, compares systems, and optionally builds Data Engineering releases from the same trajectories. It makes no model API calls.

It uses six independently seeded, noiseless RC identification tasks with three attempts each. Both scripted policies inspect the instrument and make the same five complex-response measurements. The baseline incorrectly converts its fitted cutoff from Hz to kHz while submitting a value labelled Hz. The candidate submits the correctly fitted value. Gym independently verifies the parameters and held-out frequency responses; Evaluation does not generate or replace these rewards.

The candidate runs twice to exercise reproducibility. There are 54 actual episodes in total: 18 baseline, 18 candidate and 18 repeated candidate episodes. The expected means are 0, 1 and 1, with paired differences 1 and 0. These are authored regression outcomes, not an estimate of general agent ability. Repeated deterministic attempts test identity and interchange, not independent policy sampling.

## Run

Install AI Evaluation from this checkout and the dependency revisions used by the integration:

```bash
python -m pip install -e .
python -m pip install "git+https://github.com/EnvFoundry/Verifiable-Agent-Gym@82b7100d88be2d1158d355802e6a864cde1e5684"
python integration/gym_roundtrip.py --output runs/gym-comparison
```

To retain the same episodes as versioned Data Engineering releases:

```bash
python -m pip install "git+https://github.com/EnvFoundry/Data-Engineering@c6be7171b6adf643ea3418307d7ad4d168acf033"
python integration/gym_roundtrip.py --with-data --output runs/gym-data-comparison
```

Existing source checkouts also work without installing them:

```bash
python integration/gym_roundtrip.py --gym-root ../Verifiable-Agent-Gym --data-root ../Data-Engineering
```

Without `--output`, the script uses a temporary directory and removes it when finished. With `--output`, it requires a new directory and leaves the runs, original exports, imported runs, comparison HTML and `result.json` available for inspection. The result records loaded Python source hashes as well as reference revisions, so a locally modified dependency is distinguishable from the documented source.

## What crosses each boundary

| Boundary | Preserved information |
| --- | --- |
| Gym → Evaluation | Task and suite identities, independent groups, attempts, policy versions, action/time budgets, traces and verifier outcomes |
| Original export → imported run | Byte-identical source copy and SHA-256, with original episode metadata retained per result |
| Baseline → candidate comparison | Matched task population, protocol and budget; system identities differ |
| Gym → Data Engineering | Complete episode payload, source revision, grouping and original `test` split |

The script checks that both policies receive exactly the same recorded measurement results and that the repeat run reproduces them. It checks all 18 scheduled outcomes against the original Gym run before import. This matters because Gym export-v1 omits setup failures, and a standalone export cannot establish coverage of tasks that never reached an episode. Imported coverage remains explicitly scoped to the exported population.

Gym export-v1 does not include episode durations in its top-level rows. Imported elapsed time stays unavailable; the integration does not substitute zero or infer latency from event timestamps. Task and time limits remain part of comparison conditions.

The optional Data Engineering step builds two evaluation-use releases, one per policy. All 18 trajectories in each release remain in the `test` partition. Each release payload is compared with its originating export, including reward and action history. This illustrates reusable observations across execution, evaluation and data tooling without changing their scientific meaning.
