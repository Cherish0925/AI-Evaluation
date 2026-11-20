# Understand interacting changes

A memory mechanism may help with one retriever and interfere with another. Comparing each change against a baseline cannot determine their combined behavior. Run all four combinations on the same task population:

| Cell | Factor A | Factor B |
| --- | --- | --- |
| `00` | Baseline | Baseline |
| `10` | Changed | Baseline |
| `01` | Baseline | Changed |
| `11` | Changed | Changed |

A factor can be a training recipe, model component, prompt, retrieval method, memory mechanism or inference policy. State what level 0 and level 1 mean. Record the actual configurations and hold other relevant conditions fixed. The two changes need not belong to separate software packages.

## Estimate the interaction

For sample $i$, let $y_{ab,i}$ be the mean score over its planned attempts in cell $ab$. AI Evaluation computes these contrasts before averaging across samples:

$$
A_i=\frac{(y_{10,i}-y_{00,i})+(y_{11,i}-y_{01,i})}{2},\qquad
B_i=\frac{(y_{01,i}-y_{00,i})+(y_{11,i}-y_{10,i})}{2},
$$

$$
I_i=y_{11,i}-y_{10,i}-y_{01,i}+y_{00,i}.
$$

The main effects average equally over the two levels of the other factor. The interaction is the change in A's effect when B changes; equivalently, it is the change in B's effect when A changes. The output also includes all four simple effects, such as A with baseline B and A with changed B.

The reported interaction uses the **full difference of differences**. It equals the interaction coefficient for a regression with 0/1-coded factors, twice the interaction effect often reported for a conventional two-level factorial design, and four times the interaction coefficient in a -1/+1-coded regression. This convention is explicit in each contrast's coefficients. [NIST's factorial-design guidance](https://www.itl.nist.gov/div898/handbook/pri/section6/pri615.htm) explains the underlying regression representation; use its coding convention when comparing external results.

For every contrast $c_i$, the population estimate is $\sum_i w_i c_i/\sum_i w_i$. Attempts stay within samples, and related samples stay within their independent group during paired bootstrap resampling. All four cells use the same retained samples and weights. Intervals describe sampling variation across these groups; they do not separately estimate every source of execution randomness.

```python
from ai_eval.factorial import factorial

result = factorial(
    "runs/base", "runs/retrieval", "runs/memory", "runs/both",
    "accuracy", factor_a="retrieval", factor_b="memory", slice_by="domain",
)
print(result["effects"]["interaction"])
```

Positional order is `00, 10, 01, 11`; the first digit is A. Each cell must be a distinct recorded run with matching benchmark, measurement and execution protocol. A model description may be shared across cells when an external experimental condition changes, but the condition and its levels must still be recorded and explained. Assigning four labels to repeated measurements of an unchanged system does not create a factorial intervention.

## A combination can undo both improvements

Suppose an authored example has mean completion scores `0.40, 0.60, 0.55, 0.45` in cell order. Retrieval alone adds 0.20; memory alone adds 0.15. Together they add only 0.05. The interaction is -0.30, and the averaged main effects are 0.05 and 0.00. With the new memory, the new retriever changes completion by -0.10. These numbers suggest investigating their combined behavior: for example, whether memory content displaces useful retrieval context. They do not establish that mechanism without another experiment.

For metrics where lower is better, raw contrasts retain their arithmetic meaning. The `improvement` fields reverse the sign, so a positive value consistently points toward the preferred direction. For interaction, this means a combined result better than the additive expectation; it does not mean either component is universally beneficial.

## Keep the population and interpretation clear

By default, each planned attempt must have a score in all four cells. With `missing="complete_cases"`, a sample is excluded from **every** contrast whenever any cell is incomplete. The result reports each exclusion, per-cell coverage, retained sample weight and slices that have no remaining observations. This estimates effects in the retained population; systematic missingness can change that population substantially.

The intervals are marginal for individual contrasts and slices, not a simultaneous guarantee across all reported comparisons. Choose the primary contrast and meaningful effect size before using the result for a decision. Randomization, order balancing and controlled conditions belong in the experiment itself. Post-hoc contrast arithmetic cannot recover causal identification from confounded runs.
