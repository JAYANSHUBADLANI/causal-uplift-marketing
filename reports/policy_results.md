# Part 4, targeting policy and profit simulation

Assumptions: margin = **$25.00** per incremental conversion, cost = **$0.10** per e-mail (defaults in `src/config.py`; sweep them live in the Streamlit app). Incremental conversion rates are estimated from the randomised holdout inside each targeted group, so the model only supplies the *ranking*.

## Policy comparison

| policy | n_emailed | profit |
|---|---|---|
| e-mail no one | 0 | 0.00 |
| blanket e-mail (k=100%) | 42693 | -948.79 |
| random 35% | 14943 | -332.08 |
| top 35% by uplift (optimal) | 14943 | 1,065.40 |


Profit at k* bootstrap 95% CI: [$-36, $2,238] (500 resamples).


Targeting the top 35% captures **$1,065** vs **$-949** for blanket e-mailing, `-212%` profit while contacting `35%` of customers.
