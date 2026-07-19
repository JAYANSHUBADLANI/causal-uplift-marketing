# Part 2, observational methods vs RCT truth

Confounding was manufactured by keeping treated units with probability `e*(x)` and controls with probability `1 - e*(x)`, where

```
e*(x) = clip(sigmoid(-0.20 + 0.90*z_log_history - 0.70*z_recency + 0.50*multichannel), 0.03, 0.97)
```

Sample: 21,313 customers (10,018 treated). Naive bias vs RCT truth: +0.0308 (+68.2% of the true effect).

## Method comparison

| method | estimand | estimate | ci_lo | ci_hi | abs_error_vs_rct |
|---|---|---|---|---|---|
| RCT benchmark | ATE | 0.0452 | 0.0390 | 0.0516 | 0.0000 |
| naive difference in means | ATE | 0.0761 | 0.0671 | 0.0851 | 0.0308 |
| regression adjustment (OLS, HC1) | ATE | 0.0445 | 0.0342 | 0.0548 | 0.0007 |
| IPW (stabilised, trimmed) | ATE | 0.0463 | 0.0370 | 0.0577 | 0.0011 |
| 1:1 NN matching (ATT) | ATT | 0.0554 | 0.0459 | 0.0649 | 0.0101 |
| AIPW (doubly robust, cross-fit) | ATE | 0.0464 | 0.0350 | 0.0579 | 0.0012 |


*Matching: 10,007 pairs (99.9% of treated matched within caliper 0.2 SD of the logit propensity; with replacement; paired SE is approximate. Matching estimates the ATT, which is why it is footnoted rather than compared 1:1 with the ATE rows. IPW propensities trimmed to [0.02, 0.98]; bootstrap refits the propensity model in each of 300 replicates. AIPW uses 2-fold cross-fitting with influence-function SEs.*


## Balance diagnostics

max |SMD| unadjusted = 0.976 -> IPW-weighted = 0.018 -> matched = 0.042 (threshold 0.1).
