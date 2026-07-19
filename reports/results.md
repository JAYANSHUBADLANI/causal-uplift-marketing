# Consolidated results

All numbers below were produced by running the pipeline end-to-end on the
real Hillstrom dataset (64,000 customers). Regenerate with the five
`python -m src.*` commands in the README; per-part detail lives in
`rct_ate.md`, `obs_methods.md`, `uplift_results.md`, `policy_results.md`.

## Part 1, the randomised experiment

Randomisation checks: SRM chi-square p = 0.904 (arm sizes 21,306 / 21,307 /
21,387); max covariate |SMD| = 0.014. Clean.

| treatment | outcome | control mean | treated mean | ATE | 95% CI (bootstrap) | p (Holm) |
|---|---|---|---|---|---|---|
| mens | visit | 0.1062 | 0.1828 | **+0.0766** | [0.0701, 0.0834] | 8.2e-112 |
| mens | conversion | 0.0057 | 0.0125 | +0.0068 | [0.0050, 0.0086] | 6.0e-13 |
| mens | spend | 0.653 | 1.423 | +$0.770 | [0.484, 1.052] | 3.5e-07 |
| womens | visit | 0.1062 | 0.1514 | **+0.0452** | [0.0390, 0.0516] | 1.2e-43 |
| womens | conversion | 0.0057 | 0.0088 | +0.0031 | [0.0014, 0.0048] | 3.1e-04 |
| womens | spend | 0.653 | 1.077 | +$0.424 | [0.152, 0.686] | 1.1e-03 |

All six effects survive Holm correction. The Women's-e-mail effect on visit
spans roughly 3.6-6.6pp across one-dimensional segments (history, channel),
heterogeneity that a multivariate model widens further (Part 3's top decile:
+8.5pp). See `figures/fig_rct_segments.png`.

## Part 2, observational methods vs the RCT ground truth (centerpiece)

Confounded sample built by biased subsampling (n = 21,313; engaged
customers over-represented among the treated). Target: the womens→visit
ATE, known from Part 1 to be **+0.0452 [0.0390, 0.0516]**.

| method | estimand | estimate | 95% CI | abs. error vs RCT |
|---|---|---|---|---|
| RCT benchmark | ATE | 0.0452 | [0.0390, 0.0516] | - |
| naive difference in means | ATE | 0.0761 | [0.0671, 0.0851] | **0.0308** |
| regression adjustment (OLS, HC1) | ATE | 0.0445 | [0.0342, 0.0548] | 0.0007 |
| IPW (stabilised, trimmed) | ATE | 0.0463 | [0.0370, 0.0577] | 0.0011 |
| 1:1 NN matching | ATT | 0.0554 | [0.0459, 0.0649] | 0.0101* |
| AIPW (doubly robust, cross-fit) | ATE | 0.0464 | [0.0350, 0.0579] | 0.0012 |

The naive estimate overstates the true effect by 68%; every adjustment
method recovers it to within ~0.1pp. *Matching estimates the ATT, the
effect among the (selected, engaged) treated, which is genuinely larger
under the Part 1 heterogeneity, so its gap is an estimand difference, not
an estimator failure. Balance: max |SMD| 0.98 unadjusted → 0.02 after IPW
weighting (0.04 matched), see `figures/fig_obs_love.png`.

## Part 3, uplift meta-learners (womens vs control, outcome: visit)

70/30 stratified split; Qini implemented from scratch; 5-fold OOF within
train as an overfitting check.

| learner | Qini (test) | Qini (OOF train) | uplift@10% | uplift@20% | uplift@30% |
|---|---|---|---|---|---|
| **S-learner** | **54.8** | 84.0 | +0.0849 | +0.0796 | +0.0752 |
| X-learner | 31.2 | 65.4 | +0.0558 | +0.0627 | +0.0718 |
| T-learner | 23.7 | 73.6 | +0.0440 | +0.0569 | +0.0597 |
| class-transformation | 3.9 | 40.8 | +0.0492 | +0.0416 | +0.0549 |

Bootstrap (1,000 resamples of the test split): Qini(S) − Qini(X) = **+23.6
[+9.4, +37.4]**, P(S better) = 1.00. The top OOF decile shows +8.5pp
observed uplift vs the +4.5pp average (`figures/fig_uplift_deciles.png`).

## Part 4, targeting policy (margin $25/conversion, cost $0.10/e-mail)

| policy | e-mails sent | expected profit |
|---|---|---|
| e-mail no one | 0 | $0 |
| blanket e-mail (k = 100%) | 42,693 | **−$949** |
| random 35% | 14,943 | −$332 |
| top 35% by predicted uplift (k*) | 14,943 | **+$1,065** |

Profit at k* bootstrap 95% CI: [−$36, +$2,238] (conversions are rare;
uncertainty is honest). The same campaign flips from value-destroying to
value-creating purely through *who* is contacted, the entire point of
uplift modelling. Sweep margin/cost live in the Streamlit app.
