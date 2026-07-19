# Causal Inference & Uplift Modeling: who should we target?

An end-to-end causal analysis of the **Hillstrom e-mail RCT** (64,000
customers): from randomised-experiment analysis, through observational
causal methods validated against known ground truth, to heterogeneous
treatment effects with from-scratch uplift meta-learners, ending in a
profit-optimal targeting policy.

The punchline, on real data: at $25 margin per incremental conversion and
$0.10 per e-mail, **blanket e-mailing loses ≈ $949 while e-mailing the top
35% ranked by predicted uplift makes ≈ +$1,065**, the identical campaign
flips from value-destroying to value-creating purely by choosing *who* to
contact.

## Why uplift modelling (the business framing)

A marketing e-mail only creates value when it *changes* someone's
behaviour. Segmenting customers by their two potential outcomes:

| | would act if e-mailed | would not act if e-mailed |
|---|---|---|
| **would act anyway** | Sure things, e-mail wasted | Do-not-disturbs, e-mail *destroys* value |
| **would not act anyway** | **Persuadables, the only ROI** | Lost causes, e-mail wasted |

Classic response models rank customers by P(buy), which is dominated by
sure things. Uplift models rank by the *causal increment*
τ(x) = E[Y(1) − Y(0) | X = x], i.e. they hunt persuadables. This project
builds that ranking and prices it.

## The data

Kevin Hillstrom's MineThatData E-Mail Analytics Challenge (March 2008):
64,000 customers with a purchase in the last twelve months, randomised
1:1:1 to a **Men's-merchandise e-mail**, a **Women's-merchandise e-mail**,
or **no e-mail**, tracked for two weeks. Outcomes: `visit`, `conversion`,
`spend`. Covariates: `recency`, `history` (+ segment), `mens`, `womens`,
`zip_code`, `newbie`, `channel`, all pre-treatment.

`python -m src.data_ingestion` downloads the CSV from
[minethatdata.com](http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv)
and caches it under `data/raw/`. If the network makes the download
impossible, a schema-identical synthetic sample is generated and **every**
artefact is stamped `SYNTHETIC, regenerate on real data` (plus a marker
file in `data/raw/`). All numbers in this README come from the **real**
data.

## The narrative arc

1. **Part 1, trust the experiment** (`src/rct_analysis.py`): verify the
   randomisation, estimate ATEs properly, correct for multiplicity, and
   expose the heterogeneity that motivates everything after.
2. **Part 2, pretend we couldn't randomise** (`src/observational.py`):
   manufacture confounding by biased subsampling, show the naive answer is
   badly wrong, then recover the RCT truth with four observational
   methods. Because the truth is known, this is a *validated* demo, not a
   leap of faith.
3. **Part 3, go individual** (`src/uplift/`): S/T/X/class-transformation
   meta-learners implemented from scratch over LightGBM, evaluated with
   from-scratch Qini machinery.
4. **Part 4, make the decision** (`src/policy.py`): profit simulation
   comparing blanket / none / random-k% / top-k%-by-uplift policies.

## Part 1, RCT analysis

Randomisation checks pass: SRM chi-square p = **0.904** (arm sizes
21,306 / 21,307 / 21,387) and max covariate |SMD| = **0.014**: so
difference in means identifies the ATE:

$$\widehat{\mathrm{ATE}} = \bar Y_{treat} - \bar Y_{control}$$

Each cell gets a Welch t-test **and** a percentile-bootstrap 95% CI (2,000
resamples), with **Holm** correction across the 2 treatments × 3 outcomes
family:

| treatment | outcome | control | treated | ATE | bootstrap 95% CI | p (Holm) |
|---|---|---|---|---|---|---|
| mens | visit | 0.1062 | 0.1828 | **+0.0766** | [0.0701, 0.0834] | 8.2e-112 |
| mens | conversion | 0.0057 | 0.0125 | +0.0068 | [0.0050, 0.0086] | 6.0e-13 |
| mens | spend ($) | 0.653 | 1.423 | +0.770 | [0.484, 1.052] | 3.5e-07 |
| womens | visit | 0.1062 | 0.1514 | **+0.0452** | [0.0390, 0.0516] | 1.2e-43 |
| womens | conversion | 0.0057 | 0.0088 | +0.0031 | [0.0014, 0.0048] | 3.1e-04 |
| womens | spend ($) | 0.653 | 1.077 | +0.424 | [0.152, 0.686] | 1.1e-03 |

All six effects survive Holm. But the Women's-e-mail effect on `visit`
already ranges from ~3.6pp to ~6.6pp across one-dimensional segments
(`reports/figures/fig_rct_segments.png`), and one-dimensional cuts
understate what a multivariate model can find (Part 3's top decile reaches
+8.5pp). Averages hide who is actually persuadable. Hence Parts 3-4.

## Part 2, observational methods vs RCT ground truth (centerpiece)

Real campaign data is rarely randomised, marketers target their best
customers. We recreate exactly that bias inside the Women's-e-mail +
control cell: with standardised $z_h$ (log history) and $z_r$ (recency),
each unit's *true* selection propensity is

$$e^*(x) = \mathrm{clip}\big(\sigma(-0.20 + 0.90\,z_h - 0.70\,z_r + 0.50\cdot\text{multichannel}),\ 0.03,\ 0.97\big),$$

keeping treated units with probability $e^*(x)$ and controls with
$1-e^*(x)$ (n = 21,313 after subsampling; max covariate |SMD| jumps to
0.98). Selection depends only on observables, so ignorability holds by
construction and the Part 1 ATE (+0.0452) remains the true value.

| method | estimand | estimate | 95% CI | abs. error vs RCT |
|---|---|---|---|---|
| **RCT benchmark** | ATE | **0.0452** | [0.0390, 0.0516] | - |
| naive difference in means | ATE | 0.0761 | [0.0671, 0.0851] | **0.0308** |
| regression adjustment (OLS, HC1) | ATE | 0.0445 | [0.0342, 0.0548] | 0.0007 |
| IPW (stabilised, trimmed) | ATE | 0.0463 | [0.0370, 0.0577] | 0.0011 |
| 1:1 NN matching* | ATT | 0.0554 | [0.0459, 0.0649] | 0.0101 |
| AIPW (doubly robust, cross-fit) | ATE | 0.0464 | [0.0350, 0.0579] | 0.0012 |

The naive estimate overstates the effect by **68%**: the mistake a
dashboard comparing e-mailed vs non-e-mailed customers would make. Every
adjustment method recovers the truth to within ~0.1pp. IPW uses logistic
propensities trimmed to [0.02, 0.98] with stabilised weights and a
bootstrap that *re-fits the propensity model in every replicate*; AIPW is
cross-fitted with influence-function SEs:

$$\hat\psi_i = \hat m_1(X_i) - \hat m_0(X_i) + \frac{W_i(Y_i - \hat m_1(X_i))}{\hat e(X_i)} - \frac{(1-W_i)(Y_i - \hat m_0(X_i))}{1-\hat e(X_i)}, \qquad \widehat{\mathrm{ATE}} = \frac1n\sum_i \hat\psi_i$$

\*Matching estimates the **ATT**: the effect among the (selected,
engaged) treated, which Part 1's heterogeneity says is genuinely larger,
an estimand difference, not an estimator failure. Diagnostics
(`fig_obs_overlap/love/weights/methods.png`): IPW weighting collapses the
max |SMD| from 0.98 to 0.02; matching pairs 99.9% of treated within a
0.2-SD caliper.

**The narrative:** when you cannot randomise, selection-on-observables
methods work, and here is proof against ground truth, plus the
diagnostics you should demand before believing any of them.

## Part 3, uplift modelling

Cell: **Women's e-mail vs control** (n = 42,693, treated share 0.501),
outcome **visit**. Four meta-learners, from scratch over LightGBM
(`src/uplift/meta_learners.py`): **S** (single model with the treatment as
a feature), **T** (two models), **X** (Künzel et al., imputed-effect
regressions blended by propensity), and **class transformation**
(Jaskowski & Jaroszewicz, valid here because assignment is ~50/50).

Evaluation is ranking-based (`src/uplift/qini.py`, from scratch). Sorting
by predicted uplift, the Qini curve tracks incremental visits among the
top-n:

$$Q(n) = Y_t(n) - Y_c(n)\,\frac{n_t(n)}{n_c(n)}$$

and the Qini coefficient is the area between $Q$ and the random-targeting
diagonal. Protocol: 70/30 split stratified on treatment × outcome; 5-fold
out-of-fold predictions within train as an overfitting check; final
metrics on the untouched test split.

| learner | Qini (test) | Qini (OOF train) | uplift@10% | uplift@20% | uplift@30% |
|---|---|---|---|---|---|
| **S-learner** | **54.8** | 84.0 | **+0.0849** | +0.0796 | +0.0752 |
| X-learner | 31.2 | 65.4 | +0.0558 | +0.0627 | +0.0718 |
| T-learner | 23.7 | 73.6 | +0.0440 | +0.0569 | +0.0597 |
| class-transformation | 3.9 | 40.8 | +0.0492 | +0.0416 | +0.0549 |

Bootstrap on the test split (1,000 resamples): **Qini(S) − Qini(X) = +23.6
[+9.4, +37.4]**, P(S better) = 1.00, a real difference, not noise. With
modest heterogeneity the S-learner's shrinkage toward a shared effect
surface wins; the T-learner overfits arm-specific noise, and the
class-transformation estimator pays its known variance penalty. The top
out-of-fold decile shows **+8.5pp** observed uplift vs the +4.5pp average
(`fig_uplift_deciles.png`), the model is finding persuadables, not just
likely visitors.

## Part 4, targeting policy and profit (the closing argument)

With margin $M$ per incremental conversion and cost $c$ per e-mail,
targeting the top-k% (m_k customers) yields

$$\pi(k) = m_k\big(\widehat{\Delta\mathrm{conv}}_k \cdot M - c\big),$$

where $\widehat{\Delta\mathrm{conv}}_k$ is estimated **empirically from the
randomised arms within the targeted group**, an offline policy-value
estimate that uses the model only for ranking. Defaults M = $25, c = $0.10
(`src/config.py`; both are CLI flags and live sliders in the app):

| policy | e-mails sent | expected profit |
|---|---|---|
| e-mail no one | 0 | $0 |
| blanket e-mail (k = 100%) | 42,693 | **−$949** |
| random 35% | 14,943 | −$332 |
| **top 35% by predicted uplift (k\*)** | 14,943 | **+$1,065** |

Profit-vs-k curve with k* marked: `reports/figures/fig_policy_profit.png`.
Bootstrap 95% CI at k*: [−$36, +$2,238], wide, because conversions are
rare; the honest conclusion is "directionally decisive, magnitude
uncertain, rerun at scale".

## Streamlit app

```
streamlit run app.py
```

Four tabs: **RCT results** (ATEs, CIs, segment effects), **observational
demo** (method-vs-truth table, love plot, overlap, weights), **uplift**
(Qini curves, learner table, deciles), **policy** (margin/cost sliders →
live profit curve and optimal k\*, recomputed from cached out-of-fold
scores, no retraining).

## How to run

```
git clone <this repo> && cd causal-uplift-marketing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.data_ingestion      # download + validate + feature frame
python -m src.rct_analysis        # Part 1
python -m src.observational       # Part 2  (needs Part 1's benchmark)
python -m src.uplift.run_uplift   # Part 3
python -m src.policy              # Part 4  (needs Part 3's scores)
streamlit run app.py
```

Total runtime ≈ 1 minute on a laptop. Every table/figure in `reports/` is
regenerated; notebooks in `notebooks/` are narrative companions and are
intentionally unexecuted.

## Repository structure

```
causal-uplift-marketing/
├── data/raw/  data/processed/          # gitignored; .gitkeep committed
├── notebooks/
│   ├── 01_eda_rct.ipynb                # EDA + Part 1 walkthrough
│   └── 02_causal_uplift.ipynb          # Parts 2-4 walkthrough
├── src/
│   ├── config.py                       # paths, constants, knobs
│   ├── data_ingestion.py               # download / validate / features
│   ├── rct_analysis.py                 # Part 1
│   ├── observational.py                # Part 2
│   ├── policy.py                       # Part 4
│   └── uplift/
│       ├── meta_learners.py            # S / T / X / CT, from scratch
│       ├── qini.py                     # Qini machinery, from scratch
│       └── run_uplift.py               # Part 3 runner
├── reports/                            # committed results + figures
├── app.py                              # Streamlit dashboard
└── requirements.txt
```

## Limitations (kept honest)

This is a single two-week campaign snapshot: no long-term effects
(fatigue, unsubscribes, cannibalisation) are observable, and external
validity beyond this retailer is unknown. The profit parameters are
assumptions, the app exists precisely to sweep them. The uplift ranking
optimises *visit* uplift while profit prices *conversions*; with ~310
conversions in the cell, modelling conversion uplift directly was too
noisy, and the mismatch is acknowledged. Meta-learner uncertainty is
bootstrap-only (no honest/debiased CIs for individual τ̂(x)); the matching
SE ignores control reuse and propensity estimation; the
class-transformation learner requires ~50/50 assignment; and Part 2's
validation shows the methods work *when ignorability holds by
construction*, in genuinely observational data, unobserved confounding
remains the unfixable risk.

## References

- Hillstrom, K. (2008). *The MineThatData E-Mail Analytics And Data Mining
  Challenge*. minethatdata.com.
- Radcliffe, N. J. (2007). *Using control groups to target on predicted
  lift.* Direct Marketing Analytics Journal.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019).
  *Metalearners for estimating heterogeneous treatment effects using
  machine learning.* PNAS 116(10).
- Jaskowski, M., & Jaroszewicz, S. (2012). *Uplift modeling for clinical
  trial data.* ICML Workshop on Clinical Data Analysis.
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The central role of the
  propensity score in observational studies for causal effects.*
  Biometrika 70(1).
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). *Estimation of
  regression coefficients when some regressors are not always observed.*
  JASA 89(427).
