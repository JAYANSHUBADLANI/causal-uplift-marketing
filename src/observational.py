"""Part 2, observational causal inference, benchmarked against RCT truth.

The Hillstrom data is a randomised experiment, so the ATE of the Women's
e-mail on ``visit`` is *known* (Part 1). That makes it the perfect test bed
for observational methods:

1.  **Manufacture confounding.** Build an "observational" dataset by biased
    subsampling of the Women's-e-mail + control arms. With
    ``z_h = standardised log(1 + history)`` and ``z_r = standardised
    recency``, define a true selection propensity

        e*(x) = clip( sigmoid( -0.20 + 0.90*z_h - 0.70*z_r
                               + 0.50*multichannel ), 0.03, 0.97 )

    and keep a treated customer with probability e*(x) and a control
    customer with probability 1 - e*(x). Because the two arms are (nearly)
    the same size, P(W=1 | X, kept) = e*(x): engaged customers (high
    history, recent, multichannel) are now over-represented among the
    treated, exactly the "marketers target their best customers" bias.
    Selection depends only on observed X, so ignorability holds by
    construction and the RCT ATE remains the estimand's true value.

2.  **Show the naive estimate is biased**, then recover the truth with
    regression adjustment, IPW (stabilised + trimmed), 1:1 nearest-neighbour
    propensity matching, and AIPW (doubly robust, cross-fitted):

        AIPW:  psi_i = m1(X_i) - m0(X_i)
                       + W_i (Y_i - m1(X_i)) / e(X_i)
                       - (1-W_i)(Y_i - m0(X_i)) / (1 - e(X_i))

3.  **Diagnostics**: propensity overlap, love plot (SMD before/after
    adjustment), stabilised-weight distribution.

Outputs: reports/obs_methods.{csv,md}, reports/obs_summary.json and
reports/figures/fig_obs_*.png.
"""
from __future__ import annotations

import json
import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from lightgbm import LGBMClassifier
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src import config
from src.data_ingestion import load_clean
from src.rct_analysis import welch

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

OUTCOME = config.UPLIFT_OUTCOME  # visit
# Drop one level per one-hot group so design matrices are full rank.
ADJ_FEATURES = [f for f in config.FEATURES if f not in ("zip_Urban", "ch_Web")]

OUTCOME_MODEL_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    random_state=config.SEED,
    verbose=-1,
)


# ------------------------------------------------------------- construction
def true_selection_propensity(df: pd.DataFrame) -> np.ndarray:
    """The *known* selection mechanism used to manufacture confounding."""
    z_h = (df["log_history"] - df["log_history"].mean()) / df["log_history"].std()
    z_r = (df["recency"] - df["recency"].mean()) / df["recency"].std()
    score = -0.20 + 0.90 * z_h - 0.70 * z_r + 0.50 * df["ch_Multichannel"]
    return np.clip(expit(score), 0.03, 0.97)


def make_confounded(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Biased subsample of the womens+control arms (see module docstring)."""
    sub = df[df["treatment"].isin([config.UPLIFT_TREATMENT, "control"])].copy()
    sub["W"] = (sub["treatment"] == config.UPLIFT_TREATMENT).astype(int)
    e_star = true_selection_propensity(sub)
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=len(sub))
    keep = np.where(sub["W"] == 1, u < e_star, u < 1 - e_star)
    obs = sub.loc[keep].reset_index(drop=True)
    return obs


# -------------------------------------------------------------- estimators
def naive_diff(obs: pd.DataFrame) -> dict:
    y_t = obs.loc[obs["W"] == 1, OUTCOME].to_numpy(float)
    y_c = obs.loc[obs["W"] == 0, OUTCOME].to_numpy(float)
    w = welch(y_t, y_c)
    return {"estimate": w["diff"], "ci_lo": w["ci_lo"], "ci_hi": w["ci_hi"]}


def regression_adjustment(obs: pd.DataFrame) -> dict:
    """OLS of the outcome on treatment + covariates, HC1 robust SEs.

    With a constant treatment coefficient this is the classic linear
    adjustment estimator of the ATE (a linear probability model here;
    its misspecification risk is exactly why AIPW is also reported).
    """
    X = sm.add_constant(obs[["W"] + ADJ_FEATURES].astype(float))
    fit = sm.OLS(obs[OUTCOME].astype(float), X).fit(cov_type="HC1")
    ci = fit.conf_int().loc["W"]
    return {"estimate": float(fit.params["W"]), "ci_lo": float(ci[0]), "ci_hi": float(ci[1])}


def fit_propensity(obs: pd.DataFrame) -> np.ndarray:
    """Logistic-regression propensity scores e(x) = P(W=1 | X)."""
    Xs = StandardScaler().fit_transform(obs[ADJ_FEATURES].astype(float))
    lr = LogisticRegression(max_iter=500, random_state=config.SEED)
    lr.fit(Xs, obs["W"])
    return lr.predict_proba(Xs)[:, 1]


def stabilised_weights(w: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Stabilised IPW weights with trimmed propensities."""
    e = np.clip(e, *config.PS_CLIP)
    p = w.mean()
    return np.where(w == 1, p / e, (1 - p) / (1 - e))


def ipw_hajek(y: np.ndarray, w: np.ndarray, sw: np.ndarray) -> float:
    """Hajek (self-normalised) IPW estimate of the ATE."""
    m1 = np.sum(w * sw * y) / np.sum(w * sw)
    m0 = np.sum((1 - w) * sw * y) / np.sum((1 - w) * sw)
    return float(m1 - m0)


def ipw(obs: pd.DataFrame, n_boot: int = config.B_OBS) -> tuple[dict, np.ndarray]:
    """IPW point estimate + bootstrap CI (propensity refitted per replicate)."""
    y = obs[OUTCOME].to_numpy(float)
    w = obs["W"].to_numpy(int)
    e_hat = fit_propensity(obs)
    sw = stabilised_weights(w, e_hat)
    est = ipw_hajek(y, w, sw)

    rng = np.random.default_rng(config.SEED)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(obs), len(obs))
        ob = obs.iloc[idx]
        eb = fit_propensity(ob)
        boots[b] = ipw_hajek(
            ob[OUTCOME].to_numpy(float),
            ob["W"].to_numpy(int),
            stabilised_weights(ob["W"].to_numpy(int), eb),
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"estimate": est, "ci_lo": float(lo), "ci_hi": float(hi)}, e_hat


def nn_matching(obs: pd.DataFrame, e_hat: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """1:1 nearest-neighbour matching on the logit propensity (ATT).

    With replacement, caliper = 0.2 SD of the logit propensity. The SE is
    the simple paired SE (ignores control reuse and propensity estimation,
    so it is mildly optimistic, flagged in the report).
    """
    lp = logit(np.clip(e_hat, *config.PS_CLIP))
    caliper = config.CALIPER_SD * lp.std()
    treated = np.flatnonzero(obs["W"] == 1)
    controls = np.flatnonzero(obs["W"] == 0)
    nn = NearestNeighbors(n_neighbors=1).fit(lp[controls].reshape(-1, 1))
    dist, pos = nn.kneighbors(lp[treated].reshape(-1, 1))
    ok = dist[:, 0] <= caliper
    m_t = treated[ok]
    m_c = controls[pos[ok, 0]]
    d = obs[OUTCOME].to_numpy(float)[m_t] - obs[OUTCOME].to_numpy(float)[m_c]
    est = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    matched = pd.concat([obs.iloc[m_t], obs.iloc[m_c]])
    return (
        {
            "estimate": est,
            "ci_lo": est - 1.96 * se,
            "ci_hi": est + 1.96 * se,
            "n_matched": int(ok.sum()),
            "pct_treated_matched": float(100 * ok.mean()),
        },
        matched,
    )


def aipw(obs: pd.DataFrame, n_folds: int = 2) -> dict:
    """Cross-fitted AIPW (doubly robust) estimate of the ATE.

    Nuisances per fold: logistic-regression propensity + gradient-boosted
    outcome models m1, m0 fit on the *other* fold. The influence-function
    variance gives the SE: se = sd(psi) / sqrt(n).
    """
    y = obs[OUTCOME].to_numpy(float)
    w = obs["W"].to_numpy(int)
    X = obs[config.FEATURES].astype(float).to_numpy()
    Xs = StandardScaler().fit_transform(obs[ADJ_FEATURES].astype(float))
    psi = np.empty(len(obs))
    for train, held in KFold(n_folds, shuffle=True, random_state=config.SEED).split(X):
        lr = LogisticRegression(max_iter=500, random_state=config.SEED)
        lr.fit(Xs[train], w[train])
        e = np.clip(lr.predict_proba(Xs[held])[:, 1], *config.PS_CLIP)

        m1 = LGBMClassifier(**OUTCOME_MODEL_PARAMS)
        m1.fit(X[train][w[train] == 1], y[train][w[train] == 1])
        m0 = LGBMClassifier(**OUTCOME_MODEL_PARAMS)
        m0.fit(X[train][w[train] == 0], y[train][w[train] == 0])
        mu1 = m1.predict_proba(X[held])[:, 1]
        mu0 = m0.predict_proba(X[held])[:, 1]

        psi[held] = (
            mu1
            - mu0
            + w[held] * (y[held] - mu1) / e
            - (1 - w[held]) * (y[held] - mu0) / (1 - e)
        )
    est = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(len(psi)))
    return {"estimate": est, "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se}


# -------------------------------------------------------------- diagnostics
def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    m = _wmean(x, w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def weighted_smd(obs: pd.DataFrame, sw: np.ndarray) -> pd.Series:
    """SMD of each covariate after stabilised-IPW weighting."""
    out = {}
    t = obs["W"] == 1
    for f in config.FEATURES:
        x = obs[f].to_numpy(float)
        m_t, m_c = _wmean(x[t], sw[t.to_numpy()]), _wmean(x[~t], sw[~t.to_numpy()])
        v_t, v_c = _wvar(x[t], sw[t.to_numpy()]), _wvar(x[~t], sw[~t.to_numpy()])
        denom = np.sqrt((v_t + v_c) / 2)
        out[f] = 0.0 if denom == 0 else (m_t - m_c) / denom
    return pd.Series(out)


def raw_smd(frame: pd.DataFrame) -> pd.Series:
    out = {}
    t = frame["W"] == 1
    for f in config.FEATURES:
        x_t, x_c = frame.loc[t, f].astype(float), frame.loc[~t, f].astype(float)
        denom = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2)
        out[f] = 0.0 if denom == 0 else (x_t.mean() - x_c.mean()) / denom
    return pd.Series(out)


def plot_overlap(obs: pd.DataFrame, e_hat: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(e_hat[obs["W"] == 1], bins=bins, alpha=0.55, density=True,
            label="treated (Women's e-mail)", color="#DD8452")
    ax.hist(e_hat[obs["W"] == 0], bins=bins, alpha=0.55, density=True,
            label="control", color="#4C72B0")
    for c in config.PS_CLIP:
        ax.axvline(c, color="gray", ls=":", lw=1)
    ax.set_xlabel("estimated propensity  e(x)")
    ax.set_ylabel("density")
    ax.set_title("Propensity overlap in the confounded sample")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_obs_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_love(smd_raw: pd.Series, smd_ipw: pd.Series, smd_match: pd.Series) -> None:
    order = smd_raw.abs().sort_values().index
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(smd_raw[order].abs(), y, label="unadjusted", color="#C44E52", zorder=3)
    ax.scatter(smd_ipw[order].abs(), y, label="IPW-weighted", color="#4C72B0",
               marker="s", zorder=3)
    ax.scatter(smd_match[order].abs(), y, label="matched", color="#55A868",
               marker="^", zorder=3)
    ax.axvline(0.1, color="gray", ls="--", lw=1, label="0.1 threshold")
    ax.set_yticks(y, order)
    ax.set_xlabel("|standardised mean difference|")
    ax.set_title("Love plot, covariate balance before/after adjustment")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_obs_love.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weights(sw: np.ndarray, w: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(sw[w == 1], bins=60, alpha=0.6, label="treated", color="#DD8452")
    ax.hist(sw[w == 0], bins=60, alpha=0.6, label="control", color="#4C72B0")
    ax.set_xlabel("stabilised weight")
    ax.set_ylabel("count")
    ax.set_title(
        f"Stabilised IPW weights (max={sw.max():.2f}, mean={sw.mean():.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_obs_weights.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_methods(table: pd.DataFrame, benchmark: dict) -> None:
    sub = table[table["method"] != "RCT benchmark"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.axvspan(benchmark["ci_lo"], benchmark["ci_hi"], color="#55A868", alpha=0.18,
               label="RCT 95% CI")
    ax.axvline(benchmark["ate"], color="#55A868", lw=2, label="RCT ground truth")
    y = np.arange(len(sub))
    colors = ["#C44E52" if m == "naive difference in means" else "#4C72B0"
              for m in sub["method"]]
    ax.errorbar(sub["estimate"], y,
                xerr=[sub["estimate"] - sub["ci_lo"], sub["ci_hi"] - sub["estimate"]],
                fmt="none", ecolor="gray", capsize=4, zorder=2)
    ax.scatter(sub["estimate"], y, color=colors, s=60, zorder=3)
    ax.set_yticks(y, sub["method"])
    ax.set_xlabel(f"estimated effect of Women's e-mail on {OUTCOME}")
    ax.set_title("Observational estimators vs the randomised ground truth")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_obs_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    config.ensure_dirs()
    df = load_clean()
    benchmark = json.loads((config.REPORTS / "rct_summary.json").read_text())["benchmark"]

    obs = make_confounded(df)
    n_t = int(obs["W"].sum())
    print(f"[obs] confounded sample: n={len(obs):,} (treated={n_t:,}, "
          f"control={len(obs) - n_t:,})")

    results = {"naive difference in means": naive_diff(obs)}
    results["regression adjustment (OLS, HC1)"] = regression_adjustment(obs)
    ipw_res, e_hat = ipw(obs)
    results["IPW (stabilised, trimmed)"] = ipw_res
    match_res, matched = nn_matching(obs, e_hat)
    results["1:1 NN matching (ATT)"] = {
        k: match_res[k] for k in ("estimate", "ci_lo", "ci_hi")
    }
    results["AIPW (doubly robust, cross-fit)"] = aipw(obs)

    rows = [
        {
            "method": "RCT benchmark",
            "estimand": "ATE",
            "estimate": benchmark["ate"],
            "ci_lo": benchmark["ci_lo"],
            "ci_hi": benchmark["ci_hi"],
            "abs_error_vs_rct": 0.0,
        }
    ]
    for name, r in results.items():
        rows.append(
            {
                "method": name,
                "estimand": "ATT" if "ATT" in name else "ATE",
                "estimate": r["estimate"],
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "abs_error_vs_rct": abs(r["estimate"] - benchmark["ate"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(config.REPORTS / "obs_methods.csv", index=False)

    # diagnostics
    w = obs["W"].to_numpy(int)
    sw = stabilised_weights(w, e_hat)
    smd_before = raw_smd(obs)
    smd_ipw = weighted_smd(obs, sw)
    smd_match = raw_smd(matched)
    plot_overlap(obs, e_hat)
    plot_love(smd_before, smd_ipw, smd_match)
    plot_weights(sw, w)
    plot_methods(table, benchmark)

    md = [
        config.banner_md() + "# Part 2, observational methods vs RCT truth\n",
        "Confounding was manufactured by keeping treated units with "
        "probability `e*(x)` and controls with probability `1 - e*(x)`, where\n\n"
        "```\ne*(x) = clip(sigmoid(-0.20 + 0.90*z_log_history - 0.70*z_recency "
        "+ 0.50*multichannel), 0.03, 0.97)\n```\n",
        f"Sample: {len(obs):,} customers ({n_t:,} treated). "
        f"Naive bias vs RCT truth: "
        f"{results['naive difference in means']['estimate'] - benchmark['ate']:+.4f} "
        f"({100 * (results['naive difference in means']['estimate'] - benchmark['ate']) / benchmark['ate']:+.1f}% of the true effect).\n",
        "## Method comparison\n",
        config.md_table(table),
        f"\n*Matching: {match_res['n_matched']:,} pairs "
        f"({match_res['pct_treated_matched']:.1f}% of treated matched within "
        f"caliper {config.CALIPER_SD} SD of the logit propensity; with "
        "replacement; paired SE is approximate. Matching estimates the ATT, "
        "which is why it is footnoted rather than compared 1:1 with the ATE "
        "rows. IPW propensities trimmed to "
        f"{list(config.PS_CLIP)}; bootstrap refits the propensity model in "
        f"each of {config.B_OBS} replicates. AIPW uses 2-fold cross-fitting "
        "with influence-function SEs.*\n",
        "\n## Balance diagnostics\n",
        f"max |SMD| unadjusted = {smd_before.abs().max():.3f} -> "
        f"IPW-weighted = {smd_ipw.abs().max():.3f} -> "
        f"matched = {smd_match.abs().max():.3f} (threshold 0.1).\n",
    ]
    (config.REPORTS / "obs_methods.md").write_text("\n".join(md))

    summary = {
        "synthetic": config.is_synthetic(),
        "n_obs": len(obs),
        "n_treated": n_t,
        "benchmark": benchmark,
        "methods": {name: r for name, r in results.items()},
        "matching_detail": match_res,
        "balance": {
            "max_smd_unadjusted": float(smd_before.abs().max()),
            "max_smd_ipw": float(smd_ipw.abs().max()),
            "max_smd_matched": float(smd_match.abs().max()),
        },
    }
    (config.REPORTS / "obs_summary.json").write_text(json.dumps(summary, indent=2))

    for name, r in results.items():
        print(f"[obs] {name:38s} {r['estimate']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  "
              f"|err|={abs(r['estimate'] - benchmark['ate']):.4f}")
    print(f"[obs] RCT truth {benchmark['ate']:+.4f} "
          f"[{benchmark['ci_lo']:+.4f}, {benchmark['ci_hi']:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
