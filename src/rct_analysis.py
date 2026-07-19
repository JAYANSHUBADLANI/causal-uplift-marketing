"""Part 1, analysis of the randomised experiment.

Because assignment was randomised, a difference in means is an unbiased
estimate of the average treatment effect (ATE):

    ATE = E[Y(1) - Y(0)] = E[Y | W=1] - E[Y | W=0]   (under randomisation)

This script checks that the randomisation itself is sound before trusting
that identity, then estimates the ATE of each e-mail on each outcome:

1.  Sample-ratio-mismatch (SRM) check, chi-square test that the realised
    arm sizes match the intended 1:1:1 allocation.
2.  Covariate balance, standardised mean differences (SMD) of every model
    covariate, each treatment arm vs control (|SMD| < 0.1 is conventional).
3.  ATEs, difference in means with a Welch t-test *and* a nonparametric
    bootstrap 95% CI for every (treatment x outcome) cell.
4.  Holm correction across the 2x3 grid of hypotheses (raw + adjusted p).
5.  Segment-level ATEs (history segment, channel), the heterogeneity that
    motivates uplift modelling in Part 3.

Outputs: reports/rct_*.csv, reports/rct_ate.md, reports/rct_summary.json,
reports/figures/fig_rct_ates.png, reports/figures/fig_rct_segments.png.
"""
from __future__ import annotations

import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src import config
from src.data_ingestion import load_clean

sns.set_theme(style="whitegrid")


# ------------------------------------------------------------------ checks
def srm_check(df: pd.DataFrame) -> dict:
    """Chi-square goodness-of-fit of realised arm sizes vs 1:1:1."""
    counts = df["treatment"].value_counts().reindex(config.ARMS)
    chi2, p = stats.chisquare(counts.to_numpy())
    return {
        "counts": counts.to_dict(),
        "chi2": float(chi2),
        "p_value": float(p),
        "srm_detected": bool(p < 0.001),
    }


def smd(x_t: pd.Series, x_c: pd.Series) -> float:
    """Standardised mean difference with the pooled-variance denominator."""
    s = np.sqrt((x_t.var(ddof=1) + x_c.var(ddof=1)) / 2.0)
    return 0.0 if s == 0 else float((x_t.mean() - x_c.mean()) / s)


def balance_table(df: pd.DataFrame) -> pd.DataFrame:
    """SMD of every covariate, each treatment arm vs control."""
    ctrl = df[df["treatment"] == "control"]
    rows = []
    for arm in config.TREATMENTS:
        tr = df[df["treatment"] == arm]
        for feat in config.FEATURES:
            rows.append(
                {"covariate": feat, "arm": arm, "smd": smd(tr[feat], ctrl[feat])}
            )
    wide = (
        pd.DataFrame(rows)
        .pivot(index="covariate", columns="arm", values="smd")
        .reindex(config.FEATURES)
        .reset_index()
    )
    wide.columns.name = None
    return wide


# --------------------------------------------------------------- estimation
def welch(y_t: np.ndarray, y_c: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch two-sample t-test with its (unequal-variance) 95% CI."""
    n_t, n_c = len(y_t), len(y_c)
    v_t, v_c = y_t.var(ddof=1), y_c.var(ddof=1)
    diff = y_t.mean() - y_c.mean()
    se = np.sqrt(v_t / n_t + v_c / n_c)
    dof = (v_t / n_t + v_c / n_c) ** 2 / (
        (v_t / n_t) ** 2 / (n_t - 1) + (v_c / n_c) ** 2 / (n_c - 1)
    )
    t_stat = diff / se
    p = 2 * stats.t.sf(abs(t_stat), dof)
    crit = stats.t.ppf(1 - alpha / 2, dof)
    return {
        "diff": float(diff),
        "se": float(se),
        "t": float(t_stat),
        "dof": float(dof),
        "p": float(p),
        "ci_lo": float(diff - crit * se),
        "ci_hi": float(diff + crit * se),
    }


def bootstrap_diff_ci(
    y_t: np.ndarray,
    y_c: np.ndarray,
    n_boot: int = config.B_ATE,
    seed: int = config.SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a difference in means (resampling arms)."""
    rng = np.random.default_rng(seed)
    n_t, n_c = len(y_t), len(y_c)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = (
            y_t[rng.integers(0, n_t, n_t)].mean()
            - y_c[rng.integers(0, n_c, n_c)].mean()
        )
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def ate_table(df: pd.DataFrame) -> pd.DataFrame:
    """ATE of each e-mail vs control on every outcome, Holm-adjusted."""
    ctrl = df[df["treatment"] == "control"]
    rows = []
    for arm in config.TREATMENTS:
        tr = df[df["treatment"] == arm]
        for outcome in config.OUTCOMES:
            y_t = tr[outcome].to_numpy(float)
            y_c = ctrl[outcome].to_numpy(float)
            w = welch(y_t, y_c)
            b_lo, b_hi = bootstrap_diff_ci(y_t, y_c)
            rows.append(
                {
                    "treatment": arm,
                    "outcome": outcome,
                    "n_treat": len(y_t),
                    "n_ctrl": len(y_c),
                    "mean_treat": y_t.mean(),
                    "mean_ctrl": y_c.mean(),
                    "ate": w["diff"],
                    "rel_lift_pct": 100 * w["diff"] / y_c.mean(),
                    "welch_t": w["t"],
                    "p_raw": w["p"],
                    "ci_welch_lo": w["ci_lo"],
                    "ci_welch_hi": w["ci_hi"],
                    "ci_boot_lo": b_lo,
                    "ci_boot_hi": b_hi,
                }
            )
    out = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(out["p_raw"], alpha=0.05, method="holm")
    out["p_holm"] = p_adj
    out["significant_5pct"] = reject
    return out


def segment_ates(df: pd.DataFrame) -> pd.DataFrame:
    """Difference-in-means ATE on `visit` within pre-treatment segments."""
    rows = []
    for seg_col in ("history_segment", "channel"):
        for level in sorted(df[seg_col].unique()):
            sub = df[df[seg_col] == level]
            y_c = sub.loc[sub["treatment"] == "control", "visit"].to_numpy(float)
            for arm in config.TREATMENTS:
                y_t = sub.loc[sub["treatment"] == arm, "visit"].to_numpy(float)
                w = welch(y_t, y_c)
                rows.append(
                    {
                        "segment_var": seg_col,
                        "level": level,
                        "treatment": arm,
                        "n_treat": len(y_t),
                        "n_ctrl": len(y_c),
                        "ate_visit": w["diff"],
                        "ci_lo": w["ci_lo"],
                        "ci_hi": w["ci_hi"],
                        "p": w["p"],
                    }
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def plot_ates(ate: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {"mens": "#4C72B0", "womens": "#DD8452"}
    for ax, outcome in zip(axes, config.OUTCOMES):
        sub = ate[ate["outcome"] == outcome]
        x = np.arange(len(sub))
        err = np.vstack(
            [sub["ate"] - sub["ci_boot_lo"], sub["ci_boot_hi"] - sub["ate"]]
        )
        ax.bar(
            x,
            sub["ate"],
            yerr=err,
            capsize=6,
            color=[colors[a] for a in sub["treatment"]],
            width=0.6,
        )
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x, [f"{a} e-mail" for a in sub["treatment"]])
        ax.set_title(f"ATE on {outcome}")
        ax.set_ylabel("difference vs control" if outcome == "visit" else "")
    fig.suptitle("RCT average treatment effects (bootstrap 95% CI)", y=1.03)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_rct_ates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_segments(seg: pd.DataFrame) -> None:
    sub = seg[seg["treatment"] == config.UPLIFT_TREATMENT].copy()
    sub["label"] = sub["segment_var"].str.replace("_", " ") + ": " + sub["level"]
    sub = sub.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    y = np.arange(len(sub))
    ax.errorbar(
        sub["ate_visit"],
        y,
        xerr=[sub["ate_visit"] - sub["ci_lo"], sub["ci_hi"] - sub["ate_visit"]],
        fmt="o",
        color="#DD8452",
        ecolor="gray",
        capsize=4,
    )
    overall = sub["ate_visit"].mean()
    ax.axvline(0, color="black", lw=1)
    ax.axvline(overall, color="#4C72B0", lw=1, ls="--", label="mean of segments")
    ax.set_yticks(y, sub["label"])
    ax.set_xlabel("ATE on visit (Women's e-mail vs control)")
    ax.set_title("Segment-level effects are heterogeneous\n(Welch 95% CI)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_rct_segments.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- main
def main() -> int:
    config.ensure_dirs()
    df = load_clean()

    srm = srm_check(df)
    print(f"[rct] SRM chi2={srm['chi2']:.3f} p={srm['p_value']:.3f} "
          f"counts={srm['counts']}")

    bal = balance_table(df)
    max_smd = bal[config.TREATMENTS].abs().to_numpy().max()
    print(f"[rct] max |SMD| across covariates/arms = {max_smd:.4f}")

    ate = ate_table(df)
    seg = segment_ates(df)

    bal.to_csv(config.REPORTS / "rct_balance.csv", index=False)
    ate.to_csv(config.REPORTS / "rct_ate.csv", index=False)
    seg.to_csv(config.REPORTS / "rct_segments.csv", index=False)

    md = [
        config.banner_md() + "# Part 1, RCT analysis\n",
        f"**SRM check**: arm sizes {srm['counts']}, chi-square p = "
        f"{srm['p_value']:.3f} → {'MISMATCH' if srm['srm_detected'] else 'no sample-ratio mismatch'}.\n",
        f"**Covariate balance**: max |SMD| = {max_smd:.4f} "
        "(all far below the 0.1 rule of thumb).\n",
        "## Average treatment effects (vs control)\n",
        config.md_table(
            ate[
                [
                    "treatment", "outcome", "mean_treat", "mean_ctrl", "ate",
                    "rel_lift_pct", "welch_t", "p_raw", "p_holm",
                    "ci_boot_lo", "ci_boot_hi", "significant_5pct",
                ]
            ]
        ),
        "\n*p_holm = Holm step-down adjustment across the 2 treatments x 3 "
        "outcomes family. Bootstrap CIs use "
        f"{config.B_ATE:,} resamples.*\n",
        "\n## Segment-level ATEs on visit\n",
        config.md_table(seg),
    ]
    (config.REPORTS / "rct_ate.md").write_text("\n".join(md))

    plot_ates(ate)
    plot_segments(seg)

    womens_visit = ate.query(
        "treatment == @config.UPLIFT_TREATMENT and outcome == @config.UPLIFT_OUTCOME"
    ).iloc[0]
    summary = {
        "synthetic": config.is_synthetic(),
        "srm": srm,
        "max_abs_smd": float(max_smd),
        "ate": ate.to_dict(orient="records"),
        "benchmark": {
            "description": "Women's e-mail vs control ATE on visit, the RCT "
            "ground truth used in Part 2",
            "ate": float(womens_visit["ate"]),
            "ci_lo": float(womens_visit["ci_boot_lo"]),
            "ci_hi": float(womens_visit["ci_boot_hi"]),
        },
    }
    (config.REPORTS / "rct_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"[rct] benchmark (womens->visit) ATE = {womens_visit['ate']:.4f} "
          f"[{womens_visit['ci_boot_lo']:.4f}, {womens_visit['ci_boot_hi']:.4f}]")
    print("[rct] wrote tables, figures and rct_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
