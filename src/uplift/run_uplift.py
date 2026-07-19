"""Part 3 runner, heterogeneous treatment effects with uplift meta-learners.

Cell under study: **Women's e-mail vs control**, outcome **visit** (the
highest-signal combination, per Part 1).

Protocol
--------
1.  70/30 train/test split, stratified on the (treatment x outcome) cell so
    both arms and both outcome classes keep their proportions.
2.  Each meta-learner (S, T, X, class-transformation) is fit on the training
    split and scored on the *held-out test* split.
3.  Honesty check: out-of-fold (5-fold, stratified) predictions *within the
    training split*, a large gap between OOF-train and test Qini would
    signal overfitting of the ranking.
4.  Evaluation: Qini curves, Qini coefficient, uplift@{10,20,30}%, and an
    uplift-by-decile table (all implemented from scratch in ``qini.py``).
5.  A percentile bootstrap (resampling test customers) gives a CI for the
    Qini-coefficient *difference* between the two best learners.
6.  The best learner is refit in a 5-fold scheme over the **full** cell to
    produce out-of-fold uplift scores for every customer, the input to the
    Part 4 policy simulation (``data/processed/uplift_scores.csv``).

Outputs: reports/uplift_*.{csv,md}, reports/uplift_summary.json,
reports/figures/fig_qini_curves.png, reports/figures/fig_uplift_deciles.png.
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
from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config
from src.data_ingestion import load_clean
from src.uplift.meta_learners import (
    ClassTransformation,
    SLearner,
    TLearner,
    XLearner,
)
from src.uplift.qini import decile_table, qini_coefficient, qini_curve, uplift_at_k

sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

LEARNERS = {
    "S-learner": SLearner,
    "T-learner": TLearner,
    "X-learner": XLearner,
    "class-transformation": ClassTransformation,
}
UPLIFT_AT = (0.10, 0.20, 0.30)


def load_cell():
    """The Women's-e-mail + control cell as numpy arrays."""
    df = load_clean()
    sub = df[df["treatment"].isin([config.UPLIFT_TREATMENT, "control"])].reset_index(
        drop=True
    )
    X = sub[config.FEATURES].astype(float).to_numpy()
    y = sub[config.UPLIFT_OUTCOME].to_numpy(int)
    w = (sub["treatment"] == config.UPLIFT_TREATMENT).to_numpy(int)
    return sub, X, y, w


def oof_scores(factory, X, y, w, n_folds: int = config.N_FOLDS) -> np.ndarray:
    """Out-of-fold uplift predictions (stratified on treatment x outcome)."""
    strata = 2 * w + y
    oof = np.empty(len(y))
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=config.SEED)
    for tr, te in skf.split(X, strata):
        model = factory().fit(X[tr], w[tr], y[tr])
        oof[te] = model.predict_uplift(X[te])
    return oof


def bootstrap_qini_difference(
    y, w, score_a, score_b, n_boot: int = config.B_QINI
) -> dict:
    """Percentile bootstrap for Qini(a) - Qini(b) on the test split."""
    rng = np.random.default_rng(config.SEED)
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = qini_coefficient(y[idx], w[idx], score_a[idx], seed=b) - \
            qini_coefficient(y[idx], w[idx], score_b[idx], seed=b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta": float(np.mean(diffs)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_better": float((diffs > 0).mean()),
    }


def plot_qini(test_curves: dict, y_te, w_te) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    palette = sns.color_palette("deep", len(test_curves))
    q_end = None
    for (name, (frac, q)), color in zip(test_curves.items(), palette):
        step = max(1, len(frac) // 400)
        ax.plot(frac[::step], q[::step], label=name, color=color, lw=2)
        q_end = q[-1]
    ax.plot([0, 1], [0, q_end], color="gray", ls="--", lw=1.5, label="random targeting")
    ax.set_xlabel("fraction of customers targeted (by predicted uplift)")
    ax.set_ylabel("incremental visits (Qini)")
    ax.set_title("Qini curves on the held-out test split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_qini_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_deciles(dec: pd.DataFrame, ate: float) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(dec["decile"], dec["uplift"], color="#4C72B0")
    ax.axhline(ate, color="#DD8452", ls="--", lw=1.5,
               label=f"overall ATE = {ate:.3f}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(dec["decile"])
    ax.set_xlabel("uplift-score decile (1 = highest predicted uplift)")
    ax.set_ylabel("observed uplift on visit")
    ax.set_title("Uplift by decile, best learner, out-of-fold scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_uplift_deciles.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    config.ensure_dirs()
    sub, X, y, w = load_cell()
    print(f"[uplift] cell: n={len(y):,}, treated share={w.mean():.4f}, "
          f"visit rate={y.mean():.4f}")

    strata = 2 * w + y
    idx_tr, idx_te = train_test_split(
        np.arange(len(y)),
        test_size=config.TEST_SIZE,
        stratify=strata,
        random_state=config.SEED,
    )
    X_tr, X_te = X[idx_tr], X[idx_te]
    y_tr, y_te = y[idx_tr], y[idx_te]
    w_tr, w_te = w[idx_tr], w[idx_te]

    rows, test_scores, test_curves = [], {}, {}
    for name, factory in LEARNERS.items():
        model = factory().fit(X_tr, w_tr, y_tr)
        tau_te = model.predict_uplift(X_te)
        tau_oof = oof_scores(factory, X_tr, y_tr, w_tr)
        test_scores[name] = tau_te
        test_curves[name] = qini_curve(y_te, w_te, tau_te)
        row = {
            "learner": name,
            "qini_test": qini_coefficient(y_te, w_te, tau_te),
            "qini_oof_train": qini_coefficient(y_tr, w_tr, tau_oof),
        }
        for k in UPLIFT_AT:
            row[f"uplift_at_{int(k * 100)}"] = uplift_at_k(y_te, w_te, tau_te, k)
        rows.append(row)
        print(f"[uplift] {name:22s} qini_test={row['qini_test']:.2f} "
              f"qini_oof_train={row['qini_oof_train']:.2f} "
              f"u@10={row['uplift_at_10']:.4f}")

    metrics = pd.DataFrame(rows).sort_values("qini_test", ascending=False)
    metrics.to_csv(config.REPORTS / "uplift_metrics.csv", index=False)

    # ---- bootstrap comparison of the two best learners on the test split
    best, runner_up = metrics["learner"].iloc[0], metrics["learner"].iloc[1]
    comp = bootstrap_qini_difference(
        y_te, w_te, test_scores[best], test_scores[runner_up]
    )
    print(f"[uplift] {best} vs {runner_up}: dQini={comp['delta']:.2f} "
          f"[{comp['ci_lo']:.2f}, {comp['ci_hi']:.2f}], "
          f"P(better)={comp['p_better']:.2f}")

    # ---- full-cell OOF scores from the best learner -> policy input
    tau_full = oof_scores(LEARNERS[best], X, y, w)
    scores = sub[["visit", "conversion", "spend"]].copy()
    scores.insert(0, "uplift_score", tau_full)
    scores.insert(1, "W", w)
    scores.to_csv(config.SCORES_FILE, index=False)
    print(f"[uplift] wrote OOF scores for {len(scores):,} customers "
          f"({config.SCORES_FILE.name}, learner={best})")

    dec = decile_table(y, w, tau_full)
    dec.to_csv(config.REPORTS / "uplift_deciles.csv", index=False)
    ate = float(y[w == 1].mean() - y[w == 0].mean())
    plot_qini(test_curves, y_te, w_te)
    plot_deciles(dec, ate)

    # ---- persist downsampled test Qini curves for the Streamlit app
    curve_rows = []
    for name, (frac, q) in test_curves.items():
        step = max(1, len(frac) // 300)
        for f, qq in zip(frac[::step], q[::step]):
            curve_rows.append({"learner": name, "frac": f, "qini": qq})
    pd.DataFrame(curve_rows).to_csv(config.REPORTS / "uplift_qini_curves.csv",
                                    index=False)

    md = [
        config.banner_md() + "# Part 3, uplift modelling "
        "(Women's e-mail vs control, outcome: visit)\n",
        f"Split: {1 - config.TEST_SIZE:.0%} train / {config.TEST_SIZE:.0%} test, "
        "stratified on treatment x outcome; OOF = "
        f"{config.N_FOLDS}-fold within train.\n",
        "## Learner comparison\n",
        config.md_table(metrics),
        f"\n**Bootstrap comparison ({config.B_QINI:,} resamples of the test "
        f"split)**, Qini({best}) - Qini({runner_up}): "
        f"{comp['delta']:.2f} [{comp['ci_lo']:.2f}, {comp['ci_hi']:.2f}], "
        f"P({best} better) = {comp['p_better']:.2f}.\n",
        "\n## Uplift by decile (best learner, full-cell out-of-fold scores)\n",
        config.md_table(dec),
    ]
    (config.REPORTS / "uplift_results.md").write_text("\n".join(md))

    summary = {
        "synthetic": config.is_synthetic(),
        "cell": {
            "treatment": config.UPLIFT_TREATMENT,
            "outcome": config.UPLIFT_OUTCOME,
            "n": int(len(y)),
            "treated_share": float(w.mean()),
        },
        "metrics": metrics.to_dict(orient="records"),
        "best_learner": best,
        "runner_up": runner_up,
        "qini_diff_bootstrap": comp,
        "overall_ate_visit": ate,
    }
    (config.REPORTS / "uplift_summary.json").write_text(json.dumps(summary, indent=2))
    print("[uplift] wrote metrics, deciles, curves, figures and summary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
