"""Streamlit dashboard for the causal-uplift-marketing project.

Run with:  streamlit run app.py

The app renders the artefacts produced by the pipeline (Parts 1-3) and
recomputes the Part 4 profit curve live from the saved out-of-fold uplift
scores, so the margin / cost sliders respond instantly without retraining.
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config
from src.policy import compute_profit_curve

st.set_page_config(page_title="Causal Inference & Uplift Modeling", layout="wide")


# ------------------------------------------------------------------ helpers
def load_csv(name: str) -> pd.DataFrame | None:
    path = config.REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def load_json(name: str) -> dict | None:
    path = config.REPORTS / name
    return json.loads(path.read_text()) if path.exists() else None


def show_fig(name: str, caption: str = "") -> None:
    path = config.FIGURES / name
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        st.info(f"Figure {name} not found, run the pipeline first.")


MISSING = (
    "Artefacts not found. Run the pipeline first:\n\n"
    "```\npython -m src.data_ingestion\npython -m src.rct_analysis\n"
    "python -m src.observational\npython -m src.uplift.run_uplift\n"
    "python -m src.policy\n```"
)

# -------------------------------------------------------------------- header
st.title("Causal Inference & Uplift Modeling, who should we target?")
st.caption(
    "Hillstrom e-mail RCT (64,000 customers) → randomised ATEs → "
    "observational methods vs ground truth → uplift meta-learners → "
    "profit-optimal targeting."
)
if config.is_synthetic():
    st.error(config.SYNTHETIC_BANNER)

tab_rct, tab_obs, tab_uplift, tab_policy = st.tabs(
    ["1 · RCT results", "2 · Observational demo", "3 · Uplift models", "4 · Targeting policy"]
)

# ---------------------------------------------------------------- tab 1: RCT
with tab_rct:
    rct = load_json("rct_summary.json")
    ate = load_csv("rct_ate.csv")
    if rct is None or ate is None:
        st.warning(MISSING)
    else:
        srm = rct["srm"]
        c1, c2, c3 = st.columns(3)
        c1.metric("SRM chi-square p-value", f"{srm['p_value']:.3f}")
        c2.metric("Max |SMD| across arms", f"{rct['max_abs_smd']:.4f}")
        c3.metric(
            "Benchmark ATE (womens → visit)",
            f"{rct['benchmark']['ate'] * 100:.2f} pp",
        )
        st.subheader("Average treatment effects vs control")
        cols = [
            "treatment", "outcome", "mean_treat", "mean_ctrl", "ate",
            "rel_lift_pct", "p_raw", "p_holm", "ci_boot_lo", "ci_boot_hi",
            "significant_5pct",
        ]
        st.dataframe(ate[cols], hide_index=True, use_container_width=True)
        st.caption(
            "Welch t-tests with percentile-bootstrap 95% CIs; Holm correction "
            "across the 2 treatments × 3 outcomes family."
        )
        left, right = st.columns(2)
        with left:
            show_fig("fig_rct_ates.png")
        with right:
            show_fig("fig_rct_segments.png")

# ------------------------------------------------------- tab 2: observational
with tab_obs:
    obs = load_csv("obs_methods.csv")
    obs_sum = load_json("obs_summary.json")
    if obs is None or obs_sum is None:
        st.warning(MISSING)
    else:
        st.markdown(
            "Confounding was **manufactured** by biased subsampling of the "
            "randomised data (engaged customers over-represented among the "
            "treated), so the RCT ATE remains the known ground truth."
        )
        st.subheader("Method comparison vs RCT truth")
        st.dataframe(obs, hide_index=True, use_container_width=True)
        show_fig("fig_obs_methods.png")
        left, mid = st.columns(2)
        with left:
            show_fig("fig_obs_overlap.png")
            show_fig("fig_obs_weights.png")
        with mid:
            show_fig("fig_obs_love.png")

# -------------------------------------------------------------- tab 3: uplift
with tab_uplift:
    met = load_csv("uplift_metrics.csv")
    dec = load_csv("uplift_deciles.csv")
    usum = load_json("uplift_summary.json")
    if met is None or dec is None or usum is None:
        st.warning(MISSING)
    else:
        comp = usum["qini_diff_bootstrap"]
        st.markdown(
            f"Cell: **Women's e-mail vs control**, outcome **visit**. Best "
            f"learner: **{usum['best_learner']}**. Bootstrap Qini difference "
            f"vs {usum['runner_up']}: {comp['delta']:.1f} "
            f"[{comp['ci_lo']:.1f}, {comp['ci_hi']:.1f}] "
            f"(P(better) = {comp['p_better']:.2f})."
        )
        st.subheader("Learner comparison")
        st.dataframe(met, hide_index=True, use_container_width=True)
        left, right = st.columns(2)
        with left:
            show_fig("fig_qini_curves.png")
        with right:
            show_fig("fig_uplift_deciles.png")
        st.subheader("Uplift by decile (out-of-fold)")
        st.dataframe(dec, hide_index=True, use_container_width=True)

# -------------------------------------------------------------- tab 4: policy
with tab_policy:
    if not config.SCORES_FILE.exists():
        st.warning(MISSING)
    else:
        scores = pd.read_csv(config.SCORES_FILE)
        st.markdown(
            "Profit is estimated **empirically from the randomised holdout** "
            "inside each targeted group, the model only supplies the ranking. "
            "Move the sliders; the curve recomputes instantly."
        )
        c1, c2 = st.columns(2)
        margin = c1.slider("Margin per incremental conversion ($)", 1.0, 100.0,
                           float(config.MARGIN_PER_CONVERSION), 1.0)
        cost = c2.slider("Cost per e-mail ($)", 0.01, 1.00,
                         float(config.COST_PER_EMAIL), 0.01)
        curve = compute_profit_curve(scores, margin, cost)
        valid = curve.dropna(subset=["profit_topk"])
        k_star = int(valid.loc[valid["profit_topk"].idxmax(), "k_pct"])
        best = valid.loc[valid["k_pct"] == k_star].iloc[0]
        blanket = float(valid["profit_topk"].iloc[-1])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Optimal targeting depth k*", f"{k_star}%")
        m2.metric("Profit at k*", f"${best['profit_topk']:,.0f}")
        m3.metric("Blanket e-mail profit", f"${blanket:,.0f}")
        m4.metric(
            "Uplift targeting vs blanket",
            f"${best['profit_topk'] - blanket:,.0f}",
        )

        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.plot(curve["k_pct"], curve["profit_topk"], lw=2.2, color="#4C72B0",
                label="top-k% by predicted uplift")
        ax.plot(curve["k_pct"], curve["profit_random"], lw=1.5, ls="--",
                color="gray", label="random k%")
        ax.axhline(0, color="black", lw=1, label="e-mail no one")
        ax.scatter([k_star], [best["profit_topk"]], color="#C44E52", s=60,
                   zorder=5, label=f"k* = {k_star}%")
        ax.set_xlabel("share of customers e-mailed (%)")
        ax.set_ylabel("expected profit ($)")
        ax.legend(loc="lower center")
        st.pyplot(fig, use_container_width=True)
        st.caption(
            "Blanket e-mailing can lose money while targeted e-mailing is "
            "profitable, the entire value of uplift modelling is choosing "
            "*who*, not just *whether*, to contact."
        )
