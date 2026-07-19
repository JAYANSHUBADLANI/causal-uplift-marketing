"""Part 4, from uplift scores to a targeting policy with a profit simulation.

The point of uplift modelling is a *decision*: who gets the e-mail? This
script converts the Part 3 out-of-fold uplift scores into expected profit
under four policies and finds the profit-maximising targeting depth k*.

Profit model (parameters in ``src/config.py``, override via CLI):

    profit(k) = m_k * ( inc_conv_rate_k * margin - cost )

where m_k is the number of customers in the top-k% by predicted uplift,
``inc_conv_rate_k`` the *incremental* conversion rate among them, ``margin``
the contribution margin per incremental conversion and ``cost`` the cost per
e-mail sent.

Because the underlying data is an RCT, ``inc_conv_rate_k`` is estimated
*empirically*, conversion rate of treated minus control customers within
the targeted group, an offline policy-value estimate with no reliance on
the model being well calibrated (the model only supplies the *ranking*).

Policies compared: e-mail everyone (blanket), e-mail no one, e-mail a random
k%, e-mail the top-k% by predicted uplift (k swept 1..100).

Outputs: reports/policy_curve.csv, reports/policy_results.md,
reports/policy_summary.json, reports/figures/fig_policy_profit.png.
"""
from __future__ import annotations

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src import config

sns.set_theme(style="whitegrid")


def compute_profit_curve(
    scores: pd.DataFrame,
    margin: float,
    cost: float,
    outcome: str = "conversion",
) -> pd.DataFrame:
    """Expected profit of 'target top-k% by uplift score' for k = 1..100.

    Also returns the random-targeting benchmark, which is linear in k
    between profit(0) = 0 and profit(100) = blanket.
    """
    df = scores.sort_values("uplift_score", ascending=False).reset_index(drop=True)
    y = df[outcome].to_numpy(float)
    w = df["W"].to_numpy(int)
    n = len(df)

    rows = []
    for k in range(1, 101):
        m = int(round(n * k / 100))
        yt, wt = y[:m], w[:m]
        n_t, n_c = wt.sum(), m - wt.sum()
        if n_t == 0 or n_c == 0:
            inc = np.nan
        else:
            inc = yt[wt == 1].mean() - yt[wt == 0].mean()
        profit = m * (inc * margin - cost) if np.isfinite(inc) else np.nan
        rows.append(
            {
                "k_pct": k,
                "n_targeted": m,
                "inc_conv_rate": inc,
                "profit_topk": profit,
            }
        )
    curve = pd.DataFrame(rows)
    blanket = curve["profit_topk"].iloc[-1]
    curve["profit_random"] = blanket * curve["k_pct"] / 100.0
    return curve


def bootstrap_profit_ci(
    scores: pd.DataFrame,
    k_star: int,
    margin: float,
    cost: float,
    n_boot: int = 500,
) -> tuple[float, float]:
    """Percentile bootstrap CI for profit at the chosen targeting depth."""
    rng = np.random.default_rng(config.SEED)
    n = len(scores)
    vals = np.empty(n_boot)
    y = scores["conversion"].to_numpy(float)
    w = scores["W"].to_numpy(int)
    s = scores["uplift_score"].to_numpy(float)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, wb, sb = y[idx], w[idx], s[idx]
        order = np.argsort(-sb)
        m = int(round(n * k_star / 100))
        top = order[:m]
        yt, wt = yb[top], wb[top]
        if wt.sum() == 0 or (m - wt.sum()) == 0:
            vals[b] = np.nan
            continue
        inc = yt[wt == 1].mean() - yt[wt == 0].mean()
        vals[b] = m * (inc * margin - cost)
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def plot_curve(curve: pd.DataFrame, k_star: int, margin: float, cost: float) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(curve["k_pct"], curve["profit_topk"], lw=2.2, color="#4C72B0",
            label="target top-k% by predicted uplift")
    ax.plot(curve["k_pct"], curve["profit_random"], lw=1.6, ls="--", color="gray",
            label="random k% (and k=100 -> blanket e-mail)")
    ax.axhline(0, color="black", lw=1, label="e-mail no one")
    best = curve.loc[curve["k_pct"] == k_star].iloc[0]
    ax.scatter([k_star], [best["profit_topk"]], color="#C44E52", zorder=5, s=70,
               label=f"optimal k* = {k_star}% (${best['profit_topk']:,.0f})")
    ax.scatter([100], [curve['profit_topk'].iloc[-1]], color="gray", zorder=5, s=50)
    ax.set_xlabel("k, share of customers e-mailed (%)")
    ax.set_ylabel("expected campaign profit ($)")
    ax.set_title(
        f"Profit vs targeting depth (margin=\\${margin:.0f}/conversion, "
        f"cost=\\${cost:.2f}/e-mail)"
    )
    ax.legend(loc="lower center")
    fig.tight_layout()
    fig.savefig(config.FIGURES / "fig_policy_profit.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=config.MARGIN_PER_CONVERSION,
                        help="contribution margin per incremental conversion ($)")
    parser.add_argument("--cost", type=float, default=config.COST_PER_EMAIL,
                        help="cost per e-mail sent ($)")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    scores = pd.read_csv(config.SCORES_FILE)
    curve = compute_profit_curve(scores, args.margin, args.cost)
    curve.to_csv(config.REPORTS / "policy_curve.csv", index=False)

    valid = curve.dropna(subset=["profit_topk"])
    k_star = int(valid.loc[valid["profit_topk"].idxmax(), "k_pct"])
    best = valid.loc[valid["k_pct"] == k_star].iloc[0]
    blanket = float(valid["profit_topk"].iloc[-1])
    random_at_kstar = float(best["profit_random"])
    ci_lo, ci_hi = bootstrap_profit_ci(scores, k_star, args.margin, args.cost)
    plot_curve(curve, k_star, args.margin, args.cost)

    policies = pd.DataFrame(
        [
            {"policy": "e-mail no one", "n_emailed": 0, "profit": 0.0},
            {
                "policy": "blanket e-mail (k=100%)",
                "n_emailed": int(curve["n_targeted"].iloc[-1]),
                "profit": blanket,
            },
            {
                "policy": f"random {k_star}%",
                "n_emailed": int(best["n_targeted"]),
                "profit": random_at_kstar,
            },
            {
                "policy": f"top {k_star}% by uplift (optimal)",
                "n_emailed": int(best["n_targeted"]),
                "profit": float(best["profit_topk"]),
            },
        ]
    )

    md = [
        config.banner_md() + "# Part 4, targeting policy and profit simulation\n",
        f"Assumptions: margin = **${args.margin:.2f}** per incremental "
        f"conversion, cost = **${args.cost:.2f}** per e-mail "
        f"(defaults in `src/config.py`; sweep them live in the Streamlit "
        "app). Incremental conversion rates are estimated from the "
        "randomised holdout inside each targeted group, so the model only "
        "supplies the *ranking*.\n",
        "## Policy comparison\n",
        config.md_table(policies, ndigits=2),
        f"\nProfit at k* bootstrap 95% CI: [${ci_lo:,.0f}, ${ci_hi:,.0f}] "
        "(500 resamples).\n",
        f"\nTargeting the top {k_star}% captures "
        f"**${best['profit_topk']:,.0f}** vs **${blanket:,.0f}** for blanket "
        f"e-mailing, `{(best['profit_topk'] / blanket - 1) * 100:+.0f}%` "
        f"profit while contacting `{k_star}%` of customers.\n",
    ]
    (config.REPORTS / "policy_results.md").write_text("\n".join(md))

    summary = {
        "synthetic": config.is_synthetic(),
        "margin_per_conversion": args.margin,
        "cost_per_email": args.cost,
        "k_star_pct": k_star,
        "profit_at_k_star": float(best["profit_topk"]),
        "profit_ci": [ci_lo, ci_hi],
        "profit_blanket": blanket,
        "profit_random_at_k_star": random_at_kstar,
        "policies": policies.to_dict(orient="records"),
    }
    (config.REPORTS / "policy_summary.json").write_text(json.dumps(summary, indent=2))

    print(policies.to_string(index=False))
    print(f"[policy] optimal k* = {k_star}%  profit=${best['profit_topk']:,.0f} "
          f"[{ci_lo:,.0f}, {ci_hi:,.0f}]  vs blanket ${blanket:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
