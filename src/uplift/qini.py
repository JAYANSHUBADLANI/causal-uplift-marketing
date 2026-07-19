"""Qini curve, Qini coefficient, uplift@k and decile diagnostics, from scratch.

Definitions (Radcliffe, 2007). Rank customers by predicted uplift,
descending. After targeting the first n customers, let n_t(n) / n_c(n) be
the number of treated / control customers among them and Y_t(n) / Y_c(n)
their cumulative positive outcomes. The Qini curve is

    Q(n) = Y_t(n) - Y_c(n) * n_t(n) / n_c(n),

the estimated number of *incremental* positive outcomes among the first n
customers if the whole targeted group had been treated. Random targeting
traces the straight line from (0, 0) to (N, Q(N)).

The **Qini coefficient** reported here is the area between the model's Qini
curve and that random-targeting line, with the x-axis expressed as the
fraction targeted (so units are "incremental visits x fraction"; higher is
better, 0 means no better than random).

Ties are broken by a seeded random permutation before the (stable) sort so
results are reproducible and not order-dependent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _ranked(y, w, score, seed: int = config.SEED):
    """Outcomes/treatments sorted by descending score, random tie-break."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    order = np.argsort(-score[perm], kind="stable")
    return y[perm][order], w[perm][order]


def qini_curve(y, w, score, seed: int = config.SEED):
    """Return (fraction_targeted, Q) arrays, both starting at 0."""
    yy, ww = _ranked(y, w, score, seed)
    n = len(yy)
    cum_yt = np.cumsum(yy * ww)
    cum_yc = np.cumsum(yy * (1 - ww))
    n_t = np.cumsum(ww)
    n_c = np.cumsum(1 - ww)
    ratio = np.divide(n_t, n_c, out=np.zeros(n, dtype=float), where=n_c > 0)
    q = cum_yt - cum_yc * ratio
    frac = np.arange(1, n + 1) / n
    return np.concatenate([[0.0], frac]), np.concatenate([[0.0], q])


def qini_coefficient(y, w, score, seed: int = config.SEED) -> float:
    """Area between the Qini curve and the random-targeting diagonal."""
    frac, q = qini_curve(y, w, score, seed)
    area_model = np.trapezoid(q, frac)
    area_random = q[-1] / 2.0  # triangle under the line (0,0) -> (1, Q(N))
    return float(area_model - area_random)


def uplift_at_k(y, w, score, k: float, seed: int = config.SEED) -> float:
    """Difference in outcome rates (treated - control) within the top-k share."""
    yy, ww = _ranked(y, w, score, seed)
    m = max(1, int(np.ceil(k * len(yy))))
    top_y, top_w = yy[:m], ww[:m]
    if top_w.sum() == 0 or (1 - top_w).sum() == 0:
        return float("nan")
    return float(top_y[top_w == 1].mean() - top_y[top_w == 0].mean())


def decile_table(y, w, score, seed: int = config.SEED) -> pd.DataFrame:
    """Uplift by score decile (decile 1 = highest predicted uplift)."""
    yy, ww = _ranked(y, w, score, seed)
    ss = np.sort(np.asarray(score, dtype=float))[::-1]
    edges = np.array_split(np.arange(len(yy)), 10)
    rows = []
    cum_inc = 0.0
    for d, idx in enumerate(edges, start=1):
        ty, tw, ts = yy[idx], ww[idx], ss[idx]
        n_t, n_c = int(tw.sum()), int((1 - tw).sum())
        r_t = float(ty[tw == 1].mean()) if n_t else float("nan")
        r_c = float(ty[tw == 0].mean()) if n_c else float("nan")
        uplift = r_t - r_c
        cum_inc += uplift * len(idx)
        rows.append(
            {
                "decile": d,
                "n": len(idx),
                "n_treat": n_t,
                "n_ctrl": n_c,
                "mean_score": float(ts.mean()),
                "visit_rate_treat": r_t,
                "visit_rate_ctrl": r_c,
                "uplift": uplift,
                "cum_incremental_visits": cum_inc,
            }
        )
    return pd.DataFrame(rows)
