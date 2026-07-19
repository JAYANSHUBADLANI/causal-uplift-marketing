"""Uplift meta-learners implemented from scratch on top of LightGBM.

The goal is the conditional average treatment effect (CATE)

    tau(x) = E[Y(1) - Y(0) | X = x],

estimated here without any causal-ML library so the logic of each learner is
explicit. All learners share the same lightweight LightGBM base models and
the same interface:

    learner.fit(X, w, y).predict_uplift(X_new)  ->  tau_hat

where ``w`` is the 0/1 treatment indicator and ``y`` the binary outcome.

Implemented learners
--------------------
S-learner
    One model f(X, W). tau(x) = f(x, 1) - f(x, 0). Simple, but the treatment
    indicator competes with every covariate for splits, so regularisation
    can shrink the effect towards zero.
T-learner
    Separate models f1 on the treated, f0 on the controls.
    tau(x) = f1(x) - f0(x). No shared strength between arms.
X-learner (Kunzel et al., 2019)
    Stage 1: T-learner. Stage 2: regress the *imputed* individual effects
    D1 = Y - f0(X) (treated) and D0 = f1(X) - Y (controls) on X, giving
    tau1 and tau0. Combine: tau(x) = e(x)*tau0(x) + (1 - e(x))*tau1(x).
    In an RCT the propensity e(x) is a known constant, so the empirical
    treated share is used by default.
Class transformation (Jaskowski & Jaroszewicz, 2012)
    Define Z = W*Y + (1-W)*(1-Y). When P(W=1) = 1/2,
    tau(x) = 2*P(Z=1 | x) - 1, so a single classifier on Z estimates uplift
    directly. The constructor enforces the balanced-assignment requirement
    up to a tolerance.
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor

from src import config

# Shared base-model hyperparameters: small trees + strong minimum leaf sizes,
# because uplift signal is weak relative to outcome signal.
LGBM_PARAMS = dict(
    n_estimators=150,
    learning_rate=0.08,
    num_leaves=15,
    min_child_samples=100,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=config.SEED,
    verbose=-1,
)


def _as_arrays(X, w, y=None):
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=int)
    if y is None:
        return X, w
    return X, w, np.asarray(y, dtype=float)


class SLearner:
    """Single-model learner: tau(x) = f(x, W=1) - f(x, W=0)."""

    def __init__(self, **params):
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, X, w, y):
        X, w, y = _as_arrays(X, w, y)
        self.model_ = LGBMClassifier(**self.params).fit(np.column_stack([X, w]), y)
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        ones = np.ones((len(X), 1))
        p1 = self.model_.predict_proba(np.hstack([X, ones]))[:, 1]
        p0 = self.model_.predict_proba(np.hstack([X, 0 * ones]))[:, 1]
        return p1 - p0


class TLearner:
    """Two-model learner: tau(x) = f1(x) - f0(x)."""

    def __init__(self, **params):
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, X, w, y):
        X, w, y = _as_arrays(X, w, y)
        self.model1_ = LGBMClassifier(**self.params).fit(X[w == 1], y[w == 1])
        self.model0_ = LGBMClassifier(**self.params).fit(X[w == 0], y[w == 0])
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        return (
            self.model1_.predict_proba(X)[:, 1] - self.model0_.predict_proba(X)[:, 1]
        )


class XLearner:
    """X-learner with imputed-effect regressions and propensity weighting.

    Parameters
    ----------
    propensity:
        P(W=1). If None, the empirical treated share is used, appropriate
        for a randomised experiment where assignment is independent of X.
    """

    def __init__(self, propensity: float | None = None, **params):
        self.propensity = propensity
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, X, w, y):
        X, w, y = _as_arrays(X, w, y)
        # stage 1, outcome models per arm
        self.model1_ = LGBMClassifier(**self.params).fit(X[w == 1], y[w == 1])
        self.model0_ = LGBMClassifier(**self.params).fit(X[w == 0], y[w == 0])
        # stage 2, imputed individual effects
        d1 = y[w == 1] - self.model0_.predict_proba(X[w == 1])[:, 1]
        d0 = self.model1_.predict_proba(X[w == 0])[:, 1] - y[w == 0]
        self.tau1_ = LGBMRegressor(**self.params).fit(X[w == 1], d1)
        self.tau0_ = LGBMRegressor(**self.params).fit(X[w == 0], d0)
        self.e_ = float(w.mean()) if self.propensity is None else self.propensity
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        # weight the estimate built on the *other* arm's imputations by the
        # probability of being in this arm (Kunzel et al., eq. 9)
        return self.e_ * self.tau0_.predict(X) + (1 - self.e_) * self.tau1_.predict(X)


class ClassTransformation:
    """Class-variable transformation: Z = W*Y + (1-W)*(1-Y).

    P(Z=1|x) = e*P(Y=1|x,W=1) + (1-e)*P(Y=0|x,W=0); when e = 1/2 this
    collapses to tau(x) = 2*P(Z=1|x) - 1. Requires (near-)balanced
    assignment, which holds in this experiment.
    """

    def __init__(self, tol: float = 0.05, **params):
        self.tol = tol
        self.params = {**LGBM_PARAMS, **params}

    def fit(self, X, w, y):
        X, w, y = _as_arrays(X, w, y)
        e = w.mean()
        if abs(e - 0.5) > self.tol:
            raise ValueError(
                f"class transformation requires P(W=1) close to 0.5, got {e:.3f}"
            )
        z = w * y + (1 - w) * (1 - y)
        self.model_ = LGBMClassifier(**self.params).fit(X, z)
        return self

    def predict_uplift(self, X):
        X = np.asarray(X, dtype=float)
        return 2 * self.model_.predict_proba(X)[:, 1] - 1
