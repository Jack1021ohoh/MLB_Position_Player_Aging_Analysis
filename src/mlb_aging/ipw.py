"""Survivorship-bias correction by Inverse Probability Weighting.

The ``qual=100`` PA threshold means declining players vanish from the data
before they formally retire, so the sample at older ages is positively
selected. A survival model estimates P(player appears next season) and its
inverse upweights the player-seasons least likely to be observed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from mlb_aging.metrics import MetricSpec

#: Floor on the survival probability, capping any single weight at 10x.
P_SURVIVE_FLOOR = 0.1
DEFAULT_HOLDOUT_YEARS = 5


def retirement_features(weight_col: str, perf_col: str | None) -> list[str]:
    """Feature list for the survival model.

    ``perf_col`` (a lagged metric) lets the model separate genuine decline from
    low playing time that means something else -- a defensive specialist who
    rarely bats, say. It matters most for G-weighted metrics; for PA-weighted
    ones, plate appearances already proxy hitting quality.
    """
    features = ["Age", weight_col, "experience"]
    if perf_col is not None:
        features.append(perf_col)
    return features


def add_survived_label(data: pd.DataFrame) -> pd.DataFrame:
    """Label each row with whether the player appears again the next season."""
    df = data.copy().reset_index(drop=True)
    player_season_set = set(zip(df["IDfg"], df["Season"]))
    df["survived"] = [
        1 if (pid, season + 1) in player_season_set else 0
        for pid, season in zip(df["IDfg"], df["Season"])
    ]
    return df


@dataclass(frozen=True)
class RetirementModel:
    """A fitted survival model and the scaler its features expect."""

    model: LogisticRegression
    scaler: StandardScaler
    weight_col: str
    perf_col: str | None

    def predict_survival(self, data: pd.DataFrame) -> np.ndarray:
        features = retirement_features(self.weight_col, self.perf_col)
        x = self.scaler.transform(data[features].values)
        return self.model.predict_proba(x)[:, 1]


def fit_retirement_model(
    data: pd.DataFrame, weight_col: str = "PA", perf_col: str | None = None
) -> RetirementModel:
    """Fit P(appears next season) on standardized features.

    The final season is dropped: its label is unobservable, since there is no
    following season in the data to check.
    """
    df = add_survived_label(data)
    fit_df = df[df["Season"] < df["Season"].max()]

    features = retirement_features(weight_col, perf_col)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(fit_df[features].values)

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_scaled, fit_df["survived"].values, sample_weight=fit_df[weight_col].values)

    return RetirementModel(lr, scaler, weight_col, perf_col)


@dataclass(frozen=True)
class RetirementDiagnostics:
    """Hold-out discrimination and calibration for a survival model."""

    auc: float
    brier: float
    naive_brier: float
    n_eval: int
    holdout_start: int
    holdout_end: int
    y_true: np.ndarray
    y_prob: np.ndarray

    def summary(self) -> str:
        return (
            f"Hold-out seasons: {self.holdout_start}–{self.holdout_end}  |  n={self.n_eval}\n"
            f"  AUC-ROC:     {self.auc:.4f}   (0.5 = random, 1.0 = perfect)\n"
            f"  Brier score: {self.brier:.4f}  (lower = better; "
            f"naive baseline = {self.naive_brier:.4f})"
        )


def evaluate_retirement_model(
    data: pd.DataFrame,
    weight_col: str = "PA",
    perf_col: str | None = None,
    holdout_years: int = DEFAULT_HOLDOUT_YEARS,
) -> RetirementDiagnostics:
    """Score the survival model on a temporal hold-out of the last N seasons.

    Returns the diagnostics rather than printing and plotting them, so callers
    choose their own presentation.
    """
    df = add_survived_label(data)
    df = df[df["Season"] < df["Season"].max()]

    cutoff = df["Season"].max() - holdout_years
    train_df = df[df["Season"] <= cutoff]
    eval_df = df[df["Season"] > cutoff]

    features = retirement_features(weight_col, perf_col)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[features].values)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(x_train, train_df["survived"].values, sample_weight=train_df[weight_col].values)

    x_eval = scaler.transform(eval_df[features].values)
    y_true = eval_df["survived"].values
    y_prob = lr.predict_proba(x_eval)[:, 1]

    return RetirementDiagnostics(
        auc=roc_auc_score(y_true, y_prob),
        brier=brier_score_loss(y_true, y_prob),
        naive_brier=float(np.mean(1 - y_true) * np.mean(y_true)),
        n_eval=len(eval_df),
        holdout_start=int(cutoff + 1),
        holdout_end=int(df["Season"].max()),
        y_true=y_true,
        y_prob=y_prob,
    )


def compute_ipw_weights(data: pd.DataFrame, model: RetirementModel) -> pd.DataFrame:
    """Add survival probabilities and IPW sample weights.

    Adds ``p_survive``, ``ipw_weight`` (its clipped inverse, normalised to mean
    1) and ``ipw_final_weight`` -- the playing-time weight scaled by the IPW
    term, which is what the corrected GAM is fitted on.
    """
    df = data.copy().reset_index(drop=True)
    p_survive = model.predict_survival(df)

    ipw = 1.0 / np.clip(p_survive, P_SURVIVE_FLOOR, 1.0)
    ipw = ipw / ipw.mean()

    df["p_survive"] = p_survive
    df["ipw_weight"] = ipw
    df["ipw_final_weight"] = df[model.weight_col] * ipw
    return df


def fit_ipw_weights(data: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    """Fit the survival model for ``spec`` and return the weighted frame."""
    model = fit_retirement_model(data, weight_col=spec.weight_col, perf_col=spec.lag_col)
    return compute_ipw_weights(data, model)
