"""Naive baselines the GAM is measured against.

Without a baseline, "the GAM improves accuracy" has no denominator. These are
the alternatives a reader would reach for first, scored on exactly the frame
:func:`~mlb_aging.pipeline.run_metric` scores, with the same sample weights, so
the numbers are directly comparable.

The ladder, weakest first:

``persistence``
    Predict this season from last season, ignoring age entirely. The floor --
    and a surprisingly strong one, since season-to-season variance dominates.
``delta_curve``
    The classic delta method used as ``delta_method.ipynb`` used it: a single
    population curve value per age, the same prediction for every player of
    that age. Not personalized.
``delta_lag``
    The delta method applied the way it is used in practice: take the player's
    prior season and add the mean change for their age. This is the honest
    comparison for the GAM, which also sees the prior season.

The delta curve itself is the standard construction -- per-age mean
year-over-year change, weighted by the harmonic mean of the two seasons'
playing time, then accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import mean_absolute_error

from mlb_aging.dataset import DEFAULT_DATA_DIR, build_training_frame, load_data
from mlb_aging.features import generate_test_data
from mlb_aging.metrics import MetricSpec


@dataclass(frozen=True)
class DeltaCurve:
    """Per-age mean year-over-year change, and the curve it accumulates to."""

    #: Age -> weighted mean change from the previous observed season.
    deltas: pd.Series
    #: Rows that contributed a delta.
    n_pairs: int

    @property
    def curve(self) -> pd.Series:
        """Cumulative sum of the deltas -- the aging curve, up to a constant."""
        return self.deltas.cumsum()

    @property
    def peak_age(self) -> float:
        return float(self.curve.idxmax())

    def anchored(self, data: pd.DataFrame, spec: MetricSpec) -> pd.Series:
        """The curve shifted so its weighted mean matches ``data``'s.

        The delta method recovers a *shape*; its level is arbitrary because the
        cumulative sum starts from zero. Anchoring on the training mean is what
        the original notebook approximated by adding 100 to the wRC+ curve.
        """
        curve = self.curve
        weights = data.groupby("Age")[spec.weight_col].sum().reindex(curve.index).fillna(0.0)
        offset = np.average(data[spec.target_col], weights=data[spec.weight_col])
        return curve - np.average(curve, weights=weights) + offset


def delta_curve(train_df: pd.DataFrame, spec: MetricSpec) -> DeltaCurve:
    """Fit the delta method on ``train_df``.

    Note the first season each player retains in the frame contributes no
    delta: its predecessor was dropped by :func:`~mlb_aging.features.add_lag`,
    so the earlier season's playing time is not available to weight the pair.
    """
    df = train_df.copy()
    grouped = df.groupby("IDfg")

    df["weight_prev"] = grouped[spec.weight_col].shift(1)
    df["change"] = grouped[spec.target_col].diff()
    df["weight_hmean"] = hmean(
        df[[spec.weight_col, "weight_prev"]].clip(lower=1e-9), axis=1
    )

    pairs = df.dropna(subset=["change", "weight_prev"])
    weighted_mean = lambda x: np.average(  # noqa: E731
        x, weights=pairs.loc[x.index, "weight_hmean"]
    )
    return DeltaCurve(
        deltas=pairs.groupby("Age")["change"].agg(weighted_mean),
        n_pairs=len(pairs),
    )


def predict_persistence(test_df: pd.DataFrame, spec: MetricSpec) -> np.ndarray:
    """Last season, unchanged."""
    return test_df[spec.lag_col].values


def predict_delta_curve(
    test_df: pd.DataFrame, spec: MetricSpec, curve: DeltaCurve, train_df: pd.DataFrame
) -> np.ndarray:
    """The population curve's value at the player's age -- not personalized."""
    anchored = curve.anchored(train_df, spec)
    return test_df["Age"].map(anchored).fillna(anchored.mean()).values


def predict_delta_lag(
    test_df: pd.DataFrame, spec: MetricSpec, curve: DeltaCurve
) -> np.ndarray:
    """Last season plus the mean change for the player's age."""
    step = test_df["Age"].map(curve.deltas).fillna(0.0).values
    return test_df[spec.lag_col].values + step


@dataclass(frozen=True)
class BaselineResult:
    """One baseline's score on the shared test frame."""

    name: str
    spec: MetricSpec
    test_mae: float
    test_mae_fit_weighted: float
    n_test: int

    def summary(self) -> str:
        return f"{self.name:14s} test MAE {self.test_mae:9.4f}"


def score(
    name: str, spec: MetricSpec, test_df: pd.DataFrame, predictions: np.ndarray
) -> BaselineResult:
    """Score predictions under both the eval weight and the fit weight."""
    actual = test_df[spec.target_col].values
    return BaselineResult(
        name=name,
        spec=spec,
        test_mae=mean_absolute_error(
            actual, predictions, sample_weight=test_df[spec.eval_weight_col].values
        ),
        test_mae_fit_weighted=mean_absolute_error(
            actual, predictions, sample_weight=test_df[spec.weight_col].values
        ),
        n_test=len(test_df),
    )


def run_baselines(
    spec: MetricSpec, data_dir: Path | str = DEFAULT_DATA_DIR
) -> tuple[dict[str, BaselineResult], DeltaCurve]:
    """Score every baseline for ``spec`` on the GAM's own test frame."""
    train_data, test_data = load_data(spec, data_dir=data_dir)
    train_df = build_training_frame(train_data, spec)
    test_df = generate_test_data(train_df, test_data, spec.target_col)

    curve = delta_curve(train_df, spec)
    predictions = {
        "persistence": predict_persistence(test_df, spec),
        "delta_curve": predict_delta_curve(test_df, spec, curve, train_df),
        "delta_lag": predict_delta_lag(test_df, spec, curve),
    }
    results = {
        name: score(name, spec, test_df, preds) for name, preds in predictions.items()
    }
    return results, curve
