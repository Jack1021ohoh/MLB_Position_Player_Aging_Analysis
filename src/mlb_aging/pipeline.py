"""End-to-end aging-curve runs: load, fit, trace, score."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mlb_aging.dataset import DEFAULT_DATA_DIR, build_training_frame, load_data
from mlb_aging.evaluate import evaluate
from mlb_aging.features import generate_test_data
from mlb_aging.gam import AgingCurve, aging_curve, fit_gam
from mlb_aging.ipw import (
    RetirementDiagnostics,
    evaluate_retirement_model,
    fit_ipw_weights,
)
from mlb_aging.metrics import MetricSpec


@dataclass
class AgingResult:
    """One fitted aging curve and its test score."""

    spec: MetricSpec
    curve: AgingCurve
    test_mae: float
    n_train: int
    n_test: int

    @property
    def peak_age(self) -> float:
        return self.curve.peak_age

    @property
    def peak_value(self) -> float:
        return self.curve.peak_value

    def summary(self) -> str:
        return (
            f"{self.spec.name}: peak age {self.peak_age:.2f}  |  "
            f"peak value {self.peak_value:.4f}  |  "
            f"test MAE {self.test_mae:.4f}  "
            f"(n_train={self.n_train}, n_test={self.n_test})"
        )


def cohort_ids(
    cohort_metric: MetricSpec, threshold: float, data_dir: Path | str = DEFAULT_DATA_DIR
) -> set:
    """Player ids whose career mean of ``cohort_metric`` exceeds ``threshold``.

    Lets a cohort defined by one metric (career WAR, say) be carried into
    another metric's model -- how do elite players age defensively? The
    notebook did this by reusing a positional boolean mask across frames,
    which quietly assumed the two metrics produced identically ordered rows.
    Selecting on player id removes that assumption.
    """
    train_data, _ = load_data(cohort_metric, data_dir=data_dir)
    frame = build_training_frame(train_data, cohort_metric, elite_threshold=threshold)
    return set(frame["IDfg"].unique())


def run_metric(
    spec: MetricSpec,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    ipw: bool = False,
    elite_threshold: float | None = None,
    elite_test: bool = False,
    cohort_metric: MetricSpec | None = None,
    curve_reference: float | str | None = "__spec__",
    test_threshold: float | None = None,
) -> AgingResult:
    """Fit one metric and score it on the held-out seasons.

    Parameters
    ----------
    ipw
        Fit with survivorship-corrected sample weights instead of raw playing
        time.
    elite_threshold
        Restrict *training* to players whose career mean exceeds this.
    elite_test
        Also restrict the test set to those players, matching the top-player
        notebook's "strong player" scoring.
    cohort_metric
        Apply ``elite_threshold`` to this metric's career mean instead of
        ``spec``'s, so the cohort is defined by one metric and modelled on
        another.
    curve_reference
        Override the career-mean value pinned when tracing the curve.
    test_threshold
        Restrict the *test* set independently of training, so an all-player
        model can be scored on the elite subset. ``elite_test`` is the special
        case where this equals ``elite_threshold``.
    """
    train_data, test_data = load_data(spec, data_dir=data_dir)

    if cohort_metric is not None and elite_threshold is not None:
        ids = cohort_ids(cohort_metric, elite_threshold, data_dir=data_dir)
        train_df = build_training_frame(train_data, spec)
        train_df = train_df.loc[train_df["IDfg"].isin(ids), :]
    else:
        train_df = build_training_frame(train_data, spec, elite_threshold=elite_threshold)

    weights = None
    if ipw:
        weights = fit_ipw_weights(train_df, spec)["ipw_final_weight"].values

    gam = fit_gam(train_df, spec, weights=weights)
    curve = aging_curve(gam, train_df, spec, curve_reference=curve_reference)

    test_df = generate_test_data(train_df, test_data, spec.target_col)

    threshold = test_threshold
    if threshold is None and elite_test and elite_threshold is not None and cohort_metric is None:
        threshold = elite_threshold
    if threshold is not None:
        test_df = test_df.loc[test_df[spec.career_mean_col] > threshold, :]

    return AgingResult(
        spec=spec,
        curve=curve,
        test_mae=evaluate(test_df, spec, gam),
        n_train=len(train_df),
        n_test=len(test_df),
    )


@dataclass
class IPWComparison:
    """An uncorrected and a survivorship-corrected fit of the same metric."""

    spec: MetricSpec
    baseline: AgingResult
    corrected: AgingResult
    survival_baseline: RetirementDiagnostics
    survival_with_perf: RetirementDiagnostics
    p_survive_by_age: pd.Series

    @property
    def mae_improvement(self) -> float:
        """Fractional reduction in test MAE, positive when IPW helps."""
        return 1.0 - self.corrected.test_mae / self.baseline.test_mae

    @property
    def auc_gain(self) -> float:
        """AUC added by giving the survival model the lagged metric."""
        return self.survival_with_perf.auc - self.survival_baseline.auc

    def summary(self) -> str:
        return (
            f"{self.spec.name}  (fit weight {self.spec.weight_col})\n"
            f"  Survival AUC — without {self.spec.lag_col}: "
            f"{self.survival_baseline.auc:.4f}  |  with: "
            f"{self.survival_with_perf.auc:.4f}  ({self.auc_gain:+.4f})\n"
            f"  Peak age     — original: {self.baseline.peak_age:.1f}  |  "
            f"IPW: {self.corrected.peak_age:.1f}\n"
            f"  Test MAE     — original: {self.baseline.test_mae:.4f}  |  "
            f"IPW: {self.corrected.test_mae:.4f}  "
            f"({self.mae_improvement:+.1%})"
        )


def compare_ipw(
    spec: MetricSpec, data_dir: Path | str = DEFAULT_DATA_DIR
) -> IPWComparison:
    """Fit ``spec`` with and without the survivorship correction.

    Both arms are scored identically, so the comparison is unaffected by which
    weight column ``evaluate`` uses.
    """
    train_data, _ = load_data(spec, data_dir=data_dir)
    frame = build_training_frame(train_data, spec)

    return IPWComparison(
        spec=spec,
        baseline=run_metric(spec, data_dir=data_dir),
        corrected=run_metric(spec, data_dir=data_dir, ipw=True),
        survival_baseline=evaluate_retirement_model(frame, weight_col=spec.weight_col),
        survival_with_perf=evaluate_retirement_model(
            frame, weight_col=spec.weight_col, perf_col=spec.lag_col
        ),
        p_survive_by_age=fit_ipw_weights(frame, spec).groupby("Age")["p_survive"].mean(),
    )
