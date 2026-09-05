"""Presentation helpers for the notebooks.

The analysis modules deliberately return data rather than drawing, so plotting
lives here and callers choose how to show it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mlb_aging.gam import AgingCurve
from mlb_aging.metrics import MetricSpec

#: Age ticks used across every curve plot, matching the notebooks.
AGE_TICKS = np.arange(20, 45, 5)


def plot_aging_curve(
    curve: AgingCurve,
    spec: MetricSpec | None = None,
    ax: plt.Axes | None = None,
    label: str | None = None,
    title: str | None = None,
    **kwargs,
) -> plt.Axes:
    """Draw one traced curve, marking the peak."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(curve.ages, curve.predictions, label=label, linewidth=2, **kwargs)
    ax.axvline(curve.peak_age, color="grey", linestyle=":", linewidth=1)

    ax.set_xlabel("Age")
    if spec is not None:
        ax.set_ylabel(spec.target_col)
    ax.set_title(title or (f"{spec.name} aging curve" if spec else "Aging curve"))
    ax.set_xticks(AGE_TICKS)
    ax.grid(True)
    if label:
        ax.legend()
    return ax


def plot_curve_comparison(
    baseline: AgingCurve,
    corrected: AgingCurve,
    spec: MetricSpec,
    labels: tuple[str, str] = ("Original", "IPW-Adjusted"),
) -> plt.Figure:
    """Two panels: both curves overlaid, and their difference.

    The right panel is where the survivorship correction shows up -- a
    downward adjustment at the older ages means the uncorrected curve
    overstated how well survivors hold their value.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(baseline.ages, baseline.predictions,
                 label=labels[0], color="steelblue", linewidth=2)
    axes[0].plot(corrected.ages, corrected.predictions,
                 label=labels[1], color="tomato", linewidth=2, linestyle="--")
    axes[0].set_title(f"{spec.name} aging curve: {labels[0]} vs {labels[1]}")
    axes[0].set_ylabel(spec.target_col)

    axes[1].plot(corrected.ages, corrected.predictions - baseline.predictions,
                 color="purple", linewidth=2)
    axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[1].set_title(f"Adjustment effect ({labels[1]} - {labels[0]})")

    for ax in axes:
        ax.set_xlabel("Age")
        ax.set_xticks(AGE_TICKS)
        ax.grid(True)
    axes[0].legend()

    fig.tight_layout()
    return fig


def plot_residuals_by_age(
    *profiles,
    ax: plt.Axes | None = None,
    title: str | None = None,
) -> plt.Axes:
    """Plot mean held-out residual against age bin for one or more profiles.

    A well-shaped curve gives a flat line on zero. A downward slope means the
    curve increasingly over-predicts older players -- the decline phase fitted
    too shallow. Point size tracks the number of rows behind each bin, since
    the oldest bins are thin.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4.5))

    for profile in profiles:
        table = profile.table
        centres = [interval.mid for interval in table.index]
        ax.plot(centres, table["mean_resid"], marker="o", label=profile.label or None)
        ax.scatter(centres, table["mean_resid"], s=table["n"] / 3, alpha=0.25)

    ax.axhline(0.0, color="black", lw=1, ls="--", zorder=0)
    ax.set_xlabel("Age")
    ax.set_ylabel("mean residual (actual − predicted)")
    spec = profiles[0].spec
    ax.set_title(title or f"{spec.name}: held-out residual by age")
    if any(p.label for p in profiles):
        ax.legend()
    return ax


def plot_curves_against_truth(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    truth_ages: np.ndarray,
    truth_values: np.ndarray,
    title: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Estimated curves over the known truth, all centred on their own mean.

    Every estimated level is arbitrary -- ``aging_curve`` pins the career mean at a
    reference and the delta method's cumulative sum starts from zero -- so only shape is
    comparable, and centring is what makes the picture honest rather than flattering.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    centred_truth = truth_values - truth_values.mean()
    ax.plot(truth_ages, centred_truth, color="black", linewidth=3, label="truth", zorder=5)
    ax.axvline(truth_ages[np.argmax(truth_values)], color="black", linestyle=":", linewidth=1)

    # Distinct dash patterns, because near-identical arms would otherwise hide each other:
    # two curves agreeing to within a line width is a *result*, and it has to stay visible.
    # Widths taper so an overlapping later curve reads as a dash over a thicker solid line.
    styles = [("-", 3.0), ("--", 2.2), ("-.", 1.8), (":", 1.8), ((0, (5, 1)), 1.5)]
    for index, (label, (ages, values)) in enumerate(curves.items()):
        dash, width = styles[index % len(styles)]
        resampled = np.interp(truth_ages, ages, values)
        ax.plot(
            truth_ages, resampled - resampled.mean(),
            linestyle=dash, linewidth=width, alpha=0.9, label=label,
        )

    ax.set_xlabel("Age")
    ax.set_ylabel("centred value")
    ax.set_title(title)
    ax.set_xticks(AGE_TICKS)
    ax.grid(True)
    ax.legend(fontsize=8)
    return ax


def plot_peak_error_by_arm(study, ax: plt.Axes | None = None, title: str | None = None):
    """Distribution of peak-age error across replicates, one box per arm.

    The zero line is the truth. A box sitting wholly off it is a bias the estimator has
    regardless of sample noise -- which is the distinction a single fit cannot draw.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))

    arms = list(dict.fromkeys(study["arm"]))
    data = [study.loc[study["arm"] == arm, "peak_age_error"].values for arm in arms]

    ax.boxplot(data, tick_labels=arms, showmeans=True)
    ax.axhline(0.0, color="black", lw=1.2, ls="--", zorder=0)
    ax.set_ylabel("peak age error (years)")
    ax.set_title(title or "Recovered peak minus the true peak")
    ax.grid(True, axis="y")
    ax.tick_params(axis="x", rotation=20)
    return ax
