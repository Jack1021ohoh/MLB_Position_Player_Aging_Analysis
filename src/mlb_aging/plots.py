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
