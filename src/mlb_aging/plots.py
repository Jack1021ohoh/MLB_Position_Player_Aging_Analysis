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
