"""Checks on the aging *curve*, as opposed to per-season prediction accuracy.

Test MAE is a weak instrument for curve quality. The baseline ladder shows why:
``persistence`` contains no age term at all -- it predicts last season unchanged
-- yet it lands within 1.2% of every other arm on Def and beats the GAM
outright. A score that a model with no aging signal nearly wins cannot rank
aging curves; it is dominated by how well the level is predicted, which is
mostly talent and persistence.

The two checks here look at the curve instead.

:func:`residuals_by_age` aggregates held-out residuals into age bins. Averaged
over the whole test set a systematic error in the decline phase disappears;
broken out by age it is visible and locatable. If the curve's shape is right,
mean residuals are flat in age. A bend says the curve is wrong in a specific
place, and its sign says which way.

:func:`era_curves` refits the whole curve on disjoint eras. A claim about human
aging should replicate across twenty years of different players; if the peak
moves, the peak is not stable enough to publish. This needs a control, since
each era has half the data and half-sized samples wobble on their own -- use
:func:`random_split_curves` for that, and only call an era difference real when
it exceeds what random splits of the same size produce.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_aging.dataset import DEFAULT_DATA_DIR, build_training_frame, load_data
from mlb_aging.evaluate import inference
from mlb_aging.gam import AgingCurve, aging_curve, fit_gam
from mlb_aging.metrics import MetricSpec

#: Bin edges chosen so every bin holds enough test rows to mean something.
#: The test set has 887 rows spread over twenty ages, so per-age bins are noise.
DEFAULT_AGE_BINS = (20, 27, 29, 31, 33, 35, 41)


@dataclass(frozen=True)
class ResidualProfile:
    """Mean held-out residual by age bin, for one fit on one frame."""

    spec: MetricSpec
    label: str
    #: Index: age bin. Columns: ``n``, ``mean_resid``, ``sd``.
    table: pd.DataFrame

    @property
    def bends_with_age(self) -> float:
        """Change in mean residual from the youngest bin to the oldest.

        Large and negative means the curve increasingly over-predicts older
        players -- a decline phase fitted too shallow.
        """
        return float(self.table["mean_resid"].iloc[-1] - self.table["mean_resid"].iloc[0])

    def summary(self) -> str:
        head = (
            f"{self.spec.name} — {self.label}\n"
            f"  residual = actual − predicted, {self.spec.eval_weight_col}-weighted; "
            f"negative means the curve over-predicts\n"
            f"  {'age bin':>10s} {'n':>6s} {'mean resid':>12s} {'sd':>9s}"
        )
        rows = [
            f"  {str(idx):>10s} {int(r.n):6d} {r.mean_resid:12.3f} {r.sd:9.3f}"
            for idx, r in self.table.iterrows()
        ]
        return "\n".join([head, *rows, f"  youngest → oldest bend: {self.bends_with_age:+.3f}"])


def residuals_by_age(
    frame: pd.DataFrame,
    spec: MetricSpec,
    model,
    bins: tuple[int, ...] = DEFAULT_AGE_BINS,
    label: str = "",
) -> ResidualProfile:
    """Aggregate ``frame``'s residuals under ``model`` into age bins."""
    scored = inference(frame, spec, model)
    scored["residual"] = scored[spec.target_col] - scored["prediction"]
    scored["age_bin"] = pd.cut(scored["Age"], list(bins))

    grouped = scored.groupby("age_bin", observed=True)
    table = grouped.apply(
        lambda d: pd.Series({
            "n": len(d),
            "mean_resid": np.average(d["residual"], weights=d[spec.eval_weight_col]),
            "sd": d["residual"].std(),
        }),
        include_groups=False,
    )
    return ResidualProfile(spec=spec, label=label, table=table)


def era_curves(
    spec: MetricSpec,
    eras: dict[str, tuple[int, int]],
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict[str, tuple[AgingCurve, int]]:
    """Refit and retrace the curve on each era. Returns ``{label: (curve, n)}``."""
    train_data, _ = load_data(spec, data_dir=data_dir)

    out = {}
    for label, (start, end) in eras.items():
        subset = train_data.loc[
            (train_data["Season"] >= start) & (train_data["Season"] <= end), :
        ]
        frame = build_training_frame(subset, spec)
        curve = aging_curve(fit_gam(frame, spec), frame, spec)
        out[label] = (curve, len(frame))
    return out


def random_split_curves(
    spec: MetricSpec,
    seed: int = 0,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[AgingCurve, AgingCurve]:
    """Split players (not rows) at random into halves and fit each.

    The control for :func:`era_curves`: each era holds half the data, and a
    half-sized sample moves the peak on its own. An era difference is only an
    era *effect* if it exceeds what this produces.
    """
    train_data, _ = load_data(spec, data_dir=data_dir)
    ids = train_data["IDfg"].unique().copy()
    np.random.default_rng(seed).shuffle(ids)
    half = set(ids[: len(ids) // 2])

    curves = []
    for mask in (train_data["IDfg"].isin(half), ~train_data["IDfg"].isin(half)):
        frame = build_training_frame(train_data.loc[mask, :], spec)
        curves.append(aging_curve(fit_gam(frame, spec), frame, spec))
    return curves[0], curves[1]
