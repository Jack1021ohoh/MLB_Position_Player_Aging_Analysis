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
from pygam import GAM
from pygam.utils import flatten

from mlb_aging.dataset import DEFAULT_DATA_DIR, build_training_frame, load_data
from mlb_aging.evaluate import inference
from mlb_aging.ipw import fit_ipw_weights
from mlb_aging.gam import DEFAULT_SPEC, AgingCurve, aging_curve, fit_gam
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


@dataclass(frozen=True)
class SplitFit:
    """One curve fitted on one subset of the training data."""

    label: str
    curve: AgingCurve
    #: The fitted model, kept so the grid-searched ``lam`` is inspectable.
    gam: GAM
    n_rows: int
    n_players: int

    @property
    def peak_age(self) -> float:
        return float(self.curve.peak_age)

    @property
    def lam(self) -> float:
        """The selected smoothing penalty.

        ``LAM_GRID`` is one-dimensional, so ``gridsearch`` broadcasts a single
        value across every term and this scalar is the whole selection.
        """
        return float(flatten(self.gam.lam)[0])


def _fit_subset(
    frame: pd.DataFrame,
    spec: MetricSpec,
    label: str,
    model_spec: str = DEFAULT_SPEC,
    use_ipw: bool = False,
) -> SplitFit:
    """Fit and trace one subset. IPW is refit *within* the subset, not reused."""
    weights = None
    if use_ipw:
        weights = fit_ipw_weights(frame, spec)["ipw_final_weight"].values
    gam = fit_gam(frame, spec, weights=weights, model_spec=model_spec)
    return SplitFit(
        label=label,
        curve=aging_curve(gam, frame, spec),
        gam=gam,
        n_rows=len(frame),
        n_players=int(frame["IDfg"].nunique()),
    )


def era_curves(
    spec: MetricSpec,
    eras: dict[str, tuple[int, int]],
    model_spec: str = DEFAULT_SPEC,
    use_ipw: bool = False,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> dict[str, SplitFit]:
    """Refit and retrace the curve on each era."""
    train_data, _ = load_data(spec, data_dir=data_dir)

    out = {}
    for label, (start, end) in eras.items():
        subset = train_data.loc[
            (train_data["Season"] >= start) & (train_data["Season"] <= end), :
        ]
        frame = build_training_frame(subset, spec)
        out[label] = _fit_subset(frame, spec, label, model_spec, use_ipw)
    return out


def random_split_curves(
    spec: MetricSpec,
    seed: int = 0,
    model_spec: str = DEFAULT_SPEC,
    use_ipw: bool = False,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> tuple[SplitFit, SplitFit]:
    """Split players (not rows) at random into halves and fit each.

    The control for :func:`era_curves`: each era holds half the data, and a
    half-sized sample moves the peak on its own. An era difference is only an
    era *effect* if it exceeds what this produces.

    Splitting on ``IDfg`` rather than on rows keeps a player's seasons together,
    so the two halves are independent. Splitting rows would put the same player
    on both sides and understate the spread.
    """
    train_data, _ = load_data(spec, data_dir=data_dir)
    ids = train_data["IDfg"].unique().copy()
    np.random.default_rng(seed).shuffle(ids)
    half = set(ids[: len(ids) // 2])

    in_half = train_data["IDfg"].isin(half)
    return tuple(
        _fit_subset(
            build_training_frame(train_data.loc[mask, :], spec),
            spec,
            label=f"seed {seed} {name}",
            model_spec=model_spec,
            use_ipw=use_ipw,
        )
        for name, mask in (("A", in_half), ("B", ~in_half))
    )


def peak_stability(
    spec: MetricSpec,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    model_spec: str = DEFAULT_SPEC,
    use_ipw: bool = False,
    data_dir: Path | str = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Peak age from every random half, one row per half-sample fit.

    Columns: ``seed``, ``half``, ``peak_age``, ``lam``, ``n_rows``,
    ``n_players``. The spread of ``peak_age`` is the resolution of the peak
    estimate; the paired gap within a seed is the noise floor an era difference
    has to clear.
    """
    rows = []
    for seed in seeds:
        for half, fit in zip("AB", random_split_curves(
            spec, seed=seed, model_spec=model_spec, use_ipw=use_ipw, data_dir=data_dir
        )):
            rows.append({
                "seed": seed,
                "half": half,
                "peak_age": fit.peak_age,
                "lam": fit.lam,
                "n_rows": fit.n_rows,
                "n_players": fit.n_players,
            })
    return pd.DataFrame(rows)
