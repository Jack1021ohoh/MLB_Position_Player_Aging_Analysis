"""GAM specification, fitting, and aging-curve extraction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pygam import GAM, s, te

from mlb_aging.features import generate_data
from mlb_aging.metrics import MetricSpec

#: Smoothing penalties searched by ``gridsearch``.
LAM_GRID = np.logspace(-2, 3, 20)
N_SPLINES = 5
CURVE_POINTS = 1000


def build_gam() -> GAM:
    """The shared model specification.

    Term indices refer to positions in :attr:`MetricSpec.feature_cols`:
    ``te(0, 2)`` is age x experience, ``s(1)`` the career-mean talent control,
    ``s(3)`` the lagged prior season. Terms are rebuilt per call so fitted
    state is never shared between models.
    """
    return GAM(
        te(0, 2, n_splines=N_SPLINES)
        + s(1, n_splines=N_SPLINES)
        + s(3, n_splines=N_SPLINES)
    )


def fit_gam(
    data: pd.DataFrame,
    spec: MetricSpec,
    weights: np.ndarray | None = None,
    progress: bool = False,
) -> GAM:
    """Fit the GAM, grid-searching the smoothing penalty.

    ``weights`` defaults to the metric's fitting weight column; pass IPW
    weights explicitly to fit the survivorship-corrected variant.
    """
    x, y = generate_data(data, spec.feature_cols, spec.target_col)
    if weights is None:
        weights = data[spec.weight_col].values
    gam = build_gam()
    gam.gridsearch(x, y, weights=weights, lam=LAM_GRID, progress=progress)
    return gam


@dataclass(frozen=True)
class AgingCurve:
    """A traced aging curve and the grid it was evaluated on."""

    ages: np.ndarray
    predictions: np.ndarray

    @property
    def peak_age(self) -> float:
        return float(self.ages[np.argmax(self.predictions)])

    @property
    def peak_value(self) -> float:
        return float(np.max(self.predictions))


def aging_curve(
    gam: GAM,
    data: pd.DataFrame,
    spec: MetricSpec,
    curve_reference: float | str | None = "__spec__",
) -> AgingCurve:
    """Trace the age effect, holding the other features at reference values.

    Every non-age column is pinned: experience is tied to age (assuming a
    debut at :data:`~mlb_aging.dataset.MIN_AGE`), the lag is set to the
    training mean for that age, and the career mean follows
    :attr:`MetricSpec.curve_reference` unless ``curve_reference`` overrides it.

    Because ``s(1)`` enters the model additively, the career-mean reference
    shifts the whole curve by a constant -- it changes the peak *value* but
    never the peak *age*. The original notebooks are inconsistent here: the
    all-player wRC+ curve pins 100 while the top-player ones use the training
    mean, hence the override.
    """
    reference = spec.curve_reference if curve_reference == "__spec__" else curve_reference
    age_lag_means = data.groupby(["Age"])[spec.lag_col].mean()

    xx = gam.generate_X_grid(term=0, n=CURVE_POINTS)
    if reference == "train_mean":
        xx[:, 1] = np.mean(data[spec.target_col])
    elif reference is not None:
        xx[:, 1] = reference
    xx[:, 2] = np.subtract(np.floor(xx[:, 0]), 20.0)
    xx[:, 3] = [age_lag_means[age] for age in np.floor(xx[:, 0])]

    return AgingCurve(ages=xx[:, 0], predictions=gam.predict(xx))
