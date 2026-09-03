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


#: Age x experience tensor product. The original specification, kept because
#: it scores marginally better and because the regression suite pins it.
TENSOR_SPEC = "tensor"
#: Smooth on age alone -- standard aging-curve practice, and the default.
AGE_ONLY_SPEC = "age_only"
MODEL_SPECS = (TENSOR_SPEC, AGE_ONLY_SPEC)
#: What every published number is fitted with.
DEFAULT_SPEC = AGE_ONLY_SPEC


def build_gam(model_spec: str = DEFAULT_SPEC) -> GAM:
    """The model specification. Terms are rebuilt per call, never shared.

    Term indices refer to positions in :attr:`MetricSpec.feature_cols`, so both
    variants read ``s(1)`` as the career-mean talent control and ``s(3)`` as the
    lagged prior season. They differ only in how age enters.

    ``"tensor"`` (default, and what every published number was fitted with)
        ``te(0, 2)``, the age x experience tensor product.
    ``"age_only"``
        ``s(0)``, a smooth on age alone. Experience stays in ``feature_cols``
        but no term reads it.

    The difference matters more for *reading* the model than for fitting it.
    A tensor has to be sliced to be traced, and :func:`aging_curve` slices it by
    pinning ``experience = age - 20`` -- one hypothetical career, debuting at
    20. That choice is not innocuous: peak age tracks it almost one-for-one
    (WAR peaks at 25 assuming debut at 20, at 27 assuming debut at 24), and
    debut-at-20 describes 43 of the ~2300 players in the training set. There is
    no principled slice, because age and experience are near-collinear.

    ``"age_only"`` removes the choice: with no experience term the traced curve
    is invariant to the debut assumption, so peak age is a property of the data
    rather than of the convention. This is what standard practice does -- the
    GAM approach in Baseball Prospectus's "The Delta Method, Revisited" and the
    FanGraphs Sabermetrics Library both smooth on age and control for career
    average performance, handling experience implicitly through that control
    precisely because the two are collinear. Test MAE barely moves (0.03-1.3%
    worse), so the tensor buys little beyond the ambiguity it introduces.
    """
    if model_spec == TENSOR_SPEC:
        age_term = te(0, 2, n_splines=N_SPLINES)
    elif model_spec == AGE_ONLY_SPEC:
        age_term = s(0, n_splines=N_SPLINES)
    else:
        raise ValueError(f"unknown model_spec {model_spec!r}; expected one of {MODEL_SPECS}")

    return GAM(age_term + s(1, n_splines=N_SPLINES) + s(3, n_splines=N_SPLINES))


def fit_gam(
    data: pd.DataFrame,
    spec: MetricSpec,
    weights: np.ndarray | None = None,
    progress: bool = False,
    model_spec: str = DEFAULT_SPEC,
) -> GAM:
    """Fit the GAM, grid-searching the smoothing penalty.

    ``weights`` defaults to the metric's fitting weight column; pass IPW
    weights explicitly to fit the survivorship-corrected variant.
    """
    x, y = generate_data(data, spec.feature_cols, spec.target_col)
    if weights is None:
        weights = data[spec.weight_col].values
    gam = build_gam(model_spec)
    gam.gridsearch(x, y, weights=weights, lam=LAM_GRID, progress=progress)
    return gam


@dataclass(frozen=True)
class AgingCurve:
    """A traced aging curve and the grid it was evaluated on.

    ``ages`` is **not** a sorted one-dimensional sweep. ``generate_X_grid`` on
    a tensor term returns an ``n x n`` mesh, so each age appears many times and
    the array is not monotonic -- interpolating against it directly gives
    nonsense. Every non-age feature is pinned by :func:`aging_curve` before
    prediction, so the repeats are exact duplicates; use :attr:`by_age` or
    :meth:`value_at`, which collapse them.
    """

    ages: np.ndarray
    predictions: np.ndarray

    @property
    def peak_age(self) -> float:
        return float(self.ages[np.argmax(self.predictions)])

    @property
    def peak_value(self) -> float:
        return float(np.max(self.predictions))

    @property
    def by_age(self) -> pd.Series:
        """The curve as one prediction per age, sorted ascending."""
        series = pd.Series(self.predictions, index=self.ages)
        return series.groupby(level=0).first().sort_index()

    def value_at(self, age: float) -> float:
        """The curve's value at ``age``, interpolated between grid points."""
        curve = self.by_age
        return float(np.interp(age, curve.index.values, curve.values))

    def change_between(self, start: float, end: float) -> float:
        """Signed change from ``start`` to ``end`` -- negative means decline."""
        return self.value_at(end) - self.value_at(start)

    @property
    def peaks_at_left_edge(self) -> bool:
        """True when the maximum sits on the youngest age fitted.

        Then the curve only ever declines over the observed range and the
        "peak" is an artefact of where the data starts, not a turning point.
        Spd is the case in point: no age-20 row survives ``add_lag``, so the
        curve begins at 21 and falls from there.
        """
        return bool(np.isclose(self.peak_age, self.by_age.index.min()))


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
