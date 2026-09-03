"""Shared, memoised model fits.

Every test here reads a fitted model; none mutates one. Without caching the
suite refits the same handful of GAMs a dozen times over -- `run_metric("WAR")`
alone is wanted by four separate tests -- and a full run costs three minutes for
about twenty distinct fits.

These wrappers key on plain hashable arguments so `lru_cache` can do the
deduplication. Call them instead of `run_metric` / `compare_baselines` directly;
a test that genuinely needs a fresh fit should call the pipeline itself and say
why.
"""

from __future__ import annotations

from functools import lru_cache

from mlb_aging.gam import DEFAULT_SPEC
from mlb_aging.metrics import get_metric
from mlb_aging.pipeline import compare_baselines, run_metric


@lru_cache(maxsize=None)
def fit(
    metric: str,
    *,
    ipw: bool = False,
    elite_threshold: float | None = None,
    elite_test: bool = False,
    cohort_metric: str | None = None,
    curve_reference: float | str | None = "__spec__",
    test_threshold: float | None = None,
    model_spec: str = DEFAULT_SPEC,
):
    """`run_metric`, memoised on its arguments."""
    return run_metric(
        get_metric(metric),
        ipw=ipw,
        elite_threshold=elite_threshold,
        elite_test=elite_test,
        cohort_metric=get_metric(cohort_metric) if cohort_metric else None,
        curve_reference=curve_reference,
        test_threshold=test_threshold,
        model_spec=model_spec,
    )


@lru_cache(maxsize=None)
def baselines(metric: str, *, model_spec: str = DEFAULT_SPEC):
    """`compare_baselines`, memoised on its arguments."""
    return compare_baselines(get_metric(metric), model_spec=model_spec)
