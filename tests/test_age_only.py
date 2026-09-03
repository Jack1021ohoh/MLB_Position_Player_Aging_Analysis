"""Regression tests pinning the published results.

``age_only`` -- a smooth on age alone -- is the default specification, so these
are the numbers ``mlb-aging train`` prints, the notebooks show and README.md
quotes. ``test_regression.py`` covers the older ``tensor`` code path, which is
still selectable with ``--model-spec tensor`` and still reproduces exactly what
it always did.

Everything here is transcribed from the stored notebook outputs, to the same
1e-9 tolerance used throughout.
"""

from __future__ import annotations

import pytest

from conftest import baselines, fit
from mlb_aging.metrics import ELITE_WAR_THRESHOLD, ELITE_WRC_THRESHOLD

EXACT = 1e-9
N_TEST = 887
N_TRAIN = 13636

# metric -> (peak_age, peak_value, test_mae)
ALL_PLAYERS = {
    "OPS": (26.99099099099099, 0.009180945389039474, 0.07550499949160745),
    "wRC+": (26.99099099099099, 102.27262984686107, 19.77303671002998),
    "Def": (21.98898898898899, 0.8257503759969484, 4.815618540819553),
    "Spd": (21.0, 4.439084146771299, 0.9960747363758543),
    "WAR": (27.01001001001001, 1.7081423335960508, 1.4969633162607543),
}

# metric -> (peak_age, test_mae)
IPW_RESULTS = {
    "OPS": (27.01001001001001, 0.0741945469485932),
    "wRC+": (27.01001001001001, 19.491845731861737),
    "Def": (21.98898898898899, 4.727301657158723),
    "Spd": (21.0, 0.9889032101403478),
    "WAR": (26.002002002002, 1.435728792175524),
}


@pytest.mark.parametrize("metric", sorted(ALL_PLAYERS))
def test_all_player_curves(metric):
    peak_age, peak_value, test_mae = ALL_PLAYERS[metric]
    result = fit(metric)

    assert result.n_train == N_TRAIN
    assert result.n_test == N_TEST
    assert result.curve.peak_age == pytest.approx(peak_age, abs=EXACT)
    assert result.curve.peak_value == pytest.approx(peak_value, abs=EXACT)
    assert result.test_mae == pytest.approx(test_mae, abs=EXACT)


@pytest.mark.parametrize("metric", sorted(IPW_RESULTS))
def test_ipw_corrected_curves(metric):
    peak_age, test_mae = IPW_RESULTS[metric]
    result = fit(metric, ipw=True)

    assert result.curve.peak_age == pytest.approx(peak_age, abs=EXACT)
    assert result.test_mae == pytest.approx(test_mae, abs=EXACT)


# Only Spd's maximum sits on the left edge of the fitted range. Def's is a
# nominal interior turning point that a resample moves by 3-4 years, so it is
# checked for its tiny rise rather than treated as a real peak.
PEAK_IS_LEFT_EDGE = {"OPS": False, "wRC+": False, "Def": False, "Spd": True, "WAR": False}
RISE_TO_PEAK = {"OPS": 0.0480, "wRC+": 12.3215, "Def": 0.0982, "Spd": 0.0, "WAR": 0.4854}
YOUNGEST_FITTED_AGE = 21.0


@pytest.mark.parametrize("metric", sorted(PEAK_IS_LEFT_EDGE))
def test_peak_is_interior_except_for_spd(metric):
    curve = fit(metric).curve

    assert curve.by_age.index.min() == pytest.approx(YOUNGEST_FITTED_AGE, abs=EXACT)
    assert curve.peaks_at_left_edge is PEAK_IS_LEFT_EDGE[metric]
    rise = curve.peak_value - curve.by_age.iloc[0]
    assert rise == pytest.approx(RISE_TO_PEAK[metric], abs=5e-5)


def test_age_only_grid_is_a_sorted_sweep():
    """Unlike the tensor, s(0) yields one row per grid point, already sorted."""
    curve = fit("WAR").curve

    assert len(curve.ages) == 1000
    assert len(curve.by_age) == 1000
    assert list(curve.by_age.index) == sorted(curve.by_age.index)


# The numbers a contract spanning the decline phase turns on.
DECLINE_30_TO_34 = {
    "OPS": -0.0211, "wRC+": -5.4198, "Def": -1.8566, "Spd": -0.3768, "WAR": -0.6517,
}
WAR_AT_30 = 1.4980329922982238
WAR_AT_34 = 0.846283115474359


@pytest.mark.parametrize("metric", sorted(DECLINE_30_TO_34))
def test_decline_schedule(metric):
    curve = fit(metric).curve
    assert curve.change_between(30, 34) == pytest.approx(DECLINE_30_TO_34[metric], abs=5e-5)


def test_war_reaches_replacement_level_around_38():
    """The age-only curve crosses zero far later than the tensor's 35.1."""
    curve = fit("WAR").curve

    assert curve.value_at(30) == pytest.approx(WAR_AT_30, abs=EXACT)
    assert curve.value_at(34) == pytest.approx(WAR_AT_34, abs=EXACT)
    after_peak = curve.by_age[curve.by_age.index > 27]
    crossing = after_peak[after_peak <= 0].index[0]
    assert crossing == pytest.approx(38.022, abs=0.01)


# Positive means the arm beats delta_lag. Def is negative and pinned so it
# cannot be quietly tuned away.
GAM_VS_DELTA_LAG = {
    "OPS": (0.0707, 0.0868), "wRC+": (0.0596, 0.0730), "Def": (-0.0626, -0.0431),
    "Spd": (0.0263, 0.0333), "WAR": (0.0019, 0.0427),
}


@pytest.mark.parametrize("metric", sorted(GAM_VS_DELTA_LAG))
def test_gam_improvement_over_delta_lag(metric):
    gam, gam_ipw = GAM_VS_DELTA_LAG[metric]
    comparison = baselines(metric)

    assert comparison.n_test == N_TEST
    assert comparison.improvement("gam") == pytest.approx(gam, abs=5e-5)
    assert comparison.improvement("gam_ipw") == pytest.approx(gam_ipw, abs=5e-5)


# (metric, threshold, cohort_metric, elite_test) -> (n_train, peak_age, test_mae)
ELITE_CASES = {
    "wRC+ hitters": (
        ("wRC+", ELITE_WRC_THRESHOLD, None, True),
        (3929, 28.01801801801802, 20.161914374116048),
    ),
    "wRC+ on WAR cohort": (
        ("wRC+", ELITE_WAR_THRESHOLD, "WAR", False),
        (3581, 28.01801801801802, 19.899391816663368),
    ),
    "WAR": (
        ("WAR", ELITE_WAR_THRESHOLD, "WAR", False),
        (3581, 27.01001001001001, 1.6867258943308796),
    ),
    "Def": (
        ("Def", ELITE_WAR_THRESHOLD, "WAR", False),
        (3581, 21.98898898898899, 4.625833966648357),
    ),
}


@pytest.mark.parametrize("case", sorted(ELITE_CASES))
def test_elite_cohort_curves(case):
    (metric, threshold, cohort, elite_test), (n_train, peak, mae) = ELITE_CASES[case]

    result = fit(
        metric,
        elite_threshold=threshold,
        cohort_metric=cohort,
        elite_test=elite_test,
        # every top-player section pins the training mean, even for wRC+
        curve_reference="train_mean",
    )

    assert result.n_train == n_train
    assert result.curve.peak_age == pytest.approx(peak, abs=EXACT)
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


# The all-player model scored on elite test rows only -- worse than the
# specialist models above, which is what makes the elite curves worth fitting.
ALL_MODEL_ON_ELITE_TEST = {
    "wRC+": (411, 20.563343287870808),
    "WAR": (376, 1.7480212569085207),
}


@pytest.mark.parametrize("metric", sorted(ALL_MODEL_ON_ELITE_TEST))
def test_all_player_model_scored_on_elite_test(metric):
    n_test, mae = ALL_MODEL_ON_ELITE_TEST[metric]
    threshold = ELITE_WRC_THRESHOLD if metric == "wRC+" else ELITE_WAR_THRESHOLD
    result = fit(metric, test_threshold=threshold)

    assert result.n_test == n_test
    assert result.test_mae == pytest.approx(mae, abs=5e-5)
