"""Regression tests pinning the ``tensor`` specification's results.

Every expected value here was transcribed from the stored outputs of GAM.ipynb,
GAM_IPW.ipynb and GAM_top.ipynb as they stood when ``te(0, 2)`` was the default,
and every call below therefore passes ``model_spec=TENSOR_SPEC`` explicitly.

The published default is now ``age_only`` (see ``curve_validation.ipynb``
section 5 for why), so these no longer describe what ``mlb-aging train`` prints.
They are kept unchanged as the reproduction contract for the original notebooks:
the tensor code path still exists, and these prove it still produces exactly what
it always did. ``test_age_only.py`` pins the numbers that are published now.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mlb_aging.dataset import build_training_frame, load_data
from mlb_aging.ipw import evaluate_retirement_model
from mlb_aging.metrics import ELITE_WAR_THRESHOLD, ELITE_WRC_THRESHOLD, get_metric
from conftest import baselines, fit
from mlb_aging.gam import TENSOR_SPEC

# Peak ages and MAEs are exact floats from `execute_result` cells, so they are
# compared tightly. Values printed with `:.4f` get a rounding-width tolerance.
EXACT = 1e-9
PRINTED = 5e-5

#: Rows surviving generate_test_data -- shared by every arm of every comparison.
N_TEST = 887


# --- GAM.ipynb: all players ------------------------------------------------
# metric -> (peak_age, peak_value, test_mae)
ALL_PLAYERS = {
    "OPS":  (26.002002002002,    0.01849932246726088, 0.07515022260741691),
    "wRC+": (26.002002002002,  104.72366847230786,   19.715383859491936),
    "Def":  (23.016016016016017,  1.1198520354753811,  4.7541250657038105),
    "Spd":  (21.0,                4.440684814905345,   0.986151157868963),
    "WAR":  (25.013013013013015,  1.8910338254134658,  1.49658277683825),
}

# --- GAM_IPW.ipynb ---------------------------------------------------------
# metric -> (peak_age, ipw_test_mae)   [printed to 4dp]
IPW_RESULTS = {
    "OPS":  (26.0,  0.0741),
    "wRC+": (26.0, 19.4684),
    "Def":  (23.0,  4.6579),
    "Spd":  (21.0,  0.9813),
    "WAR":  (25.0,  1.4449),
}

# metric -> survival-model hold-out AUC  [printed to 4dp]
RETIREMENT_AUC = {
    "OPS": 0.8329, "wRC+": 0.8331, "Def": 0.8006, "Spd": 0.7955, "WAR": 0.8233,
}
HOLDOUT_N = 1827
HOLDOUT_SEASONS = (2014, 2018)

# --- GAM_top.ipynb: elite cohorts ------------------------------------------
# id -> (metric, threshold, cohort_metric, elite_test, n_train, peak, peak_value, mae)
ELITE_CASES = {
    "wrc_own": ("wRC+", ELITE_WRC_THRESHOLD, None,   True,  3929,
                26.002002002002, 126.8807595176286, 20.18157138584899),
    "war_own": ("WAR",  ELITE_WAR_THRESHOLD, None,   True,  3581,
                26.002002002002,   4.102328129637835, 1.7052038147870012),
    "def_war_cohort": ("Def", ELITE_WAR_THRESHOLD, "WAR", False, None,
                       21.98898898898899, 2.8251596952456817, 4.6020162711969865),
    "wrc_war_cohort": ("wRC+", ELITE_WAR_THRESHOLD, "WAR", False, None,
                       27.01001001001001, 123.5957775818701, 19.99131833317994),
}

# The all-player models scored on the elite test subset -- the "less accurate on
# top performers" comparison in GAM_top.ipynb's prose. Training is unrestricted;
# only the test set is filtered.  metric -> (test_threshold, n_test, mae)
ALL_MODEL_ON_ELITE_TEST = {
    "wRC+": (ELITE_WRC_THRESHOLD, 411, 20.4677333736631),
    "WAR":  (ELITE_WAR_THRESHOLD, 376,  1.7429755586702773),
}


@pytest.mark.parametrize("metric", sorted(ALL_PLAYERS))
def test_all_player_curves(metric):
    peak, peak_value, mae = ALL_PLAYERS[metric]
    result = fit(metric, model_spec=TENSOR_SPEC)

    assert result.peak_age == pytest.approx(peak, abs=EXACT)
    assert result.peak_value == pytest.approx(peak_value, abs=EXACT)
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


@pytest.mark.parametrize("metric", sorted(IPW_RESULTS))
def test_ipw_corrected_curves(metric):
    peak, mae = IPW_RESULTS[metric]
    result = fit(metric, ipw=True, model_spec=TENSOR_SPEC)

    assert result.peak_age == pytest.approx(peak, abs=5e-2)
    assert result.test_mae == pytest.approx(mae, abs=PRINTED)


@pytest.mark.parametrize("metric", sorted(RETIREMENT_AUC))
def test_retirement_model_discrimination(metric):
    spec = get_metric(metric)
    train_data, _ = load_data(spec)
    frame = build_training_frame(train_data, spec)

    diagnostics = evaluate_retirement_model(
        frame, weight_col=spec.weight_col, perf_col=spec.lag_col
    )

    assert diagnostics.auc == pytest.approx(RETIREMENT_AUC[metric], abs=PRINTED)
    assert diagnostics.n_eval == HOLDOUT_N
    assert (diagnostics.holdout_start, diagnostics.holdout_end) == HOLDOUT_SEASONS


@pytest.mark.parametrize("case", sorted(ELITE_CASES))
def test_elite_cohort_curves(case):
    metric, threshold, cohort, elite_test, n_train, peak, peak_value, mae = ELITE_CASES[case]

    result = fit(
        metric,
        elite_threshold=threshold,
        cohort_metric=cohort,
        elite_test=elite_test,
        # every top-player section pins the training mean, even for wRC+
        curve_reference="train_mean",
        model_spec=TENSOR_SPEC,
    )

    if n_train is not None:
        assert result.n_train == n_train
    assert result.peak_age == pytest.approx(peak, abs=EXACT)
    assert result.peak_value == pytest.approx(peak_value, abs=EXACT)
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


@pytest.mark.parametrize("metric", sorted(ALL_MODEL_ON_ELITE_TEST))
def test_all_player_model_scored_on_elite_test(metric):
    threshold, n_test, mae = ALL_MODEL_ON_ELITE_TEST[metric]
    result = fit(metric, test_threshold=threshold, model_spec=TENSOR_SPEC)

    assert result.n_test == n_test
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


# The G-weighted alternative for the metrics whose fit and eval weights differ.
# Reported alongside test_mae so the weighting choice needs no argument.
FIT_WEIGHTED_MAE = {"Def": 4.708312735637497, "Spd": 0.9940013823141102,
                    "WAR": 1.4696828199425542}


def test_scoring_weight_is_pinned_deliberately():
    """Def, Spd and WAR are fitted G-weighted and scored PA-weighted.

    Not a defect: the fit weight is about which observations are reliable, the
    eval weight about which errors matter. This pins the choice so it stays a
    decision rather than reverting to ``evaluate()``'s default.
    """
    for metric in ("Def", "Spd", "WAR"):
        spec = get_metric(metric)
        assert spec.weight_col == "G"
        assert spec.eval_weight_col == "PA"

    for metric in ("OPS", "wRC+"):
        spec = get_metric(metric)
        assert spec.weight_col == spec.eval_weight_col == "PA"


@pytest.mark.parametrize("metric", sorted(ALL_PLAYERS))
def test_both_scoring_weights_are_reported(metric):
    result = fit(metric, model_spec=TENSOR_SPEC)
    spec = result.spec

    if metric in FIT_WEIGHTED_MAE:
        assert result.test_mae_fit_weighted == pytest.approx(
            FIT_WEIGHTED_MAE[metric], abs=EXACT
        )
        assert result.test_mae_fit_weighted != result.test_mae
    else:
        # PA-weighted metrics score identically under either weight
        assert spec.weight_col == spec.eval_weight_col
        assert result.test_mae_fit_weighted == pytest.approx(result.test_mae, abs=EXACT)


# --- delta_method.ipynb: the naive ladder -----------------------------------
# Scored on the same frame run_metric uses, so these are directly comparable to
# ALL_PLAYERS and IPW_RESULTS above. metric -> {baseline: eval-weighted MAE}
BASELINE_MAE = {
    "OPS":  {"persistence": 0.0823575147776609,
             "delta_curve": 0.08467877637307515,
             "delta_lag":   0.08124777286831522},
    "wRC+": {"persistence": 21.317703824821262,
             "delta_curve": 22.543944608639222,
             "delta_lag":   21.02688893945524},
    "Def":  {"persistence": 4.5880660019450055,
             "delta_curve": 6.594079119564908,
             "delta_lag":   4.531911222384779},
    "Spd":  {"persistence": 1.027493507742618,
             "delta_curve": 1.355467067253739,
             "delta_lag":   1.0230182287413296},
    "WAR":  {"persistence": 1.525558921911235,
             "delta_curve": 1.6644038478289973,
             "delta_lag":   1.4998261855925554},
}

# The delta method's own curve peaks, and the number of consecutive-season
# pairs it is estimated from (identical across metrics -- same row set).
DELTA_PEAK_AGE = {"OPS": 26.0, "wRC+": 26.0, "Def": 24.0, "Spd": 22.0, "WAR": 26.0}
DELTA_N_PAIRS = 11324

#: Where the GAM arms land against the strongest baseline. Def is negative on
#: purpose: the naive predictor genuinely wins there, and that must not be
#: quietly "fixed" away.
GAM_VS_DELTA_LAG = {
    "OPS":  (0.07504882959414927,   0.08848701103718748),
    "wRC+": (0.062372759172296455,  0.07411779979554689),
    "Def":  (-0.04903314129840708, -0.027790196906223752),
    "Spd":  (0.03603755029636735,   0.040747284416618346),
    "WAR":  (0.0021625230879813717, 0.03660378527075736),
}


@pytest.mark.parametrize("metric", sorted(BASELINE_MAE))
def test_naive_baselines(metric):
    comparison = baselines(metric, model_spec=TENSOR_SPEC)

    for name, expected in BASELINE_MAE[metric].items():
        assert comparison.maes[name][0] == pytest.approx(expected, abs=EXACT), name

    assert comparison.delta_curve.peak_age == DELTA_PEAK_AGE[metric]
    assert comparison.delta_curve.n_pairs == DELTA_N_PAIRS
    assert comparison.n_test == N_TEST


@pytest.mark.parametrize("metric", sorted(GAM_VS_DELTA_LAG))
def test_gam_improvement_over_delta_lag(metric):
    """Pin how much the GAM arms beat the strongest naive baseline.

    These are the numbers any "GAM + IPW improves accuracy by X" claim rests
    on, so they are pinned rather than recomputed by hand each time.
    """
    comparison = baselines(metric, model_spec=TENSOR_SPEC)
    expected_gam, expected_ipw = GAM_VS_DELTA_LAG[metric]

    assert comparison.improvement("gam") == pytest.approx(expected_gam, abs=EXACT)
    assert comparison.improvement("gam_ipw") == pytest.approx(expected_ipw, abs=EXACT)

    # The GAM arms must be scored on the same frame as the baselines.
    assert comparison.maes["gam"][0] == pytest.approx(ALL_PLAYERS[metric][2], abs=EXACT)


def test_delta_method_ladder_is_ordered_as_documented():
    """delta_lag is the strongest baseline for every metric."""
    for metric in sorted(BASELINE_MAE):
        maes = BASELINE_MAE[metric]
        assert maes["delta_lag"] < maes["persistence"] < maes["delta_curve"], metric


# --- Curve geometry: which "peaks" are real ---------------------------------
#: Spd's maximum is the youngest age fitted, not a turning point. add_lag drops
#: every player's first season, so the curve starts at 21 and only ever falls.
#: The number 21 is right; calling it a peak is not.
PEAK_IS_LEFT_EDGE = {
    "OPS": False, "wRC+": False, "Def": False, "Spd": True, "WAR": False,
}
YOUNGEST_FITTED_AGE = 21.0
#: generate_X_grid on the te(0, 2) tensor returns an n x n mesh.
CURVE_GRID_POINTS = 1_000_000
CURVE_UNIQUE_AGES = 1_000


@pytest.mark.parametrize("metric", sorted(PEAK_IS_LEFT_EDGE))
def test_peak_is_interior_except_for_spd(metric):
    curve = fit(metric, model_spec=TENSOR_SPEC).curve

    assert curve.by_age.index.min() == pytest.approx(YOUNGEST_FITTED_AGE, abs=EXACT)
    assert curve.peaks_at_left_edge is PEAK_IS_LEFT_EDGE[metric]

    rise = curve.peak_value - curve.value_at(YOUNGEST_FITTED_AGE)
    if PEAK_IS_LEFT_EDGE[metric]:
        assert rise == pytest.approx(0.0, abs=EXACT)
    else:
        assert rise > 0.0


def test_curve_ages_are_a_mesh_and_by_age_collapses_it():
    """``ages`` repeats and is unsorted; ``by_age`` must fix both."""
    curve = fit("WAR", model_spec=TENSOR_SPEC).curve

    assert len(curve.ages) == CURVE_GRID_POINTS
    assert len(curve.by_age) == CURVE_UNIQUE_AGES
    assert curve.by_age.index.is_monotonic_increasing
    # every repeat of an age carries the same prediction, so collapsing is lossless
    assert pd.Series(curve.predictions, index=curve.ages).groupby(level=0).nunique().max() == 1


# --- The decline schedule the README quotes ---------------------------------
DECLINE_30_TO_34 = {
    "OPS":  -0.02685685289440412,
    "wRC+": -6.687740147129119,
    "Def":  -3.0878889879219966,
    "Spd":  -0.4564014608176836,
    "WAR":  -0.9922813131497311,
}
WAR_AT_30 = 1.2167236828651082
WAR_AT_35 = 0.02517545987321492


@pytest.mark.parametrize("metric", sorted(DECLINE_30_TO_34))
def test_decline_schedule(metric):
    curve = fit(metric, model_spec=TENSOR_SPEC).curve
    assert curve.change_between(30, 34) == pytest.approx(
        DECLINE_30_TO_34[metric], abs=EXACT
    )


def test_war_reaches_replacement_level_around_35():
    """The README's headline decline claim: ~1 WAR lost 30->34, ~0 at 35."""
    curve = fit("WAR", model_spec=TENSOR_SPEC).curve

    assert curve.value_at(30) == pytest.approx(WAR_AT_30, abs=EXACT)
    assert curve.value_at(35) == pytest.approx(WAR_AT_35, abs=EXACT)
    assert curve.value_at(34) > 0.0 > curve.value_at(36)
