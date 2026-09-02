"""Regression tests pinning the published notebook results.

Every expected value is transcribed from the stored outputs of GAM.ipynb,
GAM_IPW.ipynb and GAM_top.ipynb. They exist so the refactor can be proven
behaviour-preserving *before* any of the known bugs are fixed -- when a fix
lands, the expectation it moves should change in the same commit.
"""

from __future__ import annotations

import pytest

from mlb_aging.dataset import build_training_frame, load_data
from mlb_aging.ipw import evaluate_retirement_model
from mlb_aging.metrics import ELITE_WAR_THRESHOLD, ELITE_WRC_THRESHOLD, get_metric
from mlb_aging.pipeline import run_metric

# Peak ages and MAEs are exact floats from `execute_result` cells, so they are
# compared tightly. Values printed with `:.4f` get a rounding-width tolerance.
EXACT = 1e-9
PRINTED = 5e-5


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
    result = run_metric(get_metric(metric))

    assert result.peak_age == pytest.approx(peak, abs=EXACT)
    assert result.peak_value == pytest.approx(peak_value, abs=EXACT)
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


@pytest.mark.parametrize("metric", sorted(IPW_RESULTS))
def test_ipw_corrected_curves(metric):
    peak, mae = IPW_RESULTS[metric]
    result = run_metric(get_metric(metric), ipw=True)

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

    result = run_metric(
        get_metric(metric),
        elite_threshold=threshold,
        cohort_metric=get_metric(cohort) if cohort else None,
        elite_test=elite_test,
        # every top-player section pins the training mean, even for wRC+
        curve_reference="train_mean",
    )

    if n_train is not None:
        assert result.n_train == n_train
    assert result.peak_age == pytest.approx(peak, abs=EXACT)
    assert result.peak_value == pytest.approx(peak_value, abs=EXACT)
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


@pytest.mark.parametrize("metric", sorted(ALL_MODEL_ON_ELITE_TEST))
def test_all_player_model_scored_on_elite_test(metric):
    threshold, n_test, mae = ALL_MODEL_ON_ELITE_TEST[metric]
    result = run_metric(get_metric(metric), test_threshold=threshold)

    assert result.n_test == n_test
    assert result.test_mae == pytest.approx(mae, abs=EXACT)


def test_eval_weight_bug_is_still_preserved():
    """Guard the deliberately-preserved scoring bug.

    Def, Spd and WAR are fitted G-weighted but scored PA-weighted, because the
    notebooks never passed ``weight_col`` to ``evaluate()``. When that is
    fixed, this test should be deleted and the MAE expectations updated.
    """
    for metric in ("Def", "Spd", "WAR"):
        spec = get_metric(metric)
        assert spec.weight_col == "G"
        assert spec.eval_weight_col == "PA"
