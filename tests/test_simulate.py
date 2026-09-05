"""Tests for the simulation study.

These pin the *generator* and the framework's validity, not any published number. Neither
``test_age_only.py`` nor ``test_regression.py`` is touched by this module, so the
reproduction contract in CLAUDE.md is unaffected.

The load-bearing test here is :func:`test_framework_recovers_a_known_peak`. A simulation
study whose estimator cannot recover a known curve under ideal conditions measures nothing,
so that case has to hold before any result in ``simulation_validation.ipynb`` is readable.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
from pygam import GAM, s

from mlb_aging.features import generate_data
from mlb_aging.gam import LAM_GRID, N_SPLINES
from mlb_aging.simulate import (
    ACCUMULATING,
    COMPARISON_AGES,
    COMPLETE_CASE,
    FIT_MIN_AGE,
    MIN_APPEARANCE_PA,
    DELTA_LAG,
    NO_TALENT_CONTROL,
    RATE,
    REAL_MOMENTS,
    CurveTargets,
    SimConfig,
    _model_frame,
    calibration_report,
    censor,
    run_simulation_study,
    simulate_careers,
    true_curve,
)

# A small league keeps these fast; the shapes under test do not need 2,312 players.
SMALL = replace(SimConfig(), n_players=400)


@lru_cache(maxsize=None)
def generated(seed: int = 0, n_players: int = 400):
    return simulate_careers(replace(SimConfig(), n_players=n_players), seed=seed)


# --------------------------------------------------------------------------------------
# The generator
# --------------------------------------------------------------------------------------


def test_generation_is_deterministic():
    """Same seed, same league -- to the suite's 1e-9 convention."""
    first, second = simulate_careers(SMALL, seed=7), simulate_careers(SMALL, seed=7)
    assert list(first.columns) == list(second.columns)
    assert len(first) == len(second)
    numeric = first.select_dtypes("number").columns
    assert np.allclose(first[numeric].values, second[numeric].values, atol=1e-9, rtol=0.0)


def test_different_seeds_give_different_leagues():
    assert len(simulate_careers(SMALL, seed=1)) != len(simulate_careers(SMALL, seed=2))


def test_no_non_appearances_survive():
    """Clipping the playing-time normal at zero leaves a point mass that has no
    counterpart in a real leaderboard, and a zero PA breaks the PA-weighted career mean."""
    frame = generated()
    assert frame["PA"].min() >= MIN_APPEARANCE_PA
    assert frame["G"].min() >= 1.0


@pytest.mark.parametrize("metric_name", ["OPS", "wRC+", "Spd", "Def", "WAR"])
def test_true_curve_hits_its_targets(metric_name):
    """The truth is calibrated in closed form, so it should match to machine precision."""
    metric = SimConfig().metric(metric_name)
    targets = metric.targets

    peak = float(true_curve(targets.peak_age, targets))
    assert peak == pytest.approx(targets.level, abs=1e-9)

    # The peak really is the maximum over the fitted range. Sampled on the comparison
    # grid, which does not land on the peak exactly, so the tolerance is the grid's own
    # quantization rather than machine precision.
    values = np.asarray(true_curve(COMPARISON_AGES, targets))
    step = COMPARISON_AGES[1] - COMPARISON_AGES[0]
    assert values.max() <= peak + 1e-9
    assert COMPARISON_AGES[np.argmax(values)] == pytest.approx(targets.peak_age, abs=step)

    rise = peak - float(true_curve(FIT_MIN_AGE, targets))
    assert rise == pytest.approx(targets.rise, abs=1e-9)

    change = float(true_curve(34.0, targets)) - float(true_curve(30.0, targets))
    assert change == pytest.approx(targets.decline_30_34, abs=1e-9)


def test_true_curve_rejects_an_uncalibratable_peak():
    """A peak past 32 puts both anchor ages on the same side of the maximum."""
    with pytest.raises(ValueError, match="30->34"):
        true_curve(30.0, CurveTargets(peak_age=33.0, rise=1.0, decline_30_34=-1.0, level=0.0))


def test_talent_correlations_are_respected():
    """WAR has to be composite, not a relabelled wRC+.

    The measured structure has Def *negatively* correlated with hitting -- good hitters
    take the easy defensive positions -- and Spd nearly orthogonal to it. A single shared
    latent talent would erase both.
    """
    frame = generated(n_players=1200)
    config = SimConfig()
    per_player = frame.groupby("IDfg")[[m.talent_col for m in config.metrics]].first()
    simulated = per_player.corr().values
    np.testing.assert_allclose(simulated, config.talent_correlation, atol=0.06)


# --------------------------------------------------------------------------------------
# Censoring
# --------------------------------------------------------------------------------------


def test_censoring_removes_rows_and_selects_upward():
    """The survivorship effect, present by construction: what qual=100 hides is worse."""
    frame = generated()
    observed = censor(frame)
    assert len(observed) < len(frame)

    old = frame["Age"] >= 34
    hidden = frame.loc[old & ~frame["qualified"], "SimWRC"]
    shown = frame.loc[old & frame["qualified"], "SimWRC"]
    assert len(hidden) > 0
    assert hidden.mean() < shown.mean()


def test_retirement_and_qualification_are_distinct():
    """The two flags must not be the same column.

    ``add_survived_label`` cannot tell them apart on real data -- a drop to 60 PA is
    labelled identically to retirement -- which is the conflation this study prices.
    """
    frame = generated()
    unqualified_but_playing = frame.loc[~frame["qualified"] & ~frame["retired"], :]
    assert len(unqualified_but_playing) > 0


def test_accumulating_metrics_carry_playing_time_into_the_target():
    """The ``kind`` axis, asserted rather than assumed.

    The signature is *spread*, not level. Def is runs above average, centred near zero,
    so playing more games does not raise it -- it amplifies it in whichever direction the
    player's talent points. A rate does the opposite: more playing time averages its noise
    down. So the test is the ratio of within-player SD in high-games seasons to low-games
    ones, which is exactly how the five metrics were classified off the real data
    (Def 1.73 and WAR 1.60 against OPS 0.73, wRC+ 0.69, Spd 0.81).

    This is the structural claim the rate-vs-accumulating comparison rests on.
    """
    frame = generated(n_players=1200)
    config = SimConfig()
    ratios = {RATE: [], ACCUMULATING: []}
    low, high = frame["G"].quantile(0.25), frame["G"].quantile(0.75)
    for metric in config.metrics:
        within = frame[metric.col] - frame.groupby("IDfg")[metric.col].transform("mean")
        ratios[metric.kind].append(
            float(within[frame["G"] >= high].std() / within[frame["G"] <= low].std())
        )

    assert min(ratios[ACCUMULATING]) > 1.0, ratios
    assert max(ratios[RATE]) < 1.0, ratios
    assert min(ratios[ACCUMULATING]) > max(ratios[RATE])


# --------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------


#: Known-unfixed: see :func:`test_ipw_weight_dispersion_is_a_known_gap`.
UNCALIBRATED = {"sd(ipw_weight)"}


def test_calibration_holds_on_the_defaults():
    """The baked-in constants reproduce the real censored sample within 10%.

    Every result downstream is conditional on this, so it is pinned rather than left to
    the notebook to notice. ``sd(ipw_weight)`` is excluded and pinned separately below --
    it does not hold, and the exclusion is deliberate rather than a loosened tolerance.
    """
    table = calibration_report(SimConfig(), seed=0)
    table = table.loc[~table["quantity"].isin(UNCALIBRATED), :]
    worst = table.loc[table["pct_error"].abs().idxmax()]
    assert abs(worst["pct_error"]) < 10.0, f"{worst['quantity']} off by {worst['pct_error']:.1f}%"


def test_ipw_weight_dispersion_is_a_known_gap():
    """The generator does not reproduce how much the IPW weights vary. Pinned, not fixed.

    Real weights have SD ~0.43; the generator produces ~0.18. IPW's whole leverage is in
    that spread -- it works by upweighting the seasons least likely to be seen again -- so
    with near-uniform weights the correction has nothing to do, and **this simulation
    cannot evaluate IPW.** Any IPW result read off it is a property of the generator.

    The cause is diagnosed: survival here depends on talent, which the retirement model
    cannot see, and on playing time, which it can but which cannot be strengthened without
    collapsing career length. It does not depend on *last season's performance*, which
    ``ipw.fit_retirement_model`` takes as ``perf_col`` and is built around. The generator
    omits a dependence the estimator is designed to detect.

    This test exists so the gap cannot be quietly forgotten. When the survival model gains
    that term, this should start failing -- and that failure is the fix landing.
    """
    table = calibration_report(SimConfig(), seed=0).set_index("quantity")
    simulated = table.loc["sd(ipw_weight)", "simulated"]
    assert simulated < 0.30, (
        f"ipw weight SD is {simulated:.3f}; if it now approaches the real 0.43 the "
        "generator has been fixed -- delete this test and re-enable the IPW conclusions"
    )


def test_calibration_matches_the_survivorship_mechanism():
    """Better players being observed more often *is* the mechanism under study."""
    table = calibration_report(SimConfig(), seed=0).set_index("quantity")
    row = table.loc["corr(talent, career length)"]
    assert row["simulated"] == pytest.approx(REAL_MOMENTS["talent_career_length_corr"], abs=0.05)


# --------------------------------------------------------------------------------------
# The framework itself
# --------------------------------------------------------------------------------------


def test_framework_recovers_a_known_peak():
    """A smooth on age alone finds a known peak on a clean panel.

    This is the study's foundation. With no differential survival and no qualification
    cut, ``s(0)`` has to land on the true peak; if it cannot, every number the notebook
    reports is an artefact of the framework rather than a property of the estimators.

    Note this is deliberately ``s(0)`` alone, *not* the published specification. The
    published spec misses by roughly a year even here, and separating "the framework
    works" from "the estimator is biased" is the entire point of running the check.
    """
    clean = replace(
        SimConfig(),
        n_players=2312,
        survival_on_talent=0.0,
        survival_on_playing_time=0.0,
        qual_threshold=0.0,
    )
    frame = simulate_careers(clean, seed=0)
    metric = clean.metric("wRC+")
    spec = replace(metric.spec, name=metric.col)
    df = _model_frame(frame, metric, metric.col)

    x, y = generate_data(df, spec.feature_cols, spec.target_col)
    gam = GAM(s(0, n_splines=N_SPLINES))
    gam.gridsearch(x, y, weights=df[spec.weight_col].values, lam=LAM_GRID, progress=False)

    ages = np.linspace(FIT_MIN_AGE, 39.0, 400)
    grid = np.tile(np.median(x, axis=0), (len(ages), 1))
    grid[:, 0] = ages
    recovered = ages[np.argmax(gam.predict(grid))]

    assert recovered == pytest.approx(metric.targets.peak_age, abs=0.6)


def test_study_returns_one_row_per_sim_metric_arm():
    study = run_simulation_study(
        SMALL, n_sims=2, metrics=("wRC+",), arms=(DELTA_LAG, COMPLETE_CASE)
    )
    assert len(study) == 2 * 1 * 2
    assert set(study["arm"]) == {DELTA_LAG, COMPLETE_CASE}
    assert study["peak_age_error"].notna().all()
    assert study["sbd"].between(0.0, 2.0).all()


def test_accumulating_metrics_gain_a_latent_rate_arm():
    """Skill alone is only separable where playing time multiplies the target."""
    study = run_simulation_study(SMALL, n_sims=1, metrics=("WAR", "wRC+"))
    assert "latent_rate" in set(study.loc[study["metric"] == "WAR", "arm"])
    assert "latent_rate" not in set(study.loc[study["metric"] == "wRC+", "arm"])


def test_the_talent_control_earns_its_place_under_selection():
    """Dropping ``s(1)`` should hurt badly once survival is selective.

    The control is a post-treatment variable and on a clean panel it *adds* bias, but on
    a realistically selected one it removes far more than it adds. Both halves matter, so
    the direction under selection is pinned here.
    """
    study = run_simulation_study(
        replace(SimConfig(), n_players=1200),
        n_sims=3,
        metrics=("wRC+",),
        arms=(COMPLETE_CASE, NO_TALENT_CONTROL),
    )
    errors = study.groupby("arm")["peak_age_error"].mean().abs()
    assert errors[NO_TALENT_CONTROL] > errors[COMPLETE_CASE]

    # And the published shape score agrees -- the control is not a peak-only effect.
    shapes = study.groupby("arm")["sbd"].mean()
    assert shapes[NO_TALENT_CONTROL] > shapes[COMPLETE_CASE]
