"""Simulation study: can the estimators recover a curve we already know?

Test MAE cannot grade an aging curve. The baseline ladder in :mod:`mlb_aging.baselines`
shows why -- ``persistence`` carries no age term at all, yet lands within 1.2% of every
other arm on Def and beats the GAM outright. A score a model with no aging signal nearly
wins cannot rank aging curves.

The deeper problem is that the true curve is never observed, so every quantity computed on
real data is a proxy. This module removes that constraint: generate careers from a curve
written down here, censor them with a rule written down here, run the estimators, and
measure the distance to the truth.

Nothing in this module touches the published pipeline. It builds frames shaped exactly
like the real ones and hands them to the same :mod:`mlb_aging.gam`, :mod:`mlb_aging.ipw`
and :mod:`mlb_aging.baselines` code.

Metric structure enters in one place -- the generator -- along a single axis, ``kind``:

``rate``
    OPS, wRC+, Spd. The value *is* the rate; playing time enters only its noise, whose
    variance falls as 1/n. Censoring on playing time is therefore *correlated* with the
    target.
``accumulating``
    Def, WAR. The value is ``rate x G``, so playing time is a **factor** of the target.
    Censoring on playing time cuts the target directly.

That split was measured, not assumed (see :data:`METRIC_SIMS`), and it reproduces what the
real data shows: de-meaned within player, Def and WAR spread *out* as games rise (residual
SD ratio high-G/low-G of 1.73 and 1.60) while OPS, wRC+ and Spd tighten (0.73, 0.69, 0.81).
Spd is FanGraphs' Speed Score, a 0-10 rate that happens to be G-weighted in the fit -- it
does not accumulate, and the data says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from pygam import GAM, s

from mlb_aging.baselines import delta_curve
from mlb_aging.dataset import MAX_AGE
from mlb_aging.features import add_career_mean, add_experience, add_lag
from mlb_aging.features import generate_data
from mlb_aging.gam import LAM_GRID, N_SPLINES, aging_curve, fit_gam
from mlb_aging.ipw import fit_ipw_weights
from mlb_aging.metrics import MetricSpec

#: Youngest age the model ever fits. ``add_lag`` drops each player's first season, so no
#: age-20 row survives even though ``MIN_AGE`` is 20. Curve targets are anchored here.
FIT_MIN_AGE = 21

RATE = "rate"
ACCUMULATING = "accumulating"

#: Below this many plate appearances a simulated season counts as not played at all --
#: injury, or a year in the minors. Such seasons are not rows on a major-league
#: leaderboard, so they are not rows here either, and dropping them leaves a gap that
#: ``add_experience`` and ``add_lag`` handle exactly as they handle a real missed year.
#: Without it, clipping the playing-time normal at zero leaves a 7.8% point mass at
#: PA = 0, which breaks the PA-weighted career mean and exists in no real sample.
MIN_APPEARANCE_PA = 1.0


@dataclass(frozen=True)
class CurveTargets:
    """The shape of a known aging curve, in the units the generator works in.

    ``rise`` is the gain from :data:`FIT_MIN_AGE` to the peak and ``decline_30_34`` the
    signed change from 30 to 34 -- the two quantities the notebooks actually quote, so a
    simulated curve calibrated to them is comparable to a published one term by term.
    """

    peak_age: float
    rise: float
    decline_30_34: float
    #: Vertical offset. Only the *shape* fields above carry meaning about aging; ``level``
    #: is a nuisance parameter solved by :func:`calibrate` so the censored simulated mean
    #: matches the real one. It is not the metric's peak value, and because ``s(1)`` enters
    #: the GAM additively it never moves an estimated peak age.
    level: float


def true_curve(age: np.ndarray | float, targets: CurveTargets) -> np.ndarray | float:
    """The known truth: an asymmetric quadratic, solved in closed form from ``targets``.

    Curvature is set separately either side of the peak so that the curve hits the
    configured rise and 30->34 decline exactly. The decline is quadratic, so its *slope*
    steepens with age -- matching "the decline steepens monotonically" in ``GAM.ipynb``.

    A peak at or below :data:`FIT_MIN_AGE` means the curve only ever falls over the
    observed range, which is Spd's real situation; the rising branch is then unused.
    """
    peak = targets.peak_age
    age = np.asarray(age, dtype=float)

    span_up = peak - FIT_MIN_AGE
    c_up = targets.rise / span_up**2 if span_up > 0 else 0.0

    span_down = (34.0 - peak) ** 2 - (30.0 - peak) ** 2
    if span_down <= 0:
        raise ValueError(
            f"peak_age {peak} leaves 30->34 spanning the maximum; calibrate on a "
            "different pair of ages for a curve that peaks this late"
        )
    c_down = -targets.decline_30_34 / span_down

    curvature = np.where(age <= peak, c_up, c_down)
    return targets.level - curvature * (age - peak) ** 2


@dataclass(frozen=True)
class MetricSim:
    """How one metric is generated. Shaped after :class:`~mlb_aging.metrics.MetricSpec`.

    Add a metric here, not by duplicating a code path -- the rule ``metrics.py`` already
    follows.

    ``var_a`` and ``var_b`` parameterise the season noise as ``var = var_a + var_b / n``,
    where ``n`` is the metric's own denominator: ``weight_col`` for a rate, always ``G``
    for an accumulating metric (whose noise lives on the per-game scale before being
    multiplied up). ``var_a`` is the floor that survives infinite playing time -- real
    year-to-year change in a player, not sampling error.

    ``talent_sd`` and the targets are on the **generating** scale: the value scale for a
    rate, the per-game scale for an accumulating metric.
    """

    name: str
    sim_name: str
    kind: str
    talent_sd: float
    var_a: float
    var_b: float
    targets: CurveTargets
    weight_col: str
    eval_weight_col: str = "PA"
    curve_reference: float | str | None = "train_mean"

    def __post_init__(self) -> None:
        if self.kind not in (RATE, ACCUMULATING):
            raise ValueError(f"unknown kind {self.kind!r}")

    @property
    def col(self) -> str:
        """Realized value column in the simulated frame."""
        return self.sim_name

    @property
    def rate_col(self) -> str:
        """Latent rate, before playing time and noise. Never visible to an estimator."""
        return f"{self.sim_name}_rate"

    @property
    def talent_col(self) -> str:
        """The player's latent talent, on the generating scale."""
        return f"{self.sim_name}_talent"

    @property
    def expected_col(self) -> str:
        """The season's noiseless expected value -- the metric before its noise draw.

        Averaged per player this is *exactly* what ``add_career_mean`` is trying to
        estimate, which makes it the right oracle. Latent talent alone is not: for an
        accumulating metric two players of equal skill but different playing time have
        different expected value, so talent under-determines the target and the career
        mean is legitimately the better control.
        """
        return f"{self.sim_name}_expected"

    @property
    def noise_denominator(self) -> str:
        return self.weight_col if self.kind == RATE else "G"

    @property
    def spec(self) -> MetricSpec:
        """The :class:`MetricSpec` the estimators see.

        Mirrors the real metric's fit/eval weighting, including the deliberate asymmetry
        of fitting Def, Spd and WAR G-weighted while scoring them PA-weighted. Simulated
        metrics are never centralized: ``centralize_data`` removes league-season effects
        and the generator creates none.
        """
        return MetricSpec(
            name=self.sim_name,
            centralized=False,
            weight_col=self.weight_col,
            eval_weight_col=self.eval_weight_col,
            curve_reference=self.curve_reference,
        )

    def noise_sd(self, denominator: np.ndarray) -> np.ndarray:
        """Season noise SD at the given playing time."""
        return np.sqrt(np.maximum(self.var_a + self.var_b / np.maximum(denominator, 1.0), 0.0))


# --------------------------------------------------------------------------------------
# Calibration. Every number below was measured off data/hitter_train_data.csv (1980-2019,
# ages 20-40, n=13,636 rows over 2,312 players); see the comments for the source.
# --------------------------------------------------------------------------------------

#: Curve targets for the three metrics whose published peaks are real interior maxima.
#: Read off the fitted age-only curves in ``tests/test_age_only.py``.
_OPS_TARGETS = CurveTargets(peak_age=26.991, rise=0.0480, decline_30_34=-0.0211,
                            level=-0.039314087732990055)
_WRC_TARGETS = CurveTargets(peak_age=26.991, rise=12.3215, decline_30_34=-5.4198,
                            level=90.78472035389495)
#: WAR on the per-game scale, refitted for this module: peak 27.010, rise +0.004297,
#: 30->34 change -0.004675. The peak matches WAR's own 27.01, as it must.
_WAR_TARGETS = CurveTargets(peak_age=27.010, rise=0.004297, decline_30_34=-0.004675,
                            level=0.011983663121547476)

#: Def and Spd have no usable published peak -- Def's nominal 21.99 scatters 21.0-25.0
#: under resampling and Spd's sits on the left edge of the data. **That is why they are
#: here.** The simulation *assigns* them an interior peak the estimator is never told, so
#: "can the peak be found at this sample size?" becomes a question with an answer. These
#: defaults are the starting point of that sweep, not a claim about real defensive aging.
_DEF_TARGETS = CurveTargets(peak_age=24.0, rise=0.0060, decline_30_34=-0.014814,
                            level=0.006511469351824029)
_SPD_TARGETS = CurveTargets(peak_age=23.0, rise=0.30, decline_30_34=-0.3768,
                            level=4.372745257000583)

METRIC_SIMS: tuple[MetricSim, ...] = (
    # kind, var_a and var_b measured by regressing squared within-player residuals on
    # 1/denominator; talent_sd is the between-player SD of the per-player mean.
    MetricSim("OPS", "SimOPS", RATE, talent_sd=0.088984,
              var_a=0.002919, var_b=0.912015, targets=_OPS_TARGETS, weight_col="PA"),
    MetricSim("wRC+", "SimWRC", RATE, talent_sd=23.858715,
              var_a=177.011436, var_b=70902.582313, targets=_WRC_TARGETS, weight_col="PA"),
    MetricSim("Spd", "SimSpd", RATE, talent_sd=1.577645,
              var_a=0.735823, var_b=29.127890, targets=_SPD_TARGETS, weight_col="G"),
    MetricSim("Def", "SimDef", ACCUMULATING, talent_sd=0.059853,
              var_a=0.001614, var_b=0.104943, targets=_DEF_TARGETS, weight_col="G"),
    MetricSim("WAR", "SimWAR", ACCUMULATING, talent_sd=0.010588,
              var_a=0.000079, var_b=0.004661, targets=_WAR_TARGETS, weight_col="G"),
)

#: Correlations between the five career means, per player, over the 2,312 players with all
#: five. Def is *negatively* correlated with hitting (-0.34) -- good hitters occupy the
#: easy defensive positions -- and Spd is nearly orthogonal to it (-0.02). A single shared
#: latent talent would erase both facts and make WAR a relabelled wRC+.
TALENT_CORRELATION = np.array(
    [
        # OPS    wRC+     Def     Spd     WAR
        [1.000,  0.982, -0.341, -0.022,  0.743],  # OPS
        [0.982,  1.000, -0.331, -0.009,  0.750],  # wRC+
        [-0.341, -0.331, 1.000,  0.113,  0.178],  # Def
        [-0.022, -0.009, 0.113,  1.000,  0.159],  # Spd
        [0.743,  0.750,  0.178,  0.159,  1.000],  # WAR
    ]
)

#: Debut ages and their counts, 1980-2019. Mode 24 (514 players); only 46 debut at 20.
DEBUT_AGES = np.arange(20, 41)
DEBUT_WEIGHTS = np.array(
    [46, 126, 270, 397, 514, 466, 352, 248, 153, 99, 75, 56, 30, 20, 19, 11, 8, 4, 2, 1, 2],
    dtype=float,
)

#: Survival to the next season, by two-year age bucket, over qualified seasons. Flat near
#: 0.82 until the early 30s, then falling away: 0.782 at 32, 0.709 at 34, 0.581 at 38.
SURVIVAL_AGES = np.array([20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40], dtype=float)
SURVIVAL_RATES = np.array(
    [0.871, 0.829, 0.828, 0.814, 0.817, 0.828, 0.782, 0.709, 0.651, 0.581, 0.400]
)

#: Plate appearances per game: 3.5113 on average, and near-constant in age (3.66 at 21 to
#: 3.47 at 39), so a single ratio with noise reproduces the observed G given PA.
PA_PER_GAME = 3.5113
PA_PER_GAME_SD = 0.7434


@dataclass(frozen=True)
class PlayingTimeModel:
    """How playing time responds to talent, last season, and age.

    Coefficients come from regressing PA on the wRC+ career mean, the lagged wRC+ and
    ``max(0, Age - 33)``: ``PA = 0.72 + 3.161*career_mean + 1.107*lag - 20.833*age_pen``,
    residual SD 158.8, R^2 0.247.

    **The one constant that cannot be measured.** That regression was fitted on qualified
    seasons only, so it describes the PA >= 100 world and says nothing about how far below
    100 a collapsing player falls. ``intercept_shift`` is therefore a free parameter,
    tuned so the *censored* simulated PA distribution matches the real one -- calibrating
    on the part that is observable. It is the study's main modelling assumption and the
    notebook states it as such.

    The dependence on **realized** prior performance, not on latent talent, is what makes
    the censoring informative: a fluke bad season costs playing time, which is exactly the
    mechanism that removes declining players before they retire.
    """

    #: 0.72 + 3.161*100 + 1.107*100 = 427.52 -- the raw fit rewritten around a league-average
    #: player, so that retuning a metric's level no longer drags playing time with it.
    intercept: float = 427.52
    on_talent: float = 3.161
    on_lag: float = 1.107
    on_age_penalty: float = -20.833
    age_knee: float = 33.0
    resid_sd: float = 158.81
    #: Solved by :func:`calibrate` against the censored PA mean.
    intercept_shift: float = -92.1219
    max_pa: float = 778.0

    def expected_pa(
        self, talent_wrc: np.ndarray, lag_deviation: np.ndarray, age: np.ndarray
    ) -> np.ndarray:
        """Expected PA from talent and last season, both as deviations from average."""
        penalty = np.maximum(0.0, age - self.age_knee)
        return (
            self.intercept
            + self.intercept_shift
            + self.on_talent * talent_wrc
            + self.on_lag * lag_deviation
            + self.on_age_penalty * penalty
        )


@dataclass(frozen=True)
class SimConfig:
    """Everything the generator needs. Defaults reproduce the real sample's shape."""

    n_players: int = 2312
    first_season: int = 1980
    n_seasons: int = 40
    qual_threshold: float = 100.0
    metrics: tuple[MetricSim, ...] = METRIC_SIMS
    talent_correlation: np.ndarray = field(default_factory=lambda: TALENT_CORRELATION)
    playing_time: PlayingTimeModel = field(default_factory=PlayingTimeModel)
    #: How strongly survival responds to talent. This *is* the survivorship mechanism --
    #: better players are observed more -- so :func:`calibrate` fits it against the real
    #: correlation between career mean and career length (0.4685), rather than guessing.
    survival_on_talent: float = 0.5161
    survival_on_playing_time: float = 0.4
    #: Added to the survival logit. :data:`SURVIVAL_RATES` was measured on *qualified*
    #: seasons, so it is P(qualifies again | qualified) -- not P(still playing). A player
    #: who drops to 60 PA and returns counts as dead there and alive here. The true hazard
    #: is therefore gentler than the measured one, and this closes the gap; :func:`calibrate`
    #: solves it against observed career length. It is the same conflation the survival
    #: label makes, showing up as a calibration constant.
    survival_boost: float = 0.9093
    #: Multiplies the PA residual SD. The real regression saw only PA >= 100, so it
    #: understates the spread of the uncensored distribution.
    pa_sd_scale: float = 1.3536
    max_games: float = 162.0

    def with_targets(self, metric_name: str, targets: CurveTargets) -> "SimConfig":
        """Return a copy with one metric's true curve replaced.

        This is how the Def and Spd peak sweeps work: assign a peak, regenerate, and ask
        whether the estimator finds it.
        """
        metrics = tuple(
            replace(m, targets=targets) if m.name == metric_name else m for m in self.metrics
        )
        return replace(self, metrics=metrics)

    def metric(self, name: str) -> MetricSim:
        for m in self.metrics:
            if m.name == name or m.sim_name == name:
                return m
        raise KeyError(f"unknown metric {name!r}; expected one of {[m.name for m in self.metrics]}")


def _draw_talents(config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw one correlated talent vector per player, on each metric's own scale."""
    sds = np.array([m.talent_sd for m in config.metrics])
    corr = np.asarray(config.talent_correlation, dtype=float)
    if corr.shape != (len(config.metrics), len(config.metrics)):
        raise ValueError("talent_correlation must be square in the number of metrics")

    cov = corr * np.outer(sds, sds)
    # Nearest-PSD guard: a measured correlation matrix can be slightly indefinite.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    cov = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
    return rng.multivariate_normal(np.zeros(len(config.metrics)), cov, size=config.n_players)


def _survival_probability(
    config: SimConfig, age: float, talent_z: float, pa: float, rng: np.random.Generator
) -> float:
    """P(the player appears again next season), on the measured age profile."""
    base = float(np.interp(age, SURVIVAL_AGES, SURVIVAL_RATES))
    base = min(max(base, 1e-3), 1 - 1e-3)
    logit = np.log(base / (1 - base))
    logit += config.survival_boost
    logit += config.survival_on_talent * talent_z
    logit += config.survival_on_playing_time * (pa - 400.0) / 200.0
    return float(1.0 / (1.0 + np.exp(-logit)))


def simulate_careers(config: SimConfig | None = None, seed: int = 0) -> pd.DataFrame:
    """Generate one synthetic league: every season lived, censored or not.

    Returns one row per player-season with ``IDfg, Season, Name, Age, G, PA``, each
    metric's realized value, latent rate and true talent, plus two flags kept
    deliberately separate:

    ``qualified``
        ``PA >= qual_threshold`` -- whether the season is *visible* to an estimator.
    ``retired``
        whether the player's career has actually ended.

    Their difference is exactly what :func:`~mlb_aging.ipw.add_survived_label` conflates on
    real data, where a drop to 60 PA is labelled identically to retirement. Here the two
    are known separately, so the cost of conflating them is measurable.
    """
    config = config or SimConfig()
    rng = np.random.default_rng(seed)

    talents = _draw_talents(config, rng)
    wrc = config.metric("wRC+")
    wrc_index = [m.name for m in config.metrics].index("wRC+")

    debut_probs = DEBUT_WEIGHTS / DEBUT_WEIGHTS.sum()
    debut_ages = rng.choice(DEBUT_AGES, size=config.n_players, p=debut_probs)
    last_season = config.first_season + config.n_seasons - 1
    debut_seasons = rng.integers(config.first_season, last_season + 1, size=config.n_players)

    records = []
    for player in range(config.n_players):
        talent = talents[player]
        talent_wrc = talent[wrc_index]
        talent_z = talent_wrc / wrc.talent_sd
        age = float(debut_ages[player])
        season = int(debut_seasons[player])
        # A debut season has no prior year; anchor playing time on the curve itself.
        prev_value = float(true_curve(age, wrc.targets)) + talent_wrc

        while age <= MAX_AGE and season <= last_season:
            expected = config.playing_time.expected_pa(
                np.array([talent_wrc]),
                np.array([prev_value - wrc.targets.level]),
                np.array([age]),
            )[0]
            pa = float(
                np.clip(
                    expected + rng.normal(0.0, config.playing_time.resid_sd * config.pa_sd_scale),
                    0.0,
                    config.playing_time.max_pa,
                )
            )
            ratio = max(rng.normal(PA_PER_GAME, PA_PER_GAME_SD), 1.0)
            # Floored at one: the player appeared, so he played at least one game. A zero
            # here becomes a zero fit weight, which pyGAM inverts.
            games = float(np.clip(round(pa / ratio), 1.0, config.max_games))

            row = {
                "IDfg": player,
                "Season": season,
                "Name": f"P{player:05d}",
                "Age": age,
                "PA": pa,
                "G": games,
            }
            for index, metric in enumerate(config.metrics):
                rate = float(true_curve(age, metric.targets)) + talent[index]
                denominator = pa if metric.noise_denominator == "PA" else games
                noise = float(rng.normal(0.0, metric.noise_sd(np.array([denominator]))[0]))
                row[metric.rate_col] = rate
                row[metric.talent_col] = talent[index]
                row[metric.expected_col] = rate if metric.kind == RATE else rate * games
                row[metric.col] = rate + noise if metric.kind == RATE else (rate + noise) * games
            records.append(row)

            prev_value = row[wrc.col]
            p_survive = _survival_probability(config, age, talent_z, pa, rng)
            if rng.random() > p_survive:
                break
            age += 1.0
            season += 1

    frame = pd.DataFrame.from_records(records)
    frame = frame.loc[frame["PA"] >= MIN_APPEARANCE_PA, :]

    frame["qualified"] = frame["PA"] >= config.qual_threshold
    # A player's career ends after his last *appearance*, which is why the flags are set
    # after the non-appearances are dropped.
    frame["retired"] = frame["Season"] == frame.groupby("IDfg")["Season"].transform("max")
    return frame.sort_values(["Season", "Name"]).reset_index(drop=True)


def censor(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame an estimator is allowed to see -- qualified seasons only."""
    return frame.loc[frame["qualified"], :].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Calibration against the observable moments
# --------------------------------------------------------------------------------------

#: Moments of the real qualified training frame (1980-2019, 13,636 rows over 2,312
#: players), the targets :func:`calibrate` solves against.
REAL_MOMENTS = {
    "n_rows": 13636,
    "n_players": 2312,
    "PA_mean": 412.5645,
    "PA_sd": 183.0,
    "G_mean": 111.7258,
    "career_length_mean": 5.897,
    #: Correlation between a player's career mean and how many qualified seasons he gets.
    #: This IS the survivorship mechanism -- better players are observed more -- so it is
    #: fitted rather than guessed.
    "talent_career_length_corr": 0.4685,
    #: SD of ``ipw_weight`` on the qualified frame (wRC+; the five metrics run 0.428-0.477).
    #: This is IPW's whole leverage -- the correction works by upweighting the seasons least
    #: likely to be seen again, so if the weights are near-uniform there is nothing for it to
    #: do. Calibrating the survivorship *mean* without its *dispersion* makes the IPW arm
    #: unevaluable, which is exactly what happened before this target was added.
    "ipw_weight_sd": 0.4277,
    "metric_mean": {"OPS": -0.0206, "wRC+": 96.3714, "Def": -0.6189, "Spd": 3.8926, "WAR": 1.4639},
    "metric_sd": {"OPS": 0.1112, "wRC+": 29.1712, "Def": 9.0195, "Spd": 1.8099, "WAR": 1.9524},
}


def _talent_career_length_corr(observed: pd.DataFrame, metric_name: str = "SimWRC") -> float:
    """Correlation between latent talent and observed career length -- the mechanism."""
    per_player = observed.groupby("IDfg").agg(
        talent=(f"{metric_name}_talent", "first"), seasons=("Season", "size")
    )
    return float(per_player["talent"].corr(per_player["seasons"]))


def _ipw_weight_sd(observed: pd.DataFrame, metric_name: str = "wRC+") -> float:
    """SD of the IPW weights the survival model produces on the simulated qualified frame."""
    metric = SimConfig().metric(metric_name)
    spec = replace(metric.spec, name=metric.col)
    frame = _model_frame(observed, metric, metric.col)
    return float(fit_ipw_weights(frame, spec)["ipw_weight"].std())


def calibration_report(
    config: SimConfig | None = None, seed: int = 0, moments: dict | None = None
) -> pd.DataFrame:
    """Simulated moments beside the real ones, on the *censored* frame.

    The censored frame is the only fair comparison: it is all the real data ever shows.
    Everything downstream is conditional on this table holding up, so the notebook prints
    it before any result.
    """
    config = config or SimConfig()
    moments = moments or REAL_MOMENTS
    observed = censor(simulate_careers(config, seed=seed))

    rows = [
        {"quantity": "rows", "real": moments["n_rows"], "simulated": len(observed)},
        {"quantity": "players", "real": moments["n_players"], "simulated": observed["IDfg"].nunique()},
        {"quantity": "PA mean", "real": moments["PA_mean"], "simulated": observed["PA"].mean()},
        {"quantity": "PA sd", "real": moments["PA_sd"], "simulated": observed["PA"].std()},
        {"quantity": "G mean", "real": moments["G_mean"], "simulated": observed["G"].mean()},
        {
            "quantity": "career length",
            "real": moments["career_length_mean"],
            "simulated": observed.groupby("IDfg").size().mean(),
        },
        {
            "quantity": "corr(talent, career length)",
            "real": moments["talent_career_length_corr"],
            "simulated": _talent_career_length_corr(observed),
        },
        {
            "quantity": "sd(ipw_weight)",
            "real": moments["ipw_weight_sd"],
            "simulated": _ipw_weight_sd(observed),
        },
    ]
    for metric in config.metrics:
        rows.append(
            {
                "quantity": f"{metric.name} mean",
                "real": moments["metric_mean"][metric.name],
                "simulated": observed[metric.col].mean(),
            }
        )
        rows.append(
            {
                "quantity": f"{metric.name} sd",
                "real": moments["metric_sd"][metric.name],
                "simulated": observed[metric.col].std(),
            }
        )

    table = pd.DataFrame(rows)
    scale = table["real"].abs().replace(0.0, np.nan)
    table["pct_error"] = (table["simulated"] - table["real"]) / scale * 100.0
    return table


def calibrate(
    config: SimConfig | None = None,
    seed: int = 0,
    moments: dict | None = None,
    rounds: int = 3,
) -> SimConfig:
    """Solve the two free constants against the observable moments.

    Two things genuinely cannot be read off the data and so are fitted here:

    ``intercept_shift``
        The real PA regression was fitted on qualified seasons, so it describes the
        PA >= 100 world and is silent about how far a collapsing player actually falls.
        Bisected so the *censored* simulated PA mean matches the real 412.6.
    each metric's ``level``
        The curve targets fix a metric's *shape* -- rise and decline -- but its height is
        the peak value at an arbitrary career-mean reference, not the mean of a censored
        sample. Shifted so the censored simulated mean matches the real one.

    Everything else in the module is measured. Fitting the level and the shift to observed
    moments is the standard move: match the part of the distribution that is visible, and
    let the generating process imply the rest.
    """
    config = config or SimConfig()
    moments = moments or REAL_MOMENTS

    def bisect(cfg, low, high, apply, measure, target, steps=16):
        for _ in range(steps):
            middle = (low + high) / 2.0
            value = measure(censor(simulate_careers(apply(cfg, middle), seed=seed)))
            if value < target:
                low = middle
            else:
                high = middle
        return apply(cfg, (low + high) / 2.0)

    for _ in range(rounds):
        config = bisect(
            config, -400.0, 200.0,
            lambda c, v: replace(c, playing_time=replace(c.playing_time, intercept_shift=v)),
            lambda f: f["PA"].mean(), moments["PA_mean"],
        )
        config = bisect(
            config, -2.0, 4.0,
            lambda c, v: replace(c, survival_boost=v),
            lambda f: f.groupby("IDfg").size().mean(), moments["career_length_mean"],
        )
        config = bisect(
            config, 0.3, 3.0,
            lambda c, v: replace(c, pa_sd_scale=v),
            lambda f: f["PA"].std(), moments["PA_sd"],
        )
        config = bisect(
            config, 0.0, 2.5,
            lambda c, v: replace(c, survival_on_talent=v),
            _talent_career_length_corr, moments["talent_career_length_corr"],
        )
        # Survival's *dispersion*, which the correlation above does not constrain. Playing
        # time is a direct feature of the retirement model while talent is not, so this is
        # the coefficient that moves the spread of predicted survival -- raising
        # survival_on_talent instead narrows it and breaks the correlation target.
        config = bisect(
            config, 0.0, 12.0,
            lambda c, v: replace(c, survival_on_playing_time=v),
            _ipw_weight_sd, moments["ipw_weight_sd"],
        )

        observed = censor(simulate_careers(config, seed=seed))
        metrics = []
        for metric in config.metrics:
            gap = moments["metric_mean"][metric.name] - observed[metric.col].mean()
            # An accumulating metric's level lives on the per-game scale.
            step = gap / observed["G"].mean() if metric.kind == ACCUMULATING else gap
            metrics.append(
                replace(metric, targets=replace(metric.targets, level=metric.targets.level + step))
            )
        config = replace(config, metrics=tuple(metrics))

    return config


# --------------------------------------------------------------------------------------
# The estimator arms, and scoring them against the truth
# --------------------------------------------------------------------------------------

#: Ages the curves are compared on. Starts at :data:`FIT_MIN_AGE` because ``add_lag`` drops
#: every player's debut season, and stops at 39 where the real data thins to ~180 rows.
COMPARISON_AGES = np.linspace(FIT_MIN_AGE, 39.0, 200)

COMPLETE_CASE = "complete_case"
COMPLETE_CASE_IPW = "complete_case_ipw"
DELTA_LAG = "delta_lag"
LATENT_RATE = "latent_rate"
NO_TALENT_CONTROL = "no_talent_control"

#: The comparable arms, in the order the notebook reports them. Every one is fitted on the
#: censored frame -- the censoring's cost is isolated by toggling ``qual_threshold`` in the
#: ablation instead, which is cleaner than carrying uncensored arms through the study.
ARMS = (COMPLETE_CASE, COMPLETE_CASE_IPW, NO_TALENT_CONTROL, DELTA_LAG)


@dataclass(frozen=True)
class TruthCurve:
    """The curve the estimator is trying to recover, on :data:`COMPARISON_AGES`."""

    ages: np.ndarray
    values: np.ndarray

    @property
    def peak_age(self) -> float:
        return float(self.ages[np.argmax(self.values)])

    def value_at(self, age: float) -> float:
        return float(np.interp(age, self.ages, self.values))

    def change_between(self, start: float, end: float) -> float:
        return self.value_at(end) - self.value_at(start)


def truth_curve(
    frame: pd.DataFrame, metric: MetricSim, ages: np.ndarray = COMPARISON_AGES
) -> TruthCurve:
    """What the estimator *should* find, in the metric's own observed units.

    For a rate that is just :func:`true_curve`. For an accumulating metric the estimator
    sees ``rate x G``, so the truth is the latent rate multiplied by the **uncensored**
    mean games at each age -- skill times opportunity.

    That product is why an accumulating metric's peak need not sit where its latent rate
    peaks: if games fall with age, the value curve turns earlier than the skill curve. The
    simulation knows both, so :func:`peak_age` here is computed from the product rather
    than read off ``targets.peak_age``.
    """
    rate = np.asarray(true_curve(ages, metric.targets), dtype=float)
    if metric.kind == RATE:
        return TruthCurve(ages=ages, values=rate)

    games = frame.groupby("Age")["G"].mean()
    profile = np.interp(ages, games.index.values.astype(float), games.values)
    return TruthCurve(ages=ages, values=rate * profile)


def _z_normalize(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    if spread == 0:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def shape_based_distance(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Shape-based distance, as Paparrizos & Gravano (2015) define it.

    SBD is a *sliding* measure: the coefficient-normalised cross-correlation is evaluated
    at every shift and the best one is taken, so two curves of the same shape score 0 even
    if one is translated. Both series are z-normalised first, and the result lies in
    [0, 2] with 0 identical.

    Used as the shape score in arXiv 2110.14017's aging-curve simulations.
    """
    x, y = _z_normalize(np.asarray(estimate, float)), _z_normalize(np.asarray(truth, float))
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.correlate(x, y, mode="full").max() / denominator)


def shape_corr_distance(estimate: np.ndarray, truth: np.ndarray) -> float:
    """``1 - Pearson correlation`` of the two curves. **Not** SBD -- a deliberate variant.

    SBD maximises over shifts, which makes it blind to translation. For an aging curve a
    translation *is* a wrong peak age, so a shift-invariant score forgives the error the
    curve exists to answer. This is SBD's zero-lag term alone, which does not forgive it.

    Reported beside :func:`shape_based_distance` rather than instead of it: the published
    metric is the published metric, and this is a deviation that has to argue for itself.
    """
    x = np.asarray(estimate, float) - np.mean(estimate)
    y = np.asarray(truth, float) - np.mean(truth)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return float("nan")
    return float(1.0 - np.dot(x, y) / denominator)


def curve_errors(
    ages: np.ndarray, values: np.ndarray, truth: TruthCurve
) -> dict[str, float]:
    """Distance from an estimated curve to the truth.

    The first three are the aging-curve literature's own scores: **curve-versus-curve MAE**
    (Nguyen & Matthews 2024), **RMSE** and **SBD** (arXiv 2110.14017). Note MAE and RMSE
    here compare a *curve to a curve*, which is a different quantity from the project's
    published test MAE of a prediction to a season.

    ``shape_corr_distance``, ``peak_age_error`` and ``decline_error`` are **not** from that
    literature. They are reported because the peak age is the number this project
    publishes, but they do not outrank the three above.

    Both curves are centred before every level-sensitive comparison. That is forced by the
    pipeline, not a change to the metrics: ``aging_curve`` pins the career mean at an
    arbitrary ``curve_reference`` and the delta method's cumulative sum starts from zero,
    so a raw level difference would score the reference convention rather than the
    estimator. ``peak_age_error`` is read at the estimator's own resolution, before
    interpolation, so a coarse comparison grid cannot blur a peak.
    """
    peak_age = float(ages[np.argmax(values)])
    resampled = np.interp(truth.ages, ages, values)

    centred_estimate = resampled - resampled.mean()
    centred_truth = truth.values - truth.values.mean()
    residual = centred_estimate - centred_truth

    estimated_decline = float(np.interp(34.0, ages, values) - np.interp(30.0, ages, values))

    return {
        # The literature's metrics.
        "curve_mae": float(np.mean(np.abs(residual))),
        "curve_rmse": float(np.sqrt(np.mean(residual**2))),
        "sbd": shape_based_distance(resampled, truth.values),
        # Ours, reported but subordinate.
        "shape_corr_distance": shape_corr_distance(resampled, truth.values),
        "peak_age": peak_age,
        "peak_age_error": peak_age - truth.peak_age,
        "decline": estimated_decline,
        "decline_error": estimated_decline - truth.change_between(30.0, 34.0),
    }


def _model_frame(frame: pd.DataFrame, metric: MetricSim, target_col: str) -> pd.DataFrame:
    """Run the real feature pipeline over a simulated frame.

    Same order as :func:`~mlb_aging.dataset.load_data`: experience, then the lag, then the
    career mean. ``add_lag`` is positional, so the frame must already be in season order --
    :func:`simulate_careers` returns it that way.
    """
    spec = replace(metric.spec, name=target_col)
    df = add_experience(frame)
    df = add_lag(df, col=spec.target_col)
    return add_career_mean(df, spec.target_col)


def fit_arm(
    generated: pd.DataFrame, metric: MetricSim, arm: str
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Fit one arm and return its traced curve, plus the sample it saw.

    There is no train/test split: the question is curve-versus-truth, not held-out
    prediction, so every arm is fitted on everything it is allowed to see.
    """
    frame = censor(generated)
    target_col = metric.rate_col if arm == LATENT_RATE else metric.col
    spec = replace(metric.spec, name=target_col)

    df = _model_frame(frame, metric, target_col)

    n_rows, n_players = len(df), int(df["IDfg"].nunique())

    if arm == DELTA_LAG:
        curve = delta_curve(df, spec).curve
        return curve.index.values.astype(float), curve.values.astype(float), n_rows, n_players

    weights = None
    if arm == COMPLETE_CASE_IPW:
        weights = fit_ipw_weights(df, spec)["ipw_final_weight"].values

    if arm == NO_TALENT_CONTROL:
        gam = _fit_without_talent_control(df, spec)
    else:
        gam = fit_gam(df, spec, weights=weights)

    traced = aging_curve(gam, df, spec).by_age
    return traced.index.values.astype(float), traced.values.astype(float), n_rows, n_players


def _fit_without_talent_control(df: pd.DataFrame, spec: MetricSpec) -> GAM:
    """The published spec with ``s(1)`` removed -- everything else identical.

    The career mean is a *post-treatment* control: it averages a player over the seasons
    he actually played, so it encodes which ages those were. Conditioning on it therefore
    reaches into the age effect it is meant to leave alone. Dropping it is how the study
    prices that.

    Term indices are positional into ``feature_cols`` and this drops a term rather than
    reordering the list, so ``s(3)`` still reads the lag -- the same discipline
    ``gam.build_gam`` documents.
    """
    x, y = generate_data(df, spec.feature_cols, spec.target_col)
    gam = GAM(s(0, n_splines=N_SPLINES) + s(3, n_splines=N_SPLINES))
    gam.gridsearch(x, y, weights=df[spec.weight_col].values, lam=LAM_GRID, progress=False)
    return gam


def run_simulation_study(
    config: SimConfig | None = None,
    n_sims: int = 25,
    metrics: tuple[str, ...] | None = None,
    arms: tuple[str, ...] = ARMS,
    seed: int = 0,
) -> pd.DataFrame:
    """Replicate the whole comparison and return one row per (sim, metric, arm).

    Careers are generated once per replicate and shared across every metric, so the
    metrics are compared on identical players under identical censoring -- which is what
    makes the rate-versus-accumulating contrast controlled rather than five studies.

    Replication is the point. A single simulated dataset would leave exactly the "is this
    movement or noise?" ambiguity that makes single-fit comparisons unreadable.
    """
    config = config or SimConfig()
    selected = [m for m in config.metrics if metrics is None or m.name in metrics]

    records = []
    for sim in range(n_sims):
        generated = simulate_careers(config, seed=seed + sim)
        for metric in selected:
            truth = truth_curve(generated, metric)
            arm_list = list(arms)
            if metric.kind == ACCUMULATING and LATENT_RATE not in arm_list:
                arm_list.append(LATENT_RATE)
            for arm in arm_list:
                # The latent rate is skill alone, so it is scored against the rate curve.
                target = (
                    TruthCurve(COMPARISON_AGES, np.asarray(true_curve(COMPARISON_AGES, metric.targets)))
                    if arm == LATENT_RATE
                    else truth
                )
                ages, values, n_rows, n_players = fit_arm(generated, metric, arm)
                records.append(
                    {
                        "sim": sim,
                        "metric": metric.name,
                        "kind": metric.kind,
                        "arm": arm,
                        "true_peak_age": target.peak_age,
                        **curve_errors(ages, values, target),
                        "n_rows": n_rows,
                        "n_players": n_players,
                    }
                )
    return pd.DataFrame(records)


def summarize_study(study: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD of each error, by metric and arm."""
    grouped = study.groupby(["metric", "kind", "arm"], sort=False)
    summary = grouped.agg(
        # Published metrics first.
        curve_mae=("curve_mae", "mean"),
        curve_rmse=("curve_rmse", "mean"),
        sbd=("sbd", "mean"),
        sbd_se=("sbd", lambda s: s.std() / len(s) ** 0.5),
        # Ours, subordinate.
        shape_corr=("shape_corr_distance", "mean"),
        peak_err_mean=("peak_age_error", "mean"),
        peak_err_se=("peak_age_error", lambda s: s.std() / len(s) ** 0.5),
        decline_err=("decline_error", "mean"),
        n_rows=("n_rows", "mean"),
    )
    return summary.reset_index()


def peak_recovery_sweep(
    metric_name: str,
    peak_ages: tuple[float, ...] = (22.0, 24.0, 26.0),
    rise_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
    config: SimConfig | None = None,
    n_sims: int = 5,
    arm: str = COMPLETE_CASE,
    seed: int = 0,
) -> pd.DataFrame:
    """Is a peak of a given height, at a given age, findable at this sample size?

    ``CLAUDE.md`` concludes Def's peak "is not measurable at this sample size -- a
    resolution problem no change of specification addresses". That was inferred from
    resampling scatter, never demonstrated, because the real peak is unknown. Here it is
    assigned: put a peak of known height at a known age, and see whether the estimator
    finds it.

    The answer is a surface, not a number -- a tall peak is findable where a shallow one
    is not -- so both the age and the height are swept. ``rise_scales`` multiplies the
    metric's configured rise, so 1.0 is the height the real fitted curve suggests.
    """
    config = config or SimConfig()
    base = config.metric(metric_name)

    records = []
    for peak_age in peak_ages:
        for scale in rise_scales:
            targets = replace(
                base.targets, peak_age=peak_age, rise=base.targets.rise * scale
            )
            trial = config.with_targets(metric_name, targets)
            for sim in range(n_sims):
                generated = simulate_careers(trial, seed=seed + sim)
                metric = trial.metric(metric_name)
                truth = truth_curve(generated, metric)
                ages, values, _, _ = fit_arm(generated, metric, arm)
                errors = curve_errors(ages, values, truth)
                records.append(
                    {
                        "metric": metric_name,
                        "assigned_peak": peak_age,
                        "rise_scale": scale,
                        "rise": targets.rise,
                        "sim": sim,
                        "true_peak": truth.peak_age,
                        "recovered_peak": errors["peak_age"],
                        "peak_age_error": errors["peak_age_error"],
                    }
                )
    return pd.DataFrame(records)


def peak_recovery_summary(sweep: pd.DataFrame, tolerance: float = 1.0) -> pd.DataFrame:
    """Collapse a sweep to bias, scatter and a hit rate within ``tolerance`` years."""
    grouped = sweep.groupby(["assigned_peak", "rise_scale"], sort=True)
    summary = grouped.agg(
        rise=("rise", "first"),
        bias=("peak_age_error", "mean"),
        scatter=("peak_age_error", "std"),
        hit_rate=("peak_age_error", lambda e: float((e.abs() <= tolerance).mean())),
    )
    return summary.reset_index()
