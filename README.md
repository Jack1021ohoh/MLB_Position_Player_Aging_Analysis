# MLB Position Player Aging Analysis

A comprehensive statistical analysis of MLB position player performance trends across career trajectories using Generalized Additive Models (GAM) with survivorship bias correction.

## Overview

This project constructs aging curves for MLB position players using historical data from 1980 to 2025. By applying Generalized Additive Models (GAM) with Inverse Probability Weighting (IPW) to correct for survivorship bias, this analysis reveals how player performance metrics evolve with age, providing actionable insights for player evaluation, contract negotiations, and roster construction.

## Key Findings

- **Peak Offensive Age**: Position players reach their offensive peak around **age 27** for both OPS and wRC+, matching the 27–28 range reported across the sabermetric literature
- **Overall Value Peak**: WAR contribution also peaks around **age 27**, though it rises more gently and holds a plateau across the late 20s
- **Defense Has No Locatable Peak**: The Def curve nominally turns at **age 22**, but the rise from the youngest fitted age to that point is only **+0.10 runs** — and refitting on random halves of the players moves it across a **3–4 year range**, with roughly half the resamples finding no interior peak at all. Defense is best described as declining from the start of a major-league career, not as peaking at a particular age
- **Speed Never Peaks in the Majors**: The Spd curve declines monotonically from **age 21** — but 21 is simply the youngest age fitted, since `add_lag` drops every player's first season. The maximum sits on the left edge of the data, not at a turning point, so speed is *already* declining when players arrive and its true peak precedes the majors
- **Survivorship Bias**: IPW correction consistently improves predictive accuracy across all metrics, with the largest gain in WAR (4.1% MAE reduction)
- **Gain over a Naive Baseline**: Measured against the delta method applied to each player's prior season, GAM + IPW reduces test MAE by **8.7% for OPS** and **7.3% for wRC+**. WAR gains 4.3% — almost all of it from the survivorship correction rather than the model itself. Def does not beat the baseline, and that is reported rather than hidden
- **Elite Player Differences**: Top players (career wRC+ ≥ 110) sustain hitting a full year longer, peaking at **age 28**; elite WAR players peak at 27 like the population but from a much higher level
- **Decline Schedule**: The average position player loses **about 0.65 WAR between ages 30 and 34** (~0.16 WAR per year) and does not reach replacement level until **age 38** — the numbers a contract spanning the decline phase actually turns on. Peak age says where the curve turns; the decline schedule says what a multi-year deal is buying
- **Decline Phase**: Decline is underway from the late 20s rather than beginning at 35, and it steepens monotonically with age — each two-year step from 28 onward costs more WAR than the one before it

## Performance Metrics Analyzed

- **OPS (On-base Plus Slugging)**: Overall offensive production
- **wRC+ (Weighted Runs Created Plus)**: Context-adjusted offensive value
- **Def (Defensive Runs Saved)**: Defensive contribution
- **Spd (Speed Score)**: Base running ability
- **WAR (Wins Above Replacement)**: Overall player value

## Methodology

### Data Collection
- **Source**: FanGraphs, via the pybaseball API — last successfully fetched **2026-03-17**
- **Training Data**: 1980–2019 seasons
- **Test Data**: 2021–2025 seasons
- **Qualification**: Minimum 100 plate appearances per season
- **Age Range**: Players aged 20–40

> **The fetch step no longer runs.** FanGraphs placed the site behind a Cloudflare bot challenge
> in April 2026 and pybaseball is unmaintained; see [Data collection](#data-collection) below.
> The committed CSVs reproduce every result here exactly, so this blocks refreshing the data,
> not reproducing the analysis.

### Statistical Approaches

#### 1. Generalized Additive Models (GAM)
- Features: a smooth on age, a career mean (player talent control), and lagged prior-season performance
- Weighted by plate appearances (hitting metrics) or games played (Def, Spd, WAR)
- Hyperparameters tuned via grid search over 20 log-spaced λ values
- Evaluated on held-out 2021–2025 test seasons

#### The age × experience tensor, and why it is not the default

An earlier specification replaced the age smooth with an **age × experience tensor product**,
`te(0, 2)`. It is still available as `--model-spec tensor`, and on raw accuracy it is **slightly
better**: about 0.4 percentage points on OPS, 0.2 on wRC+, 1.0 on Spd and 1.4 on Def when measured
against the naive baseline. (WAR is the exception — with the IPW correction applied, the age-only
model is 0.6 points *better*.) Experience does carry real information that age alone does not.

The problem is what it costs the aging curve itself. Tracing a curve means holding every
non-age feature fixed, and experience cannot be held fixed independently of age — a 30-year-old
in his tenth season and a 30-year-old in his second are different players, and the curve has to
choose. The code pins `experience = age − 20`, i.e. a player who debuted at 20 and never missed a
season. That is a coherent career, but it is also the **rarest** one in the data: 43 of ~2,300
training players debuted at 20, against 428 at the modal age of 24. Retrace the same fitted model
assuming a debut at 24 and the peak moves by one to three years — OPS from 26.0 to 27.3, Def from
23.0 all the way to 26.0.

So the tensor makes the model marginally sharper at predicting individual seasons, and makes the
general aging curve markedly **less intuitive**: it no longer has one peak age, it has a peak age
*per assumed debut year*, and no principled way to choose among them. Reporting several peak ages
for one metric is not a useful answer to "when do hitters peak?".

The age-only specification is also what standard practice does. The GAM approach in Baseball
Prospectus's *"The Delta Method, Revisited"* and the FanGraphs Sabermetrics Library both smooth on
age and let the career-average control absorb experience implicitly, precisely because age,
experience and service time are near-collinear. Adopting it costs roughly half a percentage point
of accuracy and buys one peak age per metric, landing in the literature's 27–28 range.

`curve_validation.ipynb` documents all of this: the debut-age sensitivity in section 3, the
side-by-side accuracy comparison in section 4, and the resampling check in section 5.

#### 2. Survivorship Bias Correction (IPW)
Standard datasets only include players who meet the 100 PA threshold, creating positive selection at older ages — only above-average players retain playing time. IPW corrects for this by upweighting player-seasons that are unlikely to be observed:

- A logistic regression model estimates `P(player appears next season)` using age, playing time, experience, and lagged metric performance
- IPW weight = `1 / P(survive)`, normalized and multiplied into the GAM sample weights
- Separate survival models are trained per metric, allowing `perf_col` to distinguish declining players from those with low games played for other reasons (e.g. defensive specialists)
- Adding the lagged metric value to the survival model most improved WAR (AUC +0.028); PA-weighted metrics (OPS, wRC+) saw minimal change because plate appearances already proxy hitting quality

#### 3. Elite Player Analysis
Separate models trained on top players to assess whether aging curves differ from the general population:
- **Hitting specialists**: career wRC+ ≥ 110
- **Overall value**: career WAR ≥ 2.5

#### 4. Baseline Comparison
"The GAM improves accuracy" is meaningless without something to improve on, so the model is
measured against the delta method — the standard naive approach, in which year-over-year changes
are averaged by age (weighted by the harmonic mean of the two seasons' playing time) and
accumulated into a curve.

Three baselines are scored, on the identical test frame with identical sample weights so the
comparison cannot drift through differing row sets:

- **`persistence`**: predict last season unchanged, ignoring age entirely
- **`delta_curve`**: the population curve's value at the player's age — one number per age, not personalized
- **`delta_lag`**: last season plus the mean change for that age

`delta_lag` is the reference the GAM is quoted against. It is the strongest of the three and the
only one that sees the same information the GAM does — the player's prior season and their age.
Quoting against a weaker arm would flatter the model.
#### 5. Validation Against a Known Truth

Test MAE cannot rank aging curves — `persistence` carries no age term at all yet nearly wins —
and the true curve is never observed, so every score computed on real data is a proxy.

[simulation_validation.ipynb](simulation_validation.ipynb) removes that constraint. Careers are
generated from a curve written down in `simulate.py`, censored at `qual=100`, and handed to the
same fitting code that produces the published results, so distance to the truth is computable.
Estimates are scored on the aging-curve literature's own metrics — curve-versus-curve MAE
(Nguyen & Matthews 2024), RMSE, and shape-based distance (Paparrizos & Gravano 2015) — and the
generator is calibrated against 18 moments of the real training frame.

## Project Structure

```
MLB_Position_Player_Aging_Analysis/
├── data/
│   ├── hitter_train_data.csv    # Training dataset (1980-2019)
│   ├── hitter_test_data.csv     # Test dataset (2021-2025)
│   └── outside_data.csv         # Additional data
├── src/mlb_aging/
│   ├── metrics.py               # MetricSpec: the five metrics as data
│   ├── features.py              # Centering, lag, experience, career mean
│   ├── dataset.py               # Loading and the train/test split
│   ├── gam.py                   # Model spec, fitting, curve tracing
│   ├── ipw.py                   # Survival model and IPW weights
│   ├── simulate.py              # Synthetic careers with a known aging curve
│   ├── diagnostics.py           # Residuals by age, era and random-split refits
│   ├── evaluate.py              # Weighted test scoring
│   ├── pipeline.py              # End-to-end runs and IPW comparisons
│   ├── plots.py                 # Notebook plotting helpers
│   ├── fetch.py                 # FanGraphs download (see caveat below)
│   └── cli.py                   # The `mlb-aging` command
├── tests/
│   ├── test_regression.py       # Pins the tensor specification's results to 1e-9
│   ├── test_simulate.py         # Pins the generator and the simulation's validity
│   └── test_age_only.py         # Pins the published (age-only) results to 1e-9
├── GAM.ipynb                    # Aging curves for all players
├── curve_validation.ipynb       # Are the curves trustworthy? Residuals, debut sensitivity, resampling
├── GAM_top.ipynb                # Aging curves for elite players
├── GAM_IPW.ipynb                # IPW survivorship bias correction
├── delta_method.ipynb           # Naive baseline the GAM is compared against
├── simulation_validation.ipynb  # Can the estimators recover a curve we already know?
├── pyproject.toml
├── uv.lock                      # Exact pinned resolution
└── README.md
```

The notebooks are thin drivers: all analysis logic lives in `mlb_aging`, and the
regression suite pins every number the notebooks report.

## Installation

```bash
uv sync --all-extras     # builds .venv from uv.lock
```

`uv.lock` is committed deliberately. The regression suite asserts float equality to 1e-9,
so an unpinned resolver update would be indistinguishable from a code regression. Results
reproduce exactly on pandas 3.0.5, numpy 2.5.2, pygam 0.12.0 and scikit-learn 1.9.0.

## Usage

### Command line

```bash
.venv/bin/mlb-aging train                          # all five metrics
.venv/bin/mlb-aging baselines                      # naive ladder vs both GAM arms
.venv/bin/mlb-aging train --metric WAR --ipw       # survivorship-corrected
.venv/bin/mlb-aging train --metric wRC+ --elite 110 --elite-test --curve-reference train_mean
.venv/bin/mlb-aging train --model-spec tensor      # the age x experience variant
```

### Notebooks

| Notebook | What it covers |
|---|---|
| [GAM.ipynb](GAM.ipynb) | Aging curves for all five metrics on the full population |
| [GAM_top.ipynb](GAM_top.ipynb) | Elite-player curves, compared against the general population |
| [GAM_IPW.ipynb](GAM_IPW.ipynb) | IPW-corrected curves and the survivorship-bias impact per metric |
| [delta_method.ipynb](delta_method.ipynb) | The naive delta-method baseline, and how much the GAM improves on it |
| [curve_validation.ipynb](curve_validation.ipynb) | Whether the curves are trustworthy: residuals by age, sensitivity to the debut assumption, and peak reproducibility under resampling |
| [simulation_validation.ipynb](simulation_validation.ipynb) | Whether the estimators recover a curve that is known by construction, scored on the literature's own metrics |

Point the notebook kernel at the project venv, then run top to bottom.

### Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

### Data collection

The committed CSVs are what every result above is computed from. **Refreshing them is not
currently possible**, for two independent reasons:

1. **FanGraphs is behind a Cloudflare bot challenge**, added around April 2026. Every path
   returns HTTP 403 with `cf-mitigated: challenge` — the retired `leaders-legacy.aspx` endpoint
   pybaseball uses, their modern JSON API, and the homepage alike. Browser headers do not get
   past it; neither does a residential IP, nor Google Colab.
2. **`pybaseball` is unmaintained.** 2.2.7 (September 2023) is the newest release *and* GitHub
   master still targets the retired endpoint, so installing from git changes nothing. Its
   maintainers reached the same conclusion in
   [issue #507](https://github.com/jldbc/pybaseball/issues/507).

`src/mlb_aging/fetch.py` and [fetch_data.ipynb](fetch_data.ipynb) are kept because they document
the exact query that produced the data. Running `mlb-aging fetch` fails with a
`FanGraphsUnavailable` error explaining this rather than a bare HTTP traceback.

**If you need newer data**, `pybaseball.bwar_bat()` still works — a single bulk file from
Baseball-Reference covering 1871–2026 with WAR, the components of Def (`runs_defense` +
`runs_position`) and `OPS_plus`. Retrosheet's `retrosplits` supplies the counting stats that OPS
and Speed Score are computed from. wRC+ is the one metric with no direct substitute, since it
requires FanGraphs' own linear-weight constants. Note that any re-fetch produces a *different
dataset* and so cannot reproduce the numbers above.

## Results Summary

### All Players (GAM)

| Metric | Peak Age | Test MAE |
|--------|----------|----------|
| OPS    | 27       | 0.0755   |
| wRC+   | 27       | 19.77    |
| Def    | 22\*\*   | 4.82     |
| Spd    | 21\*     | 0.996    |
| WAR    | 27       | 1.50     |

\* Spd's maximum is the **left edge of the fitted range, not a peak**. No age-20 row survives
`add_lag`, so the curve starts at 21 and falls from its first point — the rise from the youngest
age to the "peak" is exactly 0.0000.

\*\* Def's turning point is nominal. The rise from age 21 to age 22 is **+0.098 runs**, and
refitting on random halves of the players scatters the peak from 21.0 to 25.0, with half the
resamples returning the left edge. Treat Def as declining throughout, not as peaking at 22.
The genuine interior maxima are OPS (rise +0.048), wRC+ (+12.32) and WAR (+0.49).
`AgingCurve.peaks_at_left_edge` reports the edge case, and `GAM.ipynb` prints it for all five
metrics.

Test MAE is PA-weighted throughout. Def, Spd and WAR are *fitted* G-weighted, since those
metrics accrue over games played rather than plate appearances; re-scoring them G-weighted
gives 4.77, 1.004 and 1.472 respectively. The two weightings answer different questions — which
observations are reliable, versus which errors matter — and the choice moves MAE by 1–2% while
leaving every peak age and peak value untouched. `mlb-aging train` reports both.

### IPW-Corrected (Survivorship Bias Adjustment)

| Metric | Peak Age | Test MAE | MAE Improvement |
|--------|----------|----------|-----------------|
| OPS    | 27       | 0.0742   | +1.7%           |
| wRC+   | 27       | 19.49    | +1.4%           |
| Def    | 22\*\*   | 4.727    | +1.8%           |
| Spd    | 21\*     | 0.9889   | +0.7%           |
| WAR    | 26       | 1.436    | +4.1%           |

Improvement is measured against the uncorrected GAM in the table above, and is **positive when
the correction helps** — the same convention used everywhere else in this README, in
`mlb-aging train` and in the notebooks. A negative number would mean the arm is *worse* than
what it is being compared against, as Def is against the naive baseline below.

### Decline Schedule

Peak age says where a curve turns; it does not say what a contract is buying. An extension
signed at 24 runs through the peak and out the far side, and a free-agent deal is almost
entirely decline phase. Both need the *slope*, in the metric's own units, across the ages
actually being paid for. Traced from the same fitted curves (`GAM.ipynb`, final section):

| age | OPS | wRC+ | Def | Spd | WAR |
|-----|-----|------|-----|-----|-----|
| 28 | 0.008 | 102.1 | −0.05 | 4.00 | 1.66 |
| 30 | 0.003 | 100.7 | −0.68 | 3.81 | 1.50 |
| 32 | −0.007 | 98.3 | −1.57 | 3.61 | 1.21 |
| 34 | −0.019 | 95.3 | −2.54 | 3.43 | 0.85 |
| 36 | −0.033 | 91.6 | −3.43 | 3.25 | 0.44 |
| 38 | −0.049 | 87.4 | −4.15 | 3.09 | 0.01 |

Over the four years from 30 to 34 — roughly a first free-agent deal — the average position
player loses:

| metric | age 30 | age 34 | change | per year |
|--------|--------|--------|--------|----------|
| WAR    | 1.498  | 0.846  | **−0.65** | −0.16 |
| wRC+   | 100.7  | 95.3   | −5.42  | −1.36 |
| Def    | −0.68  | −2.54  | −1.86  | −0.46 |
| Spd    | 3.81   | 3.43   | −0.38  | −0.09 |
| OPS    | 0.003  | −0.019 | −0.021 | −0.005 |

**The WAR curve reaches replacement level at age 38.0**, and the decline steepens monotonically
on the way there: each two-year step from 28 onward costs more than the one before it (−0.17,
−0.29, −0.36, −0.40, −0.44 WAR).

Two cautions. These are *population* curves describing the average player at each age, not any
individual's trajectory, and players vary widely around them. And the late ages are
survivor-weighted, since only players holding 100+ PA remain — the correction for which is the
IPW arm above. That correction shifts the decline phase's *level* down by about 0.05 WAR while
leaving its *slope* essentially unchanged (0.63 vs 0.65 WAR lost from 30 to 34), so the figures
here are robust to it.

### Improvement over a Naive Baseline

Aging-curve accuracy needs a denominator. Every arm below is scored on the identical
887-row test frame with identical weights, so the numbers are directly comparable.

| Model | Prediction |
|---|---|
| `persistence` | last season, unchanged — ignores age |
| `delta_curve` | the delta method's population curve at the player's age — not personalized |
| `delta_lag` | last season **plus** the mean change for that age |

`delta_lag` is the reference: it is the strongest baseline and the only one that sees the
same information as the GAM.

| Metric | delta_lag | GAM | GAM + IPW |
|--------|-----------|-----|-----------|
| OPS    | 0.0812    | **+7.1%** | **+8.7%** |
| wRC+   | 21.03     | **+6.0%** | **+7.3%** |
| Spd    | 1.023     | +2.6%     | +3.3%     |
| WAR    | 1.500     | +0.2%     | **+4.3%** |
| Def    | 4.532     | −6.3%     | −4.3%     |

Three honest caveats belong with these numbers:

- **Def loses to the naive baseline.** This holds under either weighting (−6.1% / −4.0%
  G-weighted). It does not invalidate the defensive aging curve: individual-season MAE and
  population curve *shape* are different targets, and season-to-season defensive value is
  volatile enough that "last year, age-adjusted" is hard to beat one player-season at a time.
  This is now measured rather than asserted: against a known truth, the GAM's Def curve beats
  `delta_lag` on shape in 9 of 10 simulated leagues while still losing on per-season MAE.
- **WAR's gain is almost entirely IPW,** not the GAM — the strongest single piece of evidence
  for the survivorship correction in this project.
- **Persistence alone is within ~1.7% of `delta_lag` everywhere.** Most apparent accuracy on
  this task is simply that last season predicts this season.

### Curve Recovery Against a Known Truth

Per-season accuracy and curve accuracy are different questions, and only the first is
measurable on real data. Simulated leagues — calibrated to 18 moments of the real training
frame — make the second one computable. Replicates won out of 10, on shape-based distance:

| Metric | GAM beats delta method | career mean beats no control |
|--------|------------------------|------------------------------|
| OPS    | 9/10                   | 10/10 |
| wRC+   | **10/10**              | 10/10 |
| Spd    | 4/10                   | 5/10  |
| Def    | 9/10                   | 9/10  |
| WAR    | 9/10                   | 10/10 |

- **The GAM is the better curve estimator**, with 1.6–2.5× lower shape distance on four of five
  metrics (OPS 1.6×, Def 2.0×, wRC+ 2.2×, WAR 2.5×). Until now that claim rested entirely on
  per-season MAE, which cannot rank curves.
- **Def loses on MAE and wins on shape**, confirming the caveat above.
- **The peak reads about a year late**, and switching the survivorship mechanisms off leaves the
  bias intact — it comes from the career mean acting as a post-treatment control, not from
  selection. This is simulation-conditional and no published number has been changed on it.
- **IPW cannot be assessed this way.** The generator's survival weights are far too uniform
  (SD 0.18 against a real 0.43), so the correction has nothing to reweight. The notebook records
  the diagnosis and a failed attempt at a fix.

### Elite Players

| Cohort | Metric | Peak Age | Test MAE |
|--------|--------|----------|----------|
| career wRC+ ≥ 110 | wRC+ | 28 | 20.16 |
| career WAR ≥ 2.5  | wRC+ | 28 | 19.90 |
| career WAR ≥ 2.5  | WAR  | 27 | 1.69 |
| career WAR ≥ 2.5  | Def  | 22 | 4.63 |

Elite hitters peak a year later than the population (28 vs 27) and hold their level longer.
Elite WAR players peak at the same age as everyone else — what separates them is height, not
timing. Def peaks at 22 in the elite cohort as in the population, and is no more locatable there.

Training and test populations are independent in these runs. Restricting only the *test* set,
and scoring the all-player model on it, gives wRC+ 20.56 and WAR 1.75 — both worse than the
specialist models above, which is the evidence that elite curves differ rather than simply
being the population curve shifted up.

## Implications for Teams

1. **Free Agent Signings**: Peak performance occurs at 27 for hitting and overall value; exercise caution with long-term contracts for players past age 30, especially those valued for speed or defense
2. **Contract Valuation**: Front-load compensation — decline is continuous from the late 20s and steepens every year, so the back end of a long deal is systematically the worst-value portion
3. **Defense and Speed**: Neither has a usable peak age. Speed declines from the moment players arrive, and defensive decline is measurable but its turning point is not — do not overvalue either skill in older players, and do not build a valuation on a defensive peak age
4. **Elite vs. Average Players**: Top hitters sustain performance about a year longer than average (wRC+ peak 28 vs. 27), justifying premium valuations for proven bats in their late 20s
5. **Long-term Contracts**: A 10-year deal signed at age 26 spans nearly the entire decline phase; WAR contribution in the final years will be substantially lower than at signing, though the curve does not cross replacement level until 38

## Future Work

- Incorporate position-specific aging curves (catchers, shortstops vs. corner outfielders)
- Replace the plug-in career mean with a shrunk player effect (a mixed-model random intercept), so short careers are pulled toward the population mean rather than trusted equally
- Examine modern era (2015+) vs. historical aging differences
- Develop injury-adjusted aging models
- Extend IPW to non-qualified player seasons to capture the full distribution of aging outcomes
- Give the simulator's survival model a two-population form (established regulars vs. fringe players) so IPW becomes evaluable against a known truth
- Create interactive visualization dashboard

## License

This project is for educational and analytical purposes.

## Acknowledgments

- Data provided by pybaseball
- Statistical methods based on established sabermetric research
- Inspired by aging curve analyses in the baseball analytics community
