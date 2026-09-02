# MLB Position Player Aging Analysis

A comprehensive statistical analysis of MLB position player performance trends across career trajectories using Generalized Additive Models (GAM) with survivorship bias correction.

## Overview

This project constructs aging curves for MLB position players using historical data from 1980 to 2025. By applying Generalized Additive Models (GAM) with Inverse Probability Weighting (IPW) to correct for survivorship bias, this analysis reveals how player performance metrics evolve with age, providing actionable insights for player evaluation, contract negotiations, and roster construction.

## Key Findings

- **Peak Offensive Age**: Position players reach their offensive peak around **age 26** for both OPS and wRC+
- **Defensive Peak**: Players peak defensively earlier, around **age 23**, driven primarily by athleticism
- **Speed Peak**: Base running speed peaks at **age 21** and declines monotonically — the earliest peak of all metrics
- **Overall Value Peak**: WAR contribution peaks around **age 25**, one year before hitting, due to the earlier decline of defense and speed
- **Survivorship Bias**: IPW correction consistently improves predictive accuracy across all metrics, with the largest gain in WAR (3.5% MAE reduction)
- **Gain over a Naive Baseline**: Measured against the delta method applied to each player's prior season, GAM + IPW reduces test MAE by **8.8% for OPS** and **7.4% for wRC+**. WAR gains 3.7% — almost all of it from the survivorship correction rather than the model itself. Def does not beat the baseline, and that is reported rather than hidden
- **Elite Player Differences**: Top players (career WAR ≥ 2.5) peak in overall value one year later (age 26) and sustain hitting performance longer (wRC+ peak age 27)
- **Decline Phase**: Significant performance decline begins after age 35 across all metrics

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
- Features: age × experience tensor product, career mean (player talent control), lagged prior-season performance
- Weighted by plate appearances (hitting metrics) or games played (Def, Spd, WAR)
- Hyperparameters tuned via grid search over 20 log-spaced λ values
- Evaluated on held-out 2021–2025 test seasons

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
│   ├── evaluate.py              # Weighted test scoring
│   ├── pipeline.py              # End-to-end runs and IPW comparisons
│   ├── plots.py                 # Notebook plotting helpers
│   ├── fetch.py                 # FanGraphs download (see caveat below)
│   └── cli.py                   # The `mlb-aging` command
├── tests/test_regression.py     # Pins every published number to 1e-9
├── GAM.ipynb                    # Aging curves for all players
├── GAM_top.ipynb                # Aging curves for elite players
├── GAM_IPW.ipynb                # IPW survivorship bias correction
├── delta_method.ipynb           # Naive baseline the GAM is compared against
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
```

### Notebooks

| Notebook | What it covers |
|---|---|
| [GAM.ipynb](GAM.ipynb) | Aging curves for all five metrics on the full population |
| [GAM_top.ipynb](GAM_top.ipynb) | Elite-player curves, compared against the general population |
| [GAM_IPW.ipynb](GAM_IPW.ipynb) | IPW-corrected curves and the survivorship-bias impact per metric |
| [delta_method.ipynb](delta_method.ipynb) | The naive delta-method baseline, and how much the GAM improves on it |

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
| OPS    | 26       | 0.075    |
| wRC+   | 26       | 19.7     |
| Def    | 23       | 4.75     |
| Spd    | 21       | 0.986    |
| WAR    | 25       | 1.50     |

Test MAE is PA-weighted throughout. Def, Spd and WAR are *fitted* G-weighted, since both
metrics accrue over games played rather than plate appearances; re-scoring them G-weighted
gives 4.71, 0.994 and 1.47 respectively. The two weightings answer different questions — which
observations are reliable, versus which errors matter — and the choice moves MAE by 1–2% while
leaving every peak age and peak value untouched. `mlb-aging train` reports both.

### IPW-Corrected (Survivorship Bias Adjustment)

| Metric | Peak Age | Test MAE | MAE Improvement |
|--------|----------|----------|-----------------|
| OPS    | 26       | 0.0741   | +1.5%           |
| wRC+   | 26       | 19.47    | +1.3%           |
| Def    | 23       | 4.658    | +2.0%           |
| Spd    | 21       | 0.9813   | +0.5%           |
| WAR    | 25       | 1.445    | +3.5%           |

Improvement is measured against the uncorrected GAM in the table above, and is **positive when
the correction helps** — the same convention used everywhere else in this README, in
`mlb-aging train` and in the notebooks. A negative number would mean the arm is *worse* than
what it is being compared against, as Def is against the naive baseline below.

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
| OPS    | 0.0812    | **+7.5%** | **+8.8%** |
| wRC+   | 21.03     | **+6.2%** | **+7.4%** |
| Spd    | 1.023     | +3.6%     | +4.1%     |
| WAR    | 1.500     | +0.2%     | **+3.7%** |
| Def    | 4.532     | −4.9%     | −2.8%     |

Three honest caveats belong with these numbers:

- **Def loses to the naive baseline.** This holds under either weighting (−4.7% / −2.4%
  G-weighted). It does not invalidate the defensive aging curve: individual-season MAE and
  population curve *shape* are different targets, and season-to-season defensive value is
  volatile enough that "last year, age-adjusted" is hard to beat one player-season at a time.
- **WAR's gain is almost entirely IPW,** not the GAM — the strongest single piece of evidence
  for the survivorship correction in this project.
- **Persistence alone is within ~1.5% of `delta_lag` everywhere.** Most apparent accuracy on
  this task is simply that last season predicts this season.

### Elite Players (career WAR ≥ 2.5)

| Metric | Peak Age | Test MAE (strong players) |
|--------|----------|---------------------------|
| wRC+   | 27       | 20.0                      |
| WAR    | 26       | 1.71                      |
| Def    | 22       | 4.60                      |

## Implications for Teams

1. **Free Agent Signings**: Peak performance occurs at 25–27 depending on metric; exercise caution with long-term contracts for players past age 30, especially those valued for speed or defense
2. **Contract Valuation**: Front-load compensation — the steepest decline begins around age 34–35 across all metrics
3. **Defense and Speed**: Both peak before age 25 and decline monotonically; do not overvalue these skills in older players
4. **Elite vs. Average Players**: Top players sustain hitting performance 1–2 years longer than average (wRC+ peak 27 vs. 26), justifying premium valuations for proven stars in their late 20s
5. **Long-term Contracts**: A 10-year deal signed at age 26 spans nearly the entire decline phase; WAR contribution in the final years will be substantially lower than at signing

## Future Work

- Incorporate position-specific aging curves (catchers, shortstops vs. corner outfielders)
- Examine modern era (2015+) vs. historical aging differences
- Develop injury-adjusted aging models
- Extend IPW to non-qualified player seasons to capture the full distribution of aging outcomes
- Create interactive visualization dashboard

## License

This project is for educational and analytical purposes.

## Acknowledgments

- Data provided by pybaseball
- Statistical methods based on established sabermetric research
- Inspired by aging curve analyses in the baseball analytics community
