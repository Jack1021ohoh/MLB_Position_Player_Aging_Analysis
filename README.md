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
- **Source**: MLB statistics via pybaseball API
- **Training Data**: 1980–2019 seasons
- **Test Data**: 2021–2025 seasons
- **Qualification**: Minimum 100 plate appearances per season
- **Age Range**: Players aged 20–40

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

## Project Structure

```
MLB_Position_Player_Aging_Analysis/
├── data/
│   ├── hitter_train_data.csv    # Training dataset (1980-2019)
│   ├── hitter_test_data.csv     # Test dataset (2021-2025)
│   └── outside_data.csv         # Additional data
├── fetch_data.ipynb             # Data collection and preprocessing
├── GAM.ipynb                    # GAM aging curves for all players
├── GAM_top.ipynb                # GAM aging curves for elite players
├── GAM_IPW.ipynb                # GAM with IPW survivorship bias correction
└── README.md
```

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib scipy scikit-learn pybaseball pygam
```

### Dependencies
- Python 3.7+
- pandas, numpy: Data manipulation and numerical computing
- matplotlib: Visualization
- scipy: Statistical functions
- scikit-learn: Model evaluation and preprocessing
- pybaseball: MLB data API
- pyGAM: Generalized Additive Models

## Usage

### 1. Data Collection
Run [fetch_data.ipynb](fetch_data.ipynb) to download and prepare MLB statistics.

### 2. GAM Analysis (All Players)
Run [GAM.ipynb](GAM.ipynb) to fit aging curves for all five metrics on the full player population.

### 3. Elite Player Analysis
Run [GAM_top.ipynb](GAM_top.ipynb) to fit aging curves restricted to top players and compare with the general population.

### 4. Survivorship Bias Correction
Run [GAM_IPW.ipynb](GAM_IPW.ipynb) to fit IPW-corrected aging curves and evaluate the impact of survivorship bias on each metric.

## Results Summary

### All Players (GAM)

| Metric | Peak Age | Test MAE |
|--------|----------|----------|
| OPS    | 26       | 0.075    |
| wRC+   | 26       | 19.7     |
| Def    | 23       | 4.75     |
| Spd    | 21       | 0.986    |
| WAR    | 25       | 1.50     |

### IPW-Corrected (Survivorship Bias Adjustment)

| Metric | Peak Age | Test MAE | MAE Improvement |
|--------|----------|----------|-----------------|
| OPS    | 26       | 0.074    | −1.5%           |
| wRC+   | 26       | 19.5     | −1.3%           |
| Def    | 23       | 4.66     | −2.0%           |
| Spd    | 21       | 0.981    | −0.5%           |
| WAR    | 25       | 1.45     | −3.5%           |

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
