# MLB Position Player Aging Analysis

A comprehensive statistical analysis of MLB position player performance trends across career trajectories using advanced modeling techniques.

## Overview

This project constructs aging curves for MLB position players using historical data from 1980 to 2024. By applying both the delta method and Generalized Additive Models (GAM), this analysis reveals how player performance metrics evolve with age, providing valuable insights for player evaluation, contract negotiations, and roster construction.

## Key Findings

- **Peak Performance Age**: Position players reach their offensive peak around age 26 for OPS and wRC+
- **Defensive Peak**: Players peak defensively earlier, around age 23
- **Speed Peak**: Base running speed peaks at age 21, indicating it is primarily talent-driven
- **Overall Value Peak**: WAR contribution peaks around age 25
- **Decline Phase**: Significant performance decline begins after age 35 across all metrics
- **Model Comparison**: GAM outperforms the delta method for predictive accuracy

### Performance Metrics Analyzed

- **OPS (On-base Plus Slugging)**: Overall offensive production
- **wRC+ (Weighted Runs Created Plus)**: Context-adjusted offensive value
- **Def (Defensive Runs)**: Defensive contribution
- **Spd (Speed Score)**: Base running ability
- **WAR (Wins Above Replacement)**: Overall player value

## Methodology

### Data Collection
- **Source**: MLB statistics via pybaseball API
- **Training Data**: 1980-2019 seasons (16,625 player-seasons)
- **Test Data**: 2021-2024 seasons (1,848 player-seasons)
- **Qualification**: Minimum 100 plate appearances per season
- **Age Range**: Players aged 20-40

### Statistical Approaches

#### 1. Delta Method
- Calculates year-over-year changes in performance metrics
- Weighted by harmonic mean of consecutive season plate appearances
- Produces league-wide aging curves through cumulative changes
- Centralizes data to account for era effects

#### 2. Generalized Additive Models (GAM)
- Incorporates multiple features: age, career mean, experience, and previous season performance
- Uses tensor product splines for age-experience interactions
- Weighted by plate appearances or games played
- Provides superior predictive accuracy (e.g., OPS MAE: 0.075 vs league average 0.761)

## Project Structure

```
MLB_Position_Player_Aging_Analysis/
├── data/
│   ├── hitter_train_data.csv    # Training dataset (1980-2019)
│   ├── hitter_test_data.csv     # Test dataset (2021-2024)
│   └── outside_data.csv         # Additional data
├── fetch_data.ipynb             # Data collection and preprocessing
├── delta_method.ipynb           # Delta method implementation
├── GAM.ipynb                    # GAM analysis for all players
├── GAM_top.ipynb               # GAM analysis for elite players
└── README.md
```

## Installation

### Requirements
```bash
pip install pandas numpy matplotlib scipy scikit-learn pybaseball pygam
```

### Dependencies
- Python 3.7+
- pandas: Data manipulation
- numpy: Numerical computing
- matplotlib: Visualization
- scipy: Statistical functions
- scikit-learn: Model evaluation metrics
- pybaseball: MLB data API
- pyGAM: Generalized Additive Models

## Usage

### 1. Data Collection
Run [fetch_data.ipynb](fetch_data.ipynb) to download and prepare MLB statistics:
```python
# Fetches batting statistics with selected metrics
# Combines data from multiple decades
# Splits into training (1980-2019) and test (2021-2024) sets
```

### 2. Delta Method Analysis
Run [delta_method.ipynb](delta_method.ipynb) to generate traditional aging curves:
- Analyzes OPS and wRC+ aging patterns
- Produces year-over-year delta calculations
- Evaluates model performance on test data

### 3. GAM Analysis
Run [GAM.ipynb](GAM.ipynb) for advanced modeling:
- Fits GAM models for all five metrics
- Generates aging curves with confidence intervals
- Evaluates predictive accuracy
- Provides player-level predictions

## Results Summary

| Metric | Peak Age | Test MAE | Model Performance |
|--------|----------|----------|-------------------|
| OPS | 26 | 0.075 | 9.9% of league average |
| wRC+ | 26 | 19.49 | Superior to delta method |
| Def | 23 | 4.64 | Strong predictive power |
| Spd | 21 | 0.97 | Highly accurate |
| WAR | 25 | 1.47 | Robust validation |

## Implications for Teams

1. **Free Agent Signings**: Exercise caution with long-term contracts for players past age 30, especially those valued primarily for speed or defense
2. **Contract Valuation**: Front-load compensation for players peaking in their mid-20s
3. **Roster Construction**: Balance age distribution to maintain competitive performance
4. **Talent Development**: Defensive and speed skills are talent-driven; focus development on offensive skills
5. **Long-term Contracts**: 10+ year deals should account for significant decline in later years

## Visualization

The project includes aging curve visualizations showing:
- Performance trajectories from age 20-40
- Peak performance ages for each metric
- Comparative analysis between delta method and GAM approaches
- Polynomial trend fitting for aging patterns

## Future Work

- Incorporate position-specific aging curves
- Analyze impact of playing time on aging patterns
- Examine modern era (2015+) vs historical aging differences
- Develop injury-adjusted aging models
- Create interactive visualization dashboard

## Author

Statistical analysis of MLB player aging patterns using historical data and advanced modeling techniques.

## License

This project is for educational and analytical purposes.

## Acknowledgments

- Data provided by pybaseball
- Statistical methods based on established sabermetric research
- Inspired by aging curve analyses in baseball analytics community
