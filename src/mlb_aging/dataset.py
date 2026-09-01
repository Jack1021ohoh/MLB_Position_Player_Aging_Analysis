"""Loading and splitting the season-level batting data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_aging.features import add_career_mean, add_experience, add_lag, centralize_data
from mlb_aging.metrics import MetricSpec

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MIN_AGE = 20
MAX_AGE = 40
#: 2020 is excluded from both splits -- the COVID season was 60 games.
SPLIT_SEASON = 2020


def load_data(
    spec: MetricSpec, data_dir: Path | str = DEFAULT_DATA_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, engineer features on, and re-split the train/test data.

    Train and test are concatenated *before* experience and lag are computed,
    then split again on :data:`SPLIT_SEASON`. That ordering is load-bearing:
    2021 test rows need a lag whose prior season lives in the training range.

    Centering on the combined frame does *not* leak: :func:`centralize_data`
    groups by ``Season``, and the season ranges are disjoint (1980-2019 vs
    2021-2025), so every group lies wholly inside one split. Verified identical
    to centering each split separately, to 0.0 across all 18,840 rows.
    """
    data_dir = Path(data_dir)
    train_data = pd.read_csv(data_dir / "hitter_train_data.csv")
    test_data = pd.read_csv(data_dir / "hitter_test_data.csv")

    train_data = train_data.loc[
        (train_data["Age"] >= MIN_AGE).values & (train_data["Age"] <= MAX_AGE).values, :
    ]
    test_data = test_data.loc[
        (test_data["Age"] >= MIN_AGE).values & (test_data["Age"] <= MAX_AGE).values, :
    ]

    train_data = train_data.sort_values(by=["Season", "Name"]).reset_index(drop=True)
    test_data = test_data.sort_values(by=["Season", "Name"]).reset_index(drop=True)

    all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
    all_data = add_experience(all_data)

    if spec.centralized:
        all_data = centralize_data(all_data, spec.name)

    all_data = add_lag(all_data, col=spec.target_col)

    train_data = all_data.loc[all_data["Season"] < SPLIT_SEASON, :]
    test_data = all_data.loc[all_data["Season"] > SPLIT_SEASON, :]
    return train_data, test_data


def build_training_frame(
    train_data: pd.DataFrame, spec: MetricSpec, elite_threshold: float | None = None
) -> pd.DataFrame:
    """Attach career means and optionally restrict to elite players."""
    df = add_career_mean(train_data, spec.target_col)
    if elite_threshold is not None:
        df = df.loc[df[spec.career_mean_col] > elite_threshold, :]
    return df
