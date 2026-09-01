"""Feature engineering shared by every aging-curve model.

These functions were duplicated verbatim across GAM.ipynb, GAM_top.ipynb and
GAM_IPW.ipynb. They are ported here unchanged -- including the quirks noted in
the docstrings -- so the package reproduces the published results exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def centralize_data(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Subtract each season's PA-weighted league mean from ``col``.

    Adds ``<col>_centralized``. Note the merge on ``Season`` reorders rows,
    which matters because :func:`add_lag` is order-sensitive.
    """
    df = data.copy()
    wm = lambda x: np.average(x, weights=df.loc[x.index, "PA"])  # noqa: E731
    mean_results = df.groupby(["Season"]).agg(weighted_mean=(col, wm)).reset_index()
    df = df.merge(mean_results, on="Season")
    df[col + "_centralized"] = np.subtract(df[col], df["weighted_mean"])
    return df


def add_experience(data: pd.DataFrame) -> pd.DataFrame:
    """Add ``experience``: cumulative season gaps since a player's first row.

    This counts elapsed seasons rather than service time, so a player who
    misses a year still gains two years of "experience" on their return.
    """
    data = data.copy()
    data["experience"] = data.groupby(["IDfg"])["Season"].diff().fillna(0.0)
    data["experience"] = data.groupby(["IDfg"])["experience"].cumsum()
    return data


def add_lag(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add ``<col>_lag`` (previous row for the player) and drop unlagged rows.

    Uses positional ``shift(1)`` within each player, so the caller is
    responsible for row ordering. Every player's first observed season is
    dropped because it has no predecessor.
    """
    data = data.copy()
    data[col + "_lag"] = data.groupby(["IDfg"])[col].shift(1)
    data = data.dropna(subset=[col + "_lag"])
    return data


def add_career_mean(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add ``<col>_career_mean``: the player's PA-weighted mean of ``col``.

    This is a talent control -- a per-player constant standing in for latent
    quality, so the age terms are not left to absorb it. The value is identical
    for every row of a player, in training and test alike; test rows receive
    the same training-era constant via :func:`generate_test_data`.

    MODELLING CHOICE, not a defect: the mean is taken over the player's whole
    career, so a row's own season is included in the feature used to predict it
    (~20% of it at the median 5 rows per player, more for short careers). No
    test target leaks -- a test row's career mean is built from training
    seasons only -- so test scores stay honest. The cost is that the
    career-mean/target relationship is stronger in training than at test time.
    Leave-one-out or a shrunk estimate is the standard alternative if that
    generalization gap turns out to matter.

    Note this narrows the frame to a fixed column list, dropping anything else
    the caller had attached.
    """
    cols = ["IDfg", "Season", "Name", "Age", "G", "PA", "experience"] + [col, col + "_lag"]
    df = data.loc[:, cols].copy()
    wm = lambda x: np.average(x, weights=df.loc[x.index, "PA"])  # noqa: E731
    mean_result = df.groupby(["IDfg"]).agg(weighted_mean=(col, wm)).reset_index()
    mean_result.columns = ["IDfg", col + "_career_mean"]
    df = df.merge(mean_result, on="IDfg", how="left")
    return df


def generate_data(
    data: pd.DataFrame, feature_cols: list[str], target: str
) -> tuple[np.ndarray, np.ndarray]:
    """Split a frame into the feature matrix and target vector."""
    return data.loc[:, feature_cols].values, data[target].values


def generate_test_data(
    train_data: pd.DataFrame, test_data: pd.DataFrame, col: str
) -> pd.DataFrame:
    """Attach training-era career means to test rows.

    Because the career mean is constant within a player, taking the last
    training season is just a way of picking that single value. Players absent
    from the training era get no career mean and are dropped, so the test set
    skews toward long careers.
    """
    train_cols = ["IDfg", "Season", col + "_career_mean"]
    test_cols = ["IDfg", "Season", "Name", "Age", "G", "PA", "experience", col + "_lag", col]

    train_df = (
        train_data.loc[:, train_cols]
        .sort_values("Season")
        .groupby("IDfg", as_index=False)
        .last()[["IDfg", col + "_career_mean"]]
    )
    test_df = test_data.loc[:, test_cols]

    all_test_data = test_df.merge(train_df, on="IDfg", how="left")
    all_test_data = all_test_data.dropna()
    return all_test_data.reset_index(drop=True)
