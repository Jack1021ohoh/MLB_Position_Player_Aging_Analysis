"""Download season-level batting data from FanGraphs via pybaseball.

Replaces fetch_data.ipynb. Seasons are pulled in chunks because a single
request spanning decades is slow and prone to timing out.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_aging.dataset import DEFAULT_DATA_DIR

#: FanGraphs column ids: identifiers, playing time, the rate stats the models
#: use, plus the DEFENSE and FIELDING groups that supply Def and Fld.
SELECTED_COLS = [
    "1", "2", "3", "4", "5", "6", "7", "11", "16", "21",
    "39", "40", "50", "58", "60", "61", "DEFENSE", "FIELDING",
]

#: Minimum plate appearances for a player-season to be included. This
#: threshold is what creates the survivorship bias the IPW module corrects.
QUAL = 100

TRAIN_SEASONS = (1980, 2019)
TEST_SEASONS = (2021, 2025)
CHUNK_YEARS = 10


def _chunks(start: int, end: int, size: int) -> list[tuple[int, int]]:
    return [(y, min(y + size - 1, end)) for y in range(start, end + 1, size)]


def fetch_seasons(start: int, end: int, chunk_years: int = CHUNK_YEARS) -> pd.DataFrame:
    """Fetch ``start``..``end`` inclusive as one frame."""
    from pybaseball import batting_stats

    frames = []
    for chunk_start, chunk_end in _chunks(start, end, chunk_years):
        print(f"  fetching {chunk_start}-{chunk_end} ...", flush=True)
        chunk = batting_stats(chunk_start, chunk_end, qual=QUAL, stat_columns=SELECTED_COLS)
        print(f"    {len(chunk)} rows", flush=True)
        frames.append(chunk)

    return pd.concat(frames, axis=0).reset_index(drop=True)


def fetch_all(data_dir: Path | str = DEFAULT_DATA_DIR) -> dict[str, Path]:
    """Fetch both splits and write them to ``data_dir``.

    Unlike the notebook, this writes directly into the data directory instead
    of the working directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for label, (start, end) in (
        ("hitter_train_data", TRAIN_SEASONS),
        ("hitter_test_data", TEST_SEASONS),
    ):
        print(f"{label}: {start}-{end}")
        df = fetch_seasons(start, end)
        path = data_dir / f"{label}.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {len(df)} rows -> {path}")
        written[label] = path

    return written
