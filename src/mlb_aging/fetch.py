"""Download season-level batting data from FanGraphs via pybaseball.

.. warning::

   **This does not work and cannot be made to work from here.** It is kept
   because it documents the exact query that produced the committed CSVs, and
   because it will be the starting point if FanGraphs ever reopens.

   Two independent blockers, either one fatal:

   1. **FanGraphs is behind a Cloudflare bot challenge**, added around April
      2026. Every path returns HTTP 403 with ``cf-mitigated: challenge`` --
      the retired ``leaders-legacy.aspx``, the modern
      ``/api/leaders/major-league/data``, and the homepage alike. Browser
      headers do not help; neither does a residential IP, nor Colab, which is
      where the last successful run (2026-03-17) happened.
   2. **pybaseball is unmaintained.** 2.2.7 (September 2023) is the newest
      release *and* master still hardcodes ``leaders-legacy.aspx``, so
      installing from git changes nothing.

   The pybaseball maintainers reached the same conclusion in issue #507: the
   ``curl_cffi`` fix that resolved the earlier 2025 breakage was tried against
   this one and does not work.

   Everything here except the transport is verified correct: the chunker
   covers 1980-2019 in four chunks and 2021-2025 in one, and
   :data:`SELECTED_COLS` resolves to exactly the 20 columns in the committed
   CSVs.

Seasons are pulled in chunks because a single request spanning decades is slow
and prone to timing out.
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


class FanGraphsUnavailable(RuntimeError):
    """Raised when FanGraphs refuses the request, which is now always.

    Exists so the failure arrives as an explanation rather than as a bare
    ``HTTPError: ... status code 403`` from four frames inside pybaseball.
    """


UNAVAILABLE_MESSAGE = """\
FanGraphs refused the request. This is expected -- the fetch stage has not
worked since around April 2026 and cannot be fixed from here.

  * FanGraphs put the whole site behind a Cloudflare bot challenge. Every path
    returns 403, including their modern JSON API and the homepage. Browser
    headers, a residential IP and Google Colab have all been tried.
  * pybaseball is unmaintained: 2.2.7 (Sept 2023) is the newest release and
    master still targets the retired leaders-legacy.aspx endpoint, so
    installing from git changes nothing.

The committed CSVs in data/ still reproduce every published result, so this
blocks refreshing the data, not reproducing the analysis. See the "fetch stage"
note in CLAUDE.md for the alternatives that do still work (Baseball-Reference's
bwar_bat covers WAR, Def components and OPS+ back to 1871)."""


def fetch_seasons(start: int, end: int, chunk_years: int = CHUNK_YEARS) -> pd.DataFrame:
    """Fetch ``start``..``end`` inclusive as one frame.

    Raises :class:`FanGraphsUnavailable` in practice -- see the module note.
    """
    from pybaseball import batting_stats

    frames = []
    for chunk_start, chunk_end in _chunks(start, end, chunk_years):
        print(f"  fetching {chunk_start}-{chunk_end} ...", flush=True)
        try:
            chunk = batting_stats(
                chunk_start, chunk_end, qual=QUAL, stat_columns=SELECTED_COLS
            )
        except Exception as exc:  # noqa: BLE001 -- any transport failure means the same thing
            raise FanGraphsUnavailable(f"{UNAVAILABLE_MESSAGE}\n\nUnderlying error: {exc}") from exc
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
