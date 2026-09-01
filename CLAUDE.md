# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Statistical analysis fitting MLB position-player aging curves with pyGAM, correcting the
survivorship bias created by the `qual=100` PA sampling threshold via Inverse Probability
Weighting. See `README.md` for findings.

The analysis logic lives in the `mlb_aging` package (`src/` layout). The notebooks are being
migrated onto it; see **Migration state** below for what has and hasn't moved.

## Commands

```bash
uv sync --all-extras                               # builds .venv from uv.lock

.venv/bin/python -m pytest tests/ -q              # full regression suite (~45s)
.venv/bin/python -m pytest tests/ -q -k IPW       # one group
.venv/bin/python -m pytest tests/test_regression.py::test_all_player_curves -q

.venv/bin/mlb-aging train                          # all five metrics
.venv/bin/mlb-aging train --metric WAR --ipw       # survivorship-corrected
.venv/bin/mlb-aging train --metric wRC+ --elite 110 --elite-test --curve-reference train_mean
.venv/bin/mlb-aging train --metric Def --elite 2.5 --cohort-metric WAR --curve-reference train_mean
.venv/bin/mlb-aging fetch                          # re-download from FanGraphs (slow)
```

`uv.lock` pins the exact resolution and is committed deliberately: the regression suite asserts
float equality to 1e-9, so an unpinned resolver update would be indistinguishable from a code
regression. Run `uv lock --upgrade` to move dependencies forward, and expect to justify any test
that moves as a result. `.python-version` pins the interpreter to 3.12; the lock is universal and
also resolves for 3.11 (which selects an older numpy).

The published results reproduce **exactly** on the locked stack (pandas 3.0.5, numpy 2.5.2,
pygam 0.12.0, scikit-learn 1.9.0) — and did so on the pre-lock resolution too, so no pinning to
old versions is needed.

## The reproduction contract

The refactor is deliberately **bug-for-bug faithful**: `tests/test_regression.py` pins every peak
age, peak value, test MAE and survival-model AUC transcribed from the notebooks' stored outputs,
and all of them currently match to 1e-9.

Known defects are therefore *preserved on purpose*, each tagged `BUG-PRESERVED:` at its site:

1. **`evaluate()` scoring weight** (`metrics.py`, `MetricSpec.eval_weight_col`) — the notebooks
   never passed `weight_col`, so its `'PA'` default applied to every metric. Def, Spd and WAR are
   fitted G-weighted but scored PA-weighted. The fix is `eval_weight_col = weight_col`; it will
   move those three MAEs. `test_eval_weight_bug_is_still_preserved` guards this and should be
   deleted when it is fixed.
2. **Centering leakage** (`dataset.py`, `load_data`) — for centralized metrics the league mean is
   computed over the concatenated train+test frame, so test-season means feed training features.

When fixing either of these, change the affected expectation in the same commit — never loosen a
tolerance to make a fix pass.

### Not a bug: the career mean

`add_career_mean` produces a per-player constant used as a talent control. It is the **same value
on both sides** — verified: 0 of 2312 training players have more than one value, and test rows
receive that identical constant (max difference 0.0). The
`.sort_values("Season").groupby("IDfg").last()` in `generate_test_data` is therefore a no-op way
of selecting it, and the stale "expanding career mean" comment in `GAM.ipynb` / `GAM_top.ipynb`
describes an implementation that no longer exists — `GAM_IPW.ipynb`'s comment is the accurate one.

The one real property to keep in mind is that a row's own season is included in its own predictor
(~20% of it at the median 5 rows per player). No test target leaks, so test scores are honest; the
cost is that the career-mean/target relationship is stronger in training than at test. Leave-one-out
or a shrunk estimate is the standard alternative, worth measuring before adopting.

## Architecture

### `MetricSpec` is the core abstraction

`metrics.py` reduces the five metrics to four axes: whether the metric is season-centralized (only
OPS), the fit weight column (`PA` for OPS/wRC+, `G` for Def/Spd/WAR), the eval weight column, and
the curve reference. Derived column names (`target_col`, `lag_col`, `career_mean_col`,
`feature_cols`) hang off the spec. This replaced five near-identical copy-pasted sections per
notebook — add a metric here, not by duplicating a code path.

### Feature order is part of the model spec

```python
feature_cols = ["Age", career_mean_col, "experience", lag_col]
GAM(te(0, 2, n_splines=5) + s(1, n_splines=5) + s(3, n_splines=5))
```

`te(0, 2)` is the age × experience tensor product, `s(1)` the career-mean talent control, `s(3)`
the lagged prior season. The term indices are positional into `feature_cols`, so **reordering that
list silently changes the model.**

### `load_data` ordering is load-bearing

Train and test are concatenated *before* `experience` and `lag` are computed, then re-split on
`Season < 2020` / `Season > 2020` (2020 excluded on both sides — COVID short season). 2021 test
rows need a lag whose prior season lives in the training range. Ages are clipped to 20–40.

`add_lag` uses positional `shift(1)` within each player, so it depends on row order — and
`centralize_data`'s merge on `Season` reorders rows. Don't reorder these steps.

`experience` counts elapsed seasons since a player's first row, not service time: a missed year
still advances it by two.

### Curve tracing

`gam.aging_curve` pins every non-age column: experience tied to age, lag set to the training mean
for that age, career mean per `curve_reference`. Because `s(1)` is additive, the career-mean
reference shifts the curve **level only — it never moves the peak age.** The notebooks are
inconsistent here (all-player wRC+ pins 100; every top-player section uses the training mean),
which is why the reference is overridable per run.

### Test set shrinks toward long careers

`generate_test_data` drops test rows for players absent from the training era, since they have no
career mean to join. n_test is 887 of ~2300 test rows.

### IPW

`fit_retirement_model` → `compute_ipw_weights` → `ipw_final_weight` as the GAM sample weight.
Survival label is "does `(IDfg, Season + 1)` exist in the data"; the final season is dropped as
unlabelable. `p_survive` is floored at 0.1 (capping any weight at 10×) and normalised to mean 1.
`evaluate_retirement_model` returns a `RetirementDiagnostics` dataclass rather than printing and
plotting, so callers choose presentation.

Adding the lagged metric (`perf_col`) matters most for G-weighted metrics — WAR gains +0.028 AUC —
because games played alone can't separate genuine decline from a defensive specialist who rarely
bats. For PA-weighted metrics it's ~0, since plate appearances already proxy hitting quality.

### Cohort selection

`pipeline.cohort_ids` selects players by **IDfg** whose career mean of one metric clears a
threshold, so a cohort defined by career WAR can be modelled on Def or wRC+. `GAM_top.ipynb` did
this by reusing a positional boolean mask across differently-built frames, which quietly assumed
identical row ordering; the id-based version is verified equivalent by the regression tests.

## Migration state

- **Ported:** the helper block, GAM fitting, curve tracing, IPW, evaluation, data fetching, and all
  three notebooks' result paths — all covered by `tests/test_regression.py`.
- **Not yet done:** the notebooks still carry their own duplicated copies of the old helpers and
  have not been rewritten to import `mlb_aging`. Until they are, a change to the package does *not*
  change the notebooks, and the two can drift. `delta_method.ipynb` (a superseded delta-method
  baseline) has not been ported at all.
- `data/outside_data.csv` has a different FanGraphs-export schema and is unused.

## Editing notebooks

Prefer `NotebookEdit` over rewriting `.ipynb` JSON. The markdown cells quote specific numbers
(peak ages, MAEs, AUCs); if a change moves those, update the surrounding prose and the `README.md`
tables in the same commit.
