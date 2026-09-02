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

.venv/bin/python -m pytest tests/ -q              # full regression suite, 38 tests (~2min)
.venv/bin/python -m pytest tests/ -q -k IPW       # one group
.venv/bin/python -m pytest tests/test_regression.py::test_all_player_curves -q

.venv/bin/mlb-aging train                          # all five metrics
.venv/bin/mlb-aging baselines                      # naive ladder vs both GAM arms
.venv/bin/mlb-aging train --metric WAR --ipw       # survivorship-corrected
.venv/bin/mlb-aging train --metric wRC+ --elite 110 --elite-test --curve-reference train_mean
.venv/bin/mlb-aging train --metric Def --elite 2.5 --cohort-metric WAR --curve-reference train_mean
.venv/bin/mlb-aging fetch                          # BROKEN -- FanGraphs blocked, see below
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

No defect is currently preserved. Never loosen a tolerance to make a change pass.

### Not bugs — three claims that did not survive checking

All three were initially recorded as defects and are wrong; do not "fix" them.

**The scoring weight is a choice, not a mismatch.** Def, Spd and WAR are fitted G-weighted and
scored PA-weighted (`metrics.py`, `MetricSpec.eval_weight_col`). Nothing requires the two to
agree: the fit weight is a claim about which *observations* are reliable, the eval weight about
which *errors* matter. It did arrive as `evaluate()`'s silent default — the notebooks never passed
`weight_col` — so it is now pinned explicitly and both numbers are reported side by side
(`AgingResult.test_mae_fit_weighted`, the CLI's `fit-wt MAE` column):

| metric | PA-weighted (published) | G-weighted (matches the fit) | delta |
|--------|------------------------|------------------------------|-------|
| Def    | 4.7541                 | 4.7083                       | −0.96% |
| Spd    | 0.9862                 | 0.9940                       | +0.80% |
| WAR    | 1.4966                 | 1.4697                       | −1.80% |

The argument *for* G — that Def and Spd accrue in the field, not at the plate, so PA discounts the
defensive replacement and the pinch runner — is sound in principle but nearly inert here: the test
set has no such players. `qual=100` filtered them out before the model sees them, PA/G runs 2.06
at the 1st percentile to 4.56 at the 99th (a true defensive replacement is below 1.0), and
switching schemes relocates only 6% of total weight mass. Peak ages and peak values are computed
before scoring and are **completely independent** of this; only the MAE column moves. The IPW
comparison is unaffected either way, since both arms are scored identically.

**Centering is not leaky.** `centralize_data` groups by `Season`, and the splits occupy disjoint
season ranges (1980–2019 vs 2021–2025), so every group lies wholly within one split. Centering the
concatenated frame is identical to centering each split separately — verified to 0.0 across all
18,840 rows. A player's own season does contribute to its own league mean, but that is inherent to
centering and capped at ~0.5% of the mean by the ~428 players per season.

**The career mean is a talent control, not leakage.** `add_career_mean` produces a per-player
constant. It is the **same value on both sides** — verified: 0 of 2312 training players have more than one value, and test rows
receive that identical constant (max difference 0.0). The
`.sort_values("Season").groupby("IDfg").last()` in `generate_test_data` is therefore a no-op way
of selecting it. The notebooks used to carry a stale "expanding career mean" comment describing an
implementation that no longer exists; it went with the rewrite.

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

### Training and test populations are independent

`run_metric`'s `elite_threshold` restricts *training*; `test_threshold` restricts *scoring*.
`elite_test=True` is the special case where they are equal. GAM_top's prose needs all three
shapes: a model trained and scored on everyone, the same model scored only on elites (this is
where wRC+ 19.72 → 20.47 comes from), and a specialist model trained and scored on elites.

### Baselines give the improvement claims a denominator

`baselines.py` scores a ladder on **exactly** the frame `run_metric` scores, with the same
weights: `persistence` (last season unchanged) → `delta_curve` (the population curve as
everyone's prediction) → `delta_lag` (last season plus the age adjustment) → `gam` → `gam_ipw`.
`pipeline.compare_baselines` assembles all five.

`delta_lag` is the reference (`REFERENCE_BASELINE`) because it is the strongest naive arm and
the only one seeing what the GAM sees. Improvements against it — **positive means better**,
the convention used by `IPWComparison.mae_improvement`, the CLI and the README alike:

| metric | gam | gam + ipw |
|--------|-----|-----------|
| OPS    | +7.5% | +8.8% |
| wRC+   | +6.2% | +7.4% |
| Spd    | +3.6% | +4.1% |
| WAR    | +0.2% | +3.7% |
| Def    | **−4.9%** | **−2.8%** |

**Def losing is a real result, not a bug** — `test_gam_improvement_over_delta_lag` pins the
negative numbers so they cannot be quietly tuned away. It holds under G weighting too
(−4.7% / −2.4%). Individual-season MAE and population curve *shape* are different targets;
the defensive curve can still be the better shape estimate. WAR's split matters too: the GAM
alone is a coin flip, so essentially all of WAR's gain is the survivorship correction. And
plain `persistence` is within ~1.5% of `delta_lag` everywhere — most apparent accuracy on this
task is just "last season predicts this season."

The old `delta_method.ipynb` numbers (OPS 0.0876, wRC+ 22.47) were **not** comparable to the
GAM's: a 1038-row test frame instead of 887, ages ≥21 instead of ≥20, a population value as
every player's prediction, and a `centralize_data` lambda that closed over the global
`train_data` so test-season league means were weighted by *training* rows' PA (~0.027 OPS low).
Don't resurrect them.

### Notebook-facing helpers

`pipeline.compare_ipw` fits a metric both ways and returns an `IPWComparison` carrying both
`AgingResult`s, the survival diagnostics with and without `perf_col`, and mean `p_survive` by age
— one call per notebook section. `plots.py` is the only module that draws; everything else returns
data. `AgingResult.summary()` and `IPWComparison.summary()` produce the printed lines the
notebooks show, so their format lives in one place.

## Migration state

- **Ported:** everything the three GAM notebooks did — the helper block, GAM fitting, curve
  tracing, IPW, evaluation, data fetching — all covered by `tests/test_regression.py`.
- **The notebooks now import `mlb_aging`.** `GAM.ipynb` (87 → 17 cells), `GAM_top.ipynb`
  (108 → 21) and `GAM_IPW.ipynb` (39 → 20) hold narrative, one call per section, and plots. The
  duplicated helper block is gone, so there is one implementation and it is the tested one.
- `delta_method.ipynb` (30 -> 9 cells) is now the baseline comparison, not a superseded
  alternative: it is what the GAM's improvement claims are measured against.
- `data/outside_data.csv` has a different FanGraphs-export schema and is unused.

### The fetch stage does not work

FanGraphs put the whole site behind a Cloudflare bot challenge around **April 2026** (`cf-mitigated:
challenge` on every path, including `/api/leaders/major-league/data` and the homepage). pybaseball
is abandoned — 2.2.7 from Sept 2023 is the newest release *and* master still targets the retired
`leaders-legacy.aspx` — so installing from git does not help, and neither does Colab. `fetch.py`'s
own logic is verified correct offline; only the transport is dead. It is kept because it documents
the exact query that produced the committed CSVs. Do not build a challenge solver. The committed
data still reproduces every published number, so this blocks refreshing, not reproducing.

## Editing notebooks

Prefer `NotebookEdit` over rewriting `.ipynb` JSON. The markdown cells quote specific numbers
(peak ages, MAEs, AUCs); if a change moves those, update the surrounding prose and the `README.md`
tables in the same commit.

Re-run them with:

```bash
MPLBACKEND=Agg .venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=1800 GAM.ipynb GAM_top.ipynb GAM_IPW.ipynb
```
