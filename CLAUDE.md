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

.venv/bin/python -m pytest tests/ -q              # full regression suite, 83 tests (~3min)
.venv/bin/python -m pytest tests/ -q -k IPW       # one group
.venv/bin/python -m pytest tests/test_regression.py::test_all_player_curves -q

.venv/bin/mlb-aging train                          # all five metrics
.venv/bin/mlb-aging baselines                      # naive ladder vs both GAM arms
.venv/bin/mlb-aging train --metric WAR --ipw       # survivorship-corrected
.venv/bin/mlb-aging train --metric wRC+ --elite 110 --elite-test --curve-reference train_mean
.venv/bin/mlb-aging train --metric Def --elite 2.5 --cohort-metric WAR --curve-reference train_mean
.venv/bin/mlb-aging train --model-spec tensor      # the pre-adoption age x experience variant
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

Two suites, both pinning to 1e-9, and neither may have a tolerance loosened to make a change pass:

- **`tests/test_age_only.py`** pins what is *published* — the numbers in README.md, the notebook
  outputs and `mlb-aging train`. Every call uses the default spec.
- **`tests/test_regression.py`** pins the **`tensor`** specification, transcribed from the
  notebooks' stored outputs from before the default changed. Every call passes
  `model_spec=TENSOR_SPEC` explicitly. It is the original bug-for-bug-faithful refactor contract
  and it still holds unchanged; it just no longer describes the published results.

No defect is currently preserved.

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
| Def    | 4.8156                 | 4.7724                       | −0.90% |
| Spd    | 0.9961                 | 1.0040                       | +0.80% |
| WAR    | 1.4970                 | 1.4719                       | −1.67% |

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
GAM(s(0, n_splines=5) + s(1, n_splines=5) + s(3, n_splines=5))          # DEFAULT_SPEC
GAM(te(0, 2, n_splines=5) + s(1, n_splines=5) + s(3, n_splines=5))      # TENSOR_SPEC
```

`s(1)` is the career-mean talent control and `s(3)` the lagged prior season in both. The term
indices are positional into `feature_cols`, so **reordering that list silently changes the model.**
`experience` stays in `feature_cols` under the default even though no term reads it — dropping it
would shift `s(3)` onto the wrong column.

### `age_only` is the default; `tensor` is kept and still tested

`gam.DEFAULT_SPEC` is `AGE_ONLY_SPEC`. The tensor was the original specification and scores
marginally better (0.2–1.4pp against the naive baseline, except WAR+IPW where age-only wins by
0.6pp), but tracing a curve from it requires pinning experience, and `aging_curve` pins
`experience = floor(age) - 20` — a debut at 20, which describes 43 of ~2300 training players
against 428 at the modal debut of 24. Retracing the same fitted model at debut 24 moves the peak
one to three years (Def 23.0 to 26.0). One metric then has no single peak age, which is the
question an aging curve exists to answer, so the accuracy was traded away deliberately.

Two consequences worth knowing, both measured in `curve_validation.ipynb`:

- **The traced tensor curve is a sawtooth.** `aging_curve` pins experience *and* the lag as step
  functions of `floor(age)`, so the surface falls within each integer age and jumps at the
  boundary (WAR: 1.7929 at 24.99, 1.8721 at 25.05). `argmax` therefore chooses among ~20 step
  edges and every tensor peak is an integer. Its apparent perfect stability under resampling is
  that quantization, not precision.
- **`age_only` is not more reproducible.** That was the hoped-for second argument and it did not
  materialise; the case for the default rests on debut-invariance alone.

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

### Def has no locatable peak, and Spd's is the left edge of the data

Neither metric's "peak age" should be quoted. Def's curve nominally turns at 21.99, but the rise
from the youngest fitted age is only +0.098 runs and refitting on random halves of the players
scatters it from 21.0 to 25.0 with half the resamples returning the left edge (3-4 year spread,
`peak_stability` in `diagnostics.py`). Combined with the 3-year debut swing and a residual profile
that is a flat offset rather than a bend, the conclusion is that Def's peak is not measurable at
this sample size — a resolution problem no change of specification addresses. Say "defense
declines from the start of a major-league career".

### Spd's "peak" is the left edge of the data, not a turning point

`add_lag` drops every player's first season, so no age-20 row reaches the model and the fitted
age range starts at **21**, not at `MIN_AGE`. Spd's argmax lands exactly there: the rise from the
youngest age to the "peak" is 0.0000 and the curve declines monotonically across the whole range.
The number 21 is correct and pinned; the word *peak* is what would be wrong. Say "speed is
already declining at the youngest age observed" — its true maximum is outside the data.

`AgingCurve.peaks_at_left_edge` reports this, and `GAM.ipynb` prints it for all five metrics.
Def's is nominal (rise +0.098, and a 3-4 year spread under resampling); OPS (+0.048),
wRC+ (+12.32) and WAR (+0.485) are genuine interior maxima.

### `AgingCurve.ages` is a mesh, not a sorted sweep

This applies to `TENSOR_SPEC` only; `s(0)` yields a sorted 1000-row sweep.
`generate_X_grid(term=0)` on the `te(0, 2)` tensor returns an `n x n` mesh, so `ages` holds
1,000,000 rows with each age repeated and **is not monotonic** — `np.interp` against it directly
returns nonsense. Every non-age feature is pinned before prediction, so the repeats are exact
duplicates. Use `curve.by_age` (one prediction per age, sorted), `curve.value_at(age)` or
`curve.change_between(a, b)`. `peak_age` is unaffected, since `argmax` does not care about order.

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
where wRC+ 19.77 → 20.56 comes from), and a specialist model trained and scored on elites.

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
| OPS    | +7.1% | +8.7% |
| wRC+   | +6.0% | +7.3% |
| Spd    | +2.6% | +3.3% |
| WAR    | +0.2% | +4.3% |
| Def    | **−6.3%** | **−4.3%** |

**Def losing is a real result, not a bug** — `test_gam_improvement_over_delta_lag` pins the
negative numbers so they cannot be quietly tuned away. It holds under G weighting too
(−6.1% / −4.0%). Individual-season MAE and population curve *shape* are different targets;
the defensive curve can still be the better shape estimate. WAR's split matters too: the GAM
alone is a coin flip, so essentially all of WAR's gain is the survivorship correction. And
plain `persistence` is within ~1.7% of `delta_lag` everywhere — most apparent accuracy on this
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
- `curve_validation.ipynb` asks whether the curves are trustworthy at all: residuals by age,
  sensitivity to the debut assumption, the `age_only` vs `tensor` comparison, and peak
  reproducibility under player-level resampling. It is where the case for the current default
  is argued and where Def's peak is shown to be unmeasurable.
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
