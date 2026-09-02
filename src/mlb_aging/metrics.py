"""Per-metric configuration.

The five metrics differ along four axes only. Capturing them here removes the
copy-pasted per-metric sections that the original notebooks repeated five times.
"""

from __future__ import annotations

from dataclasses import dataclass

# Elite-player thresholds used by the top-player analysis (GAM_top.ipynb).
ELITE_WRC_THRESHOLD = 110
ELITE_WAR_THRESHOLD = 2.5


@dataclass(frozen=True)
class MetricSpec:
    """How one performance metric is modelled.

    Attributes
    ----------
    name
        Raw column name in the source CSVs.
    centralized
        Whether the metric is re-centred against the PA-weighted league mean of
        its season before modelling. Only OPS is.
    weight_col
        Sample-weight column used when *fitting* the GAM: plate appearances for
        rate-style hitting metrics, games played for the rest.
    eval_weight_col
        Sample-weight column used when *evaluating* test MAE. ``'PA'`` for all
        five metrics, so Def, Spd and WAR are fitted G-weighted but scored
        PA-weighted.

        This is a deliberate choice, not a mismatch to fix. The two weights
        answer different questions: the fit weight is a statistical claim about
        which *observations* are reliable (a 150-game season measures Def
        better than a 20-game one), while the eval weight says which *errors*
        matter to the reader. Nothing requires them to agree.

        It reached here as ``evaluate()``'s silent default -- the notebooks
        never passed ``weight_col`` -- so it is pinned explicitly now rather
        than inherited. Scoring Def and Spd by plate appearances is the weaker
        half of the choice, since both accrue in the field rather than at the
        plate; the population that would expose it (defensive replacements,
        pinch runners) is absent from the test set, which ``qual=100`` already
        filtered to regulars. Test-set PA/G runs 2.06 at the 1st percentile to
        4.56 at the 99th, and switching schemes relocates 6% of total weight
        mass, moving MAE by 1-2%. ``AgingResult.test_mae_fit_weighted`` reports
        the G-weighted number alongside, so the choice needs no argument.

        It affects nothing but the MAE column: every peak age and peak value is
        computed before scoring and is independent of this.
    curve_reference
        Value pinned into the career-mean column when tracing the aging curve.
        ``None`` leaves pyGAM's own grid value in place (OPS), a float pins a
        fixed reference (wRC+ uses 100, league average), and ``"train_mean"``
        pins the training-set mean of the metric.
    """

    name: str
    centralized: bool
    weight_col: str
    eval_weight_col: str
    curve_reference: float | str | None

    @property
    def target_col(self) -> str:
        """Column actually modelled -- suffixed when the metric is centralized."""
        return f"{self.name}_centralized" if self.centralized else self.name

    @property
    def lag_col(self) -> str:
        return f"{self.target_col}_lag"

    @property
    def career_mean_col(self) -> str:
        return f"{self.target_col}_career_mean"

    @property
    def feature_cols(self) -> list[str]:
        """Model features, in the order the GAM term indices assume.

        ``te(0, 2)`` is the age x experience tensor product, ``s(1)`` the
        career-mean talent control and ``s(3)`` the lagged prior season, so
        this ordering is part of the model specification -- not cosmetic.
        """
        return ["Age", self.career_mean_col, "experience", self.lag_col]


METRICS: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        MetricSpec("OPS",  centralized=True,  weight_col="PA", eval_weight_col="PA", curve_reference=None),
        MetricSpec("wRC+", centralized=False, weight_col="PA", eval_weight_col="PA", curve_reference=100.0),
        MetricSpec("Def",  centralized=False, weight_col="G",  eval_weight_col="PA", curve_reference="train_mean"),
        MetricSpec("Spd",  centralized=False, weight_col="G",  eval_weight_col="PA", curve_reference="train_mean"),
        MetricSpec("WAR",  centralized=False, weight_col="G",  eval_weight_col="PA", curve_reference="train_mean"),
    )
}


def get_metric(name: str) -> MetricSpec:
    try:
        return METRICS[name]
    except KeyError:
        raise KeyError(f"unknown metric {name!r}; expected one of {sorted(METRICS)}") from None
