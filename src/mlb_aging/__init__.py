"""MLB position player aging curves: GAM fits with IPW survivorship-bias correction."""

from mlb_aging.metrics import METRICS, MetricSpec, get_metric

__all__ = ["METRICS", "MetricSpec", "get_metric"]
