"""Test-set scoring and inference."""

from __future__ import annotations

import pandas as pd
from pygam import GAM
from sklearn.metrics import mean_absolute_error

from mlb_aging.features import generate_data
from mlb_aging.metrics import MetricSpec


def evaluate(data: pd.DataFrame, spec: MetricSpec, model: GAM) -> float:
    """Weighted MAE on the test frame.

    Uses :attr:`MetricSpec.eval_weight_col`, which currently reproduces the
    notebooks' PA-weighted scoring for every metric -- see the note there.
    """
    test_x, test_y = generate_data(data, spec.feature_cols, spec.target_col)
    y_preds = model.predict(test_x)
    return mean_absolute_error(
        test_y, y_preds, sample_weight=data[spec.eval_weight_col].values
    )


def inference(data: pd.DataFrame, spec: MetricSpec, model: GAM) -> pd.DataFrame:
    """Return ``data`` with a ``prediction`` column appended."""
    test_x, _ = generate_data(data, spec.feature_cols, spec.target_col)
    inference_df = data.copy()
    inference_df["prediction"] = model.predict(test_x)
    return inference_df
