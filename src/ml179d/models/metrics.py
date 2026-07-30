from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

"""
Regression metrics for the surrogate models.

Alongside the usual r2/mae/rmse this reports CV(RMSE) and NMBE, the calibration
metrics used in ASHRAE Guideline 14 and referenced by 179D workflows. Both are
normalized by the mean of the observed values, so they are comparable across
usecases with very different absolute energy magnitudes -- which matters here
because a SmallOffice in CZ1A and a RetailStripmall in CZ8 are not on the same
scale.
"""


def _as_1d(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.ravel()


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute metrics for a single target.
    """
    true = _as_1d(y_true)
    pred = _as_1d(y_pred)

    if true.shape != pred.shape:
        raise ValueError(
            f"y_true and y_pred have different shapes: {true.shape} vs {pred.shape}"
        )
    if true.size == 0:
        raise ValueError("Cannot compute metrics on an empty array.")

    error = pred - true
    mean_true = float(np.mean(true))

    sse = float(np.sum(error ** 2))
    sst = float(np.sum((true - mean_true) ** 2))

    rmse = float(np.sqrt(sse / true.size))
    mae = float(np.mean(np.abs(error)))

    # r2 is undefined when every observation is identical
    r2 = float("nan") if sst == 0 else 1.0 - sse / sst

    if mean_true == 0:
        cvrmse = float("nan")
        nmbe = float("nan")
    else:
        cvrmse = 100.0 * rmse / abs(mean_true)
        nmbe = 100.0 * float(np.sum(error)) / (true.size * mean_true)

    return {
        "n": int(true.size),
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "cvrmse_pct": cvrmse,
        "nmbe_pct": nmbe,
        "mean_observed": mean_true,
    }


def metrics_by_target(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per target column. Columns are matched by name, and both
    frames must share the same row index so predictions stay aligned to
    building ids.
    """
    if list(y_true.columns) != list(y_pred.columns):
        raise ValueError(
            f"Target columns differ: {list(y_true.columns)} vs {list(y_pred.columns)}"
        )
    if not y_true.index.equals(y_pred.index):
        raise ValueError("y_true and y_pred must share the same row index.")

    return {
        col: regression_metrics(y_true[col], y_pred[col])
        for col in y_true.columns
    }
