from __future__ import annotations

from typing import Any, Mapping, Protocol

import pandas as pd

"""
The boundary between the pipeline and the modeling layer.

pipeline.stage_dataset produces TrainingData(X, y); anything implementing
Estimator consumes it. Keeping this as a Protocol means the pipeline never
imports a concrete model, and sklearn/xgboost/lightgbm estimators satisfy it
as-is without a wrapper.
"""


class Estimator(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Estimator": ...

    def predict(self, X: pd.DataFrame) -> Any: ...


class EstimatorFactory(Protocol):
    """
    Builds an unfitted estimator for a model_type from its hyperparameters.

    model.yaml currently carries no hyperparameters -- 'ridge_poly',
    'plain_linear' and 'xgboost' appear only as transform recipes. Whatever
    config section supplies them should feed this.
    """
    def __call__(self, model_type: str, params: Mapping[str, Any]) -> Estimator: ...
