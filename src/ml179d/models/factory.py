from __future__ import annotations

from typing import Any, Callable, Dict, Mapping

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from ml179d.models.protocols import Estimator

"""
Estimator factory
-----------------
Turns the 'estimators' section of model.yaml into unfitted sklearn-compatible
estimators. As with features/registry.py, config stays data and this module is
the only place a name becomes code.

Scaling is part of the estimator, not the pipeline, so that it is fitted on
train only and travels with the persisted model. The surrogate features are on
wildly different scales (floor area in m^2 vs U-factors near 0.3), which makes
this non optional for the penalized linear models.
"""


def _build_linear(params: Mapping[str, Any]) -> Estimator:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LinearRegression(**dict(params))),
        ]
    )


def _build_ridge_poly(params: Mapping[str, Any]) -> Estimator:
    params = dict(params)
    degree = params.pop("degree", 2)
    include_bias = params.pop("include_bias", False)
    interaction_only = params.pop("interaction_only", False)

    return Pipeline(
        [
            (
                "poly",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=include_bias,
                    interaction_only=interaction_only,
                ),
            ),
            ("scale", StandardScaler()),
            ("model", Ridge(**params)),
        ]
    )


def _build_xgboost(params: Mapping[str, Any]) -> Estimator:
    # Imported lazily so the package works without xgboost installed.
    from xgboost import XGBRegressor

    return XGBRegressor(**dict(params))


BUILDERS: Dict[str, Callable[[Mapping[str, Any]], Estimator]] = {
    "linear": _build_linear,
    "ridge_poly": _build_ridge_poly,
    "xgboost": _build_xgboost,
}


def build_estimator(kind: str, params: Mapping[str, Any]) -> Estimator:
    if kind not in BUILDERS:
        raise KeyError(
            f"Unknown estimator kind '{kind}'. Available: {sorted(BUILDERS)}"
        )
    return BUILDERS[kind](params)


class ConfigEstimatorFactory:
    """
    Resolves a model_type through the 'estimators' section of model.yaml.
    """

    def __init__(self, estimators: Mapping[str, Mapping[str, Any]]):
        self._estimators = dict(estimators or {})

    def __call__(self, model_type: str, params: Mapping[str, Any] | None = None) -> Estimator:
        if model_type not in self._estimators:
            raise KeyError(
                f"model_type '{model_type}' has no entry in the 'estimators' section "
                f"of model.yaml. Available: {sorted(self._estimators)}"
            )

        spec = self._estimators[model_type]
        if "kind" not in spec:
            raise ValueError(f"estimators['{model_type}'] is missing 'kind'.")

        merged = dict(spec.get("params", {}))
        merged.update(dict(params or {}))

        return build_estimator(spec["kind"], merged)
