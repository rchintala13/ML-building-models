from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

from ml179d.models.protocols import Estimator

"""
Estimator factory
-----------------
Turns the 'estimators' section of model.yaml into unfitted sklearn-compatible
estimators. As with features/registry.py, config stays data and this module is
the only place a name becomes code.

Scaling is part of the estimator, not the data pipeline, so that it is fitted
on train only and travels with the persisted model.

The ridge_poly step order (polynomial expansion, THEN scaling) is carried over
from the previous surrogate implementation, where alpha was tuned against that
exact arrangement. Scaling before expansion would change the conditioning of
the design matrix and invalidate the tuned alpha, so do not reorder these
without retuning.
"""

SCALERS = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler,
    "none": None,
}


def _make_scaler(name: str):
    if name not in SCALERS:
        raise KeyError(f"Unknown scaler '{name}'. Available: {sorted(SCALERS)}")
    cls = SCALERS[name]
    return None if cls is None else cls()


def _build_linear(params: Mapping[str, Any]) -> Estimator:
    params = dict(params)
    scaler = _make_scaler(params.pop("scaler", "minmax"))

    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", LinearRegression(**params)))

    return Pipeline(steps)


def _build_ridge_poly(params: Mapping[str, Any]) -> Estimator:
    params = dict(params)
    degree = params.pop("degree", 2)
    include_bias = params.pop("include_bias", False)
    interaction_only = params.pop("interaction_only", False)
    scaler = _make_scaler(params.pop("scaler", "minmax"))

    steps = [
        (
            "poly_features",
            PolynomialFeatures(
                degree=degree,
                include_bias=include_bias,
                interaction_only=interaction_only,
            ),
        )
    ]
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", Ridge(**params)))

    return Pipeline(steps)


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


VALID_SCENARIOS = ("proposed", "baseline")


class ConfigEstimatorFactory:
    """
    Resolves a model_type through the 'estimators' section of model.yaml.

    Hyperparameters resolve in increasing precedence:

        params                              the model_type default
        overrides[target_set][scenario]     tuned per target and scenario
        params= argument                    explicit caller override

    The override axes exist because alpha was tuned separately per target set
    and scenario in the previous implementation -- natural gas proposed differs
    from every other combination.
    """

    def __init__(self, estimators: Mapping[str, Mapping[str, Any]]):
        self._estimators = dict(estimators or {})

    def _override_params(
        self,
        spec: Mapping[str, Any],
        model_type: str,
        target_set: Optional[str],
        scenario: Optional[str],
    ) -> Dict[str, Any]:
        overrides = spec.get("overrides") or {}
        if not overrides or target_set is None:
            return {}

        by_target = overrides.get(target_set)
        if by_target is None:
            return {}

        unknown = [s for s in by_target if s not in VALID_SCENARIOS]
        if unknown:
            raise ValueError(
                f"estimators['{model_type}'].overrides['{target_set}'] has unknown "
                f"scenario key(s) {unknown}. Expected {list(VALID_SCENARIOS)}."
            )

        if scenario is None:
            return {}
        return dict(by_target.get(scenario) or {})

    def __call__(
        self,
        model_type: str,
        params: Optional[Mapping[str, Any]] = None,
        *,
        target_set: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> Estimator:
        if model_type not in self._estimators:
            raise KeyError(
                f"model_type '{model_type}' has no entry in the 'estimators' section "
                f"of model.yaml. Available: {sorted(self._estimators)}"
            )

        spec = self._estimators[model_type]
        if "kind" not in spec:
            raise ValueError(f"estimators['{model_type}'] is missing 'kind'.")

        merged = dict(spec.get("params", {}))
        merged.update(self._override_params(spec, model_type, target_set, scenario))
        merged.update(dict(params or {}))

        return build_estimator(spec["kind"], merged)
