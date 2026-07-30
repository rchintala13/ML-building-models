from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import joblib
import pandas as pd
import sklearn

from ml179d.config import DatasetRecipe
from ml179d.features.registry import (
    BASE_FEATURE_INPUTS,
    BASE_FEATURE_OUTPUTS,
    FeatureContext,
    TransformSpec,
    apply_base_features,
    apply_transforms,
)
from ml179d.train import FittedModel

"""
Serving
-------
The contract between this repository and the web calculator.

A manifest declares, for one (usecase, target_set, model_type):

    user_inputs   what the calculator must ask the user for
    derived       what this package computes from those inputs
    base_features / transforms   the recipe, in order
    fitted_features              the exact columns and order the model expects

predict_savings replays that recipe, so the calculator never reimplements
feature engineering. It needs only the manifest, the two joblib files and this
package -- no configs/ directory and no schema.yaml.

    savings = baseline_prediction - proposed_prediction
"""

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1


# ---------------------------------------------------------------
# building the manifest
# ---------------------------------------------------------------

def user_input_columns(recipe: DatasetRecipe) -> List[str]:
    """
    Columns the caller must supply: everything the recipe needs that this
    package does not compute.
    """
    derived: set = set()
    required: set = set(recipe.features)

    for name in recipe.base_features:
        required.update(BASE_FEATURE_INPUTS.get(name, ()))
        derived.update(BASE_FEATURE_OUTPUTS.get(name, ()))

    return sorted(required - derived)


def derived_columns(recipe: DatasetRecipe) -> List[str]:
    derived: set = set()
    for name in recipe.base_features:
        derived.update(BASE_FEATURE_OUTPUTS.get(name, ()))
    return sorted(derived)


def build_manifest(
    *,
    models: Mapping[str, FittedModel],
    recipe: DatasetRecipe,
    usecase_id: str,
    building_type_slug: str,
    system_type_slug: str,
    climate_zone_slug: str,
    schema_units: Optional[Mapping[str, str]] = None,
    schema_dtypes: Optional[Mapping[str, str]] = None,
    categories: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Any]:
    """
    Assemble the serving contract for one usecase / target set / model type.
    """
    units = dict(schema_units or {})
    dtypes = dict(schema_dtypes or {})
    bounds = {
        spec.column: {"min": spec.min_value, "max": spec.max_value}
        for spec in recipe.filters
    }

    inputs = []
    for name in user_input_columns(recipe):
        entry: Dict[str, Any] = {"name": name}
        if name in dtypes:
            entry["dtype"] = dtypes[name]
        if units.get(name):
            entry["unit"] = units[name]
        if name in bounds:
            entry["bounds"] = bounds[name]
        if categories and name in categories:
            entry["categories"] = list(categories[name])
        inputs.append(entry)

    any_model = next(iter(models.values()))

    return {
        "manifest_version": MANIFEST_VERSION,
        "usecase_id": usecase_id,
        "building_type_slug": building_type_slug,
        "system_type_slug": system_type_slug,
        "climate_zone_slug": climate_zone_slug,
        "target_set": recipe.target_set,
        "model_type": recipe.model_type,
        "targets": list(any_model.target_names),
        "savings_definition": "baseline - proposed",
        "user_inputs": inputs,
        "derived": derived_columns(recipe),
        "recipe": {
            "base_features": list(recipe.base_features),
            "transforms": [
                {"name": t.name, "params": dict(t.params)} for t in recipe.transforms
            ],
        },
        "fitted_features": list(any_model.feature_names),
        "models": {
            scenario: {
                "file": f"{scenario}.joblib",
                "n_train": model.n_train,
                "test_metrics": model.test_metrics,
            }
            for scenario, model in sorted(models.items())
        },
        "provenance": {
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }


def save_manifest(manifest: Mapping[str, Any], *, directory: Path) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ---------------------------------------------------------------
# serving
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ModelBundle:
    """
    A manifest plus its fitted estimators, loaded from disk.
    """
    manifest: Dict[str, Any]
    estimators: Dict[str, Any]
    directory: Path

    @property
    def usecase_id(self) -> str:
        return self.manifest["usecase_id"]

    @property
    def required_inputs(self) -> List[str]:
        return [entry["name"] for entry in self.manifest["user_inputs"]]

    @property
    def fitted_features(self) -> List[str]:
        return list(self.manifest["fitted_features"])


def load_bundle(directory: Path) -> ModelBundle:
    """
    Load a manifest and both scenario models from a model directory.
    """
    directory = Path(directory)
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"No {MANIFEST_NAME} in {directory}")

    manifest = json.loads(manifest_path.read_text())

    estimators = {}
    for scenario, entry in manifest["models"].items():
        path = directory / entry["file"]
        if not path.exists():
            raise FileNotFoundError(f"Model file missing: {path}")
        estimators[scenario] = joblib.load(path)

    return ModelBundle(manifest=manifest, estimators=estimators, directory=directory)


def _validate_inputs(bundle: ModelBundle, user_inputs: Mapping[str, Any]) -> None:
    missing = [name for name in bundle.required_inputs if name not in user_inputs]
    if missing:
        raise KeyError(f"Missing required user input(s): {missing}")

    unexpected = [k for k in user_inputs if k not in set(bundle.required_inputs)]
    if unexpected:
        raise KeyError(
            f"Unexpected input(s) {unexpected}. This model accepts "
            f"{bundle.required_inputs}."
        )


def check_bounds(
    bundle: ModelBundle, user_inputs: Mapping[str, Any]
) -> List[str]:
    """
    Return a message per input outside the range the model was fitted on.

    Out-of-range values are extrapolation, not an error, so this reports rather
    than raises; the caller decides whether to warn or refuse.
    """
    warnings: List[str] = []

    for entry in bundle.manifest["user_inputs"]:
        bounds = entry.get("bounds")
        if not bounds:
            continue

        value = user_inputs.get(entry["name"])
        if value is None:
            continue

        low, high = bounds.get("min"), bounds.get("max")
        if low is not None and value < low:
            warnings.append(
                f"{entry['name']}={value} is below the fitted range (min {low})"
            )
        if high is not None and value > high:
            warnings.append(
                f"{entry['name']}={value} is above the fitted range (max {high})"
            )

    return warnings


def build_feature_frame(
    bundle: ModelBundle, user_inputs: Mapping[str, Any]
) -> pd.DataFrame:
    """
    Replay the recipe: user inputs -> the columns the model was fitted on.
    """
    _validate_inputs(bundle, user_inputs)

    frame = pd.DataFrame([dict(user_inputs)])

    context = FeatureContext(
        usecase_id=bundle.manifest["usecase_id"],
        building_type_slug=bundle.manifest["building_type_slug"],
        system_type_slug=bundle.manifest["system_type_slug"],
        climate_zone_slug=bundle.manifest["climate_zone_slug"],
    )

    frame = apply_base_features(
        frame, bundle.manifest["recipe"]["base_features"], context
    )
    frame = apply_transforms(
        frame,
        [
            TransformSpec(name=t["name"], params=t.get("params", {}))
            for t in bundle.manifest["recipe"]["transforms"]
        ],
    )

    missing = [c for c in bundle.fitted_features if c not in frame.columns]
    if missing:
        raise KeyError(
            f"Recipe did not produce fitted feature(s) {missing}. The manifest "
            f"and the installed ml179d version may be out of step."
        )

    return frame[bundle.fitted_features]


@dataclass(frozen=True)
class SavingsPrediction:
    usecase_id: str
    target_set: str
    target: str
    proposed: float
    baseline: float
    savings: float
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "usecase_id": self.usecase_id,
            "target_set": self.target_set,
            "target": self.target,
            "proposed": self.proposed,
            "baseline": self.baseline,
            "savings": self.savings,
            "warnings": list(self.warnings),
        }


def predict_savings(
    bundle: ModelBundle,
    user_inputs: Mapping[str, Any],
    *,
    check_range: bool = True,
) -> SavingsPrediction:
    """
    Estimate energy savings for one building.

        savings = baseline - proposed

    Both models consume the same user inputs: the baseline model was trained on
    proposed inputs against baseline energy, so the difference is the saving
    attributable to the proposed design.
    """
    for scenario in ("proposed", "baseline"):
        if scenario not in bundle.estimators:
            raise KeyError(
                f"Bundle for '{bundle.usecase_id}' has no '{scenario}' model, "
                f"so savings cannot be computed."
            )

    features = build_feature_frame(bundle, user_inputs)

    predictions = {
        scenario: float(pd.Series(estimator.predict(features)).iloc[0])
        for scenario, estimator in bundle.estimators.items()
    }

    return SavingsPrediction(
        usecase_id=bundle.usecase_id,
        target_set=bundle.manifest["target_set"],
        target=bundle.manifest["targets"][0],
        proposed=predictions["proposed"],
        baseline=predictions["baseline"],
        savings=predictions["baseline"] - predictions["proposed"],
        warnings=check_bounds(bundle, user_inputs) if check_range else [],
    )
