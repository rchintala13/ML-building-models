from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from ml179d.features.registry import TransformSpec
from ml179d.schema.types import Schema

"""
ModelConfig
-----------
Loads configs/model.yaml and resolves it into a DatasetRecipe: the concrete
list of features, targets, base feature functions, transforms and row filters
for one (target_set, model_type, usecase) combination.

Resolution order, later steps overriding earlier ones:

    base_feature_sets[target_set]        the starting feature list
    system_overrides[system_type_slug]   add/drop features for a system type
    usecase_overrides[usecase_id]        add/drop features for one usecase
    model_type_overrides[model_type]     transforms (applied after selection)

usecase_overrides may also contribute transforms and filters, which are
appended to whatever the model type already specifies.
"""


@dataclass(frozen=True, slots=True)
class FilterSpec:
    """
    One entry from a 'filters' block. Bounds are inclusive; either may be None.
    """
    column: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass(frozen=True, slots=True)
class DatasetRecipe:
    """
    Everything the pipeline needs to turn a canonical DataFrame into (X, y).
    """
    features: Tuple[str, ...]
    targets: Tuple[str, ...]
    base_features: Tuple[str, ...]
    transforms: Tuple[TransformSpec, ...]
    filters: Tuple[FilterSpec, ...]
    target_set: str
    model_type: str


def _as_filter_specs(raw: Sequence[Mapping[str, Any]]) -> Tuple[FilterSpec, ...]:
    specs: List[FilterSpec] = []
    for entry in raw or []:
        if "column" not in entry:
            raise ValueError(f"Filter entry is missing 'column': {entry}")
        specs.append(
            FilterSpec(
                column=entry["column"],
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
            )
        )
    return tuple(specs)


def _as_transform_specs(raw: Sequence[Mapping[str, Any]]) -> Tuple[TransformSpec, ...]:
    specs: List[TransformSpec] = []
    for entry in raw or []:
        if "name" not in entry:
            raise ValueError(f"Transform entry is missing 'name': {entry}")
        specs.append(
            TransformSpec(name=entry["name"], params=dict(entry.get("params", {})))
        )
    return tuple(specs)


def _apply_add_drop(
    features: List[str],
    override: Mapping[str, Any],
    *,
    source: str,
) -> List[str]:
    """
    Apply add_features / drop_features, preserving order and rejecting no-op
    drops so that a stale config name fails loudly instead of silently.
    """
    for name in override.get("add_features", []) or []:
        if name not in features:
            features.append(name)

    for name in override.get("drop_features", []) or []:
        if name not in features:
            raise ValueError(
                f"{source} drops feature '{name}', which is not in the current "
                f"feature list. Current: {features}"
            )
        features.remove(name)

    return features


@dataclass(frozen=True)
class ModelConfig:
    target_sets: Mapping[str, Sequence[str]]
    base_feature_sets: Mapping[str, Sequence[str]]
    base_features: Sequence[str]
    system_overrides: Mapping[str, Mapping[str, Any]]
    model_type_overrides: Mapping[str, Mapping[str, Any]]
    usecase_overrides: Mapping[str, Mapping[str, Any]]

    @staticmethod
    def from_yaml(path: Path) -> "ModelConfig":
        cfg = yaml.safe_load(Path(path).read_text()) or {}

        usecase_overrides = cfg.get("usecase_overrides") or {}
        if isinstance(usecase_overrides, list):
            if usecase_overrides:
                raise ValueError(
                    "'usecase_overrides' must be a mapping of usecase_id -> override, "
                    "got a non-empty list."
                )
            usecase_overrides = {}

        return ModelConfig(
            target_sets=cfg.get("target_sets", {}) or {},
            base_feature_sets=cfg.get("base_feature_sets", {}) or {},
            base_features=tuple(cfg.get("base_features", []) or ()),
            system_overrides=cfg.get("system_overrides") or {},
            model_type_overrides=cfg.get("model_type_overrides") or {},
            usecase_overrides=usecase_overrides,
        )

    def resolve(
        self,
        *,
        target_set: str,
        model_type: str,
        system_type_slug: Optional[str] = None,
        usecase_id: Optional[str] = None,
    ) -> DatasetRecipe:
        """
        Resolve the recipe for one training run.
        """
        if target_set not in self.base_feature_sets:
            raise KeyError(
                f"Unknown target_set '{target_set}'. "
                f"Available: {sorted(self.base_feature_sets)}"
            )
        if target_set not in self.target_sets:
            raise KeyError(
                f"target_set '{target_set}' has no entry in 'target_sets'."
            )
        if model_type not in self.model_type_overrides:
            raise KeyError(
                f"Unknown model_type '{model_type}'. "
                f"Available: {sorted(self.model_type_overrides)}"
            )

        features = list(self.base_feature_sets[target_set])

        if system_type_slug and system_type_slug in self.system_overrides:
            features = _apply_add_drop(
                features,
                self.system_overrides[system_type_slug],
                source=f"system_overrides['{system_type_slug}']",
            )

        uc_override: Mapping[str, Any] = {}
        if usecase_id and usecase_id in self.usecase_overrides:
            uc_override = self.usecase_overrides[usecase_id]
            features = _apply_add_drop(
                features,
                uc_override,
                source=f"usecase_overrides['{usecase_id}']",
            )

        transforms = _as_transform_specs(
            self.model_type_overrides[model_type].get("transforms", [])
        ) + _as_transform_specs(uc_override.get("transforms", []))

        filters = _as_filter_specs(uc_override.get("filters", []))

        return DatasetRecipe(
            features=tuple(features),
            targets=tuple(self.target_sets[target_set]),
            base_features=tuple(self.base_features),
            transforms=transforms,
            filters=filters,
            target_set=target_set,
            model_type=model_type,
        )


def validate_against_schema(config: ModelConfig, schema: Schema) -> None:
    """
    Every feature and target named in model.yaml must exist as a canonical
    schema column, unless it is produced by a base feature function.

    Called once at startup so a typo fails before any CSV is read.
    """
    from ml179d.features.registry import BASE_FEATURES

    canonical = set(schema.columns)
    # Columns that only exist after base features run. The '_cal' names are
    # calculated counterparts to simulated schema columns, not replacements.
    derived = {"roof_area_cal", "bldg_vol", "ACH_infiltration_cal", "sa_to_vol_ratio"}
    known = canonical | derived

    problems: List[str] = []

    for name, feats in config.base_feature_sets.items():
        unknown = [f for f in feats if f not in known]
        if unknown:
            problems.append(f"base_feature_sets['{name}']: {unknown}")

    for name, targets in config.target_sets.items():
        unknown = [t for t in targets if t not in canonical]
        if unknown:
            problems.append(f"target_sets['{name}']: {unknown}")

    for name in config.base_features:
        if name not in BASE_FEATURES:
            problems.append(
                f"base_features: unknown function '{name}', "
                f"available {sorted(BASE_FEATURES)}"
            )

    if problems:
        raise ValueError(
            "model.yaml references columns/functions that do not exist:\n  "
            + "\n  ".join(problems)
        )
