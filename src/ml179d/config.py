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

    filters_apply_to_test:
        False (the default) filters the train split only, so models are fitted
        on a restricted range but scored against the full test distribution.
        True filters both splits.
    """
    features: Tuple[str, ...]
    targets: Tuple[str, ...]
    base_features: Tuple[str, ...]
    transforms: Tuple[TransformSpec, ...]
    filters: Tuple[FilterSpec, ...]
    target_set: str
    model_type: str
    filters_apply_to_test: bool = False


FILTER_SELECTOR_FIELDS = ("target_set", "scenario", "usecase_id")


def _selector_matches(
    when: Mapping[str, Any],
    *,
    target_set: Optional[str],
    scenario: Optional[str],
    usecase_id: Optional[str],
) -> bool:
    """
    A selector matches when every field it names matches; omitted fields are
    wildcards. Same rule as the 'disallow' constraints in usecase_space.yaml.
    """
    unknown = [k for k in when if k not in FILTER_SELECTOR_FIELDS]
    if unknown:
        raise ValueError(
            f"Filter override selector has unknown field(s) {unknown}. "
            f"Expected any of {list(FILTER_SELECTOR_FIELDS)}."
        )

    actual = {
        "target_set": target_set,
        "scenario": scenario,
        "usecase_id": usecase_id,
    }

    for field_name, expected in when.items():
        current = actual[field_name]
        if current is None:
            return False
        allowed = (expected,) if isinstance(expected, str) else tuple(expected)
        if current not in allowed:
            return False

    return True


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
    estimators: Mapping[str, Mapping[str, Any]]
    filters: Mapping[str, Any]

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
            estimators=cfg.get("estimators") or {},
            filters=cfg.get("filters") or {},
        )

    def resolve_filters(
        self,
        *,
        target_set: Optional[str] = None,
        scenario: Optional[str] = None,
        usecase_id: Optional[str] = None,
    ) -> Tuple[Tuple[FilterSpec, ...], bool]:
        """
        Merge the global bounds with every matching override, in file order.

        Merging is per column, so an override that tightens one column leaves
        the other global bounds intact. Setting a bound to null removes it.
        """
        block = self.filters or {}

        min_values: Dict[str, Any] = dict(block.get("min_values") or {})
        max_values: Dict[str, Any] = dict(block.get("max_values") or {})
        apply_to_test = bool(block.get("apply_to_test", False))

        overrides = block.get("overrides") or []
        if not isinstance(overrides, list):
            raise ValueError(
                f"'filters.overrides' must be a list of rules, got "
                f"{type(overrides).__name__}."
            )

        for rule in overrides:
            if not _selector_matches(
                rule.get("when") or {},
                target_set=target_set,
                scenario=scenario,
                usecase_id=usecase_id,
            ):
                continue

            min_values.update(rule.get("min_values") or {})
            max_values.update(rule.get("max_values") or {})
            if "apply_to_test" in rule:
                apply_to_test = bool(rule["apply_to_test"])

        specs = tuple(
            FilterSpec(
                column=column,
                min_value=min_values.get(column),
                max_value=max_values.get(column),
            )
            for column in sorted(set(min_values) | set(max_values))
            # a column whose bounds were both cleared is not a filter
            if min_values.get(column) is not None or max_values.get(column) is not None
        )

        return specs, apply_to_test

    def resolve(
        self,
        *,
        target_set: str,
        model_type: str,
        system_type_slug: Optional[str] = None,
        usecase_id: Optional[str] = None,
        scenario: Optional[str] = None,
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

        if uc_override.get("filters"):
            raise ValueError(
                f"usecase_overrides['{usecase_id}'] defines 'filters'. Filters now "
                f"live in the top-level 'filters' block; use an override with "
                f"when.usecase_id instead."
            )

        filters, filters_apply_to_test = self.resolve_filters(
            target_set=target_set,
            scenario=scenario,
            usecase_id=usecase_id,
        )

        return DatasetRecipe(
            features=tuple(features),
            targets=tuple(self.target_sets[target_set]),
            base_features=tuple(self.base_features),
            transforms=transforms,
            filters=filters,
            target_set=target_set,
            model_type=model_type,
            filters_apply_to_test=filters_apply_to_test,
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
    derived = {
        "roof_area_cal",
        "bldg_vol",
        "ACH_infiltration_cal",
        "sa_to_vol_ratio",
        "ext_wall_surface_area_cal",
        "window_area_cal",
        "erv_present",
    }
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

    if not problems:
        from ml179d.features.registry import check_base_feature_order

        try:
            check_base_feature_order(config.base_features)
        except ValueError as exc:
            problems.append(str(exc))

    if problems:
        raise ValueError(
            "model.yaml references columns/functions that do not exist:\n  "
            + "\n  ".join(problems)
        )
