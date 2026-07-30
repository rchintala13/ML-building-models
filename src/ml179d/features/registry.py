from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

import pandas as pd

from ml179d.features import engineering as eng

"""
Registry
--------
Maps the function names used in configs/model.yaml to the callables in
features/engineering.py. Config is data; this module is the only place that
turns a name string into executable code.

Two kinds of entries:

BASE_FEATURES
    Derived columns computed before feature selection. Some need usecase
    context (building type), so each entry is wrapped in an adapter with a
    uniform (df, ctx) signature.

TRANSFORMS
    Model-type specific reshaping applied after feature selection. These take
    their arguments from the 'params' block in model.yaml, so they are called
    as fn(df, **params).
"""


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """
    Usecase-level facts that base feature functions may need.

    building_type_slug:
        The SLUGGED building type ('small_office', 'retail_stripmall'), not the
        raw BEM value. add_sa_to_vol_ratio matches on substrings of the lowered
        string, and 'SmallOffice'.lower() == 'smalloffice' would not match.
    """
    usecase_id: str
    building_type_slug: str
    system_type_slug: str
    climate_zone_slug: str


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """
    One entry from model.yaml -> model_type_overrides.<type>.transforms
    """
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)


BaseFeatureFn = Callable[[pd.DataFrame, FeatureContext], pd.DataFrame]


BASE_FEATURES: Dict[str, BaseFeatureFn] = {
    "add_roof_area": lambda df, ctx: eng.add_roof_area(df),
    "add_bldg_volume": lambda df, ctx: eng.add_bldg_volume(df),
    "add_ach_infiltration": lambda df, ctx: eng.add_ach_infiltration(df),
    "add_sa_to_vol_ratio": lambda df, ctx: eng.add_sa_to_vol_ratio(
        df, building_type=ctx.building_type_slug
    ),
    "add_ext_wall_surface_area": lambda df, ctx: eng.add_ext_wall_surface_area(
        df, building_type=ctx.building_type_slug
    ),
    "add_window_area": lambda df, ctx: eng.add_window_area(
        df, building_type=ctx.building_type_slug
    ),
}


TRANSFORMS: Dict[str, Callable[..., pd.DataFrame]] = {
    "add_log_transforms": eng.add_log_transforms,
    "add_piecewise_feature": eng.add_piecewise_feature,
}


def get_base_feature(name: str) -> BaseFeatureFn:
    if name not in BASE_FEATURES:
        raise KeyError(
            f"Unknown base feature '{name}'. Available: {sorted(BASE_FEATURES)}"
        )
    return BASE_FEATURES[name]


def get_transform(name: str) -> Callable[..., pd.DataFrame]:
    if name not in TRANSFORMS:
        raise KeyError(
            f"Unknown transform '{name}'. Available: {sorted(TRANSFORMS)}"
        )
    return TRANSFORMS[name]


def apply_base_features(
    df: pd.DataFrame,
    names: Sequence[str],
    ctx: FeatureContext,
) -> pd.DataFrame:
    """
    Apply base feature functions in the order given.

    Note the '_cal' convention: add_roof_area writes 'roof_area_cal' and
    add_ach_infiltration writes 'ACH_infiltration_cal', which sit alongside the
    simulated 'roof_area' and 'ACHInfiltration' schema columns rather than
    replacing them. add_bldg_volume and add_sa_to_vol_ratio read the SIMULATED
    'roof_area', so they depend on the schema column, not on add_roof_area.
    """
    for name in names:
        df = get_base_feature(name)(df, ctx)
    return df


def apply_transforms(
    df: pd.DataFrame,
    transforms: Sequence[TransformSpec],
) -> pd.DataFrame:
    """
    Apply model-type transforms in the order given.
    """
    for spec in transforms:
        df = get_transform(spec.name)(df, **dict(spec.params))
    return df
