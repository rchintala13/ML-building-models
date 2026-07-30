from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

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
    "add_erv_indicator": lambda df, ctx: eng.add_erv_indicator(df),
}


# Columns each base feature reads and writes. Declared rather than inferred so
# that the serving manifest can state which values a user must supply, and so
# ordering mistakes are caught before a model is fitted.
BASE_FEATURE_INPUTS: Dict[str, Tuple[str, ...]] = {
    "add_roof_area": ("gross_floor_area", "number_of_floors"),
    "add_bldg_volume": ("roof_area", "number_of_floors"),
    "add_ach_infiltration": ("gross_floor_area", "number_of_floors", "aspect_ratio"),
    "add_sa_to_vol_ratio": ("roof_area_cal", "aspect_ratio", "number_of_floors"),
    "add_ext_wall_surface_area": (
        "gross_floor_area",
        "number_of_floors",
        "aspect_ratio",
        "window_wall_ratio",
    ),
    "add_window_area": (
        "gross_floor_area",
        "number_of_floors",
        "aspect_ratio",
        "window_wall_ratio",
    ),
    "add_erv_indicator": ("erv_sensible_cooling",),
}

BASE_FEATURE_OUTPUTS: Dict[str, Tuple[str, ...]] = {
    "add_roof_area": ("roof_area_cal",),
    "add_bldg_volume": ("bldg_vol",),
    "add_ach_infiltration": ("ACH_infiltration_cal",),
    "add_sa_to_vol_ratio": ("sa_to_vol_ratio",),
    "add_ext_wall_surface_area": ("ext_wall_surface_area_cal",),
    "add_window_area": ("window_area_cal",),
    "add_erv_indicator": ("erv_present",),
}


def check_base_feature_order(names: Sequence[str]) -> None:
    """
    Verify each base feature's inputs are available when it runs.

    An input is available if it is not produced by any base feature (it comes
    from the CSV or from user input) or if its producer runs earlier.
    """
    produced_by = {
        column: name
        for name, columns in BASE_FEATURE_OUTPUTS.items()
        for column in columns
    }

    available: set = set()
    for name in names:
        for column in BASE_FEATURE_INPUTS.get(name, ()):
            if column in produced_by and column not in available:
                raise ValueError(
                    f"base_features order: '{name}' reads '{column}', which is "
                    f"produced by '{produced_by[column]}'. List "
                    f"'{produced_by[column]}' first."
                )
        available.update(BASE_FEATURE_OUTPUTS.get(name, ()))


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
    replacing them.

    Ordering constraint: add_sa_to_vol_ratio reads 'roof_area_cal', so
    add_roof_area must come first. Everything a served model depends on has to
    be derivable from user inputs, which is why it no longer reads the
    simulated 'roof_area'. add_bldg_volume still does.
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
