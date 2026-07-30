from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# BASE CALCULATED FEATURES
# ============================================================

def add_roof_area(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roof_area_cal"] = df["gross_floor_area"] / df["number_of_floors"]
    return df


def add_bldg_volume(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bldg_vol"] = df["roof_area"] * df["number_of_floors"]
    return df


def add_ach_infiltration(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    AF = df["gross_floor_area"] / df["number_of_floors"]
    I = 0.0115824

    val = (
        I * 120 * (
            np.sqrt(df["aspect_ratio"] * AF) +
            np.sqrt(AF / df["aspect_ratio"])
        )
        / (AF * df["number_of_floors"])
    ) * df["number_of_floors"]

    df["ACH_infiltration_cal"] = val
    return df


def add_sa_to_vol_ratio(
    df: pd.DataFrame,
    *,
    building_type: str,
) -> pd.DataFrame:
    """
    Surface-area-to-volume ratio.

    Reads roof_area_cal, the calculated per-floor footprint, rather than the
    simulated roof_area: the web calculator only has user inputs, so a feature
    depending on a simulated column could not be served. Requires add_roof_area
    to have run first.
    """
    df = df.copy()

    # safer building type check
    bt = building_type.lower()

    if "small_office" in bt:
        ceil_height = 10
    elif "retail" in bt:
        ceil_height = 17
    else:
        raise ValueError(f"Unknown building type for SA/V ratio: {building_type}")

    if "roof_area_cal" not in df.columns:
        raise KeyError(
            "add_sa_to_vol_ratio needs 'roof_area_cal'; list add_roof_area "
            "before add_sa_to_vol_ratio in base_features."
        )

    df["aspect_ratio"] = df["aspect_ratio"].replace(0, 1.9)

    term1 = np.sqrt(df["aspect_ratio"] / df["roof_area_cal"])
    term2 = np.sqrt(1.0 / (df["aspect_ratio"] * df["roof_area_cal"]))

    df["sa_to_vol_ratio"] = (
        2.0 * (term1 + term2) +
        (1.0 / (ceil_height * df["number_of_floors"]))
    )

    return df


# Values of erv_sensible_cooling that mean "no energy recovery ventilator".
# The proposed source encodes absence as 0.0 and the baseline support column
# as the sentinel 999.0, so the raw value is not comparable across scenarios.
ERV_ABSENT_VALUES = (0.0, 999.0)


def add_erv_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary 'is there an ERV' flag derived from erv_sensible_cooling.

    0.0 and 999.0 both mean no ERV and become 0; any genuine effectiveness
    fraction becomes 1. Missing values are treated as absent.

    This makes the feature identical across scenarios, which the raw column is
    not: feeding a baseline model trained on 999.0 a proposed value of 0.0 put
    it far outside its fitted range.
    """
    df = df.copy()

    values = pd.to_numeric(df["erv_sensible_cooling"], errors="coerce")
    absent = values.isna() | values.isin(ERV_ABSENT_VALUES)

    df["erv_present"] = (~absent).astype(float)
    return df


def _floor_height_m(building_type: str) -> float:
    """
    Floor-to-floor height in metres, from the prototype building geometry.

    Matches calculate_ext_wall_surface_area below. Accepts the slugged building
    type ('small_office', 'retail_stripmall').
    """
    bt = building_type.lower()

    if "small_office" in bt or "small office" in bt:
        return 10 * 0.3048
    if "retail" in bt:
        return 17 * 0.3048

    raise ValueError(f"Unknown building type for floor height: {building_type}")


def _gross_ext_wall_area(df: pd.DataFrame, *, building_type: str) -> pd.Series:
    """
    Gross exterior wall area (windows included) from user-supplied geometry.

    Treats the building as a rectangular prism with plan aspect ratio
    'aspect_ratio' and per-floor area gross_floor_area / number_of_floors.
    """
    floor_height = _floor_height_m(building_type)
    AF = df["gross_floor_area"] / df["number_of_floors"]

    perimeter_term = np.sqrt(df["aspect_ratio"] * AF) + np.sqrt(AF / df["aspect_ratio"])

    return 2 * floor_height * perimeter_term * df["number_of_floors"]


def add_ext_wall_surface_area(
    df: pd.DataFrame,
    *,
    building_type: str,
) -> pd.DataFrame:
    """
    Net exterior wall area (gross minus glazing), computed from user inputs.

    This is the servable counterpart to the simulated 'ext_wall_surface_area'
    column: the web calculator only has user inputs, so training on the
    calculated value avoids train/serve skew.
    """
    df = df.copy()
    gross = _gross_ext_wall_area(df, building_type=building_type)
    df["ext_wall_surface_area_cal"] = (1.0 - df["window_wall_ratio"]) * gross
    return df


def add_window_area(
    df: pd.DataFrame,
    *,
    building_type: str,
) -> pd.DataFrame:
    """
    Glazing area, computed from user inputs.

    Derived from the gross wall area directly rather than from
    ext_wall_surface_area_cal, so it does not depend on step order and does not
    divide by (1 - window_wall_ratio).
    """
    df = df.copy()
    gross = _gross_ext_wall_area(df, building_type=building_type)
    df["window_area_cal"] = df["window_wall_ratio"] * gross
    return df


# ============================================================
# USER INPUT CALCULATIONS (optional, separate context)
# ============================================================

def calculate_ext_wall_surface_area(user_inputs: dict) -> dict:
    bt = user_inputs["building_type"].lower()

    if "small office" in bt:
        floor_height = 10 * 0.3048
    elif "retail" in bt:
        floor_height = 17 * 0.3048
    else:
        raise ValueError(f"Unknown building type: {bt}")

    AF = user_inputs["gross_floor_area"] / user_inputs["number_of_floors"]

    ext_wall_surface_area_gross = (
        2 * floor_height *
        (
            np.sqrt(user_inputs["aspect_ratio"] * AF) +
            np.sqrt(AF / user_inputs["aspect_ratio"])
        )
    ) * user_inputs["number_of_floors"]

    ext_wall_surface_area = (
        (1 - user_inputs["window_wall_ratio"]) *
        ext_wall_surface_area_gross
    )

    user_inputs["ext_wall_surface_area"] = ext_wall_surface_area
    return user_inputs


def calculate_window_area(user_inputs: dict) -> dict:
    ext_wall_surface_area = user_inputs["ext_wall_surface_area"]

    ext_wall_surface_area_gross = (
        ext_wall_surface_area /
        (1 - user_inputs["window_wall_ratio"])
    )

    window_area = (
        user_inputs["window_wall_ratio"] *
        ext_wall_surface_area_gross
    )

    user_inputs["window_area"] = window_area
    return user_inputs


# ============================================================
# TRANSFORMS
# ============================================================

def add_log_transforms(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found for log transform.")

        df[f"{col}_log"] = np.log(df[col].clip(lower=1e-8))

    return df


# ============================================================
# PIECEWISE / BREAKPOINT FEATURES
# ============================================================

def add_piecewise_feature(
    df: pd.DataFrame,
    *,
    column: str,
    breakpoint: float,
    left_column: str | None = None,
    right_column: str | None = None,
    drop_original: bool = True,
) -> pd.DataFrame:
    """
    Replace a feature x with two piecewise linear features around a breakpoint:

        left  = min(x, breakpoint)
        right = max(0, x - breakpoint)

    This allows linear models to learn different behavior before and after
    the breakpoint.

    Parameters
    ----------
    df:
        Input DataFrame.
    column:
        Name of the original feature column.
    breakpoint:
        Breakpoint value.
    left_column:
        Optional custom name for the left piece.
    right_column:
        Optional custom name for the right piece.
    drop_original:
        If True, drop the original column after creating the piecewise columns.
    """
    df = df.copy()

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")

    left_name = left_column or f"{column}_le_{breakpoint}"
    right_name = right_column or f"{column}_gt_{breakpoint}"

    x = df[column]
    df[left_name] = np.minimum(x, breakpoint)
    df[right_name] = np.maximum(0.0, x - breakpoint)

    if drop_original:
        df = df.drop(columns=[column])

    return df


# ============================================================
# DATA FILTERING
# ============================================================

def filter_by_feature_range(
    df: pd.DataFrame,
    *,
    column: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> pd.DataFrame:
    """
    Filters rows based on feature range.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found.")

    mask = pd.Series(True, index=df.index)

    if min_value is not None:
        mask &= df[column] >= min_value

    if max_value is not None:
        mask &= df[column] <= max_value

    return df.loc[mask].copy()