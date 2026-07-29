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
    df = df.copy()

    # safer building type check
    bt = building_type.lower()

    if "small_office" in bt:
        ceil_height = 10
    elif "retail" in bt:
        ceil_height = 17
    else:
        raise ValueError(f"Unknown building type for SA/V ratio: {building_type}")

    df["aspect_ratio"] = df["aspect_ratio"].replace(0, 1.9)

    term1 = np.sqrt(df["aspect_ratio"] / df["roof_area"])
    term2 = np.sqrt(1.0 / (df["aspect_ratio"] * df["roof_area"]))

    df["sa_to_vol_ratio"] = (
        2.0 * (term1 + term2) +
        (1.0 / (ceil_height * df["number_of_floors"]))
    )

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