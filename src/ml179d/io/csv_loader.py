from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd

from ml179d.schema.types import FeatureColumnMap, Schema


def _pick_first_existing_column(
    available_columns: Iterable[str],
    candidate_columns: list[str],
) -> Optional[str]:
    """
    Return the first candidate column that exists in available_columns.
    """
    available_set = set(available_columns)
    for col in candidate_columns:
        if col in available_set:
            return col
    return None


def resolve_raw_to_canonical_column_map(
    raw_columns: Iterable[str],
    *,
    schema: Schema,
    scenario: str,
    include_roles: tuple[str, ...] = ("row_id", "id", "feature", "target"),
) -> Dict[str, str]:
    """
    Resolve raw EnergyPlus column names to canonical schema names.

    Parameters
    ----------
    raw_columns:
        Column names present in the CSV.
    schema:
        Loaded schema object.
    scenario:
        'proposed' or 'baseline'
    include_roles:
        Only schema columns with these roles will be considered.

    Returns
    -------
    dict[str, str]
        Mapping:
            raw_column_name -> canonical_column_name
    """
    scenario = scenario.lower().strip()
    if scenario not in {"proposed", "baseline"}:
        raise ValueError(f"Unexpected scenario '{scenario}'. Must be 'proposed' or 'baseline'.")

    raw_to_canonical: Dict[str, str] = {}

    for canonical_name, colspec in schema.columns.items():
        if colspec.role not in include_roles:
            continue

        matched_raw_col: Optional[str] = None

        # Scenario-specific sources take precedence when available
        if colspec.sources_by_scenario is not None:
            scenario_sources = colspec.sources_by_scenario.get(scenario, [])
            matched_raw_col = _pick_first_existing_column(raw_columns, scenario_sources)

        # Fall back to scenario-independent sources
        if matched_raw_col is None and colspec.sources is not None:
            matched_raw_col = _pick_first_existing_column(raw_columns, colspec.sources)

        if matched_raw_col is not None:
            raw_to_canonical[matched_raw_col] = canonical_name
        elif colspec.required:
            raise KeyError(
                f"Required canonical column '{canonical_name}' could not be resolved "
                f"for scenario='{scenario}'."
            )

    return raw_to_canonical


def apply_row_validity(
    df: pd.DataFrame,
    *,
    schema: Schema,
    source: str = "",
) -> tuple[pd.DataFrame, int]:
    """
    Keep only rows that satisfy every schema.row_validity condition.

    Failed simulation rows carry blank features and blank usecase identifiers,
    so they would otherwise flow into training as NaN. Returns the filtered
    frame and the number of rows dropped.
    """
    if not schema.row_validity:
        return df, 0

    mask = pd.Series(True, index=df.index)

    for column, allowed in schema.row_validity.items():
        if column not in df.columns:
            raise KeyError(
                f"{source or 'DataFrame'} is missing row_validity column "
                f"'{column}'. Remove it from schema.yaml or fix the source data."
            )
        mask &= df[column].isin(allowed)

    n_dropped = int((~mask).sum())
    return df.loc[mask].copy(), n_dropped


def load_canonical_batch_dataframe(
    csv_path: Path,
    *,
    schema: Schema,
    scenario: str,
    set_row_index: bool = True,
) -> pd.DataFrame:
    """
    Load a raw batch CSV and return a DataFrame with canonical column names.

    Parameters
    ----------
    csv_path:
        Path to the raw batch CSV.
    schema:
        Loaded schema object.
    scenario:
        'proposed' or 'baseline'
    set_row_index:
        If True, set the first row_id column (e.g. 'name') as the DataFrame index.

    Returns
    -------
    pd.DataFrame
        DataFrame with canonical column names.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df_raw = pd.read_csv(csv_path)

    # Drop failed simulation rows before anything else looks at the data.
    df_raw, n_invalid = apply_row_validity(df_raw, schema=schema, source=csv_path.name)
    if df_raw.empty:
        raise ValueError(
            f"{csv_path.name}: every row was rejected by schema.row_validity."
        )

    raw_to_canonical = resolve_raw_to_canonical_column_map(
        df_raw.columns,
        schema=schema,
        scenario=scenario,
    )

    df = df_raw.rename(columns=raw_to_canonical)

    # Keep only columns that were successfully mapped to canonical names
    canonical_columns = list(raw_to_canonical.values())
    df = df[canonical_columns].copy()

    if set_row_index:
        row_id_cols = schema.row_id_columns()
        if row_id_cols:
            row_id_col = row_id_cols[0]
            if row_id_col in df.columns:
                df = df.set_index(row_id_col, drop=False)

    df.attrs["n_invalid_rows_dropped"] = n_invalid
    return df


def validate_required_canonical_columns(
    df: pd.DataFrame,
    *,
    schema: Schema,
    roles: tuple[str, ...] = ("row_id", "id", "feature", "target"),
) -> None:
    """
    Validate that all required canonical columns for the selected roles are present.
    """
    missing = []
    for canonical_name, colspec in schema.columns.items():
        if colspec.role not in roles:
            continue
        if colspec.required and canonical_name not in df.columns:
            missing.append(canonical_name)

    if missing:
        raise KeyError(
            f"DataFrame is missing required canonical columns: {missing}"
        )


def get_canonical_columns_by_role(
    df: pd.DataFrame,
    *,
    schema: Schema,
    role: str,
) -> list[str]:
    """
    Return canonical columns of a given schema role that are present in the DataFrame.
    """
    return [
        col.name
        for col in schema.by_role(role)
        if col.name in df.columns
    ]