from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from ml179d.schema.types import (
    ColumnSpec,
    FeatureColumnMap,
    Schema,
    UsecaseColumnMap,
)


def load_schema(schema_path: Path) -> Schema:
    """
    Load schema.yaml into a structured Schema object.
    """
    schema_path = Path(schema_path)
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    cfg = yaml.safe_load(schema_path.read_text())
    if "columns" not in cfg:
        raise KeyError("schema.yaml must contain a top-level 'columns' section.")

    raw_columns: Mapping[str, Any] = cfg["columns"]
    columns: Dict[str, ColumnSpec] = {}

    for canonical_name, spec in raw_columns.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Schema entry for '{canonical_name}' must be a mapping.")

        columns[canonical_name] = ColumnSpec(
            name=canonical_name,
            role=str(spec.get("role")),
            required=bool(spec.get("required", False)),
            dtype=spec.get("dtype"),
            unit=spec.get("unit"),
            sources=list(spec.get("sources", [])) or None,
            sources_by_scenario=dict(spec.get("sources_by_scenario", {})) or None,
        )

    raw_validity = cfg.get("row_validity") or {}
    if not isinstance(raw_validity, dict):
        raise ValueError(
            f"'row_validity' must be a mapping of raw column -> allowed values, "
            f"got {type(raw_validity).__name__}."
        )

    row_validity: Dict[str, List[str]] = {}
    for column, allowed in raw_validity.items():
        values = [allowed] if isinstance(allowed, str) else list(allowed or [])
        if not values:
            raise ValueError(
                f"row_validity['{column}'] has no allowed values, which would "
                f"exclude every row."
            )
        row_validity[column] = values

    return Schema(columns=columns, row_validity=row_validity)


def _first_source(col: ColumnSpec) -> str:
    """
    Return the first raw source column from a scenario-independent column.
    """
    if not col.sources:
        raise ValueError(f"Column '{col.name}' has no 'sources' defined.")
    return col.sources[0]


def _first_source_by_scenario(col: ColumnSpec, scenario: str) -> str:
    """
    Return the first raw source column for a scenario-dependent column.
    """
    if not col.sources_by_scenario:
        raise ValueError(f"Column '{col.name}' has no 'sources_by_scenario' defined.")
    if scenario not in col.sources_by_scenario:
        raise KeyError(f"Column '{col.name}' has no sources for scenario '{scenario}'.")
    sources = col.sources_by_scenario[scenario]
    if not sources:
        raise ValueError(f"Column '{col.name}' scenario '{scenario}' has empty source list.")
    return sources[0]


def build_usecase_column_map(schema: Schema) -> UsecaseColumnMap:
    """
    Extract the minimal raw-column map needed by batch_scanner.
    Requires canonical schema columns:
      - building_type
      - climate_zone
      - system_type
    """
    bt = schema.get_column("building_type")
    cz = schema.get_column("climate_zone")
    st = schema.get_column("system_type")

    if st.sources_by_scenario is None:
        raise ValueError(
            "Canonical column 'system_type' must define 'sources_by_scenario' "
            "for proposed and baseline."
        )

    return UsecaseColumnMap(
        building_type_raw=_first_source(bt),
        climate_zone_raw=_first_source(cz),
        system_type_raw_by_scenario={
            "proposed": _first_source_by_scenario(st, "proposed"),
            "baseline": _first_source_by_scenario(st, "baseline"),
        },
    )


def build_feature_column_map(schema: Schema) -> FeatureColumnMap:
    """
    Extract canonical feature/target/row_id mappings for CSV loading and modeling.
    """
    canonical_to_sources: Dict[str, List[str]] = {}
    canonical_to_sources_by_scenario: Dict[str, Dict[str, List[str]]] = {}

    for canonical_name, col in schema.columns.items():
        if col.sources:
            canonical_to_sources[canonical_name] = list(col.sources)

        if col.sources_by_scenario:
            canonical_to_sources_by_scenario[canonical_name] = {
                scenario: list(source_list)
                for scenario, source_list in col.sources_by_scenario.items()
            }

    return FeatureColumnMap(
        canonical_to_sources=canonical_to_sources,
        canonical_to_sources_by_scenario=canonical_to_sources_by_scenario,
        feature_columns=schema.feature_columns(),
        target_columns=schema.target_columns(),
        row_id_columns=schema.row_id_columns(),
    )


def load_usecase_column_map(schema_path: Path) -> UsecaseColumnMap:
    """
    Convenience helper: schema.yaml -> UsecaseColumnMap
    """
    schema = load_schema(schema_path)
    return build_usecase_column_map(schema)


def load_feature_column_map(schema_path: Path) -> FeatureColumnMap:
    """
    Convenience helper: schema.yaml -> FeatureColumnMap
    """
    schema = load_schema(schema_path)
    return build_feature_column_map(schema)