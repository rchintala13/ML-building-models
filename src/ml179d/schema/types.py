from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass(frozen=True, slots=True)
class ColumnSpec:
    """
    Column definition and properties of the csv file from BEM team
    """

    name: str
    role: str #id, row_id, feature, target
    required: bool = False
    dtype: Optional[str] = None
    unit: Optional[str] = None
    sources: Optional[List[str]] = None 
    sources_by_scenario: Optional[Dict[str, List[str]]] = None # features which are scenario dependent proposed|baseline|baseline_prm

@dataclass(frozen=True)
class Schema:
    """
    Fully loaded schema from schema.yaml

    row_validity:
        raw column name -> allowed values. Only rows matching every entry are
        loaded. This is a whitelist rather than a blacklist so that a new
        failure spelling is excluded by default instead of silently admitted.
    """
    columns: Dict[str, ColumnSpec]
    row_validity: Dict[str, List[str]] = field(default_factory=dict)

    def get_column(self, name: str) -> ColumnSpec:
        if name not in self.columns:
            raise KeyError(f"Column '{name}' not found in schema")
        return self.columns[name]


    def by_role(self, role: str) -> List[ColumnSpec]:
        return [col for col in self.columns.values() if col.role == role]
    
    def feature_columns(self) -> List[str]:
        return [col.name for col in self.by_role("feature")]

    def target_columns(self) -> List[str]:
        return [col.name for col in self.by_role("target")]

    def id_columns(self) -> List[str]:
        return [col.name for col in self.by_role("id")]

    def row_id_columns(self) -> List[str]:
        return [col.name for col in self.by_role("row_id")]

    def categorical_columns(self) -> List[str]:
        """
        Canonical columns declared as strings. These need encoding before any
        estimator sees them.
        """
        return [col.name for col in self.columns.values() if col.dtype == "str"]

    def scenario_feature_columns(self) -> List[str]:
        return [
            col.name
            for col in self.by_role("feature")
            if col.sources_by_scenario is not None
        ]
    
@dataclass(frozen=True, slots=True)
class UsecaseColumnMap:
    """
    Minimal map needed by batch_scanner to identify a usecase.
    Values are raw EnergyPlus column names.
    """
    building_type_raw: str
    climate_zone_raw: str
    system_type_raw_by_scenario: Dict[str, str]


@dataclass(frozen=True, slots=True)
class FeatureColumnMap:
    """
    Map used by CSV loading / modeling code.

    canonical_to_sources:
        scenario-independent raw source names for each canonical column

    canonical_to_sources_by_scenario:
        scenario-dependent raw source names for each canonical column

    feature_columns:
        canonical feature column names

    target_columns:
        canonical target column names

    row_id_columns:
        canonical row identifier columns, e.g. 'name'
    """
    canonical_to_sources: Dict[str, List[str]]
    canonical_to_sources_by_scenario: Dict[str, Dict[str, List[str]]]
    feature_columns: List[str]
    target_columns: List[str]
    row_id_columns: List[str]