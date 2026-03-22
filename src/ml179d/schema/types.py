from __future__ import annotations

from dataclasses import dataclass
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
    sources_by_scenario: Optional[Dict[str, List[str]]] # features which are scenario dependent proposed|baseline|baseline_prm

@dataclass(frozen=True)
class Schema:
    """
    Fully loaded schema from schema.yaml
    """
    columns: Dict[str, ColumnSpec]

    def get_column(self, name: str) -> ColumnSpec:
        if name not in self.columns:
            raise KeyError(f"Column '{name}' not found in schema")
        
    def by_role(self, role: str) -> List[ColumnSpec]:
        return [col for col in self.columns.values() if col.role == role]