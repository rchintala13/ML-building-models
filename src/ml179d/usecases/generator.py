from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import yaml
from ml179d.usecases.types import Usecase

"""
UsecaseSpace
------------
All possible combinations of building type, system type, and climate zones
- User defined
- read from yaml file (./configs/usecase_space.yaml)

generate_usecases
-----------------
Creates list of Usecase objects using the usecase space + filter
filter defined through a validity function

generate_usecase_ids
--------------------
Creates list of usecase strings corresponding to the list of usecase objects
"""
@dataclass(frozen=True)
class UsecaseSpace:
    building_types: Tuple[str, ...]
    system_types: Tuple[str, ...]
    climate_zones: Tuple[str, ...]

    @staticmethod
    def from_yaml(path: Path) -> UsecaseSpace:
        cfg = yaml.safe_load(path.read_text())
        return UsecaseSpace(
            building_types=tuple(cfg.get("building_types", []).keys()),
            system_types=tuple(cfg.get("system_types", []).keys()),
            climate_zones=tuple(cfg.get("climate_zones", []).keys()),
        )
    

# Hook: return True if the combo should be included
ValidityFn = Callable[[Usecase], bool]


def default_validity(_: Usecase) -> bool:
    return True


def generate_usecases(
    space: UsecaseSpace,
    validity_fn: ValidityFn = default_validity,
    sort: bool = True,
) -> List[Usecase]:
    """
    Generate the cartesian product of building_types × system_types × climate_zones,
    optionally filtering by a validity function.
    """
    usecases: List[Usecase] = []
    for bt, st, cz in product(space.building_types, space.system_types, space.climate_zones):
        uc = Usecase(building_type=bt, system_type=st, climate_zone=cz)
        if validity_fn(uc):
            usecases.append(uc)

    if sort:
        usecases.sort(key=lambda u: u.id)
    return usecases


def generate_usecase_ids(
    space: UsecaseSpace,
    validity_fn: ValidityFn = default_validity,
    sort: bool = True,
) -> List[str]:
    return [u.id for u in generate_usecases(space, validity_fn=validity_fn, sort=sort)]


# Example validity rule (optional): block certain system types for certain buildings
def make_validity_rule(
    disallow: Optional[Sequence[Tuple[str, str]]] = None,
) -> ValidityFn:
    """
    disallow: list of (building_type, system_type) pairs to exclude.
    """
    disallow_set = set(disallow or [])

    def _valid(uc: Usecase) -> bool:
        if (uc.building_type, uc.system_type) in disallow_set:
            return False
        return True

    return _valid
