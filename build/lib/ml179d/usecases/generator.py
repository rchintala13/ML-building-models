from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml
from ml179d.usecases.types import Usecase
from ml179d.usecases.protocols import UsecaseResolverLike

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

load_validity_rule
------------------
Builds the validity function from the 'disallow' section of the yaml file
(./configs/usecase_space.yaml). Combinations that survive the filter are the
usecases we actually train models for.
"""
@dataclass(frozen=True)
class UsecaseSpace:
    building_types: Tuple[str, ...]
    system_types: Tuple[str, ...]
    climate_zones: Tuple[str, ...]

    @staticmethod
    def from_yaml(path: Path) -> UsecaseSpace:
        cfg = yaml.safe_load(path.read_text())
        uc = cfg.get("usecase", {})
        aliases = uc.get("aliases", {})

        bts = tuple(dict(aliases.get("building_type", {})).keys())
        sts = tuple(dict(aliases.get("system_type", {})).keys())
        czs = tuple(dict(aliases.get("climate_zone", {})).keys())
        return UsecaseSpace(
            building_types=bts,
            system_types=sts,
            climate_zones=czs,
        )
    

# Hook: return True if the combo should be included
ValidityFn = Callable[[Usecase], bool]


def default_validity(_: Usecase) -> bool:
    return True


def generate_usecases(
    space: UsecaseSpace,
    validity_fn: ValidityFn = default_validity,
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

    return usecases


def generate_usecase_ids(
    space: UsecaseSpace,
    resolver: UsecaseResolverLike,
    validity_fn: ValidityFn = default_validity,
    sort: bool = True,
) -> List[str]:
    ids =  [u.id(resolver) for u in generate_usecases(space, validity_fn=validity_fn)]

    if sort:
        ids.sort()
    return ids


# Fields a disallow constraint may reference (attributes of Usecase)
USECASE_FIELDS: Tuple[str, ...] = ("building_type", "system_type", "climate_zone")

# A single exclusion, e.g. {"building_type": "SmallOffice", "system_type": "VRF DOAS"}
# or {"system_type": "PSZ-AC with gas coil", "climate_zone": ["1A", "2A"]}.
# Values are RAW BEM values, not slugs. An omitted field means "any value".
Constraint = Mapping[str, str | Sequence[str]]


def _normalize_constraint(constraint: Constraint) -> Dict[str, Tuple[str, ...]]:
    """
    Validate one disallow entry and normalize every value to a tuple.
    """
    if not constraint:
        raise ValueError(
            "Empty disallow constraint would exclude every usecase. "
            f"Specify at least one of {USECASE_FIELDS}."
        )

    normalized: Dict[str, Tuple[str, ...]] = {}
    for field, value in constraint.items():
        if field not in USECASE_FIELDS:
            raise ValueError(
                f"Unknown disallow field '{field}'. Expected one of {USECASE_FIELDS}."
            )
        values = (value,) if isinstance(value, str) else tuple(value)
        if not values:
            raise ValueError(f"Disallow field '{field}' has an empty value list.")
        normalized[field] = values

    return normalized


def make_validity_rule(
    disallow: Optional[Sequence[Constraint]] = None,
) -> ValidityFn:
    """
    Build a validity function from a list of disallow constraints.

    A usecase is rejected when it matches ANY constraint. A usecase matches a
    constraint when EVERY field named by that constraint matches; fields the
    constraint omits are treated as wildcards.

        make_validity_rule([{"system_type": "VRF DOAS"}])
            -> blocks VRF DOAS for all building types and climate zones

        make_validity_rule([{"building_type": "SmallOffice",
                             "system_type": "VRF DOAS"}])
            -> blocks that one pair, across all climate zones
    """
    rules = [_normalize_constraint(c) for c in (disallow or [])]

    def _valid(uc: Usecase) -> bool:
        for rule in rules:
            if all(getattr(uc, field) in allowed for field, allowed in rule.items()):
                return False
        return True

    return _valid


def _check_disallow_against_space(
    disallow: Sequence[Constraint],
    space: UsecaseSpace,
) -> None:
    """
    Catch typos: every value referenced by a constraint must exist in the space.
    A misspelled value would otherwise match nothing and silently disable the rule.
    """
    axis_values = {
        "building_type": set(space.building_types),
        "system_type": set(space.system_types),
        "climate_zone": set(space.climate_zones),
    }

    for constraint in disallow:
        for field, values in _normalize_constraint(constraint).items():
            unknown = [v for v in values if v not in axis_values[field]]
            if unknown:
                raise ValueError(
                    f"Disallow rule references {field} value(s) {unknown} "
                    f"that are not in the usecase space. Expected one of "
                    f"{sorted(axis_values[field])}."
                )


def load_validity_rule(
    path: Path,
    space: Optional[UsecaseSpace] = None,
) -> ValidityFn:
    """
    Read the 'usecase.disallow' section of a yaml file into a validity function.

    A missing or empty 'disallow' section yields a rule that keeps every combination,
    matching default_validity.

    space:
        when provided, disallow values are validated against it so that typos
        raise instead of silently matching nothing.
    """
    cfg = yaml.safe_load(Path(path).read_text()) or {}
    uc = cfg.get("usecase", {}) or {}
    disallow: Sequence[Constraint] = uc.get("disallow", []) or []

    if not isinstance(disallow, list):
        raise ValueError(
            "'usecase.disallow' must be a list of mappings, "
            f"got {type(disallow).__name__}."
        )

    if space is not None:
        _check_disallow_against_space(disallow, space)

    return make_validity_rule(disallow)
