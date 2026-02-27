from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Tuple, Optional
import yaml

"""
Aliases
--------
- A usecase is defined as building_type + '_' + system_type + '_' + 'CZ_'+ climatezone using aliases
- building_type, system_type, and climate_zone from BEM raw data are of a different form
  than model names, results analysis etc. 
- The resolver maps BEM values to their aliases and vice versa
   Example: SmallOffice -> small_office

UsecaseResolver
---------------
UsecaseResolver stores the mappings from raw -> slug and slug-> raw
    - either provided  by user
    - or read from a yaml file (./configs/usecase_space.yaml)

"""

Kind = Literal["building_type", "system_type", "climate_zone"]


def default_slugify(s: str) -> str:
    """
    Conservative fallback normalization:
    - strip
    - spaces -> underscores
    - lowercase
    - remove non [a-z0-9_]
    """
    s = s.strip().replace(" ", "_").lower()
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

def invert_map(raw_to_slug: Dict[str, str]) -> Dict[str, str]:
    """
    Build slug -> raw map. If two raw values map to the same slug, error out.
    """
    slug_to_raw: Dict[str, str] = {}
    for raw, slug in raw_to_slug.items():
        if slug in slug_to_raw and slug_to_raw[slug] != raw:
            raise ValueError(
                f"Alias collision: '{raw}' and '{slug_to_raw[slug]}' both map to slug '{slug}'."
            )
        slug_to_raw[slug] = raw
    return slug_to_raw


## Class that stores mappings from raw (BEM Fields) -> slugs (aliases) and vice versa
@dataclass(frozen=True)
class UsecaseResolver:

    sep: str
    # forward maps: raw -> slug
    building_type_alias: Dict[str, str]
    system_type_alias: Dict[str, str]
    climate_zone_alias: Dict[str, str]

    # reverse maps: slug -> raw (derived)
    building_type_rev: Dict[str, str]
    system_type_rev: Dict[str, str]
    climate_zone_rev: Dict[str, str]

    @staticmethod
    def from_yaml(path: Path) -> UsecaseResolver:
        cfg = yaml.safe_load(path.read_text())
        uc = cfg.get("usecase", {})
        aliases = uc.get("aliases", {})

        bt = dict(aliases.get("building_type", {}))
        st = dict(aliases.get("system_type", {}))
        cz = dict(aliases.get("climate_zone", {}))

        return UsecaseResolver(
            sep=uc.get("sep", "_"),
            building_type_alias=bt,
            system_type_alias=st,
            climate_zone_alias=cz,
            building_type_rev=invert_map(bt),
            system_type_rev=invert_map(st),
            climate_zone_rev=invert_map(cz),
        )

    def to_slug(self, kind: Kind, raw: str) -> str:
        """
        Convert a raw/canonical value to its slug used in usecase IDs.
        """
        if kind == "building_type":
            return self.building_type_alias.get(raw, default_slugify(raw))
        if kind == "system_type":
            return self.system_type_alias.get(raw, default_slugify(raw))
        if kind == "climate_zone":
            return self.climate_zone_alias.get(raw, default_slugify(raw))
        raise ValueError(f"Unknown kind: {kind}")

    def to_raw(self, kind: Kind, slug: str) -> str:
        """
        Convert a slug back to the raw/canonical value, if it exists in aliases.
        If it doesn't exist, return the slug (or you can choose to error).
        """
        if kind == "building_type":
            return self.building_type_rev.get(slug, slug)
        if kind == "system_type":
            return self.system_type_rev.get(slug, slug)
        if kind == "climate_zone":
            return self.climate_zone_rev.get(slug, slug)
        raise ValueError(f"Unknown kind: {kind}")


    
    def parse_id(self, usecase_id: str, strict: bool = True) -> Tuple[str, str, str]:
        """
        Parse IDs like:
        small_office_VRF_DOAS_CZ5A -> (small_office, VRF_DOAS, CZ5A)
        where sep is '_' and '_' may also appear inside slugs.

        Strategy:
        - match climate_zone as a known suffix slug
        - then match system_type as a known suffix slug
        - remaining prefix must be building_type slug

        If strict=True, all parts must be present in alias VALUE sets.
        """
        sep = self.sep
        if not usecase_id or sep not in usecase_id:
            raise ValueError(f"Invalid usecase_id '{usecase_id}'. Expected separator '{sep}'.")

        bt_slugs = set(self.building_type_alias.values())
        st_slugs = set(self.system_type_alias.values())
        cz_slugs = set(self.climate_zone_alias.values())

        if not (bt_slugs and st_slugs and cz_slugs):
            raise ValueError("Alias maps must be populated to parse usecase_id with '_' separator.")

        def strip_suffix(s: str, suffix: str) -> str | None:
            # expects: <prefix><sep><suffix>
            tail = f"{sep}{suffix}"
            if s.endswith(tail):
                return s[: -len(tail)]
            return None

        # Prefer longest suffix matches to reduce ambiguity
        cz_candidates = sorted(
            [cz for cz in cz_slugs if usecase_id.endswith(f"{sep}{cz}")],
            key=len,
            reverse=True,
        )
        if not cz_candidates:
            raise ValueError(f"Could not parse climate_zone from '{usecase_id}' using known aliases.")
        cz = cz_candidates[0]

        remainder = strip_suffix(usecase_id, cz)
        assert remainder is not None  # by construction

        st_candidates = sorted(
            [st for st in st_slugs if remainder.endswith(f"{sep}{st}")],
            key=len,
            reverse=True,
        )
        if not st_candidates:
            raise ValueError(f"Could not parse system_type from '{usecase_id}' using known aliases.")
        st = st_candidates[0]

        bt_part = strip_suffix(remainder, st)
        if bt_part is None or bt_part == "":
            raise ValueError(f"Could not parse building_type from '{usecase_id}' using known aliases.")
        bt = bt_part

        if strict:
            if bt not in bt_slugs:
                raise ValueError(f"Unknown building_type slug '{bt}' in '{usecase_id}'.")
            if st not in st_slugs:
                raise ValueError(f"Unknown system_type slug '{st}' in '{usecase_id}'.")
            if cz not in cz_slugs:
                raise ValueError(f"Unknown climate_zone slug '{cz}' in '{usecase_id}'.")

        return bt, st, cz