from __future__ import annotations
from typing import Protocol, Literal

Kind = Literal["building_type", "system_type", "climate_zone"]

class UsecaseResolverLike(Protocol):
    sep: str
    def to_slug(self, kind: Kind, raw: str) -> str: ...
