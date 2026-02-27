from dataclasses import dataclass
from typing import Protocol

"""
Stores the definition of what a usecase is
"""

class UsecaseResolverLike(Protocol):
    sep: str
    def to_slug(self, kind: str, raw: str) -> str: ...


@dataclass(frozen=True, slots=True)
class Usecase:
    building_type: str
    system_type: str
    climate_zone: str

    def id(self, resolver: UsecaseResolverLike) -> str:
        return resolver.sep.join([
            resolver.to_slug("building_type", self.building_type),
            resolver.to_slug("system_type", self.system_type),
            resolver.to_slug("climate_zone", self.climate_zone),
        ])