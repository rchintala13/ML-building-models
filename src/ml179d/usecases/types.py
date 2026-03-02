from dataclasses import dataclass
from typing import Protocol

from ml179d.usecases.protocols import UsecaseResolverLike, Kind

"""
Stores the definition of what a usecase is
"""


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