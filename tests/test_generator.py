import pytest
from pathlib import Path

from ml179d.usecases.generator import UsecaseSpace, generate_usecases, generate_usecase_ids, make_validity_rule
from ml179d.usecases.resolver import UsecaseResolver, invert_map


def make_resolver(sep="_"):
    # raw -> slug maps
    bt = {"SmallOffice": "small_office", "RetailStripmall": "retail_stripmall"}
    st = {
        "HP RTU": "HP_RTU",
        "MSHP DOAS": "MSHP_DOAS",
        "PSZ-AC with electric coil": "PSZ-AC_with_electric_coil",
        "PSZ-AC with gas coil": "PSZ-AC_with_gas_coil",
        "PSZ-HP": "PSZ-HP",
        "VRF DOAS": "VRF_DOAS"}
    cz = {
        "1A": "CZ1A",
        "5A": "CZ5A",
        "7A": "CZ7"
    }

    return UsecaseResolver(
        sep=sep,
        building_type_alias=bt,
        system_type_alias=st,
        climate_zone_alias=cz,
        building_type_rev=invert_map(bt),
        system_type_rev=invert_map(st),
        climate_zone_rev=invert_map(cz),
    )


def make_generator():
    r = make_resolver()
    building_types = ['SmallOffice', 'RetailStripmall']
    system_types = ['HP RTU', 'PSZ-AC with electric coil']
    climate_zones = ['1A', '2A', '7A', '8A']

    usecase_space = 