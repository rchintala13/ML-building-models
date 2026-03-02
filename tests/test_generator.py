import pytest
from pathlib import Path
from typing import  Tuple
from itertools import product
import yaml

from ml179d.usecases.generator import UsecaseSpace, generate_usecases, generate_usecase_ids, make_validity_rule
from ml179d.usecases.resolver import UsecaseResolver, invert_map
from ml179d.usecases.types import Usecase
from ml179d.usecases.protocols import UsecaseResolverLike


 # raw -> slug maps
BT = {"SmallOffice": "small_office", "RetailStripmall": "retail_stripmall"}
ST = {
    "HP RTU": "HP_RTU",
    "MSHP DOAS": "MSHP_DOAS",
    "PSZ-AC with electric coil": "PSZ-AC_with_electric_coil",
    "PSZ-AC with gas coil": "PSZ-AC_with_gas_coil",
    "PSZ-HP": "PSZ-HP",
    "VRF DOAS": "VRF_DOAS"}
CZ = {
    "1A": "CZ1A",
    "5A": "CZ5A",
    "7A": "CZ7"
}

def make_resolver(sep="_"):
   
    return UsecaseResolver(
        sep=sep,
        building_type_alias=BT,
        system_type_alias=ST,
        climate_zone_alias=CZ,
        building_type_rev=invert_map(BT),
        system_type_rev=invert_map(ST),
        climate_zone_rev=invert_map(CZ),
    )


def make_generator(
    building_types: Tuple[str, ...],
    system_types: Tuple[str, ...],
    climate_zones: Tuple[str, ...],
    resolver: UsecaseResolverLike
):

    usecase_space = UsecaseSpace(
        building_types=building_types,
        system_types=system_types,
        climate_zones=climate_zones
    )
    usecase_ids = generate_usecase_ids(
        space = usecase_space,
        resolver = resolver,
        sort = True
    )

    return usecase_ids


def test_usecase():
    r = make_resolver()
    usecase_ = Usecase(
        building_type="SmallOffice",
        system_type= "PSZ-AC with electric coil",
        climate_zone="7A"
    )

    assert usecase_.id(r) == 'small_office_PSZ-AC_with_electric_coil_CZ7'

def test_generate_usecase_ids_structure():
    """
    test for structure of usecase_ids
    """
    r = make_resolver()

    building_types = tuple(BT.keys())
    system_types = tuple(ST.keys())
    climate_zones = tuple(CZ.keys())

    usecase_ids = make_generator(
        building_types=building_types,
        system_types=system_types,
        climate_zones=climate_zones,
        resolver = r
    )
    expected_count = len(BT) * len(ST) * len(CZ)

    # check if usecase_ids is a list
    assert isinstance(usecase_ids, list)
    # check if each instance of usecase_ids is a string
    assert all(isinstance(x, str) for x in usecase_ids)
    # check if length of usecase_ids is equal to total possible combinations
    assert len(usecase_ids) == expected_count
    # Uniqueness 
    assert len(set(usecase_ids)) == expected_count


@pytest.mark.parametrize(
    "bt, st, cz",
    product(BT.keys(), ST.keys(), CZ.keys()),
)
def test_generate_usecase_ids_contains_all(bt, st, cz):
    """
    tests for all possible combinations of bt, st, and cz.
    Works only when no validity fn is provided to make_generator
    """
    r = make_resolver()

    usecase_ids = make_generator(
        building_types=tuple(BT.keys()),
        system_types=tuple(ST.keys()),
        climate_zones=tuple(CZ.keys()),
        resolver=r,
    )

    expected_id = Usecase(bt, st, cz).id(r)

    assert expected_id in usecase_ids


def test_yaml_loading(tmp_path: Path):
    yaml_text = """
    usecase:
        sep: "_"
        aliases:
            building_type:
                SmallOffice: small_office
            system_type:
                PSZ-AC with electric coil: PSZ-AC_with_electric_coil
                MSHP DOAS: MSHP_DOAS
            climate_zone:
                7A: CZ7
    """
    cfg_path = tmp_path / "usecase_config.yaml"
    cfg_path.write_text(yaml_text)
    cfg = yaml.safe_load(cfg_path.read_text())
    uc = cfg.get("usecase", {})
    aliases = uc.get("aliases", {})
    building_types = aliases.get("building_type", [])
    system_types = aliases.get("system_type", [])
    climate_zones = aliases.get("climate_zone", [])

    expected_count = len(building_types) * len(system_types) * len(climate_zones)

    r = UsecaseResolver.from_yaml(cfg_path)
    usecase_space = UsecaseSpace.from_yaml(cfg_path)

    usecase_ids = generate_usecase_ids(
        space = usecase_space,
        resolver = r,
        sort = True
    )
    # check if usecase_ids is a list
    assert isinstance(usecase_ids, list)
    # check if each instance of usecase_ids is a string
    assert all(isinstance(x, str) for x in usecase_ids)
    # check if length of usecase_ids is equal to total possible combinations
    assert len(usecase_ids) == expected_count
    # Uniqueness
    assert len(set(usecase_ids)) == expected_count
    

