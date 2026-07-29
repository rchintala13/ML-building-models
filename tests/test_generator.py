import pytest
from pathlib import Path
from typing import  Tuple
from itertools import product
import yaml

from ml179d.usecases.generator import (
    UsecaseSpace,
    generate_usecases,
    generate_usecase_ids,
    load_validity_rule,
    make_validity_rule,
)
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


# ---------------------------------------------------------------
# validity rules
# ---------------------------------------------------------------

def full_space() -> UsecaseSpace:
    return UsecaseSpace(
        building_types=tuple(BT.keys()),
        system_types=tuple(ST.keys()),
        climate_zones=tuple(CZ.keys()),
    )


def test_validity_rule_blocks_single_pair():
    """
    A two-field constraint blocks that pair across every climate zone.
    """
    space = full_space()
    validity_fn = make_validity_rule(
        [{"building_type": "SmallOffice", "system_type": "VRF DOAS"}]
    )

    usecases = generate_usecases(space, validity_fn=validity_fn)

    assert not any(
        u.building_type == "SmallOffice" and u.system_type == "VRF DOAS"
        for u in usecases
    )
    # only the 3 climate zones of that one pair are removed
    assert len(usecases) == len(BT) * len(ST) * len(CZ) - len(CZ)
    # other building types keep VRF DOAS
    assert any(
        u.building_type == "RetailStripmall" and u.system_type == "VRF DOAS"
        for u in usecases
    )


def test_validity_rule_omitted_field_is_wildcard():
    """
    A single-field constraint blocks that value across all other axes.
    """
    space = full_space()
    validity_fn = make_validity_rule([{"system_type": "VRF DOAS"}])

    usecases = generate_usecases(space, validity_fn=validity_fn)

    assert not any(u.system_type == "VRF DOAS" for u in usecases)
    assert len(usecases) == len(BT) * (len(ST) - 1) * len(CZ)


def test_validity_rule_accepts_value_list():
    """
    A list value matches any of the listed values.
    """
    space = full_space()
    validity_fn = make_validity_rule(
        [{"system_type": "PSZ-HP", "climate_zone": ["1A", "5A"]}]
    )

    usecases = generate_usecases(space, validity_fn=validity_fn)
    blocked = [u for u in usecases if u.system_type == "PSZ-HP"]

    # PSZ-HP survives only in the one climate zone not listed
    assert {u.climate_zone for u in blocked} == {"7A"}
    assert len(usecases) == len(BT) * len(ST) * len(CZ) - len(BT) * 2


def test_validity_rule_multiple_constraints_are_or_ed():
    space = full_space()
    validity_fn = make_validity_rule(
        [
            {"system_type": "VRF DOAS"},
            {"system_type": "PSZ-HP"},
        ]
    )

    usecases = generate_usecases(space, validity_fn=validity_fn)

    assert not any(u.system_type in {"VRF DOAS", "PSZ-HP"} for u in usecases)
    assert len(usecases) == len(BT) * (len(ST) - 2) * len(CZ)


def test_validity_rule_empty_disallow_keeps_everything():
    space = full_space()

    assert len(generate_usecases(space, validity_fn=make_validity_rule())) == (
        len(BT) * len(ST) * len(CZ)
    )
    assert len(generate_usecases(space, validity_fn=make_validity_rule([]))) == (
        len(BT) * len(ST) * len(CZ)
    )


def test_validity_rule_rejects_unknown_field():
    with pytest.raises(ValueError, match="Unknown disallow field"):
        make_validity_rule([{"bldg_type": "SmallOffice"}])


def test_validity_rule_rejects_empty_constraint():
    with pytest.raises(ValueError, match="would exclude every usecase"):
        make_validity_rule([{}])


def test_validity_rule_rejects_empty_value_list():
    with pytest.raises(ValueError, match="empty value list"):
        make_validity_rule([{"system_type": []}])


# ---------------------------------------------------------------
# yaml-driven validity rules
# ---------------------------------------------------------------

DISALLOW_YAML = """
usecase:
    sep: "_"
    aliases:
        building_type:
            SmallOffice: small_office
            RetailStripmall: retail_stripmall
        system_type:
            PSZ-HP: PSZ-HP
            VRF DOAS: VRF_DOAS
        climate_zone:
            1A: CZ1A
            7A: CZ7
    disallow:
        - building_type: SmallOffice
          system_type: VRF DOAS
        - system_type: PSZ-HP
          climate_zone: [1A]
"""


def write_cfg(tmp_path: Path, text: str) -> Path:
    cfg_path = tmp_path / "usecase_space.yaml"
    cfg_path.write_text(text)
    return cfg_path


def test_load_validity_rule_from_yaml(tmp_path: Path):
    cfg_path = write_cfg(tmp_path, DISALLOW_YAML)

    space = UsecaseSpace.from_yaml(cfg_path)
    validity_fn = load_validity_rule(cfg_path, space=space)
    usecases = generate_usecases(space, validity_fn=validity_fn)

    pairs = {(u.building_type, u.system_type, u.climate_zone) for u in usecases}

    # 2 BT x 2 ST x 2 CZ = 8, minus SmallOffice/VRF DOAS (2 zones)
    # and minus PSZ-HP in 1A (2 building types) = 4
    assert len(usecases) == 4
    assert ("SmallOffice", "VRF DOAS", "1A") not in pairs
    assert ("SmallOffice", "VRF DOAS", "7A") not in pairs
    assert ("SmallOffice", "PSZ-HP", "1A") not in pairs
    assert ("RetailStripmall", "PSZ-HP", "1A") not in pairs
    assert ("RetailStripmall", "VRF DOAS", "1A") in pairs
    assert ("SmallOffice", "PSZ-HP", "7A") in pairs


def test_load_validity_rule_missing_section_keeps_everything(tmp_path: Path):
    """
    A yaml file with no 'disallow' section behaves like default_validity.
    """
    cfg_path = write_cfg(
        tmp_path,
        """
usecase:
    sep: "_"
    aliases:
        building_type:
            SmallOffice: small_office
        system_type:
            PSZ-HP: PSZ-HP
        climate_zone:
            1A: CZ1A
            7A: CZ7
""",
    )

    space = UsecaseSpace.from_yaml(cfg_path)
    validity_fn = load_validity_rule(cfg_path, space=space)

    assert len(generate_usecases(space, validity_fn=validity_fn)) == 2


def test_load_validity_rule_catches_typo_against_space(tmp_path: Path):
    """
    A misspelled value would silently match nothing, so it must raise.
    """
    cfg_path = write_cfg(
        tmp_path,
        DISALLOW_YAML.replace("- building_type: SmallOffice", "- building_type: SmallOfice"),
    )

    space = UsecaseSpace.from_yaml(cfg_path)

    with pytest.raises(ValueError, match="not in the usecase space"):
        load_validity_rule(cfg_path, space=space)

    # without a space to check against, it loads but matches nothing
    validity_fn = load_validity_rule(cfg_path)
    assert len(generate_usecases(space, validity_fn=validity_fn)) == 6


def test_load_validity_rule_rejects_non_list_disallow(tmp_path: Path):
    cfg_path = write_cfg(
        tmp_path,
        """
usecase:
    aliases:
        building_type:
            SmallOffice: small_office
    disallow:
        building_type: SmallOffice
""",
    )

    with pytest.raises(ValueError, match="must be a list of mappings"):
        load_validity_rule(cfg_path)


def test_generate_usecase_ids_respects_validity_fn(tmp_path: Path):
    """
    The filter reaches the id list, not just the Usecase objects.
    """
    cfg_path = write_cfg(tmp_path, DISALLOW_YAML)

    r = UsecaseResolver.from_yaml(cfg_path)
    space = UsecaseSpace.from_yaml(cfg_path)
    validity_fn = load_validity_rule(cfg_path, space=space)

    ids = generate_usecase_ids(space=space, resolver=r, validity_fn=validity_fn)

    assert ids == sorted(ids)
    assert len(ids) == 4
    assert "small_office_VRF_DOAS_CZ7" not in ids
    assert "small_office_PSZ-HP_CZ7" in ids


def test_project_usecase_space_yaml_loads():
    """
    The checked-in config must stay loadable, including its disallow section.
    """
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "usecase_space.yaml"

    space = UsecaseSpace.from_yaml(cfg_path)
    validity_fn = load_validity_rule(cfg_path, space=space)
    usecases = generate_usecases(space, validity_fn=validity_fn)

    assert len(space.building_types) == 2
    assert len(space.system_types) == 6
    assert len(space.climate_zones) == 16
    assert len(usecases) <= 192


