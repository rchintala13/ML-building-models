import pytest
from pathlib import Path

from ml179d.usecases.resolver import UsecaseResolver, default_slugify, invert_map

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

def test_to_slug():
    r = make_resolver()
    assert r.to_slug("building_type", "SmallOffice") == "small_office"
    assert r.to_slug("system_type", "MSHP DOAS") == "MSHP_DOAS"
    assert r.to_slug("system_type", "PSZ-AC with electric coil") == "PSZ-AC_with_electric_coil"
    assert r.to_slug("climate_zone", "7A") == "CZ7"

def test_parse_id_with_underscore_separator():
    r = make_resolver(sep="_")
    
    bt, st, cz = r.parse_id("small_office_PSZ-AC_with_electric_coil_CZ7", strict=True)
    assert (bt, st, cz) == ("small_office", "PSZ-AC_with_electric_coil", "CZ7")

def test_yaml_loading(tmp_path: Path):
    yaml_text = """
    usecase:
    sep: "_"
    aliases:
        building_type:
        SmallOffice: small_office
        system_type:
        "PSZ-AC with electric coil": PSZ-AC_with_electric_coil
        MSHP DOAS: MSHP_DOAS
        climate_zone:
        "7A": CZ7
    """
    cfg = tmp_path / "usecase_config.yaml"
    cfg.write_text(yaml_text)

    r = UsecaseResolver.from_yaml(cfg)
    assert r.sep == "_"
    assert r.to_slug("building_type", "SmallOffice") == "small_office"
    bt, st, cz = r.parse_id("small_office_MSHP_DOAS_CZ7", strict=True)
    assert (bt, st, cz) == ("small_office", "MSHP_DOAS", "CZ7")