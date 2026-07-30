import pytest
from pathlib import Path

from ml179d.usecases.resolver import UsecaseResolver, default_slugify, invert_map

CZ_PATTERN = r"([0-9][A-C])$"


def make_resolver(sep="_", extract=None):
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
        extract=dict(extract or {}),
    )


# ---------------------------------------------------------------
# raw value extraction
# ---------------------------------------------------------------

def test_climate_zone_extracted_from_ashrae_string():
    """
    reporting_179_d.in_weather_climate_zone arrives as 'ASHRAE 169-2013-4A'.
    """
    r = make_resolver(extract={"climate_zone": CZ_PATTERN})

    assert r.normalize("climate_zone", "ASHRAE 169-2013-5A") == "5A"
    assert r.to_slug("climate_zone", "ASHRAE 169-2013-5A") == "CZ5A"
    assert r.to_slug("climate_zone", "ASHRAE 169-2013-7A") == "CZ7"


def test_extraction_is_idempotent():
    """
    An already-extracted value must pass through, so slugs are stable whether
    the caller supplies the full string or the short form.
    """
    r = make_resolver(extract={"climate_zone": CZ_PATTERN})

    assert r.normalize("climate_zone", "5A") == "5A"
    assert r.to_slug("climate_zone", "5A") == r.to_slug("climate_zone", "ASHRAE 169-2013-5A")


def test_extraction_only_applies_to_configured_kinds():
    r = make_resolver(extract={"climate_zone": CZ_PATTERN})

    # no pattern for building_type, so the raw value is untouched
    assert r.normalize("building_type", "SmallOffice") == "SmallOffice"
    assert r.to_slug("building_type", "SmallOffice") == "small_office"


def test_unmatched_extraction_raises():
    """
    Without this, a non-matching value would fall through to default_slugify
    and produce a mangled usecase id instead of an error.
    """
    r = make_resolver(extract={"climate_zone": CZ_PATTERN})

    with pytest.raises(ValueError, match="Could not extract climate_zone"):
        r.normalize("climate_zone", "Nowhere Land")


def test_no_extract_config_leaves_values_alone():
    r = make_resolver()

    assert r.normalize("climate_zone", "ASHRAE 169-2013-5A") == "ASHRAE 169-2013-5A"
    # and this is exactly the mangling the extract block prevents
    assert r.to_slug("climate_zone", "ASHRAE 169-2013-5A") == "ashrae_16920135a"


def test_extract_is_read_from_yaml(tmp_path: Path):
    path = tmp_path / "usecase_space.yaml"
    path.write_text(
        """
usecase:
  sep: "_"
  extract:
    climate_zone: '([0-9][A-C])$'
  aliases:
    building_type:
      SmallOffice: small_office
    system_type:
      PSZ-HP: PSZ-HP
    climate_zone:
      4A: CZ4A
"""
    )

    r = UsecaseResolver.from_yaml(path)

    assert r.extract == {"climate_zone": "([0-9][A-C])$"}
    assert r.to_slug("climate_zone", "ASHRAE 169-2013-4A") == "CZ4A"

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
                PSZ-AC with electric coil: PSZ-AC_with_electric_coil
                MSHP DOAS: MSHP_DOAS
            climate_zone:
                7A: CZ7
    """
    cfg = tmp_path / "usecase_config.yaml"
    cfg.write_text(yaml_text)

    r = UsecaseResolver.from_yaml(cfg)
    assert r.sep == "_"
    assert r.to_slug("building_type", "SmallOffice") == "small_office"
    bt, st, cz = r.parse_id("small_office_MSHP_DOAS_CZ7", strict=True)
    assert (bt, st, cz) == ("small_office", "MSHP_DOAS", "CZ7")