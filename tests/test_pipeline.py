from pathlib import Path

import pandas as pd
import pytest

from ml179d.config import ModelConfig, validate_against_schema
from ml179d.features.registry import (
    FeatureContext,
    TransformSpec,
    apply_base_features,
    apply_transforms,
    get_base_feature,
    get_transform,
)
from ml179d.pipeline import (
    PipelineContext,
    resolve_batch_path,
    stage_catalog,
    stage_dataset,
    stage_scan,
)
from ml179d.schema import load_schema

from conftest import BATCH_FILES, MODEL_YAML, N_ROWS, SCHEMA_YAML, USECASE_ID


# ---------------------------------------------------------------
# stage 1: scan
# ---------------------------------------------------------------

def test_stage_scan_builds_index(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)

    assert len(batch_index) == 4
    assert set(batch_index["usecase_id"]) == {USECASE_ID}
    assert set(batch_index["scenario"]) == {"proposed", "baseline"}
    # 'training'/'testing' in the filename become 'train'/'test'
    assert set(batch_index["split"]) == {"train", "test"}


def test_stage_scan_skips_unrecognized_csv(project, ctx):
    (project["raw"] / "notes_export.csv").write_text("a,b\n1,2\n")

    batch_index = stage_scan(project["raw"], ctx=ctx)

    assert len(batch_index) == 4


def test_stage_scan_strict_raises_on_unrecognized_csv(project, ctx):
    (project["raw"] / "notes_export.csv").write_text("a,b\n1,2\n")

    with pytest.raises(ValueError, match="Unrecognized batch filename"):
        stage_scan(project["raw"], ctx=ctx, strict=True)


def test_stage_scan_uses_cache(project, ctx):
    cache = project["root"] / "data" / "interim" / "batch_index.parquet"

    first = stage_scan(project["raw"], ctx=ctx, cache_path=cache)
    assert cache.exists()

    # deleting the raw files must not matter once cached
    for filename in BATCH_FILES.values():
        (project["raw"] / filename).unlink()

    second = stage_scan(project["raw"], ctx=ctx, cache_path=cache)
    pd.testing.assert_frame_equal(first, second)


# ---------------------------------------------------------------
# stage 2: catalog
# ---------------------------------------------------------------

def test_stage_catalog_complete_usecase(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)
    report = stage_catalog(batch_index)

    assert report.trainable_usecase_ids == [USECASE_ID]
    assert report.incomplete.empty

    record = report.catalog[USECASE_ID]
    assert record.proposed_train == 1
    assert record.baseline_train == 2


def test_stage_catalog_reports_incomplete_usecase(project, ctx):
    (project["raw"] / BATCH_FILES["baseline_test"]).unlink()

    batch_index = stage_scan(project["raw"], ctx=ctx)
    report = stage_catalog(batch_index)

    assert report.catalog == {}
    assert list(report.incomplete["usecase_id"]) == [USECASE_ID]
    assert "baseline_test" in report.incomplete.iloc[0]["missing_slots"]


def test_stage_catalog_compares_against_expected(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)
    expected = ctx.expected_usecase_ids(project["space_path"])

    report = stage_catalog(batch_index, expected_usecase_ids=expected)

    assert expected == [USECASE_ID]
    assert report.not_in_data == []
    assert report.not_expected == []


def test_stage_catalog_flags_usecase_absent_from_data(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)

    report = stage_catalog(
        batch_index,
        expected_usecase_ids=[USECASE_ID, "small_office_PSZ-HP_CZ1A"],
    )

    assert report.not_in_data == ["small_office_PSZ-HP_CZ1A"]


def test_resolve_batch_path(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)

    path = resolve_batch_path(
        batch_index, usecase_id=USECASE_ID, scenario="proposed", split="train"
    )

    assert path.name == BATCH_FILES["proposed_train"]

    with pytest.raises(KeyError, match="No batch file"):
        resolve_batch_path(
            batch_index, usecase_id=USECASE_ID, scenario="proposed", split="nope"
        )


# ---------------------------------------------------------------
# stage 3: dataset
# ---------------------------------------------------------------

def build(project, ctx, model_type: str):
    batch_index = stage_scan(project["raw"], ctx=ctx)
    return stage_dataset(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        split="train",
        target_set="electricity",
        model_type=model_type,
    )


def test_stage_dataset_plain_linear(project, ctx):
    data = build(project, ctx, "plain_linear")

    assert data.feature_names == [
        "aspect_ratio",
        "number_of_floors",
        "gross_floor_area",
        "roof_area",
        "roof_area_cal",
    ]
    assert data.target_names == ["total_electricity_179d"]
    assert len(data) == N_ROWS
    # unmapped raw columns and row_id must not leak into X as a column,
    # but the row_id must survive as the index
    assert "some.unmapped_column" not in data.X.columns
    assert "name" not in data.X.columns
    assert data.X.index.name == "name"
    assert list(data.row_ids) == [f"bldg_{i}" for i in range(N_ROWS)]


def test_stage_dataset_computes_derived_features(project, ctx):
    data = build(project, ctx, "plain_linear")

    # roof_area_cal is calculated; roof_area comes from the CSV. Both survive,
    # and they must not be equal or the '_cal' distinction is pointless.
    expected = data.X["gross_floor_area"] / data.X["number_of_floors"]
    pd.testing.assert_series_equal(
        data.X["roof_area_cal"], expected, check_names=False
    )
    assert not data.X["roof_area"].equals(data.X["roof_area_cal"])


def test_stage_dataset_applies_model_type_transforms(project, ctx):
    data = build(project, ctx, "ridge_poly")

    # add_log_transforms appends, add_piecewise_feature replaces
    assert "roof_area_cal_log" in data.feature_names
    assert "gross_floor_area" not in data.feature_names
    assert "gross_floor_area_le_1200" in data.feature_names
    assert "gross_floor_area_gt_1200" in data.feature_names
    assert data.target_names == ["total_electricity_179d"]


def test_stage_dataset_missing_derived_feature_raises(project, ctx):
    """
    roof_area is selected but add_roof_area is not listed in base_features.
    """
    project["model_path"].write_text(
        MODEL_YAML.replace("  - add_roof_area\n  - add_bldg_volume", "  []")
    )
    ctx2 = PipelineContext.load(
        schema_path=project["schema_path"],
        usecase_space_path=project["space_path"],
        model_config_path=project["model_path"],
    )

    with pytest.raises(KeyError, match="base_features"):
        build(project, ctx2, "plain_linear")


def test_stage_dataset_unknown_slot_raises(project, ctx):
    batch_index = stage_scan(project["raw"], ctx=ctx)

    with pytest.raises(KeyError, match="No batch file"):
        stage_dataset(
            batch_index,
            ctx=ctx,
            usecase_id="does_not_exist",
            scenario="proposed",
            split="train",
            target_set="electricity",
            model_type="plain_linear",
        )


# ---------------------------------------------------------------
# config resolution
# ---------------------------------------------------------------

def test_resolve_applies_system_override(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_YAML.replace(
            "system_overrides: {}",
            "system_overrides:\n"
            "  PSZ-HP:\n"
            "    add_features:\n"
            "      - bldg_vol\n"
            "    drop_features:\n"
            "      - aspect_ratio\n",
        )
    )
    config = ModelConfig.from_yaml(path)

    recipe = config.resolve(
        target_set="electricity", model_type="plain_linear", system_type_slug="PSZ-HP"
    )

    assert "bldg_vol" in recipe.features
    assert "aspect_ratio" not in recipe.features


def test_resolve_ignores_non_matching_system_override(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(MODEL_YAML)
    config = ModelConfig.from_yaml(path)

    recipe = config.resolve(
        target_set="electricity",
        model_type="plain_linear",
        system_type_slug="not_a_real_system",
    )

    assert list(recipe.features) == list(config.base_feature_sets["electricity"])


def test_resolve_rejects_dropping_absent_feature(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_YAML.replace(
            "system_overrides: {}",
            "system_overrides:\n"
            "  PSZ-HP:\n"
            "    drop_features:\n"
            "      - not_a_feature\n",
        )
    )
    config = ModelConfig.from_yaml(path)

    with pytest.raises(ValueError, match="not in the current feature list"):
        config.resolve(
            target_set="electricity",
            model_type="plain_linear",
            system_type_slug="PSZ-HP",
        )


def test_resolve_rejects_unknown_model_type(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(MODEL_YAML)
    config = ModelConfig.from_yaml(path)

    with pytest.raises(KeyError, match="Unknown model_type"):
        config.resolve(target_set="electricity", model_type="random_forest")


def test_usecase_override_appends_transforms(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_YAML.replace(
            "usecase_overrides: {}",
            "usecase_overrides:\n"
            f"  {USECASE_ID}:\n"
            "    add_features:\n"
            "      - bldg_vol\n"
            "    transforms:\n"
            "      - name: add_log_transforms\n"
            "        params:\n"
            "          columns:\n"
            "            - bldg_vol\n",
        )
    )
    config = ModelConfig.from_yaml(path)

    recipe = config.resolve(
        target_set="electricity", model_type="ridge_poly", usecase_id=USECASE_ID
    )

    assert "bldg_vol" in recipe.features
    # usecase transforms append to the model-type transforms, they do not replace
    assert [t.name for t in recipe.transforms] == [
        "add_log_transforms",
        "add_piecewise_feature",
        "add_log_transforms",
    ]


def test_usecase_override_rejects_legacy_filters(tmp_path: Path):
    """
    Filters moved to the top-level 'filters' block; the old spelling must fail
    loudly rather than be silently ignored.
    """
    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_YAML.replace(
            "usecase_overrides: {}",
            "usecase_overrides:\n"
            f"  {USECASE_ID}:\n"
            "    filters:\n"
            "      - column: gross_floor_area\n"
            "        min_value: 400\n",
        )
    )
    config = ModelConfig.from_yaml(path)

    with pytest.raises(ValueError, match="Filters now live in the top-level"):
        config.resolve(
            target_set="electricity", model_type="ridge_poly", usecase_id=USECASE_ID
        )


def test_from_yaml_rejects_non_empty_list_usecase_overrides(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(
        MODEL_YAML.replace("usecase_overrides: {}", "usecase_overrides:\n  - a\n")
    )

    with pytest.raises(ValueError, match="must be a mapping"):
        ModelConfig.from_yaml(path)


def test_validate_against_schema_catches_typo(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(MODEL_YAML.replace("    - aspect_ratio", "    - aspct_ratio"))

    with pytest.raises(ValueError, match="aspct_ratio"):
        validate_against_schema(
            ModelConfig.from_yaml(model_path), load_schema(schema_path)
        )


def test_validate_against_schema_catches_unknown_base_feature(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML)
    model_path = tmp_path / "model.yaml"
    model_path.write_text(MODEL_YAML.replace("  - add_roof_area", "  - add_roof_are"))

    with pytest.raises(ValueError, match="unknown function"):
        validate_against_schema(
            ModelConfig.from_yaml(model_path), load_schema(schema_path)
        )


# ---------------------------------------------------------------
# registry
# ---------------------------------------------------------------

def test_registry_rejects_unknown_names():
    with pytest.raises(KeyError, match="Unknown base feature"):
        get_base_feature("add_nothing")

    with pytest.raises(KeyError, match="Unknown transform"):
        get_transform("add_nothing")


CONTEXT = FeatureContext(USECASE_ID, "small_office", "PSZ-HP", "CZ5A")


def base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gross_floor_area": [1000.0],
            "number_of_floors": [2.0],
            "aspect_ratio": [1.5],
            "roof_area": [450.0],
        }
    )


def test_add_roof_area_does_not_overwrite_simulated_roof_area():
    """
    The calculated value lands in roof_area_cal so the simulated column survives.
    """
    out = apply_base_features(base_df(), ["add_roof_area"], CONTEXT)

    assert out["roof_area_cal"].iloc[0] == 500.0
    assert out["roof_area"].iloc[0] == 450.0


def test_bldg_volume_reads_simulated_roof_area():
    out = apply_base_features(base_df(), ["add_bldg_volume"], CONTEXT)

    # 450 * 2, i.e. the CSV value -- not gross_floor_area
    assert out["bldg_vol"].iloc[0] == 900.0

    # and it fails outright when the simulated column is absent
    with pytest.raises(KeyError):
        apply_base_features(
            base_df().drop(columns=["roof_area"]), ["add_bldg_volume"], CONTEXT
        )


def test_sa_to_vol_ratio_needs_the_slug_not_the_raw_value():
    """
    add_sa_to_vol_ratio matches substrings of the lowered building type, so
    'SmallOffice' would not match. FeatureContext carries the slug for this.
    """
    steps = ["add_roof_area", "add_sa_to_vol_ratio"]

    ok = apply_base_features(base_df(), steps, CONTEXT)
    assert "sa_to_vol_ratio" in ok.columns

    with pytest.raises(ValueError, match="Unknown building type"):
        apply_base_features(
            base_df(),
            steps,
            FeatureContext(USECASE_ID, "SmallOffice", "PSZ-HP", "CZ5A"),
        )


def test_sa_to_vol_ratio_uses_the_calculated_roof_area():
    """
    The web calculator has no simulated roof_area, so this feature must derive
    from roof_area_cal and fail loudly if add_roof_area has not run.
    """
    with pytest.raises(KeyError, match="needs 'roof_area_cal'"):
        apply_base_features(base_df(), ["add_sa_to_vol_ratio"], CONTEXT)

    # a misleading simulated roof_area must not be picked up instead
    df = base_df()
    df["roof_area"] = 99999.0
    out = apply_base_features(df, ["add_roof_area", "add_sa_to_vol_ratio"], CONTEXT)

    expected_footprint = df["gross_floor_area"] / df["number_of_floors"]
    assert out["roof_area_cal"].iloc[0] == expected_footprint.iloc[0]
    # value follows roof_area_cal (500), not roof_area (99999)
    assert out["sa_to_vol_ratio"].iloc[0] > 0.1


def test_apply_transforms_runs_in_order():
    df = pd.DataFrame({"gross_floor_area": [800.0, 2000.0]})

    out = apply_transforms(
        df,
        [
            TransformSpec("add_log_transforms", {"columns": ["gross_floor_area"]}),
            TransformSpec(
                "add_piecewise_feature",
                {"column": "gross_floor_area", "breakpoint": 1200, "drop_original": True},
            ),
        ],
    )

    assert "gross_floor_area_log" in out.columns
    assert "gross_floor_area" not in out.columns
    assert list(out["gross_floor_area_le_1200"]) == [800.0, 1200.0]
    assert list(out["gross_floor_area_gt_1200"]) == [0.0, 800.0]


# ---------------------------------------------------------------
# calculated envelope areas (servable counterparts)
# ---------------------------------------------------------------

def test_calculated_envelope_areas_split_the_gross_wall():
    """
    ext_wall_surface_area_cal + window_area_cal must reconstitute the gross
    wall area, since one is (1-WWR) and the other WWR of the same quantity.
    """
    df = pd.DataFrame(
        {
            "gross_floor_area": [1000.0, 2400.0],
            "number_of_floors": [1.0, 2.0],
            "aspect_ratio": [1.5, 2.0],
            "window_wall_ratio": [0.3, 0.4],
        }
    )
    ctx = FeatureContext("u", "small_office", "PSZ-HP", "CZ5A")

    out = apply_base_features(
        df, ["add_ext_wall_surface_area", "add_window_area"], ctx
    )

    gross = out["ext_wall_surface_area_cal"] + out["window_area_cal"]
    ratio = out["window_area_cal"] / gross

    assert (out["ext_wall_surface_area_cal"] > 0).all()
    pd.testing.assert_series_equal(
        ratio, df["window_wall_ratio"], check_names=False
    )


def test_calculated_envelope_areas_depend_on_building_type():
    """
    Retail has a taller floor-to-floor height, so the same floor plan yields
    more wall area.
    """
    df = pd.DataFrame(
        {
            "gross_floor_area": [1000.0],
            "number_of_floors": [1.0],
            "aspect_ratio": [1.5],
            "window_wall_ratio": [0.3],
        }
    )

    office = apply_base_features(
        df, ["add_ext_wall_surface_area"],
        FeatureContext("u", "small_office", "PSZ-HP", "CZ5A"),
    )
    retail = apply_base_features(
        df, ["add_ext_wall_surface_area"],
        FeatureContext("u", "retail_stripmall", "HP_RTU", "CZ5A"),
    )

    assert (
        retail["ext_wall_surface_area_cal"].iloc[0]
        > office["ext_wall_surface_area_cal"].iloc[0]
    )


def test_calculated_envelope_area_rejects_unknown_building_type():
    df = pd.DataFrame(
        {
            "gross_floor_area": [1000.0],
            "number_of_floors": [1.0],
            "aspect_ratio": [1.5],
            "window_wall_ratio": [0.3],
        }
    )

    with pytest.raises(ValueError, match="Unknown building type for floor height"):
        apply_base_features(
            df, ["add_ext_wall_surface_area"],
            FeatureContext("u", "warehouse", "HP_RTU", "CZ5A"),
        )


def test_window_area_is_order_independent():
    """
    add_window_area must not depend on add_ext_wall_surface_area having run.
    """
    df = pd.DataFrame(
        {
            "gross_floor_area": [1500.0],
            "number_of_floors": [2.0],
            "aspect_ratio": [1.8],
            "window_wall_ratio": [0.35],
        }
    )
    ctx = FeatureContext("u", "retail_stripmall", "HP_RTU", "CZ5A")

    alone = apply_base_features(df, ["add_window_area"], ctx)
    after = apply_base_features(
        df, ["add_ext_wall_surface_area", "add_window_area"], ctx
    )

    assert alone["window_area_cal"].iloc[0] == after["window_area_cal"].iloc[0]


# ---------------------------------------------------------------
# erv indicator
# ---------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (0.0, 0.0),      # proposed encoding for "no ERV"
        (999.0, 0.0),    # baseline sentinel for the same thing
        (0.75, 1.0),
        (0.82, 1.0),
        (None, 0.0),     # missing treated as absent
    ],
)
def test_erv_indicator_binarizes(raw, expected):
    df = pd.DataFrame({"erv_sensible_cooling": [raw]})
    ctx = FeatureContext("u", "small_office", "PSZ-HP", "CZ5A")

    out = apply_base_features(df, ["add_erv_indicator"], ctx)

    assert out["erv_present"].iloc[0] == expected


def test_erv_indicator_makes_scenarios_agree():
    """
    The whole point: the proposed encoding (0.0) and the baseline sentinel
    (999.0) must collapse to the same feature value, or a baseline model fitted
    on 999.0 explodes when served a proposed 0.0.
    """
    ctx = FeatureContext("u", "retail_stripmall", "HP_RTU", "CZ1A")

    proposed = apply_base_features(
        pd.DataFrame({"erv_sensible_cooling": [0.0]}), ["add_erv_indicator"], ctx
    )
    baseline = apply_base_features(
        pd.DataFrame({"erv_sensible_cooling": [999.0]}), ["add_erv_indicator"], ctx
    )

    assert proposed["erv_present"].iloc[0] == baseline["erv_present"].iloc[0] == 0.0


def test_erv_indicator_keeps_the_source_column():
    df = pd.DataFrame({"erv_sensible_cooling": [0.75]})
    ctx = FeatureContext("u", "small_office", "PSZ-HP", "CZ5A")

    out = apply_base_features(df, ["add_erv_indicator"], ctx)

    assert out["erv_sensible_cooling"].iloc[0] == 0.75
    assert out["erv_present"].iloc[0] == 1.0
