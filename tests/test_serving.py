import json
from pathlib import Path

import pandas as pd
import pytest

from ml179d.cli import main
from ml179d.features.registry import check_base_feature_order
from ml179d.pipeline import stage_dataset, stage_scan
from ml179d.serving import (
    ModelBundle,
    build_feature_frame,
    check_bounds,
    load_bundle,
    predict_savings,
)
from ml179d.train import train_usecase_with_savings

from conftest import USECASE_ID, write_project

"""
The serving contract: a manifest plus two joblib files must reproduce, from
user inputs alone, exactly what training computed.
"""


@pytest.fixture
def trained(tmp_path: Path) -> dict:
    project = write_project(tmp_path)
    code = main(
        [
            "train",
            "--usecase", USECASE_ID,
            "--model-type", "plain_linear",
            "--configs", str(project["root"] / "configs"),
            "--data-root", str(project["root"] / "data"),
            "--output-dir", str(project["root"] / "outputs"),
        ]
    )
    assert code == 0

    project["model_dir"] = (
        project["root"] / "outputs" / "models" / USECASE_ID / "electricity" / "plain_linear"
    )
    return project


# ---------------------------------------------------------------
# base feature ordering
# ---------------------------------------------------------------

def test_order_check_accepts_valid_order():
    check_base_feature_order(["add_roof_area", "add_sa_to_vol_ratio"])


def test_order_check_rejects_producer_after_consumer():
    with pytest.raises(ValueError, match="List 'add_roof_area' first"):
        check_base_feature_order(["add_sa_to_vol_ratio", "add_roof_area"])


def test_order_check_ignores_non_derived_inputs():
    # add_ext_wall_surface_area reads only user-supplied columns
    check_base_feature_order(["add_ext_wall_surface_area", "add_window_area"])


# ---------------------------------------------------------------
# manifest
# ---------------------------------------------------------------

def test_manifest_is_written_next_to_the_models(trained):
    path = trained["model_dir"] / "manifest.json"

    assert path.exists()
    manifest = json.loads(path.read_text())

    assert manifest["usecase_id"] == USECASE_ID
    assert manifest["target_set"] == "electricity"
    assert manifest["savings_definition"] == "baseline - proposed"
    assert set(manifest["models"]) == {"proposed", "baseline"}


def test_manifest_separates_user_inputs_from_derived(trained):
    manifest = json.loads((trained["model_dir"] / "manifest.json").read_text())

    inputs = [e["name"] for e in manifest["user_inputs"]]
    derived = manifest["derived"]

    # roof_area_cal and bldg_vol are computed here, never supplied
    assert "roof_area_cal" in derived
    assert "roof_area_cal" not in inputs
    # the inputs those derivations need must be requested from the user
    assert "gross_floor_area" in inputs
    assert "number_of_floors" in inputs
    assert not set(inputs) & set(derived)


def test_manifest_records_units_and_recipe(trained):
    manifest = json.loads((trained["model_dir"] / "manifest.json").read_text())

    assert manifest["recipe"]["base_features"] == ["add_roof_area", "add_bldg_volume"]
    assert manifest["fitted_features"]
    assert manifest["provenance"]["sklearn"]


def test_manifest_carries_slugs_so_serving_needs_no_yaml(trained):
    """
    build_feature_frame needs the building type for geometry, and the manifest
    must supply it: the calculator has no usecase_space.yaml.
    """
    manifest = json.loads((trained["model_dir"] / "manifest.json").read_text())

    assert manifest["building_type_slug"] == "small_office"
    assert manifest["system_type_slug"] == "PSZ-HP"
    assert manifest["climate_zone_slug"] == "CZ5A"


# ---------------------------------------------------------------
# prediction
# ---------------------------------------------------------------

def user_inputs_from(bundle: ModelBundle, row: pd.Series) -> dict:
    return {name: row[name] for name in bundle.required_inputs}


def test_served_prediction_matches_training(trained, ctx):
    """
    The whole point: replaying the recipe from user inputs must reproduce the
    model's own prediction on the training frame, to floating point.
    """
    bundle = load_bundle(trained["model_dir"])

    index = stage_scan(trained["raw"], ctx=ctx)
    data = stage_dataset(
        index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        split="test",
        target_set="electricity",
        model_type="plain_linear",
    )

    # take a building and feed only its user inputs back through serving
    raw = pd.read_csv(
        trained["raw"] / "batch1_proposed_testing_2024_12_17_18_29_43_est.csv"
    ).set_index("reporting_179_d.name")
    name = data.row_ids[3]

    inputs = {
        "gross_floor_area": raw.loc[name, "reporting_179_d.in_floor_area_m_2"],
        "number_of_floors": raw.loc[name, "reporting_179_d.in_number_of_floors"],
        "aspect_ratio": raw.loc[name, "reporting_179_d.in_ns_to_ew_ratio"],
        "roof_area": raw.loc[name, "reporting_179_d.in_roof_area_m_2"],
    }
    inputs = {k: v for k, v in inputs.items() if k in set(bundle.required_inputs)}

    served = build_feature_frame(bundle, inputs)
    expected = data.X.loc[[name]]

    pd.testing.assert_frame_equal(
        served.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
    )


def test_predict_savings_is_baseline_minus_proposed(trained):
    bundle = load_bundle(trained["model_dir"])
    inputs = {
        "gross_floor_area": 1500.0,
        "number_of_floors": 2.0,
        "aspect_ratio": 1.5,
        "roof_area": 700.0,
    }
    inputs = {k: v for k, v in inputs.items() if k in set(bundle.required_inputs)}

    result = predict_savings(bundle, inputs)

    assert result.savings == pytest.approx(result.baseline - result.proposed)
    assert result.savings > 0
    assert result.usecase_id == USECASE_ID
    assert result.target == "total_electricity_179d"


def test_predict_rejects_missing_input(trained):
    bundle = load_bundle(trained["model_dir"])

    with pytest.raises(KeyError, match="Missing required user input"):
        predict_savings(bundle, {"gross_floor_area": 1500.0})


def test_predict_rejects_unexpected_input(trained):
    bundle = load_bundle(trained["model_dir"])
    inputs = {name: 1.5 for name in bundle.required_inputs}
    inputs["not_a_feature"] = 3

    with pytest.raises(KeyError, match="Unexpected input"):
        predict_savings(bundle, inputs)


def test_out_of_range_input_warns_but_predicts(trained):
    """
    Extrapolation is reported, not refused: the caller decides.
    """
    bundle = load_bundle(trained["model_dir"])

    # the fixture config has no filters, so inject a bound into the manifest
    bundle.manifest["user_inputs"][0]["bounds"] = {"min": 1e9, "max": None}
    name = bundle.manifest["user_inputs"][0]["name"]

    inputs = {n: 1.5 for n in bundle.required_inputs}
    warnings = check_bounds(bundle, inputs)

    assert any(name in w and "below the fitted range" in w for w in warnings)


def test_load_bundle_requires_a_manifest(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No manifest.json"):
        load_bundle(tmp_path)


def test_load_bundle_reports_a_missing_model_file(trained):
    (trained["model_dir"] / "baseline.joblib").unlink()

    with pytest.raises(FileNotFoundError, match="Model file missing"):
        load_bundle(trained["model_dir"])


# ---------------------------------------------------------------
# per-usecase recipe variation
# ---------------------------------------------------------------

def test_manifest_reflects_system_overrides(tmp_path: Path):
    """
    The manifest recipe must resolve exactly as stage_dataset did, including
    system_overrides, or it would describe a model that was never fitted.
    """
    from conftest import MODEL_YAML

    model_yaml = MODEL_YAML.replace(
        "system_overrides: {}",
        "system_overrides:\n"
        "  PSZ-HP:\n"
        "    add_features:\n"
        "      - bldg_vol\n",
    )
    project = write_project(tmp_path, model_yaml=model_yaml)

    code = main(
        [
            "train",
            "--usecase", USECASE_ID,
            "--model-type", "plain_linear",
            "--configs", str(project["root"] / "configs"),
            "--data-root", str(project["root"] / "data"),
            "--output-dir", str(project["root"] / "outputs"),
        ]
    )
    assert code == 0

    directory = (
        project["root"] / "outputs" / "models" / USECASE_ID / "electricity" / "plain_linear"
    )
    manifest = json.loads((directory / "manifest.json").read_text())
    sidecar = json.loads((directory / "proposed.json").read_text())

    # the system override added bldg_vol; the manifest must know about it
    assert "bldg_vol" in manifest["fitted_features"]
    assert manifest["fitted_features"] == sidecar["feature_names"]


def test_manifest_fitted_features_match_the_actual_models(trained):
    """
    The declared contract must equal what each scenario model was fitted on.
    """
    manifest = json.loads((trained["model_dir"] / "manifest.json").read_text())

    for scenario in ("proposed", "baseline"):
        sidecar = json.loads((trained["model_dir"] / f"{scenario}.json").read_text())
        assert manifest["fitted_features"] == sidecar["feature_names"]


def test_manifest_rejects_mismatched_scenario_features(trained):
    from ml179d.serving import build_manifest
    from ml179d.config import DatasetRecipe

    bundle = load_bundle(trained["model_dir"])
    manifest = json.loads((trained["model_dir"] / "manifest.json").read_text())

    import dataclasses
    from ml179d.train import FittedModel

    proposed = FittedModel(
        estimator=bundle.estimators["proposed"], usecase_id=USECASE_ID,
        scenario="proposed", target_set="electricity", model_type="plain_linear",
        feature_names=["a", "b"], target_names=["t"], n_train=10,
    )
    baseline = dataclasses.replace(proposed, scenario="baseline", feature_names=["a"])

    recipe = DatasetRecipe(
        features=("a", "b"), targets=("t",), base_features=(), transforms=(),
        filters=(), target_set="electricity", model_type="plain_linear",
    )

    with pytest.raises(ValueError, match="fitted on different features"):
        build_manifest(
            models={"proposed": proposed, "baseline": baseline},
            recipe=recipe,
            usecase_id=USECASE_ID,
            building_type_slug="small_office",
            system_type_slug="PSZ-HP",
            climate_zone_slug="CZ5A",
        )
