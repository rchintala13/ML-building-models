from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml179d.pipeline import PipelineContext, stage_scan

"""
A miniature project on disk: schema, usecase space, model config and four
batch CSVs for a single usecase. Shared by the pipeline and training tests.

Targets are exact linear functions of the features so that a linear model can
recover them, which makes metric and savings assertions deterministic.
"""

SCHEMA_YAML = """
columns:
  name:
    role: row_id
    required: true
    dtype: str
    sources:
      - reporting_179_d.name
  building_type:
    role: id
    required: true
    dtype: str
    sources:
      - reporting_179_d.in_primary_bldg_type
  climate_zone:
    role: id
    required: true
    dtype: str
    sources:
      - reporting_179_d.in_weather_climate_zone
  system_type:
    role: id
    required: true
    dtype: str
    sources_by_scenario:
      proposed:
        - support.in_hvac_system_type_proposed
      baseline:
        - support.in_hvac_system_type_proposed
  gross_floor_area:
    role: feature
    required: true
    dtype: float
    sources:
      - reporting_179_d.in_floor_area_m_2
  number_of_floors:
    role: feature
    required: true
    dtype: float
    sources:
      - reporting_179_d.in_number_of_floors
  aspect_ratio:
    role: feature
    required: true
    dtype: float
    sources:
      - reporting_179_d.in_ns_to_ew_ratio
  roof_area:
    role: feature
    required: true
    dtype: float
    sources:
      - reporting_179_d.in_roof_area_m_2
  total_electricity_179d:
    role: target
    required: true
    dtype: float
    sources:
      - reporting_179_d.out_electricity
"""

USECASE_SPACE_YAML = """
usecase:
  sep: "_"
  aliases:
    building_type:
      SmallOffice: small_office
    system_type:
      PSZ-HP: PSZ-HP
    climate_zone:
      5A: CZ5A
  disallow: []
"""

# roof_area is a simulated schema column; roof_area_cal and bldg_vol exist only
# once the base feature functions run.
MODEL_YAML = """
base_features:
  - add_roof_area
  - add_bldg_volume

target_sets:
  electricity:
    - total_electricity_179d

base_feature_sets:
  electricity:
    - aspect_ratio
    - number_of_floors
    - gross_floor_area
    - roof_area
    - roof_area_cal

system_overrides: {}

estimators:
  plain_linear:
    kind: linear
    params: {}
  ridge_poly:
    kind: ridge_poly
    params:
      degree: 2
      alpha: 0.001

model_type_overrides:
  plain_linear:
    transforms: []
  ridge_poly:
    transforms:
      - name: add_log_transforms
        params:
          columns:
            - roof_area_cal
      - name: add_piecewise_feature
        params:
          column: gross_floor_area
          breakpoint: 1200
          drop_original: true

usecase_overrides: {}
"""

USECASE_ID = "small_office_PSZ-HP_CZ5A"

BATCH_FILES = {
    "proposed_train": "batch1_proposed_training_2024_12_17_18_29_43_est.csv",
    "proposed_test": "batch1_proposed_testing_2024_12_17_18_29_43_est.csv",
    "baseline_train": "batch2_baseline_training_2024_12_17_18_29_43_est.csv",
    "baseline_test": "batch2_baseline_testing_2024_12_17_18_29_43_est.csv",
}

# electricity = intercept + a * gross_floor_area + b * roof_area, per scenario.
# Baseline uses more energy, so savings = baseline - proposed > 0.
ENERGY_COEFFS = {
    "proposed": {"intercept": 100.0, "gfa": 1.5, "roof": 0.5},
    "baseline": {"intercept": 150.0, "gfa": 1.9, "roof": 0.7},
}

N_ROWS = 40


def make_raw_frame(
    *,
    scenario: str,
    n_rows: int = N_ROWS,
    seed: int = 0,
    name_prefix: str = "bldg",
    name_offset: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    gross_floor_area = 500.0 + 60.0 * np.arange(n_rows)
    number_of_floors = 1.0 + (np.arange(n_rows) % 3)
    aspect_ratio = 1.2 + 0.05 * (np.arange(n_rows) % 7)
    roof_area = gross_floor_area / number_of_floors + rng.normal(0, 5, n_rows)

    c = ENERGY_COEFFS[scenario]
    electricity = c["intercept"] + c["gfa"] * gross_floor_area + c["roof"] * roof_area

    return pd.DataFrame(
        {
            "reporting_179_d.name": [
                f"{name_prefix}_{i + name_offset}" for i in range(n_rows)
            ],
            "reporting_179_d.in_primary_bldg_type": ["SmallOffice"] * n_rows,
            "reporting_179_d.in_weather_climate_zone": ["5A"] * n_rows,
            "support.in_hvac_system_type_proposed": ["PSZ-HP"] * n_rows,
            "reporting_179_d.in_floor_area_m_2": gross_floor_area,
            "reporting_179_d.in_number_of_floors": number_of_floors,
            "reporting_179_d.in_ns_to_ew_ratio": aspect_ratio,
            "reporting_179_d.in_roof_area_m_2": roof_area,
            "reporting_179_d.out_electricity": electricity,
            "some.unmapped_column": ["ignored"] * n_rows,
        }
    )


def write_batch_csv(path: Path, **kwargs) -> None:
    make_raw_frame(**kwargs).to_csv(path, index=False)


def write_project(root: Path, *, model_yaml: str = MODEL_YAML) -> dict:
    configs = root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    (configs / "schema.yaml").write_text(SCHEMA_YAML)
    (configs / "usecase_space.yaml").write_text(USECASE_SPACE_YAML)
    (configs / "model.yaml").write_text(model_yaml)

    raw = root / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    # The same buildings appear in proposed and baseline, which is what makes
    # them pairable for savings. Train and test hold different buildings.
    for slot, filename in BATCH_FILES.items():
        scenario, split = slot.split("_")
        write_batch_csv(
            raw / filename,
            scenario=scenario,
            seed=0 if split == "train" else 1,
            name_offset=0 if split == "train" else 1000,
        )

    return {
        "root": root,
        "raw": raw,
        "schema_path": configs / "schema.yaml",
        "space_path": configs / "usecase_space.yaml",
        "model_path": configs / "model.yaml",
    }


@pytest.fixture
def project(tmp_path: Path) -> dict:
    return write_project(tmp_path)


@pytest.fixture
def ctx(project: dict) -> PipelineContext:
    return PipelineContext.load(
        schema_path=project["schema_path"],
        usecase_space_path=project["space_path"],
        model_config_path=project["model_path"],
    )


@pytest.fixture
def batch_index(project: dict, ctx: PipelineContext) -> pd.DataFrame:
    return stage_scan(project["raw"], ctx=ctx)
