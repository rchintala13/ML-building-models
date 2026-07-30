from pathlib import Path

import pandas as pd
import pytest

from ml179d.io.csv_loader import apply_row_validity
from ml179d.pipeline import PipelineContext, stage_dataset, stage_scan
from ml179d.schema import load_schema
from ml179d.train import train_usecase_with_savings

from conftest import (
    BATCH_FILES,
    N_ROWS,
    SCHEMA_YAML,
    USECASE_ID,
    make_raw_frame,
    write_project,
)

"""
Failed simulation rows ('datapoint failure') carry blank features and blank
usecase identifiers. schema.row_validity keeps only rows whose status columns
match a whitelist.
"""

FAILED = (0, 3, 7)


def project_with_failures(tmp_path: Path, *, failures_by_slot: dict) -> dict:
    project = write_project(tmp_path)
    for slot, filename in BATCH_FILES.items():
        scenario, split = slot.split("_")
        make_raw_frame(
            scenario=scenario,
            seed=0 if split == "train" else 1,
            name_offset=0 if split == "train" else 1000,
            failed_rows=failures_by_slot.get(slot, ()),
        ).to_csv(project["raw"] / filename, index=False)
    return project


def context_for(project: dict) -> PipelineContext:
    return PipelineContext.load(
        schema_path=project["schema_path"],
        usecase_space_path=project["space_path"],
        model_config_path=project["model_path"],
    )


# ---------------------------------------------------------------
# the filter itself
# ---------------------------------------------------------------

def test_whitelist_rejects_unlisted_values(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML)
    schema = load_schema(schema_path)

    df = pd.DataFrame(
        {
            "status": ["completed", "completed", "started", "queued"],
            "status_message": [
                "completed normal",
                "datapoint failure",
                None,
                "completed normal",
            ],
        }
    )

    kept, dropped = apply_row_validity(df, schema=schema)

    # only the first row satisfies both conditions
    assert len(kept) == 1
    assert dropped == 3


def test_no_row_validity_configured_is_a_passthrough(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML.replace(
        "row_validity:\n  status:\n    - completed\n  status_message:\n    - completed normal\n",
        "",
    ))
    schema = load_schema(schema_path)

    df = pd.DataFrame({"status": ["anything"], "status_message": [None]})
    kept, dropped = apply_row_validity(df, schema=schema)

    assert schema.row_validity == {}
    assert len(kept) == 1
    assert dropped == 0


def test_missing_status_column_raises(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML)
    schema = load_schema(schema_path)

    with pytest.raises(KeyError, match="missing row_validity column"):
        apply_row_validity(pd.DataFrame({"a": [1]}), schema=schema, source="x.csv")


def test_empty_allowed_values_raises(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(SCHEMA_YAML.replace("    - completed normal", ""))

    with pytest.raises(ValueError, match="no allowed values"):
        load_schema(schema_path)


def test_scalar_allowed_value_is_accepted(tmp_path: Path):
    schema_path = tmp_path / "schema.yaml"
    schema_path.write_text(
        SCHEMA_YAML.replace("  status:\n    - completed\n", "  status: completed\n")
    )

    schema = load_schema(schema_path)

    assert schema.row_validity["status"] == ["completed"]


# ---------------------------------------------------------------
# through the pipeline
# ---------------------------------------------------------------

def test_failed_rows_never_reach_the_dataset(tmp_path: Path):
    project = project_with_failures(tmp_path, failures_by_slot={"proposed_train": FAILED})
    ctx = context_for(project)
    index = stage_scan(project["raw"], ctx=ctx)

    data = stage_dataset(
        index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        split="train",
        target_set="electricity",
        model_type="plain_linear",
    )

    assert len(data) == N_ROWS - len(FAILED)
    assert data.n_invalid_rows == len(FAILED)
    # no NaN survives into the model inputs
    assert not data.X.isna().any().any()
    assert not data.y.isna().any().any()
    # the failed buildings are gone by name
    assert not {f"bldg_{i}" for i in FAILED} & set(data.row_ids)


def test_scan_survives_a_failed_first_row(tmp_path: Path):
    """
    The usecase identifiers are blank on failed rows, so scanning must not read
    row 0 blindly.
    """
    project = project_with_failures(
        tmp_path, failures_by_slot={slot: (0, 1) for slot in BATCH_FILES}
    )
    ctx = context_for(project)

    index = stage_scan(project["raw"], ctx=ctx)

    assert len(index) == 4
    assert set(index["usecase_id"]) == {USECASE_ID}
    assert set(index["climate_zone"]) == {"5A"}


def test_all_rows_failed_raises(tmp_path: Path):
    project = project_with_failures(
        tmp_path, failures_by_slot={"proposed_train": tuple(range(N_ROWS))}
    )
    ctx = context_for(project)

    with pytest.raises(ValueError, match="no row with all of"):
        stage_scan(project["raw"], ctx=ctx)


# ---------------------------------------------------------------
# savings pairing
# ---------------------------------------------------------------

def test_savings_pairs_on_the_intersection_when_failures_differ(tmp_path: Path):
    """
    Proposed and baseline fail on different buildings, which is the normal case
    for real simulation output.
    """
    project = project_with_failures(
        tmp_path,
        failures_by_slot={"proposed_test": (0, 1), "baseline_test": (5,)},
    )
    ctx = context_for(project)
    index = stage_scan(project["raw"], ctx=ctx)

    result = train_usecase_with_savings(
        index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
        require_full_overlap=False,
    )
    savings = result.savings

    # 40 rows, minus 2 failed in proposed and 1 in baseline, no overlap
    assert len(savings.frame) == N_ROWS - 3
    assert savings.n_unpaired_proposed == 1   # baseline lost bldg_1005
    assert savings.n_unpaired_baseline == 2   # proposed lost bldg_1000, bldg_1001
    assert (savings.frame["savings_true"] > 0).all()


def test_strict_pairing_still_raises_on_mismatch(tmp_path: Path):
    project = project_with_failures(
        tmp_path, failures_by_slot={"proposed_test": (0,)}
    )
    ctx = context_for(project)
    index = stage_scan(project["raw"], ctx=ctx)

    with pytest.raises(ValueError, match="have no counterpart"):
        train_usecase_with_savings(
            index,
            ctx=ctx,
            usecase_id=USECASE_ID,
            target_set="electricity",
            model_type="plain_linear",
            require_full_overlap=True,
        )
