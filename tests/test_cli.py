import json
from pathlib import Path

import pandas as pd
import pytest

from ml179d.cli import main
from ml179d.io.artifacts import load_model
from ml179d.selection import (
    DEFAULT_MODEL_TYPE,
    Job,
    Selection,
    format_plan,
    resolve_jobs,
)

from conftest import USECASE_ID, write_project

TARGET_SETS = ("electricity",)
MODEL_TYPES = ("plain_linear", "ridge_poly")


# ---------------------------------------------------------------
# job selection
# ---------------------------------------------------------------

def plan_for(selection: Selection, ctx, expected=None, available=None):
    expected = expected if expected is not None else [USECASE_ID]
    return resolve_jobs(
        selection,
        resolver=ctx.resolver,
        expected_usecase_ids=expected,
        available_usecase_ids=available,
        known_target_sets=TARGET_SETS,
        known_model_types=MODEL_TYPES,
    )


def test_default_selection_uses_ridge_poly_only(ctx):
    plan = plan_for(Selection(target_sets=TARGET_SETS), ctx)

    assert [j.model_type for j in plan.jobs] == [DEFAULT_MODEL_TYPE]


def test_default_selection_runs_both_scenarios_and_savings(ctx):
    plan = plan_for(Selection(target_sets=TARGET_SETS), ctx)

    job = plan.jobs[0]
    assert set(job.scenarios) == {"proposed", "baseline"}
    assert job.computes_savings is True


def test_single_scenario_skips_savings(ctx):
    plan = plan_for(Selection(target_sets=TARGET_SETS, scenarios=("proposed",)), ctx)

    assert plan.jobs[0].scenarios == ("proposed",)
    assert plan.jobs[0].computes_savings is False


def test_explicit_model_types_expand(ctx):
    plan = plan_for(
        Selection(target_sets=TARGET_SETS, model_types=MODEL_TYPES), ctx
    )

    assert sorted(j.model_type for j in plan.jobs) == sorted(MODEL_TYPES)


def test_unknown_model_type_raises(ctx):
    with pytest.raises(ValueError, match="Unknown model_type"):
        plan_for(Selection(target_sets=TARGET_SETS, model_types=("magic",)), ctx)


def test_unknown_usecase_id_raises(ctx):
    with pytest.raises(ValueError, match="Unknown usecase id"):
        plan_for(Selection(target_sets=TARGET_SETS, usecase_ids=("nope",)), ctx)


def test_axis_filter_accepts_raw_or_slug(ctx):
    raw = plan_for(Selection(target_sets=TARGET_SETS, building_types=("SmallOffice",)), ctx)
    slug = plan_for(Selection(target_sets=TARGET_SETS, building_types=("small_office",)), ctx)

    assert len(raw) == len(slug) == 1


def test_axis_filter_excludes_non_matching(ctx):
    with pytest.raises(ValueError, match="matched no usecases"):
        plan_for(
            Selection(target_sets=TARGET_SETS, climate_zones=("1A",)),
            ctx,
            expected=[USECASE_ID],
        )


def test_unavailable_usecases_are_reported_not_run(ctx):
    plan = plan_for(
        Selection(target_sets=TARGET_SETS),
        ctx,
        expected=[USECASE_ID, "small_office_PSZ-HP_CZ1A"],
        available=[USECASE_ID],
    )

    assert [j.usecase_id for j in plan.jobs] == [USECASE_ID]
    assert plan.requested_but_unavailable == ("small_office_PSZ-HP_CZ1A",)
    assert "not available in the batch catalog" in format_plan(plan)


def test_job_cross_product(ctx):
    plan = plan_for(
        Selection(target_sets=TARGET_SETS, model_types=MODEL_TYPES),
        ctx,
        expected=[USECASE_ID],
    )

    assert len(plan) == 1 * len(TARGET_SETS) * len(MODEL_TYPES)


# ---------------------------------------------------------------
# cli end to end
# ---------------------------------------------------------------

@pytest.fixture
def cli_project(tmp_path: Path) -> dict:
    return write_project(tmp_path)


def run(project: dict, *argv: str) -> int:
    return main(
        [
            *argv,
            "--configs", str(project["root"] / "configs"),
            "--data-root", str(project["root"] / "data"),
            "--output-dir", str(project["root"] / "outputs"),
        ]
    )


def test_cli_scan(cli_project, capsys):
    code = run(cli_project, "scan")

    assert code == 0
    assert "Scanned 4 batch file(s)" in capsys.readouterr().out
    assert (cli_project["root"] / "data" / "interim" / "batch_index.parquet").exists()


def test_cli_catalog(cli_project, capsys):
    code = run(cli_project, "catalog")

    out = capsys.readouterr().out
    assert code == 0
    assert "Trainable usecases:   1" in out
    assert (cli_project["root"] / "outputs" / "metrics" / "catalog_coverage.csv").exists()


def test_cli_train_dry_run_defaults_to_ridge_poly(cli_project, capsys):
    code = run(cli_project, "train", "--dry-run")

    out = capsys.readouterr().out
    assert code == 0
    assert "ridge_poly" in out
    assert "plain_linear" not in out
    assert "+savings" in out
    assert "Dry run; nothing was fitted." in out
    # nothing written
    assert not (cli_project["root"] / "outputs" / "models").exists()


def test_cli_train_writes_models_and_metrics(cli_project, capsys):
    code = run(cli_project, "train", "--model-type", "plain_linear")

    out = capsys.readouterr().out
    assert code == 0, out

    root = cli_project["root"] / "outputs"
    model_dir = root / "models" / USECASE_ID / "electricity" / "plain_linear"

    assert (model_dir / "proposed.joblib").exists()
    assert (model_dir / "baseline.joblib").exists()
    assert load_model(model_dir / "proposed.joblib") is not None

    sidecar = json.loads((model_dir / "proposed.json").read_text())
    assert sidecar["scenario"] == "proposed"
    assert "roof_area" in sidecar["feature_names"]

    metrics = pd.read_csv(root / "metrics" / "metrics.csv")
    assert set(metrics["scenario"]) == {"proposed", "baseline", "savings"}
    assert {"r2", "cvrmse_pct", "nmbe_pct"} <= set(metrics.columns)


def test_cli_train_writes_savings_frame(cli_project):
    run(cli_project, "train", "--model-type", "plain_linear")

    path = (
        cli_project["root"] / "outputs" / "metrics" / "savings"
        / f"{USECASE_ID}__electricity__plain_linear.csv"
    )
    frame = pd.read_csv(path)

    assert "savings_true" in frame.columns
    assert "savings_pred" in frame.columns
    # the row id survives persistence, which is what makes savings auditable
    assert frame.columns[0] == "name"
    assert (frame["savings_true"] > 0).all()


def test_cli_train_single_scenario_skips_savings(cli_project):
    run(
        cli_project,
        "train",
        "--model-type", "plain_linear",
        "--scenario", "proposed",
    )

    root = cli_project["root"] / "outputs"
    model_dir = root / "models" / USECASE_ID / "electricity" / "plain_linear"

    assert (model_dir / "proposed.joblib").exists()
    assert not (model_dir / "baseline.joblib").exists()
    assert not (root / "metrics" / "savings").exists()

    metrics = pd.read_csv(root / "metrics" / "metrics.csv")
    assert set(metrics["scenario"]) == {"proposed"}


def test_cli_train_usecase_filter(cli_project, capsys):
    code = run(cli_project, "train", "--usecase", USECASE_ID, "--dry-run")

    assert code == 0
    assert USECASE_ID in capsys.readouterr().out


def test_cli_train_unknown_usecase_exits_nonzero(cli_project, capsys):
    code = run(cli_project, "train", "--usecase", "not_a_usecase", "--dry-run")

    assert code == 2
    assert "Unknown usecase id" in capsys.readouterr().err


def test_cli_train_rejects_unconfigured_model_type_before_fitting(cli_project, capsys):
    """
    A model type with no 'estimators' entry is rejected during planning, so no
    partial output is written.
    """
    model_path = cli_project["model_path"]
    model_path.write_text(
        model_path.read_text().replace(
            "model_type_overrides:", "model_type_overrides:\n  broken:\n    transforms: []"
        )
    )

    code = run(cli_project, "train", "--model-type", "plain_linear,broken")
    captured = capsys.readouterr()

    assert code == 2
    assert "Unknown model_type" in captured.err
    assert not (cli_project["root"] / "outputs" / "models").exists()


def test_cli_train_continues_past_a_failing_job(cli_project, capsys):
    """
    A job that fails at fit time must not abort the remaining jobs.
    """
    model_path = cli_project["model_path"]
    # a transform referencing a column that is not selected fails for ridge_poly only
    model_path.write_text(
        model_path.read_text().replace("            - roof_area_cal\n", "            - not_a_column\n")
    )

    code = run(cli_project, "train", "--model-type", "plain_linear,ridge_poly")
    out = capsys.readouterr().out

    assert code == 1
    assert "ok   " in out
    assert "FAIL " in out
    assert "1 job(s) failed" in out
    # the healthy job still produced output
    assert (
        cli_project["root"] / "outputs" / "models" / USECASE_ID / "electricity"
        / "plain_linear" / "proposed.joblib"
    ).exists()


def test_cli_requires_a_subcommand(capsys):
    with pytest.raises(SystemExit):
        main([])
