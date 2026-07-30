import numpy as np
import pandas as pd
import pytest

from ml179d.models.factory import ConfigEstimatorFactory, build_estimator
from ml179d.models.metrics import metrics_by_target, regression_metrics
from ml179d.pipeline import PipelineContext, stage_dataset, stage_scan
from ml179d.train import (
    compute_savings,
    fit,
    pair_scenarios,
    train_usecase,
    train_usecase_with_savings,
)

from conftest import (
    BATCH_FILES,
    ENERGY_COEFFS,
    MODEL_YAML,
    N_ROWS,
    USECASE_ID,
    make_raw_frame,
    write_project,
)

TARGET = "total_electricity_179d"


def data_for(batch_index, ctx, scenario: str, split: str, model_type="plain_linear"):
    return stage_dataset(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario=scenario,
        split=split,
        target_set="electricity",
        model_type=model_type,
    )


# ---------------------------------------------------------------
# estimator factory
# ---------------------------------------------------------------

def test_factory_builds_configured_estimators(ctx):
    factory = ConfigEstimatorFactory(ctx.model_config.estimators)

    assert factory("plain_linear") is not None
    assert factory("ridge_poly") is not None


def test_factory_rejects_unconfigured_model_type(ctx):
    factory = ConfigEstimatorFactory(ctx.model_config.estimators)

    with pytest.raises(KeyError, match="no entry in the 'estimators' section"):
        factory("xgboost")


def test_factory_call_params_override_config(ctx):
    factory = ConfigEstimatorFactory(ctx.model_config.estimators)

    model = factory("ridge_poly", {"alpha": 42.0})

    assert model.named_steps["model"].alpha == 42.0


def test_build_estimator_rejects_unknown_kind():
    with pytest.raises(KeyError, match="Unknown estimator kind"):
        build_estimator("magic", {})


# ---------------------------------------------------------------
# metrics
# ---------------------------------------------------------------

def test_regression_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])

    m = regression_metrics(y, y)

    assert m["r2"] == pytest.approx(1.0)
    assert m["mae"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["cvrmse_pct"] == pytest.approx(0.0)
    assert m["nmbe_pct"] == pytest.approx(0.0)
    assert m["n"] == 4


def test_regression_metrics_known_values():
    true = np.array([100.0, 200.0, 300.0])
    pred = np.array([110.0, 190.0, 300.0])

    m = regression_metrics(true, pred)

    assert m["mae"] == pytest.approx(20.0 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(200.0 / 3))
    assert m["mean_observed"] == pytest.approx(200.0)
    # errors +10 and -10 cancel, so bias is zero while rmse is not
    assert m["nmbe_pct"] == pytest.approx(0.0)
    assert m["cvrmse_pct"] > 0


def test_regression_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="different shapes"):
        regression_metrics([1.0, 2.0], [1.0])


def test_metrics_by_target_requires_matching_index():
    a = pd.DataFrame({TARGET: [1.0, 2.0]}, index=["x", "y"])
    b = pd.DataFrame({TARGET: [1.0, 2.0]}, index=["x", "z"])

    with pytest.raises(ValueError, match="same row index"):
        metrics_by_target(a, b)


# ---------------------------------------------------------------
# fitting
# ---------------------------------------------------------------

def test_train_usecase_recovers_linear_target(batch_index, ctx):
    model = train_usecase(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        target_set="electricity",
        model_type="plain_linear",
    )

    # the fixture target is exactly linear in gross_floor_area and roof_area
    assert model.train_metrics[TARGET]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert model.test_metrics[TARGET]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert model.n_train == N_ROWS
    assert model.scenario == "proposed"
    assert model.target_names == [TARGET]


def test_predictions_keep_row_id_index(batch_index, ctx):
    model = train_usecase(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        target_set="electricity",
        model_type="plain_linear",
    )
    test = data_for(batch_index, ctx, "proposed", "test")

    preds = model.predict(test.X)

    assert preds.index.name == "name"
    assert list(preds.index) == list(test.row_ids)
    assert list(preds.columns) == [TARGET]


def test_predict_rejects_missing_features(batch_index, ctx):
    model = train_usecase(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        target_set="electricity",
        model_type="plain_linear",
    )
    test = data_for(batch_index, ctx, "proposed", "test")

    with pytest.raises(KeyError, match="missing features"):
        model.predict(test.X.drop(columns=["roof_area"]))


def test_predict_is_order_insensitive(batch_index, ctx):
    """
    Columns are reindexed to the fitted order, so shuffled input is safe.
    """
    model = train_usecase(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        target_set="electricity",
        model_type="plain_linear",
    )
    test = data_for(batch_index, ctx, "proposed", "test")

    shuffled = test.X[list(reversed(test.feature_names))]

    pd.testing.assert_frame_equal(model.predict(test.X), model.predict(shuffled))


def test_fit_rejects_empty_training_data(batch_index, ctx):
    train = data_for(batch_index, ctx, "proposed", "train")
    empty = type(train)(
        X=train.X.iloc[:0],
        y=train.y.iloc[:0],
        usecase_id=train.usecase_id,
        scenario=train.scenario,
        split=train.split,
        recipe=train.recipe,
    )
    factory = ConfigEstimatorFactory(ctx.model_config.estimators)

    with pytest.raises(ValueError, match="no training rows"):
        fit(empty, estimator=factory("plain_linear"))


def test_ridge_poly_trains_through_its_transforms(batch_index, ctx):
    model = train_usecase(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario="proposed",
        target_set="electricity",
        model_type="ridge_poly",
    )

    assert "roof_area_cal_log" in model.feature_names
    assert "gross_floor_area" not in model.feature_names
    assert model.test_metrics[TARGET]["r2"] > 0.99


# ---------------------------------------------------------------
# scenario pairing
# ---------------------------------------------------------------

def test_proposed_and_baseline_share_row_ids(batch_index, ctx):
    proposed = data_for(batch_index, ctx, "proposed", "test")
    baseline = data_for(batch_index, ctx, "baseline", "test")

    assert list(proposed.row_ids) == list(baseline.row_ids)


def test_pair_scenarios_returns_shared_rows(batch_index, ctx):
    proposed = data_for(batch_index, ctx, "proposed", "test")
    baseline = data_for(batch_index, ctx, "baseline", "test")

    rows, n_p, n_b = pair_scenarios(proposed, baseline)

    assert list(rows) == list(proposed.row_ids)
    assert (n_p, n_b) == (0, 0)


def test_pair_scenarios_rejects_partial_overlap(batch_index, ctx):
    proposed = data_for(batch_index, ctx, "proposed", "test")
    baseline = data_for(batch_index, ctx, "baseline", "test")
    trimmed = type(baseline)(
        X=baseline.X.iloc[:-3],
        y=baseline.y.iloc[:-3],
        usecase_id=baseline.usecase_id,
        scenario=baseline.scenario,
        split=baseline.split,
        recipe=baseline.recipe,
    )

    with pytest.raises(ValueError, match="have no counterpart"):
        pair_scenarios(proposed, trimmed)

    rows, n_p, n_b = pair_scenarios(proposed, trimmed, require_full_overlap=False)
    assert len(rows) == N_ROWS - 3
    assert (n_p, n_b) == (3, 0)


def test_pair_scenarios_rejects_different_splits(batch_index, ctx):
    proposed = data_for(batch_index, ctx, "proposed", "train")
    baseline = data_for(batch_index, ctx, "baseline", "test")

    with pytest.raises(ValueError, match="different splits"):
        pair_scenarios(proposed, baseline)


# ---------------------------------------------------------------
# savings
# ---------------------------------------------------------------

def test_compute_savings_matches_analytic_difference(batch_index, ctx):
    result = train_usecase_with_savings(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
    )
    savings = result.savings
    frame = savings.frame

    assert list(frame.index.names) == ["name"]
    assert len(frame) == N_ROWS

    # savings = baseline - proposed, and the fixture makes baseline strictly
    # more energy intensive on every coefficient, so savings are positive
    assert ENERGY_COEFFS["baseline"]["gfa"] > ENERGY_COEFFS["proposed"]["gfa"]
    assert (frame["savings_true"] > 0).all()
    pd.testing.assert_series_equal(
        frame["savings_true"],
        frame["baseline_true"] - frame["proposed_true"],
        check_names=False,
    )
    # a perfect fit means predicted savings track observed savings
    assert savings.metrics["r2"] == pytest.approx(1.0, abs=1e-6)
    assert savings.n_unpaired_proposed == 0
    assert savings.n_unpaired_baseline == 0


def test_savings_is_computed_on_the_test_split(batch_index, ctx):
    result = train_usecase_with_savings(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
    )

    assert result.savings.split == "test"
    # test buildings use the 1000-offset names from the fixture
    assert all(str(name).startswith("bldg_10") for name in result.savings.frame.index)


def test_compute_savings_rejects_swapped_scenarios(batch_index, ctx):
    proposed = data_for(batch_index, ctx, "proposed", "test")
    baseline = data_for(batch_index, ctx, "baseline", "test")
    models = train_usecase_with_savings(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
    ).models

    with pytest.raises(ValueError, match="Expected proposed_data.scenario"):
        compute_savings(
            proposed_data=baseline,
            baseline_data=proposed,
            proposed_model=models["proposed"],
            baseline_model=models["baseline"],
        )


def test_savings_alignment_survives_shuffled_baseline_rows(tmp_path):
    """
    Savings must join on name, not on row position.
    """
    project = write_project(tmp_path)

    # rewrite the baseline test batch with its rows in reverse order
    frame = make_raw_frame(scenario="baseline", seed=1, name_offset=1000)
    frame.iloc[::-1].to_csv(project["raw"] / BATCH_FILES["baseline_test"], index=False)

    ctx = PipelineContext.load(
        schema_path=project["schema_path"],
        usecase_space_path=project["space_path"],
        model_config_path=project["model_path"],
    )
    index = stage_scan(project["raw"], ctx=ctx)

    result = train_usecase_with_savings(
        index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
    )
    frame_out = result.savings.frame

    # every row still pairs the same building with itself
    assert (frame_out["savings_true"] > 0).all()
    assert result.savings.metrics["r2"] == pytest.approx(1.0, abs=1e-6)


def test_train_usecase_with_savings_returns_both_models(batch_index, ctx):
    result = train_usecase_with_savings(
        batch_index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        target_set="electricity",
        model_type="plain_linear",
    )

    assert set(result.models) == {"proposed", "baseline"}
    assert result.models["proposed"].scenario == "proposed"
    assert result.models["baseline"].scenario == "baseline"
    # the two models differ: baseline energy is higher for the same inputs
    test_X = data_for(batch_index, ctx, "proposed", "test").X
    p = result.models["proposed"].predict(test_X)[TARGET]
    b = result.models["baseline"].predict(test_X)[TARGET]
    assert (b > p).all()
