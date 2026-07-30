from pathlib import Path

import pytest

from ml179d.config import ModelConfig
from ml179d.pipeline import PipelineContext, stage_dataset, stage_scan

from conftest import MODEL_YAML, N_ROWS, USECASE_ID, write_project

"""
Filters are configured in the top-level 'filters' block of model.yaml and
applied in stage_dataset between base features and feature selection.
"""

FILTERS_YAML = """
filters:
  min_values:
    gross_floor_area: 1000
  max_values: {}
  apply_to_test: false
  overrides:
    - when:
        scenario: baseline
      min_values:
        gross_floor_area: 1500
    - when:
        target_set: natural_gas
      max_values:
        aspect_ratio: 1.4
    - when:
        usecase_id: small_office_PSZ-HP_CZ5A
        scenario: proposed
      apply_to_test: true
"""


def config_with(filters_yaml: str) -> ModelConfig:
    import tempfile

    path = Path(tempfile.mkdtemp()) / "model.yaml"
    path.write_text(MODEL_YAML + filters_yaml)
    return ModelConfig.from_yaml(path)


# ---------------------------------------------------------------
# resolution
# ---------------------------------------------------------------

def test_global_bounds_apply_when_nothing_matches():
    config = config_with(FILTERS_YAML)

    specs, apply_to_test = config.resolve_filters(
        target_set="electricity", scenario="baseline", usecase_id="other_usecase"
    )

    # the baseline override tightens the global 1000 to 1500
    assert [(s.column, s.min_value, s.max_value) for s in specs] == [
        ("gross_floor_area", 1500, None)
    ]
    assert apply_to_test is False


def test_no_filters_section_yields_nothing():
    config = config_with("")

    specs, apply_to_test = config.resolve_filters(
        target_set="electricity", scenario="proposed", usecase_id=USECASE_ID
    )

    assert specs == ()
    assert apply_to_test is False


def test_overrides_merge_per_column():
    """
    A rule that sets a max on one column must not drop the global min on another.
    """
    config = config_with(FILTERS_YAML)

    specs, _ = config.resolve_filters(
        target_set="natural_gas", scenario="proposed", usecase_id=USECASE_ID
    )
    by_column = {s.column: s for s in specs}

    assert by_column["gross_floor_area"].min_value == 1000
    assert by_column["aspect_ratio"].max_value == 1.4
    assert by_column["aspect_ratio"].min_value is None


def test_later_matching_override_wins():
    config = config_with(FILTERS_YAML)

    specs, _ = config.resolve_filters(
        target_set="natural_gas", scenario="baseline", usecase_id=USECASE_ID
    )
    by_column = {s.column: s for s in specs}

    assert by_column["gross_floor_area"].min_value == 1500


def test_apply_to_test_override():
    config = config_with(FILTERS_YAML)

    _, proposed = config.resolve_filters(
        target_set="electricity", scenario="proposed", usecase_id=USECASE_ID
    )
    _, baseline = config.resolve_filters(
        target_set="electricity", scenario="baseline", usecase_id=USECASE_ID
    )

    assert proposed is True
    assert baseline is False


def test_null_bound_clears_the_filter():
    config = config_with(
        """
filters:
  min_values:
    gross_floor_area: 400
  overrides:
    - when:
        scenario: baseline
      min_values:
        gross_floor_area: null
"""
    )

    kept, _ = config.resolve_filters(target_set="electricity", scenario="proposed")
    cleared, _ = config.resolve_filters(target_set="electricity", scenario="baseline")

    assert [s.column for s in kept] == ["gross_floor_area"]
    assert cleared == ()


def test_selector_rejects_unknown_field():
    config = config_with(
        """
filters:
  overrides:
    - when:
        model_type: ridge_poly
      min_values:
        gross_floor_area: 400
"""
    )

    with pytest.raises(ValueError, match="unknown field"):
        config.resolve_filters(target_set="electricity", scenario="proposed")


def test_selector_accepts_a_list_of_values():
    config = config_with(
        """
filters:
  overrides:
    - when:
        scenario: [proposed, baseline]
      min_values:
        gross_floor_area: 900
"""
    )

    for scenario in ("proposed", "baseline"):
        specs, _ = config.resolve_filters(
            target_set="electricity", scenario=scenario
        )
        assert specs[0].min_value == 900


def test_overrides_must_be_a_list():
    config = config_with(
        """
filters:
  overrides:
    when:
      scenario: proposed
"""
    )

    with pytest.raises(ValueError, match="must be a list of rules"):
        config.resolve_filters(target_set="electricity", scenario="proposed")


# ---------------------------------------------------------------
# application in the pipeline
# ---------------------------------------------------------------

def build_project(tmp_path: Path, filters_yaml: str):
    project = write_project(tmp_path, model_yaml=MODEL_YAML + filters_yaml)
    ctx = PipelineContext.load(
        schema_path=project["schema_path"],
        usecase_space_path=project["space_path"],
        model_config_path=project["model_path"],
    )
    return project, ctx, stage_scan(project["raw"], ctx=ctx)


def dataset(ctx, index, scenario="proposed", split="train"):
    return stage_dataset(
        index,
        ctx=ctx,
        usecase_id=USECASE_ID,
        scenario=scenario,
        split=split,
        target_set="electricity",
        model_type="plain_linear",
    )


# fixture gross_floor_area runs 500, 560, ... 500 + 60*39 = 2840
FILTER_1000 = """
filters:
  min_values:
    gross_floor_area: 1000
  apply_to_test: false
"""


def test_filter_drops_rows_from_train(tmp_path: Path):
    _, ctx, index = build_project(tmp_path, FILTER_1000)

    train = dataset(ctx, index, split="train")

    assert len(train) < N_ROWS
    assert train.X["gross_floor_area"].min() >= 1000


def test_test_split_is_unfiltered_by_default(tmp_path: Path):
    """
    filter_train_and_test = False: fit on a restricted range, score on all of it.
    """
    _, ctx, index = build_project(tmp_path, FILTER_1000)

    train = dataset(ctx, index, split="train")
    test = dataset(ctx, index, split="test")

    assert len(train) < N_ROWS
    assert len(test) == N_ROWS
    assert test.X["gross_floor_area"].min() < 1000


def test_apply_to_test_true_filters_both_splits(tmp_path: Path):
    _, ctx, index = build_project(
        tmp_path, FILTER_1000.replace("apply_to_test: false", "apply_to_test: true")
    )

    train = dataset(ctx, index, split="train")
    test = dataset(ctx, index, split="test")

    assert len(train) < N_ROWS
    assert len(test) < N_ROWS
    assert test.X["gross_floor_area"].min() >= 1000


def test_filter_can_reference_a_derived_column(tmp_path: Path):
    """
    Filters run after base features, so roof_area_cal is available even though
    it is not a schema column.
    """
    _, ctx, index = build_project(
        tmp_path,
        """
filters:
  min_values:
    roof_area_cal: 600
  apply_to_test: true
""",
    )

    train = dataset(ctx, index, split="train")

    assert len(train) < N_ROWS
    assert train.X["roof_area_cal"].min() >= 600


def test_scenario_specific_filter_breaks_savings_pairing(tmp_path: Path):
    """
    Filtering proposed but not baseline drops different buildings from each,
    which must fail loudly rather than misalign savings.
    """
    from ml179d.train import train_usecase_with_savings

    _, ctx, index = build_project(
        tmp_path,
        """
filters:
  overrides:
    - when:
        scenario: proposed
      min_values:
        gross_floor_area: 1000
      apply_to_test: true
""",
    )

    with pytest.raises(ValueError, match="have no counterpart"):
        train_usecase_with_savings(
            index,
            ctx=ctx,
            usecase_id=USECASE_ID,
            target_set="electricity",
            model_type="plain_linear",
        )


def test_unknown_filter_column_raises(tmp_path: Path):
    _, ctx, index = build_project(
        tmp_path,
        """
filters:
  min_values:
    not_a_column: 1
""",
    )

    with pytest.raises(KeyError, match="not_a_column"):
        dataset(ctx, index, split="train")
