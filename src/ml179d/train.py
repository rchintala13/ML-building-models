from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from ml179d.models.factory import ConfigEstimatorFactory
from ml179d.models.metrics import metrics_by_target, regression_metrics
from ml179d.models.protocols import Estimator
from ml179d.pipeline import PipelineContext, TrainingData, stage_dataset

"""
Training
--------
One model per (usecase, scenario, target_set, model_type).

Why per scenario: the baseline model is trained on the SAME user inputs as the
proposed model -- schema.yaml maps the baseline scenario to the '*_proposed'
raw columns -- but targets the baseline simulation's energy. So a building's
savings are the difference between two models' predictions on one input row,
not the output of a savings model.

    savings = baseline_energy - proposed_energy

Everything here is indexed by the row_id ('name'). A building in the proposed
batch carries the same name in the baseline batch, and that is the only thing
tying the two predictions together.
"""

SAVINGS_PREFIX = "savings"


# ---------------------------------------------------------------
# fitting
# ---------------------------------------------------------------

@dataclass(frozen=True)
class FittedModel:
    """
    A fitted estimator plus the provenance needed to reuse it safely.

    feature_names is the exact column order the estimator was fitted on;
    predict() reindexes to it so a caller cannot silently pass columns in a
    different order.
    """
    estimator: Estimator
    usecase_id: str
    scenario: str
    target_set: str
    model_type: str
    feature_names: List[str]
    target_names: List[str]
    n_train: int
    train_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    test_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict, preserving the row_id index and target column names.
        """
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise KeyError(
                f"X is missing features the model was fitted on: {missing}"
            )

        raw = self.estimator.predict(X[self.feature_names])
        values = pd.DataFrame(raw, index=X.index)

        if values.shape[1] != len(self.target_names):
            raise ValueError(
                f"Estimator returned {values.shape[1]} output column(s), expected "
                f"{len(self.target_names)} for targets {self.target_names}."
            )

        values.columns = self.target_names
        return values


def _fit_target(y: pd.DataFrame):
    """
    sklearn wants a 1d y for a single target and a 2d y for several.
    """
    return y.iloc[:, 0] if y.shape[1] == 1 else y


def fit(
    train: TrainingData,
    *,
    estimator: Estimator,
    test: Optional[TrainingData] = None,
) -> FittedModel:
    """
    Fit one estimator on a training split and score it on an optional test split.
    """
    if len(train) == 0:
        raise ValueError(
            f"usecase '{train.usecase_id}' scenario '{train.scenario}': no training "
            f"rows. Check the filters in model.yaml."
        )

    estimator.fit(train.X, _fit_target(train.y))

    fitted = FittedModel(
        estimator=estimator,
        usecase_id=train.usecase_id,
        scenario=train.scenario,
        target_set=train.recipe.target_set,
        model_type=train.recipe.model_type,
        feature_names=train.feature_names,
        target_names=train.target_names,
        n_train=len(train),
    )

    train_metrics = metrics_by_target(train.y, fitted.predict(train.X))
    test_metrics: Dict[str, Dict[str, float]] = {}

    if test is not None:
        if test.feature_names != train.feature_names:
            raise ValueError(
                f"usecase '{train.usecase_id}': train and test features differ.\n"
                f"  train: {train.feature_names}\n  test:  {test.feature_names}"
            )
        test_metrics = metrics_by_target(test.y, fitted.predict(test.X))

    return FittedModel(
        estimator=fitted.estimator,
        usecase_id=fitted.usecase_id,
        scenario=fitted.scenario,
        target_set=fitted.target_set,
        model_type=fitted.model_type,
        feature_names=fitted.feature_names,
        target_names=fitted.target_names,
        n_train=fitted.n_train,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )


def train_usecase(
    batch_index: pd.DataFrame,
    *,
    ctx: PipelineContext,
    usecase_id: str,
    scenario: str,
    target_set: str,
    model_type: str,
    factory: Optional[ConfigEstimatorFactory] = None,
    evaluate: bool = True,
) -> FittedModel:
    """
    Build the train (and test) datasets for one usecase and fit a model.
    """
    factory = factory or ConfigEstimatorFactory(ctx.model_config.estimators)

    def build(split: str) -> TrainingData:
        return stage_dataset(
            batch_index,
            ctx=ctx,
            usecase_id=usecase_id,
            scenario=scenario,
            split=split,
            target_set=target_set,
            model_type=model_type,
        )

    train_data = build("train")
    test_data = build("test") if evaluate else None

    return fit(
        train_data,
        estimator=factory(model_type, target_set=target_set, scenario=scenario),
        test=test_data,
    )


# ---------------------------------------------------------------
# savings
# ---------------------------------------------------------------

@dataclass(frozen=True)
class SavingsResult:
    """
    Per-building savings, observed and predicted.

    frame columns, all indexed by row_id:
        proposed_true, proposed_pred
        baseline_true, baseline_pred
        savings_true,  savings_pred
    """
    frame: pd.DataFrame
    usecase_id: str
    target_set: str
    target: str
    split: str
    metrics: Dict[str, float]
    n_unpaired_proposed: int
    n_unpaired_baseline: int


def pair_scenarios(
    proposed: TrainingData,
    baseline: TrainingData,
    *,
    require_full_overlap: bool = True,
) -> tuple[pd.Index, int, int]:
    """
    Return the row ids present in both scenarios, plus the counts dropped
    from each side.

    Buildings are matched on the row_id index ('name'). A building simulated in
    one scenario but not the other cannot contribute to savings.
    """
    if proposed.usecase_id != baseline.usecase_id:
        raise ValueError(
            f"Cannot pair different usecases: '{proposed.usecase_id}' vs "
            f"'{baseline.usecase_id}'."
        )
    if proposed.split != baseline.split:
        raise ValueError(
            f"Cannot pair different splits: '{proposed.split}' vs '{baseline.split}'."
        )

    shared = proposed.row_ids.intersection(baseline.row_ids)

    n_unpaired_proposed = len(proposed.row_ids.difference(shared))
    n_unpaired_baseline = len(baseline.row_ids.difference(shared))

    if len(shared) == 0:
        raise ValueError(
            f"usecase '{proposed.usecase_id}': proposed and baseline share no row "
            f"ids, so savings cannot be computed."
        )

    if require_full_overlap and (n_unpaired_proposed or n_unpaired_baseline):
        raise ValueError(
            f"usecase '{proposed.usecase_id}' split '{proposed.split}': "
            f"{n_unpaired_proposed} proposed and {n_unpaired_baseline} baseline "
            f"rows have no counterpart. Pass require_full_overlap=False to drop them."
        )

    # preserve proposed ordering rather than the intersection's arbitrary order
    return proposed.row_ids[proposed.row_ids.isin(shared)], n_unpaired_proposed, n_unpaired_baseline


def compute_savings(
    *,
    proposed_data: TrainingData,
    baseline_data: TrainingData,
    proposed_model: FittedModel,
    baseline_model: FittedModel,
    target: Optional[str] = None,
    require_full_overlap: bool = True,
) -> SavingsResult:
    """
    Compute observed and predicted savings per building.

        savings = baseline_energy - proposed_energy

    Predictions for both scenarios are made from the proposed inputs, which is
    what the baseline model was trained on (see schema.yaml scenario notes).
    """
    if proposed_data.scenario != "proposed" or baseline_data.scenario != "baseline":
        raise ValueError(
            "Expected proposed_data.scenario == 'proposed' and "
            f"baseline_data.scenario == 'baseline', got "
            f"'{proposed_data.scenario}' and '{baseline_data.scenario}'."
        )

    target = target or proposed_data.target_names[0]
    for name, data in (("proposed", proposed_data), ("baseline", baseline_data)):
        if target not in data.target_names:
            raise KeyError(
                f"Target '{target}' is not in the {name} dataset "
                f"(has {data.target_names})."
            )

    rows, n_unpaired_p, n_unpaired_b = pair_scenarios(
        proposed_data, baseline_data, require_full_overlap=require_full_overlap
    )

    proposed_pred = proposed_model.predict(proposed_data.X.loc[rows])
    baseline_pred = baseline_model.predict(baseline_data.X.loc[rows])

    frame = pd.DataFrame(
        {
            "proposed_true": proposed_data.y.loc[rows, target],
            "proposed_pred": proposed_pred.loc[rows, target],
            "baseline_true": baseline_data.y.loc[rows, target],
            "baseline_pred": baseline_pred.loc[rows, target],
        }
    )
    frame["savings_true"] = frame["baseline_true"] - frame["proposed_true"]
    frame["savings_pred"] = frame["baseline_pred"] - frame["proposed_pred"]

    return SavingsResult(
        frame=frame,
        usecase_id=proposed_data.usecase_id,
        target_set=proposed_data.recipe.target_set,
        target=target,
        split=proposed_data.split,
        metrics=regression_metrics(frame["savings_true"], frame["savings_pred"]),
        n_unpaired_proposed=n_unpaired_p,
        n_unpaired_baseline=n_unpaired_b,
    )


# ---------------------------------------------------------------
# both scenarios in one call
# ---------------------------------------------------------------

@dataclass(frozen=True)
class UsecaseResult:
    """
    Both scenario models for one usecase, plus savings on the test split.
    """
    usecase_id: str
    target_set: str
    model_type: str
    models: Dict[str, FittedModel]        # scenario -> model
    savings: Optional[SavingsResult] = None


def train_usecase_with_savings(
    batch_index: pd.DataFrame,
    *,
    ctx: PipelineContext,
    usecase_id: str,
    target_set: str,
    model_type: str,
    factory: Optional[ConfigEstimatorFactory] = None,
    savings_split: str = "test",
    require_full_overlap: bool = True,
) -> UsecaseResult:
    """
    Fit the proposed and baseline models for one usecase and evaluate savings.
    """
    factory = factory or ConfigEstimatorFactory(ctx.model_config.estimators)

    models: Dict[str, FittedModel] = {}
    for scenario in ("proposed", "baseline"):
        models[scenario] = train_usecase(
            batch_index,
            ctx=ctx,
            usecase_id=usecase_id,
            scenario=scenario,
            target_set=target_set,
            model_type=model_type,
            factory=factory,
        )

    def data_for(scenario: str) -> TrainingData:
        return stage_dataset(
            batch_index,
            ctx=ctx,
            usecase_id=usecase_id,
            scenario=scenario,
            split=savings_split,
            target_set=target_set,
            model_type=model_type,
        )

    savings = compute_savings(
        proposed_data=data_for("proposed"),
        baseline_data=data_for("baseline"),
        proposed_model=models["proposed"],
        baseline_model=models["baseline"],
        require_full_overlap=require_full_overlap,
    )

    return UsecaseResult(
        usecase_id=usecase_id,
        target_set=target_set,
        model_type=model_type,
        models=models,
        savings=savings,
    )
