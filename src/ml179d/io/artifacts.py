from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import joblib
import pandas as pd

from ml179d.train import FittedModel, SavingsResult

"""
Artifacts
---------
Where trained models, metrics and savings land under outputs/.

    outputs/models/<usecase_id>/<target_set>/<model_type>/<scenario>.joblib
    outputs/models/<usecase_id>/<target_set>/<model_type>/<scenario>.json
    outputs/metrics/metrics.csv
    outputs/metrics/savings/<usecase_id>__<target_set>__<model_type>.csv

The sidecar json records the feature list and metrics next to each model, so a
persisted model can be audited without loading the pickle.
"""


def model_dir(output_root: Path, *, usecase_id: str, target_set: str, model_type: str) -> Path:
    return Path(output_root) / "models" / usecase_id / target_set / model_type


def save_model(
    model: FittedModel,
    *,
    output_root: Path,
) -> Path:
    """
    Persist one fitted model plus a json sidecar. Returns the joblib path.
    """
    directory = model_dir(
        output_root,
        usecase_id=model.usecase_id,
        target_set=model.target_set,
        model_type=model.model_type,
    )
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / f"{model.scenario}.joblib"
    joblib.dump(model.estimator, path)

    sidecar = {
        "usecase_id": model.usecase_id,
        "scenario": model.scenario,
        "target_set": model.target_set,
        "model_type": model.model_type,
        "feature_names": model.feature_names,
        "target_names": model.target_names,
        "n_train": model.n_train,
        "train_metrics": model.train_metrics,
        "test_metrics": model.test_metrics,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))

    return path


def load_model(path: Path):
    return joblib.load(Path(path))


def metrics_rows(model: FittedModel) -> List[Dict[str, Any]]:
    """
    Flatten a fitted model's metrics into one row per (target, split).
    """
    rows: List[Dict[str, Any]] = []

    for split, metrics in (("train", model.train_metrics), ("test", model.test_metrics)):
        for target, values in metrics.items():
            rows.append(
                {
                    "usecase_id": model.usecase_id,
                    "scenario": model.scenario,
                    "target_set": model.target_set,
                    "model_type": model.model_type,
                    "target": target,
                    "split": split,
                    **values,
                }
            )

    return rows


def savings_row(savings: SavingsResult) -> Dict[str, Any]:
    return {
        "usecase_id": savings.usecase_id,
        "scenario": "savings",
        "target_set": savings.target_set,
        "model_type": "",
        "target": savings.target,
        "split": savings.split,
        "n_unpaired_proposed": savings.n_unpaired_proposed,
        "n_unpaired_baseline": savings.n_unpaired_baseline,
        **savings.metrics,
    }


def save_metrics(rows: Sequence[Dict[str, Any]], *, output_root: Path) -> Optional[Path]:
    """
    Write the metrics table. Returns None when there is nothing to write.
    """
    if not rows:
        return None

    directory = Path(output_root) / "metrics"
    directory.mkdir(parents=True, exist_ok=True)

    path = directory / "metrics.csv"
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return path


def save_savings(savings: SavingsResult, *, output_root: Path, model_type: str) -> Path:
    """
    Write the per-building savings frame, indexed by row id.
    """
    directory = Path(output_root) / "metrics" / "savings"
    directory.mkdir(parents=True, exist_ok=True)

    path = (
        directory
        / f"{savings.usecase_id}__{savings.target_set}__{model_type}.csv"
    )
    savings.frame.to_csv(path)
    return path
