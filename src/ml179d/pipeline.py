from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ml179d.config import DatasetRecipe, FilterSpec, ModelConfig, validate_against_schema
from ml179d.features import engineering as eng
from ml179d.features.registry import (
    FeatureContext,
    apply_base_features,
    apply_transforms,
)
from ml179d.io.batch_catalog import (
    UsecaseBatchRecord,
    build_batch_catalog,
    build_missing_batch_report,
)
from ml179d.io.batch_scanner import scan_batches
from ml179d.io.csv_loader import load_canonical_batch_dataframe
from ml179d.schema import build_usecase_column_map, load_schema
from ml179d.schema.types import Schema
from ml179d.usecases.generator import UsecaseSpace, generate_usecase_ids, load_validity_rule
from ml179d.usecases.resolver import UsecaseResolver

"""
Pipeline
--------
Sequential stages, each consuming the previous stage's artifact:

    stage_scan     raw CSVs                -> batch_index DataFrame
    stage_catalog  batch_index             -> catalog + coverage report
    stage_dataset  batch_index + recipe    -> TrainingData (X, y)

Every stage takes explicit paths and returns in-memory objects; persistence is
the caller's decision. Nothing here fits a model -- stage_dataset is the
handoff point to the modeling layer (see models/protocols.py).
"""


# ---------------------------------------------------------------
# shared context
# ---------------------------------------------------------------

@dataclass(frozen=True)
class PipelineContext:
    """
    Config objects loaded once and reused across stages.
    """
    schema: Schema
    resolver: UsecaseResolver
    space: UsecaseSpace
    model_config: ModelConfig

    @staticmethod
    def load(
        *,
        schema_path: Path,
        usecase_space_path: Path,
        model_config_path: Path,
        validate: bool = True,
    ) -> "PipelineContext":
        schema = load_schema(Path(schema_path))
        resolver = UsecaseResolver.from_yaml(Path(usecase_space_path))
        space = UsecaseSpace.from_yaml(Path(usecase_space_path))
        model_config = ModelConfig.from_yaml(Path(model_config_path))

        if validate:
            validate_against_schema(model_config, schema)

        return PipelineContext(
            schema=schema,
            resolver=resolver,
            space=space,
            model_config=model_config,
        )

    def expected_usecase_ids(self, usecase_space_path: Path) -> List[str]:
        """
        The usecase IDs we intend to train, after the disallow filter.
        """
        validity_fn = load_validity_rule(Path(usecase_space_path), space=self.space)
        return generate_usecase_ids(
            space=self.space,
            resolver=self.resolver,
            validity_fn=validity_fn,
        )


# ---------------------------------------------------------------
# stage 1: scan
# ---------------------------------------------------------------

def stage_scan(
    raw_data_dir: Path,
    *,
    ctx: PipelineContext,
    cache_path: Optional[Path] = None,
    force: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Scan raw batch CSVs into a batch index.

    strict defaults to False here (unlike scan_batches) so that stray CSVs in
    data/raw do not abort the whole scan.
    """
    colmap = build_usecase_column_map(ctx.schema)
    return scan_batches(
        Path(raw_data_dir),
        resolver=ctx.resolver,
        colmap=colmap,
        cache_path=Path(cache_path) if cache_path else None,
        force=force,
        strict=strict,
    )


# ---------------------------------------------------------------
# stage 2: catalog
# ---------------------------------------------------------------

@dataclass(frozen=True)
class CatalogReport:
    """
    Which usecases are trainable, and what is missing for the rest.
    """
    catalog: Dict[str, UsecaseBatchRecord]
    incomplete: pd.DataFrame          # usecase_id, missing_slots
    not_in_data: List[str]            # expected but absent entirely
    not_expected: List[str]           # present in data but filtered out by disallow

    @property
    def trainable_usecase_ids(self) -> List[str]:
        return sorted(self.catalog)


def stage_catalog(
    batch_index: pd.DataFrame,
    *,
    expected_usecase_ids: Optional[List[str]] = None,
    strict: bool = False,
) -> CatalogReport:
    """
    Group the batch index into per-usecase 4-slot records.

    strict defaults to False so that a partially delivered dataset still yields
    a usable catalog; the gaps come back in the report instead of raising.
    """
    catalog = build_batch_catalog(batch_index, strict=strict)
    incomplete = build_missing_batch_report(batch_index)

    expected = set(expected_usecase_ids or [])
    found = set(batch_index["usecase_id"].unique())

    return CatalogReport(
        catalog=catalog,
        incomplete=incomplete,
        not_in_data=sorted(expected - found) if expected else [],
        not_expected=sorted(found - expected) if expected else [],
    )


def resolve_batch_path(
    batch_index: pd.DataFrame,
    *,
    usecase_id: str,
    scenario: str,
    split: str,
) -> Path:
    """
    Look up the CSV path for one (usecase, scenario, split) slot.

    UsecaseBatchRecord stores batch numbers only, so the batch index remains
    the authority on file locations.
    """
    match = batch_index[
        (batch_index["usecase_id"] == usecase_id)
        & (batch_index["scenario"] == scenario)
        & (batch_index["split"] == split)
    ]

    if match.empty:
        raise KeyError(
            f"No batch file for usecase='{usecase_id}' scenario='{scenario}' "
            f"split='{split}'."
        )
    if len(match) > 1:
        raise ValueError(
            f"Multiple batch files for usecase='{usecase_id}' scenario='{scenario}' "
            f"split='{split}': {sorted(match['filepath'])}"
        )

    return Path(match.iloc[0]["filepath"])


# ---------------------------------------------------------------
# stage 3: dataset
# ---------------------------------------------------------------

@dataclass(frozen=True)
class TrainingData:
    """
    The pipeline's output and the modeling layer's input.

    X and y are indexed by the row_id column ('name'). That index is load
    bearing: a building in the proposed batch has the same name in the baseline
    batch, and savings are computed by joining the two on it. Never reset it.
    """
    X: pd.DataFrame
    y: pd.DataFrame
    usecase_id: str
    scenario: str
    split: str
    recipe: DatasetRecipe
    # rows removed by schema.row_validity, i.e. failed simulation datapoints
    n_invalid_rows: int = 0

    @property
    def feature_names(self) -> List[str]:
        return list(self.X.columns)

    @property
    def target_names(self) -> List[str]:
        return list(self.y.columns)

    @property
    def row_ids(self) -> pd.Index:
        """
        Building identifiers, shared across scenarios.
        """
        return self.X.index

    def __len__(self) -> int:
        return len(self.X)


def _validate_row_index(df: pd.DataFrame, *, row_id_col: str, usecase_id: str, scenario: str) -> None:
    """
    The row index carries the building identity used to pair proposed with
    baseline. A missing or duplicated index would silently misalign savings,
    so both are hard errors rather than warnings.
    """
    if df.index.name != row_id_col:
        raise ValueError(
            f"usecase '{usecase_id}' scenario '{scenario}': expected the DataFrame "
            f"to be indexed by '{row_id_col}', got index name {df.index.name!r}."
        )

    if not df.index.is_unique:
        duplicates = df.index[df.index.duplicated()].unique().tolist()
        raise ValueError(
            f"usecase '{usecase_id}' scenario '{scenario}': duplicate {row_id_col} "
            f"values {duplicates[:5]}. Row ids must be unique to pair scenarios."
        )


def _apply_filters(df: pd.DataFrame, filters: Tuple[FilterSpec, ...]) -> pd.DataFrame:
    for spec in filters:
        df = eng.filter_by_feature_range(
            df,
            column=spec.column,
            min_value=spec.min_value,
            max_value=spec.max_value,
        )
    return df


def _feature_context(
    usecase_id: str,
    row: pd.Series,
    resolver: UsecaseResolver,
) -> FeatureContext:
    return FeatureContext(
        usecase_id=usecase_id,
        building_type_slug=resolver.to_slug("building_type", row["building_type"]),
        system_type_slug=resolver.to_slug("system_type", row["system_type"]),
        climate_zone_slug=resolver.to_slug("climate_zone", row["climate_zone"]),
    )


def stage_dataset(
    batch_index: pd.DataFrame,
    *,
    ctx: PipelineContext,
    usecase_id: str,
    scenario: str,
    split: str,
    target_set: str,
    model_type: str,
) -> TrainingData:
    """
    Build (X, y) for one usecase / scenario / split.

    Order of operations:
        1. load canonical DataFrame      raw column names resolved via schema
        2. base features                 derived columns (roof_area, etc.)
        3. filters                       row-level range filtering
        4. feature selection             base_feature_sets + overrides
        5. transforms                    model-type specific, may add/drop columns
        6. split into X / y

    Transforms run after selection because add_piecewise_feature drops its
    input column, which is itself a selected feature.
    """
    slot_rows = batch_index[
        (batch_index["usecase_id"] == usecase_id)
        & (batch_index["scenario"] == scenario)
        & (batch_index["split"] == split)
    ]
    if slot_rows.empty:
        raise KeyError(
            f"No batch file for usecase='{usecase_id}' scenario='{scenario}' "
            f"split='{split}'."
        )
    row = slot_rows.iloc[0]

    recipe = ctx.model_config.resolve(
        target_set=target_set,
        model_type=model_type,
        system_type_slug=ctx.resolver.to_slug("system_type", row["system_type"]),
        usecase_id=usecase_id,
        scenario=scenario,
    )

    csv_path = resolve_batch_path(
        batch_index, usecase_id=usecase_id, scenario=scenario, split=split
    )

    # 1. canonical DataFrame
    df = load_canonical_batch_dataframe(
        csv_path,
        schema=ctx.schema,
        scenario=scenario,
    )

    row_id_cols = ctx.schema.row_id_columns()
    if not row_id_cols:
        raise ValueError(
            "Schema defines no row_id column; scenarios cannot be paired for savings."
        )
    _validate_row_index(
        df, row_id_col=row_id_cols[0], usecase_id=usecase_id, scenario=scenario
    )

    # 2. base features
    df = apply_base_features(
        df,
        recipe.base_features,
        _feature_context(usecase_id, row, ctx.resolver),
    )

    # 3. filters -- train only unless the config opts test in
    if split == "train" or recipe.filters_apply_to_test:
        df = _apply_filters(df, recipe.filters)

    # 4. feature selection
    missing_features = [c for c in recipe.features if c not in df.columns]
    if missing_features:
        raise KeyError(
            f"usecase '{usecase_id}': features {missing_features} are not in the "
            f"loaded DataFrame. Either they are absent from the CSV, or they are "
            f"derived columns whose base feature function is not listed in "
            f"'base_features' in model.yaml."
        )
    missing_targets = [c for c in recipe.targets if c not in df.columns]
    if missing_targets:
        raise KeyError(
            f"usecase '{usecase_id}': targets {missing_targets} are not in the "
            f"loaded DataFrame."
        )

    selected = df[list(recipe.features) + list(recipe.targets)].copy()

    # 5. transforms
    selected = apply_transforms(selected, recipe.transforms)

    # 6. X / y
    target_cols = [c for c in recipe.targets if c in selected.columns]
    feature_cols = [c for c in selected.columns if c not in target_cols]

    return TrainingData(
        X=selected[feature_cols],
        y=selected[target_cols],
        usecase_id=usecase_id,
        scenario=scenario,
        split=split,
        recipe=recipe,
        n_invalid_rows=int(df.attrs.get("n_invalid_rows_dropped", 0)),
    )
