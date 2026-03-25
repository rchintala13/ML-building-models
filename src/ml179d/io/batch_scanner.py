from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ml179d.schema.types import UsecaseColumnMap
from ml179d.usecases.types import Usecase
from ml179d.usecases.protocols import UsecaseResolverLike

# Example filename:
# batch1_proposed_training_2024_12_17_18_29_43_est.csv
FILENAME_RE = re.compile(
    r"^batch(?P<batch>\d+)_"
    r"(?P<scenario>proposed|baseline)_"
    r"(?P<split>training|testing)_"
    r"(?P<yyyy>\d{4})_(?P<mo>\d{2})_(?P<day>\d{2})_(?P<h>\d{2})_(?P<mi>\d{2})_(?P<s>\d{2})_"
    r"(?P<tz>[A-Za-z]+)\.csv$"
)

# batch number, scenario, and split are learnt from filename
@dataclass(frozen=True)
class BatchFileMeta:
    batch_number: int
    scenario: str          # proposed|baseline
    split: str             # train|test
    filepath: Path


def parse_batch_filename(filename: str, *, strict: bool = True) -> Optional[BatchFileMeta]:
    """
    Parse batch filename into structured metadata.

    Returns None if no match and strict=False.
    """
    m = FILENAME_RE.match(filename)
    if not m:
        if strict:
            raise ValueError(f"Unrecognized batch filename format: {filename}")
        return None

    batch_number = int(m.group("batch"))
    scenario = m.group("scenario")
    split_raw = m.group("split")
    split = "train" if split_raw == "training" else "test"

    return BatchFileMeta(
        batch_number=batch_number,
        scenario=scenario,
        split=split,
        filepath=Path(filename),
    )

def _read_usecase_constants_from_csv(
    csv_path: Path,
    *,
    scenario: str,
    colmap: UsecaseColumnMap,
) -> tuple[str, str, str]:
    """
    Read the 3 usecase-defining values from a CSV:
      - building_type
      - system_type
      - climate_zone

    Only reads the first row because these columns are assumed constant
    within a file.

    Parameters
    ----------
    csv_path:
        Path to the CSV file.
    scenario:
        "proposed" or "baseline", used to select the correct system_type raw column.
    colmap:
        Schema-derived raw column map.
    """
    scenario = scenario.lower().strip()
    if scenario not in {"proposed", "baseline"}:
        raise ValueError(f"Unexpected scenario: {scenario}")

    bt_col = colmap.building_type_raw
    cz_col = colmap.climate_zone_raw
    st_col = colmap.system_type_raw_by_scenario[scenario]

    df = pd.read_csv(csv_path, nrows=1)

    required_cols = [bt_col, st_col, cz_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"{csv_path.name} is missing required usecase columns: {missing_cols}"
        )

    building_type = str(df.iloc[0][bt_col])
    system_type = str(df.iloc[0][st_col])
    climate_zone = str(df.iloc[0][cz_col])

    return building_type, system_type, climate_zone


def scan_batches(
    raw_data_dir: Path,
    *,
    resolver: UsecaseResolverLike,
    colmap: UsecaseColumnMap,
    cache_path: Optional[Path] = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Scan the raw data directory and build a batch index DataFrame.

    The resulting DataFrame contains one row per batch CSV with:
      - batch_number
      - scenario
      - split
      - building_type
      - system_type
      - climate_zone
      - usecase_id
      - filepath

    Parameters
    ----------
    raw_data_dir:
        Directory containing raw EnergyPlus CSV batch files.
    resolver:
        Usecase resolver used to generate canonical usecase IDs.
    colmap:
        Schema-derived usecase column map.
    cache_path:
        Optional parquet path for saving/loading a cached batch index.
    force:
        If False and cache_path exists, load cached result instead of rescanning.

    Returns
    -------
    pd.DataFrame
    """
    raw_data_dir = Path(raw_data_dir)

    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not force:
            return pd.read_parquet(cache_path)

    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    csv_files = sorted(raw_data_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found under: {raw_data_dir}")

    rows: list[dict] = []

    for csv_path in csv_files:
        meta = parse_batch_filename(csv_path.name)

        building_type, system_type, climate_zone = _read_usecase_constants_from_csv(
            csv_path,
            scenario=meta.scenario,
            colmap=colmap,
        )

        usecase = Usecase(
            building_type=building_type,
            system_type=system_type,
            climate_zone=climate_zone,
        )

        rows.append(
            {
                "batch_number": meta.batch_number,
                "scenario": meta.scenario,
                "split": meta.split,
                "building_type": building_type,
                "system_type": system_type,
                "climate_zone": climate_zone,
                "usecase_id": usecase.id(resolver),
                "filepath": str(csv_path.resolve()),
            }
        )

    batch_index = pd.DataFrame(rows).sort_values(
        by=["usecase_id", "scenario", "split", "batch_number"]
    ).reset_index(drop=True)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        batch_index.to_parquet(cache_path, index=False)

    return batch_index