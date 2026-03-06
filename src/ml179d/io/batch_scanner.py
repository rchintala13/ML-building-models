from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

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

def _read_usecase_constants_from_csv(csv_path: Path) -> tuple[str, str, str]:
    """
    Reads the constant columns (building_type, system_type, climate_zone) from the CSV.
    Assumes these columns exist and are constant within the file.
    Reads only a single row for speed.
    """
    df = pd.read_csv(csv_path, nrows=1)

    required = ["building_type", "system_type", "climate_zone"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path.name} missing required columns: {missing}")

    bt = str(df.loc[0, "building_type"])
    st = str(df.loc[0, "system_type"])
    cz = str(df.loc[0, "climate_zone"])
    
    return bt, st, cz