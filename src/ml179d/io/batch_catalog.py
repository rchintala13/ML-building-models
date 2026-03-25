from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import pandas as pd


REQUIRED_SLOTS = (
    "proposed_train",
    "proposed_test",
    "baseline_train",
    "baseline_test",
)


@dataclass(frozen=True, slots=True)
class UsecaseBatchRecord:
    """
    Stores the 4 required batches for one usecase.

    Each field contains the batch number.
    """
    proposed_train: int
    proposed_test: int
    baseline_train: int
    baseline_test: int

    def as_dict(self) -> dict[str, int]:
        return {
            "proposed_train": self.proposed_train,
            "proposed_test": self.proposed_test,
            "baseline_train": self.baseline_train,
            "baseline_test": self.baseline_test,
        }


def _validate_batch_index(batch_index: pd.DataFrame) -> None:
    required_cols = {
        "batch_number",
        "scenario",
        "split",
        "usecase_id",
        "filepath",
    }
    missing = required_cols - set(batch_index.columns)
    if missing:
        raise KeyError(
            f"batch_index is missing required columns: {sorted(missing)}"
        )

    valid_scenarios = {"proposed", "baseline"}
    bad_scenarios = set(batch_index["scenario"].unique()) - valid_scenarios
    if bad_scenarios:
        raise ValueError(
            f"Unexpected scenario values in batch_index: {sorted(bad_scenarios)}"
        )

    valid_splits = {"train", "test"}
    bad_splits = set(batch_index["split"].unique()) - valid_splits
    if bad_splits:
        raise ValueError(
            f"Unexpected split values in batch_index: {sorted(bad_splits)}"
        )


def _make_slot(scenario: str, split: str) -> str:
    return f"{scenario}_{split}"


def _prepare_batch_index(batch_index: pd.DataFrame) -> pd.DataFrame:
    """
    Return a normalized copy of the batch index with a 'slot' column.
    """
    _validate_batch_index(batch_index)

    df = batch_index.copy()
    df["scenario"] = df["scenario"].str.strip().str.lower()
    df["split"] = df["split"].str.strip().str.lower()
    df["slot"] = df.apply(
        lambda row: _make_slot(row["scenario"], row["split"]),
        axis=1,
    )

    return df


def build_batch_catalog(
    batch_index: pd.DataFrame,
    *,
    strict: bool = True,
) -> Dict[str, UsecaseBatchRecord]:
    """
    Build:
        usecase_id -> UsecaseBatchRecord

    Parameters
    ----------
    batch_index:
        DataFrame produced by batch_scanner.scan_batches()

    strict:
        If True, raise an error when:
          - a usecase is missing any of the 4 required batch slots
          - a usecase has duplicate rows for the same slot

        If False, incomplete usecases are skipped.

    Returns
    -------
    dict[str, UsecaseBatchRecord]
    """
    df = _prepare_batch_index(batch_index)

    catalog: Dict[str, UsecaseBatchRecord] = {}
    errors: list[str] = []

    for usecase_id, group in df.groupby("usecase_id"):
        slot_to_rows = {
            slot: slot_group
            for slot, slot_group in group.groupby("slot")
        }

        # Check missing slots
        missing_slots = [slot for slot in REQUIRED_SLOTS if slot not in slot_to_rows]
        if missing_slots:
            msg = (
                f"usecase '{usecase_id}' is missing required slots: {missing_slots}"
            )
            if strict:
                errors.append(msg)
            continue

        # Check duplicate slots
        duplicate_slots = [
            slot
            for slot, slot_group in slot_to_rows.items()
            if len(slot_group) > 1
        ]
        if duplicate_slots:
            msg = (
                f"usecase '{usecase_id}' has duplicate rows for slots: {duplicate_slots}"
            )
            if strict:
                errors.append(msg)
            continue

        record = UsecaseBatchRecord(
            proposed_train=int(slot_to_rows["proposed_train"].iloc[0]["batch_number"]),
            proposed_test=int(slot_to_rows["proposed_test"].iloc[0]["batch_number"]),
            baseline_train=int(slot_to_rows["baseline_train"].iloc[0]["batch_number"]),
            baseline_test=int(slot_to_rows["baseline_test"].iloc[0]["batch_number"]),
        )

        catalog[usecase_id] = record

    if errors:
        raise ValueError(
            "Batch catalog could not be built cleanly:\n" + "\n".join(errors)
        )

    return catalog


def batch_catalog_to_dataframe(
    catalog: Dict[str, UsecaseBatchRecord],
) -> pd.DataFrame:
    """
    Convert catalog dict into a DataFrame for saving or inspection.
    """
    rows = []
    for usecase_id, record in catalog.items():
        row = {"usecase_id": usecase_id}
        row.update(record.as_dict())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("usecase_id").reset_index(drop=True)


def find_missing_usecases(
    expected_usecase_ids: Iterable[str],
    catalog: Dict[str, UsecaseBatchRecord],
) -> list[str]:
    """
    Return expected usecases that are not present in the catalog.
    """
    expected = set(expected_usecase_ids)
    actual = set(catalog.keys())
    return sorted(expected - actual)


def build_missing_batch_report(batch_index: pd.DataFrame) -> pd.DataFrame:
    """
    Build a report of usecases missing one or more required batch slots.

    Returns a DataFrame with columns:
      - usecase_id
      - missing_slots
    """
    df = _prepare_batch_index(batch_index)

    rows = []

    for usecase_id, group in df.groupby("usecase_id"):
        slots_present = set(group["slot"].unique())
        missing_slots = [slot for slot in REQUIRED_SLOTS if slot not in slots_present]

        if missing_slots:
            rows.append(
                {
                    "usecase_id": usecase_id,
                    "missing_slots": ",".join(missing_slots),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["usecase_id", "missing_slots"])

    return pd.DataFrame(rows).sort_values("usecase_id").reset_index(drop=True)