from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ml179d.usecases.resolver import UsecaseResolver

"""
Selection
---------
Turns "what do I want to run" into an explicit list of jobs.

A job is one (usecase, target_set, model_type) plus the scenarios to fit. When
both scenarios are selected the job also computes savings, since savings need a
proposed and a baseline model for the same buildings.

Selection lives on the command line, not in yaml: which subset you are running
changes every invocation, while usecase_space.yaml and model.yaml describe the
project itself. In particular 'disallow' in usecase_space.yaml means "this
combination does not exist", which is not the same as "skip it today".
"""

DEFAULT_MODEL_TYPE = "ridge_poly"
ALL = "all"

SCENARIOS: Tuple[str, ...] = ("proposed", "baseline")


@dataclass(frozen=True, slots=True)
class Job:
    usecase_id: str
    target_set: str
    model_type: str
    scenarios: Tuple[str, ...]

    @property
    def computes_savings(self) -> bool:
        """
        Savings need both scenarios; a single-scenario job just fits one model.
        """
        return set(self.scenarios) == set(SCENARIOS)

    def describe(self) -> str:
        return (
            f"{self.usecase_id} | {self.target_set} | {self.model_type} | "
            f"{'+'.join(self.scenarios)}"
            f"{' | +savings' if self.computes_savings else ''}"
        )


@dataclass(frozen=True)
class Selection:
    """
    Raw selection from the command line. Empty/None means "everything".
    """
    usecase_ids: Tuple[str, ...] = ()
    building_types: Tuple[str, ...] = ()
    system_types: Tuple[str, ...] = ()
    climate_zones: Tuple[str, ...] = ()
    scenarios: Tuple[str, ...] = SCENARIOS
    target_sets: Tuple[str, ...] = ()
    model_types: Tuple[str, ...] = (DEFAULT_MODEL_TYPE,)


@dataclass(frozen=True)
class JobPlan:
    jobs: Tuple[Job, ...]
    requested_but_unavailable: Tuple[str, ...]
    filtered_out: int

    def __len__(self) -> int:
        return len(self.jobs)


def _axis_matches(
    slug: str,
    wanted: Sequence[str],
    *,
    kind: str,
    resolver: UsecaseResolver,
) -> bool:
    """
    Accept either the raw BEM value ('SmallOffice') or the slug ('small_office').
    """
    if not wanted:
        return True

    for value in wanted:
        if value == slug or resolver.to_slug(kind, value) == slug:
            return True
    return False


def _split_usecase(usecase_id: str, resolver: UsecaseResolver) -> Optional[Tuple[str, str, str]]:
    try:
        return resolver.parse_id(usecase_id)
    except ValueError:
        return None


def resolve_jobs(
    selection: Selection,
    *,
    resolver: UsecaseResolver,
    expected_usecase_ids: Iterable[str],
    available_usecase_ids: Optional[Iterable[str]] = None,
    known_target_sets: Sequence[str],
    known_model_types: Sequence[str],
) -> JobPlan:
    """
    Expand a selection into concrete jobs.

    expected_usecase_ids:
        the generated space after the disallow filter
    available_usecase_ids:
        usecases the batch catalog can actually supply; when given, anything
        missing is reported rather than attempted
    """
    expected = list(expected_usecase_ids)
    expected_set = set(expected)
    available = set(available_usecase_ids) if available_usecase_ids is not None else None

    target_sets = tuple(selection.target_sets) or tuple(known_target_sets)
    model_types = tuple(selection.model_types) or (DEFAULT_MODEL_TYPE,)

    for name, chosen, known in (
        ("target_set", target_sets, known_target_sets),
        ("model_type", model_types, known_model_types),
    ):
        unknown = [v for v in chosen if v not in known]
        if unknown:
            raise ValueError(
                f"Unknown {name}(s) {unknown}. Available: {sorted(known)}"
            )

    scenarios = tuple(s for s in SCENARIOS if s in set(selection.scenarios))
    if not scenarios:
        raise ValueError(
            f"No valid scenario selected. Expected any of {list(SCENARIOS)}."
        )

    # explicit ids win over axis filters
    if selection.usecase_ids:
        unknown = [u for u in selection.usecase_ids if u not in expected_set]
        if unknown:
            raise ValueError(
                f"Unknown usecase id(s) {unknown}. They are not in the generated "
                f"usecase space (check spelling, or the disallow rules)."
            )
        candidates = list(selection.usecase_ids)
        filtered_out = 0
    elif not (
        selection.building_types or selection.system_types or selection.climate_zones
    ):
        # no axis filters, so there is nothing to parse ids for
        candidates = list(expected)
        filtered_out = 0
    else:
        candidates = []
        for usecase_id in expected:
            parts = _split_usecase(usecase_id, resolver)
            if parts is None:
                raise ValueError(
                    f"Cannot parse usecase id '{usecase_id}' with the current "
                    f"aliases, so axis filters cannot be applied to it."
                )
            bt, st, cz = parts
            if (
                _axis_matches(bt, selection.building_types, kind="building_type", resolver=resolver)
                and _axis_matches(st, selection.system_types, kind="system_type", resolver=resolver)
                and _axis_matches(cz, selection.climate_zones, kind="climate_zone", resolver=resolver)
            ):
                candidates.append(usecase_id)
        filtered_out = len(expected) - len(candidates)

    if not candidates:
        raise ValueError(
            "The selection matched no usecases. Check the building type, system "
            "type and climate zone filters."
        )

    unavailable: List[str] = []
    runnable: List[str] = []
    for usecase_id in candidates:
        if available is not None and usecase_id not in available:
            unavailable.append(usecase_id)
        else:
            runnable.append(usecase_id)

    jobs = tuple(
        Job(
            usecase_id=usecase_id,
            target_set=target_set,
            model_type=model_type,
            scenarios=scenarios,
        )
        for usecase_id in runnable
        for target_set in target_sets
        for model_type in model_types
    )

    return JobPlan(
        jobs=jobs,
        requested_but_unavailable=tuple(unavailable),
        filtered_out=filtered_out,
    )


def format_plan(plan: JobPlan) -> str:
    """
    Human readable plan for --dry-run.
    """
    lines = [f"{len(plan.jobs)} job(s):"]
    lines.extend(f"  {job.describe()}" for job in plan.jobs)

    if plan.filtered_out:
        lines.append(f"\n{plan.filtered_out} usecase(s) excluded by the axis filters.")

    if plan.requested_but_unavailable:
        lines.append(
            f"\n{len(plan.requested_but_unavailable)} usecase(s) selected but not "
            f"available in the batch catalog:"
        )
        lines.extend(f"  {u}" for u in plan.requested_but_unavailable)

    return "\n".join(lines)
