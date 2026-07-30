from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from ml179d.io.artifacts import (
    metrics_rows,
    model_dir,
    save_metrics,
    save_model,
    save_savings,
    savings_row,
)
from ml179d.serving import build_manifest, save_manifest
from ml179d.pipeline import PipelineContext, stage_catalog, stage_scan
from ml179d.selection import (
    ALL,
    DEFAULT_MODEL_TYPE,
    SCENARIOS,
    Job,
    JobPlan,
    Selection,
    format_plan,
    resolve_jobs,
)
from ml179d.train import train_usecase, train_usecase_with_savings

"""
Command line entry point.

    ml179d scan       raw CSVs -> batch index
    ml179d catalog    batch index -> per-usecase coverage report
    ml179d train      fit models, evaluate, compute savings

Selection happens here rather than in yaml, because which subset you run
changes every invocation. Defaults:

    --model-type   ridge_poly   (not every configured model type)
    --scenario     all          (both, which also computes savings)
    --target-set   all

Paths default to a repo-root layout (configs/, data/, outputs/) resolved from
the working directory, never from __file__, so an installed package behaves the
same as a checkout.
"""


# ---------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------

def _csv_list(value: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--configs", type=Path, default=Path("configs"),
                        help="directory holding schema.yaml, usecase_space.yaml, model.yaml")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--raw-dir", type=Path, default=None,
                        help="defaults to <data-root>/raw")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--cache", type=Path, default=None,
                        help="parquet batch index cache; defaults to "
                             "<data-root>/interim/batch_index.parquet")


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--usecase", action="append", default=[], type=str,
                        help="usecase id; repeatable or comma separated")
    parser.add_argument("--building-type", type=_csv_list, default=(),
                        help="raw or slugged, comma separated")
    parser.add_argument("--system-type", type=_csv_list, default=())
    parser.add_argument("--climate-zone", type=_csv_list, default=())
    parser.add_argument("--scenario", type=str, default=ALL,
                        help=f"{'|'.join(SCENARIOS)}|{ALL} (default: {ALL}; "
                             f"both scenarios also computes savings)")
    parser.add_argument("--target-set", type=str, default=ALL,
                        help=f"electricity|natural_gas|{ALL} (default: {ALL})")
    parser.add_argument("--model-type", type=str, default=DEFAULT_MODEL_TYPE,
                        help=f"model type, or '{ALL}' for every configured type "
                             f"(default: {DEFAULT_MODEL_TYPE})")


def raw_dir_of(args: argparse.Namespace) -> Path:
    return args.raw_dir if args.raw_dir is not None else args.data_root / "raw"


def cache_of(args: argparse.Namespace) -> Path:
    if args.cache is not None:
        return args.cache
    return args.data_root / "interim" / "batch_index.parquet"


def load_context(args: argparse.Namespace) -> PipelineContext:
    return PipelineContext.load(
        schema_path=args.configs / "schema.yaml",
        usecase_space_path=args.configs / "usecase_space.yaml",
        model_config_path=args.configs / "model.yaml",
    )


def selection_from_args(args: argparse.Namespace, ctx: PipelineContext) -> Selection:
    usecase_ids: List[str] = []
    for entry in args.usecase:
        usecase_ids.extend(_csv_list(entry))

    scenarios = SCENARIOS if args.scenario == ALL else _csv_list(args.scenario)

    if args.target_set == ALL:
        target_sets: Tuple[str, ...] = tuple(ctx.model_config.target_sets)
    else:
        target_sets = _csv_list(args.target_set)

    if args.model_type == ALL:
        model_types: Tuple[str, ...] = tuple(ctx.model_config.estimators)
    else:
        model_types = _csv_list(args.model_type)

    return Selection(
        usecase_ids=tuple(usecase_ids),
        building_types=args.building_type,
        system_types=args.system_type,
        climate_zones=args.climate_zone,
        scenarios=scenarios,
        target_sets=target_sets,
        model_types=model_types,
    )


# ---------------------------------------------------------------
# scan
# ---------------------------------------------------------------

def cmd_scan(args: argparse.Namespace) -> int:
    ctx = load_context(args)
    cache = cache_of(args)

    batch_index = stage_scan(
        raw_dir_of(args),
        ctx=ctx,
        cache_path=cache,
        force=args.force,
        strict=args.strict,
    )

    print(f"Scanned {len(batch_index)} batch file(s) -> {cache}")
    print(f"{batch_index['usecase_id'].nunique()} distinct usecase(s) found.")
    return 0


# ---------------------------------------------------------------
# catalog
# ---------------------------------------------------------------

def cmd_catalog(args: argparse.Namespace) -> int:
    ctx = load_context(args)

    batch_index = stage_scan(
        raw_dir_of(args), ctx=ctx, cache_path=cache_of(args), force=False, strict=False
    )
    report = stage_catalog(
        batch_index,
        expected_usecase_ids=ctx.expected_usecase_ids(
            args.configs / "usecase_space.yaml"
        ),
    )

    print(f"Trainable usecases:   {len(report.catalog)}")
    print(f"Incomplete usecases:  {len(report.incomplete)}")
    print(f"Expected but absent:  {len(report.not_in_data)}")
    print(f"Present but excluded: {len(report.not_expected)}")

    if args.verbose:
        if not report.incomplete.empty:
            print("\nMissing batch slots:")
            for _, row in report.incomplete.iterrows():
                print(f"  {row['usecase_id']}: {row['missing_slots']}")
        if report.not_in_data:
            print("\nNo data for:")
            for usecase_id in report.not_in_data:
                print(f"  {usecase_id}")
        if report.not_expected:
            print("\nIn data but excluded by disallow rules:")
            for usecase_id in report.not_expected:
                print(f"  {usecase_id}")

    directory = args.output_dir / "metrics"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "catalog_coverage.csv"
    report.incomplete.to_csv(path, index=False)
    print(f"\nCoverage report -> {path}")

    return 0


# ---------------------------------------------------------------
# train
# ---------------------------------------------------------------

def build_plan(args: argparse.Namespace, ctx: PipelineContext) -> Tuple[JobPlan, pd.DataFrame]:
    batch_index = stage_scan(
        raw_dir_of(args), ctx=ctx, cache_path=cache_of(args), force=False, strict=False
    )
    report = stage_catalog(batch_index)

    plan = resolve_jobs(
        selection_from_args(args, ctx),
        resolver=ctx.resolver,
        expected_usecase_ids=ctx.expected_usecase_ids(
            args.configs / "usecase_space.yaml"
        ),
        available_usecase_ids=report.trainable_usecase_ids,
        known_target_sets=tuple(ctx.model_config.target_sets),
        known_model_types=tuple(ctx.model_config.estimators),
    )

    return plan, batch_index


def write_manifest(result, *, ctx: PipelineContext, output_dir: Path) -> Path:
    """
    Write the serving manifest next to a usecase's fitted models.
    """
    proposed = result.models["proposed"]

    building_type_slug, system_type_slug, climate_zone_slug = ctx.resolver.parse_id(
        result.usecase_id
    )

    # Must mirror stage_dataset exactly, including system_type_slug: a recipe
    # resolved differently here would describe a model that was never fitted.
    recipe = ctx.model_config.resolve(
        target_set=result.target_set,
        model_type=result.model_type,
        system_type_slug=system_type_slug,
        usecase_id=result.usecase_id,
        scenario="proposed",
    )

    categories = {}
    for name in ctx.schema.categorical_columns():
        estimator = proposed.estimator
        steps = getattr(estimator, "named_steps", {})
        if "encode" in steps:
            encoder = steps["encode"].named_transformers_["categorical"]
            columns = steps["encode"].transformers_[0][2]
            for column, values in zip(columns, encoder.named_steps["onehot"].categories_):
                categories[column] = [str(v) for v in values]
        break

    manifest = build_manifest(
        models=result.models,
        recipe=recipe,
        usecase_id=result.usecase_id,
        building_type_slug=building_type_slug,
        system_type_slug=system_type_slug,
        climate_zone_slug=climate_zone_slug,
        schema_units={n: c.unit for n, c in ctx.schema.columns.items() if c.unit},
        schema_dtypes={n: c.dtype for n, c in ctx.schema.columns.items() if c.dtype},
        categories=categories,
    )

    return save_manifest(
        manifest,
        directory=model_dir(
            output_dir,
            usecase_id=result.usecase_id,
            target_set=result.target_set,
            model_type=result.model_type,
        ),
    )


def run_job(
    job: Job,
    *,
    batch_index: pd.DataFrame,
    ctx: PipelineContext,
    output_dir: Path,
) -> Tuple[List[Dict[str, Any]], Optional[Path]]:
    """
    Run one job, persist its models, and return metric rows.
    """
    rows: List[Dict[str, Any]] = []
    savings_path: Optional[Path] = None

    if job.computes_savings:
        result = train_usecase_with_savings(
            batch_index,
            ctx=ctx,
            usecase_id=job.usecase_id,
            target_set=job.target_set,
            model_type=job.model_type,
            # Simulations fail per datapoint, so proposed and baseline rarely
            # lose the same buildings. Pair on the intersection and record the
            # unpaired counts rather than aborting the job.
            require_full_overlap=False,
        )
        for model in result.models.values():
            save_model(model, output_root=output_dir)
            rows.extend(metrics_rows(model))

        # The serving contract only makes sense with both scenarios present.
        write_manifest(result, ctx=ctx, output_dir=output_dir)

        if result.savings is not None:
            rows.append(savings_row(result.savings))
            savings_path = save_savings(
                result.savings, output_root=output_dir, model_type=job.model_type
            )
    else:
        for scenario in job.scenarios:
            model = train_usecase(
                batch_index,
                ctx=ctx,
                usecase_id=job.usecase_id,
                scenario=scenario,
                target_set=job.target_set,
                model_type=job.model_type,
            )
            save_model(model, output_root=output_dir)
            rows.extend(metrics_rows(model))

    return rows, savings_path


def cmd_train(args: argparse.Namespace) -> int:
    ctx = load_context(args)
    plan, batch_index = build_plan(args, ctx)

    print(format_plan(plan))

    if args.dry_run:
        print("\nDry run; nothing was fitted.")
        return 0

    if not plan.jobs:
        print("\nNothing to run.")
        return 1

    all_rows: List[Dict[str, Any]] = []
    failures: List[Tuple[Job, Exception]] = []

    print()
    for i, job in enumerate(plan.jobs, start=1):
        prefix = f"[{i}/{len(plan.jobs)}]"
        try:
            rows, _ = run_job(
                job, batch_index=batch_index, ctx=ctx, output_dir=args.output_dir
            )
            all_rows.extend(rows)
            print(f"{prefix} ok   {job.describe()}")
        except Exception as exc:  # keep going; report at the end
            failures.append((job, exc))
            print(f"{prefix} FAIL {job.describe()}\n        {type(exc).__name__}: {exc}")
            if args.fail_fast:
                break

    path = save_metrics(all_rows, output_root=args.output_dir)
    if path is not None:
        print(f"\nMetrics -> {path}")

    if failures:
        print(f"\n{len(failures)} job(s) failed:")
        for job, exc in failures:
            print(f"  {job.describe()}: {type(exc).__name__}: {exc}")
        return 1

    return 0


# ---------------------------------------------------------------
# entry point
# ---------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml179d",
        description="179D surrogate model pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="scan raw batch CSVs into a batch index")
    add_path_args(scan)
    scan.add_argument("--force", action="store_true", help="ignore the cache")
    scan.add_argument("--strict", action="store_true",
                      help="fail on CSVs that do not match the batch filename pattern")
    scan.set_defaults(func=cmd_scan)

    catalog = sub.add_parser("catalog", help="report per-usecase batch coverage")
    add_path_args(catalog)
    catalog.add_argument("--verbose", "-v", action="store_true")
    catalog.set_defaults(func=cmd_catalog)

    train = sub.add_parser("train", help="fit models and compute savings")
    add_path_args(train)
    add_selection_args(train)
    train.add_argument("--dry-run", action="store_true",
                       help="print the job plan without fitting anything")
    train.add_argument("--fail-fast", action="store_true",
                       help="stop at the first failing job")
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
