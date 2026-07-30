# ML179D Surrogate Modeling Pipeline

A scalable, schema-driven machine learning pipeline for building surrogate models from EnergyPlus simulation data. The system supports large-scale modeling across multiple building types, HVAC systems, and climate zones.

---

## 🚀 Overview

```
EnergyPlus CSVs
        ↓
Batch Scanner       index raw data           →  batch_index.parquet
        ↓
Batch Catalog       map usecases → batches   →  coverage report
        ↓
Dataset Builder     schema-driven (X, y)     →  canonical columns
        ↓
Training            per usecase / scenario   →  fitted models
        ↓
Savings             baseline − proposed      →  per-building results
```

Everything is driven by YAML in `configs/`. Selection of *what to run* happens on the command line; the YAML describes the project itself.

---

## ⚡ Quick Start

```bash
pip install -e .          # editable: source changes take effect immediately

ml179d scan               # raw CSVs  → data/interim/batch_index.parquet
ml179d catalog -v         # which usecases are trainable, what is missing
ml179d train --dry-run    # print the job plan without fitting
ml179d train              # fit everything trainable
```

> Install with `-e`. A plain `pip install .` copies a snapshot into
> site-packages and your edits will not be picked up.

---

## ⚙️ CLI

Three subcommands: `scan`, `catalog`, `train`.

### Common path options

| Option | Default | Meaning |
|---|---|---|
| `--configs` | `configs` | directory holding the three YAML files |
| `--data-root` | `data` | root for raw and interim data |
| `--raw-dir` | `<data-root>/raw` | EnergyPlus batch CSVs |
| `--output-dir` | `outputs` | models, metrics, savings |
| `--cache` | `<data-root>/interim/batch_index.parquet` | batch index cache |

Paths resolve from the working directory, never from `__file__`, so an installed package behaves the same as a checkout.

### `ml179d scan`

Parses batch filenames, reads one row per CSV to identify the usecase, and writes the batch index.

```bash
ml179d scan
ml179d scan --force        # ignore the cache and rescan
ml179d scan --strict       # fail on CSVs not matching the batch filename pattern
```

Non-matching CSVs are skipped by default, so stray exports in `data/raw/` do not abort the scan.

### `ml179d catalog`

Groups the batch index into per-usecase records and reports coverage.

```bash
ml179d catalog
ml179d catalog -v          # list the specific gaps
```

Reports four counts: trainable, incomplete (missing batch slots), expected but absent from the data, and present in the data but excluded by `disallow` rules. Writes `outputs/metrics/catalog_coverage.csv`.

### `ml179d train`

Resolves the selection into an explicit job list, then fits.

```bash
ml179d train                                     # all trainable usecases
ml179d train --dry-run                           # print the plan, fit nothing
ml179d train --usecase small_office_PSZ-HP_CZ5A
ml179d train --climate-zone 5A,7A --building-type SmallOffice
ml179d train --target-set electricity --model-type all
ml179d train --scenario proposed                 # single scenario, no savings
```

**Selection options**

| Option | Default | Notes |
|---|---|---|
| `--usecase` | all | usecase id; repeatable or comma separated |
| `--building-type` | all | raw (`SmallOffice`) or slug (`small_office`) |
| `--system-type` | all | raw or slug |
| `--climate-zone` | all | raw or slug |
| `--scenario` | `all` | `proposed`, `baseline`, or `all` |
| `--target-set` | `all` | `electricity`, `natural_gas`, or `all` |
| `--model-type` | **`ridge_poly`** | a model type, or `all` for every configured one |

Two defaults worth knowing:

* **`--model-type` defaults to `ridge_poly`**, not to every configured model. A bare `ml179d train` fits ridge_poly only, whether for one usecase or the whole space. Use `--model-type all` to sweep.
* **`--scenario all` (the default) also computes savings**, because savings require a proposed *and* a baseline model for the same buildings. Selecting a single scenario fits one model and skips savings.

Unknown usecase ids, target sets, and model types are rejected during planning, before anything is fitted. A job that fails at fit time is reported and the run continues; `--fail-fast` stops at the first failure. Exit codes: `0` success, `1` some jobs failed, `2` bad arguments or config.

---

## 📂 Repository Structure

```
configs/
    schema.yaml            raw → canonical column mappings
    usecase_space.yaml     BT × ST × CZ aliases + disallow rules
    model.yaml             features, filters, transforms, estimators

data/
    raw/                   EnergyPlus CSVs (gitignored)
    interim/               batch_index.parquet cache

outputs/
    models/<usecase_id>/<target_set>/<model_type>/<scenario>.joblib
                                                 /<scenario>.json
    metrics/metrics.csv
    metrics/savings/<usecase_id>__<target_set>__<model_type>.csv

src/ml179d/
    cli.py                 argparse entry point
    selection.py           CLI filters → explicit job list
    config.py              model.yaml → DatasetRecipe
    pipeline.py            scan / catalog / dataset stages
    train.py               fitting, evaluation, savings
    schema/
        types.py           ColumnSpec, Schema, column maps
        loader.py          schema.yaml → Schema
    io/
        batch_scanner.py   raw CSVs → batch index
        batch_catalog.py   usecase → 4 batch slots
        csv_loader.py      raw → canonical DataFrame
        artifacts.py       persistence under outputs/
    features/
        engineering.py     derived columns and transforms
        registry.py        yaml names → callables
    usecases/
        generator.py       usecase space + validity rules
        resolver.py        slug ↔ raw mapping
        types.py           Usecase object
    models/
        factory.py         model_type → unfitted estimator
        metrics.py         r2, mae, rmse, CV(RMSE), NMBE
        protocols.py       Estimator boundary

tests/
    conftest.py            miniature project fixture
    test_*.py
```

---

## 🧠 Key Concepts

### Usecase

```
building_type + '_' + system_type + '_' + climate_zone
```

e.g. `small_office_PSZ-HP_CZ5A`. Each usecase needs **4 batches**: `proposed_train`, `proposed_test`, `baseline_train`, `baseline_test`.

The full space is the cartesian product of the aliases in `usecase_space.yaml` (currently 2 × 6 × 16 = 192). Combinations that do not physically exist are removed by `disallow` rules:

```yaml
disallow:
  - building_type: SmallOffice
    system_type: VRF DOAS          # this pair, every climate zone
  - system_type: PSZ-AC with gas coil
    climate_zone: [1A, 2A, 2B]     # this system, these zones only
```

Every field named must match; omitted fields are wildcards. Values are **raw BEM values**, not slugs.

### Canonical vs raw vs slug

Three naming layers, easy to confuse:

* **raw** — the EnergyPlus column name, e.g. `reporting_179_d.in_floor_area_m_2`
* **canonical** — this project's stable column name, e.g. `gross_floor_area`
* **slug** — a normalized *value* used in usecase ids, e.g. `SmallOffice` → `small_office`

Canonical names are the keys under `columns:` in `schema.yaml`; raw names are its `sources`. The rename happens once in `csv_loader`, after which everything speaks canonical.

### Schema-driven columns

```yaml
window_u_factor:
  role: feature
  sources_by_scenario:
    proposed:
      - reporting_179_d.in_window_u_value_w_per_m_2_k_overall
    baseline:
      - reporting_179_d_support.in_window_u_value_w_per_m_2_k_overall_proposed
```

`sources` is a preference list — the first name present in the CSV wins — so a renamed column is a YAML edit, not a code change.

**Scenario semantics:** the `baseline` scenario reads the `*_proposed` input columns. Baseline models are trained on the same user inputs as proposed models, because only user inputs are available at prediction time. Use `baseline_prm` for the values actually used in baseline simulations.

### Savings

Savings are the difference between two models' predictions on the same inputs:

```
savings = baseline_model(X) − proposed_model(X)
```

Buildings are paired on the `name` row id, which is preserved as the DataFrame index throughout the pipeline. Duplicate or missing ids raise rather than silently misaligning results.

---

## 🔧 Configuration

### `model.yaml`

| Section | Purpose |
|---|---|
| `base_features` | derived columns computed before feature selection |
| `filters` | row-level range filters, with overrides |
| `target_sets` | `electricity`, `natural_gas` |
| `base_feature_sets` | starting feature list per target set |
| `system_overrides` | add/drop features by slugged system type |
| `estimators` | model kind + hyperparameters |
| `model_type_overrides` | transforms per model type |
| `usecase_overrides` | add/drop features by usecase id |

Resolution order for features: base set → system override → usecase override → model-type transforms.

### Filters

```yaml
filters:
  min_values:
    gross_floor_area: 400
  max_values: {}
  apply_to_test: false      # filter train only; score on the full test set
  overrides:
    - when: {target_set: natural_gas}
      min_values: {gross_floor_area: 500}
```

Applied after base features and before feature selection, so they can reference derived or unselected columns. Bounds merge per column; later matching rules win; `null` clears a bound.

> A filter that differs between proposed and baseline drops different buildings from each and breaks savings pairing. This raises rather than misaligning.

### Estimators

```yaml
ridge_poly:
  kind: ridge_poly
  params:
    degree: 2
    scaler: minmax
    alpha: 0.007
  overrides:
    natural_gas:
      proposed:
        alpha: 0.001
```

`overrides` is keyed by target set then scenario, because alpha was tuned per combination. The ridge_poly step order is `poly_features → MinMaxScaler → Ridge`; the tuned alpha assumes that order, so reordering requires retuning.

---

## 📊 Outputs

`outputs/metrics/metrics.csv` — one row per (scenario, split), plus a `savings` row:

| column | meaning |
|---|---|
| `n` | rows scored |
| `r2` | coefficient of determination |
| `mae` / `rmse` | absolute and squared error |
| `cvrmse_pct` | 100 × RMSE / mean(observed) |
| `nmbe_pct` | signed bias, ASHRAE Guideline 14 |
| `mean_observed` | denominator for the normalized metrics |

CV(RMSE) and NMBE are normalized, so they are comparable across usecases with very different absolute energy.

Each model is saved as `<scenario>.joblib` with a `<scenario>.json` sidecar recording the feature list and metrics, so a persisted model can be audited without unpickling it.

> `metrics.csv` is overwritten per run and contains only that run's jobs.

---

## 🧪 Testing

```bash
pytest              # full suite, no real data required
pytest tests/test_pipeline.py -v
```

Tests build a miniature project on disk (`tests/conftest.py`) with synthetic CSVs whose targets are exact linear functions of the features, so metric and savings assertions are deterministic. No BEM data is needed and nothing touches `data/`.

---

## 🧩 Design Principles

* **Single source of truth** → `schema.yaml` for columns, `model.yaml` for recipes
* **Config is data** → yaml names become code only in `features/registry.py` and `models/factory.py`
* **Separation of concerns** → scanning, dataset building, and modeling are independent stages
* **Paths are injected** → the package never guesses where data lives
* **Fail at plan time** → typos in config or selection raise before any model is fitted
* **Reproducibility** → cached batch index, deterministic pipelines, persisted feature lists

---

## 👤 Author

Developed as part of the ML179D surrogate modeling pipeline for building energy systems.
