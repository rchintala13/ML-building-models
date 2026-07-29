# ML179D Surrogate Modeling Pipeline

This repository contains a scalable, schema-driven machine learning pipeline for building surrogate models based on EnergyPlus simulation data. The system is designed to support large-scale modeling across multiple building types, HVAC systems, and climate zones.

---

## 🚀 Overview

The pipeline consists of the following stages:

```
EnergyPlus CSVs
        ↓
Batch Scanner (index raw data)
        ↓
Batch Catalog (map usecases → batches)
        ↓
Feature Processing (schema-driven)
        ↓
Surrogate Model Training
        ↓
Model Outputs + Metrics
```

The system is fully configurable and driven by YAML files, enabling flexibility and maintainability as datasets and requirements evolve.

---

## 📂 Repository Structure

```
configs/
    schema.yaml                # Column definitions and aliases
    usecase_config.yaml        # Slug mappings for usecases
    usecase_space.yaml         # BT × ST × CZ combinations

data/
    raw/                       # EnergyPlus CSVs (gitignored)

outputs/
    models/                    # Trained models (gitignored)
    catalogs/                  # Cached batch index/catalog
    metrics/                   # Model evaluation outputs

src/ml179d/
    schema/
        types.py               # Schema data structures
        loader.py              # YAML → Schema objects
    io/
        batch_scanner.py       # Scan raw CSVs → batch index
        batch_catalog.py       # Build usecase → batch mapping
        csv_loader.py          # Load + map CSV columns
    usecases/
        generator.py           # Generate usecase IDs
        resolver.py            # Slug ↔ raw mapping
        types.py               # Usecase object
    models/
        surrogate_model.py     # ML model training logic

tests/
    unit/
    integration/
```

---

## 🧠 Key Concepts

### 1. Usecase

A **usecase** is defined as:

```
building_type + system_type + climate_zone with '_' as seperator
```

Each usecase requires **4 batches**:

* proposed_train
* proposed_test
* baseline_train
* baseline_test

---

### 2. Schema-Driven Design

All column mappings are defined in:

```
configs/schema.yaml
```

Each column has a **canonical name** and one raw EnergyPlus column names:

```yaml
window_u_factor:
  role: feature
  sources_by_scenario:
    baseline:
      - reporting_179_d.in_window_u_value_w_per_m_2_k_overall
    proposed:
      - reporting_179_d_support.in_window_u_value_w_per_m_2_k_overall_proposed
```

This allows:

* consistent feature naming
* flexibility when raw column names change
* separation of data and code

---

### 3. Column Maps

The schema is converted into two key objects:

#### UsecaseColumnMap

Used by `batch_scanner` to extract:

* building_type
* system_type
* climate_zone

#### FeatureColumnMap

Used by modeling pipeline to:

* map raw → canonical columns
* select features and targets

---

## 🔍 Batch Scanner

The batch scanner:

* parses filenames to extract:

  * batch number
  * scenario (proposed/baseline)
  * split (train/test)
* reads a single row from each CSV
* computes `usecase_id`
* builds a batch index

Cached at:

```
outputs/catalogs/batch_index.parquet
```

---

## 📊 Batch Catalog

The batch catalog maps:

```
usecase_id → {4 required batches}
```

Ensures each usecase has:

* proposed_train
* proposed_test
* baseline_train
* baseline_test

---

## 🏗 Training Pipeline

The main training workflow:

1. Load schema and resolver
2. Load or scan batch index
3. Build batch catalog
4. Loop through usecases
5. Train 4 models per usecase:

   * proposed electricity
   * proposed gas
   * baseline electricity
   * baseline gas
6. Save:

   * trained models
   * evaluation metrics

---

## ⚙️ Running the Pipeline

### Train all usecases

```bash
python train.py
```

### Force re-scan of raw data

```bash
python train.py --force-rescan
```

### Train subset of usecases

```bash
python train.py --usecases-file usecases_subset.txt
```

---

## 🧪 Testing

### Run all CI-safe tests

```bash
pytest -m "not data"
```

### Run integration tests (including data)

```bash
pytest -m integration
```

### Run raw data integrity tests

```bash
pytest -m data
```

---

## ✅ Integration Tests

The repo includes system-level validation tests to ensure:

* all generated usecases exist in raw data
* each usecase has all 4 required batches
* schema mappings are correct

Data-dependent tests are marked:

```python
@pytest.mark.data
```

These are excluded from CI.

---

## 🚫 Git Ignore Policy

The following are intentionally not tracked:

* `data/raw/` (large simulation data)
* `outputs/models/` (trained models)
* `outputs/catalogs/` (generated metadata)


## 🧩 Design Principles

* **Single source of truth** → schema.yaml
* **Separation of concerns** → scanner vs modeling vs schema
* **Scalability** → supports thousands of usecases
* **Reproducibility** → cached batch index and deterministic pipelines
* **Testability** → unit + integration + data validation

---

## 👤 Author

Developed as part of ML179D surrogate modeling pipeline for building energy systems.

---
