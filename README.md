# Data-Driven Urban Rail Train Frequency and Capacity Optimization using TfL NUMBAT Data

Academic Operations Research decision-support system that turns official **TfL NUMBAT**
demand and scheduled-frequency data into validated, reproducible service-frequency
recommendations.

> **Not an TfL operational timetable.** All optimised values are model recommendations
> under the documented, user-editable assumptions. Capacities and budgets are never
> fabricated — they are mandatory/optional configuration inputs with sources.

## Problem statement

Urban rail operators must decide how many trains to schedule each 15-minute period so
that (a) demand fits within train capacity, (b) service stays within operational and
fleet limits, and (c) crowding, coverage and service-usage goals are balanced. NUMBAT
publishes link-level passenger loads and scheduled frequencies, but contains **no** train
capacities or fleet budgets — so the modelling layer must be explicit about every
externally supplied parameter.

## Methodology

1. **Ingest & validate** the official workbook (flexible sheet/column detection, quality
   report: duplicates, negatives, nulls, 15-min gaps, inconsistent links).
2. **Clean** into a canonical long-format table (originals preserved, raw data untouched).
3. **Reconstruct** the network (NetworkX) with branch/disconnection detection.
4. **Analyse** demand, frequency, utilisation (utilisation only where capacity is sourced).
5. **Optimise** with three Pyomo/HiGHS models, **validate** every constraint, run
   **scenarios**, and export a **reproducible run manifest**.

## Models

| Model | Objective | Key constraints |
|-------|-----------|-----------------|
| **1 — Minimum Service Frequency** (MILP) | min Σ x (integer trains) | capacity·x ≥ link demand × multiplier; min/max frequency; optional train-equivalent budget. Cross-checked against closed-form `ceil(max load / capacity)`. |
| **2 — Capacity-Constrained Allocation** (MILP) | max Σ y (served line-boarding demand) | y ≤ demand; y ≤ capacity·x; Σ x ≤ budget; bounds. Measures **line-level demand coverage**, not unique journeys. |
| **3 — Weighted Goal Programming** (MILP) | min w₁·unmet + w₂·freq-dev + w₃·crowding + w₄·Σx | explicit linear deviation variables; weights from `config/goal_weights.yaml`; crowding goal per link vs target occupancy. |

**Scenarios:** baseline, +10 % demand, +20 % demand, reduced service budget, and
target-utilisation sweeps — all rerun through the same model code.

## Technology stack

Python 3.11+ · pandas · NumPy · openpyxl · pyarrow · **Pyomo + HiGHS (highspy, free)** ·
NetworkX · Plotly · Streamlit · PyYAML · pytest

## Directory structure

```
├── app.py                 Streamlit dashboard (8 sections)
├── instructions.md        full manual: data, assumptions, viva prep (READ THIS)
├── config/                ALL manual inputs (CSV/YAML — no code edits needed)
├── data/raw/              official NUMBAT workbook goes here (never modified)
├── data/processed/        dataset_profile.json + cleaned_data.parquet
├── src/                   ingestion, preprocessing, analysis, network,
│                          models, scenarios, validation, visualization, utils
├── scripts/               profile · preprocess · run_eda · run_model1/2/3 ·
│                          run_scenarios · generate_report
├── tests/                 pytest suite (uses clearly-labelled TEST DATA only)
└── outputs/               figures/, runs/ (manifests), reports/, results, validation
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import pyomo.environ as pyo; print(pyo.SolverFactory('appsi_highs').available())"
```

## Execution

```powershell
# 1. download NUMBAT (https://crowding.data.tfl.gov.uk/, prefer 2025 TWT) into data\raw\
python scripts\profile_dataset.py      # detect schema + quality profile
# 2. fill config\capacity_by_line.csv   (MANDATORY, sourced)
python scripts\preprocess.py           # cleaned_data.parquet
python scripts\run_eda.py              # figures/tables
python scripts\run_model1.py           # min service frequency
python scripts\run_model2.py           # capacity-constrained allocation
python scripts\run_model3.py           # weighted goal programming
python scripts\run_scenarios.py        # sensitivity sweep
python scripts\generate_report.py      # HTML report
python -m pytest tests -q              # test suite
```

## Dashboard

```powershell
streamlit run app.py
```

Sections: **Data Overview · Data Exploration · Optimization · Parameters · Results ·
Scenarios · Validation · Export** — every run writes
`outputs/runs/<timestamp>/` with a config snapshot, results, validation report, dataset
profile hash and metadata so any slide number can be reproduced.

## Results

- `outputs/model1_results.csv`, `model2_results.csv`, `model3_results.csv`
- `outputs/scenario_results.csv` — scenario deltas vs baseline
- `outputs/validation_report.json|txt` — solver status + constraint checks
  (infeasible runs produce an **explanation**, never misleading zeros)
- `outputs/figures/` — demand/frequency/utilisation/before-after charts
- `outputs/runs/<ts>/` — full reproducibility manifest
- `outputs/reports/report_latest.html` — consolidated HTML report

## Limitations

Typical-day (TWT) data · scheduled (not actual) frequencies · externally sourced
capacities/budgets · simplified service-group model · no circulation/crew/depot/disruption
modelling · deterministic demand · goal programming returns one weighted compromise, not a
unique “true optimum”. Full list: `instructions.md` §F.
