# instructions.md — Data-Driven Urban Rail Train Frequency and Capacity Optimization using TfL NUMBAT Data

> **This file is the operational manual of the project.** Read sections A–C before the
> first run, and section L before the viva.

---

## A. PROJECT OVERVIEW

### What the project does

This system is an **academic Operations Research decision-support tool**. It:

1. loads and validates an official **TfL NUMBAT** workbook,
2. standardises worksheets/columns through configurable schema detection,
3. performs exploratory demand & scheduled-service analysis,
4. reconstructs the rail network at line/link level (NetworkX),
5. compares passenger demand against scheduled train frequency,
6. solves **three optimisation models** (Pyomo + HiGHS),
7. runs demand / fleet / capacity **sensitivity scenarios**,
8. **validates** every solver result against the constraints,
9. produces a **baseline vs optimised** comparison,
10. serves everything through an interactive **Streamlit** dashboard,
11. exports CSV / JSON / PNG / HTML artefacts, and
12. writes a **run manifest** so every number in your slides can be reproduced.

### What each model represents

| Model | Name | Level | Objective | Represents |
|-------|------|-------|-----------|------------|
| **Model 1** | Minimum Service Frequency Optimization | service group × 15 min | **min Σ x** (integer trains) | The *cheapest feasible scheduled frequency* that still covers the busiest link on each service group, given capacity, frequency bounds and (optionally) a train-equivalent service budget. |
| **Model 2** | Capacity-Constrained Service Allocation | service group × 15 min | **max Σ y** (served line boardings) | How to *allocate a limited service budget* to maximise **served line-boarding demand** (line-level demand coverage). |
| **Model 3** | Weighted Goal Programming | service group × 15 min (+ per-link crowding goals) | **min w₁d⁻ᵘ + w₂dᶠ + w₃dᶜ + w₄Σx** | A *compromise* between four explicit goals: unmet line demand, target frequency, crowding above target occupancy, and service usage. Weights come from `config/goal_weights.yaml`. |

**What this project is NOT:** it is *not* an TfL operational timetable, *not* a real-time
control system, and *not* a rolling-stock circulation model. All optimised numbers are
**model recommendations under the configured assumptions**.

---

## B. DATA ACQUISITION

### Official source (only official TfL sources are used)

- **URL:** https://crowding.data.tfl.gov.uk/
- Publisher: Transport for London (TfL) — NUMBAT = **N**etwork **U**tilisation **M**anagement
  **B**usiness **A**nalysis **T**ool, the published annual rail patronage/crowding dataset.

### Which release to download

- Prefer **NUMBAT 2025**.
- Prefer the **Tuesday–Wednesday–Thursday (TWT)** weekday profile — this is the canonical
  “typical weekday” used across the project (see `preferred_day_type: "TWT"` and
  `day_type_filter` in `config/`).
- Download the **Excel (.xlsx) workbook** release.

> The software deliberately does **not** hardcode a filename. Any `.xlsx` you place in
> `data/raw/` (or point to with `DATA_FILE`) is accepted; detection is by content.

### Where to put it

```
project_root/
└── data/
    └── raw/          <-- put the downloaded NUMBAT .xlsx here
```

Files in `data/raw/` are **never modified** by this project.

### Optional: explicit path instead of data/raw

Windows PowerShell (current session):

```powershell
$env:DATA_FILE = "C:\path\to\numbat_2025.xlsx"
python scripts/profile_dataset.py
```

cmd.exe:

```bat
set DATA_FILE=C:\path\to\numbat_2025.xlsx
python scripts\profile_dataset.py
```

Or per invocation:

```powershell
python scripts\profile_dataset.py --data-file "C:\path\to\numbat_2025.xlsx"
```

### How to verify the downloaded file is correct

1. It opens in Excel and contains worksheets whose names/columns resemble
   `Link Load`, `Link Frequency`, `Line Boarders` (names vary by release — the
   profiler prints exactly what it found).
2. Run `python scripts/profile_dataset.py`. Success criteria:
   - roles `link_load`, `link_frequency`, `line_boarders` are all **detected**,
   - the quality summary shows 0 blocking issues (or you understand why not),
   - `data/processed/dataset_profile.json` is written.
3. Check the profile JSON: the `workbook` field shows your filename, `sheets[]` shows
   row/column counts, and `cleaned_summary.lines` lists the lines you expect.

If detection fails, the profiler prints every sheet and column it saw and **stops**
(it will not guess). Fix `config/schema_mapping.yaml` (section C.10) and rerun.

---

## C. MANUAL INTERVENTION — every input you must provide or verify

Nothing in this table is invented by the code. Missing **mandatory** inputs stop the
pipeline with an explicit error.

### C.1 NUMBAT workbook download  — **MANDATORY**

| Item | Value |
|------|-------|
| File path | `data/raw/*.xlsx` (any official NUMBAT workbook) |
| Meaning | The demand & scheduled-frequency dataset itself |
| Unit | passengers per 15-min period (Link Load, Line Boarders); trains per 15-min period (Link Frequency) |
| Where to get it | https://crowding.data.tfl.gov.uk/ |
| Mandatory? | **Yes** |
| If omitted | Profiler/app print download instructions and exit — no data is fabricated |

### C.2 capacity_per_train — **MANDATORY for all three models**

| Item | Value |
|------|-------|
| File path | `config/capacity_by_line.csv` |
| Columns | `mode, line, service_group, capacity_per_train, capacity_definition, source, source_url, notes` |
| Meaning | Passengers one train can carry on that line/service group |
| Unit | **passengers per train** (state whether crush or seated in `capacity_definition`) |
| Where to get it | TfL rolling-stock / fleet capacity statements, train manufacturer datasheets, DfT vehicle specs — anything citable. Record it in `source` + `source_url`. |
| Mandatory? | **Yes** for Models 1–3 |
| If omitted | Models refuse to run: `MISSING TRAIN CAPACITY for service groups: [...]`. Utilisation metrics show *“No capacity configured”* instead of numbers. Capacity is **never guessed**. |

`line` must match the labels printed in `dataset_profile.json → cleaned_summary.lines`.

### C.3 service-group / branch mapping — **OPTIONAL**

| Item | Value |
|------|-------|
| File path | `config/service_groups.csv` |
| Columns | `mode, line, service_group, from_station, to_station, notes` |
| Meaning | Splits a branching line into separate decision units (one frequency variable each) |
| Unit | n/a (topology) |
| Where to get it | NUMBAT link ordering + TfL route maps, if a line genuinely branches |
| Mandatory? | No. Default: `service_group = line` |
| If omitted | One decision variable per line. If branches exist, the dashboard warns and suggests configuring this file. |

### C.4 minimum frequency — **OPTIONAL**

| Item | Value |
|------|-------|
| File path | `config/service_constraints.csv` → `min_frequency_per_15min` |
| Meaning | Lower bound on x[r,t] (signalling minimum, service pledge, etc.) |
| Unit | trains per 15-minute period |
| Where to get it | Operator service standards / signalling headway studies — cite it in `notes` |
| Mandatory? | No |
| If omitted | No explicit lower bound beyond non-negativity (x ≥ 0) |

### C.5 maximum frequency — **OPTIONAL**

| Item | Value |
|------|-------|
| File path | `config/service_constraints.csv` → `max_frequency_per_15min` |
| Meaning | Upper bound on x[r,t] (turnback/signalling/platform capacity limit) |
| Unit | trains per 15-minute period |
| Where to get it | Infrastructure capability statements — cite in `notes` |
| Mandatory? | No |
| If omitted | No explicit upper bound (a configured budget still constrains) |
| Interaction | If `max < ceil(max_link_load/capacity)`, Model 1 becomes **infeasible** and explains exactly that. |

### C.6 fleet / service budget — **OPTIONAL (but required for a meaningful Model 2 trade-off)**

| Item | Value |
|------|-------|
| File path | `config/fleet_budget.csv` |
| Columns | `time_period, fleet_budget, source, notes` |
| Meaning | Cap on total train-equivalents scheduled across **all** service groups in that period |
| Unit | trains (train-equivalents) per 15-minute period |
| Where to get it | Your own assumption/sponsor input — **label it clearly** in `source` |
| Mandatory? | No. Empty file ⇒ budget constraint **disabled** (stated in every run’s warnings) |
| If omitted | Model 1/3 run without a budget; Model 2 allocates up to max-frequency bounds (warning issued; use `--require-budget` to make it strict) |
| ⚠ | This is a **simplified train-equivalent service budget**, NOT the TfL physical fleet and NOT a circulation plan. |

### C.7 target frequency — **OPTIONAL**

| Item | Value |
|------|-------|
| File path | `config/service_constraints.csv` → `target_frequency_per_15min` |
| Meaning | Goal-2 target in Model 3 |
| Unit | trains per 15-minute period |
| Where to get it | Planned service levels (or leave blank) |
| Mandatory? | No |
| If omitted | Target defaults to the **observed NUMBAT scheduled frequency** (documented in `src/models/common.py`) |

### C.8 Goal-Programming weights — **MANDATORY for Model 3 (defaults provided)**

| Item | Value |
|------|-------|
| File path | `config/goal_weights.yaml` |
| Keys | `w1_unmet_demand`, `w2_frequency_target`, `w3_crowding`, `w4_service_usage`, `normalize_terms` |
| Meaning | Relative importance of the four goals |
| Unit | dimensionless (with `normalize_terms: true` the four sums are divided by documented scales first, making weights comparable) |
| Where to get it | Stakeholder/judgement — **state and justify your choice in the report** |
| Mandatory? | Model 3 will not run if keys are missing (`ConfigError`) |
| Editable in UI? | Yes — *Parameters* section of the dashboard |
| If changed | Every run snapshot saves the exact weights used |

### C.9 target utilisation — **MANDATORY for Model 3 crowding goal (default provided)**

| Item | Value |
|------|-------|
| File path | `config/scenario_parameters.yaml` → `target_utilization` (override: `goal_weights.yaml → target_utilization_override`, CLI `--target-utilization`, dashboard slider) |
| Meaning | Acceptable occupancy level: utilisation = load / (capacity × frequency) ≤ target |
| Unit | ratio (0–1+; e.g. 0.80 = 80 % of capacity) |
| Where to get it | Service quality policy / crowding standard — cite your choice |
| Mandatory? | Default 0.80 ships in config; you should confirm it suits your study |
| If omitted/invalid | Model 3 raises a clear error if outside (0, 2] |

### C.10 schema mapping (only if auto-detection fails) — **CONDITIONAL**

| Item | Value |
|------|-------|
| File path | `config/schema_mapping.yaml` |
| Meaning | Sheet/column aliases, day-type filter, default mode, line/station alias normalisation |
| When to edit | Only when the profiler prints `SCHEMA DETECTION FAILED` with the real names |
| Rule | Add the **real** sheet/column names printed in the error to the relevant `aliases` list, then rerun. Never guess a mapping. |

### C.11 Day-type filter — **VERIFY**

| Item | Value |
|------|-------|
| File path | `config/schema_mapping.yaml` → `day_type_filter`, `config/scenario_parameters.yaml` → `preferred_day_type` |
| Meaning | Which day profile to analyse (TWT weekday profile by default) |
| If the filter matches nothing | All day types are kept and a warning is recorded in the profile — inspect `day_types_present`. |

### C.12 Scenario parameters — **OPTIONAL (defaults provided)**

| Item | Value |
|------|-------|
| File path | `config/scenario_parameters.yaml` |
| Keys | `baseline/increased/high_demand_multiplier`, `fleet_multiplier`, `reduced_fleet_multiplier`, `target_utilization`, `target_utilization_scenarios`, `peak_windows`, `solver` |
| Meaning | Multipliers and thresholds for the sensitivity analysis |
| If omitted | File ships with 1.00 / 1.10 / 1.20 / 0.85 / 0.80 defaults |

---

## D. DATA DICTIONARY

### NUMBAT variables (dataset — observed/scheduled facts)

| Field | Meaning | Unit | Used by |
|-------|---------|------|---------|
| `link_load` | **Link Load** — passengers between consecutive stations in a 15-min period | passengers / period | capacity constraints (M1, M3-G3), crowding, EDA |
| `link_frequency` | **Link Frequency** — *scheduled* trains per 15-min period on that link (**planned supply, not observed operation**) | trains / period | baseline comparison, baseline utilisation |
| `line_boarders` | **Line Boarders** — boardings on a line | passengers / period | Model 2 objective, Model 3 Goal 1, line-level EDA |
| `station` + `boarders/alighters/entry/exit` | station-level activity | passengers / period | exploratory analysis only |
| `mode` | transport mode label (e.g. Underground) | – | filtering |
| `line` | line name | – | grouping |
| `from_station`, `to_station` | consecutive station pair | – | network edges |
| `direction` | direction label (if present) | – | topology/branch analysis |
| `day_type` | day profile (e.g. TWT) | – | filtering (preserved) |
| `time_period` | canonical 15-min label `HH:MM-HH:MM` (raw label kept in `time_raw`) | – | indexing |
| `service_group` | decision unit (line by default, or configured mapping) | – | all models |

### Project configuration (assumptions — externally supplied)

| Field | File | Unit | Notes |
|-------|------|------|-------|
| `capacity_per_train` | capacity_by_line.csv | passengers/train | **must have a source** |
| `min/max/target_frequency_per_15min` | service_constraints.csv | trains/15 min | optional bounds |
| `fleet_budget` | fleet_budget.csv | train-equivalents/15 min | simplified budget, optional |
| `w1..w4` | goal_weights.yaml | dimensionless | GP priorities |
| `target_utilization` | scenario_parameters.yaml | ratio | crowding threshold |
| `*_demand_multiplier`, `fleet_multiplier` | scenario_parameters.yaml | × | scenario scalars |

### Derived fields

| Field | Definition |
|-------|------------|
| `baseline_utilization` | `link_load / (capacity_per_train × link_frequency)` — NaN if capacity missing |
| `optimized_utilization` | `scaled_link_load / (capacity_per_train × x)` |
| `demand_satisfaction` (M1) | `min(1, capacity·x / scaled_link_load)` averaged/attained per group-period |
| `coverage_ratio` (M2/M3) | `y / D` (served line boardings ÷ line boardings) |
| `unmet_demand` | `D − y` |
| `analytical_frequency` | `ceil(max_link_load × multiplier / capacity)` clipped to bounds |
| `crowded links` | links with `utilisation > target_utilization` under the solution |

---

## E. ASSUMPTIONS

1. NUMBAT represents a **typical day** of demand (TWT profile), not every day.
2. `Link Frequency` is **scheduled/planned** supply, not trains actually operated.
3. Train frequency decisions `x[r,t]` are **integers** (trains are indivisible).
4. **Capacity per train is an external, sourced input** — never inferred from NUMBAT.
5. A **service group** is a consistent service-frequency entity: every link in the group
   shares one frequency decision. Default group = line.
6. The fleet constraint is a **simplified train-equivalent service budget**, not a
   rolling-stock circulation, crew, or depot plan.
7. Demand is treated as **deterministic** (no stochastic uncertainty unless you extend it).
8. `link_load × demand_multiplier` scales demand proportionally (uniform shock).
9. Summed link loads are **link-passenger exposures, not unique journeys** — OD flows are
   never reconstructed from link sums.
10. Model 2/3 demand is **line boardings** (line-level coverage), not unique OD journeys.
11. Peak windows (`peak_windows` in scenario config) are used for **EDA only**, never inside
    the optimisers.
12. Where min/max/budget are unspecified, the corresponding constraint is **absent**, and the
    run records that fact in its warnings/manifest — absence is not silently filled.
13. Goal-3 crowding is modelled at **link level** against `target_utilization × capacity × x`.
14. Model 3 weighted-sum returns **one compromise point** on the Pareto front; it does not
    prove a unique “true optimum”.

---

## F. KNOWN LIMITATIONS

- **Typical-day data:** NUMBAT is a typical weekday profile, not a specific date.
- **Not real-time:** no live train positions, no disruption states.
- **Scheduled ≠ actual frequency:** improvements compare against the schedule, not realised
  operations.
- **Capacity/fleet assumptions:** results are only as credible as your sourced capacities
  and budget; crush vs seated capacity materially changes conclusions.
- **Simplified service-group model:** branches must be configured manually; no through-routing
  or interchange effects between groups.
- **No detailed train circulation:** no unit rotations, no depot/turnback paths, no stabling.
- **No crew constraints:** no driver/depot rostering, no break allowances.
- **No infrastructure limits beyond configured bounds:** no platform-length, power-supply,
  or signalling-block modelling unless encoded as min/max frequency.
- **No disruption/stochastic modelling:** demand multipliers are deterministic sensitivities.
- **No equity/objective beyond the four GP goals:** walk-access, interchange penalties,
  accessibility and fares are out of scope.

---

## G. INSTALLATION

Requires **Python 3.11+**.

### Windows PowerShell

```powershell
cd "D:\College\Sem7\OR\Open Project"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### cmd.exe

```bat
cd /d "D:\College\Sem7\OR\Open Project"
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Generic (any OS)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify the solver

```powershell
python -c "import pyomo.environ as pyo; o=pyo.SolverFactory('appsi_highs'); print('HiGHS OK')"
```

Expected: `HiGHS OK` (HiGHS comes from the `highspy` package — free, no licence).

---

## H. FIRST RUN — exact sequence

```powershell
# 1. download the official NUMBAT workbook from https://crowding.data.tfl.gov.uk/
#    (prefer NUMBAT 2025, TWT weekday profile)

# 2. put the file in data/raw\
#    e.g. data\raw\Numbat_2025.xlsx

# 3. run the profiler (lists sheets, detects schema, writes the quality profile)
python scripts\profile_dataset.py

# 4. inspect the profile
#    - open data\processed\dataset_profile.json
#    - check: detected roles, quality summary, lines, day types, 15-min gaps
#    - if SCHEMA DETECTION FAILED: fix config\schema_mapping.yaml (see C.10) and rerun

# 5. configure capacities  (MANDATORY — sourced, never guessed)
#    edit config\capacity_by_line.csv  (see C.2)

# 6. configure constraints  (optional but recommended)
#    edit config\service_constraints.csv   (min/max/target frequency)
#    edit config\fleet_budget.csv          (train-equivalent budget, or leave empty)
#    edit config\service_groups.csv        (only if a line branches)
#    edit config\goal_weights.yaml         (Model 3 priorities)
#    edit config\scenario_parameters.yaml  (multipliers, target utilisation)

# 7. preprocess (writes data\processed\cleaned_data.parquet)
python scripts\preprocess.py

# 7b. exploratory figures + tables (optional but recommended)
python scripts\run_eda.py

# 8. run Model 1
python scripts\run_model1.py

# 9. run Model 2
python scripts\run_model2.py

# 10. run Model 3
python scripts\run_model3.py

# 11. run scenarios
python scripts\run_scenarios.py --model model1

# 12. launch the dashboard
streamlit run app.py
```

Useful filters on every model script:

```powershell
python scripts\run_model1.py --mode Underground --lines Victoria Central `
    --demand-multiplier 1.1 --target-utilization 0.75 --fleet-multiplier 0.9
```

Generate the HTML report at any point:

```powershell
python scripts\generate_report.py
```

Run the test suite:

```powershell
python -m pytest tests -q
```

---

## I. DASHBOARD

```powershell
streamlit run app.py
```

Then open the printed URL (default http://localhost:8501).

| Section | What it shows |
|---------|---------------|
| **Data Overview** | dataset, year, day types, modes/lines/stations/links/periods, detected schema, quality report, network stats, configuration status, source URL |
| **Data Exploration** | demand heatmap & time series, line boarders, peak/off-peak, frequency time series, demand-vs-frequency scatter, line comparison, top links, utilisation heatmap, underserved periods — all interactive Plotly |
| **Optimization** | model selection, the mathematical formulation of the selected model, **Run optimisation** button (sidebar) |
| **Parameters** | mode/line/time scope, demand multiplier, fleet multiplier, target utilisation, min/max/target frequency tables, capacity table, budget table, GP weights |
| **Results** | baseline vs optimised frequency, change by service group, summary metrics (train-equivalents, average/peak frequency, satisfaction, unmet demand, utilisation, up/down counts), full table, CSV download |
| **Scenarios** | baseline / +10 % / +20 % / reduced fleet / utilisation sweeps with deltas vs baseline and CSV download |
| **Validation** | solver status, every constraint check, infeasibility explanations, warnings |
| **Export** | results CSV, model-run JSON, validation JSON/TXT, figure files, list of reproducible run directories |

Every optimisation run from the dashboard writes
`outputs/runs/YYYYMMDD_HHMMSS_ui_<model>/` containing `config_snapshot/`,
`results.csv`, `validation_report.json`, `dataset_profile.json`, `model_parameters.json`
and `run_metadata.json`.

---

## J. TROUBLESHOOTING

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NUMBAT WORKBOOK NOT FOUND` | no `.xlsx` in `data/raw/`, no `DATA_FILE` | Section B: download and place the file, or set `DATA_FILE` |
| `SCHEMA DETECTION FAILED` listing sheets/columns | release uses different sheet/column names | Add the printed names to `aliases` in `config/schema_mapping.yaml`; rerun profiler |
| `Required sheet/role '...' could not be identified` | a needed worksheet is absent/renamed | Same as above; verify the workbook really contains Link Load / Frequency / Line Boarders |
| `MISSING TRAIN CAPACITY for service groups: [...]` | `capacity_by_line.csv` empty or line labels don’t match | Fill capacities with sourced values; copy **exact** line labels from `dataset_profile.json` |
| `Model 1 is INFEASIBLE ... max_frequency ... needs >= N` | frequency ceiling below capacity requirement | Raise `max_frequency_per_15min`, review capacity source, or lower demand multiplier |
| `fleet_budget[t] is smaller than sum of minimum frequencies` | budget too tight | Raise/remove budget rows or reduce minimum frequencies |
| `Model 2 needs a train-equivalent service budget` with `--require-budget` | `fleet_budget.csv` empty | Add sourced budget rows, or drop the flag (documented as budget-disabled) |
| `No supported MILP solver found` | `highspy` missing/broken | `pip install highspy` ; verify with the command in section G |
| Duplicate rows reported | source contains repeated keys | Duplicates are **removed** (first kept) and counted in the profile — investigate the source if unexpected |
| Negative values reported | source has impossible negatives | Rows are **excluded** (never flipped/filled) and counted — fix the source |
| Null demand rows reported | missing loads/boardings | Rows are **excluded, never imputed** — see `null_demand_rows` in profile |
| `missing_15min_periods` in profile | a series skips 15-min slots | Inspect the affected line/day; data gaps are reported, not interpolated |
| `inconsistent_link_relationships` > 0 | same pair+period with conflicting direction | Check source direction labels; filter or correct at source |
| `Invalid line mappings` / model runs on wrong lines | line labels differ between config and data | Align config `line` values with profile line labels (aliases available in `schema_mapping.yaml`) |
| `ConfigError: ... Configuration file not found` | wrong path / renamed config file | Restore the file names under `config/` |
| Dashboard shows *“cleaned dataset not found”* | profiling/preprocessing not run yet | Run section H steps 3 and 7 |
| PNG export fails (kaleido/Chrome) | kaleido cannot find Chrome | HTML figures are still written (`outputs/figures/*.html`); or `pip install -U kaleido` with Chrome installed |
| `st.session_state` widget errors after editing app | widget key conflict | Restart `streamlit run app.py` (state is per-session) |
| Results look “zero/empty” | solver not optimal | Check the **Validation** section — infeasible runs show an explanation instead of zeros |

---

## K. HOW TO INTERPRET RESULTS

### Model 1 (`outputs/model1_results.csv`)

| Column | Interpretation |
|--------|----------------|
| `observed_frequency` | median **scheduled** frequency NUMBAT reports for that group-period (baseline) |
| `optimized_frequency` | MILP recommendation `x[r,t]` (integer trains / 15 min) |
| `frequency_change` | positive = model wants **more** service; negative = model indicates the schedule exceeds the capacity requirement **under the configured assumptions** (not a deployable cut) |
| `max_link_load` | busiest link demand on the group in that period (unscaled, for comparison) |
| `baseline_utilization` | `max_link_load / (capacity × observed_frequency)` |
| `optimized_utilization` | scaled load ÷ supplied capacity — > 1 means capacity still exceeded (only possible if bounds/budget forced it — check validation) |
| `demand_satisfaction` | fraction of the busiest-link demand covered by `capacity × x` (1.0 = fully covered) |
| `analytical_frequency` / `matches_analytical` | closed-form `ceil(max/cap)` cross-check |
| `solver_status` | must be `optimal`/`feasible` to interpret the row |

### Model 2 (`outputs/model2_results.csv`)

| Column | Interpretation |
|--------|----------------|
| `demand` | line boardings (unscaled) for the group-period |
| `trains_allocated` | integer trains assigned under the budget |
| `demand_served` (y) | **served line-boarding demand** — line-level coverage, *not* unique journeys |
| `unmet_demand` | `D − y` |
| `coverage_ratio` | `y / D` (0–1) |
| `utilization` | `y / (capacity × x)` — how full allocated capacity is |
| `capacity_supplied` | `capacity × x` |

### Model 3 (`outputs/model3_results.csv`)

| Column | Interpretation |
|--------|----------------|
| `d_unmet_minus` | Goal-1 shortfall: passengers of line demand not served |
| `d_freq_minus` / `d_freq_plus` | Goal-2 under/over-shoot vs target frequency (trains) |
| `d_crowd_plus_group` | number of links on the group still above target occupancy |
| `target_frequency`, `demand_served`, `coverage_ratio` | as Model 2 |
| goal totals (in run metadata) | weighted-sum components — compare across weight settings |

### Scenarios (`outputs/scenario_results.csv`)

Each row is one scenario: `total_train_equivalents`, `demand_satisfaction`,
`unmet_demand`, `mean/max_utilization`, `crowded_links`, plus `delta_*_vs_baseline`
columns. Read them as **sensitivity**: how fragile is the baseline recommendation to
+10/+20 % demand, a smaller budget, or a different occupancy target?

### Validation (`outputs/validation_report.json|txt`)

`overall_passed: true` means: solver optimal, frequencies integer, capacity constraints
hold for **every link**, bounds/budget respected, `y ≤ D`, `y ≤ cap·x`, no negatives,
units consistent — and (Model 1) the analytical cross-check agrees or the difference is
explained by configured bounds. **Never present results with a failed report.**

### Before vs after metrics (dashboard *Results*)

- *total train-equivalent service* — sum of x over all group-periods vs baseline sum.
- *average / peak frequency* — across group-periods.
- *demand satisfaction / unmet demand* — coverage of the busiest links (M1) or line
  boardings (M2/M3).
- *max/mean utilisation* — occupancy of supplied capacity.
- *crowded links* — links above `target_utilization`.
- *groups needing increased service / could be reduced* — count of Δ > 0 and Δ < 0.

Reductions are **model recommendations under assumptions**, never operational cut plans.

---

## L. PRESENTATION / VIVA PREPARATION

**Q: What is NUMBAT?**
TfL’s Network Utilisation Management Business Analysis Tool — the published annual dataset
of rail demand and scheduled service metrics (link loads, frequencies, boardings, OD flows,
journey times) derived from ticketing/operational sources. Source: crowding.data.tfl.gov.uk.

**Q: Why was it selected?**
It is the only *official, public, consistent* TfL dataset that exposes link-level demand
**and** scheduled frequency together at a uniform 15-minute resolution — exactly the two
sides of a capacity-coverage optimisation.

**Q: What does Link Load mean?**
Passengers travelling between two *consecutive* stations in a 15-minute period. It is a
link-level exposure measure: summing across a journey double-counts passengers, so we never
call link sums “passenger journeys”.

**Q: What does Link Frequency mean?**
The *scheduled* number of trains serving that link per 15-minute period — **planned supply**,
not trains actually observed running.

**Q: Why Line Boarders in Model 2?**
Model 2 optimises *coverage of demand that presents to the line* (boardings). Boardings are
the demand a frequency decision on that line must serve; link loads would double count
transfers/through passengers and are reserved for capacity/crowding constraints (M1/M3-G3).

**Q: Why are capacities external inputs?**
NUMBAT contains no train capacities. Inventing them would fabricate TfL data. We require
sourced `capacity_per_train` values with `source`/`source_url` so every constraint is
auditable.

**Q: What assumptions are you making?**
Section E: typical-day demand, scheduled frequencies, integer trains, sourced capacities,
service-group = consistent frequency entity, deterministic demand, simplified budget.

**Q: Why is the fleet constraint simplified?**
A real fleet constraint needs circulation, maintenance, depot and crew rules — a separate
rolling-stock problem. We use an explicit **train-equivalent service budget** per period,
label it as such, and leave it optional/disabled by default.

**Q: LP vs MILP vs Goal Programming?**
- *LP*: continuous variables, one linear objective — would allow “0.7 trains”.
- *MILP*: integer `x` (trains are indivisible) + continuous `y` where sensible; solved by
  branch-and-cut (HiGHS).
- *Goal Programming*: multiple conflicting goals become explicit deviation variables with
  weights; you optimise a *weighted compromise*, not a single physical quantity.

**Q: What does sensitivity analysis mean here?**
Re-solving identical models under shifted parameters (demand ×1.10/×1.20, budget ×0.85,
target occupancy 0.70/0.80/0.90) to show which conclusions are robust and which are
parameter-driven.

**Q: What does “optimal” mean in this model?**
*Mathematically*: HiGHS proves no feasible solution has a better objective under *these*
constraints, data and weights. *Operationally*: it is only as good as the sourced capacity,
bounds and budget — it is **not** an TfL timetable.

**Q: Limitations and future work?**
Section F. Natural extensions: stochastic/robust demand, interchange-aware assignment,
full rolling-stock circulation and crew constraints, disruption scenarios, multi-objective
Pareto frontiers instead of single weights, calibration against observed crowding.

**Demo script (5 minutes):**
1. Data Overview → official source, day type, scale of the network.
2. Exploration → demand heatmap, frequency series, utilisation (state capacity source).
3. Optimization → show formulation, set multiplier 1.10, Run.
4. Results → baseline vs optimised, metrics, Δ by group.
5. Scenarios → +10/+20 %, reduced fleet, utilisation sweep.
6. Validation → all checks green; show an intentionally infeasible config once to
   demonstrate honest failure reporting.
7. Export → run manifest proving reproducibility.

---

## M. FINAL RESULTS CHECKLIST

Copy this before the presentation:

```
[ ] correct dataset year (NUMBAT 2025) — verified in Data Overview / profile JSON
[ ] correct day type (TWT weekday) — verified in day_type_filter_applied
[ ] official source documented (https://crowding.data.tfl.gov.uk/) in report & manifest
[ ] capacity sources documented (source + source_url columns filled for every line)
[ ] assumptions documented (section E reproduced in report/README)
[ ] optimization solver successful (solver_status = optimal/feasible on every run)
[ ] validation passed (overall_passed = true in validation_report.json)
[ ] baseline vs optimized comparison generated (dashboard Results + figure)
[ ] scenarios generated (outputs/scenario_results.csv)
[ ] figures generated (outputs/figures/)
[ ] dashboard working (all 8 sections render, Run optimisation succeeds)
[ ] results reproducible (outputs/runs/<ts>/ contains config_snapshot + metadata + hash)
[ ] no fabricated data/parameters anywhere (grep configs for placeholder values removed)
[ ] disclaimers shown (dashboard banner + report disclaimer present)
```

---

## Appendix — file map

```
config/          all manual inputs (capacities, bounds, budget, weights, schema, scenarios)
data/raw/        official NUMBAT workbook (never modified)
data/processed/  dataset_profile.json + cleaned_data.parquet (generated)
src/ingestion/   workbook reading, sheet/column detection
src/preprocessing/ cleaning pipeline + quality report
src/analysis/    EDA functions
src/network/     NetworkX reconstruction, branch/disconnection checks
src/models/      Model 1/2/3 + solver utilities + shared data prep
src/scenarios/   reusable scenario runner
src/validation/  mandatory validation reports
src/visualization/ Plotly figure builders
src/utils/       config, time parsing, manifests, IO
scripts/         profile, preprocess, EDA, run_model1/2/3, scenarios, report
tests/           pytest suite (TEST DATA only — never project results)
outputs/         figures/, runs/ (manifests), reports/, *_results.csv, validation_report.*
app.py           Streamlit dashboard
```
