# NASA C-MAPSS Predictive Maintenance: Engineering Log

## Phase 1: Infrastructure & Repository Architecture
* **What Was Done:** Provisioned a cloud-based development environment using GitHub Codespaces with a modular folder hierarchy (`src/`, `notebooks/`, `data/raw/`, `data/processed/`).
* **Why:** Eliminates local OS/dependency mismatch issues, ensures full Linux environment reproducibility, and keeps project artifacts strictly organized.
* **Interview Takeaway:** *"I structured the project using production-grade software engineering patterns—separating data ingestion (`src/data_loader.py`), feature generation (`src/features.py`), and modeling (`src/train.py`) to prevent codebase duplication and simplify CI/CD migration."*

---

## Phase 2: Telemetry Ingestion & Schema Enforcement
* **What Was Done:** Ingested raw `train_FD001.txt` data via `pd.read_csv` using whitespace regex delimiters (`sep=r"\s+"`) and assigned explicit headers (2 Index/Time, 3 Operational Settings, 21 Sensor Channels).
* **Why:** The raw NASA C-MAPSS dataset is unformatted without headers. Explicit column mapping enables deterministic feature indexing.
* **Interview Takeaway:** *"Raw telemetry datasets frequently lack structured metadata. I enforced strict schema definition early in the pipeline to prevent downstream data-type mismatches."*

---

## Phase 3: Variance Profiling & Low-Variance Channel Pruning
* **What Was Done:** Evaluated feature variance ($\sigma$) across all 24 telemetry parameters and dropped 7 zero-variance channels: `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`.
* **Why:** Sensors with $\sigma < 10^{-4}$ exhibit flatline outputs across all 20,631 operational cycles. Retaining static channels introduces dimensional noise without adding predictive signal, unnecessarily increasing model complexity.
* **Interview Takeaway:** *"I performed variance profiling to eliminate non-informative channels early, reducing dataset dimensionality from 26 to 19 clean features without losing active thermodynamic signals."*

---

## Phase 4: Degradation Target Construction
* **What Was Done:** Calculated initial Remaining Useful Life ($RUL_{i,t} = T_{i,\text{max}} - t$) grouped by engine unit (`unit_nr`).
* **Why:** Transform raw run-to-failure cycle sequences into a continuous regression target representing time-to-failure.
* **Interview Takeaway:** *"Because the training data consists of run-to-failure runs, I calculated absolute RUL per engine unit to establish the ground-truth target for supervised learning."*

---

## Phase 5: Time-Series Feature Engineering & Target Clipping
* **What Was Done:** 
  1. Generated 10-cycle rolling averages (`_mov_avg`) and rolling standard deviations (`_mov_std`) grouped by `unit_nr`, adding 30 engineered feature channels.
  2. Applied piecewise linear target clipping at $RUL = 125$ cycles.
* **Why:** 
  * **Rolling Metrics:** Raw sensors contain high-frequency noise from transient throttle changes. Moving averages capture long-term wear trends, while moving standard deviations quantify increasing signal jitter/vibration as failure nears. Grouping by `unit_nr` prevents window bleed across engine boundaries.
  * **Piecewise Clipping:** Real machinery exhibits zero thermodynamic degradation during early flight cycles. Capping target $RUL$ at 125 prevents regression algorithms from trying to fit false degradation trends on early baseline noise.
* **Interview Takeaway:** *"To isolate wear drift from operational noise, I engineered 10-cycle moving window statistics and applied a piecewise linear target cap at 125 cycles to reflect early-stage component health accurately."*

---

## Architecture Matrix Summary

| Phase | Input Shape | Output Shape | Primary Operation | Key Module |
| :--- | :--- | :--- | :--- | :--- |
| **01. Ingestion** | Raw text file | `(20631, 26)` | Schema Mapping & Parsing | `src/data_loader.py` |
| **02. Pruning** | `(20631, 26)` | `(20631, 19)` | Variance Filtering ($\sigma < 10^{-4}$) | `src/data_loader.py` |
| **03. RUL Target** | `(20631, 19)` | `(20631, 20)` | Grouped Maximum Cycle Calculation | `src/data_loader.py` |
| **04. Feature Eng.** | `(20631, 20)` | `(20631, 51)` | Rolling Window Stats + RUL Clipping | `src/features.py` |

---

## Phase 6: Experiment Tracking Log

| Exp ID | Model Architecture | Features Used | Validation RMSE | Validation MAE | Key Takeaway / Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Random Forest (100 Trees) | Active + 10-cycle MovAvg/Std | TBD | TBD | Baseline GroupKFold cross-validation run. |