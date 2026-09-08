# Engineering Log: NASA C-MAPSS RUL Prediction

## System Architecture Log

### 01. Code Base Decoupling
To keep processing and training modular, code is isolated into single-responsibility scripts under `src/`:
* `data_loader.py` — Schema definition and ingestion
* `features.py` — Window statistics and target capping
* `train.py` — Model training with grouped cross-validation
* `evaluate_test.py` — Out-of-sample evaluation on final engine cycles
* `explain.py` — SHAP feature attribution
* `predict.py` — CLI for inference and risk classification

All file paths use Python's `pathlib` for cross-platform execution.

### 02. Data Ingestion & Schema
Mapped 26 unformatted space-delimited text columns into 3 operational settings and 21 thermodynamic sensor channels (`FD001`).

### 03. Channel Pruning
Standard deviation profiling identified 7 non-informative static channels ($\sigma < 10^{-4}$): `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, and `sensor_19`. Dropping these reduced matrix width from 26 to 19 columns without removing dynamic wear signals.

### 04. Piecewise Target Clipping
Calculated raw target values as $RUL_{i,t} = T_{i,\text{max}} - t$ grouped by `unit_nr`. Applied piecewise linear capping at 125 cycles ($RUL = \min(125, T_{\text{max}} - t)$) to prevent early-life predictions from fitting non-existent wear on healthy engines.

### 05. Time-Series Feature Construction
Engineered 10-cycle rolling aggregations for active sensor streams grouped strictly by `unit_nr`:
* **Rolling Mean:** Filters high-frequency operational noise to capture sensor drift.
* **Rolling Std Dev:** Captures physical vibration and thermal variance near failure boundaries.

Grouping by unit ID ensures rolling windows do not cross engine boundaries.

### 06. Validation Strategy (`GroupKFold`)
Standard random $K$-fold causes temporal data leakage across nearby cycles of the same engine. Evaluated models using 5-fold `GroupKFold` split by `unit_nr`, isolating full engine lifespans to test true out-of-sample generalization.

### 07. Out-of-Sample Test Evaluation
Tested models on `test_FD001.txt` using only the final recorded flight cycle per engine (`groupby("unit_nr").last()`) against ground truth targets (`RUL_FD001.txt`).

### 08. Experiment Tracking

| Exp ID | Model Architecture | Features | Val RMSE | Val MAE | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Random Forest (100 Trees) | Active Sensors + 10-cycle Rolling Stats | 19.33 | 13.87 | Baseline model ($\pm 0.49$ CV std). |
| **EXP-02** | XGBoost Regressor | Active Sensors + 10-cycle Rolling Stats | 18.61 | 13.54 | Residual fitting improved RMSE by ~0.72 cycles. |
| **EXP-03** | XGBoost (Final Production) | Final Cycle per Test Engine | **18.12** | **12.76** | Evaluated on out-of-sample test set. |

### 09. Model Interpretability (Tree-SHAP)
Extracted global feature attributions using Tree-SHAP:
* Moving averages of high-pressure turbine outlet temperature (`sensor_4`), discharge pressure (`sensor_11`), and bypass ratio (`sensor_15`) drive over 60% of model decisions.
* Rolling standard deviation features (`_mov_std`) show positive SHAP values in late-stage engine life, confirming the model uses signal jitter as an indicator of imminent failure.

### 10. CLI Inference & Thresholding
`src/predict.py` ingests engine logs, applies feature transformations, matches column ordering against `model.feature_names_in_`, and maps RUL predictions to operational risk tiers:
* $RUL \le 20$: **CRITICAL**
* $20 < RUL \le 50$: **WARNING**
* $RUL > 50$: **HEALTHY**
