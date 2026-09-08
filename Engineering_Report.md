# Engineering Report: Predictive Maintenance Analytics Pipeline for Turbofan Remaining Useful Life (RUL) Estimation

Author: Christian Sultan (B1nary-Codes)

Dataset: NASA C-MAPSS Telemetry (FD001)

Domain: Predictive Maintenance & Aerospace Condition-Based Monitoring

## Executive Summary:

Unscheduled component maintenance and in-flight engine shutdowns represent significant cost and safety risks in aerospace operations. Traditional preventive maintenance relies on fixed operational interval servicing, which frequently leads to premature part replacement or undetected structural degradation.

This engineering deliverable presents an end-to-end Machine Learning Engineering pipeline designed to estimate the Remaining Useful Life (RUL) of turbofan engines using high-dimensional thermodynamic sensor telemetry. Using low-variance channel pruning, temporal rolling-window feature extraction, piecewise target formulation, and gradient boosted regression (XGBoost), the system achieves an out-of-sample Test RMSE of 18.12 flight cycles and Test MAE of 12.76 flight cycles. Model decisions were verified using Tree-SHAP interpretability analysis, confirming alignment with known physical turbine thermodynamic failure modes.

## 2. Dataset Architecture & Preprocessing

The NASA C-MAPSS FD001 dataset contains simulated run-to-failure telemetry across 100 engine units operating under sea-level conditions. Raw inputs consist of space-delimited text streams mapping 3 operational settings and 21 thermodynamic sensor channels across 20,631 total flight cycles.

`Raw Telemetry Stream (26 Cols) ──> Variance Profiling ──> Active Feature Matrix (19 Cols)`

### 2.1 Low-Variance Channel Pruning

Profiling standard deviations across all 20,631 historical training cycles revealed seven static channels with negligible variance ($\sigma < 10^{-4}$): setting_3, sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, and sensor_19. These streams represent locked control variables or uncalibrated sensor channels. Pruning these zero-variance attributes reduced matrix dimensions from 26 to 19 base attributes without degrading downstream signal density or predictive capacity.

### 2.2 Piecewise Linear Target Formulation

Unclipped linear targets ($RUL_{i,t} = T_{i,\text{max}} - t$) force regression algorithms to estimate structural decay during initial operational states where mechanical components experience zero physical wear. To match true component degradation physics, targets were capped at a upper threshold of $125$ cycles:$$RUL_{\text{piecewise}} = \min(125, T_{i,\text{max}} - t)$$Capping target values prevents the model from fitting non-existent wear signatures on healthy engines, sharpening accuracy near critical operational failure boundaries.

## 3. Feature Engineering & Validation Strategy

### 3.1 Time-Series Feature Extraction

High-frequency operational noise was isolated from structural degradation signals using 10-cycle window rolling aggregations:

- Rolling Mean: Low-pass filter isolating underlying thermodynamic component drift.
- Rolling Standard Deviation: Measures thermal instability and sensor jitter near failure boundaries.

Aggregations are calculated strictly within groupby("unit_nr") blocks to prevent window leakage across adjacent engine boundaries.

## 3.2 Leakage-Free Validation (GroupKFold)

Standard cross-validation randomly shuffles rows, introducing severe temporal data leakage between adjacent cycles of identical engine units. To evaluate true fleet generalization, a 5-fold GroupKFold validation strategy grouped by unit_nr was enforced. Entire engine lifespans were isolated into distinct train or validation folds, guaranteeing out-of-sample evaluation on unseen engine units.

## 4. Experimental Results & Discussion

Model performance was evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE). Quadratic penalization in RMSE heavily weights large over-predictions, mitigating catastrophic operational maintenance risks.

### Performance Benchmarks

| Experiment | Algorithm | Feature Matrix | CV RMSE | CV MAE | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EXP-01** | Random Forest | Active Sensors + 10-Cycle Stats | 19.33 | 13.87 | — | — |
| **EXP-02** | XGBoost Regressor | Active Sensors + 10-Cycle Stats | 18.61 | 13.54 | — | — |
| **EXP-03** | XGBoost (Production) | Final Test Cycle per Engine | — | — | **18.12** | **12.76** |

XGBoost sequential error fitting outperformed ensemble bagging by 0.72 RMSE cycles. The close alignment between cross-validation RMSE (18.61) and out-of-sample test RMSE (18.12) validates zero feature leakage and confirms operational generalization.

## 5. Model Explainability & CLI Deployment
### 5.1 Tree-SHAP Physical Verification
Tree-SHAP attributions confirmed the model relies on true thermodynamic failure drivers:

Moving averages of high-pressure turbine outlet temperature (sensor_4), discharge pressure (sensor_11), and bypass ratio (sensor_15) account for >60% of total prediction weight.

Rolling standard deviation features (_mov_std) show strong positive SHAP value shifts during late engine lifespans, verifying that physical sensor jitter is utilized as a leading failure indicator.

> **High-Pressure Turbine Temp (Sensor 4)**  
> `████████████████████` 38.7%  
> **Discharge Pressure (Sensor 11)**  
> `██████████` 19.2%  
> **Bypass Ratio (Sensor 15)**  
> `█████` 9.8%  
> **Other Active Features**  
> `██████████████` 32.3%

### 5.2 Command-Line Operational Interface
The inference pipeline (src/predict.py) parses incoming raw unit telemetry, executes variance pruning and rolling transformations, aligns schema attributes against model.feature_names_in_, and maps predicted RUL to actionable risk tiers:
1. $RUL \le 20$: CRITICAL (Immediate Maintenance Required)
1. $20 < RUL \le 50$: WARNING (Schedule Fleet Inspection)
1. $RUL > 50$: HEALTHY (Normal Flight Operations)

## 6. Conclusion & Future Improvements
This project demonstrates a production-grade time-series engineering pipeline capable of estimating turbofan Remaining Useful Life within ~12.8 cycles on unseen operational engines. Key future enhancements include:

- Integrating sequence models (LSTM / Temporal Convolutional Networks) to capture complex long-range temporal dynamics.
  
- Extending support to multi-condition datasets (FD002–FD004) involving variable operational regimes and multiple failure modes.
