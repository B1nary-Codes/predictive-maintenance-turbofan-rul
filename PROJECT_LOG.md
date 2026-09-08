# NASA C-MAPSS Turbofan Engine RUL Prediction Pipeline

## Project Overview
This repository contains an end-to-end machine learning and time-series engineering pipeline for predicting the Remaining Useful Life (RUL) of high-bypass turbofan engines using the NASA C-MAPSS dataset. The system ingests raw sensor telemetry, executes low-variance channel pruning, engineers time-series rolling features, and deploys an XGBoost regressor with Tree-SHAP interpretability for real-time maintenance risk assessment.

* **Domain:** Aerospace Predictive Maintenance, Machine Learning, Reliability Engineering
* **Primary Tools:** Python (Pandas, NumPy, XGBoost, Scikit-Learn), SHAP, VS Code, Git/GitHub
* **Key Methodologies:** GroupKFold Cross-Validation, Piecewise Target Clipping, Tree-SHAP Explainability
* **Development Log:** See [`PROJECT_LOG.md`](./PROJECT_LOG.md) for chronological technical updates and architectural decisions.

---

## Technical Specifications & Constraints
* **Dataset:** NASA C-MAPSS FD001 (20,631 operational flight cycles across 100 engine units)
* **Feature Dimensions:** 26 raw columns pruned to 19 active channels, expanded to 51 time-series features
* **Target Definition:** Piecewise linear RUL capped at 125 cycles ($RUL_{\text{piecewise}} = \min(125, T_{\text{max}} - t)$)
* **Validation Metrics:** Out-of-sample Test RMSE: 18.12 cycles | Test MAE: 12.76 cycles
* **Inference Alert Schema:** Critical ($RUL \le 20$), Warning ($20 < RUL \le 50$), Healthy ($RUL > 50$)

---

## Repository Structure
```text
predictive-maintenance-turbofan-rul/
├── README.md
├── PROJECT_LOG.md
├── Engineering_Report.pdf
├── data/                # NASA C-MAPSS raw text files
├── models/              # Serialized XGBoost model artifacts
├── src/
│   ├── data_loader.py   # Schema mapping, ingestion, and variance pruning
│   ├── features.py      # Rolling aggregations and piecewise target clipping
│   ├── train.py         # GroupKFold cross-validation and XGBoost training
│   ├── evaluate_test.py # Final cycle extraction and out-of-sample test evaluation
│   ├── explain.py       # Tree-SHAP interpretability and feature attribution
│   └── predict.py       # Production CLI inference pipeline
└── results/             # SHAP summary plots and evaluation metrics
