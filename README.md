# NASA C-MAPSS Turbofan Predictive Maintenance

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)
![SHAP](https://img.shields.io/badge/Explainable_AI-SHAP-orange.svg)

An end-to-end Machine Learning Engineering pipeline predicting the Remaining Useful Life (RUL) of turbofan engines using time-series thermodynamic sensor telemetry from the NASA C-MAPSS dataset (`FD001`).

---

## 📌 Executive Summary

Modern aerospace operations rely on condition-based predictive maintenance to prevent catastrophic mid-flight component failure and optimize servicing schedules. This project builds a production-grade machine learning model that predicts engine RUL within **~12.8 flight cycles** on unseen out-of-sample engines.

### Key Performance Results
* **Cross-Validation RMSE:** `18.61 ± 0.57 cycles` (5-Fold GroupKFold)
* **Out-of-Sample Test RMSE:** `18.12 cycles`
* **Out-of-Sample Test MAE:** `12.76 cycles`

---

## 🛠 Project Architecture

predictive-maintenance-turbofan-rul/
├── assets/                  # High-resolution visual artifacts & SHAP plots
├── data/
│   ├── raw/                 # C-MAPSS text files (train_FD001, test_FD001, RUL_FD001)
│   └── processed/           # Filtered, rolling-window feature matrices
├── models/                  # Serialized binary model artifacts (.joblib)
├── src/
│   ├── data_loader.py       # Ingestion, schema validation, zero-variance pruning
│   ├── features.py          # 10-cycle rolling aggregations & piecewise linear RUL clipping
│   ├── train.py             # GroupKFold CV benchmark & model training
│   ├── evaluate_test.py     # Out-of-sample evaluation on test datasets
│   └── explain.py           # Tree-SHAP interpretability generation
├── PROJECT_LOG.md           # Engineering decision log & mathematical rationales
└── README.md


---

## 🔍 Model Interpretability & Explainability (XAI)

To ensure predictions are actionable for flight safety engineers, we used **Tree-SHAP** to quantify how individual sensor streams drive RUL estimates.

![SHAP Feature Importance](assets/shap_summary.png)

### Key Engineering Insights:
1. **Thermodynamic Signal Dominance:** Moving averages of high-pressure turbine outlet temperatures and core speed indicators drive over 60% of model output decisions.
2. **Instability Detection:** Standard deviation features (`_mov_std`) show strong positive SHAP impact during late engine lifespans, capturing structural vibration and thermal instability prior to breakdown.

---

## 🚀 Pipeline Execution

### 1. Ingestion & Feature Generation
```bash
python src/data_loader.py
python src/features.py
