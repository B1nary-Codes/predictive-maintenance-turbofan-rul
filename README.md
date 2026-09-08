# NASA C-MAPSS Turbofan Predictive Maintenance

Predicting Remaining Useful Life (RUL) for NASA C-MAPSS turbofan engines (`FD001`) using time-series sensor telemetry and XGBoost.

## Benchmark Results

Evaluated on out-of-sample test engines (`test_FD001.txt`) using the final recorded cycle per unit:

| Model | Features | Val RMSE | Test RMSE | Test MAE |
| :--- | :--- | :--- | :--- | :--- |
| Random Forest Baseline | Active Sensors + 10-cycle Rolling Stats | 19.33 | — | — |
| XGBoost Regressor | Active Sensors + 10-cycle Rolling Stats | 18.61 | **18.12** | **12.76** |

## Key Design Decisions

* **Low-Variance Pruning:** Dropped 7 static channels (`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) that show zero variance across runs.
* **Signal Smoothing:** Applied 10-cycle rolling means and standard deviations grouped strictly by engine ID to reduce sensor noise without leaking temporal data.
* **Piecewise Target Clipping:** Capped healthy engine RUL targets at 125 cycles to match physical degradation trends and keep early-life states from skewing predictions.
* **Leakage Prevention:** Used 5-fold `GroupKFold` validation partitioned by `unit_nr` so full engine lifespans stay isolated during training.
* **Explainability:** Added Tree-SHAP analysis showing high-pressure turbine temperature (`sensor_4`) and discharge pressure (`sensor_11`) drive over 60% of model outputs.

## Project Structure

```text
predictive-maintenance-turbofan-rul/
├── assets/             # Visual outputs and SHAP plots
├── data/               # Raw and processed telemetry
├── models/             # Exported XGBoost model weights
├── src/
│   ├── data_loader.py  # Ingestion and channel pruning
│   ├── features.py     # Rolling aggregations and target clipping
│   ├── train.py        # GroupKFold training loop
│   ├── evaluate_test.py# Out-of-sample test set evaluation
│   ├── explain.py      # SHAP feature importance scripts
│   └── predict.py      # CLI inference tool
├── PROJECT_LOG.md      # Architectural log
└── README.md
```
## Quickstart

```bash
# Setup environment
git clone [https://github.com/B1nary-Codes/predictive-maintenance-turbofan-rul.git](https://github.com/B1nary-Codes/predictive-maintenance-turbofan-rul.git)
cd predictive-maintenance-turbofan-rul
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline & CLI inference
python src/data_loader.py
python src/features.py
python src/train.py
python src/evaluate_test.py
python src/explain.py
python src/predict.py --unit 1
