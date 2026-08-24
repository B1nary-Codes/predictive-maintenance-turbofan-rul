# predictive-maintenance-turbofan-rul
Engineering data analytics pipeline evaluating sensor degradation telemetry to estimate Remaining Useful Life (RUL) in industrial turbofan engines.

# Turbofan Engine Predictive Maintenance & RUL Estimation

## Overview & Engineering Problem
Rotating mechanical components operating under extreme thermal and mechanical stresses degrade over time. Unexpected equipment failure leads to costly operational downtime and safety hazards. This repository implements an engineering-focused data pipeline to process multi-channel sensor telemetry (pressures, temperatures, shaft speeds) from industrial turbofan engines, isolating degradation signals to estimate Remaining Useful Life (RUL) and establish predictive maintenance windows.

## Key Objectives
* **Sensor Data Hygiene:** Ingest dynamic time-series sensor streams and identify/drop invariant or non-informative channels.
* **Feature Engineering:** Apply sliding-window statistics, rolling standard deviations, and degradation trend rates to extract clean signals from operational noise.
* **Data Modeling & SQL:** Structure run-to-failure cycles to query operational degradation thresholds across distinct engine units.
* **RUL Estimation:** Predict continuous operational cycles remaining before functional failure to optimize maintenance scheduling.
* **Engineering Trade-Off Synthesis:** Evaluate model performance using domain-relevant error metrics that penalize late maintenance predictions over early replacements.

## Dataset Profile
* **Source:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) Dataset
* **Telemetry:** 21 sensor channels, 3 operational settings across run-to-failure engine units
* **Target Metric:** Remaining Useful Life (RUL) in operational cycles

## Repository Structure
```text
predictive-maintenance-turbofan-rul/
├── data/
│   ├── raw/                 # Raw NASA C-MAPSS telemetry files
│   └── processed/           # Filtered DataFrames with engineered features
├── notebooks/
│   ├── 01_eda_and_sensor_cleaning.ipynb
│   └── 02_feature_engineering_rul.ipynb
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Ingestion & channel filtering
│   └── features.py          # Rolling statistics & signal processing
├── .gitignore
├── README.md
└── requirements.txt
