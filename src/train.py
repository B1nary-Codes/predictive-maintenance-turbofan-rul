import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

FEATURED_PATH = Path("data/processed/train_FD001_featured.csv")
MODEL_DIR = Path("models")

def evaluate_model(model_cls, model_kwargs, name="Model"):
    """Evaluates a regression model using 5-fold Groupd Cross-Validation"""
    df = pd.read_csv(FEATURED_PATH)

    ignore_cols = ["unit_nr", "time_cycles", "RUL", "RUL_clipped"]
    X = df.drop(columns=ignore_cols)
    y = df["RUL_clipped"]
    groups = df["unit_nr"]

    gkf = GroupKFold(n_splits=5)
    rmse_scores, mae_scores = [], []

    print(f"--- Running 5-Fold Group Cross-Validation: {name} ---")
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        model = model_cls(**model_kwargs)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        mae = mean_absolute_error(y_val, preds)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    mean_rmse = np.mean(rmse_scores)
    mean_mae = np.mean(mae_scores)

    print(f"Mean CV RMSE: {mean_rmse:.2f} (+/- {np.std(rmse_scores):.2f}) cycles")
    print(f"Mean CV MAE:  {mean_mae:.2f} (+/- {np.std(mae_scores):.2f}) cycles\n")
    return mean_rmse, mean_mae

def train_final_model():
    """Trains XGBoost model on full processed dataset and exports artifact."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURED_PATH)

    ignore_cols = ["unit_nr", "time_cycles", "RUL", "RUL_clipped"]
    X = df.drop(columns=ignore_cols)
    y = df["RUL_clipped"]

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X, y)

    # Save trained b1nary artifact
    model_file = MODEL_DIR / "xgb_model.joblib"
    joblib.dump(model, model_file)
    print(f"Model serialized to {model_file}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\n--- Top 10 Key Features")
    print(importances.head(10).to_string())

if __name__ == "__main__":
    evaluate_model(
        RandomForestRegressor,
        {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "Random Forest Baseline (EXP-01)"
    )

    evaluate_model(
        XGBRegressor,
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 5, "random_state": 42, "n_jobs": -1},
        "XGBoost Model (EXP-02)"
    )

    train_final_model()