import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import COLUMN_NAMES, clean_telemetry_data
from features import generate_rolling_features

RAW_DIR = Path("data/raw")
MODEL_PATH = Path("models/xgb_model.joblib")

def process_test_data():
    """Ingests raw test telemetry and extracts the final flight cycle per engine."""
    # Load raw unheadered test data
    test_df = pd.read_csv(RAW_DIR / "test_FD001.txt", sep=r"\s+", header=None, names=COLUMN_NAMES)
    
    # Apply feature cleaning and rolling features
    clean_test_df, _ = clean_telemetry_data(test_df)
    active_sensors = [col for col in clean_test_df.columns if col.startswith("sensor_")]
    featured_test_df = generate_rolling_features(clean_test_df, sensor_cols=active_sensors, window_size=10)
    
    final_cycles = featured_test_df.groupby("unit_nr").last().reset_index()
    return final_cycles

def evaluate_on_test_set():
    """Loads trained XGBoost model and calculates out-of-sample test RMSE and MAE."""
    test_df = process_test_data()
    
    # Load ground truth target RUL values
    true_rul = pd.read_csv(RAW_DIR / "RUL_FD001.txt", header=None, names=["RUL_true"])
    
    # Apply identical piecewise linear capping at 125 cycles
    y_true = true_rul["RUL_true"].clip(upper=125)
    
    # Prepare feature matrix
    ignore_cols = ["unit_nr", "time_cycles"]
    X_test = test_df.drop(columns=ignore_cols)
    
    # Load model artifact and generate predictions
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print("\n--- Final Out-of-Sample Test Set Metrics ---")
    print(f"Test Set RMSE: {rmse:.2f} cycles")
    print(f"Test Set MAE:  {mae:.2f} cycles")

if __name__ == "__main__":
    evaluate_on_test_set()