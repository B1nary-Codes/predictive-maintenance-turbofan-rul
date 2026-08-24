import pandas as pd
from pathlib import Path
from typing import List, Tuple

INDEX_COLS = ["unit_nr", "time_cycles"]
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS

DATA_PATH = Path("data/raw")

def load_train_data(filename: str = "train_FD001.txt") -> pd.DataFrame:
    """Loads raw C-MAPSS telemetry file and applies column schema."""
    file_path = DATA_PATH / filename
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)

def clean_telemetry_data(df: pd.DataFrame, threshold: float = 1e-4) -> Tuple[pd.DataFrame, List[str]]:
    """Identifies and drops invariant columns (operational settings and sensors with near-zero std)."""
    # Exclude index columns from variance check
    feature_cols = SETTING_COLS + SENSOR_COLS
    std_series = df[feature_cols].std()
    
    dead_cols = std_series[std_series < threshold].index.tolist()
    cleaned_df = df.drop(columns=dead_cols)
    
    return cleaned_df, dead_cols

if __name__ == "__main__":
    raw_df = load_train_data()
    clean_df, dropped_cols = clean_telemetry_data(raw_df)
    
    print(f"Raw telemetry shape: {raw_df.shape}")
    print(f"Dropped non-informative channels ({len(dropped_cols)}): {dropped_cols}")
    print(f"Cleaned telemetry shape: {clean_df.shape}")