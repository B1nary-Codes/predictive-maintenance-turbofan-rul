import pandas as pd
from pathlib import Path
from typing import List, Tuple

# Define schema column mappings for raw unheadered C-MAPSS text files
INDEX_COLS = ["unit_nr", "time_cycles"]                  # Metadata: Engine ID and current operational flight cycle
SETTING_COLS = ["setting_1", "setting_2", "setting_3"]   # Operational settings: Throttle, altitude, environmental controls
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]     # 21 Raw thermodynamic and pressure sensor channels
COLUMN_NAMES = INDEX_COLS + SETTING_COLS + SENSOR_COLS   # Complete 26-column schema order

DATA_PATH = Path("data/raw")

def load_train_data(filename: str = "train_FD001.txt") -> pd.DataFrame:
    """
    Reads space-delimited raw telemetry files into a pandas DataFrame.
    Uses regex delimiter "\s+" to handle variable whitespace spacing between columns.
    """
    file_path = DATA_PATH / filename
    return pd.read_csv(file_path, sep=r"\s+", header=None, names=COLUMN_NAMES)

def clean_telemetry_data(df: pd.DataFrame, threshold: float = 1e-4) -> Tuple[pd.DataFrame, List[str]]:
    """
    Identifies and removes low-variance (invariant) features.
    
    Sensors with standard deviation near zero reflect flatline sensors or non-variable 
    operating conditions. Dropping them eliminates dimensional noise without losing 
    predictive degradation signal.
    """
    # Isolate non-index columns to test for dynamic variance
    feature_cols = SETTING_COLS + SENSOR_COLS
    std_series = df[feature_cols].std()
    
    # Identify channels whose standard deviation falls below variance threshold
    dead_cols = std_series[std_series < threshold].index.tolist()
    
    # Drop invariant channels across all dataset rows
    cleaned_df = df.drop(columns=dead_cols)
    
    return cleaned_df, dead_cols

def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates absolute Remaining Useful Life (RUL) target per engine.
    
    RUL_i,t = Max_Cycle_i - Current_Cycle_t
    `transform('max')` broadcasts each engine's lifetime maximum cycle count 
    back to all individual rows for that specific unit ID.
    """
    # Group by engine unit and find maximum operational lifespan reached
    max_cycles = df.groupby("unit_nr")["time_cycles"].transform("max")
    
    # Target RUL counts down to zero at total component failure
    df["RUL"] = max_cycles - df["time_cycles"]
    return df

if __name__ == "__main__":
    # 1. Pipeline Execution: Load raw telemetry
    raw_df = load_train_data()
    
    # 2. Pipeline Execution: Filter invariant low-variance channels
    clean_df, dropped_cols = clean_telemetry_data(raw_df)
    
    # 3. Pipeline Execution: Compute absolute target regression variable
    processed_df = add_rul_target(clean_df)
    
    # 4. Save processed output to disk for feature engineering stage
    output_path = Path("data/processed/train_FD001_processed.csv")
    processed_df.to_csv(output_path, index=False)
    print(f"Ingestion and Target Calculation Complete. File exported to {output_path}")