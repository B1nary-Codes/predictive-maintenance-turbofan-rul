import pandas as pd
from pathlib import Path
from typing import List

def generate_rolling_features(
    df: pd.DataFrame, 
    sensor_cols: List[str], 
    window_size: int = 10
) -> pd.DataFrame:
    """
    Extracts time-series contextual features using sliding window aggregations.
    
    - Rolling Mean: Smooths high-frequency sensor noise to isolate long-term thermal wear drift.
    - Rolling Std: Captures signal jitter/vibration variance that escalates prior to failure.
    - Grouping by "unit_nr" ensures sliding windows do NOT bleed across different engine lifespans.
    - "min_periods=1" prevents generating NaN values during early engine cycles (< 10 cycles).
    """
    df_feat = df.copy()
    
    # Isolate sensor columns grouped strictly by unit ID
    grouped = df_feat.groupby("unit_nr")[sensor_cols]
    
    # Compute moving statistics and strip grouping index formatting
    rolling_mean = grouped.rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_std = grouped.rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    
    # Assign distinct column suffixes for model clarity
    rolling_mean.columns = [f"{col}_mov_avg" for col in sensor_cols]
    rolling_std.columns = [f"{col}_mov_std" for col in sensor_cols]
    
    # Concatenate original attributes with new time-series rolling features
    return pd.concat([df_feat, rolling_mean, rolling_std], axis=1)

def apply_piecewise_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """
    Caps ground-truth RUL target at a maximum operational healthy threshold.
    
    Physical rationale: Turbofan engines do not exhibit observable thermodynamic 
    degradation during early flight life. A brand-new engine at cycle 5 operates identically 
    to cycle 30. Clipping RUL at 125 cycles prevents the regression algorithms from attempting 
    to learn non-existent wear patterns during initial healthy operation.
    """
    df_clipped = df.copy()
    df_clipped["RUL_clipped"] = df_clipped["RUL"].clip(upper=max_rul)
    return df_clipped

if __name__ == "__main__":
    processed_path = Path("data/processed/train_FD001_processed.csv")
    df = pd.read_csv(processed_path)
    
    # Dynamically extract remaining active sensor channels
    active_sensors = [col for col in df.columns if col.startswith("sensor_")]
    
    # 1. Feature Generation: Compute 10-cycle moving statistics
    featured_df = generate_rolling_features(df, sensor_cols=active_sensors, window_size=10)
    
    # 2. Target Engineering: Apply piecewise linear clipping threshold
    final_df = apply_piecewise_rul(featured_df, max_rul=125)
    
    # 3. Save feature matrix to processed storage
    output_path = Path("data/processed/train_FD001_featured.csv")
    final_df.to_csv(output_path, index=False)
    print(f"Feature engineering pipeline successful. Saved to {output_path}")