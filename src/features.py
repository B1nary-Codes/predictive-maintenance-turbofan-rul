import pandas as pd
from pathlib import Path
from typing import List

def generate_rolling_features(
    df: pd.DataFrame, 
    sensor_cols: List[str], 
    window_size: int = 10
) -> pd.DataFrame:
    """Computes rolling mean and standard deviation per engine unit."""
    df_feat = df.copy()
    grouped = df_feat.groupby("unit_nr")[sensor_cols]
    
    rolling_mean = grouped.rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
    rolling_std = grouped.rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)
    
    rolling_mean.columns = [f"{col}_mov_avg" for col in sensor_cols]
    rolling_std.columns = [f"{col}_mov_std" for col in sensor_cols]
    
    return pd.concat([df_feat, rolling_mean, rolling_std], axis=1)

def apply_piecewise_rul(df: pd.DataFrame, max_rul: int = 125) -> pd.DataFrame:
    """Clips RUL target at max threshold to model healthy initial operating state."""
    df_clipped = df.copy()
    df_clipped["RUL_clipped"] = df_clipped["RUL"].clip(upper=max_rul)
    return df_clipped

if __name__ == "__main__":
    processed_path = Path("data/processed/train_FD001_processed.csv")
    df = pd.read_csv(processed_path)
    
    active_sensors = [col for col in df.columns if col.startswith("sensor_")]
    
    # 1. Generate 10-cycle rolling metrics
    featured_df = generate_rolling_features(df, sensor_cols=active_sensors, window_size=10)
    
    # 2. Apply piecewise linear target clipping
    final_df = apply_piecewise_rul(featured_df, max_rul=125)
    
    # 3. Export featured dataset
    output_path = Path("data/processed/train_FD001_featured.csv")
    final_df.to_csv(output_path, index=False)
    
    print(f"Final feature matrix saved to {output_path}")
    print(f"Shape: {final_df.shape}")