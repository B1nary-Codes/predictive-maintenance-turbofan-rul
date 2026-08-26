import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap
from pathlib import Path

FEATURED_PATH = Path("data/processed/train_FD001_featured.csv")
MODEL_PATH = Path("models/xgb_model.joblib")
ASSETS_DIR = Path("assets")

def generate_shap_analysis():
    """Computes Tree-SHAP values for the trained XGBoost model to quantify 
    exact feature contributions to remaining useful life predictions."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load dataset and isolated feature matrix X
    df = pd.read_csv(FEATURED_PATH)
    ignore_cols = ["unit_nr", "time_cycles", "RUL", "RUL_clipped"]
    X = df.drop(columns=ignore_cols)
    
    # Load trained model artifact
    model = joblib.load(MODEL_PATH)
    
    print("Computing Tree-SHAP values across training feature matrix...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    
    # Save Summary Bar Plot
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Turbofan RUL Prediction: SHAP Feature Importance Summary", fontsize=14, pad=15)
    plt.tight_layout()
    
    plot_path = ASSETS_DIR / "shap_summary.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"SHAP feature importance plot saved to {plot_path}")

if __name__ == "__main__":
    generate_shap_analysis()