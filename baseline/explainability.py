#!/usr/bin/env python3
"""
baseline/explainability.py
Explainable Artificial Intelligence (XAI) analysis using SHAP (SHapley Additive exPlanations).
Computes Shapley values for the trained Random Forest model and generates:
1. Feature Importance Bar Plot (comparing deep YOLO features vs handcrafted features)
2. SHAP Beeswarm Plot showing impact on defect prediction
Saved to docs/images/
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

def run_shap_analysis(features_csv="data/baseline/val_merged_features.csv", sample_size=100):
    print("=" * 65)
    print(" Explainable AI (XAI) with SHAP for Railway Fault Detection")
    print("=" * 65)

    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    imputer_path = os.path.join(MODELS_DIR, "imputer.pkl")
    meta_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    for p in [model_path, scaler_path, imputer_path, meta_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required artifact not found: {p}. Run baseline/train_rf.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    feature_names = joblib.load(meta_path)

    if not os.path.exists(features_csv):
        features_csv = "data/baseline/train_merged_features.csv"

    df = pd.read_csv(features_csv)
    X_raw = df[feature_names]
    
    if len(X_raw) > sample_size:
        X_sample = X_raw.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_raw

    X_imp = imputer.transform(X_sample)
    X_scaled = scaler.transform(X_imp)
    X_df = pd.DataFrame(X_scaled, columns=feature_names)

    print(f"[*] Initializing SHAP TreeExplainer on {len(X_df)} samples...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    # For binary classification, shap_values can be a list [class0, class1] or 3D array
    if isinstance(shap_values, list):
        shap_vals_defect = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_vals_defect = shap_values[:, :, 1]
    else:
        shap_vals_defect = shap_values

    # Plot 1: Bar plot of feature importances
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_defect, X_df, plot_type="bar", show=False, max_display=15)
    plt.title("Top 15 Most Influential Features (SHAP Mean |Value|)")
    plt.tight_layout()
    bar_path = os.path.join(DOCS_IMG_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"[✓] Saved SHAP feature importance plot to: {bar_path}")

    # Plot 2: Detailed Beeswarm plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals_defect, X_df, show=False, max_display=12)
    plt.title("SHAP Feature Impact on Defect Prediction (Beeswarm)")
    plt.tight_layout()
    beeswarm_path = os.path.join(DOCS_IMG_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=200)
    plt.close()
    print(f"[✓] Saved SHAP beeswarm plot to: {beeswarm_path}")

    # Compute breakdown: Deep features vs Handcrafted features
    mean_abs_shap = np.abs(shap_vals_defect).mean(axis=0)
    yolo_indices = [i for i, name in enumerate(feature_names) if name.startswith("yolo_")]
    manual_indices = [i for i, name in enumerate(feature_names) if not name.startswith("yolo_")]

    yolo_importance = float(np.sum(mean_abs_shap[yolo_indices]))
    manual_importance = float(np.sum(mean_abs_shap[manual_indices]))
    total = yolo_importance + manual_importance + 1e-8

    print("\n" + "=" * 65)
    print(" SHAP FEATURE CONTRIBUTION BREAKDOWN")
    print("=" * 65)
    print(f"  YOLOv8 Deep Features Contribution:     {yolo_importance / total * 100:.2f}%")
    print(f"  Handcrafted Domain Features (GLCM/etc): {manual_importance / total * 100:.2f}%")
    print("=" * 65)

    return {
        "bar_plot": bar_path,
        "beeswarm_plot": beeswarm_path,
        "yolo_ratio": yolo_importance / total,
        "manual_ratio": manual_importance / total
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SHAP Explainability Analysis")
    parser.add_argument("--features", default="data/baseline/val_merged_features.csv", help="Features CSV path")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to explain")
    args = parser.parse_args()

    run_shap_analysis(args.features, args.samples)
