#!/usr/bin/env python3
"""
baseline/train_rf.py
Random Forest training, evaluation, and serialization pipeline.
- Loads train and validation merged feature CSVs
- Standardizes feature representations with StandardScaler
- Trains RandomForestClassifier (200 trees, max depth 15, balanced class weights)
- Computes comprehensive evaluation metrics (Accuracy, Precision, Recall, F1, Jaccard, MCC, ROC/AUC)
- Saves confusion matrix and ROC curves
- Serializes trained model and scaler to models/ directory
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    jaccard_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, classification_report
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

def train_and_evaluate(train_csv, val_csv=None, output_model_dir=MODELS_DIR):
    print("=" * 65)
    print(" Baseline Random Forest Training & Evaluation")
    print("=" * 65)

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Train features file not found: {train_csv}")

    df_train = pd.read_csv(train_csv)
    print(f"[*] Loaded training features: {df_train.shape}")

    # Drop non-feature columns
    feature_cols = [c for c in df_train.columns if c not in ['image_path', 'label']]
    X_train_raw = df_train[feature_cols]
    y_train_raw = df_train['label']

    if val_csv and os.path.exists(val_csv):
        df_val = pd.read_csv(val_csv)
        print(f"[*] Loaded validation features: {df_val.shape}")
        X_test_raw = df_val[feature_cols]
        y_test = df_val['label']
        X_train = X_train_raw
        y_train = y_train_raw
    else:
        print("[*] Performing 80/20 stratified train/test split...")
        X_train, X_test_raw, y_train, y_test = train_test_split(
            X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
        )

    # Impute missing values if any
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test_raw)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    print(f"[*] Training Random Forest Classifier on CPU...")
    print(f"  - Features: {len(feature_cols)} (Deep + Handcrafted)")
    print(f"  - Trees: 200, Max Depth: 15, Class Weight: balanced")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    print("  [✓] Model training completed!")

    # Predictions
    y_pred = rf.predict(X_test_scaled)
    y_proba = rf.predict_proba(X_test_scaled)[:, 1]

    # Metrics computation
    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred, zero_division=0) * 100
    rec = recall_score(y_test, y_pred, zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, zero_division=0) * 100
    jaccard = jaccard_score(y_test, y_pred, zero_division=0) * 100
    mcc = matthews_corrcoef(y_test, y_pred) * 100
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr) * 100

    print("\n" + "=" * 65)
    print(" MODEL PERFORMANCE EVALUATION")
    print("=" * 65)
    print(f"  Accuracy:       {acc:.2f}%")
    print(f"  Precision:      {prec:.2f}%")
    print(f"  Recall:         {rec:.2f}%")
    print(f"  F1-Score:       {f1:.2f}%")
    print(f"  Jaccard Index:  {jaccard:.2f}%")
    print(f"  MCC:            {mcc:.2f}%")
    print(f"  ROC-AUC:        {roc_auc:.2f}%")
    print("-" * 65)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=["Non Defective", "Defective"]))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Defective", "Defective"])
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(DOCS_IMG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[✓] Saved confusion matrix plot to {cm_path}")

    # Save ROC Curve Plot
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(DOCS_IMG_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"[✓] Saved ROC curve plot to {roc_path}")

    # Serialize Model, Scaler, Imputer and Feature Column List
    model_file = os.path.join(output_model_dir, "random_forest_model.pkl")
    scaler_file = os.path.join(output_model_dir, "scaler.pkl")
    imputer_file = os.path.join(output_model_dir, "imputer.pkl")
    meta_file = os.path.join(output_model_dir, "feature_names.pkl")

    joblib.dump(rf, model_file)
    joblib.dump(scaler, scaler_file)
    joblib.dump(imputer, imputer_file)
    joblib.dump(feature_cols, meta_file)

    print(f"[✓] Saved trained Random Forest model to: {model_file}")
    print(f"[✓] Saved scaler to: {scaler_file}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "jaccard": jaccard,
        "mcc": mcc,
        "auc": roc_auc,
        "model_file": model_file,
        "scaler_file": scaler_file
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline Random Forest Classifier")
    parser.add_argument("--train-csv", default="data/baseline/train_merged_features.csv", help="Path to train features CSV")
    parser.add_argument("--val-csv", default="data/baseline/val_merged_features.csv", help="Path to val features CSV")
    args = parser.parse_args()

    train_and_evaluate(args.train_csv, args.val_csv)
