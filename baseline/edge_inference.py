#!/usr/bin/env python3
"""
baseline/edge_inference.py
Simulates real-time edge / serverless cloud inference for the baseline classifier.
Performs end-to-end CPU feature extraction + scaling + Random Forest prediction
and benchmarks processing latency per image.
"""

import os
import sys
import time
import json
import argparse
import joblib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import feature extractors from baseline
from baseline.extract_features import extract_yolo_features, extract_manual_features, get_yolo_backbone

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

class BaselineEdgeInference:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.model = joblib.load(os.path.join(models_dir, "random_forest_model.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.imputer = joblib.load(os.path.join(models_dir, "imputer.pkl"))
        self.feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        self.backbone = get_yolo_backbone()

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        t0 = time.perf_counter()

        # Step 1: Feature Extraction
        t_yolo_start = time.perf_counter()
        yolo_feats = extract_yolo_features(image_path, self.backbone)
        t_yolo = (time.perf_counter() - t_yolo_start) * 1000

        t_man_start = time.perf_counter()
        manual_feats = extract_manual_features(image_path)
        t_manual = (time.perf_counter() - t_man_start) * 1000

        # Step 2: Feature Fusion
        combined = np.array(yolo_feats + manual_feats).reshape(1, -1)
        imputed = self.imputer.transform(combined)
        scaled = self.scaler.transform(imputed)

        # Step 3: Random Forest Inference
        t_rf_start = time.perf_counter()
        pred = int(self.model.predict(scaled)[0])
        probs = self.model.predict_proba(scaled)[0]
        confidence = float(np.max(probs))
        t_rf = (time.perf_counter() - t_rf_start) * 1000

        total_latency = (time.perf_counter() - t0) * 1000

        result = {
            "image_path": image_path,
            "prediction": "Defective" if pred == 1 else "Non-defective",
            "prediction_code": pred,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Non-defective": round(float(probs[0]) * 100, 2),
                "Defective": round(float(probs[1]) * 100, 2)
            },
            "latency_ms": {
                "yolo_feature_extraction": round(t_yolo, 2),
                "manual_feature_extraction": round(t_manual, 2),
                "random_forest_inference": round(t_rf, 2),
                "total_cpu_latency": round(total_latency, 2)
            }
        }
        return result

def main():
    parser = argparse.ArgumentParser(description="Baseline Edge Inference Benchmark")
    parser.add_argument("--image", required=True, help="Path to input railway track image")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="Path to models directory")
    args = parser.parse_args()

    infer = BaselineEdgeInference(args.models_dir)
    res = infer.predict(args.image)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
