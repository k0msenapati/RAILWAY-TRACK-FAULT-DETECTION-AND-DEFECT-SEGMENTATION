#!/usr/bin/env python3
"""
segmentation/train_yolo_seg.py
Trains / Fine-tunes YOLOv8n-seg on Railway Track Defect Segmentation data.
Configured strictly for CPU execution (lightweight memory footprint and fast epochs).
"""

import os
import sys
import argparse
import shutil
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(PROJECT_ROOT, "data", "segmentation", "data.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_segmentation_model(data_yaml=DATA_YAML, epochs=15, imgsz=416, batch=8):
    print("=" * 65)
    print(" YOLOv8n-seg Training on Railway Track Defect Dataset (CPU)")
    print("=" * 65)

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}. Run scripts/download_segmentation_data.py first.")

    # Load pre-trained nano segmentation model
    pretrained_weights = os.path.join(MODELS_DIR, "yolov8n-seg.pt")
    if os.path.exists(pretrained_weights):
        model = YOLO(pretrained_weights)
    else:
        model = YOLO("yolov8n-seg.pt")

    print(f"[*] Starting fine-tuning for {epochs} epochs on CPU...")
    print(f"  - Input image resolution: {imgsz}x{imgsz}")
    print(f"  - Batch size: {batch}")
    print(f"  - Device: CPU")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device="cpu",
        workers=2,
        project=os.path.join(PROJECT_ROOT, "runs", "segment"),
        name="railway_seg_cpu",
        exist_ok=True,
        verbose=True,
        plots=True
    )

    # Copy best weights to models/
    best_weights = os.path.join(PROJECT_ROOT, "runs", "segment", "railway_seg_cpu", "weights", "best.pt")
    target_weights = os.path.join(MODELS_DIR, "railway_yolov8n_seg.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"\n[✓] Best model weights saved to: {target_weights}")
    else:
        # Fallback to last.pt or save current model
        model.save(target_weights)
        print(f"\n[✓] Saved fine-tuned model weights to: {target_weights}")

    return target_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n Segmentation Model on CPU")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--data", default=DATA_YAML, help="Path to data.yaml")
    args = parser.parse_args()

    train_segmentation_model(args.data, args.epochs, args.imgsz, args.batch)
