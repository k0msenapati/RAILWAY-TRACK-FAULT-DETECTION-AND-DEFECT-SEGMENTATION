#!/usr/bin/env python3
"""
baseline/extract_features.py
Dual feature extraction pipeline for railway track fault classification:
1. Deep visual feature extraction via YOLOv8n-cls backbone (CPU)
2. Handcrafted domain feature extraction (GLCM, Canny, RGB means, Color histogram - 20 features)
3. Feature fusion: x_fused = x_deep || x_hand
Outputs:
- train_merged_features.csv
- val_merged_features.csv
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO

import cv2
from skimage.feature import graycomatrix, graycoprops, canny
from skimage.color import rgb2gray

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLO Preprocessing
yolo_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

_yolo_backbone = None

def get_yolo_backbone():
    global _yolo_backbone
    if _yolo_backbone is None:
        model_path = os.path.join(MODELS_DIR, "yolov8n-cls.pt")
        # Load YOLO model onto CPU
        model = YOLO(model_path if os.path.exists(model_path) else "yolov8n-cls.pt")
        # Use full feature extraction backbone (layers 0 through 8)
        _yolo_backbone = torch.nn.Sequential(*[model.model.model[i] for i in range(9)])
        _yolo_backbone.eval()
    return _yolo_backbone

def extract_yolo_features(image_path, backbone=None):
    """Extracts deep embeddings from YOLOv8 classification backbone on CPU."""
    if backbone is None:
        backbone = get_yolo_backbone()
    
    img = Image.open(image_path).convert('RGB')
    tensor = yolo_preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        feature_map = backbone(tensor)
        pooled = F.adaptive_avg_pool2d(feature_map, (1, 1))
        flattened = pooled.flatten().detach().cpu().numpy().astype(float)
    
    return flattened.tolist()

def extract_manual_features(image_path):
    """
    Extracts 20 handcrafted domain features:
    - 4 GLCM texture metrics (contrast, dissimilarity, homogeneity, energy)
    - 3 RGB mean intensities
    - 1 Canny edge pixel sum
    - 12 Color histogram bins (4 bins per channel R, G, B)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    image_resized = cv2.resize(image, (224, 224))
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # GLCM Texture Features
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    dissimilarity = float(graycoprops(glcm, 'dissimilarity')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])

    # RGB Mean Features
    r_mean = float(np.mean(image_resized[:, :, 2]))
    g_mean = float(np.mean(image_resized[:, :, 1]))
    b_mean = float(np.mean(image_resized[:, :, 0]))

    # Edge Feature (Canny)
    edges = canny(rgb2gray(image_resized))
    edge_sum = float(np.sum(edges))

    # Color Histogram (4 bins per channel)
    hist_b = cv2.calcHist([image_resized], [0], None, [4], [0, 256]).flatten()
    hist_g = cv2.calcHist([image_resized], [1], None, [4], [0, 256]).flatten()
    hist_r = cv2.calcHist([image_resized], [2], None, [4], [0, 256]).flatten()
    color_hist = np.concatenate([hist_r, hist_g, hist_b]).tolist()

    manual_features = [
        contrast, dissimilarity, homogeneity, energy,
        r_mean, g_mean, b_mean,
        edge_sum
    ] + color_hist
    return manual_features

def process_dataset(data_dir, output_csv, sample_limit=None):
    """
    Processes all images in data_dir, extracts both YOLO and manual features,
    and writes out the combined feature dataframe.
    """
    backbone = get_yolo_backbone()
    records = []

    for label_name, label_val in [("Defective", 1), ("Non defective", 0)]:
        cat_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(cat_dir):
            continue

        images = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if sample_limit:
            images = images[:sample_limit]

        for fname in tqdm(images, desc=f"Extracting features from {label_name}"):
            fpath = os.path.join(cat_dir, fname)
            try:
                yolo_feats = extract_yolo_features(fpath, backbone)
                manual_feats = extract_manual_features(fpath)
                row = [fpath, label_val] + yolo_feats + manual_feats
                records.append(row)
            except Exception as e:
                print(f"[!] Error processing {fpath}: {e}")

    if not records:
        print(f"[!] No records extracted from {data_dir}")
        return None

    num_yolo = len(yolo_feats)
    num_manual = len(manual_feats)
    yolo_cols = [f"yolo_{i}" for i in range(num_yolo)]
    manual_cols = [
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity", "glcm_energy",
        "r_mean", "g_mean", "b_mean", "edge_sum"
    ] + [f"color_hist_{i}" for i in range(12)]

    columns = ["image_path", "label"] + yolo_cols + manual_cols
    df = pd.DataFrame(records, columns=columns)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[✓] Saved {len(df)} feature records to {output_csv} (Total features: {num_yolo + num_manual})")
    return df

def main():
    parser = argparse.ArgumentParser(description="Dual Feature Extraction for Railway Fault Detection")
    parser.add_argument("--data-root", default="data/baseline", help="Root directory containing Train and Validation")
    parser.add_argument("--output-dir", default="data/baseline", help="Directory to save merged feature CSVs")
    parser.add_argument("--sample-run", action="store_true", help="Run on small sample for quick testing")
    args = parser.parse_args()

    limit = 10 if args.sample_run else None

    # Process Train split
    train_dir = os.path.join(args.data_root, "Train")
    if os.path.exists(train_dir):
        train_csv = os.path.join(args.output_dir, "train_merged_features.csv")
        print(f"\n[*] Processing Train dataset at: {train_dir}")
        process_dataset(train_dir, train_csv, sample_limit=limit)

    # Process Validation split
    val_dir = os.path.join(args.data_root, "Validation")
    if os.path.exists(val_dir):
        val_csv = os.path.join(args.output_dir, "val_merged_features.csv")
        print(f"\n[*] Processing Validation dataset at: {val_dir}")
        process_dataset(val_dir, val_csv, sample_limit=limit)

if __name__ == "__main__":
    main()
