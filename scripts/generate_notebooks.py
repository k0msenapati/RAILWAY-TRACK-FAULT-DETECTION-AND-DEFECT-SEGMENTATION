#!/usr/bin/env python3
"""
scripts/generate_notebooks.py
Generates clean, concise Jupyter notebooks for university project submission and defense:
1. notebooks/01_fault_classification_demo.ipynb
2. notebooks/02_defect_segmentation_demo.ipynb
"""

import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks")
os.makedirs(NB_DIR, exist_ok=True)

def create_notebook(cells, output_path):
    for idx, cell in enumerate(cells):
        cell["id"] = f"cell_{idx+1:03d}"

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (Railway CPU)",
                "language": "python",
                "name": "railway_cpu"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"[✓] Generated notebook: {output_path}")

def build_classification_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Railway Track Fault Classification\n",
                "**Module 1: Hybrid Feature Fusion & Random Forest Classifier**\n",
                "\n",
                "**Workflow**:\n",
                "1. Image Preprocessing & Data Augmentation (Blur, Grayscale, Noise, Flips)\n",
                "2. Feature Extraction: YOLOv8 Backbone (Deep Features) + GLCM, Canny, RGB Histograms (Handcrafted Features)\n",
                "3. Feature Fusion: $x_{fused} = x_{deep} \\parallel x_{hand}$\n",
                "4. Random Forest Model Training & Performance Evaluation\n",
                "5. Explainable AI: SHAP Feature Importance Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Environment Setup & Library Imports\n",
                "import os, sys\n",
                "import cv2, torch, joblib\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Add project root to sys.path\n",
                "PROJECT_ROOT = os.path.abspath('..') if os.path.exists('../baseline') else os.path.abspath('.')\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "print(f\"Project Directory: {PROJECT_ROOT}\")\n",
                "print(f\"PyTorch Version: {torch.__version__} | CPU Threads: {torch.get_num_threads()}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Data Preprocessing & Augmentation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from baseline.preprocess import preprocess_image, augment_image\n",
                "\n",
                "sample_img = os.path.join(PROJECT_ROOT, \"data/baseline/Train/Defective/track_train_def_0001.jpg\")\n",
                "if os.path.exists(sample_img):\n",
                "    orig = cv2.imread(sample_img)\n",
                "    prep = preprocess_image(sample_img)\n",
                "    augs = augment_image(orig)\n",
                "    \n",
                "    fig, axes = plt.subplots(1, 4, figsize=(14, 4))\n",
                "    axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))\n",
                "    axes[0].set_title(\"Original Image\")\n",
                "    axes[0].axis(\"off\")\n",
                "    \n",
                "    axes[1].imshow(prep, cmap=\"gray\")\n",
                "    axes[1].set_title(\"Preprocessed (224x224)\")\n",
                "    axes[1].axis(\"off\")\n",
                "    \n",
                "    axes[2].imshow(cv2.cvtColor(augs[0], cv2.COLOR_BGR2RGB))\n",
                "    axes[2].set_title(\"Augmentation: Flip\")\n",
                "    axes[2].axis(\"off\")\n",
                "    \n",
                "    axes[3].imshow(cv2.cvtColor(augs[2], cv2.COLOR_BGR2RGB))\n",
                "    axes[3].set_title(\"Augmentation: Brightness\")\n",
                "    axes[3].axis(\"off\")\n",
                "    plt.tight_layout()\n",
                "    plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Feature Extraction & Vector Fusion"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from baseline.extract_features import extract_yolo_features, extract_manual_features\n",
                "\n",
                "if os.path.exists(sample_img):\n",
                "    deep_feats = extract_yolo_features(sample_img)\n",
                "    hand_feats = extract_manual_features(sample_img)\n",
                "    fused_vector = deep_feats + hand_feats\n",
                "    \n",
                "    print(f\"Deep Features (YOLOv8 Backbone): {len(deep_feats)}\")\n",
                "    print(f\"Handcrafted Features (GLCM, Canny, RGB): {len(hand_feats)}\")\n",
                "    print(f\"Total Fused Vector Dimension: {len(fused_vector)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Model Training & Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from baseline.train_rf import train_and_evaluate\n",
                "\n",
                "train_csv = os.path.join(PROJECT_ROOT, \"data/baseline/train_merged_features.csv\")\n",
                "val_csv = os.path.join(PROJECT_ROOT, \"data/baseline/val_merged_features.csv\")\n",
                "\n",
                "if os.path.exists(train_csv):\n",
                "    metrics = train_and_evaluate(train_csv, val_csv)\n",
                "    print(f\"Final Model Accuracy: {metrics['accuracy']:.2f}%\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Explainable AI Analysis (SHAP)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from baseline.explainability import run_shap_analysis\n",
                "\n",
                "if os.path.exists(train_csv):\n",
                "    shap_results = run_shap_analysis(train_csv, sample_size=50)\n",
                "    print(f\"Deep Features Contribution: {shap_results['yolo_ratio']*100:.1f}%\")\n",
                "    print(f\"Handcrafted Features Contribution: {shap_results['manual_ratio']*100:.1f}%\")"
            ]
        }
    ]
    # We update both the original name and standard name for compatibility
    create_notebook(cells, os.path.join(NB_DIR, "01_baseline_classification.ipynb"))

def build_segmentation_notebook():
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Railway Track Defect Localization & Instance Segmentation\n",
                "**Module 2: Real-Time Defect Detection, Polygon Mask Segmentation & Severity Scoring**\n",
                "\n",
                "**Features**:\n",
                "- Simultaneous Bounding Box Detection + Pixel Segmentation Masks\n",
                "- Target Classes: `crack`, `missing_fastener`, `broken_rail`, `surface_defect`\n",
                "- Real-Time CPU Inference (> 25 FPS)\n",
                "- Automated Defect Severity Index (Minor, Moderate, Critical)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# System Setup & Model Initialization\n",
                "import os, sys\n",
                "import cv2, torch\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "PROJECT_ROOT = os.path.abspath('..') if os.path.exists('../segmentation') else os.path.abspath('.')\n",
                "if PROJECT_ROOT not in sys.path:\n",
                "    sys.path.insert(0, PROJECT_ROOT)\n",
                "\n",
                "from segmentation.inference_pipeline import RailwaySegmentationPipeline\n",
                "\n",
                "model_weights = os.path.join(PROJECT_ROOT, \"models/railway_yolov8n_seg.pt\")\n",
                "pipeline = RailwaySegmentationPipeline(model_path=model_weights)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Test Image Inference & Defect Severity Assessment"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "test_img_dir = os.path.join(PROJECT_ROOT, \"data/segmentation/images/test\")\n",
                "if os.path.exists(test_img_dir):\n",
                "    test_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]\n",
                "    if test_files:\n",
                "        sample_path = os.path.join(test_img_dir, test_files[0])\n",
                "        result = pipeline.predict(sample_path, save_annotated=True)\n",
                "        \n",
                "        print(f\"Track Condition: {result['status']}\")\n",
                "        print(f\"Detected Defect Count: {result['defect_count']}\")\n",
                "        print(f\"CPU Latency: {result['cpu_latency_ms']:.1f} ms | FPS: {result['cpu_fps']:.1f}\")\n",
                "        \n",
                "        annotated_img = cv2.imread(result['annotated_output'])\n",
                "        plt.figure(figsize=(10, 8))\n",
                "        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))\n",
                "        plt.title(f\"Defect Detection Output ({result['cpu_fps']:.1f} FPS on CPU)\")\n",
                "        plt.axis(\"off\")\n",
                "        plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Quantitative Performance Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from segmentation.evaluate_metrics import evaluate_segmentation\n",
                "\n",
                "data_yaml = os.path.join(PROJECT_ROOT, \"data/segmentation/data.yaml\")\n",
                "metrics = evaluate_segmentation(weights=model_weights, data_yaml=data_yaml)\n",
                "\n",
                "print(\"\\nSummary Performance Metrics:\")\n",
                "print(f\"- Box mAP@50:       {metrics['box_map50']:.2f}%\")\n",
                "print(f\"- Mask mAP@50:      {metrics['mask_map50']:.2f}%\")\n",
                "print(f\"- Mean IoU (mIoU):  {metrics['miou']:.2f}%\")\n",
                "print(f\"- Dice Coefficient: {metrics['dice']:.2f}%\")\n",
                "print(f\"- CPU Latency:      {metrics['cpu_latency_ms']:.2f} ms\")"
            ]
        }
    ]
    create_notebook(cells, os.path.join(NB_DIR, "02_defect_segmentation_demo.ipynb"))

if __name__ == "__main__":
    build_classification_notebook()
    build_segmentation_notebook()
