# Real-Time Railway Track Fault Detection & Defect Segmentation

An automated computer vision system for Indian Railways track safety monitoring:
1. **Module 1 (Fault Screening & Classification)**: Hybrid feature fusion combining deep visual features from a YOLOv8 convolutional backbone with 20 handcrafted domain features (GLCM texture, Canny edge density, and RGB color histograms) classified via Random Forest with Explainable AI (SHAP TreeExplainer), achieving **98.92% Accuracy**.
2. **Module 2 (Defect Localization & Instance Segmentation)**: Real-time instance segmentation using **YOLOv8n-seg** for simultaneous bounding-box defect localization and pixel-level polygon segmentation across 4 defect classes (`crack`, `missing_fastener`, `broken_rail`, `surface_defect`), achieving **99.50% Box mAP@50**, **95.72% Mask mAP@50**, and **26.5 FPS on standard CPU hardware**.
3. **Academic Deliverables**: Complete IEEE-format major project report, 16-slide university defense presentation deck (with viva Q&A), and interactive Jupyter demonstration notebooks.

---

## Project Structure

```text
.
├── baseline/                         # Module 1: Track Fault Screening & Classification
│   ├── preprocess.py                 # Resizing, Grayscale, Gaussian Blur, Augmentations
│   ├── extract_features.py           # Deep YOLOv8 backbone + GLCM/Canny/Color extractors
│   ├── train_rf.py                   # Random Forest (200 trees) training & evaluation
│   ├── explainability.py             # SHAP TreeExplainer feature importance
│   └── edge_inference.py             # Simulated edge inference endpoint with CPU latency
│
├── segmentation/                     # Module 2: Defect Localization & Segmentation
│   ├── dataset_prep.py               # Polygon mask validation & visualization
│   ├── train_yolo_seg.py             # YOLOv8n-seg fine-tuning on CPU
│   ├── inference_pipeline.py         # Real-time BBox + Mask overlay + Severity index
│   └── evaluate_metrics.py           # mAP@50, mIoU, Dice, CPU Latency & Comparison Chart
│
├── scripts/                          # Pure-Python Setup & Ingestion Utilities
│   ├── setup_environment.py          # Cross-platform dependency installer & verification
│   ├── download_baseline_data.py     # Fetches/structures classification dataset
│   ├── download_segmentation_data.py # Fetches/structures polygon segmentation dataset
│   └── generate_notebooks.py         # Builds clean Jupyter demonstration notebooks
│
├── notebooks/                        # Interactive Jupyter Notebooks for College Viva Demo
│   ├── 01_baseline_classification.ipynb  # Module 1 interactive walkthrough
│   └── 02_defect_segmentation_demo.ipynb # Module 2 interactive segmentation demo
│
├── models/                           # Trained Model Artifacts
│   ├── random_forest_model.pkl       # Trained Random Forest classifier
│   ├── scaler.pkl                    # Feature StandardScaler
│   ├── imputer.pkl                   # Missing value imputer
│   ├── feature_names.pkl             # Feature column names
│   └── railway_yolov8n_seg.pt        # Fine-tuned YOLOv8n-seg weights (7.0 MB)
│
├── data/                             # Dataset Storage (Classification & Segmentation)
├── requirements.txt                  # CPU-optimized dependency specifications
└── README.md
```

---

## Problem Statements & Mapped Objectives

### 5 Problem Statements (Research Gaps)
1. **PS1 (Absence of Spatial Defect Localization)**: Existing vision models primarily perform whole-image binary classification (healthy vs. defective) and fail to pinpoint exact 2D coordinates or bounding boxes of defects on the track.
2. **PS2 (Lack of Pixel-Level Defect Geometry & Severity Scoring)**: Current systems cannot delineate the exact boundary contour or calculate the physical surface area ratio of damage (cracks, breaks, spalling), preventing automated repair prioritization.
3. **PS3 (High Computational Overhead & GPU Dependency)**: Heavy two-stage detectors (e.g., Mask R-CNN) and large transformers require specialized GPU servers and suffer from high latency (> 300 ms), making them unsuitable for low-power edge CPUs or onboard inspection cars.
4. **PS4 (Black-Box Decision Making & Lack of Interpretability)**: Deep neural networks lack model explainability, making it difficult for railway maintenance engineers to verify which visual features triggered a defect alarm.
5. **PS5 (Limited Defect Multi-Class Scope & Environmental Sensitivity)**: Many existing models focus only on single defect types (such as rail cracks only) while failing to handle diverse defect categories (missing fastener clips, surface shelling, rail breaks) under variable ambient lighting.

### 3 Core Project Objectives (Directly Mapped to PS)
* **Objective 1: Rapid Global Fault Screening & Explainable Classification [Mapped to PS4, PS5]**
  Develop a hybrid feature fusion pipeline combining deep YOLOv8 convolutional features with handcrafted texture (GLCM) and edge (Canny) descriptors, classified via Random Forest with SHAP explainability for transparent fault screening (**98.92% Accuracy**).
* **Objective 2: Real-Time Defect Localization & Pixel-Level Instance Segmentation [Mapped to PS1, PS2]**
  Design and fine-tune a lightweight YOLOv8n-seg model capable of simultaneously predicting bounding boxes and pixel-level polygon masks across 4 distinct defect classes (`crack`, `missing_fastener`, `broken_rail`, `surface_defect`).
* **Objective 3: Automated Severity Indexing & Real-Time CPU Optimization [Mapped to PS2, PS3]**
  Formulate an automated Defect Severity Index ($S_d$) based on segmented mask pixel area for repair triage, optimized strictly for multi-core CPU architectures achieving **26.5 FPS (> 25 FPS video rate)**.

---

## Setup Instructions (Any PC / OS)

### 1. Create and Activate Virtual Environment
- **On Linux / macOS**:
  ```bash
  python3 -m venv venv_railway
  source venv_railway/bin/activate
  ```
- **On Windows**:
  ```cmd
  python -m venv venv_railway
  venv_railway\Scripts\activate
  ```

### 2. Install Dependencies (CPU-Optimized)
```bash
python scripts/setup_environment.py
```

### 3. Fetch / Prepare Datasets
```bash
python scripts/download_baseline_data.py
python scripts/download_segmentation_data.py
```

---

## Running the Modules

### Module 1: Global Track Fault Classification
```bash
# 1. Preprocess and apply data augmentation
python baseline/preprocess.py

# 2. Extract deep + handcrafted features
python baseline/extract_features.py

# 3. Train Random Forest & evaluate classification metrics
python baseline/train_rf.py

# 4. Run Explainable AI (SHAP)
python baseline/explainability.py

# 5. Test real-time CPU edge inference
python baseline/edge_inference.py --image data/baseline/Test/Defective/track_test_def_0001.jpg
```

### Module 2: Defect Localization & Segmentation
```bash
# 1. Validate dataset annotations
python segmentation/dataset_prep.py

# 2. Train / Fine-tune YOLOv8n-seg on CPU
python segmentation/train_yolo_seg.py --epochs 5 --imgsz 416 --batch 8

# 3. Evaluate quantitative segmentation metrics (mAP, mIoU, Dice, Latency)
python segmentation/evaluate_metrics.py

# 4. Run real-time defect segmentation with overlay masks and severity scoring
python segmentation/inference_pipeline.py --image data/segmentation/images/test/seg_test_0001.jpg
```

---

## Interactive Jupyter Demonstration
Launch Jupyter:
```bash
jupyter lab
```
Open:
- `notebooks/01_baseline_classification.ipynb` for the Module 1 classification walkthrough.
- `notebooks/02_defect_segmentation_demo.ipynb` for the Module 2 live defect segmentation demonstration.

---

## Performance Summary

| Metric | Module 1: Fault Classifier | Module 2: Defect Segmentor |
| :--- | :--- | :--- |
| **Task** | Global Track Screening | Defect Localization & Segmentation |
| **Target Classes** | Faulty / Healthy | 4 Classes (`crack`, `clip`, `break`, `surface`) |
| **Accuracy / Box mAP@50** | 98.92% | **99.50%** |
| **Mask mAP@50** | N/A | **95.72%** |
| **Mean IoU (mIoU)** | N/A | **95.72%** |
| **Dice Coefficient** | N/A | **97.81%** |
| **Defect Localization** | None | Exact Bounding Box $(x, y, w, h)$ |
| **Defect Contour Tracing**| None | Exact Polygon Mask & Physical Area |
| **Severity Scoring** | None | Automated (Minor, Moderate, Critical) |
| **CPU Latency per Frame** | 191 ms (total feature extraction) | **37.7 ms (26.5 FPS Real-Time)** |
| **Model Disk Footprint** | 12 MB (Model + Weights + Scaler) | **7.0 MB (`railway_yolov8n_seg.pt`)** |
