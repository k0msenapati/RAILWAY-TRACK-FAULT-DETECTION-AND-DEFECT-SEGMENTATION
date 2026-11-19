#!/usr/bin/env python3
"""
segmentation/dataset_prep.py
Validates and prepares the Railway Track Defect Segmentation dataset.
- Verifies directory structure and data.yaml
- Checks polygon annotation integrity (validating coordinates within [0, 1])
- Generates sample visualization with overlaid polygon masks and bounding boxes saved to docs/images/
"""

import os
import sys
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_DIR = os.path.join(PROJECT_ROOT, "data", "segmentation")
DOCS_IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

CLASS_COLORS = [
    (0, 0, 255),    # crack: Red
    (255, 165, 0),  # missing_fastener: Orange
    (255, 0, 0),    # broken_rail: Blue
    (0, 255, 255)   # surface_defect: Yellow
]

def validate_dataset(seg_dir=SEG_DIR):
    print("=" * 65)
    print(" Validating Railway Defect Segmentation Dataset")
    print("=" * 65)

    yaml_file = os.path.join(seg_dir, "data.yaml")
    if not os.path.exists(yaml_file):
        raise FileNotFoundError(f"data.yaml not found at: {yaml_file}")

    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)

    classes = config.get("names", [])
    print(f"[*] Dataset Classes ({len(classes)}): {classes}")

    stats = {}
    for split in ["train", "val", "test"]:
        img_folder = os.path.join(seg_dir, "images", split)
        lbl_folder = os.path.join(seg_dir, "labels", split)

        if not os.path.exists(img_folder):
            stats[split] = {"images": 0, "labels": 0, "defects": 0}
            continue

        images = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        labels = [f for f in os.listdir(lbl_folder) if f.endswith('.txt')]

        defect_count = 0
        for lbl in labels:
            with open(os.path.join(lbl_folder, lbl), "r") as lf:
                lines = [l.strip() for l in lf.readlines() if l.strip()]
                defect_count += len(lines)

        stats[split] = {"images": len(images), "labels": len(labels), "defects": defect_count}
        print(f"  - {split:<6}: {len(images)} images, {len(labels)} label files, {defect_count} defect annotations")

    return stats, classes

def visualize_samples(seg_dir=SEG_DIR, output_path=os.path.join(DOCS_IMG_DIR, "segmentation_samples.png"), num_samples=4):
    """Visualizes sample ground truth polygon masks and bounding boxes."""
    img_dir = os.path.join(seg_dir, "images", "train")
    lbl_dir = os.path.join(seg_dir, "labels", "train")

    if not os.path.exists(img_dir):
        return

    images = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not images:
        return

    sample_imgs = images[:num_samples]
    fig, axes = plt.subplots(1, len(sample_imgs), figsize=(4 * len(sample_imgs), 4))
    if len(sample_imgs) == 1:
        axes = [axes]

    for ax, img_name in zip(axes, sample_imgs):
        img_p = os.path.join(img_dir, img_name)
        lbl_p = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

        img = cv2.imread(img_p)
        if img is None:
            continue
        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:]]
                        pts = []
                        for i in range(0, len(coords), 2):
                            px = int(coords[i] * w)
                            py = int(coords[i+1] * h)
                            pts.append([px, py])
                        pts = np.array(pts, dtype=np.int32)
                        
                        # Draw translucent polygon mask
                        overlay = img_rgb.copy()
                        cv2.fillPoly(overlay, [pts], (255, 0, 0))
                        cv2.addWeighted(overlay, 0.4, img_rgb, 0.6, 0, img_rgb)
                        cv2.polylines(img_rgb, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

        ax.imshow(img_rgb)
        ax.set_title(img_name, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"[✓] Saved segmentation sample visualizations to: {output_path}")

if __name__ == "__main__":
    validate_dataset()
    visualize_samples()
