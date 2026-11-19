#!/usr/bin/env python3
"""
scripts/download_segmentation_data.py
Fetches or generates the Railway Defect Localization & Segmentation Dataset.
Formats data directly into YOLOv8-seg polygon annotation format:
- images/train, images/val, images/test
- labels/train, labels/val, labels/test (class x1 y1 x2 y2 ... xn yn)
- data.yaml specification
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEG_DIR = os.path.join(PROJECT_ROOT, "data", "segmentation")

CLASSES = ["crack", "missing_fastener", "broken_rail", "surface_defect"]

def ensure_dirs():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(SEG_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(SEG_DIR, "labels", split), exist_ok=True)

def write_data_yaml():
    yaml_path = os.path.join(SEG_DIR, "data.yaml")
    content = f"""# Railway Defect Segmentation Dataset Configuration (YOLOv8-seg)
path: {os.path.abspath(SEG_DIR)}
train: images/train
val: images/val
test: images/test

# Number of classes
nc: {len(CLASSES)}

# Class names
names: {CLASSES}
"""
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  [✓] Created YAML configuration at: {yaml_path}")

def count_segmentation_data():
    counts = {}
    total = 0
    for split in ["train", "val", "test"]:
        img_p = os.path.join(SEG_DIR, "images", split)
        lbl_p = os.path.join(SEG_DIR, "labels", split)
        if os.path.exists(img_p):
            imgs = len([f for f in os.listdir(img_p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            lbls = len([f for f in os.listdir(lbl_p) if f.endswith('.txt')])
            counts[split] = {"images": imgs, "labels": lbls}
            total += imgs
        else:
            counts[split] = {"images": 0, "labels": 0}
    return total, counts

def generate_realistic_segmentation_data():
    """
    Generates realistic railway images with ground truth polygon defect masks
    formatted for YOLOv8 segmentation.
    """
    print("\n[*] Generating polygon-annotated railway defect segmentation dataset...")
    try:
        import cv2
    except ImportError:
        print("[!] OpenCV not installed yet. Please run: pip install opencv-python-headless")
        return False

    split_targets = {
        "train": 120,
        "val": 30,
        "test": 15
    }

    for split, target_count in split_targets.items():
        img_dir = os.path.join(SEG_DIR, "images", split)
        lbl_dir = os.path.join(SEG_DIR, "labels", split)

        existing = len([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        needed = target_count - existing
        if needed <= 0:
            print(f"  [✓] {split}: {existing} samples already present.")
            continue

        print(f"  [*] Generating {needed} annotated segmentation samples for '{split}'...")
        for i in range(needed):
            idx = existing + i + 1
            w, h = 640, 640  # Standard YOLO input resolution

            # Ballast background with realistic texture
            base_val = np.random.randint(85, 120)
            img = np.full((h, w, 3), base_val, dtype=np.uint8)
            noise = np.random.normal(0, 15, (h, w, 3)).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Sleepers / Ties (horizontal concrete or wood ties)
            for tie_y in range(50, h, 140):
                tie_c = (np.random.randint(55, 75), np.random.randint(65, 85), np.random.randint(75, 95))
                cv2.rectangle(img, (20, tie_y), (w - 20, tie_y + 60), tie_c, -1)
                # Fasteners / Clips at rail junctions
                for fx in [int(w * 0.28), int(w * 0.38), int(w * 0.62), int(w * 0.72)]:
                    cv2.rectangle(img, (fx - 8, tie_y + 15), (fx + 8, tie_y + 45), (160, 160, 170), -1)

            # Two longitudinal steel rails
            rail_left_x = int(w * 0.33)
            rail_right_x = int(w * 0.67)
            rail_w = 40

            for rx in [rail_left_x, rail_right_x]:
                for dx in range(-rail_w // 2, rail_w // 2):
                    brightness = int(185 + 45 * np.cos(dx / (rail_w / 2) * np.pi / 2))
                    cv2.line(img, (rx + dx, 0), (rx + dx, h), (brightness, brightness, brightness + 15), 1)

            # Choose defect class to simulate
            cls_idx = np.random.choice([0, 1, 2, 3])
            cls_name = CLASSES[cls_idx]
            target_rail = np.random.choice([rail_left_x, rail_right_x])
            defect_y = np.random.randint(120, h - 120)

            polygon_pts = []

            if cls_name == "crack":
                # Jagged polygon along crack line
                cx, cy = target_rail - 10, defect_y
                pts = [(cx, cy), (cx + 5, cy - 8), (cx + 12, cy - 3), (cx + 20, cy + 12),
                       (cx + 16, cy + 22), (cx + 8, cy + 18), (cx, cy + 10)]
                poly = np.array(pts, dtype=np.int32)
                cv2.fillPoly(img, [poly], (25, 20, 20))
                polygon_pts = pts

            elif cls_name == "broken_rail":
                # Transverse rail break across rail head
                x1, y1 = target_rail - rail_w // 2, defect_y - 12
                x2, y2 = target_rail + rail_w // 2, defect_y + 12
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                poly = np.array(pts, dtype=np.int32)
                cv2.fillPoly(img, [poly], (15, 10, 10))
                polygon_pts = pts

            elif cls_name == "missing_fastener":
                # Missing fastener location (fastener tie junction)
                tie_nearest = min(range(50, h, 140), key=lambda ty: abs(ty - defect_y))
                fx = target_rail + (rail_w // 2 + 10)
                fy = tie_nearest + 30
                pts = [(fx - 12, fy - 12), (fx + 12, fy - 12), (fx + 12, fy + 12), (fx - 12, fy + 12)]
                poly = np.array(pts, dtype=np.int32)
                # Overwrite missing fastener area with empty tie/rust texture
                cv2.fillPoly(img, [poly], (40, 50, 70))
                polygon_pts = pts

            else:  # surface_defect / spall
                # Oval spall / shell defect on rail surface
                cx, cy = target_rail, defect_y
                rx_r, ry_r = 14, 24
                # Approximate ellipse with 8 polygon vertices
                pts = []
                for angle in np.linspace(0, 2 * np.pi, 9)[:-1]:
                    px = int(cx + rx_r * np.cos(angle) + np.random.randint(-2, 3))
                    py = int(cy + ry_r * np.sin(angle) + np.random.randint(-2, 3))
                    pts.append((px, py))
                poly = np.array(pts, dtype=np.int32)
                cv2.fillPoly(img, [poly], (50, 60, 120))
                polygon_pts = pts

            # Normalize polygon coordinates to [0, 1] for YOLOv8-seg
            normalized_poly = []
            for px, py in polygon_pts:
                norm_x = max(0.0, min(1.0, px / float(w)))
                norm_y = max(0.0, min(1.0, py / float(h)))
                normalized_poly.extend([f"{norm_x:.5f}", f"{norm_y:.5f}"])

            # Save image
            img_filename = f"seg_{split}_{idx:04d}.jpg"
            img_path = os.path.join(img_dir, img_filename)
            cv2.imwrite(img_path, img)

            # Save label in YOLO format: <class> <x1> <y1> <x2> <y2> ...
            lbl_filename = f"seg_{split}_{idx:04d}.txt"
            lbl_path = os.path.join(lbl_dir, lbl_filename)
            with open(lbl_path, "w") as lf:
                lf.write(f"{cls_idx} " + " ".join(normalized_poly) + "\n")

    return True

def main():
    print("=" * 65)
    print(" Railway Track Defect Segmentation - Dataset Setup")
    print(f" Target Classes: {CLASSES}")
    print("=" * 65)

    ensure_dirs()
    write_data_yaml()

    total, counts = count_segmentation_data()
    print(f"[*] Current segmentation dataset status: {total} images")
    for s in counts:
        print(f"  - {s}: {counts[s]['images']} images, {counts[s]['labels']} label files")

    if total >= 165:
        print("\n[✓] Segmentation dataset is already prepared!")
        return

    generate_realistic_segmentation_data()

    total, counts = count_segmentation_data()
    print("\n" + "=" * 65)
    print(f"[✓] Segmentation dataset ready! Total images: {total}")
    for s in counts:
        print(f"  - {s}: {counts[s]['images']} images, {counts[s]['labels']} labels")
    print(f"Dataset root: {SEG_DIR}")
    print("=" * 65)

if __name__ == "__main__":
    main()
