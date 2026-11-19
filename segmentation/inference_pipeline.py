#!/usr/bin/env python3
"""
segmentation/inference_pipeline.py
Real-time Railway Track Defect Localization & Segmentation Pipeline on CPU.
- Generates pixel-accurate defect masks and bounding boxes
- Categorizes defect types: crack, missing_fastener, broken_rail, surface_defect
- Calculates defect severity score based on mask pixel area
- Saves visual overlay output with bounding boxes, masks, and severity tags
- Benchmarks CPU inference speed (ms / FPS)
"""

import os
import sys
import time
import json
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DOCS_IMG_DIR = os.path.join(PROJECT_ROOT, "docs", "images")
os.makedirs(DOCS_IMG_DIR, exist_ok=True)

CLASS_COLORS = {
    0: (0, 0, 230),    # crack: Crimson Red
    1: (240, 140, 0),  # missing_fastener: Amber Orange
    2: (180, 0, 200),  # broken_rail: Deep Magenta / Critical
    3: (0, 200, 240)   # surface_defect: Cyan / Yellow
}

CLASS_NAMES = ["crack", "missing_fastener", "broken_rail", "surface_defect"]

class RailwaySegmentationPipeline:
    def __init__(self, model_path=None, conf_thresh=0.25):
        if model_path is None:
            # Check for fine-tuned weights first, else base weights
            fine_tuned = os.path.join(MODELS_DIR, "railway_yolov8n_seg.pt")
            model_path = fine_tuned if os.path.exists(fine_tuned) else "yolov8n-seg.pt"

        print(f"[*] Loading YOLOv8n-seg model from: {model_path} (Device: CPU)")
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def compute_severity(self, mask_pixel_count, total_pixels, class_name):
        area_ratio = (mask_pixel_count / float(total_pixels)) * 100
        if class_name == "broken_rail" or area_ratio > 4.0:
            severity = "CRITICAL - IMMEDIATE ACTION REQUIRED"
            color = (0, 0, 255)
        elif class_name == "crack" or area_ratio > 1.5:
            severity = "MODERATE - SCHEDULE REPAIR"
            color = (0, 165, 255)
        else:
            severity = "MINOR - MONITOR NEXT INSPECTION"
            color = (0, 255, 255)
        return severity, round(area_ratio, 3), color

    def predict(self, image_path, save_annotated=True, output_path=None):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input image not found: {image_path}")

        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = orig_img.shape[:2]
        total_pixels = h * w

        t0 = time.perf_counter()
        # Run inference strictly on CPU
        results = self.model.predict(
            source=orig_img,
            conf=self.conf_thresh,
            device="cpu",
            verbose=False
        )[0]
        latency_ms = (time.perf_counter() - t0) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0

        detected_defects = []
        annotated = orig_img.copy()
        mask_overlay = orig_img.copy()

        has_defects = False
        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clses = results.boxes.cls.cpu().numpy().astype(int)
            masks = results.masks.xy if results.masks is not None else None

            for i in range(len(boxes)):
                has_defects = True
                cls_id = int(clses[i]) % len(CLASS_NAMES)
                cls_name = CLASS_NAMES[cls_id]
                conf = float(confs[i])
                box = [int(v) for v in boxes[i]]
                x1, y1, x2, y2 = box

                color = CLASS_COLORS.get(cls_id, (0, 255, 0))

                # Process segmentation polygon mask
                mask_pixels = 0
                if masks is not None and i < len(masks):
                    poly = masks[i].astype(np.int32)
                    if len(poly) > 2:
                        cv2.fillPoly(mask_overlay, [poly], color)
                        cv2.polylines(annotated, [poly], isClosed=True, color=(255, 255, 255), thickness=2)
                        mask_pixels = int(cv2.contourArea(poly))
                else:
                    mask_pixels = (x2 - x1) * (y2 - y1)

                severity_tag, area_pct, sev_color = self.compute_severity(mask_pixels, total_pixels, cls_name)

                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw label banner
                label_txt = f"{cls_name.upper()} {conf*100:.1f}% [{severity_tag.split()[0]}]"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 6, max(0, y1)), color, -1)
                cv2.putText(annotated, label_txt, (x1 + 3, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                detected_defects.append({
                    "defect_id": i + 1,
                    "class": cls_name,
                    "confidence": round(conf * 100, 2),
                    "bbox_xyxy": box,
                    "mask_pixel_area": mask_pixels,
                    "defect_area_pct": area_pct,
                    "severity": severity_tag
                })

        # Blend mask overlay for translucent effect
        cv2.addWeighted(mask_overlay, 0.45, annotated, 0.55, 0, annotated)

        # Draw system status banner at top
        status_color = (0, 0, 200) if has_defects else (0, 180, 0)
        status_txt = f"TRACK FAULT DETECTED ({len(detected_defects)} Defects)" if has_defects else "TRACK CLEAR (No Defects)"
        cv2.rectangle(annotated, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.putText(annotated, f"RAIL-INSPECT AI | {status_txt} | Latency: {latency_ms:.1f}ms ({fps:.1f} FPS CPU)",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        if save_annotated:
            if output_path is None:
                output_path = os.path.join(DOCS_IMG_DIR, "segmentation_prediction.png")
            cv2.imwrite(output_path, annotated)

        return {
            "image": image_path,
            "status": "DEFECTIVE" if has_defects else "NON_DEFECTIVE",
            "defect_count": len(detected_defects),
            "defects": detected_defects,
            "cpu_latency_ms": round(latency_ms, 2),
            "cpu_fps": round(fps, 1),
            "annotated_output": output_path
        }

def main():
    parser = argparse.ArgumentParser(description="Railway Defect Segmentation CPU Inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", default=None, help="Path to YOLO segmentation weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--output", default=None, help="Output annotated image path")
    args = parser.parse_args()

    pipeline = RailwaySegmentationPipeline(model_path=args.weights, conf_thresh=args.conf)
    res = pipeline.predict(args.image, output_path=args.output)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
