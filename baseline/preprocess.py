#!/usr/bin/env python3
"""
baseline/preprocess.py
Preprocessing and data augmentation pipeline for railway track fault classification.
Preprocesses track images (resizing to 224x224, grayscale conversion, Gaussian blur, normalization)
and applies data augmentation (horizontal/vertical flip, rotations, brightness adjustment, noise, translation).
"""

import os
import random
import cv2
import numpy as np
from tqdm import tqdm

IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    """
    Standard preprocessing from baseline:
    - Resize to 224x224
    - Convert to grayscale
    - Gaussian blur (3x3 kernel)
    - Normalize pixel values to [0, 1]
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    img_resized = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = blurred / 255.0
    return normalized

def flip_image(image):
    flipped_h = cv2.flip(image, 1)  # Horizontal flip
    flipped_v = cv2.flip(image, 0)  # Vertical flip
    return [flipped_h, flipped_v]

def rotate_image(image, angle=None):
    if angle is None:
        angle = random.choice([90, 180, 270])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h))

def adjust_brightness(image, factor=None):
    if factor is None:
        factor = random.uniform(0.5, 1.5)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image):
    noise = np.random.normal(0, 25, image.shape).astype(np.int16)
    noisy = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy

def translate_image(image, tx=20, ty=20):
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, matrix, (w, h))

def augment_image(image):
    augmented = []
    augmented.extend(flip_image(image))
    augmented.append(rotate_image(image))
    augmented.append(adjust_brightness(image))
    augmented.append(add_noise(image))
    augmented.append(translate_image(image))
    return augmented

def augment_dataset(source_base, output_base, max_augment_per_image=5):
    """
    Iterates through dataset categories, applies augmentations, and saves to output directory.
    """
    for split in ["Train", "Validation"]:
        split_src = os.path.join(source_base, split)
        split_dst = os.path.join(output_base, split)

        if not os.path.exists(split_src):
            continue

        for cat in ["Defective", "Non defective"]:
            cat_src = os.path.join(split_src, cat)
            cat_dst = os.path.join(split_dst, cat)
            os.makedirs(cat_dst, exist_ok=True)

            if not os.path.exists(cat_src):
                continue

            images = [f for f in os.listdir(cat_src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"[*] Augmenting {split}/{cat}: {len(images)} base images...")

            for img_name in tqdm(images, desc=f"Augmenting {split}/{cat}"):
                img_path = os.path.join(cat_src, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                # Save original resized
                base_resized = cv2.resize(img, IMG_SIZE)
                cv2.imwrite(os.path.join(cat_dst, f"orig_{img_name}"), base_resized)

                # Generate and save augmentations
                aug_imgs = augment_image(base_resized)
                for idx, a_img in enumerate(aug_imgs[:max_augment_per_image]):
                    stem, ext = os.path.splitext(img_name)
                    cv2.imwrite(os.path.join(cat_dst, f"{stem}_aug{idx}{ext}"), a_img)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Preprocessing & Augmentation")
    parser.add_argument("--src", default="data/baseline", help="Source dataset path")
    parser.add_argument("--dst", default="data/baseline/Augmented", help="Augmented output path")
    args = parser.parse_args()

    augment_dataset(args.src, args.dst)
    print("[✓] Preprocessing & Data Augmentation completed!")
