"""
us_preprocessing.py — Canonical Ultrasound Preprocessing Module
================================================================
Single source-of-truth preprocessing function used by:
  - augment_dataset.py
  - ultrasound_efficientnet.py
  - gradcam_efficientnet.py
  - gradcam_multimodal.py

Pipeline (applied identically at train, val, test, and Grad-CAM time):
  1. Load image (BGR via OpenCV)
  2. Crop: 15 %–90 % height, 10 %–90 % width  → removes machine UI / text borders
  3. Resize to target_size
  4. Convert to grayscale
  5. CLAHE (clipLimit=2.0, tileGridSize=8×8)  → local contrast enhancement
  6. Stack gray × 3  → 3-channel for pretrained CNN
  7. Return UINT8 array  (do NOT divide by 255 here;
     let efficientnet.preprocess_input() handle normalisation)

For models that need [0,1] range (e.g. DenseNet with manual /255),
call  preprocess_bladder_image(..., normalize=True).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ── Default crop / contrast constants ────────────────────────────────────────
CROP_H_START: float = 0.15   # crop top 15 %
CROP_H_END:   float = 0.90   # keep up to 90 % of height
CROP_W_START: float = 0.10   # crop left 10 %
CROP_W_END:   float = 0.90   # keep up to 90 % of width

CLAHE_CLIP_LIMIT: float      = 2.0
CLAHE_TILE_GRID:  Tuple[int, int] = (8, 8)

# Default target size for EfficientNetB3
DEFAULT_TARGET_SIZE: Tuple[int, int] = (260, 260)


# ── Canonical config dict (save to JSON for fusion script compatibility) ──────
PREPROCESSING_CONFIG: dict = {
    "crop_h_start":    CROP_H_START,
    "crop_h_end":      CROP_H_END,
    "crop_w_start":    CROP_W_START,
    "crop_w_end":      CROP_W_END,
    "clahe_clip_limit": CLAHE_CLIP_LIMIT,
    "clahe_tile_grid":  list(CLAHE_TILE_GRID),
    "default_target_size": list(DEFAULT_TARGET_SIZE),
    "normalize_for_efficientnet": True,
    "description": (
        "Crop 15-90% height / 10-90% width, grayscale, CLAHE, "
        "stack x3, then efficientnet.preprocess_input()"
    ),
}


def preprocess_bladder_image(
    path: str,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    normalize: bool = False,
) -> np.ndarray:
    """
    Preprocess a single bladder ultrasound image.

    Parameters
    ----------
    path : str
        Full path to image file.
    target_size : (H, W)
        Output spatial dimensions. Default 260×260 for EfficientNetB3.
        Use (224, 224) for DenseNet121 / comparison models.
    normalize : bool
        If True, divide by 255 → float32 in [0, 1].
        If False (default), return uint8 [0, 255] — let
        efficientnet.preprocess_input() handle normalisation.

    Returns
    -------
    np.ndarray
        (H, W, 3) float32 if normalize else uint8.
        Returns a zero array on load failure (with a console warning).
    """
    img = cv2.imread(path)
    if img is None:
        print(f"  [WARN] Cannot load image: {path}")
        dtype = np.float32 if normalize else np.uint8
        return np.zeros((*target_size, 3), dtype=dtype)

    h, w = img.shape[:2]

    # ── Step 2: Crop ──────────────────────────────────────────────────────────
    y1 = int(h * CROP_H_START)
    y2 = int(h * CROP_H_END)
    x1 = int(w * CROP_W_START)
    x2 = int(w * CROP_W_END)
    img = img[y1:y2, x1:x2]

    # ── Step 3: Resize ────────────────────────────────────────────────────────
    img = cv2.resize(img, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_AREA)

    # ── Step 4: Grayscale ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── Step 5: CLAHE ─────────────────────────────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_GRID)
    gray = clahe.apply(gray)

    # ── Step 6: Stack to 3-channel ────────────────────────────────────────────
    img_3ch = np.stack([gray, gray, gray], axis=-1)   # (H, W, 3)  uint8

    if normalize:
        return img_3ch.astype(np.float32) / 255.0

    return img_3ch.astype(np.float32)   # float32 uint8-range for preprocess_input


def save_preprocessing_config(out_path: str) -> None:
    """Write PREPROCESSING_CONFIG to a JSON file."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(PREPROCESSING_CONFIG, f, indent=2)
    print(f"  ✓ Preprocessing config saved → {out_path}")

preprocess_ultrasound_v1 = preprocess_bladder_image

def preprocess_ultrasound_v2(path, target_size=(260, 260)):
    import cv2
    import numpy as np

    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]

    # Step 1: Aggressive crop — remove 20% top, 15% bottom, 15% each side
    # This removes machine UI, text overlays, measurement markers
    top    = int(h * 0.20)
    bottom = int(h * 0.85)
    left   = int(w * 0.15)
    right  = int(w * 0.85)
    img = img[top:bottom, left:right]

    # Step 2: Elliptical mask — forces model to focus on center (bladder region)
    # Blacks out borders completely so model CANNOT attend to edge artifacts
    h2, w2 = img.shape[:2]
    mask = np.zeros((h2, w2), dtype=np.uint8)
    center = (w2 // 2, h2 // 2)
    axes = (int(w2 * 0.46), int(h2 * 0.46))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)

    # Step 3: Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 4: CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 5: Convert back to 3-channel RGB (required for EfficientNetB3)
    rgb = cv2.merge([enhanced, enhanced, enhanced])

    # Step 6: Resize to EfficientNetB3 native size
    rgb = cv2.resize(rgb, target_size)

    # Step 7: Normalize to [0, 1]
    return rgb.astype(np.float32) / 255.0

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import random
    import glob

    print("=== Testing preprocessor_v2 ===")
    sample_images = glob.glob("data/balanced/*/*.jpg")
    if len(sample_images) >= 3:
        samples = random.sample(sample_images, 3)
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        fig.suptitle("Preprocessing v2 Comparison", fontsize=16)
        
        for i, path in enumerate(samples):
            # Original
            orig = cv2.imread(path)
            orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            
            # Processed
            processed = preprocess_ultrasound_v2(path)
            
            axes[i, 0].imshow(orig_rgb)
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")
            
            axes[i, 1].imshow(processed)
            axes[i, 1].set_title("Preprocessed v2")
            axes[i, 1].axis("off")
            
        plt.tight_layout()
        os.makedirs("results/graphs", exist_ok=True)
        plt.savefig("results/graphs/preprocessing_comparison.png")
        print("Saved preprocessing comparison to results/graphs/preprocessing_comparison.png")
