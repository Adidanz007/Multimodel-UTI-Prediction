"""
augment_dataset.py — Leakage-Free Augmentation (Abnormal Only, Train Split Only)
==================================================================================
PURPOSE
  Balance the training set by augmenting ONLY abnormal images that land in the
  training split.  This prevents data leakage that would occur if augmented
  copies of the same source image appeared in both train and test sets.

FLOW
  1. Collect all RAW images  (normal=2411, abnormal=994)
  2. Stratified 70/15/15 split  (seed=42 — same seed used in training script)
  3. Clear the OLD augmented/abnormal directory
  4. Augment ONLY train-split abnormal images
       target count = train-normal count  ->  balanced training set
  5. Save  data/augmented/abnormal/aug_<name>.jpg
  6. Save split manifest  results/metrics/raw_data_split.csv
       (path, label, split)  — training script reads this to avoid re-splitting
  7. Print final counts and verify ? 5 000 total images on disk

DATASET MATH (approximate)
  Raw                : 2411 normal + 994 abnormal = 3405 total
  Train split (70%)  : ~1688 normal + ~696 abnormal
  Val   split (15%)  : ~362  normal + ~149 abnormal
  Test  split (15%)  : ~361  normal + ~149 abnormal
  Augmented needed   : 1688 – 696 = ~992  (train abnormal -> match train normal)
  ?????????????????????????????????????????????????????
  On-disk total      : 3405 raw + 992 aug = ~4397  (under 5000 ?)
  Train balance      : ~1688 normal  vs  ~1688 abnormal  (1:1 ?)
  Val / Test         : raw only — clean, unbiased evaluation
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path

import albumentations as A
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ?? Seed ?????????????????????????????????????????????????????????????????????
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ?? Paths ?????????????????????????????????????????????????????????????????????
BASE_DIR        = Path(__file__).resolve().parent.parent
RAW_NORMAL      = BASE_DIR / "data" / "raw" / "ultrasound_images" / "normal"
RAW_ABNORMAL    = BASE_DIR / "data" / "raw" / "ultrasound_images" / "abnormal"
AUG_DIR         = BASE_DIR / "data" / "augmented"
AUG_ABNORMAL    = AUG_DIR  / "abnormal"
AUG_NORMAL      = AUG_DIR  / "normal"     # will be cleared — not needed
SPLIT_CSV       = BASE_DIR / "results" / "metrics" / "raw_data_split.csv"
GRAPHS_DIR      = BASE_DIR / "results" / "graphs"
IMG_EXTS        = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ?? Augmentation pipeline ?????????????????????????????????????????????????????
def build_augmenter() -> A.Compose:
    """
    Conservative augmentation pipeline for bladder ultrasound.
    Each transform fires independently -> different combo every time.
    Excluded: heavy zoom, wide rotations, colour jitter, cutout.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            p=0.5,
        ),
        A.ElasticTransform(alpha=1.0, sigma=5.0, p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
    ])


# ?? Helpers ???????????????????????????????????????????????????????????????????
def collect(directory: Path) -> list[Path]:
    return sorted([p for p in directory.rglob("*")
                   if p.suffix.lower() in IMG_EXTS])


def load_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def save_jpg(img_rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ?? Main ??????????????????????????????????????????????????????????????????????
def main() -> None:
    print("=" * 70)
    print("  AUGMENT DATASET — Leakage-Free, Train Abnormal Only")
    print("=" * 70)

    # ?? 1. Collect raw images ?????????????????????????????????????????????????
    print("\n[DIR] Collecting raw images …")
    normal_paths   = collect(RAW_NORMAL)
    abnormal_paths = collect(RAW_ABNORMAL)
    n_normal  = len(normal_paths)
    n_abnormal = len(abnormal_paths)
    print(f"  Raw normal   : {n_normal}")
    print(f"  Raw abnormal : {n_abnormal}")
    print(f"  Total raw    : {n_normal + n_abnormal}")

    if n_normal == 0 or n_abnormal == 0:
        raise RuntimeError("Raw images not found — check paths.")

    # ?? 2. Stratified 70/15/15 split on raw images (SAME SEED as training) ???
    print("\n[SPLIT]  Stratified 70 / 15 / 15 split on raw images (seed=42) …")

    all_paths  = [(str(p), 0) for p in normal_paths] + \
                 [(str(p), 1) for p in abnormal_paths]
    paths_arr  = [x[0] for x in all_paths]
    labels_arr = [x[1] for x in all_paths]

    train_p, rem_p, train_l, rem_l = train_test_split(
        paths_arr, labels_arr,
        test_size=0.30, stratify=labels_arr, random_state=SEED,
    )
    val_p, test_p, val_l, test_l = train_test_split(
        rem_p, rem_l,
        test_size=0.50, stratify=rem_l, random_state=SEED,
    )

    n_train_normal   = sum(1 for l in train_l if l == 0)
    n_train_abnormal = sum(1 for l in train_l if l == 1)
    n_val_normal     = sum(1 for l in val_l   if l == 0)
    n_val_abnormal   = sum(1 for l in val_l   if l == 1)
    n_test_normal    = sum(1 for l in test_l  if l == 0)
    n_test_abnormal  = sum(1 for l in test_l  if l == 1)

    print(f"  Train : {len(train_p):>5}  "
          f"(normal={n_train_normal}, abnormal={n_train_abnormal})")
    print(f"  Val   : {len(val_p):>5}  "
          f"(normal={n_val_normal},  abnormal={n_val_abnormal})")
    print(f"  Test  : {len(test_p):>5}  "
          f"(normal={n_test_normal},  abnormal={n_test_abnormal})")

    # ?? 3. Save split manifest ????????????????????????????????????????????????
    print(f"\n[SAVE] Saving split manifest -> {SPLIT_CSV}")
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        [(p, l, "train") for p, l in zip(train_p, train_l)] +
        [(p, l, "val")   for p, l in zip(val_p,   val_l  )] +
        [(p, l, "test")  for p, l in zip(test_p,  test_l )]
    )
    pd.DataFrame(rows, columns=["image_path", "label", "split"]
                 ).to_csv(str(SPLIT_CSV), index=False)
    print(f"  ? Saved {len(rows)} rows.")

    # ?? 4. Clear old augmented directories ???????????????????????????????????
    print("\n[DEL]  Clearing old augmented directories …")
    for d in [AUG_ABNORMAL, AUG_NORMAL]:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Cleared {d}")
    AUG_ABNORMAL.mkdir(parents=True, exist_ok=True)
    AUG_NORMAL.mkdir(parents=True, exist_ok=True)    # keep empty, won't be used

    # ?? 5. Augment train-split abnormal images ????????????????????????????????
    n_aug_needed = n_train_normal - n_train_abnormal   # target balance
    print(f"\n[RUN] Augmenting train abnormal images …")
    print(f"   Train normal   : {n_train_normal}")
    print(f"   Train abnormal : {n_train_abnormal}")
    print(f"   Aug needed     : {n_aug_needed}  (to reach 1:1 balance in train)")

    # Collect train-split abnormal paths
    train_abnormal_paths = [p for p, l in zip(train_p, train_l) if l == 1]
    print(f"   Source images  : {len(train_abnormal_paths)} train-abnormal")

    augmenter  = build_augmenter()
    aug_counter = 0
    sample_images: list[np.ndarray] = []   # for grid visualisation

    # Distribute augmentations evenly: each source image gets at least floor copies,
    # and (n_aug_needed % n_train_abnormal) images get one extra.
    base_copies = n_aug_needed // len(train_abnormal_paths)
    extra       = n_aug_needed  %  len(train_abnormal_paths)

    print(f"   Base copies/img: {base_copies}, {extra} images get +1 extra")

    for img_idx, img_path in enumerate(train_abnormal_paths):
        n_copies = base_copies + (1 if img_idx < extra else 0)
        if n_copies == 0:
            continue

        rgb = load_rgb(Path(img_path))
        if rgb is None:
            print(f"  [WARN] Cannot load: {img_path}")
            continue

        for _ in range(n_copies):
            aug_img = augmenter(image=rgb)["image"]
            out_name = f"aug_{aug_counter:04d}_{Path(img_path).stem}.jpg"
            save_jpg(aug_img, AUG_ABNORMAL / out_name)

            if len(sample_images) < 8:
                sample_images.append(aug_img)

            aug_counter += 1

        if (img_idx + 1) % 100 == 0:
            print(f"  … processed {img_idx + 1}/{len(train_abnormal_paths)} "
                  f"source images  ({aug_counter} augmented)")

    print(f"  ? Augmented {aug_counter} abnormal images saved.")

    # ?? 6. Final count summary ????????????????????????????????????????????????
    on_disk_aug_ab  = len(collect(AUG_ABNORMAL))
    total_on_disk   = n_normal + n_abnormal + on_disk_aug_ab

    print("\n" + "=" * 70)
    print("  FINAL DATASET SUMMARY")
    print("=" * 70)
    print(f"  Raw normal          : {n_normal:>5}")
    print(f"  Raw abnormal        : {n_abnormal:>5}")
    print(f"  Augmented abnormal  : {on_disk_aug_ab:>5}  "
          f"(train-split source only — NO leakage)")
    print(f"  ??????????????????????????????????")
    print(f"  Total on disk       : {total_on_disk:>5}  "
          f"({'? ?5000' if total_on_disk <= 5000 else '[!] exceeds 5000'})")
    print()
    print("  TRAINING SET (after combining raw train + augmented):")
    print(f"    Normal   (raw)    : {n_train_normal}")
    print(f"    Abnormal (raw)    : {n_train_abnormal}")
    print(f"    Abnormal (aug)    : {aug_counter}")
    print(f"    Ratio             : {n_train_normal / (n_train_abnormal + aug_counter):.2f}:1")
    print()
    print("  VAL SET  (raw only — clean):")
    print(f"    Normal  : {n_val_normal},  Abnormal: {n_val_abnormal}")
    print()
    print("  TEST SET (raw only — clean, unbiased):")
    print(f"    Normal  : {n_test_normal},  Abnormal: {n_test_abnormal}")
    print("=" * 70)

    # ?? 7. Sample grid visualisation ?????????????????????????????????????????
    print("\n?  Saving augmentation sample grid …")
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
    _save_sample_grid(sample_images, GRAPHS_DIR / "augmentation_samples_train_abnormal.png")

    print("\n[OK]  Augmentation complete! Run ultrasound_efficientnet.py next.")


def _save_sample_grid(samples: list[np.ndarray], out_path: Path) -> None:
    n = min(len(samples), 8)
    if n == 0:
        return
    rows, cols = 2, 4
    fig, axes = plt.subplots(rows, cols, figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")
    for idx, ax in enumerate(axes.flat):
        if idx < n:
            ax.imshow(cv2.resize(samples[idx], (224, 224)))
            ax.set_title("Train Abnormal (aug)", color="#ff6b6b",
                         fontsize=9, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Augmented Train-Abnormal Samples (leakage-free)",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ? Grid saved -> {out_path}")


if __name__ == "__main__":
    main()
