"""
build_fusion_dataset.py — Create deterministic clinical-image pairing table
=============================================================================
Pairs 4,000 clinical rows (from reduced_4000_dataset.csv) with 4,000 balanced
bladder images by label alignment: infected→abnormal, non-infected→normal.

Output: data/processed/fusion_pairs.csv
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINICAL_CSV = os.path.join(BASE_DIR, "data", "processed", "reduced_4000_dataset.csv")
BALANCED_DIR = os.path.join(BASE_DIR, "data", "balanced")
OUTPUT_CSV   = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")


def build_fusion_dataset():
    print("=" * 60)
    print("  TASK 1 — Build Master Paired Fusion Table")
    print("=" * 60)

    # ── Step 1: Load clinical data ──────────────────────────────────────────
    print("\n[1] Loading clinical CSV ...")
    if not os.path.exists(CLINICAL_CSV):
        # Fallback name
        alt = os.path.join(BASE_DIR, "data", "processed", "clinical_4k.csv")
        if os.path.exists(alt):
            clinical_csv = alt
        else:
            raise FileNotFoundError(f"Clinical CSV not found at {CLINICAL_CSV}")
    else:
        clinical_csv = CLINICAL_CSV

    df = pd.read_csv(clinical_csv)
    print(f"  Clinical CSV shape: {df.shape}")
    print(f"  Label column: uti_label")
    print(f"  Label distribution:\n{df['uti_label'].value_counts().to_string()}")

    # Sort: infected (label=1) first, then non-infected (label=0)
    df_infected = df[df["uti_label"] == 1].reset_index(drop=True)
    df_normal   = df[df["uti_label"] == 0].reset_index(drop=True)
    print(f"\n  Infected rows:     {len(df_infected)}")
    print(f"  Non-infected rows: {len(df_normal)}")

    # Cap at 2000 each
    if len(df_infected) > 2000:
        df_infected = df_infected.sample(n=2000, random_state=SEED).reset_index(drop=True)
    if len(df_normal) > 2000:
        df_normal = df_normal.sample(n=2000, random_state=SEED).reset_index(drop=True)

    # Concatenate: infected first (rows 0-1999), normal second (rows 2000-3999)
    df_sorted = pd.concat([df_infected, df_normal], ignore_index=True)
    print(f"  Final sorted clinical rows: {len(df_sorted)}")

    # ── Step 2: Load image file lists ───────────────────────────────────────
    print("\n[2] Loading image file lists ...")
    abnormal_dir = os.path.join(BALANCED_DIR, "abnormal")
    normal_dir   = os.path.join(BALANCED_DIR, "normal")

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    abnormal_imgs = sorted([
        f for f in os.listdir(abnormal_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    normal_imgs = sorted([
        f for f in os.listdir(normal_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])

    print(f"  Abnormal images: {len(abnormal_imgs)}")
    print(f"  Normal images:   {len(normal_imgs)}")

    # Ensure we have enough
    n_infected = min(len(df_infected), len(abnormal_imgs))
    n_normal   = min(len(df_normal),   len(normal_imgs))

    # ── Step 3: Create pairing table ────────────────────────────────────────
    print("\n[3] Creating deterministic index-based pairing ...")

    rows = []
    # Pair infected clinical rows with abnormal images
    for i in range(n_infected):
        abs_path = os.path.abspath(os.path.join(abnormal_dir, abnormal_imgs[i]))
        rows.append({
            "pair_id": i,
            "clinical_row_index": i,
            "image_filename": abs_path,
            "label": 1,
        })

    # Pair non-infected clinical rows with normal images
    for i in range(n_normal):
        abs_path = os.path.abspath(os.path.join(normal_dir, normal_imgs[i]))
        rows.append({
            "pair_id": n_infected + i,
            "clinical_row_index": n_infected + i,
            "image_filename": abs_path,
            "label": 0,
        })

    pairs_df = pd.DataFrame(rows)
    print(f"  Total pairs created: {len(pairs_df)}")

    # ── Step 4: Stratified split ────────────────────────────────────────────
    print("\n[4] Performing stratified 70/15/15 split ...")

    train_idx, temp_idx = train_test_split(
        pairs_df.index, test_size=0.30, stratify=pairs_df["label"],
        random_state=SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=pairs_df.loc[temp_idx, "label"],
        random_state=SEED
    )

    pairs_df["split"] = ""
    pairs_df.loc[train_idx, "split"] = "train"
    pairs_df.loc[val_idx,   "split"] = "val"
    pairs_df.loc[test_idx,  "split"] = "test"

    # ── Step 5: Save and report ─────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pairs_df.to_csv(OUTPUT_CSV, index=False)

    # Also save the sorted clinical subset for embedding extraction
    sorted_csv = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")
    df_sorted.to_csv(sorted_csv, index=False)
    print(f"  Sorted clinical CSV saved: {sorted_csv}")

    # Report
    train_pairs = pairs_df[pairs_df["split"] == "train"]
    val_pairs   = pairs_df[pairs_df["split"] == "val"]
    test_pairs  = pairs_df[pairs_df["split"] == "test"]

    print(f"\n=== Fusion Pairs Report ===")
    print(f"Total pairs:        {len(pairs_df)}")
    print(f"Train pairs:        {len(train_pairs)}  "
          f"(Infected: {(train_pairs['label']==1).sum()}, "
          f"Normal: {(train_pairs['label']==0).sum()})")
    print(f"Validation pairs:    {len(val_pairs)}  "
          f"(Infected: {(val_pairs['label']==1).sum()}, "
          f"Normal: {(val_pairs['label']==0).sum()})")
    print(f"Test pairs:          {len(test_pairs)}  "
          f"(Infected: {(test_pairs['label']==1).sum()}, "
          f"Normal: {(test_pairs['label']==0).sum()})")
    print(f"\nPairing method: deterministic index alignment")
    print(f"Clinical source: {clinical_csv}")
    print(f"Image source:    {BALANCED_DIR}")
    print(f"Saved to:        {OUTPUT_CSV}")
    print(f"\n✓ TASK 1 COMPLETE")

    return pairs_df, df_sorted


if __name__ == "__main__":
    build_fusion_dataset()
