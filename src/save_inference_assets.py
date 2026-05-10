"""
save_inference_assets.py — Package models for Flask web app
============================================================
Reads the exact feature names from the XGBoost booster and writes
models/inference_config.json for the Flask inference pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import joblib
import numpy as np
import pandas as pd

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")


def save_inference_config():
    print("=" * 60)
    print("  TASK 4a — Save Inference Configuration")
    print("=" * 60)

    # Load clinical model
    payload   = joblib.load(os.path.join(MODELS_DIR, "clinical_model.pkl"))
    model_obj = payload["model"] if isinstance(payload, dict) else payload

    # Get feature names from the XGBoost booster (ground truth)
    booster_features = model_obj.get_booster().feature_names
    if booster_features is None:
        # Fallback: read from CSV
        csv_path = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")
        _META = {"uti_label", "split", "abxUTI", "alt_diag"}
        sample = pd.read_csv(csv_path, nrows=1)
        booster_features = [c for c in sample.columns if c not in _META]

    print(f"  Clinical features (booster): {len(booster_features)}")

    # Read fusion AUC if available
    fusion_auc = "TBD"
    comp_csv = os.path.join(METRICS_DIR, "fusion_comparison.csv")
    if os.path.exists(comp_csv):
        comp_df = pd.read_csv(comp_csv)
        fusion_row = comp_df[comp_df["model"].str.contains("Fusion", case=False)]
        if len(fusion_row) > 0:
            fusion_auc = float(fusion_row.iloc[0]["AUC"])

    config = {
        "clinical_model":        "models/clinical_model.pkl",
        "image_model":           "models/ultrasound_efficientnet_best.keras",
        "fusion_model":          "models/fusion_model_best.keras",
        "clinical_features":     booster_features,
        "image_input_size":      [260, 260],
        "image_preprocessing":   "v2_elliptical_clahe",
        "fusion_threshold":      0.50,
        "clinical_feature_dim":  len(booster_features),
        "image_embedding_dim":   256,
        "model_version":         "1.0",
        "trained_on":            "4000 paired samples",
        "clinical_auc":          0.8415,
        "image_auc":             0.8843,
        "fusion_auc":            fusion_auc,
    }

    config_path = os.path.join(MODELS_DIR, "inference_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"  Saved: {config_path}")
    print(f"  Feature dim: {len(booster_features)}")
    print(f"  Fusion AUC:  {fusion_auc}")
    print(f"\n✓ TASK 4a COMPLETE")
    return config_path


if __name__ == "__main__":
    save_inference_config()
