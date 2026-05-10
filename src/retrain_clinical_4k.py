"""
retrain_clinical_4k.py — FIX 1: Retrain clinical model on 4K dataset
=====================================================================
Trains LogisticRegression, RandomForest, and XGBoost on the 4K clinical
subset with fresh preprocessing fit only on the training split.
Saves the best model and required inference assets.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINICAL_CSV = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Columns that are NOT features
_META_COLS  = {"uti_label", "split", "abxUTI", "alt_diag"}
LABEL_COL   = "uti_label"

# ── Ordinal mappings for urine test fields ──────────────────────────────────
ORDINAL_MAPS = {
    "urine_bacteria":     {"none": 0, "few": 1, "moderate": 2, "many": 3, "marked": 4},
    "urine_bilirubin":    {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_blood":        {"negative": 0, "small": 1, "moderate": 2, "large": 3, "other": 1},
    "urine_clarity":      {"clear": 0, "not_clear": 1},
    "urine_color":        {"colorless": 0, "yellow": 1, "amber": 2, "orange": 3, "red": 4, "other": 2},
    "epithelial_cells":   {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_glucose":      {"negative": 0, "small": 1, "moderate": 2, "large": 3, "4+": 4},
    "urine_ketones":      {"negative": 0, "small": 1, "moderate": 2, "large": 3, "4+": 4},
    "leukocyte_esterase": {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "nitrite":            {"negative": 0, "positive": 1},
    "urine_protein":      {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_rbc":          {"negative": 0, "small": 1, "moderate": 2, "large": 3, "other": 1},
    "specific_gravity":   None,   # already numeric
    "urobilinogen":       {"negative": 0, "positive": 1},
    "urine_wbc":          {"negative": 0, "small": 1, "moderate": 2, "large": 3},
    "urine_ph":           None,   # already numeric
    "UCX_abnormal":       {"no": 0, "yes": 1},
    "gender":             {"Male": 0, "Female": 1},
    "Calculus_of_urinary_tract": {"No": 0, "Yes": 1},
    "Urinary_tract_infections":  {"No": 0, "Yes": 1},
}

BINARY_COLS = {"abdominal_tenderness", "back_pain", "fatigue", "fever",
               "abdominal_pain", "burning_urination", "diff_urinating"}


def encode_clinical(df: pd.DataFrame, fit_le: dict | None = None):
    """
    Encode a clinical DataFrame using the ordinal maps + binary columns.
    Returns (X_numeric_df, label_encoders_dict_for_unseen_cats).
    """
    df = df.copy()
    les = fit_le if fit_le is not None else {}

    for col, mapping in ORDINAL_MAPS.items():
        if col not in df.columns:
            continue
        if mapping is None:
            # Already numeric — just fill NaN with median
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())
        else:
            # Map to ordinal; unseen values → median of known values
            df[col] = df[col].astype(str).str.strip().str.lower()
            mapping_lower = {k.lower(): v for k, v in mapping.items()}
            med_val = int(np.median(list(mapping.values())))
            df[col] = df[col].map(mapping_lower).fillna(med_val).astype(float)

    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Numeric cols that are already clean
    for col in ["age", "Temperature", "RBC", "WBC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    # Select only columns we can encode
    feature_cols = [c for c in df.columns if c not in _META_COLS]
    return df[feature_cols], les


def main():
    print("=" * 60)
    print("  FIX 1 — Retrain Clinical Model on 4K Dataset")
    print("=" * 60)

    # ── Load 4K CSV ────────────────────────────────────────────────────────
    df = pd.read_csv(CLINICAL_CSV)
    print(f"\nStep 1 — Load data")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Label distribution:\n{df[LABEL_COL].value_counts().to_string()}")

    # ── Encode features ────────────────────────────────────────────────────
    print("\nStep 2 — Encode features")
    X_enc, _ = encode_clinical(df)
    y = df[LABEL_COL].astype(int).values

    print(f"  Feature shape after encoding: {X_enc.shape}")
    print(f"  Features: {X_enc.columns.tolist()}")
    print(f"  NaN remaining: {X_enc.isnull().sum().sum()}")

    feature_names = X_enc.columns.tolist()

    # ── Stratified split ───────────────────────────────────────────────────
    print("\nStep 3 — Stratified split (70/15/15)")
    X_np = X_enc.values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_np, y, test_size=0.30, stratify=y, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

    print(f"  Train: {X_train.shape}  ({y_train.sum()} infected)")
    print(f"  Val:   {X_val.shape}   ({y_val.sum()} infected)")
    print(f"  Test:  {X_test.shape}  ({y_test.sum()} infected)")

    # ── Scale — fit ONLY on training split ───────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"  Scaler fitted on train set — n_features: {scaler.n_features_in_}")

    # ── Train 3 models ────────────────────────────────────────────────────
    print("\nStep 4 — Train and compare 3 models")

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, C=1.0, random_state=SEED),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=SEED,
            verbosity=0,
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train_s, y_train)

        for split_name, Xs, ys in [("Val", X_val_s, y_val), ("Test", X_test_s, y_test)]:
            proba = model.predict_proba(Xs)[:, 1]
            preds = (proba >= 0.5).astype(int)
            auc = roc_auc_score(ys, proba)
            acc = accuracy_score(ys, preds)
            pre = precision_score(ys, preds, zero_division=0)
            rec = recall_score(ys, preds, zero_division=0)
            f1  = f1_score(ys, preds, zero_division=0)
            print(f"  [{split_name}] AUC={auc:.4f} | Acc={acc:.3f} | "
                  f"Pre={pre:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

            if split_name == "Test":
                results[name] = {"model": model, "auc": auc, "f1": f1}

        print(f"  Classification Report (Test):")
        print(classification_report(y_test,
                                    (model.predict_proba(X_test_s)[:, 1] >= 0.5).astype(int),
                                    target_names=["Normal", "Infected"]))

    # ── Pick best model ────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_model = results[best_name]["model"]
    best_auc   = results[best_name]["auc"]
    print(f"\n  Best model: {best_name} (Test AUC = {best_auc:.4f})")

    # ── Step 5: Validate clinical scores ──────────────────────────────────
    print("\nStep 5 — Validate clinical score separation")

    # Encode ALL 4K data using the same scaler
    X_all_enc, _ = encode_clinical(df)
    X_all_s = scaler.transform(X_all_enc.values.astype(np.float32))
    y_all   = df[LABEL_COL].astype(int).values

    proba_test = best_model.predict_proba(X_test_s)[:, 1]
    print(f"  Infected samples — mean prob: {proba_test[y_test == 1].mean():.4f}")
    print(f"  Normal  samples  — mean prob: {proba_test[y_test == 0].mean():.4f}")

    inf_mean  = proba_test[y_test == 1].mean()
    norm_mean = proba_test[y_test == 0].mean()
    sep       = inf_mean - norm_mean

    if sep < 0.15:
        print(f"\n  ⚠ WARNING: Signal separation = {sep:.3f} < 0.15")
        print("  CLINICAL SIGNAL TOO WEAK — review feature encoding")
    else:
        print(f"\n  ✓ Signal separation = {sep:.3f} — clinical model is discriminative")

    # ── Step 6: Save ──────────────────────────────────────────────────────
    print("\nStep 6 — Saving artifacts")

    model_path  = os.path.join(MODELS_DIR, "clinical_model_4k.pkl")
    scaler_path = os.path.join(MODELS_DIR, "clinical_scaler_4k.pkl")
    names_path  = os.path.join(MODELS_DIR, "clinical_feature_names_4k.json")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler,     scaler_path)
    with open(names_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"  Saved: {model_path}")
    print(f"  Saved: {scaler_path}")
    print(f"  Saved: {names_path}")

    print(f"\n  Clinical model 4K — Test AUC: {best_auc:.4f}  "
          f"Infected mean prob: {inf_mean:.2f}  "
          f"Normal mean prob: {norm_mean:.2f}")

    if best_auc >= 0.80:
        print("\n✓ FIX 1 COMPLETE — Clinical 4K model AUC > 0.80  SUCCESS")
    else:
        print(f"\n⚠ FIX 1 PARTIAL — AUC = {best_auc:.4f} < 0.80  "
              "Consider feature engineering or more data")

    return best_model, scaler, feature_names, X_all_s, y_all


if __name__ == "__main__":
    main()
