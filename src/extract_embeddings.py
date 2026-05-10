"""
extract_embeddings.py — Extract embeddings from both pretrained models
======================================================================
The XGBoost model was trained with a numpy array (no column names in booster).
The clinical_scaler.pkl WAS fitted with a DataFrame → its feature_names_in_
attribute contains the exact 107 one-hot-encoded feature names the model expects.

Strategy (clinical side):
  1. Load scaler → read scaler.feature_names_in_ (107 exact feature names).
  2. One-hot encode the 4K clinical data with pd.get_dummies().
  3. Reindex to those 107 names (missing categories → 0, extras → dropped).
  4. Pass directly to XGBClassifier. NO scaling applied (XGBoost is tree-based).

Outputs:
  results/embeddings/clinical_proba.npy      — (4000, 2)
  results/embeddings/clinical_features.npy   — (4000, 107)
  results/embeddings/image_embeddings.npy    — (4000, 256)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_CSV   = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
CLINICAL_CSV = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")
CLINICAL_PKL = os.path.join(BASE_DIR, "models", "clinical_model.pkl")
SCALER_PKL   = os.path.join(BASE_DIR, "models", "clinical_scaler.pkl")
IMAGE_MODEL  = os.path.join(BASE_DIR, "models", "ultrasound_efficientnet_best.keras")
EMB_DIR      = os.path.join(BASE_DIR, "results", "embeddings")

_META_COLS = {"uti_label", "split", "abxUTI", "alt_diag"}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fill_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN: median for numeric, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            mode_val = df[col].mode(dropna=True)
            df[col] = df[col].fillna(
                mode_val.iloc[0] if len(mode_val) > 0 else "unknown"
            )
    return df


def _get_schema_from_scaler() -> list[str] | None:
    """
    Load the clinical scaler and return its feature_names_in_ list.
    The scaler was fitted on the one-hot-encoded training data with a DataFrame,
    so it stores the exact 107 feature names the model expects.
    Returns None if scaler is unavailable or has no feature names.
    """
    if not os.path.exists(SCALER_PKL):
        print("  [WARN] clinical_scaler.pkl not found.")
        return None

    scaler = joblib.load(SCALER_PKL)
    print(f"  Scaler type: {type(scaler).__name__}")

    if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
        names = list(scaler.feature_names_in_)
        print(f"  Scaler feature_names_in_: {len(names)} features")
        print(f"  First 8: {names[:8]}")
        return names

    print("  [WARN] Scaler has no feature_names_in_.")
    return None


def _get_schema_from_full_csv(n_expected: int) -> list[str] | None:
    """
    Fallback: load the full clinical CSV, apply get_dummies with ALL
    31 non-meta columns, and find the subset of selected columns that
    produces exactly n_expected features by trying the top-k selection.
    This is a brute-force fallback when the scaler gives no names.
    """
    candidates = [
        os.path.join(BASE_DIR, "data", "processed", "clinical_cleaned.csv"),
        os.path.join(BASE_DIR, "data", "raw", "clinical_dataset.csv"),
    ]
    full_csv = next((p for p in candidates if os.path.exists(p)), None)
    if full_csv is None:
        return None

    print(f"  [FALLBACK] Loading full CSV: {os.path.basename(full_csv)}")
    df = pd.read_csv(full_csv)
    feat_cols = [c for c in df.columns if c not in _META_COLS]
    X = _fill_nans(df[feat_cols])

    # Try encoding the full set
    X_enc = pd.get_dummies(X, drop_first=False)
    print(f"  Full encoding: {X_enc.shape[1]} columns (model expects {n_expected})")

    if X_enc.shape[1] == n_expected:
        return list(X_enc.columns)

    # The training pipeline used select_top_features (top_k=25 from config).
    # Re-run feature selection on the full data to find the 25 best raw features,
    # then encode only those to see if we hit n_expected.
    print("  Running feature selection to find the right 25 raw features ...")
    try:
        y = df["uti_label"].astype(int) if "uti_label" in df.columns else None
        if y is None:
            return None

        from sklearn.ensemble import RandomForestClassifier

        X_dummy = pd.get_dummies(X, drop_first=False)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_dummy, y)
        importances = pd.Series(rf.feature_importances_, index=X_dummy.columns)

        # Aggregate dummy columns back to original feature names
        agg = {}
        for col in feat_cols:
            prefix = f"{col}_"
            mask = (importances.index == col) | importances.index.str.startswith(prefix)
            agg[col] = float(importances[mask].sum())

        top25 = sorted(agg, key=agg.get, reverse=True)[:25]
        print(f"  Top-25 features: {top25}")

        X_top = _fill_nans(df[top25])
        X_top_enc = pd.get_dummies(X_top, drop_first=False)
        print(f"  Top-25 encoding: {X_top_enc.shape[1]} columns")

        if X_top_enc.shape[1] == n_expected:
            print("  ✓ Top-25 encoding matches model expectation!")
            return list(X_top_enc.columns)
        else:
            print(f"  Top-25 encoding still doesn't match ({X_top_enc.shape[1]} != {n_expected}).")

    except Exception as e:
        print(f"  [WARN] Feature selection fallback failed: {e}")

    return None


def _encode_to_schema(X_raw: pd.DataFrame, schema: list[str]) -> np.ndarray:
    """One-hot encode X_raw then reindex to exactly `schema` columns."""
    X_enc = pd.get_dummies(X_raw, drop_first=False)
    X_aligned = X_enc.reindex(columns=schema, fill_value=0)
    return X_aligned.values.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Clinical extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_clinical_embeddings():
    """Extract clinical embeddings using the 80K-trained XGBoost model."""
    print("\n[CLIN] Step 1 — Extracting clinical embeddings ...")

    # Load model
    payload   = joblib.load(CLINICAL_PKL)
    model_obj = payload["model"] if isinstance(payload, dict) else payload
    print(f"  Model type:        {type(model_obj).__name__}")
    n_expected = getattr(model_obj, "n_features_in_", None)
    print(f"  n_features_in_:    {n_expected}")

    # Load 4K CSV
    df4k      = pd.read_csv(CLINICAL_CSV)
    feat_cols = [c for c in df4k.columns if c not in _META_COLS]
    X_raw     = _fill_nans(df4k[feat_cols])
    print(f"  4K CSV: {df4k.shape}  |  feature cols: {len(feat_cols)}")

    # ── Obtain the exact 107-feature schema ───────────────────────────────
    # Priority 1: scaler.feature_names_in_  (most reliable)
    schema = _get_schema_from_scaler()

    # Priority 2: booster.feature_names
    if schema is None:
        bf = model_obj.get_booster().feature_names
        if bf is not None:
            schema = list(bf)
            print(f"  Schema from booster.feature_names: {len(schema)}")

    # Priority 3: full-CSV brute-force
    if schema is None and n_expected is not None:
        schema = _get_schema_from_full_csv(n_expected)

    # Priority 4: direct encoding (no schema — last resort)
    if schema is None:
        print("  [LAST RESORT] No schema found — using raw get_dummies output.")
        print("  WARNING: This may cause a shape mismatch in XGBoost predict.")
        X_arr = pd.get_dummies(_fill_nans(X_raw), drop_first=False).values.astype(np.float32)
    else:
        # Validate schema length against model expectation
        if n_expected is not None and len(schema) != n_expected:
            print(f"  [WARN] Schema has {len(schema)} features, model expects {n_expected}.")

        X_arr = _encode_to_schema(X_raw, schema)

    print(f"  X_encoded shape: {X_arr.shape}  (model expects: {n_expected})")

    # ── predict_proba ──────────────────────────────────────────────────────
    # XGBoost is tree-based — NO scaler applied.
    print("  Running predict_proba ...")
    try:
        clinical_proba = model_obj.predict_proba(X_arr)
    except Exception as e:
        raise RuntimeError(
            f"predict_proba failed: X shape={X_arr.shape}, "
            f"model expects {n_expected}.\nError: {e}\n"
            "Check that clinical_scaler.pkl has correct feature_names_in_."
        ) from e

    print(f"  ✓ Clinical proba:    {clinical_proba.shape}")
    print(f"  ✓ Clinical features: {X_arr.shape}")
    return clinical_proba, X_arr, X_raw


# ─────────────────────────────────────────────────────────────────────────────
# Image extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_embeddings(fusion_pairs: pd.DataFrame) -> np.ndarray:
    """Extract image embeddings from EfficientNetB3 'features' layer (256-dim)."""
    print("\n[IMG] Step 2 — Extracting image embeddings ...")

    import tensorflow as tf
    tf.random.set_seed(SEED)

    sys.path.insert(0, BASE_DIR)
    from src.us_preprocessing import preprocess_ultrasound_v2

    print(f"  Loading: {os.path.basename(IMAGE_MODEL)}")
    full_model = tf.keras.models.load_model(IMAGE_MODEL, compile=False)
    print(f"  Input: {full_model.input_shape}  Output: {full_model.output_shape}")

    # Show last 10 layers
    print("  Last 10 layers:")
    for layer in full_model.layers[-10:]:
        try:
            shape = layer.output_shape
        except Exception:
            shape = "?"
        print(f"    {layer.name:35s} {shape}")

    # Locate embedding layer (Dense before output sigmoid)
    embed_name = None
    for cand in ["features", "dense_1", "dense_256"]:
        try:
            full_model.get_layer(cand)
            embed_name = cand
            break
        except ValueError:
            pass

    if embed_name is None:
        dense_layers = [l for l in full_model.layers
                        if isinstance(l, tf.keras.layers.Dense)
                        and l.name != "output"]
        if dense_layers:
            embed_name = dense_layers[-1].name

    if embed_name is None:
        raise RuntimeError("Cannot locate embedding Dense layer in image model.")

    # Walk back from any Dropout to the previous Dense
    layer_names = [l.name for l in full_model.layers]
    idx = layer_names.index(embed_name)
    while idx >= 0 and isinstance(full_model.layers[idx], tf.keras.layers.Dropout):
        idx -= 1
    if isinstance(full_model.layers[idx], tf.keras.layers.Dense):
        embed_name = full_model.layers[idx].name

    print(f"  Embedding layer: '{embed_name}'")
    embed_model = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(embed_name).output,
        name="embed_extractor",
    )
    print(f"  Embedding output shape: {embed_model.output_shape}")

    image_paths = fusion_pairs["image_filename"].tolist()
    n           = len(image_paths)
    embed_dim   = embed_model.output_shape[-1]
    embeddings  = np.zeros((n, embed_dim), dtype=np.float32)

    BATCH = 16
    b_imgs, b_idx = [], []

    for i, path in enumerate(tqdm(image_paths, desc="  Image embeddings")):
        img = preprocess_ultrasound_v2(path, target_size=(260, 260))
        if img is None:
            print(f"\n  [WARN] Unreadable image at index {i}: {path}")
            continue

        img_s = tf.keras.applications.efficientnet.preprocess_input(
            (img * 255.0).astype(np.float32)
        )
        b_imgs.append(img_s)
        b_idx.append(i)

        if len(b_imgs) == BATCH or i == n - 1:
            if b_imgs:
                arr = np.array(b_imgs, dtype=np.float32)
                emb = embed_model.predict(arr, verbose=0)
                for j, k in enumerate(b_idx):
                    embeddings[k] = emb[j]
                b_imgs, b_idx = [], []

    print(f"\n  ✓ Image embeddings shape: {embeddings.shape}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  TASK 2 — Extract Embeddings from Pretrained Models")
    print("=" * 60)

    fusion_pairs = pd.read_csv(FUSION_CSV)
    print(f"  Fusion pairs: {fusion_pairs.shape[0]} rows")

    os.makedirs(EMB_DIR, exist_ok=True)

    # Step 1 — Clinical
    clinical_proba, clinical_features, _ = extract_clinical_embeddings()

    # Step 2 — Image
    image_embeddings = extract_image_embeddings(fusion_pairs)

    # Step 3 — Alignment check
    print("\n[VERIFY] Step 3 — Alignment check ...")
    n = fusion_pairs.shape[0]
    assert clinical_proba.shape[0] == n,    f"clinical_proba mismatch: {clinical_proba.shape[0]} != {n}"
    assert clinical_features.shape[0] == n, f"clinical_features mismatch: {clinical_features.shape[0]} != {n}"
    assert image_embeddings.shape[0] == n,  f"image_embeddings mismatch: {image_embeddings.shape[0]} != {n}"
    print("  All shapes aligned with fusion_pairs.csv ✓")

    # Save
    np.save(os.path.join(EMB_DIR, "clinical_proba.npy"),    clinical_proba)
    np.save(os.path.join(EMB_DIR, "clinical_features.npy"), clinical_features)
    np.save(os.path.join(EMB_DIR, "image_embeddings.npy"),  image_embeddings)

    print("\n=== Embedding Extraction Report ===")
    print(f"  Clinical proba embeddings:   {clinical_proba.shape}")
    print(f"  Clinical feature embeddings: {clinical_features.shape}")
    print(f"  Image embeddings:            {image_embeddings.shape}")
    print(f"  Saved to: {EMB_DIR}")
    print("\n✓ TASK 2 COMPLETE")


if __name__ == "__main__":
    main()
