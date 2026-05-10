"""
predict_v2.py — FIX 5: Correct predict_uti_risk() function
============================================================
Uses the 4K-retrained clinical model with proper ordinal encoding
and the best fusion strategy from multimodal_fusion_v2.py.

Usage:
    from src.predict_v2 import predict_uti_risk
    result = predict_uti_risk(clinical_dict, image_path)
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

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Lazy-loaded globals
_loaded           = False
_clin_model       = None
_clin_scaler      = None
_feature_names    = None
_image_model      = None
_embed_model      = None
_pca              = None
_fusion_model     = None
_fusion_type      = None
_preprocess_fn    = None
_threshold        = 0.50

# Import encoding helpers
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from retrain_clinical_4k import encode_clinical, ORDINAL_MAPS, BINARY_COLS, _META_COLS


def _load_models():
    global _loaded, _clin_model, _clin_scaler, _feature_names
    global _image_model, _embed_model, _pca, _fusion_model, _fusion_type, _preprocess_fn, _threshold

    if _loaded:
        return

    import tensorflow as tf

    print("[predict_v2] Loading models ...")

    # 1. Clinical (4K-trained)
    _clin_model  = joblib.load(os.path.join(MODELS_DIR, "clinical_model_4k.pkl"))
    _clin_scaler = joblib.load(os.path.join(MODELS_DIR, "clinical_scaler_4k.pkl"))
    with open(os.path.join(MODELS_DIR, "clinical_feature_names_4k.json")) as f:
        _feature_names = json.load(f)
    print(f"  Clinical 4K model: {type(_clin_model).__name__}  "
          f"({len(_feature_names)} features)")

    # 2. Image model
    img_path    = os.path.join(MODELS_DIR, "ultrasound_efficientnet_best.keras")
    _image_model = tf.keras.models.load_model(img_path, compile=False)

    embed_name = None
    for cand in ["features", "dense_256", "dense_1"]:
        try:
            layer = _image_model.get_layer(cand)
            if isinstance(layer, tf.keras.layers.Dense) and layer.output.shape[-1] > 1:
                embed_name = cand
                break
        except ValueError:
            pass
    if embed_name is None:
        for layer in reversed(_image_model.layers):
            if isinstance(layer, tf.keras.layers.Dense) and layer.output.shape[-1] > 1:
                embed_name = layer.name
                break

    _embed_model = tf.keras.Model(
        inputs=_image_model.input,
        outputs=_image_model.get_layer(embed_name).output,
    )
    print(f"  Image embed layer: '{embed_name}' dim={_embed_model.output.shape[-1]}")

    # 3. PCA
    pca_path = os.path.join(MODELS_DIR, "fusion_pca.pkl")
    if os.path.exists(pca_path):
        _pca = joblib.load(pca_path)
        print(f"  PCA: {_pca.n_components_} components")
    else:
        print("  ⚠ fusion_pca.pkl not found — run multimodal_fusion_v2.py first")

    # 4. Fusion model (try keras then pkl)
    config_path = os.path.join(MODELS_DIR, "fusion_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        _fusion_type = cfg.get("fusion_type", "unknown")
    else:
        _fusion_type = "unknown"

    keras_path = os.path.join(MODELS_DIR, "fusion_model_final.keras")
    pkl_path   = os.path.join(MODELS_DIR, "fusion_model_final.pkl")

    if os.path.exists(keras_path):
        _fusion_model = tf.keras.models.load_model(keras_path, compile=False)
        _fusion_type  = "keras"
        print(f"  Fusion model: keras")
    elif os.path.exists(pkl_path):
        _fusion_model = joblib.load(pkl_path)
        print(f"  Fusion model: {type(_fusion_model).__name__} ({_fusion_type})")
    else:
        raise FileNotFoundError(
            "Fusion model not found. Run src/multimodal_fusion_v2.py first."
        )

    # 5. Preprocessor
    from us_preprocessing import preprocess_ultrasound_v2
    _preprocess_fn = preprocess_ultrasound_v2

    # 6. Threshold
    thresh_path = os.path.join(MODELS_DIR, "fusion_threshold.txt")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            _threshold = float(f.read().strip())
        print(f"  Optimal threshold: {_threshold:.4f}")
    else:
        print("  ⚠ fusion_threshold.txt not found, using 0.50")

    _loaded = True
    print("[predict_v2] All models loaded ✓")


def _encode_single(clinical_data_dict: dict) -> np.ndarray:
    """
    Encode a single clinical data dict using the 4K ordinal encoding.
    Returns scaled array of shape (1, N_features).
    """
    # Remove meta columns
    clin = {k: v for k, v in clinical_data_dict.items() if k not in _META_COLS}
    df   = pd.DataFrame([clin])

    # Apply ordinal encoding (same logic as retrain_clinical_4k)
    for col, mapping in ORDINAL_MAPS.items():
        if col not in df.columns:
            df[col] = 0.0
        if mapping is None:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            mapping_lower = {k.lower(): v for k, v in mapping.items()}
            med_val = int(np.median(list(mapping.values())))
            df[col] = df[col].astype(str).str.strip().str.lower() \
                              .map(mapping_lower).fillna(med_val).astype(float)

    for col in BINARY_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(float)

    for col in ["age", "Temperature", "RBC", "WBC"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Keep only expected feature columns
    for col in _feature_names:
        if col not in df.columns:
            df[col] = 0.0

    X = df[_feature_names].values.astype(np.float32)
    return _clin_scaler.transform(X)


def predict_uti_risk(clinical_data_dict: dict, image_path: str) -> dict:
    """
    Predict UTI risk from clinical features + bladder ultrasound image.

    Args:
        clinical_data_dict: {feature_name: value} — any/all clinical features.
        image_path:         path to bladder ultrasound image.

    Returns dict:
        prediction          — 'UTI Positive' | 'UTI Negative'
        confidence          — 'High' | 'Medium' | 'Low'
        fusion_risk_score   — float 0-1  (main output)
        clinical_risk_score — float 0-1
        image_risk_score    — float 0-1
        interpretation      — human-readable explanation
    """
    import tensorflow as tf
    _load_models()

    # Step 1: Clinical encoding
    X_clin = _encode_single(clinical_data_dict)            # (1, N)
    clinical_prob = float(_clin_model.predict_proba(X_clin)[0, 1])
    clin_proba_vec = np.array([[1 - clinical_prob, clinical_prob]])  # (1, 2)

    # Sanity check
    if abs(clinical_prob - 0.5) < 0.05:
        print("  ⚠ CLINICAL MODEL WARNING: Score near 0.5 — encoding may be incomplete")

    # Step 2: Image preprocessing
    img = _preprocess_fn(image_path, target_size=(260, 260))
    if img is None:
        pred = "UTI Positive" if clinical_prob > 0.5 else "UTI Negative"
        return {
            "prediction":          pred,
            "confidence":          "Low",
            "fusion_risk_score":   round(clinical_prob, 4),
            "clinical_risk_score": round(clinical_prob, 4),
            "image_risk_score":    0.5,
            "interpretation":      "Image unreadable — clinical-only result.",
        }

    # Step 3: Image embedding + probability
    img_s = tf.keras.applications.efficientnet.preprocess_input(
        (img * 255.0).astype(np.float32)
    )
    img_batch   = np.expand_dims(img_s, 0)
    image_prob  = float(_image_model.predict(img_batch, verbose=0)[0, 0])
    img_embed   = _embed_model.predict(img_batch, verbose=0)  # (1, 256)

    # Step 4: PCA reduce
    if _pca is not None:
        img_pca = _pca.transform(img_embed)                  # (1, 32)
    else:
        # Fallback: use first 32 dims
        img_pca = img_embed[:, :32]

    # Step 5: Build fusion input and predict
    if _fusion_type == "logistic_regression":
        X_fusion = np.column_stack([
            [[clinical_prob]],
            [[image_prob]],
            [[clinical_prob * image_prob]],
        ])
    else:
        X_fusion = np.column_stack([
            clin_proba_vec,           # (1, 2)
            [[image_prob]],           # (1, 1)
            img_pca,                  # (1, 32)
        ])  # (1, 35)

    if _fusion_type == "keras":
        fusion_prob = float(_fusion_model.predict(X_fusion, verbose=0)[0, 0])
    else:
        fusion_prob = float(_fusion_model.predict_proba(X_fusion)[0, 1])

    # Step 6: Interpret
    prediction = "UTI Positive" if fusion_prob >= _threshold else "UTI Negative"
    margin = max(fusion_prob, 1 - fusion_prob)
    if fusion_prob >= 0.75 or fusion_prob <= 0.25:
        confidence = "High"
    elif fusion_prob >= 0.60 or fusion_prob <= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    interp = (
        f"Clinical analysis suggests "
        f"{'infection' if clinical_prob > 0.5 else 'no infection'} "
        f"(score: {clinical_prob:.2f}). "
        f"Ultrasound analysis suggests "
        f"{'abnormality' if image_prob > 0.5 else 'normal appearance'} "
        f"(score: {image_prob:.2f}). "
        f"Combined multimodal risk score: {fusion_prob:.2f}."
    )

    return {
        "prediction":          prediction,
        "confidence":          confidence,
        "fusion_risk_score":   round(fusion_prob,    4),
        "clinical_risk_score": round(clinical_prob,  4),
        "image_risk_score":    round(image_prob,     4),
        "interpretation":      interp,
    }


def test_on_samples():
    """Test predict_uti_risk on 5 samples (3 infected, 2 normal)."""
    print("\n" + "=" * 60)
    print("  FIX 5 — Validate predict_uti_risk() on 5 samples")
    print("=" * 60)

    pairs_csv = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
    clin_csv  = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")

    pairs   = pd.read_csv(pairs_csv)
    clin_df = pd.read_csv(clin_csv)

    test_pairs = pairs[pairs["split"] == "test"] if "split" in pairs.columns else pairs

    # Pick 3 infected, 2 normal
    samples = []
    for label, count in [(1, 3), (0, 2)]:
        subset = test_pairs[test_pairs["label"] == label]
        for i in range(min(count, len(subset))):
            samples.append(subset.iloc[i])

    print(f"\n{'Sample':>6}  {'True':>8}  {'Pred':>14}  {'Clin':>6}  {'Img':>6}  {'Fusion':>6}  {'OK?':>5}")
    print("-" * 60)

    correct = 0
    for i, row in enumerate(samples):
        clin_row = clin_df.iloc[int(row["clinical_row_index"])].to_dict()
        result   = predict_uti_risk(clin_row, row["image_filename"])

        true_label = "Infected" if row["label"] == 1 else "Normal"
        ok = "✓" if (
            (row["label"] == 1 and result["prediction"] == "UTI Positive") or
            (row["label"] == 0 and result["prediction"] == "UTI Negative")
        ) else "✗"
        if ok == "✓":
            correct += 1

        print(f"  {i+1:>4}  {true_label:>8}  {result['prediction']:>14}  "
              f"{result['clinical_risk_score']:>6.2f}  "
              f"{result['image_risk_score']:>6.2f}  "
              f"{result['fusion_risk_score']:>6.2f}  {ok:>5}")

        # Check clinical sanity
        if row["label"] == 1 and result["clinical_risk_score"] < 0.3:
            print(f"         ⚠ CLINICAL MODEL ERROR — infected sample has score "
                  f"{result['clinical_risk_score']:.2f} < 0.30")

    print(f"\n  Correct: {correct}/{len(samples)}")
    if correct >= 4:
        print("  ✓ FIX 5 COMPLETE — 4+ of 5 samples correct  SUCCESS")
    else:
        print("  ⚠ FIX 5 PARTIAL — fewer than 4 correct  check model quality")


if __name__ == "__main__":
    test_on_samples()
