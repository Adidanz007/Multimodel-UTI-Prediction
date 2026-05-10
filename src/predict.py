"""
predict.py — Single inference function for the Flask web app
=============================================================
The clinical_scaler.pkl's feature_names_in_ gives us the exact 107
one-hot-encoded column names the XGBoost model was trained on.
We use this as the encoding schema (no scaling applied to XGBoost).

Usage:
    from src.predict import predict_uti_risk
    result = predict_uti_risk(clinical_dict, image_path)
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_META_COLS = {"uti_label", "split", "abxUTI", "alt_diag"}

# Lazy-loaded globals
_models_loaded     = False
_xgb_model         = None
_feature_schema    = None   # exact 107 one-hot column names
_image_embed_model = None
_fusion_model      = None
_preprocess_fn     = None


def _fill_nans(df: pd.DataFrame) -> pd.DataFrame:
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


def _resolve_feature_schema(model_obj) -> list[str] | None:
    """
    Get the exact feature schema in priority order:
      1. clinical_scaler.pkl → feature_names_in_  (most reliable)
      2. booster.feature_names
      3. None (will fall back to raw encoding)
    """
    scaler_path = os.path.join(MODELS_DIR, "clinical_scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, "feature_names_in_") and scaler.feature_names_in_ is not None:
            return list(scaler.feature_names_in_)

    bf = model_obj.get_booster().feature_names
    if bf is not None:
        return list(bf)

    return None


def _load_models():
    global _models_loaded, _xgb_model, _feature_schema
    global _image_embed_model, _fusion_model, _preprocess_fn

    if _models_loaded:
        return

    import tensorflow as tf
    print("[predict] Loading models ...")

    # 1. Clinical XGBoost
    payload     = joblib.load(os.path.join(MODELS_DIR, "clinical_model.pkl"))
    _xgb_model  = payload["model"] if isinstance(payload, dict) else payload
    _feature_schema = _resolve_feature_schema(_xgb_model)

    if _feature_schema:
        print(f"  XGBoost loaded — schema: {len(_feature_schema)} features")
    else:
        print("  XGBoost loaded — no schema found, will use raw encoding")

    # 2. Image embedding model
    img_path   = os.path.join(MODELS_DIR, "ultrasound_efficientnet_best.keras")
    full_model = tf.keras.models.load_model(img_path, compile=False)

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
                        if isinstance(l, tf.keras.layers.Dense) and l.name != "output"]
        if dense_layers:
            embed_name = dense_layers[-1].name

    _image_embed_model = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(embed_name).output,
    )
    print(f"  Image embedding model — layer: {embed_name}, "
          f"dim: {_image_embed_model.output_shape[-1]}")

    # 3. Fusion model
    _fusion_model = tf.keras.models.load_model(
        os.path.join(MODELS_DIR, "fusion_model_best.keras"), compile=False
    )
    print("  Fusion model loaded")

    # 4. Image preprocessor
    sys.path.insert(0, BASE_DIR)
    from src.us_preprocessing import preprocess_ultrasound_v2
    _preprocess_fn = preprocess_ultrasound_v2

    _models_loaded = True
    print("[predict] All models loaded ✓")


def _encode_clinical(clinical_data_dict: dict) -> tuple[np.ndarray, float]:
    """
    Encode clinical dict to match XGBoost's expected 107-feature schema.
    Returns (X array [1, 107], risk_score float).
    """
    # Drop metadata keys if present
    clin = {k: v for k, v in clinical_data_dict.items() if k not in _META_COLS}
    df   = _fill_nans(pd.DataFrame([clin]))
    df_enc = pd.get_dummies(df, drop_first=False)

    if _feature_schema:
        df_aligned = df_enc.reindex(columns=_feature_schema, fill_value=0)
    else:
        # No schema — pass raw (may fail for unseen categories)
        df_aligned = df_enc

    X = df_aligned.values.astype(np.float32)
    risk = float(_xgb_model.predict_proba(X)[0, 1])
    return X, risk


def predict_uti_risk(clinical_data_dict: dict, image_path: str) -> dict:
    """
    Predict UTI risk from clinical features + bladder ultrasound image.

    Args:
        clinical_data_dict: {feature_name: value} — any/all clinical features.
        image_path:         path to bladder ultrasound image.

    Returns dict:
        fusion_risk_score   — float 0-1  (main output for Flask)
        clinical_risk_score — float 0-1
        image_risk_score    — float 0-1
        prediction          — 'UTI Positive' | 'UTI Negative'
        confidence          — 'High' | 'Medium' | 'Low'
    """
    import tensorflow as tf
    _load_models()

    # Clinical
    clin_features, clinical_risk_score = _encode_clinical(clinical_data_dict)

    # Image
    img = _preprocess_fn(image_path, target_size=(260, 260))
    if img is None:
        pred = "UTI Positive" if clinical_risk_score > 0.5 else "UTI Negative"
        return {
            "fusion_risk_score":   round(clinical_risk_score, 4),
            "clinical_risk_score": round(clinical_risk_score, 4),
            "image_risk_score":    0.5,
            "prediction":          pred,
            "confidence":          "Low",
            "error":               "Image unreadable — clinical-only result",
        }

    img_s = tf.keras.applications.efficientnet.preprocess_input(
        (img * 255.0).astype(np.float32)
    )
    img_embedding    = _image_embed_model.predict(np.expand_dims(img_s, 0), verbose=0)
    image_risk_score = float(np.mean(img_embedding > 0))

    # Fusion
    fusion_score = float(
        _fusion_model.predict([clin_features, img_embedding], verbose=0)[0, 0]
    )

    prediction = "UTI Positive" if fusion_score > 0.5 else "UTI Negative"
    margin     = max(fusion_score, 1 - fusion_score)
    confidence = "High" if margin >= 0.8 else ("Medium" if margin >= 0.6 else "Low")

    return {
        "fusion_risk_score":   round(fusion_score, 4),
        "clinical_risk_score": round(clinical_risk_score, 4),
        "image_risk_score":    round(image_risk_score, 4),
        "prediction":          prediction,
        "confidence":          confidence,
    }


def test_on_samples():
    """Test predict_uti_risk on 3 sample pairs from the test set."""
    print("\n" + "=" * 60)
    print("  TASK 4b — Test predict_uti_risk() on 3 samples")
    print("=" * 60)

    pairs   = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv"))
    clin_df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv"))

    test_pairs = pairs[pairs["split"] == "test"]
    samples    = []
    for label in [1, 0, 1]:
        subset = test_pairs[test_pairs["label"] == label]
        if len(subset) > 0:
            samples.append(subset.iloc[len(samples) % len(subset)])
    if len(samples) < 3:
        samples = [test_pairs.iloc[i] for i in range(min(3, len(test_pairs)))]

    for i, row in enumerate(samples):
        print(f"\n--- Sample {i+1} ---")
        clin_dict = clin_df.iloc[int(row["clinical_row_index"])].to_dict()
        result    = predict_uti_risk(clin_dict, row["image_filename"])

        true_label = "Infected" if row["label"] == 1 else "Normal"
        print(f"  True label:           {true_label}")
        print(f"  Prediction:           {result['prediction']}")
        print(f"  Fusion risk score:    {result['fusion_risk_score']}")
        print(f"  Clinical risk score:  {result['clinical_risk_score']}")
        print(f"  Image risk score:     {result['image_risk_score']}")
        print(f"  Confidence:           {result['confidence']}")

    print("\n✓ TASK 4b COMPLETE — predict_uti_risk() tested on 3 samples")


if __name__ == "__main__":
    test_on_samples()
