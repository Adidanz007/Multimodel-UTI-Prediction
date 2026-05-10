"""
extract_embeddings_v2.py — FIX 2 + FIX 3: Re-extract all embeddings
=====================================================================
FIX 2: Uses the new 4K-trained clinical model (clinical_model_4k.pkl)
        to produce meaningful clinical probability signals.
FIX 3: Finds the correct 256-dim embedding layer and verifies output range.

Outputs:
  results/embeddings/clinical_proba_4k.npy      — (4000, 2)
  results/embeddings/clinical_features_4k.npy   — (4000, N_encoded)
  results/embeddings/image_embeddings_fixed.npy  — (4000, 256)
  results/embeddings/image_proba.npy             — (4000, 1)
"""

from __future__ import annotations

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm

SEED = 42
np.random.seed(SEED)

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_CSV   = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
CLINICAL_CSV = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
IMAGE_MODEL  = os.path.join(BASE_DIR, "models", "ultrasound_efficientnet_best.keras")
EMB_DIR      = os.path.join(BASE_DIR, "results", "embeddings")
os.makedirs(EMB_DIR, exist_ok=True)

# ── Import the encoding function from retrain_clinical_4k ────────────────────
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
from retrain_clinical_4k import encode_clinical


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Clinical embeddings using 4K model
# ─────────────────────────────────────────────────────────────────────────────

def extract_clinical_4k():
    """
    Use the 4K-trained model to extract meaningful clinical embeddings.
    Verifies that infected > normal by at least 0.15.
    """
    print("\n[CLIN] FIX 2 — Extract clinical embeddings (4K model)")

    # Load model + scaler + feature names
    model_path  = os.path.join(MODELS_DIR, "clinical_model_4k.pkl")
    scaler_path = os.path.join(MODELS_DIR, "clinical_scaler_4k.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found. Run src/retrain_clinical_4k.py first."
        )

    clin_model = joblib.load(model_path)
    scaler     = joblib.load(scaler_path)
    print(f"  Model type:  {type(clin_model).__name__}")
    print(f"  Scaler type: {type(scaler).__name__}")

    # Load and encode 4K clinical data
    df = pd.read_csv(CLINICAL_CSV)
    y  = df["uti_label"].astype(int).values
    print(f"  4K CSV shape: {df.shape}")

    X_enc, _ = encode_clinical(df)
    print(f"  Encoded shape: {X_enc.shape}")
    print(f"  NaN check: {X_enc.isnull().sum().sum()}")

    X_scaled = scaler.transform(X_enc.values.astype(np.float32))
    print(f"  Scaled shape: {X_scaled.shape}")

    # Signal A: probability scores
    clinical_proba = clin_model.predict_proba(X_scaled)   # (4000, 2)
    print(f"\n  Signal A (proba) shape: {clinical_proba.shape}")

    # Signal B: scaled feature vector
    clinical_features = X_scaled                           # (4000, N)
    print(f"  Signal B (features) shape: {clinical_features.shape}")

    # Verify separation
    inf_idx  = np.where(y == 1)[0]
    norm_idx = np.where(y == 0)[0]
    inf_mean  = clinical_proba[inf_idx, 1].mean()
    norm_mean = clinical_proba[norm_idx, 1].mean()
    sep = inf_mean - norm_mean

    print(f"\n  Clinical proba — Infected: {inf_mean:.4f}")
    print(f"  Clinical proba — Normal:   {norm_mean:.4f}")
    print(f"  Separation:                {sep:.4f}")

    if sep < 0.15:
        print("  ⚠ CLINICAL SIGNAL TOO WEAK — cannot fuse reliably")
        print("    Check that retrain_clinical_4k.py ran successfully")
    else:
        print("  ✓ Clinical signal is discriminative")

    # Save
    np.save(os.path.join(EMB_DIR, "clinical_proba_4k.npy"),    clinical_proba)
    np.save(os.path.join(EMB_DIR, "clinical_features_4k.npy"), clinical_features)
    print(f"  Saved: clinical_proba_4k.npy    {clinical_proba.shape}")
    print(f"  Saved: clinical_features_4k.npy {clinical_features.shape}")

    return clinical_proba, clinical_features, y


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Image embeddings — find correct layer, verify output range
# ─────────────────────────────────────────────────────────────────────────────

def extract_image_embeddings_fixed(fusion_pairs: pd.DataFrame):
    """
    Extract 256-dim image embeddings and image probability scores.
    Verifies embedding values are in reasonable range (mostly positive, 0-5).
    """
    print("\n[IMG] FIX 3 — Extract image embeddings (fixed)")

    import tensorflow as tf
    tf.random.set_seed(SEED)

    sys.path.insert(0, BASE_DIR)
    from src.us_preprocessing import preprocess_ultrasound_v2

    print(f"  Loading: {os.path.basename(IMAGE_MODEL)}")
    full_model = tf.keras.models.load_model(IMAGE_MODEL, compile=False)
    print(f"  Input:  {full_model.input_shape}")
    print(f"  Output: {full_model.output.shape}")

    # Print ALL layers and shapes
    print("\n  All layers:")
    for layer in full_model.layers:
        try:
            shape = str(layer.output.shape)
        except Exception:
            shape = "(multiple)"
        print(f"    {layer.name:40s}  {shape}")

    # Find the 256-unit Dense layer BEFORE final output sigmoid
    # Walk backwards: skip Dropout and output Dense(1)
    embed_name = None
    for layer in reversed(full_model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            out_dim = layer.output.shape[-1]
            if out_dim > 1:  # not the output layer
                embed_name = layer.name
                print(f"\n  Selected embedding layer: '{embed_name}' "
                      f"(output_dim={out_dim})")
                break

    # Also check named candidates
    for cand in ["features", "dense_256", "dense_1"]:
        try:
            layer = full_model.get_layer(cand)
            if isinstance(layer, tf.keras.layers.Dense) and layer.output.shape[-1] > 1:
                embed_name = cand
                print(f"  Found named candidate: '{embed_name}'")
                break
        except ValueError:
            pass

    if embed_name is None:
        raise RuntimeError("Cannot find 256-dim Dense embedding layer.")

    # Build embedding extractor
    embed_model = tf.keras.Model(
        inputs=full_model.input,
        outputs=full_model.get_layer(embed_name).output,
        name="embed_extractor",
    )
    print(f"  Embedding model output: {embed_model.output.shape}")

    # Test on 5 images to verify output range
    print("\n  Verifying embedding range on 5 test images ...")
    test_imgs = []
    for path in fusion_pairs["image_filename"].head(10).tolist():
        img = preprocess_ultrasound_v2(path, target_size=(260, 260))
        if img is not None:
            img_s = tf.keras.applications.efficientnet.preprocess_input(
                (img * 255.0).astype(np.float32)
            )
            test_imgs.append(img_s)
        if len(test_imgs) == 5:
            break

    if test_imgs:
        test_batch = np.array(test_imgs, dtype=np.float32)
        test_emb   = embed_model.predict(test_batch, verbose=0)
        print(f"  Embedding stats (5 images):")
        print(f"    mean={test_emb.mean():.4f}, std={test_emb.std():.4f}, "
              f"min={test_emb.min():.4f}, max={test_emb.max():.4f}")
        if test_emb.max() < 0.01:
            print("  ⚠ WARNING: All values near zero → wrong layer selected!")
            # Try the layer before it
            all_dense = [l for l in full_model.layers
                         if isinstance(l, tf.keras.layers.Dense)]
            for d_layer in reversed(all_dense):
                if d_layer.name != embed_name and d_layer.output.shape[-1] > 1:
                    print(f"  Trying previous Dense: '{d_layer.name}'")
                    embed_model2 = tf.keras.Model(
                        inputs=full_model.input,
                        outputs=d_layer.output)
                    test_emb2 = embed_model2.predict(test_batch, verbose=0)
                    print(f"    mean={test_emb2.mean():.4f}, max={test_emb2.max():.4f}")
                    if test_emb2.max() > 0.1:
                        embed_model = embed_model2
                        embed_name  = d_layer.name
                        print(f"  ✓ Using '{embed_name}' instead")
                        break
    else:
        print("  ⚠ Could not load test images for range check")

    # ── Extract all 4000 embeddings ────────────────────────────────────────
    image_paths = fusion_pairs["image_filename"].tolist()
    n           = len(image_paths)
    embed_dim   = embed_model.output.shape[-1]
    embeddings  = np.zeros((n, embed_dim), dtype=np.float32)
    probas      = np.zeros((n, 1),         dtype=np.float32)

    BATCH = 16
    b_imgs, b_idx = [], []

    for i, path in enumerate(tqdm(image_paths, desc="  Images")):
        img = preprocess_ultrasound_v2(path, target_size=(260, 260))
        if img is None:
            print(f"\n  [WARN] Unreadable: {path}")
            continue

        img_s = tf.keras.applications.efficientnet.preprocess_input(
            (img * 255.0).astype(np.float32)
        )
        b_imgs.append(img_s)
        b_idx.append(i)

        if len(b_imgs) == BATCH or i == n - 1:
            if b_imgs:
                arr  = np.array(b_imgs, dtype=np.float32)
                emb  = embed_model.predict(arr,  verbose=0)
                prob = full_model.predict(arr,   verbose=0)
                for j, k in enumerate(b_idx):
                    embeddings[k] = emb[j]
                    probas[k]     = prob[j]
                b_imgs, b_idx = [], []

    print(f"\n  Image embeddings shape: {embeddings.shape}")
    print(f"  Image probas shape:     {probas.shape}")
    print(f"  Embedding stats: mean={embeddings.mean():.4f}, "
          f"std={embeddings.std():.4f}, min={embeddings.min():.4f}, "
          f"max={embeddings.max():.4f}")

    # Verify signal
    labels = fusion_pairs["label"].values
    abn_idx  = np.where(labels == 1)[0]
    norm_idx = np.where(labels == 0)[0]
    abn_prob  = probas[abn_idx].mean()
    norm_prob = probas[norm_idx].mean()
    print(f"\n  Image proba — Abnormal: {abn_prob:.4f}")
    print(f"  Image proba — Normal:   {norm_prob:.4f}")
    print(f"  Separation:             {abn_prob - norm_prob:.4f}")

    np.save(os.path.join(EMB_DIR, "image_embeddings_fixed.npy"), embeddings)
    np.save(os.path.join(EMB_DIR, "image_proba.npy"),            probas)
    print(f"  Saved: image_embeddings_fixed.npy {embeddings.shape}")
    print(f"  Saved: image_proba.npy            {probas.shape}")

    return embeddings, probas


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  FIX 2 + FIX 3 — Re-extract All Embeddings")
    print("=" * 60)

    fusion_pairs = pd.read_csv(FUSION_CSV)
    print(f"  Fusion pairs: {fusion_pairs.shape[0]} rows")

    # FIX 2: Clinical
    clinical_proba, clinical_features, y = extract_clinical_4k()

    # FIX 3: Image
    image_embeddings, image_proba = extract_image_embeddings_fixed(fusion_pairs)

    # Alignment check
    n = fusion_pairs.shape[0]
    assert clinical_proba.shape[0]    == n, f"clinical_proba rows mismatch"
    assert image_embeddings.shape[0]  == n, f"image_embeddings rows mismatch"
    print(f"\n  All {n} embeddings aligned ✓")

    print(f"\n=== Embedding Report ===")
    print(f"  clinical_proba_4k:      {clinical_proba.shape}")
    print(f"  clinical_features_4k:   {clinical_features.shape}")
    print(f"  image_embeddings_fixed: {image_embeddings.shape}")
    print(f"  image_proba:            {image_proba.shape}")
    print(f"\n✓ FIX 2 + FIX 3 COMPLETE")


if __name__ == "__main__":
    main()
