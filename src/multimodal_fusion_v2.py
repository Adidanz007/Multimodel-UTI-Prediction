"""
multimodal_fusion_v2.py — FIX 4: Correct, regularised fusion model
===================================================================
3 fusion strategies trained and compared on the same test set.
Strategy A: Simple LR on 3 probability values
Strategy B: Tiny NN (35 inputs, ~5K params) with heavy regularisation
Strategy C: XGBoost fusion with 5-fold CV

Runs PRE-TRAINING SIGNAL CHECK before any training.
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
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, f1_score, classification_report
)
from xgboost import XGBClassifier

SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMB_DIR    = os.path.join(BASE_DIR, "results", "embeddings")
FUSION_CSV = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)


def load_embeddings():
    """Load all embeddings and verify they exist."""
    paths = {
        "clinical_proba": os.path.join(EMB_DIR, "clinical_proba_4k.npy"),
        "image_embeddings": os.path.join(EMB_DIR, "image_embeddings_fixed.npy"),
        "image_proba": os.path.join(EMB_DIR, "image_proba.npy"),
    }

    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing embeddings: {missing}\n"
            "Run src/extract_embeddings_v2.py first."
        )

    clin      = np.load(paths["clinical_proba"])    # (4000, 2)
    imgs      = np.load(paths["image_embeddings"])  # (4000, 256)
    img_proba = np.load(paths["image_proba"])       # (4000, 1)

    print(f"  clinical_proba:   {clin.shape}")
    print(f"  image_embeddings: {imgs.shape}")
    print(f"  image_proba:      {img_proba.shape}")
    return clin, imgs, img_proba


def signal_check(clin, img_proba, imgs, labels):
    """Pre-training signal check — must show >0.10 separation for each."""
    print("\n=== PRE-TRAINING SIGNAL CHECK ===")
    inf_idx  = np.where(labels == 1)[0]
    norm_idx = np.where(labels == 0)[0]

    c_inf  = clin[inf_idx, 1].mean()
    c_norm = clin[norm_idx, 1].mean()
    i_inf  = img_proba[inf_idx].mean()
    i_norm = img_proba[norm_idx].mean()
    e_inf  = imgs[inf_idx].mean()
    e_norm = imgs[norm_idx].mean()

    print(f"  Clinical P(infected) — Infected: {c_inf:.4f}  Normal: {c_norm:.4f}  "
          f"Sep: {c_inf - c_norm:.4f}")
    print(f"  Image proba         — Infected: {i_inf:.4f}  Normal: {i_norm:.4f}  "
          f"Sep: {i_inf - i_norm:.4f}")
    print(f"  Image embed mean    — Infected: {e_inf:.4f}  Normal: {e_norm:.4f}  "
          f"Sep: {e_inf - e_norm:.4f}")

    warns = []
    if abs(c_inf - c_norm) < 0.10:
        warns.append("CLINICAL SIGNAL TOO WEAK — clinical model may be broken")
    if abs(i_inf - i_norm) < 0.10:
        warns.append("IMAGE SIGNAL TOO WEAK — image model may be broken")

    if warns:
        for w in warns:
            print(f"  ⚠ {w}")
        print("  Fusion will proceed but performance may be limited.")
    else:
        print("  ✓ Both signals are discriminative — ready to fuse")

    return warns


def get_splits(labels, fusion_pairs):
    """Get train/val/test indices from fusion_pairs split column if available."""
    if "split" in fusion_pairs.columns:
        train_idx = fusion_pairs[fusion_pairs["split"] == "train"].index.tolist()
        val_idx   = fusion_pairs[fusion_pairs["split"] == "val"].index.tolist()
        test_idx  = fusion_pairs[fusion_pairs["split"] == "test"].index.tolist()

        if len(train_idx) > 0 and len(test_idx) > 0:
            print(f"  Using fusion_pairs splits: "
                  f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
            return np.array(train_idx), np.array(val_idx), np.array(test_idx)

    # Fall back to stratified split
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        idx, test_size=0.30, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED)
    print(f"  Created splits: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    return train_idx, val_idx, test_idx


def eval_model(name, model, X_test, y_test, sklearn=True):
    """Evaluate and print metrics. Returns AUC."""
    if sklearn:
        proba = model.predict_proba(X_test)[:, 1]
    else:
        proba = model.predict(X_test, verbose=0).flatten()

    preds = (proba >= 0.5).astype(int)
    auc   = roc_auc_score(y_test, proba)
    acc   = accuracy_score(y_test, preds)
    rec   = recall_score(y_test, preds, zero_division=0)
    f1    = f1_score(y_test, preds, zero_division=0)
    print(f"  {name:35s} AUC={auc:.4f} Acc={acc:.3f} Recall={rec:.3f} F1={f1:.3f}")
    return auc, proba


def main():
    print("=" * 60)
    print("  FIX 4 — Build Correct, Regularised Fusion Model")
    print("=" * 60)

    # Load data
    fusion_pairs = pd.read_csv(FUSION_CSV)
    labels = fusion_pairs["label"].values if "label" in fusion_pairs.columns \
             else np.zeros(len(fusion_pairs), dtype=int)

    print("\n[LOAD] Loading embeddings ...")
    clin, imgs, img_proba = load_embeddings()
    n = clin.shape[0]

    # Signal check
    warnings_list = signal_check(clin, img_proba, imgs, labels)

    # Splits
    print("\n[SPLITS]")
    train_idx, val_idx, test_idx = get_splits(labels, fusion_pairs)
    y = labels

    # ──────────────────────────────────────────────────────────────────────
    # PCA: reduce image embeddings to 32 dims (fit on train only)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PCA] Reducing image embeddings: 256 → 32 dims")
    pca = PCA(n_components=32, random_state=SEED)
    imgs_pca_train = pca.fit_transform(imgs[train_idx])
    imgs_pca_all   = pca.transform(imgs)
    print(f"  Explained variance (32 PCs): {pca.explained_variance_ratio_.sum():.3f}")

    # Save PCA for inference
    joblib.dump(pca, os.path.join(MODELS_DIR, "fusion_pca.pkl"))

    # ──────────────────────────────────────────────────────────────────────
    # Strategy A: Simple Logistic Regression on 3 probability values
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STRATEGY A] Simple LR fusion (3 features)")
    X_simple = np.column_stack([
        clin[:, 1],                         # clinical P(infected)
        img_proba.flatten(),                # image P(abnormal)
        clin[:, 1] * img_proba.flatten(),   # interaction
    ])
    print(f"  X_simple shape: {X_simple.shape}")

    lr_fusion = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    lr_fusion.fit(X_simple[train_idx], y[train_idx])
    auc_A, proba_A = eval_model("Strategy A (LR fusion)", lr_fusion, X_simple[test_idx], y[test_idx])

    joblib.dump(lr_fusion, os.path.join(MODELS_DIR, "fusion_strategy_A.pkl"))

    # ──────────────────────────────────────────────────────────────────────
    # Strategy B: Tiny neural fusion (35-dim input, ~5K params)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STRATEGY B] Tiny NN fusion (35-dim input)")
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    tf.random.set_seed(SEED)

    X_fusion = np.column_stack([clin, img_proba, imgs_pca_all])  # (4000, 35)
    print(f"  X_fusion shape: {X_fusion.shape}")

    inp = tf.keras.Input(shape=(X_fusion.shape[1],))
    x   = layers.Dense(64, activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(32, activation="relu",
                       kernel_regularizer=regularizers.l2(0.01))(x)
    x   = layers.Dropout(0.4)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model_B = tf.keras.Model(inp, out, name="fusion_tiny")
    model_B.summary()

    model_B.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    best_B_path = os.path.join(MODELS_DIR, "fusion_model_v2_best.keras")
    history = model_B.fit(
        X_fusion[train_idx], y[train_idx],
        validation_data=(X_fusion[val_idx], y[val_idx]),
        epochs=150,
        batch_size=64,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                best_B_path, monitor="val_auc", mode="max", save_best_only=True),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_auc", patience=25, mode="max", restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
        ],
        verbose=1,
    )

    # Load best weights
    model_B = tf.keras.models.load_model(best_B_path, compile=False)
    proba_B = model_B.predict(X_fusion[test_idx], verbose=0).flatten()
    preds_B = (proba_B >= 0.5).astype(int)
    auc_B   = roc_auc_score(y[test_idx], proba_B)
    acc_B   = accuracy_score(y[test_idx], preds_B)
    rec_B   = recall_score(y[test_idx], preds_B, zero_division=0)
    f1_B    = f1_score(y[test_idx], preds_B, zero_division=0)
    print(f"  Strategy B (Tiny NN)              "
          f"AUC={auc_B:.4f} Acc={acc_B:.3f} Recall={rec_B:.3f} F1={f1_B:.3f}")

    # Train/val AUC gap check
    best_val_auc = max(history.history["val_auc"])
    best_trn_auc = history.history["auc"][history.history["val_auc"].index(best_val_auc)]
    gap = best_trn_auc - best_val_auc
    print(f"  Train/val AUC gap: {gap:.4f} "
          f"({'OK' if gap < 0.10 else '⚠ OVERFIT'})")

    # ──────────────────────────────────────────────────────────────────────
    # Strategy C: XGBoost fusion with 5-fold CV
    # ──────────────────────────────────────────────────────────────────────
    print("\n[STRATEGY C] XGBoost fusion (5-fold CV)")
    X_xgb = np.column_stack([clin, img_proba, imgs_pca_all])  # same as X_fusion

    xgb_fusion = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0,
        eval_metric="logloss", random_state=SEED, verbosity=0,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    # CV on train+val combined
    train_val_idx = np.concatenate([train_idx, val_idx])
    cv_aucs = cross_val_score(
        xgb_fusion, X_xgb[train_val_idx], y[train_val_idx],
        cv=skf, scoring="roc_auc"
    )
    print(f"  Strategy C CV AUC: {cv_aucs.mean():.4f} ± {cv_aucs.std():.4f}")

    # Final train on full train split
    xgb_fusion.fit(X_xgb[train_idx], y[train_idx])
    auc_C, proba_C = eval_model(
        "Strategy C (XGB fusion)", xgb_fusion, X_xgb[test_idx], y[test_idx])

    joblib.dump(xgb_fusion, os.path.join(MODELS_DIR, "fusion_strategy_C.pkl"))

    # ──────────────────────────────────────────────────────────────────────
    # Individual model baselines (test set)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[BASELINES]")
    clin_model = joblib.load(os.path.join(MODELS_DIR, "clinical_model_4k.pkl"))
    clin_scaler = joblib.load(os.path.join(MODELS_DIR, "clinical_scaler_4k.pkl"))
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

    # Clinical-only: use clinical proba directly
    auc_clin = roc_auc_score(y[test_idx], clin[test_idx, 1])
    print(f"  {'Clinical only (4K XGB)':35s} AUC={auc_clin:.4f}")

    # Image-only: use image proba
    auc_img = roc_auc_score(y[test_idx], img_proba[test_idx].flatten())
    rec_img = recall_score(y[test_idx],
                           (img_proba[test_idx].flatten() >= 0.5).astype(int),
                           zero_division=0)
    print(f"  {'Image only (EfficientNetB3)':35s} AUC={auc_img:.4f} Recall={rec_img:.3f}")

    # ──────────────────────────────────────────────────────────────────────
    # Full comparison table
    # ──────────────────────────────────────────────────────────────────────
    all_aucs = {
        "Clinical only (4K XGB)":        auc_clin,
        "Image only (EfficientNetB3)":   auc_img,
        "Strategy A (LR fusion)":        auc_A,
        "Strategy B (Tiny NN fusion)":   auc_B,
        "Strategy C (XGB fusion 5-fold)": auc_C,
    }

    print(f"""
{'=' * 60}
  FUSION v2 — FULL COMPARISON (test set)
{'=' * 60}
  {"Model":<38} {"AUC":>6}
  {"-" * 46}""")
    for name, auc in all_aucs.items():
        marker = " ← BEST" if auc == max(all_aucs.values()) else ""
        print(f"  {name:<38} {auc:.4f}{marker}")
    print("=" * 60)

    # Pick and save best fusion strategy
    best_fusion_name = max(
        {"A": auc_A, "B": auc_B, "C": auc_C}, key=lambda k: {"A": auc_A, "B": auc_B, "C": auc_C}[k]
    )
    best_auc = max(auc_A, auc_B, auc_C)
    print(f"\n  Best fusion: Strategy {best_fusion_name} (AUC={best_auc:.4f})")

    final_path_pkl   = os.path.join(MODELS_DIR, "fusion_model_final.pkl")
    final_path_keras = os.path.join(MODELS_DIR, "fusion_model_final.keras")

    if best_fusion_name == "B":
        model_B.save(final_path_keras)
        print(f"  Saved: {final_path_keras}")
        fusion_type = "keras"
    elif best_fusion_name == "C":
        joblib.dump(xgb_fusion, final_path_pkl)
        print(f"  Saved: {final_path_pkl}")
        fusion_type = "xgboost"
    else:
        joblib.dump(lr_fusion, final_path_pkl)
        print(f"  Saved: {final_path_pkl}")
        fusion_type = "logistic_regression"

    # Save config
    config = {
        "best_strategy":   f"Strategy {best_fusion_name}",
        "fusion_type":     fusion_type,
        "auc":             round(best_auc, 4),
        "auc_clinical":    round(auc_clin, 4),
        "auc_image":       round(auc_img,  4),
        "pca_n_components": 32,
        "fusion_input_dim": int(X_fusion.shape[1]),
    }
    with open(os.path.join(MODELS_DIR, "fusion_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # ROC plot
    # ──────────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, proba, auc in [
            ("Clinical only",    clin[test_idx, 1],          auc_clin),
            ("Image only",       img_proba[test_idx].flatten(), auc_img),
            ("Strategy A (LR)",  proba_A,                    auc_A),
            ("Strategy B (NN)",  proba_B,                    auc_B),
            ("Strategy C (XGB)", proba_C,                    auc_C),
        ]:
            fpr, tpr, _ = roc_curve(y[test_idx], proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Fusion v2 — ROC Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        roc_path = os.path.join(GRAPHS_DIR, "fusion_v2_roc_comparison.png")
        plt.savefig(roc_path, dpi=150)
        plt.close()
        print(f"\n  ROC plot saved: {roc_path}")
    except Exception as e:
        print(f"  [WARN] Could not save ROC plot: {e}")

    # ──────────────────────────────────────────────────────────────────────
    # Validation checklist
    # ──────────────────────────────────────────────────────────────────────
    df4k = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv"))
    y4k  = df4k["uti_label"].values
    clin_inf_mean  = clin[y4k == 1, 1].mean()
    clin_norm_mean = clin[y4k == 0, 1].mean()
    img_inf_mean   = img_proba[y4k == 1].mean()
    img_norm_mean  = img_proba[y4k == 0].mean()

    checks = [
        ("Clinical 4K model AUC > 0.80",               auc_clin >= 0.80),
        ("Clinical infected mean prob > 0.60",          clin_inf_mean > 0.60),
        ("Clinical normal mean prob < 0.40",            clin_norm_mean < 0.40),
        ("Image infected mean prob > 0.60",             img_inf_mean > 0.60),
        ("Image normal mean prob < 0.40",               img_norm_mean < 0.40),
        ("Train/val AUC gap < 0.10 (no overfit)",       gap < 0.10),
        ("Best fusion AUC > 0.88",                      best_auc > 0.88),
        ("Fusion AUC > both individual model AUCs",
         best_auc > auc_clin and best_auc > auc_img),
        ("models/fusion_model_final saved",
         os.path.exists(final_path_pkl) or os.path.exists(final_path_keras)),
    ]

    print(f"\n{'=' * 60}")
    print("  FUSION v2 VALIDATION CHECKLIST")
    print("=" * 60)
    passes = 0
    for desc, result in checks:
        status = "PASS ✓" if result else "FAIL ✗"
        if result:
            passes += 1
        print(f"  [{status}] {desc}")
    print(f"\n  {passes}/{len(checks)} checks passed")
    print("=" * 60)

    if best_auc > max(auc_clin, auc_img):
        print("\n✓ FIX 4 COMPLETE — Fusion outperforms both individual models  SUCCESS")
    else:
        print("\n⚠ FIX 4 PARTIAL — Fusion does not yet outperform both individual models")
        print("  Consider: more epochs, different regularisation, or better clinical signal")


if __name__ == "__main__":
    main()
