"""
multimodal_fusion.py — Train and evaluate the multimodal fusion model
=====================================================================
Combines clinical embeddings (from XGBoost preprocessor) with image
embeddings (from EfficientNetB3 features layer) using a neural fusion
architecture. Compares all 3 models on the same test set.

Outputs:
  models/fusion_model_best.keras
  results/metrics/fusion_comparison.csv
  results/graphs/fusion_roc_comparison.png
  results/graphs/fusion_confusion_matrix.png
  results/graphs/fusion_training_history.png
  results/embeddings/fusion_test_predictions.csv
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib

# Reproducibility
SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUSION_CSV = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
EMB_DIR    = os.path.join(BASE_DIR, "results", "embeddings")
MODELS_DIR = os.path.join(BASE_DIR, "models")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")
METRICS_DIR = os.path.join(BASE_DIR, "results", "metrics")


def load_data():
    """Load embeddings and fusion pairs, split by train/val/test."""
    print("\n[DATA] Loading embeddings and fusion pairs ...")

    pairs = pd.read_csv(FUSION_CSV)
    clinical_features = np.load(os.path.join(EMB_DIR, "clinical_features.npy"))
    image_embeddings  = np.load(os.path.join(EMB_DIR, "image_embeddings.npy"))
    clinical_proba    = np.load(os.path.join(EMB_DIR, "clinical_proba.npy"))

    print(f"  Fusion pairs:       {pairs.shape[0]}")
    print(f"  Clinical features:  {clinical_features.shape}")
    print(f"  Image embeddings:   {image_embeddings.shape}")
    print(f"  Clinical proba:     {clinical_proba.shape}")

    train_mask = pairs["split"] == "train"
    val_mask   = pairs["split"] == "val"
    test_mask  = pairs["split"] == "test"

    clin_dim = clinical_features.shape[1]
    img_dim  = image_embeddings.shape[1]

    data = {
        "X_clin_train": clinical_features[train_mask],
        "X_img_train":  image_embeddings[train_mask],
        "y_train":      pairs.loc[train_mask, "label"].values.astype(np.float32),

        "X_clin_val": clinical_features[val_mask],
        "X_img_val":  image_embeddings[val_mask],
        "y_val":      pairs.loc[val_mask, "label"].values.astype(np.float32),

        "X_clin_test": clinical_features[test_mask],
        "X_img_test":  image_embeddings[test_mask],
        "y_test":      pairs.loc[test_mask, "label"].values.astype(np.float32),

        "clin_proba_test": clinical_proba[test_mask],

        "clin_dim": clin_dim,
        "img_dim":  img_dim,
        "pairs":    pairs,
    }

    print(f"\n  Train: {data['y_train'].shape[0]} "
          f"(pos={int(data['y_train'].sum())}, neg={int((1-data['y_train']).sum())})")
    print(f"  Val:   {data['y_val'].shape[0]} "
          f"(pos={int(data['y_val'].sum())}, neg={int((1-data['y_val']).sum())})")
    print(f"  Test:  {data['y_test'].shape[0]} "
          f"(pos={int(data['y_test'].sum())}, neg={int((1-data['y_test']).sum())})")

    return data


def build_fusion_model(clin_dim, img_dim):
    """Build the multimodal fusion architecture."""
    import tensorflow as tf

    # Branch 1: Clinical encoder
    clin_input = tf.keras.Input(shape=(clin_dim,), name="clinical_input")
    c = tf.keras.layers.Dense(128, activation="relu")(clin_input)
    c = tf.keras.layers.BatchNormalization()(c)
    c = tf.keras.layers.Dropout(0.3)(c)
    c = tf.keras.layers.Dense(64, activation="relu")(c)
    c = tf.keras.layers.Dropout(0.2)(c)
    c = tf.keras.layers.Dense(32, activation="relu")(c)

    # Branch 2: Image encoder
    img_input = tf.keras.Input(shape=(img_dim,), name="image_input")
    i = tf.keras.layers.Dense(128, activation="relu")(img_input)
    i = tf.keras.layers.BatchNormalization()(i)
    i = tf.keras.layers.Dropout(0.3)(i)
    i = tf.keras.layers.Dense(64, activation="relu")(i)

    # Fusion head
    fused = tf.keras.layers.Concatenate()([c, i])  # (96,)
    f = tf.keras.layers.Dense(128, activation="relu")(fused)
    f = tf.keras.layers.BatchNormalization()(f)
    f = tf.keras.layers.Dropout(0.4)(f)
    f = tf.keras.layers.Dense(64, activation="relu")(f)
    f = tf.keras.layers.Dropout(0.3)(f)
    output = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(f)

    model = tf.keras.Model(
        inputs=[clin_input, img_input],
        outputs=output,
        name="multimodal_fusion",
    )
    return model


def train_fusion(data):
    """Train the fusion model in 2 stages."""
    import tensorflow as tf
    tf.random.set_seed(SEED)

    from sklearn.utils.class_weight import compute_class_weight

    model = build_fusion_model(data["clin_dim"], data["img_dim"])
    model.summary()

    # Class weights
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=data["y_train"])
    class_weight_dict = {0: float(cw[0]), 1: float(cw[1])}
    print(f"\n  Class weights: {class_weight_dict}")

    best_path = os.path.join(MODELS_DIR, "fusion_model_best.keras")

    callbacks_s1 = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
    ]

    # ── Stage 1: lr=1e-3 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 1 — Train fusion model (lr=1e-3)")
    print("=" * 60)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    h1 = model.fit(
        [data["X_clin_train"], data["X_img_train"]], data["y_train"],
        validation_data=([data["X_clin_val"], data["X_img_val"]], data["y_val"]),
        epochs=50,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=callbacks_s1,
        verbose=1,
    )

    # ── Stage 2: lr=1e-4 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STAGE 2 — Fine-tune fusion model (lr=1e-4)")
    print("=" * 60)

    callbacks_s2 = [
        tf.keras.callbacks.ModelCheckpoint(
            best_path, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=8, mode="max",
            restore_best_weights=True, verbose=1),
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    h2 = model.fit(
        [data["X_clin_train"], data["X_img_train"]], data["y_train"],
        validation_data=([data["X_clin_val"], data["X_img_val"]], data["y_val"]),
        epochs=30,
        batch_size=16,
        class_weight=class_weight_dict,
        callbacks=callbacks_s2,
        verbose=1,
    )

    return model, h1, h2, best_path


def evaluate_all_models(data, best_path):
    """Evaluate clinical-only, image-only, and fusion on the SAME test set."""
    import tensorflow as tf
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, precision_score, recall_score,
        f1_score, roc_curve, confusion_matrix, classification_report,
    )

    print("\n" + "=" * 60)
    print("  EVALUATION — All 3 models on same test set")
    print("=" * 60)

    y_test = data["y_test"]

    # ── 1) Clinical only (XGBoost) ────────────────────────────────────────
    clin_proba = data["clin_proba_test"][:, 1]  # probability of class 1
    clin_pred = (clin_proba >= 0.5).astype(int)
    clin_auc  = roc_auc_score(y_test, clin_proba)
    clin_acc  = accuracy_score(y_test, clin_pred)
    clin_prec = precision_score(y_test, clin_pred, zero_division=0)
    clin_rec  = recall_score(y_test, clin_pred, zero_division=0)
    clin_f1   = f1_score(y_test, clin_pred, zero_division=0)

    # ── 2) Image only (use a simple sigmoid on image embeddings) ──────────
    # Train a small single-layer model on image embeddings for fair comparison
    img_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu",
                              input_shape=(data["img_dim"],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    img_model.compile(
        optimizer="adam", loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")]
    )
    img_model.fit(
        data["X_img_train"], data["y_train"],
        validation_data=(data["X_img_val"], data["y_val"]),
        epochs=30, batch_size=32, verbose=0,
    )
    img_proba = img_model.predict(data["X_img_test"], verbose=0).flatten()
    img_pred  = (img_proba >= 0.5).astype(int)
    img_auc   = roc_auc_score(y_test, img_proba)
    img_acc   = accuracy_score(y_test, img_pred)
    img_prec  = precision_score(y_test, img_pred, zero_division=0)
    img_rec   = recall_score(y_test, img_pred, zero_division=0)
    img_f1    = f1_score(y_test, img_pred, zero_division=0)

    # ── 3) Fusion model ──────────────────────────────────────────────────
    fusion_model = tf.keras.models.load_model(best_path)
    fusion_proba = fusion_model.predict(
        [data["X_clin_test"], data["X_img_test"]], verbose=0
    ).flatten()
    fusion_pred = (fusion_proba >= 0.5).astype(int)
    fusion_auc  = roc_auc_score(y_test, fusion_proba)
    fusion_acc  = accuracy_score(y_test, fusion_pred)
    fusion_prec = precision_score(y_test, fusion_pred, zero_division=0)
    fusion_rec  = recall_score(y_test, fusion_pred, zero_division=0)
    fusion_f1   = f1_score(y_test, fusion_pred, zero_division=0)

    # ── Print comparison ──────────────────────────────────────────────────
    print(f"""
============================================================
  FINAL MULTIMODAL COMPARISON — SAME TEST SET ({len(y_test)} pairs)
============================================================
  Model                  AUC      Acc    Prec   Recall  F1
  Clinical only (XGB)    {clin_auc:.4f}   {clin_acc*100:.1f}%   {clin_prec*100:.1f}%   {clin_rec*100:.1f}%   {clin_f1:.2f}
  Image only (EffNet)    {img_auc:.4f}   {img_acc*100:.1f}%   {img_prec*100:.1f}%   {img_rec*100:.1f}%   {img_f1:.2f}
  Fusion (Multimodal)    {fusion_auc:.4f}   {fusion_acc*100:.1f}%   {fusion_prec*100:.1f}%   {fusion_rec*100:.1f}%   {fusion_f1:.2f}  ← TARGET 0.90+
============================================================
""")

    # Detailed fusion report
    print("  Fusion model — detailed classification report:")
    print(classification_report(y_test, fusion_pred,
                                target_names=["Normal", "Infected"]))

    # ── Save comparison CSV ───────────────────────────────────────────────
    os.makedirs(METRICS_DIR, exist_ok=True)
    comp_df = pd.DataFrame([
        {"model": "Clinical (XGBoost)", "AUC": clin_auc, "Accuracy": clin_acc,
         "Precision": clin_prec, "Recall": clin_rec, "F1": clin_f1},
        {"model": "Image (EfficientNetB3)", "AUC": img_auc, "Accuracy": img_acc,
         "Precision": img_prec, "Recall": img_rec, "F1": img_f1},
        {"model": "Fusion (Multimodal)", "AUC": fusion_auc, "Accuracy": fusion_acc,
         "Precision": fusion_prec, "Recall": fusion_rec, "F1": fusion_f1},
    ])
    comp_path = os.path.join(METRICS_DIR, "fusion_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  Comparison saved: {comp_path}")

    # ── Save test predictions ─────────────────────────────────────────────
    test_mask = data["pairs"]["split"] == "test"
    pred_df = pd.DataFrame({
        "pair_id":     data["pairs"].loc[test_mask, "pair_id"].values,
        "true_label":  y_test.astype(int),
        "clin_pred":   clin_proba,
        "img_pred":    img_proba,
        "fusion_pred": fusion_proba,
    })
    pred_path = os.path.join(EMB_DIR, "fusion_test_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Test predictions saved: {pred_path}")

    return {
        "clin_auc": clin_auc, "img_auc": img_auc, "fusion_auc": fusion_auc,
        "clin_proba": clin_proba, "img_proba": img_proba, "fusion_proba": fusion_proba,
        "y_test": y_test, "fusion_pred": fusion_pred,
    }


def save_plots(data, results, h1, h2):
    """Save ROC comparison, confusion matrix, and training history."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, confusion_matrix
    import seaborn as sns

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    DARK_BG = "#0d1117"

    # ── ROC Comparison ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor("#161b22")

    for name, proba, auc_val, color in [
        ("Clinical (XGB)", results["clin_proba"], results["clin_auc"], "#ffa657"),
        ("Image (EffNet)", results["img_proba"], results["img_auc"], "#3fb950"),
        ("Fusion",         results["fusion_proba"], results["fusion_auc"], "#58a6ff"),
    ]:
        fpr, tpr, _ = roc_curve(results["y_test"], proba)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], "#8b949e", ls="--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("ROC Comparison — All 3 Models (Same Test Set)",
                 color="white", fontsize=14, fontweight="bold")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    plt.tight_layout()
    roc_path = os.path.join(GRAPHS_DIR, "fusion_roc_comparison.png")
    plt.savefig(roc_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  ROC comparison saved: {roc_path}")

    # ── Confusion Matrix ──────────────────────────────────────────────────
    cm = confusion_matrix(results["y_test"], results["fusion_pred"])
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Infected"],
                yticklabels=["Normal", "Infected"],
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size": 16, "color": "white"}, ax=ax)
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Fusion Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(GRAPHS_DIR, "fusion_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {cm_path}")

    # ── Training History ──────────────────────────────────────────────────
    def cat(h, key):
        return h.history.get(key, [])

    s1_auc = cat(h1, "auc")
    s1_val_auc = cat(h1, "val_auc")
    s1_loss = cat(h1, "loss")
    s1_val_loss = cat(h1, "val_loss")
    s2_auc = cat(h2, "auc")
    s2_val_auc = cat(h2, "val_auc")
    s2_loss = cat(h2, "loss")
    s2_val_loss = cat(h2, "val_loss")

    all_auc = s1_auc + s2_auc
    all_val_auc = s1_val_auc + s2_val_auc
    all_loss = s1_loss + s2_loss
    all_val_loss = s1_val_loss + s2_val_loss
    epochs = list(range(1, len(all_auc) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    ax1.plot(epochs, all_auc, "#58a6ff", lw=2, label="Train AUC")
    ax1.plot(epochs, all_val_auc, "#3fb950", lw=2, ls="--", label="Val AUC")
    ax1.set_title("AUC per Epoch", color="white", fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("AUC", color="white")
    ax1.legend(facecolor="#161b22", labelcolor="white")
    if s1_auc:
        ax1.axvline(len(s1_auc), color="#8b949e", ls=":", lw=1.5)
        ax1.text(len(s1_auc) + 0.3, 0.5, "S1→S2", color="#8b949e", fontsize=9)

    ax2.plot(epochs, all_loss, "#f85149", lw=2, label="Train Loss")
    ax2.plot(epochs, all_val_loss, "#ffa657", lw=2, ls="--", label="Val Loss")
    ax2.set_title("Loss per Epoch", color="white", fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Loss", color="white")
    ax2.legend(facecolor="#161b22", labelcolor="white")
    if s1_loss:
        ax2.axvline(len(s1_loss), color="#8b949e", ls=":", lw=1.5)

    fig.suptitle("Fusion Model — 2-Stage Training",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    hist_path = os.path.join(GRAPHS_DIR, "fusion_training_history.png")
    plt.savefig(hist_path, dpi=150, facecolor=DARK_BG, bbox_inches="tight")
    plt.close()
    print(f"  Training history saved: {hist_path}")


def main():
    print("=" * 60)
    print("  TASK 3 — Build and Train the Fusion Model")
    print("=" * 60)

    data = load_data()

    # Train
    model, h1, h2, best_path = train_fusion(data)

    # Evaluate all 3 models
    results = evaluate_all_models(data, best_path)

    # Save plots
    save_plots(data, results, h1, h2)

    print(f"\n✓ TASK 3 COMPLETE")
    print(f"  Best fusion model: {best_path}")
    print(f"  Fusion AUC: {results['fusion_auc']:.4f}")

    return results


if __name__ == "__main__":
    main()
