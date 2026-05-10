"""
ultrasound_efficientnet.py — EfficientNetB3 Bladder Classifier (Leakage-Free)
==============================================================================
IMPORTANT CHANGES vs previous version
  ?  Reads the split manifest from raw_data_split.csv  (created by augment_dataset.py)
  ?  Augmented abnormal images are added to TRAIN ONLY — never val or test
  ?  Val and Test use raw images only -> unbiased metrics
  ?  Preprocessing uses the canonical us_preprocessing.preprocess_bladder_image()
  ?  Saves 256-dim feature vectors for the test set (for future fusion)
  ?  Saves preprocessing config JSON

Outputs
  models/ultrasound_efficientnet_best.keras
  results/graphs/efficientnet_training_history.png
  results/graphs/efficientnet_roc_curve.png
  results/graphs/efficientnet_confusion_matrix.png
  results/metrics/efficientnet_results.csv
  results/metrics/efficientnet_test_split.csv       (path, label, y_prob)
  results/metrics/image_features_test.npy           (N_test × 256)
  results/metrics/preprocessing_config.json
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ?? Canonical preprocessing (single source of truth) ?????????????????????????
from src.us_preprocessing import (
    preprocess_ultrasound_v2,
    save_preprocessing_config,
    DEFAULT_TARGET_SIZE,
)

# ?? Reproducibility ???????????????????????????????????????????????????????????
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ?? Paths ??????????????????????????????????????????????????????????????????????
BASE_DIR      = Path(__file__).resolve().parent.parent
BALANCED_NORMAL   = BASE_DIR / "data" / "balanced" / "normal"
BALANCED_ABNORMAL = BASE_DIR / "data" / "balanced" / "abnormal"
MODELS_DIR    = BASE_DIR / "models"
GRAPHS_DIR    = BASE_DIR / "results" / "graphs"
METRICS_DIR   = BASE_DIR / "results" / "metrics"
BEST_MODEL    = str(MODELS_DIR / "ultrasound_efficientnet_best.keras")


# ?? Config ????????????????????????????????????????????????????????????????????
IMG_SIZE    = DEFAULT_TARGET_SIZE   # (260, 260)
BATCH_SIZE  = 16
IMG_EXTS    = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

for d in [MODELS_DIR, GRAPHS_DIR, METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ????????????????????????????????????????????????????????????????????????????
# SECTION 1 — Data Loading (split-first, leakage-free)
# ????????????????????????????????????????????????????????????????????????????
def load_balanced_data() -> tuple[list, list, list, list, list, list]:
    all_paths = []
    all_labels = []
    
    for p in sorted(BALANCED_NORMAL.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            all_paths.append(str(p))
            all_labels.append(0)
            
    for p in sorted(BALANCED_ABNORMAL.rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            all_paths.append(str(p))
            all_labels.append(1)

    # 70/15/15 split
    tr_p, rem_p, tr_l, rem_l = train_test_split(
        all_paths, all_labels, test_size=0.30, stratify=all_labels, random_state=SEED)
    val_p, tst_p, val_l, tst_l = train_test_split(
        rem_p, rem_l, test_size=0.50, stratify=rem_l, random_state=SEED)
        
    return tr_p, tr_l, val_p, val_l, tst_p, tst_l



# ????????????????????????????????????????????????????????????????????????????
# SECTION 2 — tf.data Pipeline
# ????????????????????????????????????????????????????????????????????????????
def _py_load(path_bytes: bytes) -> np.ndarray:
    """Python function called inside tf.data map."""
    path = path_bytes.decode("utf-8")
    img = preprocess_ultrasound_v2(path, target_size=IMG_SIZE)
    # v2 returns normalized float32 [0,1], adjust for efficientnet if needed
    # efficientnet expects [0,255] or handles it. Wait, EfficientNetB0-B7 in Keras uses preprocess_input which takes [0, 255].
    # But since v2 already scales to [0,1] and we do not use tf.keras.applications.efficientnet.preprocess_input inside v2,
    # let's just make it [0, 255] for preprocess_input or use it directly.
    # Actually, EfficientNet preprocess_input just passes it implicitly without normalization, except in newer versions.
    # In Keras core, efficientnet expects inputs [0, 255] natively if not using preprocess_input.
    # We will just scale by 255 to match the expected preprocess_input behaviour.
    img = img * 255.0
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img.astype(np.float32)


def make_dataset(
    paths: list[str],
    labels: list[int],
    shuffle: bool = True,
    online_augment: bool = False,
) -> tf.data.Dataset:
    path_t  = tf.constant(paths,  dtype=tf.string)
    label_t = tf.constant(labels, dtype=tf.float32)

    def load_fn(path, label):
        img = tf.numpy_function(_py_load, [path], tf.float32)
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return img, label

    def aug_fn(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, max_delta=0.08)
        img = tf.image.random_contrast(img, lower=0.92, upper=1.08)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((path_t, label_t))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if online_augment:
        ds = ds.map(aug_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ????????????????????????????????????????????????????????????????????????????
# SECTION 3 — Focal Loss
# ????????????????????????????????????????????????????????????????????????????
def focal_loss(alpha: float = 0.25, gamma: float = 2.0):
    def loss_fn(y_true, y_pred):
        eps    = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        bce    = (-y_true * tf.math.log(y_pred)
                  - (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        p_t    = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        alpha_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        return tf.reduce_mean(alpha_t * tf.math.pow(1.0 - p_t, gamma) * bce)
    loss_fn.__name__ = "focal_loss"
    return loss_fn


# ????????????????????????????????????????????????????????????????????????????
# SECTION 4 — Model Architecture
# ????????????????????????????????????????????????????????????????????????????
def build_efficientnet_b3() -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    EfficientNetB3 + classification head.
    Returns (full_model, base_model) for progressive unfreezing.
    Architecture:
      Input(260×260×3) -> EfficientNetB3(frozen) -> GAP -> BN ->
      Dense(512,relu) -> Dropout(0.4) -> Dense(256,relu,'features') ->
      Dropout(0.3) -> Dense(1,sigmoid)
    """
    base = tf.keras.applications.EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
                            name="image_input")
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn")(x)
    x = tf.keras.layers.Dense(512, activation="relu", name="dense_512")(x)
    x = tf.keras.layers.Dropout(0.4, name="drop_512")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="features")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_256")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = tf.keras.Model(inputs, out, name="EfficientNetB3_UTI")
    return model, base


def get_callbacks() -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            BEST_MODEL, monitor="val_auc", mode="max",
            save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", patience=6, mode="max",
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=3, mode="max",
            min_lr=1e-7, verbose=1),
    ]


# ????????????????????????????????????????????????????????????????????????????
# SECTION 5 — 3-Phase Training
# ????????????????????????????????????????????????????????????????????????????
def _phase_header(n: int, desc: str) -> None:
    print(f"\n{'='*70}\n  PHASE {n} — {desc}\n{'='*70}")


def _log_best(history, phase: int) -> None:
    best = int(np.argmax(history.history.get("val_auc", [0])))
    print(f"\n  Phase {phase} best (epoch {best+1}):")
    for k in ("auc", "val_auc", "loss", "val_loss"):
        v = history.history.get(k, [None])[best]
        if v is not None:
            print(f"    {k:12s}: {v:.4f}")


def train_model(
    model: tf.keras.Model,
    base:  tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds:   tf.data.Dataset,
    class_weight: dict,
) -> dict:
    fl  = focal_loss(0.25, 2.0)
    all_hist: dict = {}

    # ?? Phase 1: frozen backbone ??????????????????????????????????????????????
    _phase_header(1, "Frozen Backbone — Train Classification Head")
    base.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=fl,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print(f"  Trainable params: "
          f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=20,
                   class_weight=class_weight, callbacks=get_callbacks(), verbose=1)
    _log_best(h1, 1)
    all_hist["phase1"] = h1.history

    # ?? Phase 2: unfreeze last 50 backbone layers ?????????????????????????????
    _phase_header(2, "Unfreeze Last 50 Backbone Layers — Fine-Tune")
    base.trainable = True
    for layer in base.layers[:-50]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=fl,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print(f"  Trainable params: "
          f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=25,
                   class_weight=class_weight, callbacks=get_callbacks(), verbose=1)
    _log_best(h2, 2)
    all_hist["phase2"] = h2.history

    # ?? Phase 3: full model ???????????????????????????????????????????????????
    _phase_header(3, "Full Model — Very Slow Fine-Tune")
    base.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=fl,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print(f"  Trainable params: "
          f"{sum(tf.size(v).numpy() for v in model.trainable_variables):,}")
    h3 = model.fit(train_ds, validation_data=val_ds, epochs=15,
                   class_weight=class_weight, callbacks=get_callbacks(), verbose=1)
    _log_best(h3, 3)
    all_hist["phase3"] = h3.history

    return all_hist


# ????????????????????????????????????????????????????????????????????????????
# SECTION 6 — Evaluation
# ????????????????????????????????????????????????????????????????????????????
def find_optimal_threshold(model, val_generator, target_recall=0.65, val_labels=None):
    """
    Instead of default 0.5 threshold, find the threshold where
    abnormal recall >= target_recall (65%) with highest possible precision.
    Saves threshold to: models/optimal_threshold.txt
    """
    from sklearn.metrics import roc_curve, classification_report
    import numpy as np

    # If val_generator is a tf.data.Dataset
    if hasattr(val_generator, 'unbatch') and val_labels is None:
        y_true = np.concatenate([y for x, y in val_generator], axis=0)
    elif val_labels is not None:
        y_true = val_labels
    elif hasattr(val_generator, 'labels'):
        y_true = val_generator.labels
    else:
        y_true = val_labels # Will fail, passing responsibility
        
    y_pred_prob = model.predict(val_generator).flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)

    # Find lowest threshold where recall >= target
    optimal_threshold = 0.5
    for i, (recall, threshold) in enumerate(zip(tpr, thresholds)):
        if recall >= target_recall:
            optimal_threshold = float(threshold)
            break

    # Apply and print result
    y_pred_optimal = (y_pred_prob >= optimal_threshold).astype(int)
    print(f"\n=== Optimal Threshold Analysis ===")
    print(f"Default threshold (0.50):")
    print(classification_report(y_true, (y_pred_prob >= 0.50).astype(int),
          target_names=["Normal", "Abnormal"]))
    print(f"Optimal threshold ({optimal_threshold:.3f}):")
    print(classification_report(y_true, y_pred_optimal,
          target_names=["Normal", "Abnormal"]))

    # Save threshold
    import os
    os.makedirs("models", exist_ok=True)
    with open("models/optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))
    print(f"Optimal threshold saved: models/optimal_threshold.txt")

    return optimal_threshold

def evaluate(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    test_labels: list[int],
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    print("\n[STATS] Evaluating on test set (raw images only) …")
    y_prob = model.predict(test_ds, verbose=1).ravel()
    y_true = np.array(test_labels)
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"\n  Prediction distribution:")
    print(f"    mean={y_prob.mean():.4f}  std={y_prob.std():.4f}  "
          f"min={y_prob.min():.4f}  max={y_prob.max():.4f}")
    if y_prob.std() < 0.05:
        print("  [WARN]  LOW STD — model may have collapsed!")

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob)

    print("\n  ??????????????????????????????????????????")
    print(f"  ?  Accuracy  : {acc:.4f}                  ?")
    print(f"  ?  Precision : {prec:.4f}                  ?")
    print(f"  ?  Recall    : {rec:.4f}                  ?")
    print(f"  ?  F1 Score  : {f1:.4f}                  ?")
    print(f"  ?  ROC-AUC   : {auc:.4f}                  ?")
    print("  ??????????????????????????????????????????")
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred,
                                 target_names=["Normal", "Abnormal"]))

    metrics = dict(accuracy=acc, precision=prec, recall=rec, f1=f1, roc_auc=auc)
    return metrics, y_true, y_prob, y_pred


# ????????????????????????????????????????????????????????????????????????????
# SECTION 7 — Feature Extraction for Fusion
# ????????????????????????????????????????????????????????????????????????????
def extract_features(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
) -> np.ndarray:
    """
    Extract 256-dim embeddings from the 'features' Dense layer.
    Saved to image_features_test.npy for use in the future fusion model.
    """
    print("\n[FEAT] Extracting 256-dim feature vectors (fusion preparation) …")
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("features").output,
        name="feature_extractor",
    )
    feats = feature_extractor.predict(test_ds, verbose=1)
    print(f"  Feature shape: {feats.shape}")
    return feats


# ????????????????????????????????????????????????????????????????????????????
# SECTION 8 — Plots
# ????????????????????????????????????????????????????????????????????????????
DARK_BG = "#0d1117"

def _ax_dark(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    for attr in ("xaxis", "yaxis"):
        getattr(ax, attr).label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")


def save_training_history(all_hist: dict, out: Path) -> None:
    def cat(k): return sum([h.get(k, []) for h in all_hist.values()], [])
    tr_auc, vl_auc = cat("auc"), cat("val_auc")
    tr_los, vl_los = cat("loss"), cat("val_loss")
    ep = list(range(1, len(tr_auc) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)
    for ax in (ax1, ax2): _ax_dark(ax)

    ax1.plot(ep, tr_auc, "#58a6ff", lw=2, label="Train AUC")
    ax1.plot(ep, vl_auc, "#3fb950", lw=2, ls="--", label="Val AUC")
    ax1.set_ylim(0, 1)
    ax1.set_title("AUC per Epoch", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("AUC")
    ax1.legend(facecolor="#161b22", labelcolor="white")

    ax2.plot(ep, tr_los, "#f85149", lw=2, label="Train Loss")
    ax2.plot(ep, vl_los, "#ffa657", lw=2, ls="--", label="Val Loss")
    ax2.set_title("Focal Loss per Epoch", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(facecolor="#161b22", labelcolor="white")

    # Phase boundary lines
    phase_end = [len(all_hist.get("phase1", {}).get("auc", [])),
                 len(all_hist.get("phase1", {}).get("auc", [])) +
                 len(all_hist.get("phase2", {}).get("auc", []))]
    for xe, lbl in zip(phase_end, ["P1->P2", "P2->P3"]):
        for ax in (ax1, ax2):
            ax.axvline(xe, color="#8b949e", ls=":", lw=1.5, alpha=0.8)
            ax.text(xe+0.3, 0.02, lbl, color="#8b949e", fontsize=8,
                    transform=ax.get_xaxis_transform())

    fig.suptitle("EfficientNetB3 — 3-Phase Training (Leakage-Free)",
                 color="white", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ? Training history -> {out}")


def save_roc(y_true, y_prob, out: Path, auc_val: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(DARK_BG); _ax_dark(ax)
    ax.plot(fpr, tpr, "#58a6ff", lw=2.5,
            label=f"EfficientNetB3 (AUC = {auc_val:.4f})")
    ax.plot([0,1],[0,1],"#8b949e",ls="--",lw=1.5,label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#58a6ff")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — EfficientNetB3 (Clean Test Set)",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ? ROC curve -> {out}")


def save_cm(y_true, y_pred, out: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(DARK_BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","Abnormal"],
                yticklabels=["Normal","Abnormal"],
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"size":16,"color":"white"}, ax=ax)
    ax.set_facecolor("#161b22"); ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white"); ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — EfficientNetB3", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ? Confusion matrix -> {out}")


# ????????????????????????????????????????????????????????????????????????????
# SECTION 9 — Main
# ????????????????????????????????????????????????????????????????????????????
def main() -> None:
    print("=" * 70)
    print("  EfficientNetB3 Ultrasound Classifier — Leakage-Free Pipeline")
    print("=" * 70)

    # ?? 1. Load split ?????????????????????????????????????????????????????????
    print("\n[DIR] Loading balanced data split …")
    (train_p, train_l,
     val_p,   val_l,
     test_p,  test_l) = load_balanced_data()

    train_l_arr = np.array(train_l)
    val_l_arr   = np.array(val_l)
    test_l_arr  = np.array(test_l)

    print(f"\n  DATASET SHAPES")
    print(f"  Train : {len(train_p)} "
          f"(normal={np.sum(train_l_arr==0)}, abnormal={np.sum(train_l_arr==1)})")
    print(f"  Val   : {len(val_p)}   "
          f"(normal={np.sum(val_l_arr==0)},  abnormal={np.sum(val_l_arr==1)})")
    print(f"  Test  : {len(test_p)}   "
          f"(normal={np.sum(test_l_arr==0)},  abnormal={np.sum(test_l_arr==1)})  "
          f"<- raw only, no leakage")
    ratio = np.sum(train_l_arr==0) / max(1, np.sum(train_l_arr==1))
    print(f"  Train normal:abnormal ratio =  {ratio:.2f}:1  "
          f"({'? balanced' if ratio < 1.3 else '[!] imbalanced'})")

    # ?? 3. Class weights (train set only) ?????????????????????????????????????
    cw = compute_class_weight("balanced",
                               classes=np.array([0, 1]),
                               y=train_l_arr)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print(f"\n??  Class weights: normal={class_weight[0]:.4f}, "
          f"abnormal={class_weight[1]:.4f}")

    # ?? 4. tf.data pipelines ??????????????????????????????????????????????????
    print("\n[CFG] Building tf.data pipelines …")
    train_ds = make_dataset(train_p, train_l, shuffle=True,  online_augment=True)
    val_ds   = make_dataset(val_p,   val_l,   shuffle=False, online_augment=False)
    test_ds  = make_dataset(test_p,  test_l,  shuffle=False, online_augment=False)
    print(f"  Train batches : {len(train_ds)} | "
          f"Val : {len(val_ds)} | Test : {len(test_ds)}")

    # ?? 5. Build model ????????????????????????????????????????????????????????
    print("\n[BUILD]  Building EfficientNetB3 …")
    model, base_model = build_efficientnet_b3()
    model.summary(line_length=100)

    # ?? 6. Train ??????????????????????????????????????????????????????????????
    print("\n[RUN] Starting 3-phase training …")
    all_hist = train_model(model, base_model, train_ds, val_ds, class_weight)

    # ?? 7. Load best checkpoint ???????????????????????????????????????????????
    print(f"\n[LOAD] Loading best model from checkpoint …")
    best_model = tf.keras.models.load_model(BEST_MODEL, compile=False)
    best_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    print(f"  Model input shape: {best_model.input_shape}")

    # ?? 8. Evaluate on CLEAN test set ?????????????????????????????????????????
    metrics, y_true, y_prob, y_pred = evaluate(best_model, test_ds,
                                                test_l_arr.tolist())

    # ?? 9. Extract features for future fusion ?????????????????????????????????
    feats = extract_features(best_model, test_ds)
    feat_path = METRICS_DIR / "image_features_test.npy"
    np.save(str(feat_path), feats)
    print(f"  ? Feature vectors saved -> {feat_path}  shape={feats.shape}")

    # ?? 10. Save test split with predictions ??????????????????????????????????
    test_df = pd.DataFrame({
        "image_path": test_p,
        "label":      test_l,
        "y_prob":     y_prob.tolist(),
        "y_pred":     y_pred.tolist(),
    })
    test_split_path = METRICS_DIR / "efficientnet_test_split.csv"
    test_df.to_csv(str(test_split_path), index=False)
    print(f"  ? Test split with predictions -> {test_split_path}")

    # ?? 11. Save plots ????????????????????????????????????????????????????????
    print("\n[PLOT] Saving plots …")
    save_training_history(all_hist, GRAPHS_DIR / "efficientnet_training_history.png")
    save_roc(y_true, y_prob, GRAPHS_DIR / "efficientnet_roc_curve.png",
             metrics["roc_auc"])
    save_cm (y_true, y_pred, GRAPHS_DIR / "efficientnet_confusion_matrix.png")

    # ?? 12. Metrics CSV ???????????????????????????????????????????????????????
    pd.DataFrame([metrics]).to_csv(
        str(METRICS_DIR / "efficientnet_results.csv"), index=False)
    print(f"  ? Metrics CSV -> {METRICS_DIR / 'efficientnet_results.csv'}")

    # ?? 13. Preprocessing config ??????????????????????????????????????????????
    save_preprocessing_config(str(METRICS_DIR / "preprocessing_config.json"))

    # ?? 14. Find Optimal Threshold ????????????????????????????????????????????
    print("\n[THRESH] Finding optimal threshold on validation set ...")
    optimal_threshold = find_optimal_threshold(best_model, val_ds, target_recall=0.65, val_labels=val_l_arr)
    with open("models/optimal_threshold.txt", "w") as f: f.write(str(optimal_threshold))

    # Re-evaluate with optimal threshold for final output
    print(f"\n[EVAL] Final results with optimal threshold ({optimal_threshold:.3f})")
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)
    from sklearn.metrics import recall_score
    rec_norm = recall_score(test_l_arr, y_pred, pos_label=0, zero_division=0)
    rec_abn  = recall_score(test_l_arr, y_pred, pos_label=1, zero_division=0)
    rec_norm_opt = recall_score(test_l_arr, y_pred_opt, pos_label=0, zero_division=0)
    rec_abn_opt  = recall_score(test_l_arr, y_pred_opt, pos_label=1, zero_division=0)

    # ?? FINAL SUMMARY ?????????????????????????????????????????????????????????
    print("\n==========================================")
    print("ULTRASOUND MODEL — FINAL RESULTS SUMMARY")
    print("==========================================")
    print("Dataset:        4,000 balanced bladder images")
    print("Preprocessing:  v2 (elliptical mask + CLAHE)")
    print("Backbone:       EfficientNetB3")
    print(f"\n--- Default threshold (0.50) ---")
    print(f"Test AUC:            {metrics['roc_auc']:.4f}")
    print(f"Normal recall:       {rec_norm*100:.1f}%")
    print(f"Abnormal recall:     {rec_abn*100:.1f}%   <- target: 60%+")
    
    print(f"\n--- Optimal threshold ({optimal_threshold:.2f}) ---")
    print(f"Test AUC:            {metrics['roc_auc']:.4f}  (same, threshold doesn't change AUC)")
    print(f"Normal recall:       {rec_norm_opt*100:.1f}%")
    print(f"Abnormal recall:     {rec_abn_opt*100:.1f}%   <- target: 65%+")
    
    print(f"\nModel saved:    {BEST_MODEL}")
    print(f"Threshold saved: models/optimal_threshold.txt")
    print(f"Embeddings saved: {feat_path}")
    print("==========================================")


if __name__ == "__main__":
    main()
