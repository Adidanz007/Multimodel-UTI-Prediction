"""
================================================================================
TASK 3 — DenseNet121 vs EfficientNetB3 Comparison
================================================================================
Loads both saved models and evaluates them on the SAME held-out test set.
Outputs:
  results/metrics/model_comparison.csv
  results/graphs/model_comparison_roc.png
  Console: formatted comparison table
================================================================================
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
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
RAW_DIR     = BASE_DIR / "data" / "raw" / "ultrasound_images"
AUG_DIR     = BASE_DIR / "data" / "augmented"
MODELS_DIR  = BASE_DIR / "models"
GRAPHS_DIR  = BASE_DIR / "results" / "graphs"
METRICS_DIR = BASE_DIR / "results" / "metrics"

DENSENET_PATH    = str(MODELS_DIR / "ultrasound_best.keras")
EFFICIENTNET_PATH= str(MODELS_DIR / "ultrasound_efficientnet_best.keras")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BATCH_SIZE = 16

# ── Ensure output directories ─────────────────────────────────────────────────
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Shared Preprocessing helpers (same as ultrasound_efficientnet.py)
# ────────────────────────────────────────────────────────────────────────────
def preprocess_efficientnet(path: str, img_size=(260, 260)) -> np.ndarray:
    """CLAHE pipeline → EfficientNet normalisation."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*img_size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    crop_h = int(h * 0.10)
    img = img[crop_h: h - crop_h, :, :]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    img_3ch = np.stack([gray, gray, gray], axis=-1)
    img_3ch = cv2.resize(img_3ch, (img_size[1], img_size[0]))
    img_3ch = tf.keras.applications.efficientnet.preprocess_input(
        img_3ch.astype(np.float32)
    )
    return img_3ch


def preprocess_densenet(path: str, img_size=(224, 224)) -> np.ndarray:
    """Standard RGB resize + /255 pipeline compatible with the existing DenseNet model."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((*img_size, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size[1], img_size[0]))
    return img.astype(np.float32) / 255.0


# ────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Data Collection (mirroring the EfficientNet script's split)
# ────────────────────────────────────────────────────────────────────────────
def collect_all_images() -> tuple[list[str], list[int]]:
    paths: list[str] = []
    labels: list[int] = []
    class_map = {"normal": 0, "abnormal": 1}
    sources = [(RAW_DIR, "raw"), (AUG_DIR, "augmented")]

    for source_dir, source_name in sources:
        for class_name, label in class_map.items():
            class_dir = source_dir / class_name
            if not class_dir.exists():
                continue
            found = [str(p) for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
            paths.extend(found)
            labels.extend([label] * len(found))

    return paths, labels


def get_test_split(paths, labels):
    """Re-create the exact same test split used during training (same SEED)."""
    paths  = np.array(paths)
    labels = np.array(labels, dtype=int)

    _, rem_paths, _, rem_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    _, test_paths, _, test_labels = train_test_split(
        rem_paths, rem_labels, test_size=0.50, stratify=rem_labels, random_state=SEED
    )
    return test_paths.tolist(), test_labels.tolist()


# ────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset builders
# ────────────────────────────────────────────────────────────────────────────
def make_tf_dataset(paths, labels, preprocess_fn, img_size):
    path_t  = tf.constant(paths,  dtype=tf.string)
    label_t = tf.constant(labels, dtype=tf.float32)

    def load_fn(path, label):
        img = tf.numpy_function(
            func=lambda p: preprocess_fn(p.decode("utf-8"), img_size),
            inp=[path],
            Tout=tf.float32,
        )
        img.set_shape([img_size[0], img_size[1], 3])
        return img, label

    return (
        tf.data.Dataset.from_tensor_slices((path_t, label_t))
        .map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


# ────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Run evaluation for one model
# ────────────────────────────────────────────────────────────────────────────
def evaluate_one_model(model, ds, y_true) -> dict:
    y_prob = model.predict(ds, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_prob": y_prob,
        "y_pred": y_pred,
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":   roc_auc_score(y_true, y_prob),
    }


# ────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Comparison table
# ────────────────────────────────────────────────────────────────────────────
def print_comparison_table(results: dict) -> None:
    names = list(results.keys())
    def fmt(v): return f"{v*100:.2f}%"
    def fmt_auc(v): return f"{v:.4f}"

    print("\n")
    print("┌─────────────────┬──────────┬───────────┬────────┬──────────┐")
    print("│ Model           │ Accuracy │ Precision │ Recall │ ROC-AUC  │")
    print("├─────────────────┼──────────┼───────────┼────────┼──────────┤")
    for name, res in results.items():
        short = name[:15].ljust(15)
        print(f"│ {short} │ {fmt(res['accuracy']):>8} │ {fmt(res['precision']):>9} │ "
              f"{fmt(res['recall']):>6} │ {fmt_auc(res['roc_auc']):>8} │")
    print("└─────────────────┴──────────┴───────────┴────────┴──────────┘")


# ────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Combined ROC curve
# ────────────────────────────────────────────────────────────────────────────
DARK_BG = "#0d1117"
COLORS  = {"DenseNet121": "#ffa657", "EfficientNetB3": "#58a6ff"}

def save_comparison_roc(y_true, results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_true, res["y_prob"])
        color = COLORS.get(name, "#ff6b6b")
        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f"{name}  (AUC = {res['roc_auc']:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.07, color=color)

    ax.plot([0, 1], [0, 1], color="#8b949e", linestyle="--", lw=1.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve Comparison — DenseNet121 vs EfficientNetB3",
                 fontsize=13, fontweight="bold")
    ax.legend(facecolor="#161b22", labelcolor="white", fontsize=11)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"\n  ✓ Combined ROC → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Main
# ────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 70)
    print("  TASK 3 — Model Comparison: DenseNet121 vs EfficientNetB3")
    print("=" * 70)

    # ── Load test data ────────────────────────────────────────────────────
    print("\n📂 Collecting images and reconstructing test split …")
    all_paths, all_labels = collect_all_images()
    test_paths, test_labels = get_test_split(all_paths, all_labels)
    y_true = np.array(test_labels)
    print(f"  Test images : {len(test_paths)}  (normal={np.sum(y_true==0)}, abnormal={np.sum(y_true==1)})")

    results: dict = {}

    # ── DenseNet121 ───────────────────────────────────────────────────────
    if not Path(DENSENET_PATH).exists():
        print(f"\n  [WARN] DenseNet model not found at {DENSENET_PATH} — skipping")
    else:
        print(f"\n📥 Loading DenseNet121 from {DENSENET_PATH} …")
        densenet_model = tf.keras.models.load_model(DENSENET_PATH, compile=False)
        densenet_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        # DenseNet was trained on 224×224 RGB / 255
        densenet_ds = make_tf_dataset(test_paths, test_labels, preprocess_densenet, (224, 224))
        print("  Evaluating DenseNet121 …")
        results["DenseNet121"] = evaluate_one_model(densenet_model, densenet_ds, y_true)
        print(f"  DenseNet121 AUC: {results['DenseNet121']['roc_auc']:.4f}")

    # ── EfficientNetB3 ────────────────────────────────────────────────────
    if not Path(EFFICIENTNET_PATH).exists():
        print(f"\n  [WARN] EfficientNetB3 model not found at {EFFICIENTNET_PATH} — skipping")
    else:
        print(f"\n📥 Loading EfficientNetB3 from {EFFICIENTNET_PATH} …")
        efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_PATH, compile=False)
        efficientnet_model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC()],
        )
        efficientnet_ds = make_tf_dataset(test_paths, test_labels, preprocess_efficientnet, (260, 260))
        print("  Evaluating EfficientNetB3 …")
        results["EfficientNetB3"] = evaluate_one_model(efficientnet_model, efficientnet_ds, y_true)
        print(f"  EfficientNetB3 AUC: {results['EfficientNetB3']['roc_auc']:.4f}")

    if not results:
        print("\n  [ERROR] No models found — run training scripts first.")
        return

    # ── Print comparison table ─────────────────────────────────────────────
    print_comparison_table(results)

    # ── Save comparison CSV ───────────────────────────────────────────────
    rows = []
    for name, res in results.items():
        rows.append({
            "model":     name,
            "accuracy":  res["accuracy"],
            "precision": res["precision"],
            "recall":    res["recall"],
            "f1":        res["f1"],
            "roc_auc":   res["roc_auc"],
        })
    csv_path = METRICS_DIR / "model_comparison.csv"
    pd.DataFrame(rows).to_csv(str(csv_path), index=False)
    print(f"\n  ✓ Comparison CSV → {csv_path}")

    # ── Save combined ROC curve ───────────────────────────────────────────
    save_comparison_roc(y_true, results, GRAPHS_DIR / "model_comparison_roc.png")

    # ── Improvement summary ───────────────────────────────────────────────
    if "DenseNet121" in results and "EfficientNetB3" in results:
        dn_auc  = results["DenseNet121"]["roc_auc"]
        en_auc  = results["EfficientNetB3"]["roc_auc"]
        gain    = (en_auc - dn_auc) * 100
        print(f"\n  AUC Improvement: {dn_auc:.4f} → {en_auc:.4f}  ({gain:+.2f}% gain)")

    print("\n✅  Comparison complete!")


if __name__ == "__main__":
    main()
