"""
gradcam_efficientnet.py — Proper Grad-CAM for EfficientNetB3
=============================================================
Implements true Grad-CAM (Selvaraju et al. 2017):
  CAM = ReLU( ?_c  ?_c  *  A^c )
  where  ?_c = 1/Z * ?_{i,j}  ?y / ?A^c_{ij}  (global average of gradients)
  and    A^c is the activation map of the last conv layer channel c.

Compared to input-gradient saliency, Grad-CAM:
  ? Attends to class-discriminative SPATIAL REGIONS in the image
  ? Is resolution-invariant  (heatmap upsample to original size)
  ? Works with nested / sequential models

HOW IT HANDLES EfficientNetB3 NESTED MODEL
  The outer model structure is:
      Input(260×260×3) -> EfficientNetB3_backbone -> GAP -> BN -> Dense -> Dense -> Output
  
  This script:
  1. Finds the backbone (first sub-model in outer model)
  2. Locates the last Conv2D layer  (typically 'top_conv' in EfficientNetB3)
  3. Builds a GradCAM bridge model:
       outer_input -> [backbone_conv_output, full_model_output]
  4. Computes grad-cam on that bridge model
  5. Falls back to input-gradient saliency if the architecture is unexpected

OUTPUTS
  results/gradcam_efficientnet/normal_NN.png        (4 normal samples)
  results/gradcam_efficientnet/abnormal_NN.png      (4 abnormal samples)
  results/gradcam_efficientnet/attention_grid.png   (8-panel summary)
  results/metrics/gradcam_attention_analysis.csv    (center/edge ratios)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# ?? Canonical preprocessing ???????????????????????????????????????????????????
from src.us_preprocessing import preprocess_bladder_image, DEFAULT_TARGET_SIZE

# ?? Paths ?????????????????????????????????????????????????????????????????????
BASE_DIR      = Path(__file__).resolve().parent.parent
BEST_MODEL    = BASE_DIR / "models"  / "ultrasound_efficientnet_best.keras"
TEST_SPLIT    = BASE_DIR / "results" / "metrics" / "efficientnet_test_split.csv"
GRADCAM_DIR   = BASE_DIR / "results" / "gradcam_efficientnet"
METRICS_DIR   = BASE_DIR / "results" / "metrics"
GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True,  exist_ok=True)

DARK_BG = "#0d1117"
N_SAMPLES = 4   # samples per class


# ????????????????????????????????????????????????????????????????????????????
# SECTION 1 — Dynamic model interrogation
# ????????????????????????????????????????????????????????????????????????????
def get_model_input_size(model: tf.keras.Model) -> tuple[int, int]:
    sh = model.input_shape  # (None, H, W, C)
    return int(sh[1]), int(sh[2])


def find_backbone(model: tf.keras.Model) -> tf.keras.Model | None:
    """Return the first nested sub-model (the CNN backbone)."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(layer.layers) > 10:
            return layer
    return None


def find_last_conv_layer(backbone: tf.keras.Model) -> str | None:
    """
    Return the name of the last Conv2D layer in the backbone.
    Prefers 'top_conv' (EfficientNet standard); otherwise scans in reverse.
    """
    # EfficientNetB3 canonical last conv layer
    for preferred in ("top_conv", "block7a_project_conv",
                      "block6d_project_conv", "block6a_project_conv"):
        try:
            backbone.get_layer(preferred)
            return preferred
        except ValueError:
            pass

    # Generic reverse scan
    for layer in reversed(backbone.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    return None


def build_gradcam_model(
    model: tf.keras.Model,
) -> tuple[tf.keras.Model | None, str | None]:
    """
    Build a model that outputs (last_conv_activation, final_prediction)
    from the same forward pass — enabling Grad-CAM via GradientTape.

    Returns (gradcam_model, conv_layer_name) or (None, None) on failure.
    """
    backbone = find_backbone(model)
    if backbone is None:
        print("  [WARN] No backbone sub-model found — will use saliency fallback.")
        return None, None

    conv_name = find_last_conv_layer(backbone)
    if conv_name is None:
        print("  [WARN] No Conv2D layer found in backbone — saliency fallback.")
        return None, None

    print(f"  [TARGET] Grad-CAM target layer: '{conv_name}'  "
          f"in backbone '{backbone.name}'")

    # Build backbone model with dual outputs: [conv_out, backbone_final]
    backbone_dual = tf.keras.Model(
        inputs=backbone.inputs,
        outputs=[backbone.get_layer(conv_name).output, backbone.output],
        name="backbone_dual",
    )

    # Rebuild the outer model head on top
    img_input = tf.keras.Input(shape=model.input_shape[1:])
    conv_out, backbone_out = backbone_dual(img_input)

    # Thread backbone_out through remaining head layers
    backbone_pos = next(i for i, l in enumerate(model.layers)
                        if l is backbone)
    x = backbone_out
    for layer in model.layers[backbone_pos + 1:]:
        x = layer(x)   # reuse weights (no new weights created)

    gradcam_model = tf.keras.Model(
        inputs=img_input,
        outputs=[conv_out, x],
        name="gradcam_model",
    )
    return gradcam_model, conv_name


# ????????????????????????????????????????????????????????????????????????????
# SECTION 2 — Grad-CAM computation
# ????????????????????????????????????????????????????????????????????????????
def compute_gradcam(
    gradcam_model: tf.keras.Model,
    img: np.ndarray,   # preprocessed, shape (H, W, 3), float32
) -> np.ndarray:
    """
    Returns Grad-CAM heatmap, shape (H, W), values in [0, 1].
    """
    batch = tf.cast(tf.expand_dims(img, 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = gradcam_model(batch, training=False)
        tape.watch(conv_outputs)
        loss = predictions[:, 0]   # scalar

    grads = tape.gradient(loss, conv_outputs)   # (1, h, w, C)

    if grads is None:
        print("  [WARN] Gradients are None — check model connectivity.")
        return np.zeros(img.shape[:2], dtype=np.float32)

    # ?_c = mean over spatial dims
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2])     # (C,)

    conv_out  = conv_outputs[0]                               # (h, w, C)
    cam       = tf.tensordot(conv_out, pooled_grads, axes=[[2], [0]])  # (h, w)
    cam       = tf.nn.relu(cam)
    cam_max   = tf.reduce_max(cam)
    cam       = cam / (cam_max + 1e-8)

    return cam.numpy().astype(np.float32)


def compute_saliency_fallback(
    model: tf.keras.Model,
    img: np.ndarray,
) -> np.ndarray:
    """Input-gradient saliency as fallback when Grad-CAM cannot be built."""
    batch = tf.cast(tf.expand_dims(img, 0), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(batch)
        pred = model(batch, training=False)
        loss = pred[:, 0]
    grads = tape.gradient(loss, batch)[0]   # (H, W, 3)
    sal   = tf.reduce_max(tf.abs(grads), axis=-1)  # (H, W)
    sal   = sal / (tf.reduce_max(sal) + 1e-8)
    return sal.numpy().astype(np.float32)


def overlay_heatmap(
    original_rgb: np.ndarray,   # (H, W, 3) uint8
    cam: np.ndarray,            # (h, w) float [0,1]
    alpha: float = 0.45,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Resize cam to match original, apply JET colormap, blend with image.
    Returns (H, W, 3) uint8.
    """
    if target_size is None:
        target_size = (original_rgb.shape[1], original_rgb.shape[0])  # (W, H)
    heatmap = cv2.resize(cam, (target_size[0], target_size[1]))
    heatmap_u8 = (heatmap * 255.0).astype(np.uint8)
    heatmap_rgb = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(original_rgb,
                             (target_size[0], target_size[1]))
    blended = (1 - alpha) * img_resized.astype(np.float32) + \
               alpha      * heatmap_rgb.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


# ????????????????????????????????????????????????????????????????????????????
# SECTION 3 — Attention analysis (center vs edge)
# ????????????????????????????????????????????????????????????????????????????
def center_edge_ratio(cam: np.ndarray, center_frac: float = 0.5) -> float:
    """
    Fraction of total activation in the central region vs the edge.
    Ratio > 1 -> model focuses on center (likely the bladder).
    Ratio < 1 -> model focuses on borders/artefacts (bad sign).
    """
    h, w    = cam.shape
    cy, cx  = h // 2, w // 2
    dy, dx  = int(h * center_frac / 2), int(w * center_frac / 2)
    center  = cam[cy-dy:cy+dy, cx-dx:cx+dx].sum()
    total   = cam.sum() + 1e-8
    edge    = total - center
    return float(center / (edge + 1e-8))


# ????????????????????????????????????????????????????????????????????????????
# SECTION 4 — Per-sample save
# ????????????????????????????????????????????????????????????????????????????
def _ax_style(ax, title: str) -> None:
    ax.set_facecolor("#161b22")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=4)
    ax.axis("off")


def save_single_gradcam(
    img_path: str,
    label: int,
    y_prob: float,
    cam: np.ndarray,
    out_path: Path,
) -> None:
    original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    prep_img  = preprocess_bladder_image(img_path, normalize=True)  # float [0,1]
    prep_disp = (prep_img * 255).astype(np.uint8)
    overlay   = overlay_heatmap(original, cam)
    heatmap   = cv2.applyColorMap(
        (cv2.resize(cam, (original.shape[1], original.shape[0])) * 255).astype(np.uint8),
        cv2.COLORMAP_JET)
    heatmap   = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    cls_name = "Abnormal" if label == 1 else "Normal"
    cls_col  = "#ff6b6b" if label == 1 else "#69db7c"

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor(DARK_BG)
    _ax_style(axes[0], "Original"); axes[0].imshow(original)
    _ax_style(axes[1], "Preprocessed (CLAHE)"); axes[1].imshow(prep_disp)
    _ax_style(axes[2], "Grad-CAM Heatmap"); axes[2].imshow(heatmap)
    _ax_style(axes[3], f"Overlay  ->  p={y_prob:.3f}"); axes[3].imshow(overlay)

    ratio = center_edge_ratio(cam)
    fig.suptitle(
        f"True class: {cls_name} | Pred prob: {y_prob:.3f} | "
        f"Center/Edge ratio: {ratio:.2f}  "
        f"({'? focused on bladder' if ratio > 1.0 else '[!] border focus'})",
        color=cls_col, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()


# ????????????????????????????????????????????????????????????????????????????
# SECTION 5 — Summary grid + attention CSV
# ????????????????????????????????????????????????????????????????????????????
def save_attention_grid(
    sample_info: list[dict],
    out_path: Path,
) -> None:
    """8-panel grid: 4 normal + 4 abnormal overlays."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.patch.set_facecolor(DARK_BG)

    for ax, info in zip(axes.flat, sample_info):
        original = cv2.cvtColor(cv2.imread(info["path"]), cv2.COLOR_BGR2RGB)
        overlay  = overlay_heatmap(original, info["cam"], target_size=(224, 224))
        ax.imshow(overlay)
        col   = "#ff6b6b" if info["label"] == 1 else "#69db7c"
        title = (f"{'Abnormal' if info['label'] else 'Normal'}  "
                 f"p={info['y_prob']:.2f}  CE={info['ratio']:.2f}")
        ax.set_title(title, color=col, fontsize=9, fontweight="bold", pad=3)
        ax.axis("off")

    fig.suptitle(
        "Grad-CAM Attention Map — EfficientNetB3 (top row: Normal, bottom: Abnormal)\n"
        "CE ratio = center/edge  (>1 means model focuses on bladder region ?)",
        color="white", fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ? Attention grid -> {out_path}")


# ????????????????????????????????????????????????????????????????????????????
# SECTION 6 — Main
# ????????????????????????????????????????????????????????????????????????????
def main() -> None:
    print("=" * 70)
    print("  Grad-CAM for EfficientNetB3 — Bladder Attention Validation")
    print("=" * 70)

    # ?? 1. Load best model ????????????????????????????????????????????????????
    if not BEST_MODEL.exists():
        print(f"[ERR] Model not found: {BEST_MODEL}")
        print("   Run ultrasound_efficientnet.py first.")
        sys.exit(1)

    print(f"\n[LOAD] Loading model: {BEST_MODEL}")
    model = tf.keras.models.load_model(str(BEST_MODEL), compile=False)
    h, w  = get_model_input_size(model)
    print(f"  Model input size: {h}×{w}")

    # ?? 2. Build Grad-CAM model ???????????????????????????????????????????????
    print("\n[SCAN] Building Grad-CAM model …")
    gradcam_model, conv_name = build_gradcam_model(model)
    use_saliency = (gradcam_model is None)
    method_name  = "Saliency (fallback)" if use_saliency else \
                   f"Grad-CAM [{conv_name}]"
    print(f"  Method: {method_name}")

    # ?? 3. Load test split ????????????????????????????????????????????????????
    if not TEST_SPLIT.exists():
        print(f"[ERR] Test split CSV not found: {TEST_SPLIT}")
        print("   Run ultrasound_efficientnet.py first.")
        sys.exit(1)

    df = pd.read_csv(str(TEST_SPLIT))
    # Accept both with and without 'y_prob' column
    if "y_prob" not in df.columns:
        df["y_prob"] = 0.5
    print(f"\n[LIST] Test split: {len(df)} samples "
          f"| normal={len(df[df.label==0])} | abnormal={len(df[df.label==1])}")

    # ?? 4. Select 4 normal + 4 abnormal ??????????????????????????????????????
    normal_df   = df[df.label == 0].sample(n=min(N_SAMPLES, len(df[df.label==0])),
                                            random_state=SEED)
    abnormal_df = df[df.label == 1].sample(n=min(N_SAMPLES, len(df[df.label==1])),
                                            random_state=SEED)
    selected    = pd.concat([normal_df, abnormal_df], ignore_index=True)
    print(f"  Selected {len(selected)} samples for Grad-CAM visualisation.")

    # ?? 5. Preprocess + Grad-CAM loop ?????????????????????????????????????????
    print(f"\n[CAM] Running {method_name} …")
    sample_info   = []
    attention_rows = []

    for _, row in selected.iterrows():
        img_path  = row["image_path"]
        label     = int(row["label"])
        y_prob    = float(row["y_prob"])

        if not Path(img_path).exists():
            print(f"  [SKIP] File not found: {img_path}")
            continue

        # Preprocess
        img_prep = preprocess_bladder_image(img_path,
                                            target_size=(h, w),
                                            normalize=False)
        img_efn  = tf.keras.applications.efficientnet.preprocess_input(
            img_prep.copy())

        # Compute CAM
        if not use_saliency:
            cam = compute_gradcam(gradcam_model, img_efn)
        else:
            cam = compute_saliency_fallback(model, img_efn)

        ratio = center_edge_ratio(cam)
        cls   = "abnormal" if label == 1 else "normal"
        idx   = sum(1 for s in sample_info if s["label"] == label)

        # Save per-sample figure
        out_path = GRADCAM_DIR / f"{cls}_{idx:02d}.png"
        save_single_gradcam(img_path, label, y_prob, cam, out_path)
        print(f"  [{cls:8s}]  p={y_prob:.3f}  CE_ratio={ratio:.2f}  -> {out_path.name}")

        sample_info.append({"path": img_path, "label": label,
                             "y_prob": y_prob, "cam": cam, "ratio": ratio})
        attention_rows.append({
            "image_path": img_path,
            "label": label,
            "y_prob": y_prob,
            "center_edge_ratio": round(ratio, 4),
            "method": method_name,
            "focus_ok": ratio > 1.0,
        })

    # ?? 6. Summary grid ???????????????????????????????????????????????????????
    if sample_info:
        save_attention_grid(sample_info,
                            GRADCAM_DIR / "attention_grid.png")

    # ?? 7. Attention CSV ??????????????????????????????????????????????????????
    if attention_rows:
        att_df = pd.DataFrame(attention_rows)
        att_path = METRICS_DIR / "gradcam_attention_analysis.csv"
        att_df.to_csv(str(att_path), index=False)
        print(f"  ? Attention analysis CSV -> {att_path}")

        # Summary
        print("\n" + "=" * 70)
        print("  GRAD-CAM ATTENTION ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"  Method : {method_name}")
        print(f"  Samples: {len(attention_rows)} "
              f"({sum(1 for r in attention_rows if r['label']==0)} normal, "
              f"{sum(1 for r in attention_rows if r['label']==1)} abnormal)")
        avg_ratio = np.mean([r["center_edge_ratio"] for r in attention_rows])
        focus_ok  = sum(1 for r in attention_rows if r["focus_ok"])
        print(f"  Mean center/edge ratio : {avg_ratio:.2f}  "
              f"({'[OK] model focuses on bladder' if avg_ratio > 1.0 else '[WARN]  border-focus detected'})")
        print(f"  Samples with CE>1      : {focus_ok}/{len(attention_rows)}")
        for r in attention_rows:
            cls_name = "Abnormal" if r["label"] else "Normal"
            ok = "?" if r["focus_ok"] else "[X]"
            print(f"    {ok}  {cls_name:8s}  p={r['y_prob']:.3f}  "
                  f"CE={r['center_edge_ratio']:.2f}")
        print("=" * 70)


SEED = 42

if __name__ == "__main__":
    main()
