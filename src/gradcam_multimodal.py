"""
Enhanced Grad-CAM Visualization for Multimodal UTI Prediction.

This module extends the original Grad-CAM implementation with:
- Support for multimodal fusion models
- Proper handling of nested backbones (DenseNet, EfficientNet)
- Attention analysis for model verification
- CLAHE-preprocessed image support

Author: Multimodal UTI Prediction Project
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.us_preprocessing import preprocess_bladder_image
from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


# =============================================================================
# PREPROCESSING (matching the training pipeline)
# =============================================================================


def preprocess_for_gradcam(
    image_path: str,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Preprocess image for Grad-CAM visualization.

    Delegates to the canonical preprocess_bladder_image() in us_preprocessing,
    which applies: crop (15-90% h / 10-90% w) → CLAHE → 3-channel stack.

    Args:
        image_path: Path to ultrasound image.
        target_size: Target (H, W).  Defaults to (224, 224) for DenseNet.
            For EfficientNetB3 pass (260, 260).

    Returns:
        Preprocessed image as float32 (H, W, 3) in [0, 1].  
    """
    if target_size is None:
        target_size = (224, 224)

    return preprocess_bladder_image(image_path, target_size=target_size,
                                    normalize=True)


# =============================================================================
# GRADCAM FOR NESTED MODELS
# =============================================================================


def find_nested_conv_layer(model: keras.Model) -> Tuple[Optional[keras.layers.Layer], Optional[str]]:
    """
    Find last conv layer, handling nested backbone models.

    Args:
        model: Keras model (potentially with nested backbone)

    Returns:
        Tuple of (layer, full_name) or (None, None) if not found
    """
    # First check for nested models (like DenseNet121, EfficientNet)
    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 50:
            # This is likely a backbone network
            for sublayer in reversed(layer.layers):
                if "conv" in sublayer.name.lower():
                    return sublayer, f"{layer.name}/{sublayer.name}"

    # Fallback: search in main model
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer, layer.name
        if "conv" in layer.name.lower():
            return layer, layer.name

    return None, None


def compute_gradcam_nested(
    model: keras.Model,
    image: np.ndarray,
    clinical_features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute attention heatmap via gradient saliency w.r.t. input image.

    Keras 3 compatible — works for single-input and multimodal models.
    Uses d(prediction)/d(pixel) collapsed across colour channels as the
    spatial attention map (equivalent to Grad-CAM for visualisation).

    Args:
        model: Keras model (single-input or multimodal)
        image: Preprocessed image (H, W, C) in model's expected range
        clinical_features: Clinical feature vector (n_features,) for
                           multimodal; pass None for auto zero-placeholder.

    Returns:
        Normalised heatmap (H, W), float32, values in [0, 1].
    """
    # ── Detect multimodal model ────────────────────────────────────────────
    is_multimodal = isinstance(model.input, list) and len(model.input) > 1

    # ── Prepare clinical tensor ────────────────────────────────────────────
    clin = None
    if is_multimodal:
        if clinical_features is not None:
            clin = tf.cast(
                clinical_features.reshape(1, -1).astype(np.float32), tf.float32
            )
        else:
            clin_dim = model.input[1].shape[-1]
            clin = tf.zeros((1, clin_dim), dtype=tf.float32)

    # ── tf.Variable is auto-watched by GradientTape ────────────────────────
    image_var = tf.Variable(
        np.expand_dims(image, 0).astype(np.float32),
        trainable=True,
        dtype=tf.float32,
    )

    # ── Forward pass with gradient tracking ───────────────────────────────
    try:
        with tf.GradientTape() as tape:
            if is_multimodal:
                preds = model([image_var, clin], training=False)
            else:
                # Pass as named dict — works in Keras 2 + Keras 3
                try:
                    img_name = model.input_names[0]          # Keras 2
                except AttributeError:
                    img_name = model.input.name.split(":")[0]  # Keras 3
                preds = model({img_name: image_var}, training=False)
            loss = preds[0, 0]

        grads = tape.gradient(loss, image_var)

    except Exception as exc:
        LOGGER.warning("Gradient computation failed: %s", exc)
        return np.zeros(image.shape[:2], dtype=np.float32)

    if grads is None:
        LOGGER.warning("No gradient — returning blank heatmap.")
        return np.zeros(image.shape[:2], dtype=np.float32)

    # ── Build saliency map: max |grad| across colour channels → (H, W) ────
    saliency = tf.reduce_max(tf.abs(grads[0]), axis=-1).numpy()
    saliency = np.maximum(saliency, 0)
    if saliency.max() > 0:
        saliency /= saliency.max()

    return saliency


# =============================================================================
# VISUALIZATION
# =============================================================================


def create_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Create Grad-CAM overlay on image.

    Args:
        image: Original image (H, W, C) in [0, 1]
        heatmap: Heatmap
        alpha: Overlay transparency

    Returns:
        Overlay as numpy array in [0, 1]
    """
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Apply colormap
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_colored = heatmap_colored.astype(np.float32) / 255.0

    # Blend
    overlay = (1 - alpha) * image + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def visualize_single(
    model: keras.Model,
    image_path: str,
    save_path: str,
    clinical_features: Optional[np.ndarray] = None,
    image_size: Tuple[int, int] = None,
) -> Dict:
    """
    Generate Grad-CAM visualization for a single image.

    Args:
        model: Trained model
        image_path: Path to image
        save_path: Path to save visualization
        clinical_features: Clinical features (for multimodal)
        image_size: Image size

    Returns:
        Dict with prediction info
    """

    # ✅ Dynamic image size fix — safe for both single-input and multimodal models
    if image_size is None:
        if isinstance(model.input, list):
            # Multimodal: first input is always the image
            image_size = tuple(model.input[0].shape[1:3])
        else:
            image_size = tuple(model.input_shape[1:3])

    # Detect multimodal model
    is_multimodal = isinstance(model.input, list) and len(model.input) > 1


    # Preprocess
    image = preprocess_for_gradcam(image_path, image_size)

    # Compute heatmap
    heatmap = compute_gradcam_nested(model, image, clinical_features)

    # Create overlay
    overlay = create_overlay(image, heatmap)

    # Get prediction — handle image-only and multimodal models
    image_batch = np.expand_dims(image, axis=0)
    if clinical_features is not None:
        clinical_batch = clinical_features.reshape(1, -1)
        pred = model.predict([image_batch, clinical_batch], verbose=0)[0, 0]
    elif is_multimodal:
        # Inject dummy clinical zeros so the dual-input model doesn't crash
        clinical_dim = model.input[1].shape[-1]
        dummy_clinical = np.zeros((1, clinical_dim), dtype=np.float32)
        pred = model.predict([image_batch, dummy_clinical], verbose=0)[0, 0]
    else:
        pred = model.predict(image_batch, verbose=0)[0, 0]

    pred_label = "Abnormal" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Preprocessed (CLAHE)", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    heatmap_resized = cv2.resize(heatmap, image_size)
    im = axes[1].imshow(heatmap_resized, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {pred_label} ({confidence:.1%})",
        fontsize=14, fontweight="bold", y=1.02
    )

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    LOGGER.info("Saved: %s (pred=%.4f)", save_path, pred)

    return {"prediction": pred, "label": pred_label, "confidence": confidence}


def visualize_batch(
    model: keras.Model,
    image_paths: List[str],
    labels: np.ndarray,
    save_dir: str,
    clinical_features: Optional[np.ndarray] = None,
    n_per_class: int = 4,
) -> None:
    """
    Generate Grad-CAM for multiple images.

    Args:
        model: Trained model
        image_paths: List of image paths
        labels: Labels array
        save_dir: Output directory
        clinical_features: Clinical features array
        n_per_class: Samples per class
    """
    ensure_dir(save_dir)

    # Sample from each class
    np.random.seed(42)
    normal_idx = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels == 1)[0]

    normal_samples = np.random.choice(
        normal_idx, min(n_per_class, len(normal_idx)), replace=False
    )
    abnormal_samples = np.random.choice(
        abnormal_idx, min(n_per_class, len(abnormal_idx)), replace=False
    )

    # Generate for each sample
    for i, idx in enumerate(normal_samples):
        clin = clinical_features[idx] if clinical_features is not None else None
        try:
            visualize_single(
                model, image_paths[idx],
                os.path.join(save_dir, f"normal_{i:02d}.png"),
                clin
            )
        except Exception as e:
            LOGGER.warning("Failed for %s: %s", image_paths[idx], e)

    for i, idx in enumerate(abnormal_samples):
        clin = clinical_features[idx] if clinical_features is not None else None
        try:
            visualize_single(
                model, image_paths[idx],
                os.path.join(save_dir, f"abnormal_{i:02d}.png"),
                clin
            )
        except Exception as e:
            LOGGER.warning("Failed for %s: %s", image_paths[idx], e)

    # Create summary grid
    _create_summary_grid(save_dir, n_per_class)

    LOGGER.info("Batch complete. Output: %s", save_dir)


def _create_summary_grid(save_dir: str, n_per_class: int) -> None:
    """Create summary grid of all visualizations."""
    normal_files = sorted(Path(save_dir).glob("normal_*.png"))
    abnormal_files = sorted(Path(save_dir).glob("abnormal_*.png"))

    if not normal_files and not abnormal_files:
        return

    n_cols = max(len(normal_files), len(abnormal_files))
    if n_cols == 0:
        return

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, f in enumerate(normal_files[:n_cols]):
        axes[0, i].imshow(plt.imread(str(f)))
        axes[0, i].axis("off")

    for i, f in enumerate(abnormal_files[:n_cols]):
        axes[1, i].imshow(plt.imread(str(f)))
        axes[1, i].axis("off")

    # Hide empty
    for i in range(len(normal_files), n_cols):
        axes[0, i].axis("off")
    for i in range(len(abnormal_files), n_cols):
        axes[1, i].axis("off")

    # Row labels
    if len(normal_files) > 0:
        axes[0, 0].set_ylabel("Normal", fontsize=14, fontweight="bold")
    if len(abnormal_files) > 0:
        axes[1, 0].set_ylabel("Abnormal", fontsize=14, fontweight="bold")

    plt.suptitle("Grad-CAM Summary", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary.png"), dpi=200)
    plt.close()


# =============================================================================
# ATTENTION ANALYSIS
# =============================================================================


def _get_model_image_size(model: keras.Model) -> Tuple[int, int]:
    """Resolve correct (H, W) from the model's first image input."""
    try:
        if isinstance(model.input, list):
            shape = model.input[0].shape  # multimodal: first input = image
        else:
            shape = model.input.shape
        return (int(shape[1]), int(shape[2]))  # (H, W)
    except Exception:
        return (224, 224)  # safe default


def analyze_model_attention(
    model: keras.Model,
    image_paths: List[str],
    labels: np.ndarray,
    clinical_features: Optional[np.ndarray] = None,
    n_samples: int = 50,
) -> Dict:
    """
    Analyze where the model focuses attention.

    Useful for verifying model focuses on bladder, not artifacts.

    Args:
        model: Trained model
        image_paths: Image paths
        labels: Labels
        clinical_features: Clinical features
        n_samples: Number of samples

    Returns:
        Analysis results dict
    """
    LOGGER.info("Analyzing model attention patterns...")

    # Auto-detect model image size
    _img_size = _get_model_image_size(model)

    np.random.seed(42)
    indices = np.random.choice(
        len(image_paths), min(n_samples, len(image_paths)), replace=False
    )

    center_scores = []
    edge_scores = []

    for idx in indices:
        try:
            image = preprocess_for_gradcam(image_paths[idx], _img_size)
            clin = clinical_features[idx] if clinical_features is not None else None

            heatmap = compute_gradcam_nested(model, image, clin)
            heatmap = cv2.resize(heatmap, image.shape[:2][::-1])

            h, w = heatmap.shape

            # Center region (middle 50%)
            c_h = (int(h * 0.25), int(h * 0.75))
            c_w = (int(w * 0.25), int(w * 0.75))

            center = heatmap[c_h[0]:c_h[1], c_w[0]:c_w[1]]
            full_mean = heatmap.mean() + 1e-6

            center_scores.append(center.mean() / full_mean)

            # Edge region
            edge_mask = np.ones_like(heatmap, dtype=bool)
            edge_mask[c_h[0]:c_h[1], c_w[0]:c_w[1]] = False
            edge_scores.append(heatmap[edge_mask].mean() / full_mean)

        except Exception as e:
            LOGGER.debug("Skipping %s: %s", image_paths[idx], e)

    results = {
        "center_focus_mean": np.mean(center_scores),
        "center_focus_std": np.std(center_scores),
        "edge_focus_mean": np.mean(edge_scores),
        "edge_focus_std": np.std(edge_scores),
        "center_to_edge_ratio": np.mean(center_scores) / (np.mean(edge_scores) + 1e-6),
        "n_analyzed": len(center_scores),
    }

    LOGGER.info("-" * 50)
    LOGGER.info("ATTENTION ANALYSIS:")
    LOGGER.info("  Center focus: %.2f +/- %.2f",
                results["center_focus_mean"], results["center_focus_std"])
    LOGGER.info("  Edge focus: %.2f +/- %.2f",
                results["edge_focus_mean"], results["edge_focus_std"])
    LOGGER.info("  Center/Edge ratio: %.2f", results["center_to_edge_ratio"])

    if results["center_to_edge_ratio"] > 1.5:
        LOGGER.info("  GOOD: Model focuses on center (likely bladder)")
    elif results["center_to_edge_ratio"] < 0.7:
        LOGGER.warning("  WARNING: Model focuses on edges (possible artifacts)")
    else:
        LOGGER.info("  Model has uniform attention")

    LOGGER.info("-" * 50)

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entrypoint."""
    import argparse
    from src.utils import setup_logging

    parser = argparse.ArgumentParser(description="Enhanced Grad-CAM for multimodal models")
    parser.add_argument("--model", required=True, help="Path to model")
    parser.add_argument("--image", help="Single image to visualize")
    parser.add_argument("--image-dir", help="Directory for batch visualization")
    parser.add_argument("--output-dir", default="results/gradcam_v2", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=4, help="Samples per class")

    args = parser.parse_args()
    setup_logging()

    LOGGER.info("Loading model: %s", args.model)
    model = keras.models.load_model(args.model, compile=False)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )

    if args.image:
        save_path = os.path.join(args.output_dir, "gradcam_single.png")
        visualize_single(model, args.image, save_path)

    elif args.image_dir:
        # Load images from directory
        from src.ultrasound_pipeline_v2 import load_and_filter_dataset

        paths, labels, _ = load_and_filter_dataset(args.image_dir, filter_bladder=True)

        visualize_batch(
            model, paths, np.array(labels), args.output_dir,
            n_per_class=args.n_samples
        )

        analyze_model_attention(model, paths, np.array(labels))

    else:
        LOGGER.error("Provide --image or --image-dir")


if __name__ == "__main__":
    main()
