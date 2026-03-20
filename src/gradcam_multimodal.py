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

from src.utils import ensure_dir

LOGGER = logging.getLogger(__name__)


# =============================================================================
# PREPROCESSING (matching the training pipeline)
# =============================================================================


def preprocess_for_gradcam(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Preprocess image for Grad-CAM visualization.

    Uses the same preprocessing as training:
    - Crop to remove borders/UI (15-90% height, 10-90% width)
    - CLAHE contrast enhancement
    - Normalize to [0, 1]

    Args:
        image_path: Path to ultrasound image
        target_size: Target (width, height)

    Returns:
        Preprocessed image as float32 (H, W, 3)
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = image.shape[:2]

    # Crop
    y1, y2 = int(h * 0.15), int(h * 0.90)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    cropped = image[y1:y2, x1:x2]

    # Resize
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # Grayscale + CLAHE
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # To RGB
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Normalize
    normalized = rgb.astype(np.float32) / 255.0

    return normalized


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
    Compute Grad-CAM for models with nested backbones.

    Handles both standalone image models and multimodal fusion models.

    Args:
        model: Keras model
        image: Preprocessed image (H, W, C) in [0, 1]
        clinical_features: Clinical features for multimodal (n_features,)

    Returns:
        Heatmap as numpy array
    """
    # Find the backbone model and target layer
    backbone = None
    target_layer = None

    for layer in model.layers:
        if isinstance(layer, keras.Model) and len(layer.layers) > 50:
            backbone = layer
            # Find last conv in backbone
            for sublayer in reversed(backbone.layers):
                if "conv" in sublayer.name.lower():
                    target_layer = sublayer
                    break
            break

    if target_layer is None:
        LOGGER.warning("Could not find conv layer")
        return np.zeros((7, 7))

    # Build gradient model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    # Prepare inputs
    image_batch = np.expand_dims(image, axis=0)

    # Compute gradients
    with tf.GradientTape() as tape:
        if clinical_features is not None:
            clinical_batch = clinical_features.reshape(1, -1)
            conv_output, predictions = grad_model([image_batch, clinical_batch])
        else:
            conv_output, predictions = grad_model(image_batch)

        loss = predictions[0, 0]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        return np.zeros((7, 7))

    # Pool gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight conv output
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(len(pooled_grads)):
        conv_output[:, :, i] *= pooled_grads[i]

    # Average and normalize
    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


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
    image_size: Tuple[int, int] = (224, 224),
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
    # Preprocess
    image = preprocess_for_gradcam(image_path, image_size)

    # Compute heatmap
    heatmap = compute_gradcam_nested(model, image, clinical_features)

    # Create overlay
    overlay = create_overlay(image, heatmap)

    # Get prediction
    image_batch = np.expand_dims(image, axis=0)
    if clinical_features is not None:
        clinical_batch = clinical_features.reshape(1, -1)
        pred = model.predict([image_batch, clinical_batch], verbose=0)[0, 0]
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

    np.random.seed(42)
    indices = np.random.choice(
        len(image_paths), min(n_samples, len(image_paths)), replace=False
    )

    center_scores = []
    edge_scores = []

    for idx in indices:
        try:
            image = preprocess_for_gradcam(image_paths[idx])
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
    model = keras.models.load_model(args.model)

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
