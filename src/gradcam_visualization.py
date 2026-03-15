"""Generate Grad-CAM visualizations for ultrasound model predictions."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def _get_last_conv_layer_name(model: tf.keras.Model) -> str:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    raise ValueError("No Conv2D layer found for Grad-CAM")


def _make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model, last_conv_layer_name: str) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def _load_sample_images(split_dir: str, max_images: int) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for class_name, label in [("normal", 0), ("abnormal", 1)]:
        class_dir = Path(split_dir) / "test" / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.glob("**/*"):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                samples.append((str(image_path), label))
    np.random.shuffle(samples)
    return samples[:max_images]


def run_gradcam(config_path: str = "config/config.yaml") -> None:
    """Generate Grad-CAM overlays for a subset of test images."""
    config = load_config(config_path)
    set_global_seed(int(config["project"]["seed"]))

    model = tf.keras.models.load_model(config["ultrasound"]["model_output"])
    image_size = tuple(config["ultrasound"]["image_size"])
    max_images = int(config["explainability"]["gradcam_examples"])

    split_dir = config["paths"]["ultrasound_split_dir"]
    samples = _load_sample_images(split_dir, max_images)

    graphs_dir = config["paths"]["results_graphs_dir"]
    ensure_dir(graphs_dir)

    last_conv_layer_name = _get_last_conv_layer_name(model)

    for idx, (image_path, label) in enumerate(samples):
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, image_size)
        img_array = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        heatmap = _make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        heatmap_resized = cv2.resize(heatmap, (image_rgb.shape[1], image_rgb.shape[0]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), 0.6, heatmap_color, 0.4, 0)
        output_path = os.path.join(graphs_dir, f"gradcam_{idx}_label_{label}.png")
        cv2.imwrite(output_path, overlay)

    LOGGER.info("Saved Grad-CAM outputs to %s", graphs_dir)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    run_gradcam(args.config)


if __name__ == "__main__":
    main()
