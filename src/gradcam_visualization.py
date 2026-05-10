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


def _find_backbone(model: tf.keras.Model) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    raise ValueError("No nested backbone model found for Grad-CAM")


def _find_last_conv_in_model(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found for Grad-CAM")


def _build_classifier_head(model: tf.keras.Model, backbone: tf.keras.Model) -> tf.keras.Model:
    backbone_idx = model.layers.index(backbone)
    head_layers = model.layers[backbone_idx + 1 :]
    if not head_layers:
        raise ValueError("No classifier head layers found after backbone")

    classifier_input = tf.keras.Input(shape=tuple(backbone.output.shape[1:]))
    x = classifier_input
    for layer in head_layers:
        x = layer(x)
    return tf.keras.Model(classifier_input, x)


def _make_gradcam_heatmap(img_array: np.ndarray, model: tf.keras.Model) -> tuple[np.ndarray, str]:
    backbone = _find_backbone(model)
    last_conv_layer = _find_last_conv_in_model(backbone)
    classifier_head = _build_classifier_head(model, backbone)

    backbone_model = tf.keras.Model(
        backbone.input,
        [last_conv_layer.output, backbone.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = backbone_model(img_array)
        predictions = classifier_head(backbone_output)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if tf.equal(max_val, 0):
        return np.zeros_like(heatmap.numpy()), f"{backbone.name}/{last_conv_layer.name}"
    heatmap = heatmap / max_val
    return heatmap.numpy(), f"{backbone.name}/{last_conv_layer.name}"


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

    for idx, (image_path, label) in enumerate(samples):
        image = cv2.imread(image_path)
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, image_size)
        img_array = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        heatmap, conv_name = _make_gradcam_heatmap(img_array, model)
        if idx == 0:
            LOGGER.info("Using Grad-CAM conv layer: %s", conv_name)
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
