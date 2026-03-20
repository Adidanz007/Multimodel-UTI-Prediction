"""
Improved Ultrasound Image Classification Pipeline (Standalone).

This module provides a standalone, research-grade ultrasound classifier
with proper preprocessing, bladder filtering, DenseNet121 backbone,
and two-stage training strategy.

Use this module to:
1. Train and evaluate the image model independently
2. Verify preprocessing and augmentation work correctly
3. Generate a strong image model for multimodal fusion

Key improvements over the original pipeline:
- Proper ultrasound preprocessing (cropping, CLAHE contrast enhancement)
- Bladder-specific image filtering
- DenseNet121 backbone (better for medical imaging)
- Two-stage training (frozen -> fine-tuning)
- Comprehensive debugging and collapse detection

Author: Multimodal UTI Prediction Project
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
import cv2
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
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import Sequence as KerasSequence

from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class UltrasoundConfig:
    """Configuration for ultrasound image training."""

    # Image settings
    image_size: Tuple[int, int] = (224, 224)
    crop_height: Tuple[float, float] = (0.15, 0.90)
    crop_width: Tuple[float, float] = (0.10, 0.90)

    # Data settings
    test_size: float = 0.15
    val_size: float = 0.15
    filter_bladder_only: bool = True

    # Training settings
    batch_size: int = 16
    stage1_epochs: int = 20
    stage2_epochs: int = 30
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-5

    # Architecture
    backbone: str = "DenseNet121"
    feature_dim: int = 256
    dropout_rate: float = 0.5
    layers_to_unfreeze: int = 20

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    # Seed
    seed: int = 42


# =============================================================================
# ULTRASOUND PREPROCESSING
# =============================================================================


def preprocess_ultrasound(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    crop_height: Tuple[float, float] = (0.15, 0.90),
    crop_width: Tuple[float, float] = (0.10, 0.90),
) -> np.ndarray:
    """
    Preprocess ultrasound image with medical imaging best practices.

    Pipeline:
    1. Load image (OpenCV)
    2. Crop to remove borders/UI/text (15-90% height, 10-90% width)
    3. Resize to target size (224x224)
    4. Convert to grayscale
    5. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    6. Convert to 3-channel RGB (for pretrained CNN)
    7. Normalize to [0, 1]

    Args:
        image_path: Path to ultrasound image
        target_size: Target (width, height)
        crop_height: (start_ratio, end_ratio) for height
        crop_width: (start_ratio, end_ratio) for width

    Returns:
        Preprocessed image as float32 (H, W, 3) in [0, 1]
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    h, w = image.shape[:2]

    # Crop to remove borders and UI
    y1, y2 = int(h * crop_height[0]), int(h * crop_height[1])
    x1, x2 = int(w * crop_width[0]), int(w * crop_width[1])
    cropped = image[y1:y2, x1:x2]

    # Resize
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert to RGB (3 channels for pretrained models)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    return normalized


def preprocess_ultrasound_simple(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Simple preprocessing for already-loaded images (for augmentation pipeline).

    Args:
        image: Image array (from cv2.imread)
        target_size: Target size

    Returns:
        Preprocessed image
    """
    # Crop (15-90% height, 10-90% width)
    h, w = image.shape[:2]
    y1, y2 = int(h * 0.15), int(h * 0.90)
    x1, x2 = int(w * 0.10), int(w * 0.90)
    cropped = image[y1:y2, x1:x2]

    # Resize
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # Grayscale + CLAHE
    if len(resized.shape) == 3:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = resized

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # To RGB
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return rgb


# =============================================================================
# BLADDER FILTERING
# =============================================================================


BLADDER_KEYWORDS = ["bladder", "UB", "ub", "urinary", "vesica"]
EXCLUDE_KEYWORDS = ["liver", "kidney", "ovary", "prostate", "gallbladder", "spleen"]


def is_bladder_related(filename: str) -> bool:
    """
    Check if filename indicates bladder-related image.

    Args:
        filename: Image filename

    Returns:
        True if bladder-related
    """
    filename_lower = filename.lower()

    # Exclude non-bladder organs
    for kw in EXCLUDE_KEYWORDS:
        if kw in filename_lower:
            return False

    # Check for bladder keywords
    for kw in BLADDER_KEYWORDS:
        if kw.lower() in filename_lower:
            return True

    # If no keywords match, assume it's bladder (for datasets without organ labels)
    return True


def load_and_filter_dataset(
    image_dir: str,
    filter_bladder: bool = True,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load and filter ultrasound dataset.

    Expected structure:
        image_dir/
            normal/
                image1.jpg
                ...
            abnormal/
                image1.jpg
                ...

    Args:
        image_dir: Root directory
        filter_bladder: If True, filter to bladder images only

    Returns:
        (image_paths, labels, stats_dict)
    """
    LOGGER.info("=" * 60)
    LOGGER.info("LOADING DATASET")
    LOGGER.info("=" * 60)

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    class_map = {"normal": 0, "abnormal": 1}

    paths, labels = [], []
    stats = {"total": 0, "included": 0, "excluded": 0, "corrupt": 0, "normal": 0, "abnormal": 0}

    for class_name, label in class_map.items():
        class_dir = Path(image_dir) / class_name
        if not class_dir.exists():
            LOGGER.warning("Directory not found: %s", class_dir)
            continue

        for f in class_dir.glob("**/*"):
            if f.suffix.lower() not in valid_ext:
                continue

            stats["total"] += 1

            # Filter bladder-only
            if filter_bladder and not is_bladder_related(f.name):
                stats["excluded"] += 1
                continue

            # Validate readable
            img = cv2.imread(str(f))
            if img is None:
                stats["corrupt"] += 1
                continue

            paths.append(str(f))
            labels.append(label)
            stats["included"] += 1
            stats[class_name] += 1

    if not paths:
        raise RuntimeError(f"No valid images in {image_dir}")

    # Print stats
    LOGGER.info("Total found: %d", stats["total"])
    LOGGER.info("Included (bladder): %d", stats["included"])
    LOGGER.info("Excluded (other organs): %d", stats["excluded"])
    LOGGER.info("Corrupt/unreadable: %d", stats["corrupt"])
    LOGGER.info("-" * 40)
    LOGGER.info("Normal: %d (%.1f%%)", stats["normal"], 100 * stats["normal"] / stats["included"])
    LOGGER.info("Abnormal: %d (%.1f%%)", stats["abnormal"], 100 * stats["abnormal"] / stats["included"])

    # Class imbalance
    if stats["normal"] > 0 and stats["abnormal"] > 0:
        ratio = max(stats["normal"], stats["abnormal"]) / min(stats["normal"], stats["abnormal"])
        LOGGER.info("Imbalance ratio: %.2f:1", ratio)
        if ratio > 3:
            LOGGER.warning("SEVERE IMBALANCE - use class weights!")

    LOGGER.info("=" * 60)

    return paths, labels, stats


# =============================================================================
# DATA AUGMENTATION
# =============================================================================


def get_augmentation_pipeline() -> A.Compose:
    """
    Get Albumentations augmentation pipeline for medical ultrasound.

    Augmentations are conservative to preserve anatomical features:
    - Horizontal flip (bladder is symmetric)
    - Small rotations
    - Brightness/contrast adjustments
    - Slight blur (simulates different machine settings)
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15),
            mode=cv2.BORDER_REFLECT,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


# =============================================================================
# DATA GENERATOR
# =============================================================================


@dataclass
class ImageRecord:
    """Container for image path and label."""
    path: str
    label: int


class UltrasoundDataGenerator(KerasSequence):
    """
    Custom data generator for ultrasound images.

    Features:
    - On-the-fly preprocessing with CLAHE
    - Optional augmentation
    - Efficient batch loading
    """

    def __init__(
        self,
        records: List[ImageRecord],
        image_size: Tuple[int, int],
        batch_size: int,
        augment: bool = False,
        shuffle: bool = True,
    ):
        self.records = list(records)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.augmenter = get_augmentation_pipeline() if augment else None
        self.indices = np.arange(len(self.records))

        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        images, labels = [], []
        for i in batch_idx:
            rec = self.records[i]
            try:
                img = preprocess_ultrasound(rec.path, self.image_size)

                if self.augmenter:
                    # Albumentations expects uint8
                    img_uint8 = (img * 255).astype(np.uint8)
                    aug = self.augmenter(image=img_uint8)
                    img = aug["image"].astype(np.float32) / 255.0

                images.append(img)
                labels.append(rec.label)
            except Exception as e:
                LOGGER.warning("Error loading %s: %s", rec.path, e)

        return np.array(images, dtype=np.float32), np.array(labels, dtype=np.float32)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================


def build_image_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    backbone: str = "DenseNet121",
    feature_dim: int = 256,
    dropout_rate: float = 0.5,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Build ultrasound classification model with transfer learning.

    Architecture:
        Input (224x224x3)
        -> DenseNet121/EfficientNetB0 (pretrained ImageNet)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dense(256, relu)
        -> Dropout(0.5)
        -> Dense(1, sigmoid)

    Args:
        input_shape: Input shape
        backbone: Backbone name (DenseNet121, EfficientNetB0, ResNet50)
        feature_dim: Dense layer units
        dropout_rate: Dropout rate
        freeze_base: Freeze backbone initially

    Returns:
        Keras Model
    """
    LOGGER.info("Building model with %s backbone...", backbone)

    # Select backbone
    backbone_map = {
        "DenseNet121": tf.keras.applications.DenseNet121,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
    }

    if backbone not in backbone_map:
        raise ValueError(f"Unknown backbone: {backbone}")

    base = backbone_map[backbone](
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )
    base.trainable = not freeze_base

    # Build model
    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.BatchNormalization(name="bn")(x)
    x = layers.Dense(feature_dim, activation="relu", name="features")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = models.Model(inputs, outputs, name=f"ultrasound_{backbone.lower()}")

    LOGGER.info("  Backbone: %s (frozen=%s)", backbone, freeze_base)
    LOGGER.info("  Feature dim: %d", feature_dim)
    LOGGER.info("  Total params: %d", model.count_params())

    return model


def unfreeze_layers(model: tf.keras.Model, num_layers: int = 20) -> None:
    """
    Unfreeze last N layers of backbone for fine-tuning.

    BatchNorm layers are kept frozen for stability.

    Args:
        model: Keras model
        num_layers: Layers to unfreeze from end
    """
    # Find backbone (second layer after input)
    base = model.layers[1]
    total = len(base.layers)
    unfreeze_from = max(0, total - num_layers)

    LOGGER.info("Unfreezing last %d layers (from %d/%d)", num_layers, unfreeze_from, total)

    base.trainable = True
    for i, layer in enumerate(base.layers):
        if i < unfreeze_from:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False  # Keep BN frozen
        else:
            layer.trainable = True

    trainable = sum(1 for l in base.layers if l.trainable)
    LOGGER.info("Trainable layers: %d/%d", trainable, total)


# =============================================================================
# TRAINING
# =============================================================================


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    LOGGER.info("Class weights: %s", weight_dict)
    return weight_dict


def train_stage1(
    model: tf.keras.Model,
    train_gen: UltrasoundDataGenerator,
    val_gen: UltrasoundDataGenerator,
    class_weights: Dict[int, float],
    epochs: int = 20,
    lr: float = 1e-4,
    save_dir: str = "models",
) -> Dict[str, List[float]]:
    """
    Stage 1: Train top layers with frozen backbone.

    Args:
        model: Model with frozen backbone
        train_gen: Training generator
        val_gen: Validation generator
        class_weights: Class weight dict
        epochs: Number of epochs
        lr: Learning rate
        save_dir: Directory for checkpoints

    Returns:
        Training history dict
    """
    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 1: Training top layers (backbone frozen)")
    LOGGER.info("=" * 60)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=5,
            mode="max", min_lr=1e-7, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(save_dir, "us_stage1_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )

    LOGGER.info("Stage 1 best val_auc: %.4f", max(history.history["val_auc"]))
    return history.history


def train_stage2(
    model: tf.keras.Model,
    train_gen: UltrasoundDataGenerator,
    val_gen: UltrasoundDataGenerator,
    class_weights: Dict[int, float],
    epochs: int = 30,
    lr: float = 1e-5,
    layers_to_unfreeze: int = 20,
    save_dir: str = "models",
) -> Dict[str, List[float]]:
    """
    Stage 2: Fine-tune last N layers of backbone.

    Args:
        model: Model from stage 1
        train_gen: Training generator
        val_gen: Validation generator
        class_weights: Class weight dict
        epochs: Number of epochs
        lr: Learning rate (lower than stage 1)
        layers_to_unfreeze: Number of backbone layers to unfreeze
        save_dir: Directory for checkpoints

    Returns:
        Training history dict
    """
    LOGGER.info("=" * 60)
    LOGGER.info("STAGE 2: Fine-tuning last %d layers", layers_to_unfreeze)
    LOGGER.info("=" * 60)

    unfreeze_layers(model, layers_to_unfreeze)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    cbs = [
        callbacks.EarlyStopping(
            monitor="val_auc", patience=10, mode="max",
            restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc", factor=0.5, patience=5,
            mode="max", min_lr=1e-8, verbose=1
        ),
        callbacks.ModelCheckpoint(
            os.path.join(save_dir, "us_stage2_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=cbs,
        verbose=1,
    )

    LOGGER.info("Stage 2 best val_auc: %.4f", max(history.history["val_auc"]))
    return history.history


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_model(
    model: tf.keras.Model,
    data_gen: UltrasoundDataGenerator,
    split_name: str = "test",
    save_dir: str = "results",
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with debugging checks.

    Generates:
    - ROC curve
    - Confusion matrix
    - Prediction distribution
    - Classification report

    Also checks for model collapse (predicting only one class).

    Args:
        model: Trained model
        data_gen: Data generator
        split_name: Name for plots (test, validation)
        save_dir: Directory to save results

    Returns:
        Metrics dict
    """
    ensure_dir(save_dir)

    LOGGER.info("=" * 60)
    LOGGER.info("EVALUATING ON %s SET", split_name.upper())
    LOGGER.info("=" * 60)

    # Collect predictions
    y_true, y_prob = [], []
    for batch_x, batch_y in data_gen:
        pred = model.predict(batch_x, verbose=0).ravel()
        y_true.extend(batch_y.tolist())
        y_prob.extend(pred.tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    LOGGER.info("Metrics:")
    for name, val in metrics.items():
        LOGGER.info("  %s: %.4f", name, val)

    # Debugging checks
    LOGGER.info("-" * 40)
    LOGGER.info("DEBUGGING:")
    LOGGER.info("  Mean pred: %.4f (expect ~0.3-0.7)", y_prob.mean())
    LOGGER.info("  Std pred:  %.4f (expect >0.1)", y_prob.std())
    LOGGER.info("  Min pred:  %.4f", y_prob.min())
    LOGGER.info("  Max pred:  %.4f", y_prob.max())
    LOGGER.info("  Unique classes predicted: %d", len(np.unique(y_pred)))

    if len(np.unique(y_pred)) == 1:
        LOGGER.error("MODEL COLLAPSED - only predicting one class!")
    elif y_prob.std() < 0.05:
        LOGGER.warning("Low variance - model may not be learning well")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {split_name}", fontweight="bold")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(save_dir, f"us_roc_{split_name}.png"), dpi=300)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Abnormal"],
                yticklabels=["Normal", "Abnormal"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {split_name}", fontweight="bold")
    plt.savefig(os.path.join(save_dir, f"us_cm_{split_name}.png"), dpi=300)
    plt.close()

    # Prediction distribution
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label="Normal", color="blue")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label="Abnormal", color="red")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title("By True Class")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(y_prob, bins=50, color="green", alpha=0.7)
    plt.axvline(0.5, color="red", ls="--", label="Threshold")
    plt.xlabel("Probability")
    plt.ylabel("Count")
    plt.title("Overall Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"us_dist_{split_name}.png"), dpi=300)
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"])
    with open(os.path.join(save_dir, f"us_report_{split_name}.txt"), "w") as f:
        f.write(report)

    LOGGER.info("=" * 60)
    return metrics


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================


def train_ultrasound_pipeline(
    image_dir: str,
    output_dir: str = "models",
    results_dir: str = "results",
    config: Optional[UltrasoundConfig] = None,
) -> Dict[str, Any]:
    """
    Complete training pipeline for ultrasound classifier.

    Steps:
    1. Load and filter to bladder images
    2. Split into train/val/test
    3. Build DenseNet121 model
    4. Stage 1: Train with frozen backbone
    5. Stage 2: Fine-tune last layers
    6. Evaluate on test set
    7. Save model and results

    Args:
        image_dir: Directory with normal/abnormal subdirs
        output_dir: Directory for model files
        results_dir: Directory for plots/metrics
        config: Training configuration

    Returns:
        Dict with model path and test metrics
    """
    if config is None:
        config = UltrasoundConfig()

    set_global_seed(config.seed)
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    LOGGER.info("=" * 70)
    LOGGER.info("ULTRASOUND IMAGE CLASSIFICATION PIPELINE")
    LOGGER.info("=" * 70)

    # 1. Load data
    LOGGER.info("\n[1] Loading and filtering dataset...")
    paths, labels, stats = load_and_filter_dataset(
        image_dir,
        filter_bladder=config.filter_bladder_only,
    )

    # Create records
    records = [ImageRecord(p, l) for p, l in zip(paths, labels)]
    labels_arr = np.array(labels)

    # 2. Split
    LOGGER.info("\n[2] Creating train/val/test splits...")
    train_val, test_rec = train_test_split(
        records, test_size=config.test_size,
        stratify=labels_arr, random_state=config.seed
    )
    train_val_labels = np.array([r.label for r in train_val])

    val_ratio = config.val_size / (1 - config.test_size)
    train_rec, val_rec = train_test_split(
        train_val, test_size=val_ratio,
        stratify=train_val_labels, random_state=config.seed
    )

    LOGGER.info("Train: %d, Val: %d, Test: %d",
                len(train_rec), len(val_rec), len(test_rec))

    # Verify alignment
    LOGGER.info("\nSample verification:")
    for i, rec in enumerate(train_rec[:3]):
        LOGGER.info("  Train[%d]: %s -> label %d", i, Path(rec.path).name, rec.label)

    # 3. Class weights
    LOGGER.info("\n[3] Computing class weights...")
    train_labels = np.array([r.label for r in train_rec])
    class_weights = compute_class_weights(train_labels)

    # 4. Generators
    LOGGER.info("\n[4] Creating data generators...")
    train_gen = UltrasoundDataGenerator(
        train_rec, config.image_size, config.batch_size,
        augment=True, shuffle=True
    )
    val_gen = UltrasoundDataGenerator(
        val_rec, config.image_size, config.batch_size,
        augment=False, shuffle=False
    )
    test_gen = UltrasoundDataGenerator(
        test_rec, config.image_size, config.batch_size,
        augment=False, shuffle=False
    )

    # 5. Build model
    LOGGER.info("\n[5] Building model...")
    model = build_image_model(
        input_shape=(*config.image_size, 3),
        backbone=config.backbone,
        feature_dim=config.feature_dim,
        dropout_rate=config.dropout_rate,
        freeze_base=True,
    )

    # 6. Stage 1
    LOGGER.info("\n[6] Stage 1 training...")
    hist1 = train_stage1(
        model, train_gen, val_gen, class_weights,
        epochs=config.stage1_epochs,
        lr=config.stage1_lr,
        save_dir=output_dir,
    )

    # 7. Stage 2
    LOGGER.info("\n[7] Stage 2 training...")
    hist2 = train_stage2(
        model, train_gen, val_gen, class_weights,
        epochs=config.stage2_epochs,
        lr=config.stage2_lr,
        layers_to_unfreeze=config.layers_to_unfreeze,
        save_dir=output_dir,
    )

    # 8. Evaluate
    LOGGER.info("\n[8] Evaluating...")
    val_metrics = evaluate_model(model, val_gen, "validation", results_dir)
    test_metrics = evaluate_model(model, test_gen, "test", results_dir)

    # 9. Save model
    LOGGER.info("\n[9] Saving model...")
    model_path = os.path.join(output_dir, "ultrasound_model_v2.keras")
    model.save(model_path)
    LOGGER.info("Model saved: %s", model_path)

    # Save history
    combined = {
        k: hist1[k] + hist2[k]
        for k in ["loss", "val_loss", "accuracy", "val_accuracy", "auc", "val_auc"]
    }
    pd.DataFrame(combined).to_csv(
        os.path.join(results_dir, "us_training_history.csv"), index=False
    )

    # Plot history
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (train_key, val_key, title) in zip(axes, [
        ("loss", "val_loss", "Loss"),
        ("accuracy", "val_accuracy", "Accuracy"),
        ("auc", "val_auc", "AUC"),
    ]):
        ax.plot(combined[train_key], label="Train")
        ax.plot(combined[val_key], label="Val")
        ax.set_xlabel("Epoch")
        ax.set_title(title, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "us_history.png"), dpi=300)
    plt.close()

    # Summary
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TRAINING COMPLETED")
    LOGGER.info("=" * 70)
    LOGGER.info("Test ROC-AUC: %.4f", test_metrics["roc_auc"])
    LOGGER.info("Test Accuracy: %.4f", test_metrics["accuracy"])
    LOGGER.info("Test F1: %.4f", test_metrics["f1"])
    LOGGER.info("Model: %s", model_path)
    LOGGER.info("=" * 70)

    return {
        "model_path": model_path,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "history": combined,
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train improved ultrasound image classifier"
    )
    parser.add_argument(
        "--image-dir",
        default="data/raw/ultrasound_images",
        help="Directory with normal/abnormal subdirs"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for model"
    )
    parser.add_argument(
        "--results-dir",
        default="results/ultrasound_v2",
        help="Directory for plots and metrics"
    )
    parser.add_argument(
        "--backbone",
        default="DenseNet121",
        choices=["DenseNet121", "EfficientNetB0", "ResNet50", "MobileNetV2"],
        help="Backbone architecture"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()
    setup_logging()

    config = UltrasoundConfig(
        backbone=args.backbone,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    results = train_ultrasound_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
        config=config,
    )

    LOGGER.info("\nDone! Test AUC: %.4f", results["test_metrics"]["roc_auc"])


if __name__ == "__main__":
    main()
