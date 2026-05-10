"""
Multimodal Deep Fusion Pipeline for UTI Prediction.

This module combines clinical tabular data with bladder ultrasound images
using a deep learning feature fusion architecture.

Key improvements:
1. Proper ultrasound preprocessing (cropping, histogram equalization)
2. Bladder-specific image filtering
3. DenseNet121 backbone with transfer learning
4. Feature-level deep fusion (not just probability stacking)
5. Two-stage training strategy
6. Comprehensive debugging and validation

Author: Multimodal UTI Prediction Project
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import joblib
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
from tensorflow.keras import Model, callbacks, layers, models, optimizers
from tensorflow.keras.utils import Sequence as KerasSequence

from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================


@dataclass
class MultimodalRecord:
    """Container for aligned image and clinical data."""

    image_path: str
    clinical_features: np.ndarray
    label: int
    patient_id: Optional[str] = None


@dataclass
class TrainingConfig:
    """Training configuration parameters."""

    # Image settings
    image_size: Tuple[int, int] = (224, 224)

    # Training settings
    batch_size: int = 16
    stage1_epochs: int = 20
    stage2_epochs: int = 30
    stage1_lr: float = 1e-4
    stage2_lr: float = 1e-5

    # Architecture settings
    image_feature_dim: int = 256
    clinical_feature_dim: int = 64
    fusion_hidden_dim: int = 128
    dropout_rate: float = 0.5

    # Fine-tuning
    layers_to_unfreeze: int = 20

    # Callbacks
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5


# =============================================================================
# ULTRASOUND PREPROCESSING (CRITICAL FOR MODEL PERFORMANCE)
# =============================================================================


def preprocess_ultrasound(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
    crop_height: Tuple[float, float] = (0.15, 0.90),
    crop_width: Tuple[float, float] = (0.10, 0.90),
) -> np.ndarray:
    """
    Preprocess ultrasound image with proper medical imaging pipeline.

    This function implements the following critical preprocessing steps:
    1. Load image using OpenCV
    2. Crop to remove black borders, machine UI, text overlays
    3. Resize to target size
    4. Convert to grayscale
    5. Apply histogram equalization (contrast enhancement)
    6. Convert back to 3-channel RGB (required for pretrained CNNs)
    7. Normalize to [0, 1]

    Args:
        image_path: Path to ultrasound image file
        target_size: Target (width, height) for resizing
        crop_height: Tuple (start_ratio, end_ratio) for height cropping
        crop_width: Tuple (start_ratio, end_ratio) for width cropping

    Returns:
        Preprocessed image as float32 array of shape (H, W, 3)

    Raises:
        ValueError: If image cannot be loaded
    """
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")

    original_h, original_w = image.shape[:2]

    # Step 2: Crop to remove borders/UI/text overlays
    y_start = int(original_h * crop_height[0])
    y_end = int(original_h * crop_height[1])
    x_start = int(original_w * crop_width[0])
    x_end = int(original_w * crop_width[1])

    cropped = image[y_start:y_end, x_start:x_end]

    # Step 3: Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)

    # Step 4: Convert to grayscale
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Step 5: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # CLAHE is better than regular histogram equalization for medical images
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(grayscale)

    # Step 6: Convert back to 3-channel RGB (required by pretrained models)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # Step 7: Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    return normalized


def preprocess_ultrasound_batch(
    image_paths: List[str],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Preprocess a batch of ultrasound images.

    Args:
        image_paths: List of paths to ultrasound images
        target_size: Target size for images

    Returns:
        Batch of preprocessed images as float32 array
    """
    images = []
    for path in image_paths:
        try:
            img = preprocess_ultrasound(path, target_size)
            images.append(img)
        except ValueError as e:
            LOGGER.warning("Skipping image: %s", e)
            continue

    return np.array(images, dtype=np.float32)


# =============================================================================
# BLADDER-SPECIFIC DATA FILTERING
# =============================================================================


# Keywords that indicate bladder-related ultrasound images
BLADDER_KEYWORDS = [
    "bladder", "urinary bladder", "UB", "ub", "vesica",
    "urinary_bladder", "bladder_us", "bladder_ultrasound"
]

# Keywords for organs to EXCLUDE (non-bladder organs)
EXCLUDE_KEYWORDS = [
    "liver", "kidney", "ovary", "prostate", "gallbladder",
    "spleen", "pancreas", "uterus", "testis", "heart"
]


def is_bladder_image(filename: str, metadata: Optional[Dict] = None) -> bool:
    """
    Determine if an image is bladder-related based on filename/metadata.

    This function checks:
    1. Filename patterns for bladder keywords
    2. Optional metadata dictionary for organ type
    3. Excludes images with non-bladder organ keywords

    Args:
        filename: Image filename (not full path)
        metadata: Optional dictionary with 'organ_type' or 'region' keys

    Returns:
        True if image is bladder-related, False otherwise
    """
    filename_lower = filename.lower()

    # Check if filename contains any exclude keywords
    for exclude_kw in EXCLUDE_KEYWORDS:
        if exclude_kw.lower() in filename_lower:
            return False

    # Check if filename contains bladder keywords
    for bladder_kw in BLADDER_KEYWORDS:
        if bladder_kw.lower() in filename_lower:
            return True

    # Check metadata if provided
    if metadata:
        organ_type = str(metadata.get("organ_type", "")).lower()
        region = str(metadata.get("region", "")).lower()

        for bladder_kw in BLADDER_KEYWORDS:
            if bladder_kw.lower() in organ_type or bladder_kw.lower() in region:
                return True

    # If no bladder keywords found but also no exclude keywords,
    # return True to include (assumes dataset is bladder-focused)
    # Change this logic if your dataset has mixed organs by default
    return True


def load_filtered_dataset(
    image_dir: str,
    metadata_csv: Optional[str] = None,
    filter_bladder_only: bool = True,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """
    Load and filter dataset to bladder-related images only.

    Directory structure expected:
        image_dir/
            normal/
                image1.jpg
                image2.jpg
                ...
            abnormal/
                image1.jpg
                ...

    Args:
        image_dir: Root directory containing 'normal' and 'abnormal' subdirs
        metadata_csv: Optional CSV with columns [filename, organ_type, ...]
        filter_bladder_only: If True, filter to bladder images only

    Returns:
        Tuple of (image_paths, labels, statistics_dict)
    """
    LOGGER.info("=" * 60)
    LOGGER.info("LOADING AND FILTERING DATASET")
    LOGGER.info("=" * 60)
    LOGGER.info("Image directory: %s", image_dir)
    LOGGER.info("Filter bladder only: %s", filter_bladder_only)

    # Load metadata if provided
    metadata_dict: Dict[str, Dict] = {}
    if metadata_csv and os.path.exists(metadata_csv):
        LOGGER.info("Loading metadata from: %s", metadata_csv)
        meta_df = pd.read_csv(metadata_csv)
        for _, row in meta_df.iterrows():
            metadata_dict[row["filename"]] = row.to_dict()

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    class_map = {"normal": 0, "abnormal": 1}

    image_paths: List[str] = []
    labels: List[int] = []

    stats = {
        "total_found": 0,
        "bladder_included": 0,
        "non_bladder_excluded": 0,
        "corrupt_skipped": 0,
        "normal": 0,
        "abnormal": 0,
    }

    for class_name, label in class_map.items():
        class_dir = Path(image_dir) / class_name

        if not class_dir.exists():
            LOGGER.warning("Class directory not found: %s", class_dir)
            continue

        for file_path in class_dir.glob("**/*"):
            if file_path.suffix.lower() not in valid_extensions:
                continue

            stats["total_found"] += 1
            filename = file_path.name

            # Filter bladder-only if enabled
            if filter_bladder_only:
                metadata = metadata_dict.get(filename, None)
                if not is_bladder_image(filename, metadata):
                    stats["non_bladder_excluded"] += 1
                    continue

            # Validate image can be read
            test_img = cv2.imread(str(file_path))
            if test_img is None:
                LOGGER.warning("Cannot read image: %s", file_path)
                stats["corrupt_skipped"] += 1
                continue

            image_paths.append(str(file_path))
            labels.append(label)
            stats["bladder_included"] += 1
            stats[class_name] += 1

    if not image_paths:
        raise RuntimeError(f"No valid images found in {image_dir}")

    # Print statistics
    LOGGER.info("-" * 40)
    LOGGER.info("DATASET STATISTICS:")
    LOGGER.info("-" * 40)
    LOGGER.info("Total images found:     %d", stats["total_found"])
    LOGGER.info("Bladder images included: %d", stats["bladder_included"])
    LOGGER.info("Non-bladder excluded:   %d", stats["non_bladder_excluded"])
    LOGGER.info("Corrupt/unreadable:     %d", stats["corrupt_skipped"])
    LOGGER.info("-" * 40)
    LOGGER.info("Normal (class 0):       %d (%.1f%%)",
                stats["normal"], 100 * stats["normal"] / len(image_paths))
    LOGGER.info("Abnormal (class 1):     %d (%.1f%%)",
                stats["abnormal"], 100 * stats["abnormal"] / len(image_paths))

    # Class imbalance warning
    if stats["normal"] > 0 and stats["abnormal"] > 0:
        ratio = max(stats["normal"], stats["abnormal"]) / min(stats["normal"], stats["abnormal"])
        LOGGER.info("Class imbalance ratio:  %.2f:1", ratio)
        if ratio > 3.0:
            LOGGER.warning("SEVERE CLASS IMBALANCE DETECTED! Using class weights is recommended.")

    LOGGER.info("=" * 60)

    # Print sample filenames for verification
    LOGGER.info("Sample filenames (first 5 of each class):")
    normal_samples = [p for p, l in zip(image_paths, labels) if l == 0][:5]
    abnormal_samples = [p for p, l in zip(image_paths, labels) if l == 1][:5]

    LOGGER.info("Normal samples:")
    for s in normal_samples:
        LOGGER.info("  - %s", Path(s).name)

    LOGGER.info("Abnormal samples:")
    for s in abnormal_samples:
        LOGGER.info("  - %s", Path(s).name)

    return image_paths, labels, stats


# =============================================================================
# IMAGE MODEL (DenseNet121 with Transfer Learning)
# =============================================================================


def build_image_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    feature_dim: int = 256,
    dropout_rate: float = 0.5,
    freeze_base: bool = True,
) -> tf.keras.Model:
    """
    Build DenseNet121-based image feature extractor.

    Architecture:
        Input (224x224x3)
        -> DenseNet121 (pretrained, optionally frozen)
        -> GlobalAveragePooling2D
        -> BatchNormalization
        -> Dense(256, relu)
        -> Dropout(0.5)
        -> Output: Feature vector (256-dim)

    Args:
        input_shape: Input image shape
        feature_dim: Dimension of output feature vector
        dropout_rate: Dropout rate for regularization
        freeze_base: If True, freeze all base layers initially

    Returns:
        Keras Model outputting feature vector (not prediction)
    """
    LOGGER.info("Building DenseNet121 image feature extractor...")

    # Load pretrained DenseNet121
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
    )

    # Freeze base layers initially
    base_model.trainable = not freeze_base

    # Build feature extractor
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Pass through DenseNet121
    x = base_model(inputs, training=False)

    # Global pooling
    x = layers.GlobalAveragePooling2D(name="image_global_pool")(x)

    # Batch normalization (helps with transfer learning)
    x = layers.BatchNormalization(name="image_bn")(x)

    # Dense layer for feature learning
    x = layers.Dense(feature_dim, activation="relu", name="image_features")(x)

    # Dropout for regularization
    outputs = layers.Dropout(dropout_rate, name="image_dropout")(x)

    model = Model(inputs, outputs, name="image_feature_extractor")

    LOGGER.info("  - Backbone: DenseNet121 (frozen=%s)", freeze_base)
    LOGGER.info("  - Feature dimension: %d", feature_dim)
    LOGGER.info("  - Total params: %d", model.count_params())

    return model


def unfreeze_image_model_layers(
    model: tf.keras.Model,
    num_layers: int = 20,
) -> None:
    """
    Unfreeze last N layers of DenseNet121 for fine-tuning.

    BatchNormalization layers are kept frozen for stability.

    Args:
        model: Image feature extractor model
        num_layers: Number of layers from end to unfreeze
    """
    # Find the DenseNet121 base model (should be second layer after input)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        LOGGER.warning("Could not find base model for unfreezing")
        return

    # Make base trainable
    base_model.trainable = True

    total_layers = len(base_model.layers)
    unfreeze_from = max(0, total_layers - num_layers)

    LOGGER.info("Unfreezing last %d layers of DenseNet121 (from %d/%d)",
                num_layers, unfreeze_from, total_layers)

    # Freeze early layers, unfreeze later layers
    for i, layer in enumerate(base_model.layers):
        if i < unfreeze_from:
            layer.trainable = False
        else:
            # Keep BatchNorm frozen for stability
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
    LOGGER.info("Trainable layers in base: %d/%d", trainable_count, total_layers)


# =============================================================================
# CLINICAL MODEL FEATURE EXTRACTOR
# =============================================================================


class ClinicalFeatureExtractor:
    """
    Wrapper for the existing clinical model to extract features.

    The clinical model is an sklearn Pipeline. This class wraps it
    to extract features before the final prediction layer.

    For XGBoost, we use the leaf indices as features.
    For other models, we use the preprocessed features.
    """

    def __init__(self, clinical_model_path: str):
        """
        Load clinical model from file.

        Args:
            clinical_model_path: Path to clinical model pickle file
        """
        LOGGER.info("Loading clinical model from: %s", clinical_model_path)

        payload = joblib.load(clinical_model_path)
        self.pipeline = payload["model"]
        self.selected_features = payload["selected_features"]
        self.model_name = payload.get("model_name", "unknown")

        LOGGER.info("  - Model type: %s", self.model_name)
        LOGGER.info("  - Selected features: %d", len(self.selected_features))

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features from clinical data.

        Uses the preprocessor from the pipeline to transform data,
        then returns the transformed features.

        Args:
            df: DataFrame with clinical features (must have required columns)

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        # Select required features
        df_selected = df[self.selected_features].copy()

        # Get preprocessor from pipeline
        preprocessor = self.pipeline.named_steps.get("preprocessor")

        if preprocessor is not None:
            # Transform using pipeline's preprocessor
            features = preprocessor.transform(df_selected)
        else:
            # No preprocessor, use raw values
            features = df_selected.values

        return features.astype(np.float32)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions from clinical model.

        Args:
            df: DataFrame with clinical features

        Returns:
            Array of shape (n_samples,) with probability of positive class
        """
        df_selected = df[self.selected_features].copy()
        proba = self.pipeline.predict_proba(df_selected)[:, 1]
        return proba


def build_clinical_feature_model(
    input_dim: int,
    feature_dim: int = 64,
    dropout_rate: float = 0.3,
) -> tf.keras.Model:
    """
    Build a simple neural network for clinical feature processing.

    This processes the features extracted from the sklearn clinical model
    and outputs a fixed-dimension feature vector for fusion.

    Architecture:
        Input (n_clinical_features)
        -> Dense(128, relu)
        -> BatchNormalization
        -> Dropout(0.3)
        -> Dense(64, relu)
        -> Output: Feature vector (64-dim)

    Args:
        input_dim: Number of input clinical features
        feature_dim: Dimension of output feature vector
        dropout_rate: Dropout rate

    Returns:
        Keras Model outputting feature vector
    """
    LOGGER.info("Building clinical feature model...")

    inputs = layers.Input(shape=(input_dim,), name="clinical_input")

    x = layers.Dense(128, activation="relu", name="clinical_dense1")(inputs)
    x = layers.BatchNormalization(name="clinical_bn1")(x)
    x = layers.Dropout(dropout_rate, name="clinical_dropout1")(x)

    outputs = layers.Dense(feature_dim, activation="relu", name="clinical_features")(x)

    model = Model(inputs, outputs, name="clinical_feature_model")

    LOGGER.info("  - Input dimension: %d", input_dim)
    LOGGER.info("  - Feature dimension: %d", feature_dim)

    return model


# =============================================================================
# MULTIMODAL FUSION MODEL
# =============================================================================


def build_fusion_model(
    image_input_shape: Tuple[int, int, int] = (224, 224, 3),
    clinical_input_dim: int = 25,
    config: Optional[TrainingConfig] = None,
) -> tf.keras.Model:
    """
    Build multimodal fusion model combining image and clinical features.

    Architecture:
        Image Input (224x224x3)           Clinical Input (n_features)
              |                                    |
        DenseNet121 (frozen)              Dense(128, relu)
              |                                    |
        GlobalAvgPool                      Dropout + Dense(64)
              |                                    |
        BatchNorm + Dense(256)                     |
              |                                    |
              +----------------+-------------------+
                               |
                          Concatenate
                               |
                      Dense(128, relu)
                               |
                         Dropout(0.5)
                               |
                       Dense(64, relu)
                               |
                      Dense(1, sigmoid)

    Args:
        image_input_shape: Input shape for images
        clinical_input_dim: Number of clinical features
        config: Training configuration

    Returns:
        Compiled Keras Model for binary classification
    """
    if config is None:
        config = TrainingConfig()

    LOGGER.info("=" * 60)
    LOGGER.info("BUILDING MULTIMODAL FUSION MODEL")
    LOGGER.info("=" * 60)

    # ========== IMAGE BRANCH ==========
    image_input = layers.Input(shape=image_input_shape, name="image_input")

    # DenseNet121 backbone
    densenet = tf.keras.applications.DenseNet121(
        include_top=False,
        weights="imagenet",
        input_shape=image_input_shape,
    )
    densenet.trainable = False  # Freeze initially

    x_img = densenet(image_input, training=False)
    x_img = layers.GlobalAveragePooling2D(name="img_global_pool")(x_img)
    x_img = layers.BatchNormalization(name="img_bn")(x_img)
    x_img = layers.Dense(config.image_feature_dim, activation="relu", name="img_features")(x_img)
    x_img = layers.Dropout(config.dropout_rate, name="img_dropout")(x_img)

    LOGGER.info("Image branch: DenseNet121 -> GlobalPool -> Dense(%d)",
                config.image_feature_dim)

    # ========== CLINICAL BRANCH ==========
    clinical_input = layers.Input(shape=(clinical_input_dim,), name="clinical_input")

    x_clin = layers.Dense(128, activation="relu", name="clin_dense1")(clinical_input)
    x_clin = layers.BatchNormalization(name="clin_bn1")(x_clin)
    x_clin = layers.Dropout(0.3, name="clin_dropout1")(x_clin)
    x_clin = layers.Dense(config.clinical_feature_dim, activation="relu", name="clin_features")(x_clin)

    LOGGER.info("Clinical branch: Dense(128) -> Dense(%d)", config.clinical_feature_dim)

    # ========== FUSION ==========
    # Concatenate image and clinical features
    fused = layers.Concatenate(name="fusion_concat")([x_img, x_clin])

    # Fusion layers
    x = layers.Dense(config.fusion_hidden_dim, activation="relu", name="fusion_dense1")(fused)
    x = layers.Dropout(config.dropout_rate, name="fusion_dropout1")(x)
    x = layers.Dense(64, activation="relu", name="fusion_dense2")(x)

    # Output
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    LOGGER.info("Fusion: Concat -> Dense(%d) -> Dense(64) -> Sigmoid",
                config.fusion_hidden_dim)

    # Build model
    model = Model(
        inputs=[image_input, clinical_input],
        outputs=output,
        name="multimodal_fusion"
    )

    LOGGER.info("-" * 40)
    LOGGER.info("Total parameters: %d", model.count_params())
    LOGGER.info("Trainable parameters: %d",
                sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    LOGGER.info("=" * 60)

    return model


def unfreeze_fusion_model_cnn(
    model: tf.keras.Model,
    num_layers: int = 20,
) -> None:
    """
    Unfreeze last N layers of the CNN in the fusion model for fine-tuning.

    Args:
        model: Multimodal fusion model
        num_layers: Number of CNN layers to unfreeze
    """
    # Find DenseNet121 in the model
    densenet = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "densenet" in layer.name.lower():
            densenet = layer
            break

    if densenet is None:
        # Try to find by checking for many layers (DenseNet has ~400 layers)
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and len(layer.layers) > 100:
                densenet = layer
                break

    if densenet is None:
        LOGGER.warning("Could not find DenseNet121 in fusion model")
        return

    # Make it trainable
    densenet.trainable = True

    total_layers = len(densenet.layers)
    unfreeze_from = max(0, total_layers - num_layers)

    LOGGER.info("Unfreezing last %d layers of DenseNet121", num_layers)

    for i, layer in enumerate(densenet.layers):
        if i < unfreeze_from:
            layer.trainable = False
        else:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

    trainable_count = sum(1 for layer in densenet.layers if layer.trainable)
    LOGGER.info("Trainable layers: %d/%d", trainable_count, total_layers)


# =============================================================================
# DATA GENERATOR FOR MULTIMODAL TRAINING
# =============================================================================


class MultimodalDataGenerator(KerasSequence):
    """
    Data generator for multimodal (image + clinical) training.

    Yields batches of (image_array, clinical_array), label_array.
    """

    def __init__(
        self,
        image_paths: List[str],
        clinical_features: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 16,
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        shuffle: bool = True,
    ):
        """
        Initialize multimodal data generator.

        Args:
            image_paths: List of paths to ultrasound images
            clinical_features: Array of clinical features (n_samples, n_features)
            labels: Array of labels (n_samples,)
            batch_size: Batch size
            image_size: Target image size (width, height)
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle at epoch end
        """
        self.image_paths = np.array(image_paths)
        self.clinical_features = clinical_features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.batch_size = batch_size
        self.image_size = image_size
        self.augment = augment
        self.shuffle = shuffle

        self.indices = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        """Return number of batches per epoch."""
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx: int):
        """
        Get batch of data.

        Returns:
            Tuple of ([images, clinical], labels)
        """
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Load and preprocess images
        images = []
        for i in batch_indices:
            try:
                img = preprocess_ultrasound(self.image_paths[i], self.image_size)

                if self.augment:
                    img = self._augment_image(img)

                images.append(img)
            except Exception as e:
                LOGGER.warning("Error loading image %s: %s", self.image_paths[i], e)
                # Use zeros as fallback
                images.append(np.zeros((*self.image_size, 3), dtype=np.float32))

        # Get clinical features and labels
        clinical = self.clinical_features[batch_indices]
        labels = self.labels[batch_indices]

        # Return as dict with input names matching model.input_names
        return {"image_input": np.array(images, dtype=np.float32),
                "clinical_input": clinical}, labels

    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """Apply simple augmentation to image."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)

        # Random brightness adjustment
        brightness_factor = np.random.uniform(0.85, 1.15)
        image = np.clip(image * brightness_factor, 0, 1)

        # Random contrast adjustment
        contrast_factor = np.random.uniform(0.85, 1.15)
        mean = image.mean()
        image = np.clip((image - mean) * contrast_factor + mean, 0, 1)

        return image.astype(np.float32)

    def on_epoch_end(self) -> None:
        """Shuffle indices at epoch end."""
        if self.shuffle:
            np.random.shuffle(self.indices)


# =============================================================================
# TRAINING PIPELINE
# =============================================================================


def compute_class_weights(labels: np.ndarray) -> Dict[int, float]:
    """
    Compute balanced class weights for imbalanced dataset.

    Args:
        labels: Array of class labels

    Returns:
        Dictionary mapping class index to weight
    """
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    class_weights = {int(c): float(w) for c, w in zip(classes, weights)}

    LOGGER.info("Class weights: %s", class_weights)
    return class_weights


def train_pipeline(
    image_dir: str,
    clinical_model_path: str,
    clinical_data_path: str,
    output_dir: str = "models",
    config: Optional[TrainingConfig] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Main training pipeline for multimodal fusion model.

    This function:
    1. Loads and filters bladder-only images
    2. Loads clinical model and extracts features
    3. Creates train/val/test splits
    4. Builds multimodal fusion model
    5. Trains with two-stage strategy
    6. Evaluates and saves results

    Args:
        image_dir: Directory with ultrasound images
        clinical_model_path: Path to trained clinical model
        clinical_data_path: Path to clinical CSV data
        output_dir: Directory to save outputs
        config: Training configuration
        seed: Random seed

    Returns:
        Dictionary with training results and metrics
    """
    if config is None:
        config = TrainingConfig()

    set_global_seed(seed)
    ensure_dir(output_dir)

    LOGGER.info("=" * 70)
    LOGGER.info("MULTIMODAL FUSION TRAINING PIPELINE")
    LOGGER.info("=" * 70)

    # ========================================
    # STEP 1: Load and filter image dataset
    # ========================================
    LOGGER.info("\n[STEP 1] Loading and filtering image dataset...")

    image_paths, image_labels, stats = load_filtered_dataset(
        image_dir,
        filter_bladder_only=True,
    )

    # ========================================
    # STEP 2: Load clinical model and data
    # ========================================
    LOGGER.info("\n[STEP 2] Loading clinical model and data...")

    clinical_extractor = ClinicalFeatureExtractor(clinical_model_path)

    # Load clinical data
    clinical_df = pd.read_csv(clinical_data_path)
    LOGGER.info("Clinical data shape: %s", clinical_df.shape)

    # For this implementation, we assume images and clinical data can be aligned
    # In a real scenario, you would have a patient_id mapping both
    #
    # IMPORTANT: This is a simplified version. In production:
    # - Each image should have a patient_id
    # - Clinical data should be matched by patient_id
    # - For now, we sample clinical data randomly to demonstrate the pipeline

    # Extract clinical features
    try:
        # Try to extract features for all available clinical data
        clinical_features = clinical_extractor.extract_features(clinical_df)
        n_clinical_features = clinical_features.shape[1]
        LOGGER.info("Clinical features shape: %s", clinical_features.shape)
    except Exception as e:
        LOGGER.error("Error extracting clinical features: %s", e)
        raise

    # Align data (simplified - in production use patient IDs)
    # Here we randomly sample clinical features to match image count
    n_images = len(image_paths)
    n_clinical = clinical_features.shape[0]

    if n_images > n_clinical:
        LOGGER.warning("More images than clinical records. Sampling with replacement.")
        clinical_indices = np.random.choice(n_clinical, size=n_images, replace=True)
    else:
        clinical_indices = np.random.choice(n_clinical, size=n_images, replace=False)

    aligned_clinical = clinical_features[clinical_indices]
    labels = np.array(image_labels)

    LOGGER.info("Aligned dataset size: %d samples", n_images)

    # ========================================
    # STEP 3: Split data
    # ========================================
    LOGGER.info("\n[STEP 3] Creating train/val/test splits...")

    # First split: train+val vs test
    (
        train_val_paths,
        test_paths,
        train_val_clinical,
        test_clinical,
        train_val_labels,
        test_labels,
    ) = train_test_split(
        image_paths,
        aligned_clinical,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=seed,
    )

    # Second split: train vs val
    (
        train_paths,
        val_paths,
        train_clinical,
        val_clinical,
        train_labels,
        val_labels,
    ) = train_test_split(
        train_val_paths,
        train_val_clinical,
        train_val_labels,
        test_size=0.18,  # ~15% of total
        stratify=train_val_labels,
        random_state=seed,
    )

    LOGGER.info("Train: %d, Val: %d, Test: %d",
                len(train_paths), len(val_paths), len(test_paths))

    # ========================================
    # STEP 4: Compute class weights
    # ========================================
    LOGGER.info("\n[STEP 4] Computing class weights...")

    class_weights = compute_class_weights(train_labels)

    # ========================================
    # STEP 5: Create data generators
    # ========================================
    LOGGER.info("\n[STEP 5] Creating data generators...")

    train_gen = MultimodalDataGenerator(
        image_paths=train_paths,
        clinical_features=train_clinical,
        labels=train_labels,
        batch_size=config.batch_size,
        image_size=config.image_size,
        augment=True,
        shuffle=True,
    )

    val_gen = MultimodalDataGenerator(
        image_paths=val_paths,
        clinical_features=val_clinical,
        labels=val_labels,
        batch_size=config.batch_size,
        image_size=config.image_size,
        augment=False,
        shuffle=False,
    )

    test_gen = MultimodalDataGenerator(
        image_paths=test_paths,
        clinical_features=test_clinical,
        labels=test_labels,
        batch_size=config.batch_size,
        image_size=config.image_size,
        augment=False,
        shuffle=False,
    )

    # ========================================
    # STEP 6: Build fusion model
    # ========================================
    LOGGER.info("\n[STEP 6] Building multimodal fusion model...")

    model = build_fusion_model(
        image_input_shape=(*config.image_size, 3),
        clinical_input_dim=n_clinical_features,
        config=config,
    )

    # ========================================
    # STEP 7: Stage 1 - Train with frozen CNN
    # ========================================
    LOGGER.info("\n[STEP 7] Stage 1: Training with frozen CNN...")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.stage1_lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    stage1_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=config.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=config.reduce_lr_patience,
            mode="max",
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, "fusion_stage1_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_stage1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.stage1_epochs,
        class_weight=class_weights,
        callbacks=stage1_callbacks,
        verbose=1,
    )

    best_val_auc_stage1 = max(history_stage1.history["val_auc"])
    LOGGER.info("Stage 1 completed. Best val_auc: %.4f", best_val_auc_stage1)

    # ========================================
    # STEP 8: Stage 2 - Fine-tune CNN layers
    # ========================================
    LOGGER.info("\n[STEP 8] Stage 2: Fine-tuning CNN layers...")

    # Unfreeze last N layers of CNN
    unfreeze_fusion_model_cnn(model, num_layers=config.layers_to_unfreeze)

    # Recompile with lower learning rate
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.stage2_lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    stage2_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=config.early_stopping_patience,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=config.reduce_lr_patience,
            mode="max",
            min_lr=1e-8,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, "fusion_stage2_best.keras"),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
    ]

    history_stage2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.stage2_epochs,
        class_weight=class_weights,
        callbacks=stage2_callbacks,
        verbose=1,
    )

    best_val_auc_stage2 = max(history_stage2.history["val_auc"])
    LOGGER.info("Stage 2 completed. Best val_auc: %.4f", best_val_auc_stage2)

    # ========================================
    # STEP 9: Comprehensive evaluation
    # ========================================
    LOGGER.info("\n[STEP 9] Comprehensive evaluation...")

    test_metrics = evaluate_multimodal_model(
        model=model,
        data_generator=test_gen,
        labels=test_labels,
        split_name="test",
        save_dir=output_dir,
    )

    val_metrics = evaluate_multimodal_model(
        model=model,
        data_generator=val_gen,
        labels=val_labels,
        split_name="validation",
        save_dir=output_dir,
    )

    # ========================================
    # STEP 10: Save final model
    # ========================================
    LOGGER.info("\n[STEP 10] Saving final model...")

    model_path = os.path.join(output_dir, "multimodal_fusion_model.keras")
    model.save(model_path)
    LOGGER.info("Model saved to: %s", model_path)

    # Save training history
    history_combined = {
        "loss": history_stage1.history["loss"] + history_stage2.history["loss"],
        "val_loss": history_stage1.history["val_loss"] + history_stage2.history["val_loss"],
        "accuracy": history_stage1.history["accuracy"] + history_stage2.history["accuracy"],
        "val_accuracy": history_stage1.history["val_accuracy"] + history_stage2.history["val_accuracy"],
        "auc": history_stage1.history["auc"] + history_stage2.history["auc"],
        "val_auc": history_stage1.history["val_auc"] + history_stage2.history["val_auc"],
    }

    history_df = pd.DataFrame(history_combined)
    history_path = os.path.join(output_dir, "fusion_training_history.csv")
    history_df.to_csv(history_path, index=False)

    # Plot training history
    plot_training_history(history_combined, os.path.join(output_dir, "fusion_training_history.png"))

    # ========================================
    # Final summary
    # ========================================
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("TRAINING COMPLETED")
    LOGGER.info("=" * 70)
    LOGGER.info("Best Validation AUC: %.4f", max(best_val_auc_stage1, best_val_auc_stage2))
    LOGGER.info("Test ROC-AUC:        %.4f", test_metrics["roc_auc"])
    LOGGER.info("Test Accuracy:       %.4f", test_metrics["accuracy"])
    LOGGER.info("Test F1 Score:       %.4f", test_metrics["f1"])
    LOGGER.info("=" * 70)

    return {
        "model_path": model_path,
        "test_metrics": test_metrics,
        "val_metrics": val_metrics,
        "history_path": history_path,
        "best_val_auc": max(best_val_auc_stage1, best_val_auc_stage2),
    }


# =============================================================================
# EVALUATION AND VISUALIZATION
# =============================================================================


def evaluate_multimodal_model(
    model: tf.keras.Model,
    data_generator: MultimodalDataGenerator,
    labels: np.ndarray,
    split_name: str = "test",
    save_dir: str = "results",
) -> Dict[str, float]:
    """
    Comprehensive evaluation of multimodal model with debugging checks.

    Generates:
    - ROC curve
    - Confusion matrix
    - Prediction distribution analysis
    - Classification report

    Args:
        model: Trained multimodal model
        data_generator: Data generator for evaluation
        labels: True labels
        split_name: Name of split (test, validation)
        save_dir: Directory to save plots

    Returns:
        Dictionary of metrics
    """
    ensure_dir(save_dir)

    LOGGER.info("=" * 60)
    LOGGER.info("EVALUATING ON %s SET", split_name.upper())
    LOGGER.info("=" * 60)

    # Collect predictions
    y_prob_list = []
    y_true_list = []

    for inputs, batch_labels in data_generator:
        pred = model.predict(inputs, verbose=0).ravel()
        y_prob_list.extend(pred.tolist())
        y_true_list.extend(batch_labels.tolist())

    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list)
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # Print metrics
    LOGGER.info("Metrics:")
    LOGGER.info("  Accuracy:  %.4f", metrics["accuracy"])
    LOGGER.info("  Precision: %.4f", metrics["precision"])
    LOGGER.info("  Recall:    %.4f", metrics["recall"])
    LOGGER.info("  F1 Score:  %.4f", metrics["f1"])
    LOGGER.info("  ROC-AUC:   %.4f", metrics["roc_auc"])

    # Debugging checks
    LOGGER.info("-" * 40)
    LOGGER.info("DEBUGGING CHECKS:")
    LOGGER.info("  Mean prediction:  %.4f (expect ~0.3-0.7)", y_prob.mean())
    LOGGER.info("  Std prediction:   %.4f (expect >0.1)", y_prob.std())
    LOGGER.info("  Min prediction:   %.4f", y_prob.min())
    LOGGER.info("  Max prediction:   %.4f", y_prob.max())

    unique_preds = len(np.unique(y_pred))
    LOGGER.info("  Unique predictions: %d (expect 2)", unique_preds)

    if unique_preds == 1:
        LOGGER.error("MODEL COLLAPSED! Predicting only one class.")
    elif y_prob.std() < 0.05:
        LOGGER.warning("Low prediction variance - model may not be learning.")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"ROC Curve - {split_name.capitalize()} Set", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fusion_roc_{split_name}.png"), dpi=300)
    plt.close()

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Abnormal"],
        yticklabels=["Normal", "Abnormal"],
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(f"Confusion Matrix - {split_name.capitalize()}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fusion_cm_{split_name}.png"), dpi=300)
    plt.close()

    # Plot prediction distribution
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_prob[y_true == 0], bins=30, alpha=0.7, label="Normal", color="blue")
    plt.hist(y_prob[y_true == 1], bins=30, alpha=0.7, label="Abnormal", color="red")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Prediction Distribution by Class")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(y_prob, bins=50, color="green", alpha=0.7)
    plt.axvline(0.5, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Overall Prediction Distribution")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fusion_dist_{split_name}.png"), dpi=300)
    plt.close()

    # Save classification report
    report = classification_report(y_true, y_pred, target_names=["Normal", "Abnormal"])
    report_path = os.path.join(save_dir, f"fusion_report_{split_name}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    LOGGER.info("=" * 60)

    return metrics


def plot_training_history(history: Dict[str, List[float]], save_path: str) -> None:
    """Plot training history curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    axes[0].plot(history["loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(history["accuracy"], label="Train", linewidth=2)
    axes[1].plot(history["val_accuracy"], label="Validation", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # AUC
    axes[2].plot(history["auc"], label="Train", linewidth=2)
    axes[2].plot(history["val_auc"], label="Validation", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_title("AUC", fontweight="bold")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    LOGGER.info("Training history plot saved to: %s", save_path)


# =============================================================================
# GRAD-CAM VISUALIZATION
# =============================================================================


def compute_gradcam(
    model: tf.keras.Model,
    image: np.ndarray,
    clinical_features: np.ndarray,
    layer_name: Optional[str] = None,
) -> np.ndarray:
    """
    Compute Grad-CAM heatmap for multimodal model.

    Grad-CAM highlights the regions of the image that most influenced
    the model's prediction, helping verify the model focuses on
    clinically relevant areas (bladder region).

    Args:
        model: Trained multimodal fusion model
        image: Preprocessed image array (H, W, C)
        clinical_features: Clinical feature vector (1, n_features)
        layer_name: Target conv layer name (auto-detected if None)

    Returns:
        Grad-CAM heatmap as numpy array
    """
    # Find the last conv layer in DenseNet121
    if layer_name is None:
        # Auto-detect: find DenseNet and get last conv layer
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and len(layer.layers) > 100:
                # This is likely DenseNet121
                for sublayer in reversed(layer.layers):
                    if "conv" in sublayer.name.lower():
                        layer_name = sublayer.name
                        break
                break

    if layer_name is None:
        LOGGER.warning("Could not find conv layer for Grad-CAM")
        return np.zeros(image.shape[:2])

    # Create gradient model
    # Find the target layer
    target_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            for sublayer in layer.layers:
                if sublayer.name == layer_name:
                    target_layer = sublayer
                    break

    if target_layer is None:
        LOGGER.warning("Target layer %s not found", layer_name)
        return np.zeros(image.shape[:2])

    # Build gradient model
    grad_model = Model(
        inputs=model.inputs,
        outputs=[target_layer.output, model.output]
    )

    # Compute gradients
    image_batch = np.expand_dims(image, axis=0)
    clinical_batch = clinical_features.reshape(1, -1)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model([image_batch, clinical_batch])
        loss = predictions[0]

    # Compute gradients of output w.r.t. conv layer
    grads = tape.gradient(loss, conv_output)

    if grads is None:
        LOGGER.warning("Gradients are None")
        return np.zeros(image.shape[:2])

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the conv output by the pooled gradients
    conv_output = conv_output[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(len(pooled_grads)):
        conv_output[:, :, i] *= pooled_grads[i]

    # Average across channels
    heatmap = np.mean(conv_output, axis=-1)

    # ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap


def visualize_gradcam(
    model: tf.keras.Model,
    image_path: str,
    clinical_features: np.ndarray,
    save_path: str,
    image_size: Tuple[int, int] = (224, 224),
) -> None:
    """
    Generate and save Grad-CAM visualization.

    Creates a side-by-side plot showing:
    - Original preprocessed image
    - Grad-CAM heatmap overlay

    Args:
        model: Trained multimodal model
        image_path: Path to ultrasound image
        clinical_features: Clinical feature vector
        save_path: Path to save visualization
        image_size: Image size for preprocessing
    """
    # Preprocess image
    image = preprocess_ultrasound(image_path, image_size)

    # Compute Grad-CAM
    heatmap = compute_gradcam(model, image, clinical_features)

    # Resize heatmap to match image
    heatmap_resized = cv2.resize(heatmap, image_size)

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Preprocessed Image", fontweight="bold")
    axes[0].axis("off")

    # Heatmap
    axes[1].imshow(heatmap_resized, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap", fontweight="bold")
    axes[1].axis("off")

    # Overlay
    overlay = image.copy()
    heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
    blended = 0.6 * overlay + 0.4 * heatmap_colored
    blended = np.clip(blended, 0, 1)

    axes[2].imshow(blended)
    axes[2].set_title("Overlay", fontweight="bold")
    axes[2].axis("off")

    # Get prediction
    image_batch = np.expand_dims(image, axis=0)
    clinical_batch = clinical_features.reshape(1, -1)
    pred = model.predict([image_batch, clinical_batch], verbose=0)[0, 0]

    plt.suptitle(f"Prediction: {pred:.4f} ({'Abnormal' if pred > 0.5 else 'Normal'})",
                 fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    LOGGER.info("Grad-CAM visualization saved to: %s", save_path)


def generate_gradcam_examples(
    model: tf.keras.Model,
    image_paths: List[str],
    clinical_features: np.ndarray,
    labels: np.ndarray,
    save_dir: str,
    n_examples: int = 8,
) -> None:
    """
    Generate Grad-CAM visualizations for sample images.

    Selects examples from both classes to visualize model attention.

    Args:
        model: Trained multimodal model
        image_paths: List of image paths
        clinical_features: Clinical features array
        labels: Labels array
        save_dir: Directory to save visualizations
        n_examples: Number of examples per class
    """
    ensure_dir(save_dir)

    LOGGER.info("Generating Grad-CAM examples...")

    # Get indices for each class
    normal_idx = np.where(labels == 0)[0]
    abnormal_idx = np.where(labels == 1)[0]

    # Sample indices
    np.random.seed(42)
    normal_samples = np.random.choice(normal_idx, min(n_examples, len(normal_idx)), replace=False)
    abnormal_samples = np.random.choice(abnormal_idx, min(n_examples, len(abnormal_idx)), replace=False)

    # Generate visualizations
    for i, idx in enumerate(normal_samples):
        save_path = os.path.join(save_dir, f"gradcam_normal_{i}.png")
        try:
            visualize_gradcam(
                model=model,
                image_path=image_paths[idx],
                clinical_features=clinical_features[idx],
                save_path=save_path,
            )
        except Exception as e:
            LOGGER.warning("Failed to generate Grad-CAM for %s: %s", image_paths[idx], e)

    for i, idx in enumerate(abnormal_samples):
        save_path = os.path.join(save_dir, f"gradcam_abnormal_{i}.png")
        try:
            visualize_gradcam(
                model=model,
                image_path=image_paths[idx],
                clinical_features=clinical_features[idx],
                save_path=save_path,
            )
        except Exception as e:
            LOGGER.warning("Failed to generate Grad-CAM for %s: %s", image_paths[idx], e)

    LOGGER.info("Grad-CAM visualizations saved to: %s", save_dir)


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================


def main() -> None:
    """Command-line interface entrypoint."""
    parser = argparse.ArgumentParser(
        description="Train multimodal fusion model for UTI prediction"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Override image directory from config"
    )
    parser.add_argument(
        "--clinical-model",
        default=None,
        help="Override clinical model path from config"
    )
    parser.add_argument(
        "--clinical-data",
        default=None,
        help="Override clinical data path from config"
    )
    parser.add_argument(
        "--output-dir",
        default="models",
        help="Output directory for models and results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    setup_logging()

    # Load config
    config = load_config(args.config)

    # Override with CLI args if provided
    image_dir = args.image_dir or config["paths"]["ultrasound_raw_dir"]
    clinical_model = args.clinical_model or config["clinical"]["model_output"]
    clinical_data = args.clinical_data or config["paths"]["clinical_raw"]

    # Training config
    train_config = TrainingConfig(
        batch_size=args.batch_size,
    )

    LOGGER.info("Starting multimodal fusion training...")
    LOGGER.info("Image directory: %s", image_dir)
    LOGGER.info("Clinical model: %s", clinical_model)
    LOGGER.info("Clinical data: %s", clinical_data)

    # Run training pipeline
    results = train_pipeline(
        image_dir=image_dir,
        clinical_model_path=clinical_model,
        clinical_data_path=clinical_data,
        output_dir=args.output_dir,
        config=train_config,
        seed=args.seed,
    )

    LOGGER.info("\nTraining completed!")
    LOGGER.info("Model saved to: %s", results["model_path"])
    LOGGER.info("Test ROC-AUC: %.4f", results["test_metrics"]["roc_auc"])


if __name__ == "__main__":
    main()
