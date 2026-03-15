"""Train ultrasound image model using transfer learning."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import cv2
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.utils import Sequence as KerasSequence

from src.utils import end_run_timer, ensure_dir, load_config, set_global_seed, setup_logging, start_run_timer

LOGGER = logging.getLogger(__name__)


@dataclass
class ImageRecord:
    path: str
    label: int


class UltrasoundSequence(KerasSequence):
    """Custom image sequence with optional Albumentations augmentation."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        image_size: Tuple[int, int],
        batch_size: int,
        augment: bool,
        aug_cfg: Dict[str, float],
    ) -> None:
        self.records = list(records)
        self.image_size = image_size
        self.batch_size = batch_size
        self.augment = augment
        self.augmenter = (
            A.Compose(
                [
                    A.Rotate(limit=aug_cfg["rotation_limit"], p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=aug_cfg["width_shift_limit"],
                        scale_limit=aug_cfg["zoom_limit"],
                        rotate_limit=0,
                        p=0.5,
                        border_mode=cv2.BORDER_REFLECT,
                    ),
                    A.HorizontalFlip(p=0.5 if aug_cfg["horizontal_flip"] else 0.0),
                    A.RandomBrightnessContrast(
                        brightness_limit=(aug_cfg["brightness_min"] - 1.0, aug_cfg["brightness_max"] - 1.0),
                        contrast_limit=0.0,
                        p=0.5,
                    ),
                ]
            )
            if augment
            else None
        )

    def __len__(self) -> int:
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch = self.records[idx * self.batch_size : (idx + 1) * self.batch_size]
        images: List[np.ndarray] = []
        labels: List[int] = []

        for rec in batch:
            image = cv2.imread(rec.path)
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size)

            if self.augmenter is not None:
                image = self.augmenter(image=image)["image"]

            image = image.astype(np.float32) / 255.0
            images.append(image)
            labels.append(rec.label)

        return np.array(images), np.array(labels)


def _collect_raw_images(raw_dir: str) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    class_map = {"normal": 0, "abnormal": 1}

    for class_name, label in class_map.items():
        class_dir = Path(raw_dir) / class_name
        if not class_dir.exists():
            continue
        for file_path in class_dir.glob("**/*"):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                records.append(ImageRecord(path=str(file_path), label=label))

    if not records:
        raise RuntimeError(f"No images found in {raw_dir}")
    return records


def _write_split(records: Sequence[ImageRecord], split_dir: str, split_name: str) -> None:
    for rec in records:
        class_name = "abnormal" if rec.label == 1 else "normal"
        target_dir = Path(split_dir) / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(rec.path).name
        if not target_path.exists():
            shutil.copy2(rec.path, target_path)


def _create_or_refresh_split(config: Dict[str, object]) -> None:
    raw_dir = str(config["paths"]["ultrasound_raw_dir"])
    split_dir = str(config["paths"]["ultrasound_split_dir"])

    train_normal = Path(split_dir) / "train" / "normal"
    train_abnormal = Path(split_dir) / "train" / "abnormal"
    if train_normal.exists() and train_abnormal.exists() and any(train_normal.iterdir()) and any(train_abnormal.iterdir()):
        LOGGER.info("Using existing ultrasound split at %s", split_dir)
        return

    records = _collect_raw_images(raw_dir)
    labels = [r.label for r in records]

    test_size = float(config["ultrasound"]["test_size"])
    val_size = float(config["ultrasound"]["val_size"])

    train_val, test = train_test_split(
        records,
        test_size=test_size,
        stratify=labels,
        random_state=int(config["project"]["seed"]),
    )
    train_val_labels = [r.label for r in train_val]
    val_ratio_within_train_val = val_size / (1.0 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_ratio_within_train_val,
        stratify=train_val_labels,
        random_state=int(config["project"]["seed"]),
    )

    _write_split(train, split_dir, "train")
    _write_split(val, split_dir, "val")
    _write_split(test, split_dir, "test")
    LOGGER.info("Created stratified ultrasound split at %s", split_dir)


def _load_split_records(split_dir: str, split_name: str) -> List[ImageRecord]:
    records: List[ImageRecord] = []
    for class_name, label in [("normal", 0), ("abnormal", 1)]:
        class_dir = Path(split_dir) / split_name / class_name
        if not class_dir.exists():
            continue
        for file_path in class_dir.glob("**/*"):
            if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                records.append(ImageRecord(path=str(file_path), label=label))
    return records


def _build_backbone(name: str, input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    name_to_fn = {
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
        "MobileNetV2": tf.keras.applications.MobileNetV2,
    }
    if name not in name_to_fn:
        raise ValueError(f"Unsupported backbone: {name}")
    return name_to_fn[name](include_top=False, weights="imagenet", input_shape=input_shape)


def _build_model(config: Dict[str, object]) -> tf.keras.Model:
    image_size = tuple(config["ultrasound"]["image_size"])
    input_shape = (image_size[0], image_size[1], 3)
    backbone_name = str(config["ultrasound"]["backbone"])

    base_model = _build_backbone(backbone_name, input_shape)
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=float(config["ultrasound"]["learning_rate"])),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def train_ultrasound_model(config_path: str = "config/config.yaml") -> Dict[str, float]:
    """Train transfer-learning model for ultrasound classification."""
    config = load_config(config_path)
    set_global_seed(int(config["project"]["seed"]))

    ensure_dir(str(config["paths"]["models_dir"]))
    ensure_dir(str(config["paths"]["results_metrics_dir"]))

    _create_or_refresh_split(config)
    split_dir = str(config["paths"]["ultrasound_split_dir"])

    train_records = _load_split_records(split_dir, "train")
    val_records = _load_split_records(split_dir, "val")
    test_records = _load_split_records(split_dir, "test")

    if not train_records or not val_records or not test_records:
        raise RuntimeError("Ultrasound train/val/test split is empty")

    image_size = tuple(config["ultrasound"]["image_size"])
    batch_size = int(config["ultrasound"]["batch_size"])

    train_seq = UltrasoundSequence(
        records=train_records,
        image_size=image_size,
        batch_size=batch_size,
        augment=True,
        aug_cfg=config["augmentation"],
    )
    val_seq = UltrasoundSequence(
        records=val_records,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        aug_cfg=config["augmentation"],
    )
    test_seq = UltrasoundSequence(
        records=test_records,
        image_size=image_size,
        batch_size=batch_size,
        augment=False,
        aug_cfg=config["augmentation"],
    )

    model = _build_model(config)

    class_weight = {
        0: float(config["ultrasound"]["class_weight_normal"]),
        1: float(config["ultrasound"]["class_weight_abnormal"]),
    }

    callback_list = [
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=int(config["ultrasound"]["early_stopping_patience"]),
            mode="max",
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            factor=0.5,
            patience=int(config["ultrasound"]["lr_plateau_patience"]),
            mode="max",
            min_lr=1e-6,
        ),
    ]

    mlflow.set_experiment(str(config["project"]["mlflow_experiment"]))
    start_ts = start_run_timer()
    with mlflow.start_run(run_name="ultrasound_transfer_learning"):
        history = model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=int(config["ultrasound"]["epochs"]),
            class_weight=class_weight,
            callbacks=callback_list,
            verbose=1,
        )

        y_true: List[int] = []
        y_prob: List[float] = []

        for batch_x, batch_y in test_seq:
            pred = model.predict(batch_x, verbose=0).ravel()
            y_true.extend(batch_y.tolist())
            y_prob.extend(pred.tolist())

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)

        metrics = {
            "test_accuracy": accuracy_score(y_true_arr, y_pred_arr),
            "test_precision": precision_score(y_true_arr, y_pred_arr, zero_division=0),
            "test_recall": recall_score(y_true_arr, y_pred_arr, zero_division=0),
            "test_f1": f1_score(y_true_arr, y_pred_arr, zero_division=0),
            "test_roc_auc": roc_auc_score(y_true_arr, y_prob_arr),
            "training_seconds": end_run_timer(start_ts),
        }

        mlflow.log_params(
            {
                "backbone": config["ultrasound"]["backbone"],
                "image_size": image_size,
                "batch_size": batch_size,
                "epochs": config["ultrasound"]["epochs"],
                "class_weight_normal": class_weight[0],
                "class_weight_abnormal": class_weight[1],
            }
        )
        mlflow.log_metrics(metrics)

        history_df = pd.DataFrame(history.history)
        history_path = os.path.join(str(config["paths"]["results_metrics_dir"]), "ultrasound_training_history.csv")
        history_df.to_csv(history_path, index=False)
        mlflow.log_artifact(history_path)

    model_output = str(config["ultrasound"]["model_output"])
    model.save(model_output)
    LOGGER.info("Saved ultrasound model to %s", model_output)

    metrics_path = os.path.join(str(config["paths"]["results_metrics_dir"]), "ultrasound_test_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    return metrics


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train ultrasound model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    metrics = train_ultrasound_model(args.config)
    LOGGER.info("Ultrasound model test ROC-AUC: %.4f", metrics["test_roc_auc"])


if __name__ == "__main__":
    main()
