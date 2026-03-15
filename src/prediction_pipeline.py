"""Inference pipeline for multimodal UTI prediction."""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import cv2
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import load_config


def _predict_clinical(clinical_payload: Dict[str, object], clinical_model_path: str) -> float:
    model_payload = joblib.load(clinical_model_path)
    model = model_payload["model"]
    selected_features = model_payload["selected_features"]

    row = pd.DataFrame([clinical_payload])
    row = row.reindex(columns=selected_features, fill_value=0)
    return float(model.predict_proba(row)[:, 1][0])


def _predict_ultrasound(image_path: str, ultrasound_model_path: str, image_size: Tuple[int, int]) -> float:
    model = tf.keras.models.load_model(ultrasound_model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Unable to read image at {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)

    return float(model.predict(image, verbose=0).ravel()[0])


def predict_multimodal(
    clinical_payload: Dict[str, object],
    image_path: str,
    config_path: str = "config/config.yaml",
) -> Dict[str, float]:
    """Run multimodal inference and return probabilities."""
    config = load_config(config_path)

    clinical_prob = _predict_clinical(clinical_payload, config["clinical"]["model_output"])
    ultrasound_prob = _predict_ultrasound(
        image_path,
        config["ultrasound"]["model_output"],
        tuple(config["ultrasound"]["image_size"]),
    )

    fusion_payload = joblib.load(config["fusion"]["model_output"])
    weighted_prob = (
        float(fusion_payload["weighted_clinical"]) * clinical_prob
        + float(fusion_payload["weighted_ultrasound"]) * ultrasound_prob
    )

    stacker = fusion_payload["stacking_model"]
    stacked_prob = float(stacker.predict_proba(np.array([[clinical_prob, ultrasound_prob]]))[:, 1][0])

    final_prob = stacked_prob
    final_label = int(final_prob >= 0.5)

    return {
        "clinical_probability": clinical_prob,
        "ultrasound_probability": ultrasound_prob,
        "weighted_fusion_probability": weighted_prob,
        "stacked_fusion_probability": stacked_prob,
        "final_probability": final_prob,
        "final_prediction": final_label,
    }


def main() -> None:
    """Simple CLI for prediction using csv row and image path."""
    parser = argparse.ArgumentParser(description="Run multimodal prediction")
    parser.add_argument("--clinical-csv", required=True, help="CSV containing one row of clinical features")
    parser.add_argument("--image", required=True, help="Path to ultrasound image")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    payload = pd.read_csv(args.clinical_csv).iloc[0].to_dict()
    result = predict_multimodal(payload, args.image, args.config)
    print(result)


if __name__ == "__main__":
    main()
