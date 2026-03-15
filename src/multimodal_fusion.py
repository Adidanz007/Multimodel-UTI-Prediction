"""Multimodal fusion strategies for combining clinical and ultrasound predictions."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["clinical_prob", "ultrasound_prob", "label"]


def train_fusion_models(
    prediction_csv: str,
    config_path: str = "config/config.yaml",
) -> Dict[str, float]:
    """Train weighted and stacked fusion models on aligned prediction table."""
    config = load_config(config_path)
    set_global_seed(int(config["project"]["seed"]))

    df = pd.read_csv(prediction_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in fusion csv: {missing}")

    x = df[["clinical_prob", "ultrasound_prob"]]
    y = df["label"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=int(config["project"]["seed"]),
        stratify=y,
    )

    wc = float(config["fusion"]["weighted_clinical"])
    wu = float(config["fusion"]["weighted_ultrasound"])

    weighted_prob = wc * x_test["clinical_prob"].to_numpy() + wu * x_test["ultrasound_prob"].to_numpy()
    weighted_pred = (weighted_prob >= 0.5).astype(int)

    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(x_train, y_train)
    stacked_prob = stacker.predict_proba(x_test)[:, 1]
    stacked_pred = (stacked_prob >= 0.5).astype(int)

    weighted_metrics = {
        "weighted_accuracy": accuracy_score(y_test, weighted_pred),
        "weighted_precision": precision_score(y_test, weighted_pred, zero_division=0),
        "weighted_recall": recall_score(y_test, weighted_pred, zero_division=0),
        "weighted_f1": f1_score(y_test, weighted_pred, zero_division=0),
        "weighted_roc_auc": roc_auc_score(y_test, weighted_prob),
    }
    stacked_metrics = {
        "stacked_accuracy": accuracy_score(y_test, stacked_pred),
        "stacked_precision": precision_score(y_test, stacked_pred, zero_division=0),
        "stacked_recall": recall_score(y_test, stacked_pred, zero_division=0),
        "stacked_f1": f1_score(y_test, stacked_pred, zero_division=0),
        "stacked_roc_auc": roc_auc_score(y_test, stacked_prob),
    }

    mlflow.set_experiment(str(config["project"]["mlflow_experiment"]))
    with mlflow.start_run(run_name="multimodal_fusion"):
        mlflow.log_param("weighted_clinical", wc)
        mlflow.log_param("weighted_ultrasound", wu)
        mlflow.log_metrics({**weighted_metrics, **stacked_metrics})

    output_path = str(config["fusion"]["model_output"])
    joblib.dump(
        {
            "weighted_clinical": wc,
            "weighted_ultrasound": wu,
            "stacking_model": stacker,
        },
        output_path,
    )

    metrics_dir = str(config["paths"]["results_metrics_dir"])
    ensure_dir(metrics_dir)
    metrics_path = os.path.join(metrics_dir, "fusion_metrics.csv")
    pd.DataFrame([{**weighted_metrics, **stacked_metrics}]).to_csv(metrics_path, index=False)

    LOGGER.info("Saved multimodal fusion model to %s", output_path)
    return {**weighted_metrics, **stacked_metrics}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")
    parser.add_argument("--prediction-csv", required=True, help="CSV with clinical_prob, ultrasound_prob, label")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    metrics = train_fusion_models(args.prediction_csv, args.config)
    LOGGER.info("Fusion stacked ROC-AUC: %.4f", metrics["stacked_roc_auc"])


if __name__ == "__main__":
    main()
