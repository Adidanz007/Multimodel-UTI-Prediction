"""Evaluation helpers for binary classification models."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils import ensure_dir, load_config, setup_logging

LOGGER = logging.getLogger(__name__)


def evaluate_binary_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_prefix: str,
    config_path: str = "config/config.yaml",
) -> Dict[str, float]:
    """Compute metrics and save confusion matrix, ROC, and PR plots."""
    config = load_config(config_path)
    graphs_dir = config["paths"]["results_graphs_dir"]
    metrics_dir = config["paths"]["results_metrics_dir"]
    ensure_dir(graphs_dir)
    ensure_dir(metrics_dir)

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }

    # Confusion matrix.
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax_cm)
    fig_cm.tight_layout()
    fig_cm.savefig(os.path.join(graphs_dir, f"{output_prefix}_confusion_matrix.png"), dpi=200)
    plt.close(fig_cm)

    # ROC curve.
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    fig_roc.tight_layout()
    fig_roc.savefig(os.path.join(graphs_dir, f"{output_prefix}_roc_curve.png"), dpi=200)
    plt.close(fig_roc)

    # Precision-Recall curve.
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(recall, precision)
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    fig_pr.tight_layout()
    fig_pr.savefig(os.path.join(graphs_dir, f"{output_prefix}_precision_recall_curve.png"), dpi=200)
    plt.close(fig_pr)

    metrics_path = os.path.join(metrics_dir, f"{output_prefix}_metrics.csv")
    import pandas as pd

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    LOGGER.info("Saved evaluation outputs with prefix %s", output_prefix)
    return metrics


def main() -> None:
    """Minimal CLI to evaluate from saved arrays in a CSV."""
    parser = argparse.ArgumentParser(description="Evaluate binary prediction CSV")
    parser.add_argument("--csv", required=True, help="CSV with columns: y_true,y_prob")
    parser.add_argument("--prefix", default="model", help="Output file prefix")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    import pandas as pd

    setup_logging()
    df = pd.read_csv(args.csv)
    metrics = evaluate_binary_predictions(df["y_true"].to_numpy(), df["y_prob"].to_numpy(), args.prefix, args.config)
    LOGGER.info("Evaluation ROC-AUC: %.4f", metrics["roc_auc"])


if __name__ == "__main__":
    main()
