"""CLI wrapper for binary evaluation plots and metrics."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.evaluation import evaluate_binary_predictions
from src.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
	"""Evaluate saved predictions from CSV containing y_true and y_prob."""
	parser = argparse.ArgumentParser(description="Evaluate model predictions")
	parser.add_argument("--csv", required=True, help="Path to prediction CSV with y_true,y_prob")
	parser.add_argument("--prefix", default="model", help="Output prefix")
	parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
	args = parser.parse_args()

	setup_logging()
	df = pd.read_csv(args.csv)
	metrics = evaluate_binary_predictions(
		y_true=df["y_true"].to_numpy(),
		y_prob=df["y_prob"].to_numpy(),
		output_prefix=args.prefix,
		config_path=args.config,
	)
	LOGGER.info("Evaluation completed. ROC-AUC %.4f", metrics["roc_auc"])


if __name__ == "__main__":
	main()
