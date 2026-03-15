"""Phase 1 training orchestrator."""

from __future__ import annotations

import argparse
import logging

from src.clinical_model_training import train_clinical_models
from src.multimodal_fusion import train_fusion_models
from src.ultrasound_model_training import train_ultrasound_model
from src.utils import setup_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
	"""Run clinical and ultrasound training, optionally fusion training."""
	parser = argparse.ArgumentParser(description="Train multimodal UTI models")
	parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
	parser.add_argument(
		"--fusion-csv",
		default=None,
		help="Optional aligned csv with columns clinical_prob, ultrasound_prob, label",
	)
	args = parser.parse_args()

	setup_logging()

	clinical_result = train_clinical_models(args.config)
	LOGGER.info("Clinical training finished with best model %s", clinical_result["best_model_name"])

	ultrasound_result = train_ultrasound_model(args.config)
	LOGGER.info("Ultrasound training finished with ROC-AUC %.4f", ultrasound_result["test_roc_auc"])

	if args.fusion_csv:
		fusion_result = train_fusion_models(args.fusion_csv, args.config)
		LOGGER.info("Fusion training finished with stacked ROC-AUC %.4f", fusion_result["stacked_roc_auc"])
	else:
		LOGGER.info("Fusion training skipped. Provide --fusion-csv to train fusion model.")


if __name__ == "__main__":
	main()
