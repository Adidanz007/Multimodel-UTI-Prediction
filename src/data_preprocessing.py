"""Clinical data preprocessing module."""

from __future__ import annotations

import argparse
import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def preprocess_clinical_data(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
	"""Clean clinical data and return train/test split."""
	config = load_config(config_path)
	set_global_seed(config["project"]["seed"])

	raw_path = config["paths"]["clinical_raw"]
	cleaned_path = config["paths"]["clinical_cleaned"]
	target_col = config["clinical"]["target_col"]
	drop_columns = config["clinical"].get("drop_columns", [])

	LOGGER.info("Loading clinical dataset from %s", raw_path)
	df = pd.read_csv(raw_path)

	if drop_columns:
		valid_drop = [col for col in drop_columns if col in df.columns]
		df = df.drop(columns=valid_drop)

	before = len(df)
	df = df.drop_duplicates().reset_index(drop=True)
	LOGGER.info("Removed %d duplicate rows", before - len(df))

	if target_col not in df.columns:
		raise ValueError(f"Target column '{target_col}' not found in dataset")

	y = df[target_col]
	X = df.drop(columns=[target_col])

	# Missing value handling split by dtype.
	for col in X.columns:
		if pd.api.types.is_numeric_dtype(X[col]):
			X[col] = X[col].fillna(X[col].median())
		else:
			mode_value = X[col].mode(dropna=True)
			X[col] = X[col].fillna(mode_value.iloc[0] if not mode_value.empty else "unknown")

	cleaned_df = pd.concat([X, y], axis=1)
	ensure_dir(config["paths"]["results_metrics_dir"])
	ensure_dir(config["paths"]["results_graphs_dir"])
	ensure_dir("data/processed")
	cleaned_df.to_csv(cleaned_path, index=False)
	LOGGER.info("Saved cleaned clinical dataset to %s", cleaned_path)

	x_train, x_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=config["clinical"]["test_size"],
		random_state=config["project"]["seed"],
		stratify=y,
	)

	return x_train, x_test, y_train, y_test


def main() -> None:
	"""CLI entrypoint."""
	parser = argparse.ArgumentParser(description="Preprocess clinical data")
	parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
	args = parser.parse_args()

	setup_logging()
	preprocess_clinical_data(args.config)


if __name__ == "__main__":
	main()
