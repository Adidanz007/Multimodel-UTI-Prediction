"""Feature engineering and selection for clinical tabular model."""

from __future__ import annotations

import logging
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
	"""Build preprocessor for numeric/categorical columns."""
	categorical_cols = [col for col in x.columns if x[col].dtype == "object"]
	numerical_cols = [col for col in x.columns if col not in categorical_cols]

	preprocessor = ColumnTransformer(
		transformers=[
			("num", StandardScaler(), numerical_cols),
			("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
		]
	)
	return preprocessor


def select_top_features(
	x_train: pd.DataFrame,
	y_train: pd.Series,
	top_k: int,
) -> Tuple[pd.DataFrame, List[str]]:
	"""Select top-k original features using random forest importance."""
	encoded = pd.get_dummies(x_train, drop_first=False)
	model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
	model.fit(encoded, y_train)

	importances = pd.Series(model.feature_importances_, index=encoded.columns)

	# Aggregate dummy-column importances back to original feature names.
	aggregated_importance = {}
	for col in x_train.columns:
		prefix = f"{col}_"
		mask = importances.index == col
		mask |= importances.index.str.startswith(prefix)
		aggregated_importance[col] = float(importances[mask].sum())

	grouped = pd.Series(aggregated_importance).sort_values(ascending=False)
	selected = grouped.head(min(top_k, len(grouped))).index.tolist()

	valid_selected = [col for col in selected if col in x_train.columns]
	if not valid_selected:
		LOGGER.warning("Feature selection returned no original columns; using full set")
		return x_train, list(x_train.columns)

	LOGGER.info("Selected %d features", len(valid_selected))
	return x_train[valid_selected].copy(), valid_selected
