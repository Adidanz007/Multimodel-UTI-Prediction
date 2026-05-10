"""SHAP explainability utilities for clinical model."""

from __future__ import annotations

import argparse
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_preprocessing import preprocess_clinical_data
from src.utils import ensure_dir, load_config, set_global_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def run_shap_explainability(config_path: str = "config/config.yaml") -> None:
    """Generate SHAP summary and feature importance plots."""
    config = load_config(config_path)
    set_global_seed(int(config["project"]["seed"]))

    _, x_test, _, _ = preprocess_clinical_data(config_path)

    model_payload = joblib.load(config["clinical"]["model_output"])
    model = model_payload["model"]
    selected_features = model_payload["selected_features"]

    x_eval = x_test[selected_features].copy()
    sample_size = min(int(config["explainability"]["shap_sample_size"]), len(x_eval))
    x_sample = x_eval.sample(n=sample_size, random_state=int(config["project"]["seed"]))

    transformed = model.named_steps["preprocessor"].transform(x_sample)
    transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    estimator = model.named_steps["model"]

    estimator_name = estimator.__class__.__name__.lower()
    is_tree_model = (
        "forest" in estimator_name
        or "xgb" in estimator_name
        or "tree" in estimator_name
        or "boost" in estimator_name
    )

    if is_tree_model:
        explainer = shap.TreeExplainer(estimator)
        raw_values = explainer.shap_values(transformed_dense)

        if isinstance(raw_values, list):
            values = raw_values[-1]
        else:
            values = raw_values

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, tuple, np.ndarray)):
            base_value = float(np.array(expected_value).reshape(-1)[-1])
        else:
            base_value = float(expected_value)

        shap_values = shap.Explanation(
            values=values,
            base_values=np.full(transformed_dense.shape[0], base_value),
            data=transformed_dense,
            feature_names=feature_names,
        )
    else:
        def predict_positive(x: np.ndarray) -> np.ndarray:
            return estimator.predict_proba(x)[:, 1]

        explainer = shap.Explainer(predict_positive, transformed_dense, feature_names=feature_names)
        shap_values = explainer(transformed_dense)

    graphs_dir = config["paths"]["results_graphs_dir"]
    ensure_dir(graphs_dir)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, features=transformed_dense, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "shap_summary_plot.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "shap_feature_importance_plot.png"), dpi=200)
    plt.close()

    # Save one local explanation as waterfall plot.
    shap.plots.waterfall(shap_values[0], max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_dir, "shap_individual_explanation.png"), dpi=200)
    plt.close()

    LOGGER.info("Saved SHAP explainability plots to %s", graphs_dir)


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Run SHAP explainability for clinical model")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    run_shap_explainability(args.config)


if __name__ == "__main__":
    main()
