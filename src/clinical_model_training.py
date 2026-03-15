"""Train classical ML models on clinical tabular data with CV and tuning."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.data_preprocessing import preprocess_clinical_data
from src.feature_engineering import build_preprocessor, select_top_features
from src.utils import end_run_timer, ensure_dir, load_config, set_global_seed, setup_logging, start_run_timer

LOGGER = logging.getLogger(__name__)


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


def _build_candidate_models(seed: int) -> Dict[str, Tuple[Any, Dict[str, List[Any]]]]:
    candidates: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, n_jobs=None),
            {
                "model__C": np.logspace(-3, 2, 20),
                "model__solver": ["liblinear", "lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "model__n_estimators": [200, 400, 600],
                "model__max_depth": [None, 8, 16, 24],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost"] = (
            XGBClassifier(
                random_state=seed,
                n_estimators=300,
                learning_rate=0.05,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=-1,
            ),
            {
                "model__n_estimators": [200, 300, 500],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.01, 0.05, 0.1],
                "model__subsample": [0.7, 0.9, 1.0],
                "model__colsample_bytree": [0.7, 0.9, 1.0],
            },
        )
    return candidates


def train_clinical_models(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Train and compare clinical models with 5-fold CV and tuning."""
    config = load_config(config_path)
    set_global_seed(config["project"]["seed"])

    ensure_dir(config["paths"]["models_dir"])
    ensure_dir(config["paths"]["results_metrics_dir"])

    mlflow.set_experiment(config["project"]["mlflow_experiment"])

    x_train, x_test, y_train, y_test = preprocess_clinical_data(config_path)
    x_train_selected, selected_features = select_top_features(
        x_train,
        y_train,
        top_k=config["clinical"]["feature_selection_top_k"],
    )
    x_test_selected = x_test[selected_features].copy()

    preprocessor = build_preprocessor(x_train_selected)
    cv = StratifiedKFold(
        n_splits=config["clinical"]["cv_folds"],
        shuffle=True,
        random_state=config["project"]["seed"],
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    candidates = _build_candidate_models(config["project"]["seed"])
    benchmark_rows: List[Dict[str, Any]] = []

    best_model_name = None
    best_model_pipeline = None
    best_model_auc = -1.0

    for model_name, (model, search_space) in candidates.items():
        start_ts = start_run_timer()
        with mlflow.start_run(run_name=f"clinical_{model_name}"):
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", model),
                ]
            )

            tuner = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=search_space,
                n_iter=config["clinical"]["randomized_search_iter"],
                scoring=config["clinical"]["scoring"],
                cv=cv,
                random_state=config["project"]["seed"],
                n_jobs=-1,
                refit=True,
                verbose=1,
            )
            tuner.fit(x_train_selected, y_train)

            best_estimator = tuner.best_estimator_
            cv_scores = cross_validate(
                best_estimator,
                x_train_selected,
                y_train,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
            )

            y_pred = best_estimator.predict(x_test_selected)
            y_prob = best_estimator.predict_proba(x_test_selected)[:, 1]

            test_metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }

            elapsed = end_run_timer(start_ts)
            mlflow.log_params({
                "model": model_name,
                "selected_features": len(selected_features),
                "cv_folds": config["clinical"]["cv_folds"],
                "search_iterations": config["clinical"]["randomized_search_iter"],
                **tuner.best_params_,
            })
            mlflow.log_metrics({
                "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
                "cv_precision_mean": float(np.mean(cv_scores["test_precision"])),
                "cv_recall_mean": float(np.mean(cv_scores["test_recall"])),
                "cv_f1_mean": float(np.mean(cv_scores["test_f1"])),
                "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
                "test_accuracy": test_metrics["accuracy"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_f1": test_metrics["f1"],
                "test_roc_auc": test_metrics["roc_auc"],
                "training_seconds": elapsed,
            })

            benchmark_rows.append(
                {
                    "model": model_name,
                    "cv_accuracy": float(np.mean(cv_scores["test_accuracy"])),
                    "cv_precision": float(np.mean(cv_scores["test_precision"])),
                    "cv_recall": float(np.mean(cv_scores["test_recall"])),
                    "cv_f1": float(np.mean(cv_scores["test_f1"])),
                    "cv_roc_auc": float(np.mean(cv_scores["test_roc_auc"])),
                    "test_accuracy": test_metrics["accuracy"],
                    "test_precision": test_metrics["precision"],
                    "test_recall": test_metrics["recall"],
                    "test_f1": test_metrics["f1"],
                    "test_roc_auc": test_metrics["roc_auc"],
                    "best_params": tuner.best_params_,
                }
            )

            if test_metrics["roc_auc"] > best_model_auc:
                best_model_auc = test_metrics["roc_auc"]
                best_model_name = model_name
                best_model_pipeline = best_estimator

            LOGGER.info("Completed %s with test ROC-AUC %.4f", model_name, test_metrics["roc_auc"])

    if best_model_pipeline is None:
        raise RuntimeError("No clinical model trained successfully")

    model_output = config["clinical"]["model_output"]
    payload = {
        "model": best_model_pipeline,
        "selected_features": selected_features,
        "model_name": best_model_name,
    }
    joblib.dump(payload, model_output)
    LOGGER.info("Saved best clinical model to %s", model_output)

    benchmark_df = pd.DataFrame(benchmark_rows).sort_values(by="test_roc_auc", ascending=False)
    benchmark_path = os.path.join(config["paths"]["results_metrics_dir"], "clinical_model_benchmark.csv")
    benchmark_df.to_csv(benchmark_path, index=False)

    return {
        "best_model_name": best_model_name,
        "best_model_roc_auc": best_model_auc,
        "benchmark_path": benchmark_path,
        "model_path": model_output,
    }


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Train clinical ML models")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    setup_logging()
    result = train_clinical_models(args.config)
    LOGGER.info("Best clinical model: %s (ROC-AUC: %.4f)", result["best_model_name"], result["best_model_roc_auc"])


if __name__ == "__main__":
    main()
