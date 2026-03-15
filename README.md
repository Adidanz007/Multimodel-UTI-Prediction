# Multimodal UTI Prediction

Research-grade multimodal AI pipeline for predicting urinary tract infection (UTI) from:

1. Clinical tabular data
2. Bladder ultrasound images

This repository implements Phase 1 of the project (model development).

## Project Structure

```
data/
	raw/
		clinical_dataset.csv
		ultrasound_images/
	processed/
		clinical_cleaned.csv
		ultrasound_split/
src/
	data_preprocessing.py
	feature_engineering.py
	clinical_model_training.py
	ultrasound_model_training.py
	multimodal_fusion.py
	evaluation.py
	prediction_pipeline.py
	explainability.py
	gradcam_visualization.py
	train_model.py
models/
results/
config/
	config.yaml
```

## Phase 1 Capabilities

- Clinical ML pipeline:
	- Missing value handling, duplicate removal, feature selection
	- Logistic Regression, Random Forest, XGBoost
	- 5-fold stratified CV + hyperparameter tuning
	- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC

- Ultrasound DL pipeline:
	- Stratified train/val/test split
	- Transfer learning (EfficientNetB0/ResNet50/MobileNetV2)
	- Albumentations data augmentation (train only)
	- Class weights for imbalance handling
	- Early stopping and LR scheduling

- Multimodal fusion:
	- Weighted averaging
	- Stacked ensemble

- Explainability:
	- SHAP for clinical model
	- Grad-CAM for ultrasound model

- Engineering standards:
	- Config-driven pipeline (`config/config.yaml`)
	- Reproducibility with global seeds
	- Structured logging
	- MLflow experiment tracking

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` to control:

- Dataset paths
- Training parameters
- Augmentation settings
- Fusion weights
- Explainability settings

## Run Training

Train clinical and ultrasound models:

```bash
python -m src.train_model --config config/config.yaml
```

Train fusion model (requires aligned prediction CSV with columns `clinical_prob`, `ultrasound_prob`, `label`):

```bash
python -m src.train_model --config config/config.yaml --fusion-csv path/to/fusion_predictions.csv
```

## Run Explainability

```bash
python -m src.explainability --config config/config.yaml
python -m src.gradcam_visualization --config config/config.yaml
```

## Evaluate Predictions

Evaluate a prediction table with columns `y_true`, `y_prob`:

```bash
python -m src.evaluate_model --csv path/to/predictions.csv --prefix clinical
```

## Run Inference

```bash
python -m src.prediction_pipeline --clinical-csv path/to/one_row.csv --image path/to/image.png
```