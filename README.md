# Multimodal UTI Prediction

A multimodal AI system for UTI risk prediction combining clinical
biomarkers and bladder ultrasound images.

## Model Performance
| Model | AUC | Accuracy |
|-------|-----|----------|
| Clinical (XGBoost) | 0.8105 | 74.0% |
| Image (EfficientNetB3) | 0.8843 | 80.3% |
| **Fusion (Calibrated LR)** | **0.9145** | **84.2%** |

## Project Structure
project/
├── data/processed/        # Cleaned datasets
├── models/                # Saved model files
├── results/graphs/        # All visualizations
├── src/                   # All source scripts
└── webapp/                # Flask web application (next phase)

## Setup
```bash
pip install -r requirements.txt
python src/predict_final.py   # Test inference
```