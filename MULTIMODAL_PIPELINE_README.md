# Multimodal UTI Prediction - Quick Reference

## Overview

This project implements a multimodal deep learning pipeline for UTI prediction combining:
- **Ultrasound images** (bladder-focused with DenseNet121)
- **Clinical tabular data** (XGBoost, AUC ~0.92)

## New Files Created

| File | Purpose |
|------|---------|
| `src/multimodal_deep_fusion.py` | Main multimodal fusion model with feature-level fusion |
| `src/ultrasound_pipeline_v2.py` | Improved standalone ultrasound classifier |
| `src/gradcam_multimodal.py` | Enhanced Grad-CAM for multimodal models |
| `run_pipeline.py` | Example usage script |

## Key Improvements

### 1. Ultrasound Preprocessing
- **Cropping**: Removes 15% top, 10% bottom, 10% left/right (UI/borders)
- **CLAHE**: Contrast Limited Adaptive Histogram Equalization
- **Normalization**: [0, 1] range

### 2. Bladder Filtering
- Filters images by keywords: "bladder", "UB", "urinary"
- Excludes: liver, kidney, ovary, prostate, etc.

### 3. Model Architecture
```
Image Input (224x224x3)           Clinical Input (n_features)
        |                                    |
   DenseNet121 (frozen)              Dense(128, relu)
        |                                    |
   GlobalAvgPool                     Dropout + Dense(64)
        |                                    |
   BatchNorm + Dense(256)                    |
        |                                    |
        +----------------+-------------------+
                         |
                    Concatenate
                         |
                  Dense(128, relu)
                         |
                    Dropout(0.5)
                         |
                   Dense(64, relu)
                         |
                  Dense(1, sigmoid)
```

### 4. Two-Stage Training
1. **Stage 1**: Frozen CNN backbone (LR=1e-4, ~20 epochs)
2. **Stage 2**: Fine-tune last 20 layers (LR=1e-5, ~30 epochs)

## Quick Start

### 1. Train Standalone Ultrasound Model
```bash
cd d:/Multimodel-UTI-Prediction
source .venv/Scripts/activate

# Option A: Use the run_pipeline script
python run_pipeline.py --task ultrasound

# Option B: Run module directly
python -m src.ultrasound_pipeline_v2 --image-dir data/raw/ultrasound_images
```

### 2. Train Multimodal Fusion Model
```bash
python run_pipeline.py --task multimodal

# Or directly:
python -m src.multimodal_deep_fusion \
    --image-dir data/raw/ultrasound_images \
    --clinical-model models/clinical_model.pkl \
    --clinical-data data/raw/clinical_dataset.csv
```

### 3. Generate Grad-CAM Visualizations
```bash
python run_pipeline.py --task gradcam

# Or directly:
python -m src.gradcam_multimodal \
    --model models/ultrasound_model_v2.keras \
    --image-dir data/raw/ultrasound_images
```

### 4. Test Preprocessing
```bash
python run_pipeline.py --task preprocess
```

## Configuration

### UltrasoundConfig (ultrasound_pipeline_v2.py)
```python
UltrasoundConfig(
    image_size=(224, 224),
    crop_height=(0.15, 0.90),  # Remove UI
    crop_width=(0.10, 0.90),
    filter_bladder_only=True,
    backbone="DenseNet121",
    batch_size=16,
    stage1_epochs=20,
    stage2_epochs=30,
    stage1_lr=1e-4,
    stage2_lr=1e-5,
    layers_to_unfreeze=20,
    dropout_rate=0.5,
)
```

### TrainingConfig (multimodal_deep_fusion.py)
```python
TrainingConfig(
    image_size=(224, 224),
    batch_size=16,
    stage1_epochs=20,
    stage2_epochs=30,
    stage1_lr=1e-4,
    stage2_lr=1e-5,
    image_feature_dim=256,
    clinical_feature_dim=64,
    fusion_hidden_dim=128,
    dropout_rate=0.5,
    layers_to_unfreeze=20,
)
```

## Expected Outputs

### Model Files
```
models/
├── ultrasound_model_v2.keras       # Standalone ultrasound model
├── multimodal_fusion_model.keras   # Fusion model
├── us_stage1_best.keras            # Stage 1 checkpoint
├── us_stage2_best.keras            # Stage 2 checkpoint
└── fusion_stage1_best.keras        # Fusion stage 1
```

### Results
```
results/
├── ultrasound_v2/
│   ├── us_roc_test.png             # ROC curve
│   ├── us_cm_test.png              # Confusion matrix
│   ├── us_dist_test.png            # Prediction distribution
│   └── us_history.png              # Training history
├── gradcam_v2/
│   ├── normal_00.png               # Grad-CAM for normal
│   ├── abnormal_00.png             # Grad-CAM for abnormal
│   └── summary.png                 # Summary grid
└── fusion_*.png                    # Fusion model results
```

## Debugging Checklist

### Model Collapse Detection
The pipeline automatically checks for:
- Mean prediction (~0.3-0.7 expected)
- Prediction std (>0.1 expected)
- Both classes being predicted

If you see "MODEL COLLAPSED" warning:
1. Increase class weights for minority class
2. Use stronger augmentation
3. Reduce learning rate
4. Check data quality

### Grad-CAM Attention Analysis
Check that `center_to_edge_ratio > 1.5`:
- **>1.5**: Good - model focuses on center (bladder)
- **<0.7**: Warning - model focuses on edges (artifacts)

## Data Requirements

### Image Directory Structure
```
data/raw/ultrasound_images/
├── normal/
│   ├── normal_0001.jpg
│   ├── normal_0002.jpg
│   └── ...
└── abnormal/
    ├── abnormal_0001.jpg
    └── ...
```

### Clinical Data
CSV with columns including: `age`, `gender`, urinalysis features, symptoms, etc.

## Important Notes

1. **Bladder Filtering**: Current dataset doesn't have organ labels in filenames. The filter is ready for when you have properly labeled data.

2. **Patient Alignment**: In production, each image should have a patient_id mapping to clinical data. The current implementation samples clinical data randomly for demonstration.

3. **Class Imbalance**: Class weights are computed automatically. For severe imbalance (>3:1), consider SMOTE or focal loss.

4. **GPU Memory**: Reduce batch_size to 8 if you encounter OOM errors.

## Troubleshooting

### Import Errors
```bash
# Activate virtual environment
source .venv/Scripts/activate

# Or on Windows CMD:
.venv\Scripts\activate
```

### CUDA/GPU Issues
```python
# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

### Missing Dependencies
```bash
pip install tensorflow opencv-python albumentations pandas scikit-learn matplotlib seaborn
```
