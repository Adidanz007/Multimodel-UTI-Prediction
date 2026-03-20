"""
Example Usage: Multimodal UTI Prediction Pipeline

This script demonstrates how to use the improved multimodal fusion pipeline
for UTI prediction from ultrasound images and clinical data.

Run sections individually based on your needs:
1. Train standalone ultrasound model
2. Train multimodal fusion model
3. Generate Grad-CAM visualizations
4. Make predictions with trained model

Author: Multimodal UTI Prediction Project
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import setup_logging

setup_logging()


# =============================================================================
# 1. TRAIN STANDALONE ULTRASOUND MODEL
# =============================================================================

def train_ultrasound_only():
    """
    Train the improved ultrasound image classifier.

    This trains ONLY the image model (no clinical data).
    Use this to:
    - Verify the image preprocessing is working
    - Get a baseline image-only performance
    - Debug the image pipeline before multimodal fusion
    """
    from src.ultrasound_pipeline_v2 import (
        train_ultrasound_pipeline,
        UltrasoundConfig,
    )

    # Configuration
    config = UltrasoundConfig(
        # Image settings
        image_size=(224, 224),
        crop_height=(0.15, 0.90),  # Remove top 15%, bottom 10% (UI/borders)
        crop_width=(0.10, 0.90),   # Remove left/right 10% (UI/borders)

        # Dataset
        filter_bladder_only=True,  # IMPORTANT: Only use bladder images
        test_size=0.15,
        val_size=0.15,

        # Training
        backbone="DenseNet121",    # Better for medical imaging
        batch_size=16,
        stage1_epochs=20,          # Frozen backbone
        stage2_epochs=30,          # Fine-tuning
        stage1_lr=1e-4,
        stage2_lr=1e-5,
        layers_to_unfreeze=20,     # Unfreeze last 20 layers of DenseNet

        # Regularization
        dropout_rate=0.5,

        # Seed
        seed=42,
    )

    # Train
    results = train_ultrasound_pipeline(
        image_dir="data/raw/ultrasound_images",
        output_dir="models",
        results_dir="results/ultrasound_v2",
        config=config,
    )

    print("\n" + "=" * 60)
    print("ULTRASOUND MODEL TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {results['model_path']}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")
    print("=" * 60)

    return results


# =============================================================================
# 2. TRAIN MULTIMODAL FUSION MODEL
# =============================================================================

def train_multimodal():
    """
    Train the multimodal fusion model combining:
    - Ultrasound images (DenseNet121 backbone)
    - Clinical features (from existing XGBoost model)

    This is the MAIN model that should outperform either modality alone.
    """
    from src.multimodal_deep_fusion import (
        train_pipeline,
        TrainingConfig,
    )

    # Configuration
    config = TrainingConfig(
        # Image settings
        image_size=(224, 224),

        # Training
        batch_size=16,
        stage1_epochs=20,          # Frozen CNN + clinical features
        stage2_epochs=30,          # Fine-tune CNN
        stage1_lr=1e-4,
        stage2_lr=1e-5,

        # Architecture
        image_feature_dim=256,     # CNN output dimension
        clinical_feature_dim=64,   # Clinical branch output
        fusion_hidden_dim=128,     # Fusion layer size
        dropout_rate=0.5,

        # Fine-tuning
        layers_to_unfreeze=20,

        # Callbacks
        early_stopping_patience=10,
        reduce_lr_patience=5,
    )

    # Train
    results = train_pipeline(
        image_dir="data/raw/ultrasound_images",
        clinical_model_path="models/clinical_model.pkl",
        clinical_data_path="data/raw/clinical_dataset.csv",
        output_dir="models",
        config=config,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("MULTIMODAL FUSION TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model saved: {results['model_path']}")
    print(f"Test ROC-AUC: {results['test_metrics']['roc_auc']:.4f}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1: {results['test_metrics']['f1']:.4f}")
    print(f"Best Validation AUC: {results['best_val_auc']:.4f}")
    print("=" * 60)

    return results


# =============================================================================
# 3. GENERATE GRAD-CAM VISUALIZATIONS
# =============================================================================

def generate_gradcam():
    """
    Generate Grad-CAM visualizations to verify model attention.

    This helps ensure the model focuses on:
    - The bladder region (correct anatomical location)
    - NOT on artifacts, UI elements, or text overlays
    """
    import numpy as np
    import tensorflow as tf

    from src.gradcam_multimodal import (
        visualize_batch,
        analyze_model_attention,
    )
    from src.ultrasound_pipeline_v2 import load_and_filter_dataset

    # Load model
    model_path = "models/ultrasound_model_v2.keras"
    print(f"Loading model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # Load dataset
    paths, labels, _ = load_and_filter_dataset(
        "data/raw/ultrasound_images",
        filter_bladder=True,
    )

    # Generate visualizations
    visualize_batch(
        model=model,
        image_paths=paths,
        labels=np.array(labels),
        save_dir="results/gradcam_v2",
        n_per_class=4,
    )

    # Analyze attention patterns
    results = analyze_model_attention(
        model=model,
        image_paths=paths,
        labels=np.array(labels),
        n_samples=50,
    )

    print("\n" + "=" * 60)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Output directory: results/gradcam_v2")
    print(f"Center/Edge attention ratio: {results['center_to_edge_ratio']:.2f}")
    if results['center_to_edge_ratio'] > 1.5:
        print("Model focuses on CENTER (good for bladder detection)")
    elif results['center_to_edge_ratio'] < 0.7:
        print("WARNING: Model focuses on EDGES (possible artifact issue)")
    print("=" * 60)

    return results


# =============================================================================
# 4. MAKE PREDICTIONS
# =============================================================================

def predict_single_sample():
    """
    Make prediction for a single sample using the multimodal model.
    """
    import numpy as np
    import pandas as pd
    import tensorflow as tf

    from src.multimodal_deep_fusion import (
        preprocess_ultrasound,
        ClinicalFeatureExtractor,
    )

    # Load models
    fusion_model = tf.keras.models.load_model("models/multimodal_fusion_model.keras")
    clinical_extractor = ClinicalFeatureExtractor("models/clinical_model.pkl")

    # Example: load a sample image
    image_path = "data/raw/ultrasound_images/abnormal/abnormal_0001.jpg"
    image = preprocess_ultrasound(image_path, target_size=(224, 224))

    # Example: load clinical data for this patient
    # In real usage, you would have this data for the specific patient
    clinical_df = pd.read_csv("data/raw/clinical_dataset.csv")
    sample_clinical = clinical_df.iloc[0:1]  # First patient as example

    # Extract clinical features
    clinical_features = clinical_extractor.extract_features(sample_clinical)

    # Prepare inputs
    image_batch = np.expand_dims(image, axis=0)

    # Predict
    prediction = fusion_model.predict([image_batch, clinical_features], verbose=0)[0, 0]

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Probability of UTI (abnormal): {prediction:.4f}")
    print(f"Predicted class: {'ABNORMAL (UTI)' if prediction > 0.5 else 'NORMAL'}")
    print(f"Confidence: {max(prediction, 1-prediction):.1%}")
    print("=" * 60)

    return prediction


# =============================================================================
# 5. QUICK PREPROCESSING TEST
# =============================================================================

def test_preprocessing():
    """
    Test the preprocessing pipeline on a sample image.

    Useful for verifying cropping and CLAHE enhancement work correctly.
    """
    import matplotlib.pyplot as plt
    import cv2

    from src.multimodal_deep_fusion import preprocess_ultrasound

    # Load sample image
    image_path = "data/raw/ultrasound_images/normal/normal_0001.jpg"

    # Original
    original = cv2.imread(image_path)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    # Preprocessed
    preprocessed = preprocess_ultrasound(image_path)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(preprocessed)
    axes[1].set_title("Preprocessed (Cropped + CLAHE)", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.suptitle("Ultrasound Preprocessing Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/preprocessing_test.png", dpi=300)
    plt.close()

    print("Preprocessing test saved to: results/preprocessing_test.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal UTI Prediction Pipeline")
    parser.add_argument(
        "--task",
        choices=["ultrasound", "multimodal", "gradcam", "predict", "preprocess"],
        default="ultrasound",
        help="Task to run"
    )

    args = parser.parse_args()

    if args.task == "ultrasound":
        print("\nTraining standalone ultrasound model...")
        train_ultrasound_only()

    elif args.task == "multimodal":
        print("\nTraining multimodal fusion model...")
        train_multimodal()

    elif args.task == "gradcam":
        print("\nGenerating Grad-CAM visualizations...")
        generate_gradcam()

    elif args.task == "predict":
        print("\nMaking prediction...")
        predict_single_sample()

    elif args.task == "preprocess":
        print("\nTesting preprocessing...")
        test_preprocessing()
