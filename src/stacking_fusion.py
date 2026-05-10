"""
stacking_fusion.py
===================
Improvement 2 & 3: Stacking ensemble fusion model with calibrated probability outputs.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMB_DIR    = os.path.join(BASE_DIR, "results", "embeddings")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")
FUSION_CSV = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")

os.makedirs(GRAPHS_DIR, exist_ok=True)

def main():
    print("=" * 60)
    print("  IMPROVEMENT 2 & 3 — Stacking Ensemble & Calibration")
    print("=" * 60)

    # 1. Load Data
    fusion_pairs = pd.read_csv(FUSION_CSV)
    labels = fusion_pairs["label"].values if "label" in fusion_pairs.columns else np.zeros(len(fusion_pairs), dtype=int)

    # Make sure we load the engineered clinical embeddings if they exist
    clin_proba_path = os.path.join(EMB_DIR, "clinical_proba_engineered.npy")
    clin_feat_path  = os.path.join(EMB_DIR, "clinical_features_engineered.npy")

    if not os.path.exists(clin_proba_path):
        print("  ⚠ clinical_proba_engineered.npy not found! Falling back to 4k embeddings.")
        clin_proba_path = os.path.join(EMB_DIR, "clinical_proba_4k.npy")
        clin_feat_path  = os.path.join(EMB_DIR, "clinical_features_4k.npy")

    clin_proba    = np.load(clin_proba_path)
    clin_features = np.load(clin_feat_path)
    img_proba     = np.load(os.path.join(EMB_DIR, "image_proba.npy"))
    img_embed     = np.load(os.path.join(EMB_DIR, "image_embeddings_fixed.npy"))

    # PCA for image
    pca = joblib.load(os.path.join(MODELS_DIR, "fusion_pca.pkl"))
    imgs_pca_32 = pca.transform(img_embed)

    # 2. Build full feature matrix
    X_full = np.column_stack([
        clin_proba[:, 1:2],                   # clinical P(infected)
        img_proba,                            # image P(abnormal)
        clin_proba[:, 1:2] * img_proba,       # interaction
        imgs_pca_32,                          # PCA image embeddings (32)
        clin_features,                        # clinical features (31 or 41)
    ])
    print(f"Full fusion feature matrix: {X_full.shape}")

    # 3. Splits
    if "split" in fusion_pairs.columns:
        train_idx = fusion_pairs[fusion_pairs["split"] == "train"].index.tolist()
        val_idx   = fusion_pairs[fusion_pairs["split"] == "val"].index.tolist()
        test_idx  = fusion_pairs[fusion_pairs["split"] == "test"].index.tolist()
    else:
        idx = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    X_train_full = X_full[train_idx]
    y_train      = labels[train_idx]
    X_val_full   = X_full[val_idx]
    y_val        = labels[val_idx]
    X_test_full  = X_full[test_idx]
    y_test       = labels[test_idx]

    # 4. Define base models
    base_models = [
        ('lr',   LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)),
        ('rf',   RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=SEED)),
        ('xgb',  XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_alpha=0.5, reg_lambda=2.0,
                               eval_metric='logloss', random_state=SEED)),
        ('gb',   GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=SEED)),
        ('svm',  SVC(probability=True, kernel='rbf', C=1.0, random_state=SEED)),
    ]

    # 5. Out-of-fold predictions
    print("\nGenerating out-of-fold predictions for meta-learner...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_predictions = np.zeros((len(X_train_full), len(base_models)))

    for fold_idx, (fold_train, fold_val) in enumerate(skf.split(X_train_full, y_train)):
        print(f"  Fold {fold_idx+1}/5 ...")
        X_fold_train, X_fold_val = X_train_full[fold_train], X_train_full[fold_val]
        y_fold_train = y_train[fold_train]

        for model_idx, (name, model) in enumerate(base_models):
            model.fit(X_fold_train, y_fold_train)
            oof_predictions[fold_val, model_idx] = model.predict_proba(X_fold_val)[:, 1]

    print(f"  OOF predictions shape: {oof_predictions.shape}")

    # 6. Train base models on full training set
    print("\nTraining base models on full train split...")
    test_predictions = np.zeros((len(X_test_full), len(base_models)))
    val_predictions  = np.zeros((len(X_val_full), len(base_models)))

    for model_idx, (name, model) in enumerate(base_models):
        model.fit(X_train_full, y_train)
        test_predictions[:, model_idx] = model.predict_proba(X_test_full)[:, 1]
        val_predictions[:, model_idx]  = model.predict_proba(X_val_full)[:, 1]
        auc = roc_auc_score(y_test, test_predictions[:, model_idx])
        print(f"  Base model [{name}] Test AUC: {auc:.4f}")

    # 7. Train Meta-Learner
    meta_learner = LogisticRegression(C=0.1, random_state=SEED)
    meta_learner.fit(oof_predictions, y_train)

    stacking_proba = meta_learner.predict_proba(test_predictions)[:, 1]
    stacking_auc   = roc_auc_score(y_test, stacking_proba)
    stacking_acc   = accuracy_score(y_test, (stacking_proba >= 0.50).astype(int))

    print(f"\n  STACKING ENSEMBLE — Test AUC: {stacking_auc:.4f} | Acc: {stacking_acc:.4f}")

    # 8. Save models
    for name, model in base_models:
        joblib.dump(model, os.path.join(MODELS_DIR, f'stack_base_{name}.pkl'))
    joblib.dump(meta_learner, os.path.join(MODELS_DIR, 'stack_meta_learner.pkl'))

    # ─────────────────────────────────────────────────────────────────────────────
    # IMPROVEMENT 3: Calibrate the Current Best Model
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[IMPROVEMENT 3] Calibrating the Logistic Regression Fusion Model...")
    lr_model = joblib.load(os.path.join(MODELS_DIR, "fusion_strategy_A.pkl"))
    
    # We need X_simple for LR
    X_simple = np.column_stack([clin_proba[:, 1], img_proba.flatten(), clin_proba[:, 1] * img_proba.flatten()])
    X_train_simple = X_simple[train_idx]
    X_val_simple   = X_simple[val_idx]
    X_test_simple  = X_simple[test_idx]

    # Combine train and val for cv=5 calibration
    X_calib_train = np.vstack([X_train_simple, X_val_simple])
    y_calib_train = np.concatenate([y_train, y_val])

    calibrated_fusion = CalibratedClassifierCV(
        estimator=LogisticRegression(C=1.0, random_state=SEED),
        method='isotonic',
        cv=5
    )
    calibrated_fusion.fit(X_calib_train, y_calib_train)

    cal_proba = calibrated_fusion.predict_proba(X_test_simple)[:, 1]
    cal_auc   = roc_auc_score(y_test, cal_proba)
    cal_acc   = accuracy_score(y_test, (cal_proba >= 0.50).astype(int))
    cal_f1    = f1_score(y_test, (cal_proba >= 0.50).astype(int))
    print(f"  Calibrated LR Fusion — Test AUC: {cal_auc:.4f} | Acc: {cal_acc:.4f} | F1: {cal_f1:.4f}")

    joblib.dump(calibrated_fusion, os.path.join(MODELS_DIR, 'fusion_model_calibrated.pkl'))

    # Save calibration chart
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0D1117')
    ax.set_facecolor('#0D1117')

    fusion_proba_test = lr_model.predict_proba(X_test_simple)[:, 1]

    prob_true_orig, prob_pred_orig = calibration_curve(y_test, fusion_proba_test, n_bins=10)
    prob_true_cal, prob_pred_cal   = calibration_curve(y_test, cal_proba, n_bins=10)

    ax.plot([0,1],[0,1], color='#555555', linestyle='--', label='Perfect calibration')
    ax.plot(prob_pred_orig, prob_true_orig, color='#F5A623', marker='o', linewidth=2, label='Original LR Fusion')
    ax.plot(prob_pred_cal,  prob_true_cal,  color='#00A896', marker='s', linewidth=2, label='Calibrated LR Fusion')
    ax.set_xlabel('Mean predicted probability', color='white', fontsize=11)
    ax.set_ylabel('Fraction of positives', color='white', fontsize=11)
    ax.set_title('Calibration Curve — Reliability Diagram', color='white', fontsize=12, fontweight='bold')
    ax.legend(facecolor='#1C2333', labelcolor='white', fontsize=10)
    ax.tick_params(colors='white')
    ax.spines[:].set_color('#333333')

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, 'calibration_curve.png'), dpi=150, facecolor='#0D1117', bbox_inches='tight')
    print("  Saved: calibration_curve.png")

    print("\n  Complete! Please run generate_visualizations.py to update the final graphs and report.")

if __name__ == "__main__":
    main()
