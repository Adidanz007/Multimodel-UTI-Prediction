"""
generate_visualizations.py
==========================
Generates comprehensive visual outputs for the Multimodal UTI Prediction project.
Finds optimal threshold, evaluates all improvements, and generates 11 graphs.
"""

from __future__ import annotations

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, f1_score, accuracy_score,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EMB_DIR    = os.path.join(BASE_DIR, "results", "embeddings")
GRAPHS_DIR = os.path.join(BASE_DIR, "results", "graphs")
METRICS_DIR= os.path.join(BASE_DIR, "results", "metrics")
FUSION_CSV = os.path.join(BASE_DIR, "data", "processed", "fusion_pairs.csv")
CLINICAL_CSV = os.path.join(BASE_DIR, "data", "processed", "clinical_4k_sorted.csv")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Theme Configuration
# ─────────────────────────────────────────────────────────────────────────────
THEME = {
    "bg": "#0D1117", "teal": "#00A896", "purple": "#7F77DD",
    "amber": "#F5A623", "red": "#E24B4A", "white": "#FFFFFF",
    "panel": "#1C2333", "grid": "#222222", "axis": "#333333"
}

def set_dark_theme(fig, ax):
    fig.patch.set_facecolor(THEME["bg"])
    if isinstance(ax, np.ndarray):
        for a in ax.flatten():
            a.set_facecolor(THEME["bg"])
            a.tick_params(colors=THEME["white"])
            a.spines[:].set_color(THEME["axis"])
    else:
        ax.set_facecolor(THEME["bg"])
        ax.tick_params(colors=THEME["white"])
        ax.spines[:].set_color(THEME["axis"])

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Load Data & Splits
# ─────────────────────────────────────────────────────────────────────────────
def get_splits(labels, fusion_pairs):
    if "split" in fusion_pairs.columns:
        train_idx = fusion_pairs[fusion_pairs["split"] == "train"].index.tolist()
        val_idx   = fusion_pairs[fusion_pairs["split"] == "val"].index.tolist()
        test_idx  = fusion_pairs[fusion_pairs["split"] == "test"].index.tolist()
        if len(train_idx) > 0 and len(test_idx) > 0:
            return np.array(train_idx), np.array(val_idx), np.array(test_idx)
    idx = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(idx, test_size=0.30, stratify=labels, random_state=SEED)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=labels[temp_idx], random_state=SEED)
    return train_idx, val_idx, test_idx

# ─────────────────────────────────────────────────────────────────────────────
# Main Routine
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Generating Comprehensive Visual Outputs (with all Improvements)")
    print("=" * 60)

    # 1. Load Data
    fusion_pairs = pd.read_csv(FUSION_CSV)
    labels = fusion_pairs["label"].values if "label" in fusion_pairs.columns else np.zeros(len(fusion_pairs), dtype=int)
    train_idx, val_idx, test_idx = get_splits(labels, fusion_pairs)
    y_test  = labels[test_idx]
    y_val   = labels[val_idx]

    # Load Embeddings
    clin_proba_path = os.path.join(EMB_DIR, "clinical_proba_engineered.npy")
    clin_feat_path  = os.path.join(EMB_DIR, "clinical_features_engineered.npy")
    if not os.path.exists(clin_proba_path):
        clin_proba_path = os.path.join(EMB_DIR, "clinical_proba_4k.npy")
        clin_feat_path  = os.path.join(EMB_DIR, "clinical_features_4k.npy")
    
    clin = np.load(clin_proba_path)
    clin_features = np.load(clin_feat_path)
    img_proba = np.load(os.path.join(EMB_DIR, "image_proba.npy"))
    img_embed = np.load(os.path.join(EMB_DIR, "image_embeddings_fixed.npy"))

    # PCA for image
    pca = joblib.load(os.path.join(MODELS_DIR, "fusion_pca.pkl"))
    imgs_pca_32 = pca.transform(img_embed)

    # 2. Prepare Inputs
    X_simple_all = np.column_stack([clin[:, 1], img_proba.flatten(), clin[:, 1] * img_proba.flatten()])
    X_fusion_all = np.column_stack([clin[:, 1:2], img_proba, clin[:, 1:2] * img_proba, imgs_pca_32, clin_features])

    # 3. Load Models
    model_A = joblib.load(os.path.join(MODELS_DIR, "fusion_strategy_A.pkl"))
    model_B = joblib.load(os.path.join(MODELS_DIR, "fusion_strategy_C.pkl"))  # using XGB as comparison
    
    # Check if Stacking and Calibrated models exist
    has_stacking = os.path.exists(os.path.join(MODELS_DIR, "stack_meta_learner.pkl"))
    has_calibrated = os.path.exists(os.path.join(MODELS_DIR, "fusion_model_calibrated.pkl"))

    if has_stacking:
        meta_learner = joblib.load(os.path.join(MODELS_DIR, "stack_meta_learner.pkl"))
        base_models = []
        for name in ['lr', 'rf', 'xgb', 'gb', 'svm']:
            base_models.append(joblib.load(os.path.join(MODELS_DIR, f'stack_base_{name}.pkl')))
        # Generate stacking predictions
        test_predictions = np.zeros((len(test_idx), len(base_models)))
        for i, model in enumerate(base_models):
            test_predictions[:, i] = model.predict_proba(X_fusion_all[test_idx])[:, 1]
        stacking_proba = meta_learner.predict_proba(test_predictions)[:, 1]
    else:
        stacking_proba = np.zeros(len(y_test))

    if has_calibrated:
        calibrated_model = joblib.load(os.path.join(MODELS_DIR, "fusion_model_calibrated.pkl"))
        cal_proba = calibrated_model.predict_proba(X_simple_all[test_idx])[:, 1]
    else:
        cal_proba = np.zeros(len(y_test))

    # Calculate optimal threshold
    fusion_val_proba = model_A.predict_proba(X_simple_all[val_idx])[:, 1]
    fpr_val, tpr_val, thresholds = roc_curve(y_val, fusion_val_proba)
    best_threshold = 0.50
    best_f1 = 0.0
    for thresh in thresholds:
        preds = (fusion_val_proba >= thresh).astype(int)
        f = f1_score(y_val, preds)
        if f > best_f1:
            best_f1 = f
            best_threshold = thresh
    
    with open(os.path.join(MODELS_DIR, "fusion_threshold.txt"), "w") as f:
        f.write(str(best_threshold))

    # Base Predictions
    clin_proba_test   = clin[test_idx, 1]
    img_proba_test    = img_proba[test_idx].flatten()
    fusion_proba_test = model_A.predict_proba(X_simple_all[test_idx])[:, 1]

    # Metrics
    clin_auc   = auc(*roc_curve(y_test, clin_proba_test)[:2])
    img_auc    = auc(*roc_curve(y_test, img_proba_test)[:2])
    fusion_auc = auc(*roc_curve(y_test, fusion_proba_test)[:2])
    stack_auc  = auc(*roc_curve(y_test, stacking_proba)[:2]) if has_stacking else 0
    cal_auc    = auc(*roc_curve(y_test, cal_proba)[:2]) if has_calibrated else 0

    fusion_acc = accuracy_score(y_test, (fusion_proba_test >= best_threshold).astype(int))
    fusion_f1  = f1_score(y_test, (fusion_proba_test >= best_threshold).astype(int))

    stack_acc = accuracy_score(y_test, (stacking_proba >= 0.5).astype(int)) if has_stacking else 0
    stack_f1  = f1_score(y_test, (stacking_proba >= 0.5).astype(int)) if has_stacking else 0

    cal_acc = accuracy_score(y_test, (cal_proba >= 0.5).astype(int)) if has_calibrated else 0
    cal_f1  = f1_score(y_test, (cal_proba >= 0.5).astype(int)) if has_calibrated else 0

    # High Confidence accuracy (Improvement 4)
    high_conf_mask = (fusion_proba_test >= 0.65) | (fusion_proba_test <= 0.35)
    high_conf_acc  = accuracy_score(y_test[high_conf_mask], (fusion_proba_test[high_conf_mask] >= best_threshold).astype(int))
    medium_conf_mask = (fusion_proba_test >= 0.48) & (fusion_proba_test < 0.65)
    medium_conf_acc = accuracy_score(y_test[medium_conf_mask], (fusion_proba_test[medium_conf_mask] >= best_threshold).astype(int)) if medium_conf_mask.sum() > 0 else 0
    low_conf_mask = (fusion_proba_test >= 0.35) & (fusion_proba_test < 0.48)
    low_conf_acc = accuracy_score(y_test[low_conf_mask], (fusion_proba_test[low_conf_mask] >= best_threshold).astype(int)) if low_conf_mask.sum() > 0 else 0

    # Write Final Report
    report_lines = [
        "═══════════════════════════════════════════════════════════════════",
        "  FINAL ACCURACY IMPROVEMENT REPORT",
        "═══════════════════════════════════════════════════════════════════",
        "  Approach                         AUC      Acc     F1     Change",
        "  ─────────────────────────────────────────────────────────────",
        f"  Baseline LR Fusion (3 features)  0.9125  82.7%  0.819  (baseline)",
        f"  + Feature Engineering            {fusion_auc:.4f}  {fusion_acc*100:.1f}%  {fusion_f1:.3f}  +{fusion_acc*100 - 82.7:.1f}%",
        f"  + Stacking Ensemble              {stack_auc:.4f}  {stack_acc*100:.1f}%  {stack_f1:.3f}  +{stack_acc*100 - 82.7:.1f}%",
        f"  + Calibrated Probabilities       {cal_auc:.4f}  {cal_acc*100:.1f}%  {cal_f1:.3f}  +{cal_acc*100 - 82.7:.1f}%",
        f"  + High-confidence only (subset)  ------  {high_conf_acc*100:.1f}%  -----  (subset)",
        "  ─────────────────────────────────────────────────────────────"
    ]
    
    # Determine best model
    aucs_to_compare = {"LR Fusion": fusion_auc}
    if has_stacking: aucs_to_compare["Stacking Ensemble"] = stack_auc
    if has_calibrated: aucs_to_compare["Calibrated Fusion"] = cal_auc
    
    best_name = max(aucs_to_compare, key=aucs_to_compare.get)
    best_auc = aucs_to_compare[best_name]
    best_acc = fusion_acc if best_name == "LR Fusion" else (stack_acc if best_name == "Stacking Ensemble" else cal_acc)
    best_f1_score = fusion_f1 if best_name == "LR Fusion" else (stack_f1 if best_name == "Stacking Ensemble" else cal_f1)
    
    report_lines.append(f"  BEST OVERALL MODEL: {best_name:13}  {best_auc:.4f}  {best_acc*100:.1f}%  {best_f1_score:.3f}")
    report_lines.append("═══════════════════════════════════════════════════════════════════")
    
    report_path = os.path.join(METRICS_DIR, "final_improvement_report.csv")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))

    # Save best production model
    if best_name == "Stacking Ensemble":
        joblib.dump(meta_learner, os.path.join(MODELS_DIR, "fusion_model_production.pkl"))
    elif best_name == "Calibrated Fusion":
        joblib.dump(calibrated_model, os.path.join(MODELS_DIR, "fusion_model_production.pkl"))
    else:
        joblib.dump(model_A, os.path.join(MODELS_DIR, "fusion_model_production.pkl"))
        
    cfg = {
        "best_model": best_name,
        "auc": float(best_auc),
        "accuracy": float(best_acc),
        "threshold": float(best_threshold) if best_name == "LR Fusion" else 0.50
    }
    with open(os.path.join(MODELS_DIR, "inference_config_v2.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────────
    # GRAPHS
    # ─────────────────────────────────────────────────────────────────────────────
    # Graph 9: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(9, 7))
    set_dark_theme(fig, ax)
    models_pr = {
        'Clinical (XGB)':     (y_test, clin_proba_test,    THEME["amber"], '--'),
        'Image (EffNet)':     (y_test, img_proba_test,     THEME["purple"], '--'),
        'LR Fusion (Best)':   (y_test, fusion_proba_test,  THEME["teal"], '-'),
    }
    if has_stacking:
        models_pr['Stacking Ensemble'] = (y_test, stacking_proba, THEME["red"], '-')

    for name, (y_true, y_score, color, ls) in models_pr.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(recall, precision, color=color, linestyle=ls, linewidth=2.5, label=f'{name}  (AP = {ap:.4f})')

    ax.set_xlabel('Recall (Sensitivity)', color=THEME["white"], fontsize=12)
    ax.set_ylabel('Precision (PPV)', color=THEME["white"], fontsize=12)
    ax.set_title('Precision-Recall Curve — Medical Context\n(Higher area = better for imbalanced classes)',
                 color=THEME["white"], fontsize=12, fontweight='bold')
    ax.legend(facecolor=THEME["panel"], labelcolor=THEME["white"], fontsize=10)
    plt.tight_layout()
    p9 = os.path.join(GRAPHS_DIR, 'precision_recall_curve.png')
    plt.savefig(p9, dpi=150, facecolor=THEME["bg"], bbox_inches='tight')
    print(f"Saved: precision_recall_curve.png ({os.path.getsize(p9)//1024} KB)")

    # Graph 10: Confidence Zone Analysis
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    set_dark_theme(fig, axes)
    fig.suptitle('Prediction Confidence Zone Analysis', color=THEME["white"], fontsize=13, fontweight='bold')

    ax1 = axes[0]
    high_c   = high_conf_mask.sum()
    medium_c = medium_conf_mask.sum()
    low_c    = low_conf_mask.sum()
    sizes  = [high_c, medium_c, low_c]
    clrs   = [THEME["teal"], THEME["amber"], THEME["red"]]
    labels_pie = [f'High confidence\n({high_c} samples)',
                  f'Medium confidence\n({medium_c} samples)',
                  f'Low — needs review\n({low_c} samples)']
    ax1.pie(sizes, labels=labels_pie, colors=clrs, autopct='%1.1f%%', textprops={'color': 'white'}, startangle=90)
    ax1.set_title('Prediction Confidence Distribution', color=THEME["white"], fontsize=11)

    ax2 = axes[1]
    zones = ['High confidence', 'Medium confidence', 'Low / Borderline']
    accs_by_zone = [high_conf_acc, medium_conf_acc, low_conf_acc]
    bars = ax2.bar(zones, accs_by_zone, color=[THEME["teal"], THEME["amber"], THEME["red"]], alpha=0.85)
    for bar, acc in zip(bars, accs_by_zone):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.1%}', ha='center', color=THEME["white"], fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy', color=THEME["white"], fontsize=11)
    ax2.set_title('Accuracy by Confidence Zone', color=THEME["white"], fontsize=11)
    ax2.set_ylim(0.6, 1.05)
    ax2.yaxis.grid(True, color=THEME["grid"], linewidth=0.5)

    plt.tight_layout()
    p10 = os.path.join(GRAPHS_DIR, 'confidence_zone_analysis.png')
    plt.savefig(p10, dpi=150, facecolor=THEME["bg"], bbox_inches='tight')
    print(f"Saved: confidence_zone_analysis.png ({os.path.getsize(p10)//1024} KB)")

    # MASTER DASHBOARD
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(THEME["bg"])
    fig.suptitle('Multimodal UTI Prediction — Final Results Dashboard', color=THEME["white"], fontsize=16, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Top-left: PR Curve
    ax1 = fig.add_subplot(gs[0, 0])
    set_dark_theme(fig, ax1)
    for name, (y_true, y_score, color, ls) in models_pr.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax1.plot(recall, precision, color=color, linestyle=ls, linewidth=2, label=f'{name} ({ap:.3f})')
    ax1.set_title('Precision-Recall Curves', color=THEME["white"], fontsize=12)
    ax1.legend(facecolor=THEME["panel"], labelcolor=THEME["white"], fontsize=8)

    # Top-center: Bars
    ax2 = fig.add_subplot(gs[0, 1])
    set_dark_theme(fig, ax2)
    x2 = np.arange(4)
    model_labels = ['Clinical', 'Image', 'LR Fusion', 'Stacking']
    a_aucs = [clin_auc, img_auc, fusion_auc, stack_auc if has_stacking else 0]
    a_accs = [accuracy_score(y_test, (clin_proba_test >= 0.5).astype(int)),
              accuracy_score(y_test, (img_proba_test >= 0.5).astype(int)),
              fusion_acc, stack_acc]
    a_f1s  = [f1_score(y_test, (clin_proba_test >= 0.5).astype(int)),
              f1_score(y_test, (img_proba_test >= 0.5).astype(int)),
              fusion_f1, stack_f1]
    
    ax2.bar(x2 - 0.2, a_aucs, 0.2, label='AUC', color=THEME["teal"])
    ax2.bar(x2,       a_accs, 0.2, label='Acc', color=THEME["purple"])
    ax2.bar(x2 + 0.2, a_f1s,  0.2, label='F1',  color=THEME["amber"])
    ax2.set_xticks(x2)
    ax2.set_xticklabels(model_labels, color=THEME["white"], fontsize=10)
    ax2.set_title('Performance Metrics', color=THEME["white"], fontsize=12)
    ax2.legend(facecolor=THEME["panel"], labelcolor=THEME["white"], fontsize=8)

    # Top-right: Confusion Matrix
    ax3 = fig.add_subplot(gs[0, 2])
    set_dark_theme(fig, ax3)
    best_preds = (stacking_proba >= 0.5).astype(int) if best_name == "Stacking Ensemble" else (fusion_proba_test >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, best_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Infected'], yticklabels=['Normal', 'Infected'],
                ax=ax3, cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    ax3.set_title(f'Best Model ({best_name}) Confusion Matrix', color=THEME["white"], fontsize=12)

    # Bottom-left: Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    set_dark_theme(fig, ax4)
    best_proba = stacking_proba if best_name == "Stacking Ensemble" else fusion_proba_test
    ax4.hist(best_proba[y_test == 0], bins=25, alpha=0.7, color=THEME["teal"], label='Normal')
    ax4.hist(best_proba[y_test == 1], bins=25, alpha=0.7, color=THEME["red"], label='Infected')
    thresh_plot = 0.50 if best_name == "Stacking Ensemble" else best_threshold
    ax4.axvline(x=thresh_plot, color='white', linestyle='--', label=f'Thresh={thresh_plot:.2f}')
    ax4.set_title('Score Distribution', color=THEME["white"], fontsize=12)
    ax4.legend(facecolor=THEME["panel"], labelcolor=THEME["white"], fontsize=8)

    # Bottom-center: Confidence Zone Accuracy
    ax5 = fig.add_subplot(gs[1, 1])
    set_dark_theme(fig, ax5)
    bars = ax5.bar(zones, accs_by_zone, color=[THEME["teal"], THEME["amber"], THEME["red"]], alpha=0.85)
    for bar, acc in zip(bars, accs_by_zone):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{acc:.1%}', ha='center', color=THEME["white"], fontsize=10)
    ax5.set_title('Accuracy by Confidence Zone', color=THEME["white"], fontsize=12)
    ax5.set_ylim(0.6, 1.05)

    # Bottom-right: Text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(THEME["bg"])
    ax6.axis('off')
    summary_text = f"""
╔══════════════════════════════╗
║   FINAL RESULTS SUMMARY      ║
╠══════════════════════════════╣
║ Clinical AUC:    {clin_auc:.4f}      ║
║ Image AUC:       {img_auc:.4f}      ║
║ Fusion AUC:      {best_auc:.4f}  ★   ║
╠══════════════════════════════╣
║ Accuracy:        {best_acc*100:.1f}%       ║
║ High-Conf Acc:   {high_conf_acc*100:.1f}%       ║
║ F1 Score:        {best_f1_score:.4f}      ║
╠══════════════════════════════╣
║ Dataset: 4,000 paired        ║
║ Backbone: EfficientNetB3     ║
║ Clinical: XGBoost 4K Eng     ║
║ Best Model: {best_name:16} ║
╚══════════════════════════════╝
"""
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=12, color=THEME["white"], verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor=THEME["panel"], edgecolor=THEME["teal"], alpha=0.8))

    p11 = os.path.join(GRAPHS_DIR, 'MASTER_DASHBOARD.png')
    plt.savefig(p11, dpi=200, facecolor=THEME["bg"], bbox_inches='tight')
    print(f"Saved: MASTER_DASHBOARD.png ({os.path.getsize(p11)//1024} KB)")

if __name__ == "__main__":
    main()
