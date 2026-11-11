"""
generate_unified_plots.py
=========================
Generates 300-DPI publication graphics for the multi-dataset unified evaluation.
- Figure 1: Single-Dataset ROC (MalwareMemoryDump Test Set)
- Figure 2: Cross-Dataset ROC (MalVol-25 Zero-Shot Test Set)
- Figure 3: Classifier Transfer Performance Drop Boxplot/Bar
- Figure 4: Global Feature Importance
"""
import sys, os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score

plt.rc('font', family='serif')
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "results" / "unified" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "results" / "models" / "unified_ensemble.joblib"
TRAIN_CSV  = ROOT / "data" / "unified" / "train_val_test.csv"
MV_CSV     = ROOT / "data" / "unified" / "malvol25_holdout.csv"

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y, list(df.columns[:-1])

def main():
    logger.info("Generating Q1 Analysis Plots...")
    
    X_train_full, y_train_full, feature_names = load_data(TRAIN_CSV)
    X_mv, y_mv, _ = load_data(MV_CSV)
    
    # We use the train_test_split logic to isolate the TEST split used in training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.10, stratify=y_train_full, random_state=42)
        
    logger.info("Loading unified ensemble model...")
    model = joblib.load(MODEL_PATH)
    
    # Predict on standard test set (In-Distribution)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Predict on MalVol25 (Out-Of-Distribution)
    y_mv_pred = model.predict(X_mv)
    y_mv_proba = model.predict_proba(X_mv)[:, 1]
    
    # ── 1. Side-by-Side ROC Curves ──────────────────────────────────────
    logger.info("Plotting In-Dist vs Out-of-Dist ROC...")
    plt.figure(figsize=(7, 6))
    
    fpr_td, tpr_td, _ = roc_curve(y_test, y_test_proba)
    auc_td = auc(fpr_td, tpr_td)
    plt.plot(fpr_td, tpr_td, color='#1f77b4', lw=2.5, 
             label=f'In-Distribution (CIC-MalMem-2022)\nAUC = {auc_td:.3f}')
             
    fpr_mv, tpr_mv, _ = roc_curve(y_mv, y_mv_proba)
    auc_mv = auc(fpr_mv, tpr_mv)
    plt.plot(fpr_mv, tpr_mv, color='#d62728', lw=2.5, linestyle='--',
             label=f'Cross-Dataset (MalVol-25)\nAUC = {auc_mv:.3f}')
             
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Zero-Shot Generalisation Robustness (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_dataset_roc.png", dpi=300)
    plt.close()

    # ── 2. Side-by-Side Confusion Matrices ────────────────────────AAAAAA
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    cm_td = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_td, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
    axes[0].set_title("In-Distribution Matrix (Test Set)")
    axes[0].set_ylabel("True Class")
    axes[0].set_xlabel("Predicted Class")
    
    cm_mv = confusion_matrix(y_mv, y_mv_pred)
    sns.heatmap(cm_mv, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
    axes[1].set_title("Cross-Dataset Matrix (MalVol-25)")
    axes[1].set_ylabel("True Class")
    axes[1].set_xlabel("Predicted Class")
    
    plt.tight_layout()
    plt.savefig(FIG_DIR / "cross_dataset_confusion.png", dpi=300)
    plt.close()

    # ── 3. Base Learner Transfer Degradation ──────────────────────────
    logger.info("Evaluating Base Learner degradation...")
    results = []
    
    from sklearn.metrics import accuracy_score
    for name, est in model.named_estimators_.items():
        td_acc = accuracy_score(y_test, est.predict(X_test))
        mv_acc = accuracy_score(y_mv, est.predict(X_mv))
        results.append({"Model": name.upper(), "Dataset": "CIC-MalMem-2022 (In-Dist)", "Accuracy": td_acc})
        results.append({"Model": name.upper(), "Dataset": "MalVol-25 (Cross-Data)", "Accuracy": mv_acc})
        
    df_res = pd.DataFrame(results)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df_res, x="Model", y="Accuracy", hue="Dataset", palette="muted", edgecolor='black')
    plt.ylim([0, 1.05])
    plt.title("Algorithm Robustness to Dataset Distribution Shift")
    plt.ylabel("Testing Accuracy")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "transfer_degradation_bars.png", dpi=300)
    plt.close()

    # ── 4. Feature Importance ─────────────────────────────────────────
    rf = model.named_estimators_['rf']
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_importances, y=top_features, palette="rocket")
    plt.title('Top 15 Absolute Feature Importances (Gini Impurity)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(FIG_DIR / "unified_feature_importance.png", dpi=300)
    plt.close()

    logger.success(f"All Q1 figures saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
