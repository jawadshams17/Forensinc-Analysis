"""
step7_thesis_visualizations.py
==============================
Generates a comprehensive suite of thesis-grade visualizations.
MODEL: Single XGBoost (McNemar p>0.05 proved ensemble adds no value — step8).

Includes: Data Stats, PCA/tSNE, Model Comparisons, SHAP, Learning Curves,
Confidence Distributions, and Error Analysis.

Outputs to results/unified/thesis_figures/
"""
import sys, json, time
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import shap

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit("pip install xgboost lightgbm shap")

warnings.filterwarnings("ignore")
plt.rc("font", family="serif", size=10)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

ROOT    = Path(__file__).parent
DATA_CSV = Path(r"F:\FORENSICS\MalwareMemoryDump.csv")
FEAT_MAP = ROOT / "data" / "unified" / "feature_map.json"
OUT_DIR = ROOT / "results" / "unified" / "thesis_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RS = 42

def load_and_prep_data():
    logger.info("Loading dataset...")
    df = pd.read_csv(DATA_CSV)
    df["Class"] = df["SubType"].str.strip()  # Mapped to 'SubType' in CIC-2022
    
    with open(FEAT_MAP) as f:
        features = json.load(f)["features"]
        
    df["binary_label"] = (df["Class"] != "Benign").astype(int)
    
    X = df[features].values.astype(np.float64)
    y = df["binary_label"].values.astype(int)
    
    # 80/20 train/test split
    X_tr, X_te, y_tr, y_te, cls_tr, cls_te = train_test_split(
        X, y, df["Class"].values, test_size=0.2, random_state=RS, stratify=y
    )
    return X, y, df, features, X_tr, X_te, y_tr, y_te, cls_te

def get_models():
    """Three independent models for the side-by-side comparison chart only.
    The PRIMARY model for all analysis is XGBoost alone (McNemar-validated).
    """
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS),
        "XGBoost (Primary)": XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                                           eval_metric="logloss", random_state=RS, verbosity=0),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=RS, verbosity=-1)
    }

def plot_distributions(df):
    logger.info("  -> Plotting Distributions")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Binary Label Dist
    sns.countplot(data=df, x="binary_label", ax=axes[0], palette="muted")
    axes[0].set_xticklabels(["Benign", "Malicious"])
    axes[0].set_title("Benign vs Malicious Distribution")
    
    # Attack Type Dist
    sns.countplot(data=df, y="Class", ax=axes[1], order=df["Class"].value_counts().index, palette="pastel")
    axes[1].set_title("Attack-Type Distribution")
    
    plt.tight_layout()
    plt.savefig(OUT_DIR / "1_data_distributions.png", dpi=300)
    plt.close()

def plot_correlation_heatmap(df, features):
    logger.info("  -> Correlation Heatmap")
    curves = {}
    
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_tr, y_tr)
        t_tr = time.time() - t0
        
        t0 = time.time()
        yp = clf.predict(X_te)
        try: ypr = clf.predict_proba(X_te)[:, 1]
        except: ypr = clf.decision_function(X_te)
        t_inf = time.time() - t0
        
        acc = accuracy_score(y_te, yp)
        pre = precision_score(y_te, yp)
        rec = recall_score(y_te, yp)
        f1  = f1_score(y_te, yp)
        auc = roc_auc_score(y_te, ypr)
        cm  = confusion_matrix(y_te, yp)
        
        results[name] = {"Acc": acc, "Prec": pre, "Recall": rec, "F1": f1, "AUC": auc, "T_train": t_tr, "T_inf": t_inf}
        curves[name] = {"ypr": ypr, "yp": yp, "cm": cm, "clf": clf}
        
    df_res = pd.DataFrame(results).T
    
    # Plot 4: Performance Bar Chart
    df_res[["Acc", "Prec", "Recall", "F1", "AUC"]].plot(kind="bar", figsize=(10, 5), colormap="viridis")
    plt.title("Model Comparison: Overall Performance")
    plt.ylim([0.9, 1.05])
    plt.legend(loc="lower right")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "4_model_comparison_metrics.png", dpi=300)
    plt.close()
    
    # Plot 5: Time Comparison
    df_res[["T_train", "T_inf"]].plot(kind="bar", figsize=(8, 5), color=["#d62728", "#1f77b4"])
    plt.title("Model Comparison: Computational Time (Seconds)")
    plt.yscale("log")
    plt.xticks(rotation=0)
    plt.ylabel("Time (Seconds, Log Scale)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "5_model_time_comparison.png", dpi=300)
    plt.close()
    
    # Plot 6: ROC and PR
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, c in curves.items():
        fpr, tpr, _ = roc_curve(y_te, c["ypr"])
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={df_res.loc[name, 'AUC']:.3f})")
        
        prec, rec, _ = precision_recall_curve(y_te, c["ypr"])
        axes[1].plot(rec, prec, label=f"{name}")
        
    axes[0].plot([0,1],[0,1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "6_roc_pr_curves.png", dpi=300)
    plt.close()
    
    # Plot 7: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, c) in zip(axes, curves.items()):
        sns.heatmap(c["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "7_confusion_matrices.png", dpi=300)
    plt.close()
    
    return curves["XGBoost (Primary)"]["clf"], curves["XGBoost (Primary)"]["yp"], curves["XGBoost (Primary)"]["ypr"]

def plot_advanced_analysis(clf, X_te, y_te, yp, ypr, cls_te, features):
    logger.info("  -> Error Analysis & Confidences")
    
    # 8. Detection by Attack Category (Detection Rate)
    # Using the predictions vs true class
    df_err = pd.DataFrame({"True_Class": cls_te, "True_Label": y_te, "Pred_Label": yp})
    df_err["Correct"] = (df_err["True_Label"] == df_err["Pred_Label"])
    
    cat_acc = df_err.groupby("True_Class")["Correct"].mean().sort_values()
    plt.figure(figsize=(8, 4))
    sns.barplot(x=cat_acc.values, y=cat_acc.index, palette="mako")
    plt.xlim(0, 1.05)
    plt.title("Detection Rate by Attack Category")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "8_detection_by_category.png", dpi=300)
    plt.close()
    
    # 9. Prediction Confidence Distribution
    plt.figure(figsize=(8, 5))
    sns.kdeplot(ypr[y_te==0], fill=True, label="True Benign", color="#1f77b4")
    sns.kdeplot(ypr[y_te==1], fill=True, label="True Malicious", color="#d62728")
    plt.title("Prediction Confidence Distribution (XGBoost)")
    plt.xlabel("Probability of Malware")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "9_confidence_distribution.png", dpi=300)
    plt.close()

def plot_shap(clf, X_tr, features):
    logger.info("  -> SHAP Explainability (Subsampled N=500)")
    idx = np.random.choice(X_tr.shape[0], 500, replace=False)
    X_sub = X_tr[idx]
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sub)
    
    # Ensure 2D (some xgboost versions return 2D, some 3D for binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sub, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "10_shap_summary.png", dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sub, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "11_shap_feature_importance.png", dpi=300)
    plt.close()

def plot_learning_curve_custom(clf, X, y):
    logger.info("  -> Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring="accuracy"
    )
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), "o-", label="Training score", color="r")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), "o-", label="Cross-validation score", color="g")
    plt.title("Learning Curve (XGBoost)")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "12_learning_curve.png", dpi=300)
    plt.close()

def main():
    logger.info("=" * 60)
    logger.info(" STEP 7: THESIS & PAPER COMPREHENSIVE VISUALIZATIONS")
    logger.info("=" * 60)
    
    X, y, df, features, X_tr, X_te, y_tr, y_te, cls_te = load_and_prep_data()
    
    plot_distributions(df)
    plot_correlation_heatmap(df, features)
    plot_manifold(X, y, df["Class"].values)
    
    xgb_clf, yp_xgb, ypr_xgb = evaluate_models_and_plot(X_tr, X_te, y_tr, y_te, cls_te)
    
    plot_advanced_analysis(xgb_clf, X_te, y_te, yp_xgb, ypr_xgb, cls_te, features)
    plot_shap(xgb_clf, X_tr, features)
    
    # Subsample for learning curve to save time (N=10000)
    idx = np.random.choice(X.shape[0], min(10000, X.shape[0]), replace=False)
    plot_learning_curve_custom(xgb_clf, X[idx], y[idx])
    
    logger.success(f"Done. Generated 12 thesis-grade visualization files in {OUT_DIR.relative_to(ROOT)}.")

if __name__ == "__main__":
    main()
