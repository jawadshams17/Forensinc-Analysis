"""
step9_phase2_experiments.py
============================
Phase 2 — The two experiments demanded by the adversarial review panel.

EXPERIMENT A: True Rank-Deficiency Table
  Train XGBoost on CIC-MalMem-2022 source domain.
  Record SHAP feature importances on CIC test set (in-dist).
  Record SHAP feature importances on MalVol-25 (OOD target).
  Report the top-3 features that were CRITICAL in CIC but VANISH on MalVol-25.

EXPERIMENT B: Benign-Only OOD Rejection Test
  Train XGBoost on CIC Benign (Win7 VM baseline).
  Test on MalVol-25 Benign subset only (Win10, N≈15).
  Report False Positive Rate as OS-generational fragility baseline.

Outputs:
    results/unified/step9_rank_deficiency.json
    results/unified/step9_benign_ood.json
    results/unified/thesis_figures/step9_shap_drift.png
    results/unified/thesis_figures/step9_benign_ood_fpr.png
"""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("pip install xgboost shap")

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

ROOT    = Path(__file__).parent
DUMP_CSV = Path(r"F:\FORENSICS\MalwareMemoryDump.csv")
MV_CSV   = ROOT / "data" / "malvol25" / "extracted" / "MalVol25_features.csv"
FEAT_MAP = ROOT / "data" / "unified" / "feature_map.json"
OUT_DIR  = ROOT / "results" / "unified"
FIG_DIR  = ROOT / "results" / "unified" / "thesis_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

RS = 42

def make_model():
    return XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                         eval_metric="logloss", random_state=RS, verbosity=0)

def load_cic(feat_names):
    df = pd.read_csv(DUMP_CSV)
    df["label"] = (df["SubType"].str.strip() != "Benign").astype(int)
    X = df[feat_names].values.astype(np.float64)
    y = df["label"].values.astype(int)
    return X, y

def load_malvol25(feat_names):
    mv = pd.read_csv(MV_CSV)
    mv.columns = [c.replace(".", "_") for c in mv.columns]
    meta = {"Category", "Label", "Family", "OS_Version", "File", "OS", "Class"}
    meta_norm = {m.replace(".", "_") for m in meta}
    mv["label"] = (mv["Label"] != 0).astype(int)
    # Benign samples only for Experiment B
    mv_benign = mv[mv["label"] == 0].copy()
    # Check available features
    available = [f for f in feat_names if f in mv.columns]
    X_all    = mv[available].values.astype(np.float64)
    y_all    = mv["label"].values.astype(int)
    X_benign = mv_benign[available].values.astype(np.float64)
    y_benign = mv_benign["label"].values.astype(int)
    return X_all, y_all, X_benign, y_benign, available


def run_rank_deficiency(feat_names):
    """Experiment A: Top-3 features relied on CIC but vanish on MalVol-25."""
    logger.info("=== EXP A: Rank-Deficiency Table ===")
    X, y = load_cic(feat_names)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)

    model = make_model()
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)

    # CIC in-distribution SHAP
    logger.info("  Computing SHAP on CIC test set (N=500 subsample)...")
    idx_cic = np.random.RandomState(RS).choice(X_te.shape[0], 500, replace=False)
    sv_cic  = explainer.shap_values(X_te[idx_cic])
    if len(sv_cic.shape) == 3:
        sv_cic = sv_cic[:, :, 1]
    shap_cic = np.abs(sv_cic).mean(axis=0)

    # MalVol-25 OOD SHAP
    logger.info("  Computing SHAP on MalVol-25 OOD set...")
    X_mv, y_mv, _, _, available = load_malvol25(feat_names)
    # Align to available features only
    av_idx = [feat_names.index(f) for f in available]
    X_te_aligned  = X_te[:, av_idx]
    X_mv_aligned  = X_mv

    # Rebuild model on available features only for fair SHAP comparison
    model_mv = make_model()
    model_mv.fit(X_tr[:, av_idx], y_tr)
    explainer_mv = shap.TreeExplainer(model_mv)
    sv_mv = explainer_mv.shap_values(X_mv_aligned)
    if len(sv_mv.shape) == 3:
        sv_mv = sv_mv[:, :, 1]
    shap_mv = np.abs(sv_mv).mean(axis=0)

    # Build the rank-deficiency table
    shap_cic_aligned = np.abs(
        explainer_mv.shap_values(X_te_aligned[:500])
    ).mean(axis=0)
    if len(shap_cic_aligned.shape) > 1:
        shap_cic_aligned = shap_cic_aligned[:, 1]

    feat_table = pd.DataFrame({
        "feature":   available,
        "shap_cic":  shap_cic_aligned,
        "shap_mv25": shap_mv,
    })
    feat_table["drift"] = feat_table["shap_cic"] - feat_table["shap_mv25"]
    feat_table = feat_table.sort_values("shap_cic", ascending=False)

    top3_vanishing = feat_table.head(10).nlargest(3, "drift")[["feature", "shap_cic", "shap_mv25", "drift"]]
    logger.info(f"  Top-3 features critically relied on CIC but vanishing on MalVol-25:\n{top3_vanishing}")

    # Plot SHAP drift
    plot_df = feat_table.head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_df))
    ax.bar(x - 0.2, plot_df["shap_cic"],  width=0.4, label="CIC (In-Dist)",  color="#1f77b4")
    ax.bar(x + 0.2, plot_df["shap_mv25"], width=0.4, label="MalVol-25 (OOD)", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["feature"], rotation=45, ha="right", fontsize=8)
    ax.set_title("Model Reliance Drift: Top 15 Features\n(Blue=CIC In-Dist Reliance | Red=MalVol-25 OOD Reliance)")
    ax.set_ylabel("Mean |SHAP Value|")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step9_shap_drift.png", dpi=300)
    plt.close()

    result = {
        "top3_vanishing": top3_vanishing.to_dict(orient="records"),
        "full_table": feat_table[["feature","shap_cic","shap_mv25","drift"]].to_dict(orient="records")
    }
    with open(OUT_DIR / "step9_rank_deficiency.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.success("  Exp A complete.")


def run_benign_ood(feat_names):
    """Experiment B: FPR of CIC model on MalVol-25 Benign subset (Windows 10 OOD)."""
    logger.info("=== EXP B: Benign-Only OOD Rejection (OS-Gen Fragility) ===")

    X_cic, y_cic = load_cic(feat_names)
    _, _, X_benign_mv, y_benign_mv, available = load_malvol25(feat_names)

    if len(X_benign_mv) == 0:
        logger.warning("  MalVol-25 has zero Benign samples — skipping Exp B.")
        return

    av_idx = [feat_names.index(f) for f in available]

    # Train on CIC source
    model = make_model()
    X_tr, _, y_tr, _ = train_test_split(X_cic, y_cic, test_size=0.2, random_state=RS, stratify=y_cic)
    model.fit(X_tr[:, av_idx], y_tr)

    # Predict on MalVol-25 Benign subset only (ground truth = 0, all Benign)
    y_pred_benign = model.predict(X_benign_mv)
    n_total = len(y_benign_mv)
    n_fp = int(np.sum(y_pred_benign == 1))   # Benign flagged as Malware = FP
    fpr = n_fp / n_total if n_total > 0 else None

    logger.info(f"  MalVol-25 Benign samples (N={n_total})")
    logger.info(f"  False Positives (Win10 Benign flagged as Malware): {n_fp}/{n_total}")
    logger.info(f"  False Positive Rate (OS-Generational Fragility): {fpr:.3f}")

    # Bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["True Negative\n(Correct)", "False Positive\n(Fragility Error)"],
           [n_total - n_fp, n_fp],
           color=["#2ca02c", "#d62728"])
    ax.set_title(f"Benign-Only OOD Rejection Test\n(Model: CIC Win7 → Target: MalVol-25 Win10, N={n_total})")
    ax.set_ylabel("Sample Count")
    for bar, val in zip(ax.patches, [n_total - n_fp, n_fp]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val} ({val/n_total*100:.0f}%)", ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step9_benign_ood_fpr.png", dpi=300)
    plt.close()

    with open(OUT_DIR / "step9_benign_ood.json", "w") as f:
        json.dump({"n_benign_mv25": n_total, "n_fp": n_fp, "fpr": fpr,
                   "interpretation": (
                       "FPR measures OS-generational fragility: the fraction of Win10 Benign "
                       "samples that a Win7-trained model incorrectly classifies as Malware."
                   )}, f, indent=2)
    logger.success("  Exp B complete.")


def main():
    logger.info("=" * 70)
    logger.info("  STEP 9: PHASE 2 EXPERIMENTS (Rank-Deficiency + Benign OOD)")
    logger.info("=" * 70)

    with open(FEAT_MAP) as f:
        feat_names = json.load(f)["features"]

    run_rank_deficiency(feat_names)
    run_benign_ood(feat_names)
    logger.success("Phase 2 complete. See results/unified/step9_*.json and thesis_figures/step9_*.png")


if __name__ == "__main__":
    main()
