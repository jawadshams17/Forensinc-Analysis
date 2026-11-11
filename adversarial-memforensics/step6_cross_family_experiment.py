"""
step6_cross_family_experiment.py
================================
Q1 Hardening — Path 3: The Cross-Family Zero-Shot Generalisation

MODEL: Single XGBoost (McNemar p>0.05 confirmed in step8 — ensemble unjustified).

This experiment replaces the N=30 target dataset limitation by artificially
creating a massive OOD target dataset from within CIC-MalMem-2022.

Split:
  Source Domain: Benign (50%) + Ransomware + Trojans  (~33k samples)
  Target Domain: Benign (50%) + Spyware               (~24k samples)

Outputs:
    results/unified/step6_cross_family.json
    results/unified/figures/step6_cross_family_roc.png
"""
import sys, json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("pip install xgboost")

warnings.filterwarnings("ignore")
plt.rc("font", family="serif", size=11)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "results" / "unified"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DUMP_CSV = Path(r"F:\FORENSICS\MalwareMemoryDump.csv")
FEAT_MAP = ROOT / "data" / "unified" / "feature_map.json"

RS = 42


def make_model():
    """Single XGBoost — validated in step8 (McNemar p>0.05, ensemble unjustified)."""
    return XGBClassifier(n_estimators=100, max_depth=5, use_label_encoder=False,
                         eval_metric="logloss", random_state=RS, verbosity=0)


def main():
    logger.info("=" * 70)
    logger.info("  STEP 6: CROSS-FAMILY ZERO-SHOT  (N≈25k PROOF)")
    logger.info("=" * 70)

    with open(FEAT_MAP) as f:
        feat_names = json.load(f)["features"]

    df = pd.read_csv(DUMP_CSV)
    df["SubType"] = df["SubType"].str.strip()

    df_benign     = df[df["SubType"] == "Benign"].copy()
    df_ransomware = df[df["SubType"] == "Ransomware"].copy()
    df_trojan     = df[df["SubType"] == "Trojan"].copy()
    df_spyware    = df[df["SubType"] == "Spyware"].copy()

    logger.info(f"  Benign={len(df_benign):,}  Ransomware={len(df_ransomware):,}  "
                f"Trojan={len(df_trojan):,}  Spyware={len(df_spyware):,}")

    benign_src, benign_tgt = train_test_split(df_benign, test_size=0.5, random_state=RS)

    df_src = pd.concat([benign_src, df_ransomware, df_trojan])
    df_tgt = pd.concat([benign_tgt, df_spyware])

    df_src["label"] = (df_src["SubType"] != "Benign").astype(int)
    df_tgt["label"] = (df_tgt["SubType"] != "Benign").astype(int)

    X_src = df_src[feat_names].values.astype(np.float64)
    y_src = df_src["label"].values.astype(int)
    X_tgt = df_tgt[feat_names].values.astype(np.float64)
    y_tgt = df_tgt["label"].values.astype(int)

    logger.info(f"  Source={len(X_src):,}  Target={len(X_tgt):,}")

    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_src, y_src, test_size=0.2, random_state=RS, stratify=y_src
    )

    logger.info("  Training XGBoost on Source Domain...")
    model = make_model()
    model.fit(Xs_tr, ys_tr)

    # In-distribution evaluation
    yp_src  = model.predict(Xs_te)
    ypr_src = model.predict_proba(Xs_te)[:, 1]
    acc_src = accuracy_score(ys_te, yp_src)
    f1_src  = f1_score(ys_te, yp_src, average="macro")
    auc_src = roc_auc_score(ys_te, ypr_src)
    logger.info(f"  [In-Distribution]  Acc={acc_src:.4f}  F1={f1_src:.4f}  AUC={auc_src:.4f}")

    # Zero-Shot Cross-Family (OOD)
    yp_tgt  = model.predict(X_tgt)
    ypr_tgt = model.predict_proba(X_tgt)[:, 1]
    acc_tgt = accuracy_score(y_tgt, yp_tgt)
    f1_tgt  = f1_score(y_tgt, yp_tgt, average="macro")
    auc_tgt = roc_auc_score(y_tgt, ypr_tgt)
    logger.info(f"  [Zero-Shot OOD]    Acc={acc_tgt:.4f}  F1={f1_tgt:.4f}  AUC={auc_tgt:.4f}")

    # ROC figure
    fig, ax = plt.subplots(figsize=(8, 6))
    fpr_s, tpr_s, _ = roc_curve(ys_te, ypr_src)
    fpr_t, tpr_t, _ = roc_curve(y_tgt, ypr_tgt)
    ax.plot(fpr_s, tpr_s, label=f"In-Distribution (Ransom+Trojan)\nAUC={auc_src:.3f}",
            color="#1f77b4", lw=2.5)
    ax.plot(fpr_t, tpr_t, label=f"Zero-Shot Cross-Family (Spyware)\nAUC={auc_tgt:.3f}",
            color="#d62728", lw=2.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance")
    ax.set_xlabel("False Positive Rate");  ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-Family Zero-Shot Generalisation\n(XGBoost | CIC-MalMem-2022 binary Benign vs Malicious)")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step6_cross_family_roc.png", dpi=300)
    plt.close()

    result = {
        "model": "XGBoost (McNemar-validated, ensemble dropped)",
        "source_size": int(len(X_src)),
        "target_size": int(len(X_tgt)),
        "acc_indist":  round(acc_src, 4),
        "f1_indist":   round(f1_src, 4),
        "auc_indist":  round(auc_src, 4),
        "acc_oos":     round(acc_tgt, 4),
        "f1_oos":      round(f1_tgt, 4),
        "auc_oos":     round(auc_tgt, 4),
    }
    with open(OUT_DIR / "step6_cross_family.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.success("  Step 6 done. N=30 objection dissolved.")


if __name__ == "__main__":
    main()
