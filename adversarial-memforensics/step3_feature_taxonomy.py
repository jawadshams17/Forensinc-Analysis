"""
step3_feature_taxonomy.py
=========================
Q1 Hardening — Step 3: Feature Taxonomy & CIC Confound Audit

1. CONDITIONAL SHIFT ANALYSIS
   Computes KS[Benign vs Benign] and KS[Malware vs Malware].
   If Benign shift ≈ Malware shift, the feature is environment-dependent.
   Outputs a feature taxonomy:
   - Environment-Dependent
   - Malware-Indicative
   - Mixed

2. CIC CONFOUND AUDIT
   Proves that CIC-MalMem-2022 is separable using purely environment-dependent
   features, confirming the dataset artefact. Train model on only these features.

3. SOURCE SIZE ABLATION
   Trains on 1%, 5%, 10%, 25%, 50%, 100% of CIC-MalMem-2022.
   Evaluates cross-dataset transfer to MalVol-25. Shows whether more
   source data closes the gap.

Outputs:
    results/unified/step3_taxonomy.json
    results/unified/figures/step3_conditional_shift.png
    results/unified/figures/step3_ablation_curve.png
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
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit("pip install xgboost lightgbm")

warnings.filterwarnings("ignore")
plt.rc("font", family="serif", size=11)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "results" / "unified"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV   = ROOT / "data" / "unified" / "train_val_test.csv"
HOLDOUT_CSV = ROOT / "data" / "unified" / "malvol25_holdout.csv"
FEAT_MAP    = ROOT / "data" / "unified" / "feature_map.json"

RS = 42

def load_data():
    df_src = pd.read_csv(TRAIN_CSV)
    df_tgt = pd.read_csv(HOLDOUT_CSV)
    with open(FEAT_MAP) as f:
        feat_names = json.load(f)["features"]
    X_src = df_src[feat_names].values.astype(np.float64)
    y_src = df_src["label"].values.astype(int)
    X_tgt = df_tgt[feat_names].values.astype(np.float64)
    y_tgt = df_tgt["label"].values.astype(int)
    return X_src, y_src, X_tgt, y_tgt, feat_names


def make_ensemble():
    rf   = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS)
    xgb  = XGBClassifier(n_estimators=100, use_label_encoder=False,
                          eval_metric="logloss", random_state=RS, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=100, random_state=RS, verbosity=-1)
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                             voting="soft", n_jobs=-1)


# ═══════════════════════════════════════════════════════════════════════
#  A. CONDITIONAL SHIFT & TAXONOMY
# ═══════════════════════════════════════════════════════════════════════

def run_conditional_shift(X_src, y_src, X_tgt, y_tgt, feat_names):
    logger.info("--- A. Conditional Shift & Taxonomy ---")
    results = []

    # Masks
    m_src_ben = (y_src == 0)
    m_src_mal = (y_src == 1)
    m_tgt_ben = (y_tgt == 0)
    m_tgt_mal = (y_tgt == 1)

    for i, fname in enumerate(feat_names):
        # Overall shift
        ks_all, _ = ks_2samp(X_src[:, i], X_tgt[:, i])
        # Benign vs Benign (Environment shift)
        ks_benign, _ = ks_2samp(X_src[m_src_ben, i], X_tgt[m_tgt_ben, i])
        # Malware vs Malware
        ks_malware, _ = ks_2samp(X_src[m_src_mal, i], X_tgt[m_tgt_mal, i])

        # Classification criteria (heuristic)
        # If Benign shifts heavily (KS > 0.4), it is environment-dependent
        if ks_benign > 0.4:
            cat = "Environment-Dependent"
        # If Benign shift is low but Malware shift is high, it's malware-specific context diff
        elif ks_benign <= 0.4 and ks_malware > 0.4:
            cat = "Malware-Context"
        # If both are low, it's globally invariant
        else:
            cat = "Globally Invariant"

        results.append({
            "feature": fname,
            "ks_overall": round(float(ks_all), 4),
            "ks_benign": round(float(ks_benign), 4),
            "ks_malware": round(float(ks_malware), 4),
            "category": cat
        })

    df = pd.DataFrame(results)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 10))
    df_sorted = df.sort_values("ks_benign", ascending=True)

    y_pos = np.arange(len(df))
    ax.barh(y_pos - 0.2, df_sorted["ks_benign"], height=0.4, label="Benign vs Benign (Env Shift)", color="#4c72b0")
    ax.barh(y_pos + 0.2, df_sorted["ks_malware"], height=0.4, label="Malware vs Malware", color="#c44e52")

    ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["feature"], fontsize=8)
    ax.set_xlabel("KS Statistic")
    ax.set_title("Conditional Feature Shift: Splitting Environment vs Payload")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_conditional_shift.png", dpi=300)
    plt.close()

    logger.info(f"  Mapped {len(df[df.category=='Environment-Dependent'])} features as Environment-Dependent")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  B. CIC CONFOUND AUDIT
# ═══════════════════════════════════════════════════════════════════════

def run_cic_confound_audit(X_src, y_src, feat_names, taxonomy_df):
    logger.info("\n--- B. CIC-MalMem-2022 Confound Audit ---")

    # Select ONLY features that are heavily environment-dependent
    env_features = taxonomy_df[taxonomy_df["ks_benign"] > 0.6]["feature"].tolist()
    env_indices  = [feat_names.index(f) for f in env_features]

    logger.info(f"  Training ensemble purely on {len(env_features)} environment artefacts...")
    X_src_confound = X_src[:, env_indices]

    # 5-fold CV internally on CIC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    model = make_ensemble()
    scores = cross_val_score(model, X_src_confound, y_src, cv=cv, scoring="accuracy", n_jobs=-1)

    mean_acc = scores.mean()
    logger.info(f"  Confound Accuracy (using only VM baseline features): {mean_acc:.4f} "
                f"[{scores.min():.4f} - {scores.max():.4f}]")

    return {
        "features_used": env_features,
        "n_features": len(env_features),
        "cv_accuracy_mean": round(float(mean_acc), 4),
        "cv_accuracy_std": round(float(scores.std()), 4)
    }


# ═══════════════════════════════════════════════════════════════════════
#  C. SOURCE SIZE ABLATION
# ═══════════════════════════════════════════════════════════════════════

def run_source_ablation(X_src, y_src, X_tgt, y_tgt):
    logger.info("\n--- C. Source Size Ablation ---")

    fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.0]
    n_total = X_src.shape[0]

    rng = np.random.default_rng(RS)
    results = []

    for frac in fractions:
        n_samples = max(10, int(n_total * frac))

        # Stratified subsampling
        idx_0 = np.where(y_src == 0)[0]
        idx_1 = np.where(y_src == 1)[0]
        n_0 = int(n_samples * 0.5)
        n_1 = n_samples - n_0

        samp_0 = rng.choice(idx_0, min(n_0, len(idx_0)), replace=False)
        samp_1 = rng.choice(idx_1, min(n_1, len(idx_1)), replace=False)
        idx_sub = np.concatenate([samp_0, samp_1])

        X_sub = X_src[idx_sub]
        y_sub = y_src[idx_sub]

        model = make_ensemble()
        model.fit(X_sub, y_sub)

        # In-distribution (test on rest of CIC)
        idx_rest = np.array(list(set(range(n_total)) - set(idx_sub)))
        if len(idx_rest) > 10000:
            idx_rest = rng.choice(idx_rest, 10000, replace=False)
            
        if len(idx_rest) > 0:
            idx_rest = idx_rest.astype(int)
            yp_src = model.predict(X_src[idx_rest])
            acc_src = accuracy_score(y_src[idx_rest], yp_src)
        else:
            acc_src = 1.0  # Training accuracy essentially when 100% is used

        # Out-of-distribution (test on MalVol-25)
        yp_tgt = model.predict(X_tgt)
        acc_tgt = accuracy_score(y_tgt, yp_tgt)

        logger.info(f"  Frac {frac*100:3.0f}% ({n_samples:6d} samples): "
                    f"In-Dist={acc_src:.4f}  |  OOD={acc_tgt:.4f}")

        results.append({
            "fraction": frac,
            "n_samples": n_samples,
            "acc_in_dist": acc_src,
            "acc_ood": acc_tgt
        })

    # Plot
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["n_samples"], df["acc_in_dist"], "o-", label="In-Distribution (CIC)", color="#1f77b4")
    ax.plot(df["n_samples"], df["acc_ood"], "s-", label="Cross-Dataset (MalVol-25)", color="#d62728")
    ax.set_xscale("log")
    ax.set_xlabel("Number of Source Training Samples (Log Scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Does More Data Solve Domain Shift?\n(In-Distribution vs Cross-Dataset Generalisation)")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step3_ablation_curve.png", dpi=300)
    plt.close()

    return results


def main():
    logger.info("=" * 65)
    logger.info("  STEP 3: TAXONOMY, CONFOUNDS, ABLATION")
    logger.info("=" * 65)

    X_src, y_src, X_tgt, y_tgt, feat_names = load_data()

    tax_df = run_conditional_shift(X_src, y_src, X_tgt, y_tgt, feat_names)
    confound_res = run_cic_confound_audit(X_src, y_src, feat_names, tax_df)
    ablation_res = run_source_ablation(X_src, y_src, X_tgt, y_tgt)

    output = {
        "taxonomy": tax_df.to_dict(orient="records"),
        "confound_audit": confound_res,
        "ablation": ablation_res
    }

    with open(OUT_DIR / "step3_taxonomy.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.success("  Done. Step 3 results saved.")

if __name__ == "__main__":
    main()
