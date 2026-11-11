"""
root_cause_and_defence.py
=========================
Two-part Q1 novelty engine:

  Part 1 — ROOT-CAUSE ANALYSIS (C2)
    For each of the 41 shared features, compute:
      - Kolmogorov-Smirnov (KS) test statistic + p-value
      - Cohen's d effect size
      - Mean/std per dataset
    Rank features by shift severity. Produce publication figures.

  Part 2 — DOMAIN ADAPTATION DEFENCES (C3)
    Three progressively sophisticated defences:
      D1: Z-Score Normalisation  (standardise per-dataset)
      D2: Drift-Aware Feature Selection  (drop top-K drifting features)
      D3: CORAL (CORrelation ALignment)  (align covariance matrices)
    Each is evaluated on the MalVol-25 holdout and compared.

Outputs:
    results/unified/feature_shift_analysis.csv
    results/unified/defence_results.json
    results/unified/figures/feature_drift_heatmap.png
    results/unified/figures/defence_comparison.png
    results/unified/figures/drift_vs_accuracy.png
    results/unified/figures/defence_confusion_grid.png
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
from scipy import stats as sp_stats
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

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

RANDOM_STATE = 42
N_BOOTSTRAP  = 1000

# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def load_unified():
    """Load both datasets aligned on common features."""
    df_src = pd.read_csv(TRAIN_CSV)
    df_tgt = pd.read_csv(HOLDOUT_CSV)
    with open(FEAT_MAP) as f:
        feat_names = json.load(f)["features"]
    X_src = df_src[feat_names].values.astype(np.float64)
    y_src = df_src["label"].values.astype(int)
    X_tgt = df_tgt[feat_names].values.astype(np.float64)
    y_tgt = df_tgt["label"].values.astype(int)
    return X_src, y_src, X_tgt, y_tgt, feat_names


def bootstrap_metric(y_true, y_pred, metric_fn, n=N_BOOTSTRAP):
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def train_ensemble(X_train, y_train):
    """Train a fresh RF+XGB+LGBM soft-voting ensemble."""
    rf   = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE)
    xgb  = XGBClassifier(n_estimators=200, use_label_encoder=False,
                          eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE, verbosity=-1)
    ens  = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                             voting="soft", n_jobs=-1)
    ens.fit(X_train, y_train)
    return ens


def evaluate_on_target(model, X_tgt, y_tgt, label=""):
    """Evaluate a trained model on the target (MalVol-25) set."""
    y_pred  = model.predict(X_tgt)
    y_proba = model.predict_proba(X_tgt)[:, 1]
    acc = accuracy_score(y_tgt, y_pred)
    f1  = f1_score(y_tgt, y_pred, average="macro", zero_division=0)
    try:
        auc_val = roc_auc_score(y_tgt, y_proba)
    except ValueError:
        auc_val = 0.0
    acc_m, acc_lo, acc_hi = bootstrap_metric(y_tgt, y_pred, accuracy_score)
    f1_m, f1_lo, f1_hi   = bootstrap_metric(y_tgt, y_pred,
                            lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    cm = confusion_matrix(y_tgt, y_pred)
    logger.info(f"  [{label}]  Acc={acc:.4f} [{acc_lo:.4f}–{acc_hi:.4f}]  "
                f"F1={f1:.4f}  AUC={auc_val:.4f}")
    return {
        "label": label, "accuracy": round(acc, 4),
        "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "f1_macro": round(f1, 4),
        "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "roc_auc": round(auc_val, 4),
        "confusion_matrix": cm.tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  PART 1 — ROOT-CAUSE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def part1_feature_shift(X_src, X_tgt, feat_names):
    logger.info("=" * 60)
    logger.info("  PART 1: FEATURE DISTRIBUTION SHIFT ANALYSIS")
    logger.info("=" * 60)

    records = []
    for i, fname in enumerate(feat_names):
        src_col = X_src[:, i]
        tgt_col = X_tgt[:, i]

        # KS test
        ks_stat, ks_p = sp_stats.ks_2samp(src_col, tgt_col)

        # Cohen's d
        pooled_std = np.sqrt((np.var(src_col) + np.var(tgt_col)) / 2)
        cohens_d = (np.mean(tgt_col) - np.mean(src_col)) / pooled_std if pooled_std > 0 else 0.0

        # Feature group (prefix before first underscore)
        group = fname.split("_")[0]

        records.append({
            "feature": fname,
            "group": group,
            "src_mean": round(float(np.mean(src_col)), 4),
            "src_std": round(float(np.std(src_col)), 4),
            "tgt_mean": round(float(np.mean(tgt_col)), 4),
            "tgt_std": round(float(np.std(tgt_col)), 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_p),
            "cohens_d": round(float(cohens_d), 4),
            "significant": ks_p < 0.05,
        })

    shift_df = pd.DataFrame(records).sort_values("ks_statistic", ascending=False)
    shift_df.to_csv(OUT_DIR / "feature_shift_analysis.csv", index=False)

    n_sig = shift_df["significant"].sum()
    logger.info(f"  Features with significant shift (p<0.05): {n_sig}/{len(feat_names)}")
    logger.info(f"  Top 5 drifting features:")
    for _, row in shift_df.head(5).iterrows():
        logger.info(f"    {row['feature']:>40s}  KS={row['ks_statistic']:.3f}  "
                     f"d={row['cohens_d']:.2f}  p={row['ks_pvalue']:.2e}")

    # ── Figure: Feature Drift Heatmap ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    pivot = shift_df.set_index("feature")[["ks_statistic"]].T
    # Reorder by KS value
    order = shift_df.sort_values("ks_statistic", ascending=True)["feature"].tolist()
    pivot = pivot[order]

    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5,
                ax=ax, cbar_kws={"label": "KS Statistic"}, vmin=0, vmax=1)
    ax.set_title("Feature Distribution Shift: CIC-MalMem-2022 → MalVol-25\n"
                 "(Kolmogorov-Smirnov Statistic per Feature)", fontsize=13)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_drift_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success(f"  Saved feature_drift_heatmap.png")

    # ── Figure: Grouped Drift Bar Chart ──────────────────────────────
    group_drift = shift_df.groupby("group")["ks_statistic"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    group_drift.plot(kind="barh", ax=ax, color=sns.color_palette("rocket", len(group_drift)),
                     edgecolor="black")
    ax.set_xlabel("Mean KS Statistic")
    ax.set_title("Distribution Shift by Feature Group")
    ax.axvline(0.5, color="red", linestyle="--", alpha=0.7, label="High drift threshold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "grouped_drift_bars.png", dpi=300)
    plt.close()
    logger.success(f"  Saved grouped_drift_bars.png")

    return shift_df


# ═══════════════════════════════════════════════════════════════════════
#  PART 2 — DOMAIN ADAPTATION DEFENCES
# ═══════════════════════════════════════════════════════════════════════

def defence_baseline(X_src, y_src, X_tgt, y_tgt):
    """D0: No adaptation (baseline)."""
    model = train_ensemble(X_src, y_src)
    return evaluate_on_target(model, X_tgt, y_tgt, label="D0: No Adaptation (Baseline)")


def defence_zscore(X_src, y_src, X_tgt, y_tgt):
    """D1: Per-dataset Z-Score normalisation."""
    scaler_src = StandardScaler().fit(X_src)
    scaler_tgt = StandardScaler().fit(X_tgt)
    X_src_z = scaler_src.transform(X_src)
    X_tgt_z = scaler_tgt.transform(X_tgt)
    model = train_ensemble(X_src_z, y_src)
    return evaluate_on_target(model, X_tgt_z, y_tgt, label="D1: Z-Score Normalisation")


def defence_feature_selection(X_src, y_src, X_tgt, y_tgt, shift_df, feat_names,
                               drop_threshold=0.7):
    """D2: Drop features with KS > threshold, retrain on stable features only."""
    stable_feats = shift_df[shift_df["ks_statistic"] < drop_threshold]["feature"].tolist()
    stable_idx   = [feat_names.index(f) for f in stable_feats]
    logger.info(f"  D2: Keeping {len(stable_idx)}/{len(feat_names)} stable features "
                f"(KS < {drop_threshold})")
    X_src_s = X_src[:, stable_idx]
    X_tgt_s = X_tgt[:, stable_idx]
    # Also z-score normalise the selected features
    scaler_src = StandardScaler().fit(X_src_s)
    scaler_tgt = StandardScaler().fit(X_tgt_s)
    X_src_s = scaler_src.transform(X_src_s)
    X_tgt_s = scaler_tgt.transform(X_tgt_s)
    model = train_ensemble(X_src_s, y_src)
    result = evaluate_on_target(model, X_tgt_s, y_tgt,
                                 label=f"D2: Feature Selection (KS<{drop_threshold})")
    result["n_features_kept"] = len(stable_idx)
    result["features_dropped"] = [f for f in feat_names if f not in stable_feats]
    return result


def defence_coral(X_src, y_src, X_tgt, y_tgt):
    """
    D3: CORAL (CORrelation ALignment).
    Aligns the source covariance to the target covariance so the ensemble
    sees the same second-order statistics at test time.

    Reference: Sun, Feng & Saenko, "Return of Frustratingly Easy Domain Adaptation", AAAI 2016.
    """
    # Z-score normalise first
    scaler_src = StandardScaler().fit(X_src)
    scaler_tgt = StandardScaler().fit(X_tgt)
    Xs = scaler_src.transform(X_src)
    Xt = scaler_tgt.transform(X_tgt)

    # Compute covariance matrices
    n_feat = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(n_feat) * 1e-6  # regularise
    Ct = np.cov(Xt, rowvar=False) + np.eye(n_feat) * 1e-6

    # Whitening transform: Cs^{-1/2}
    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-8)
    Cs_neg_half = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T

    # Re-colouring transform: Ct^{1/2}
    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-8)
    Ct_half = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T

    # CORAL transform: whiten source, re-colour to target
    A_coral = Cs_neg_half @ Ct_half
    Xs_aligned = Xs @ A_coral

    model = train_ensemble(Xs_aligned, y_src)
    return evaluate_on_target(model, Xt, y_tgt, label="D3: CORAL Alignment")


def part2_defences(X_src, y_src, X_tgt, y_tgt, shift_df, feat_names):
    logger.info("=" * 60)
    logger.info("  PART 2: DOMAIN ADAPTATION DEFENCES")
    logger.info("=" * 60)

    results = []

    # D0: Baseline
    logger.info("\n--- D0: No Adaptation ---")
    r0 = defence_baseline(X_src, y_src, X_tgt, y_tgt)
    results.append(r0)

    # D1: Z-Score
    logger.info("\n--- D1: Z-Score Normalisation ---")
    r1 = defence_zscore(X_src, y_src, X_tgt, y_tgt)
    results.append(r1)

    # D2: Feature Selection (try multiple thresholds)
    logger.info("\n--- D2: Drift-Aware Feature Selection ---")
    best_d2 = None
    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        r2 = defence_feature_selection(X_src, y_src, X_tgt, y_tgt,
                                        shift_df, feat_names, drop_threshold=thresh)
        results.append(r2)
        if best_d2 is None or r2["accuracy"] > best_d2["accuracy"]:
            best_d2 = r2

    # D3: CORAL
    logger.info("\n--- D3: CORAL Alignment ---")
    r3 = defence_coral(X_src, y_src, X_tgt, y_tgt)
    results.append(r3)

    # Save all results
    with open(OUT_DIR / "defence_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.success(f"  Defence results saved to defence_results.json")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════

def generate_figures(results, shift_df):
    logger.info("=" * 60)
    logger.info("  GENERATING Q1 PUBLICATION FIGURES")
    logger.info("=" * 60)

    # ── Figure: Defence Comparison Bar Chart ──────────────────────────
    # Pick only the key entries
    key_labels = ["D0: No Adaptation (Baseline)",
                  "D1: Z-Score Normalisation",
                  "D3: CORAL Alignment"]
    # Find the best D2 result
    d2_results = [r for r in results if r["label"].startswith("D2:")]
    if d2_results:
        best_d2 = max(d2_results, key=lambda r: r["accuracy"])
        key_labels.append(best_d2["label"])

    plot_data = []
    for r in results:
        if r["label"] in key_labels:
            short = r["label"].split(":")[0] + ":" + r["label"].split(":")[1][:25]
            plot_data.append({
                "Defence": short.strip(),
                "Accuracy": r["accuracy"],
                "F1-Macro": r["f1_macro"],
                "CI_lo": r["accuracy_ci"][0],
                "CI_hi": r["accuracy_ci"][1],
            })

    df_plot = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(df_plot))
    width = 0.35
    bars1 = ax.bar(x - width/2, df_plot["Accuracy"],  width, label="Accuracy",
                   color="#4c72b0", edgecolor="black", zorder=3)
    bars2 = ax.bar(x + width/2, df_plot["F1-Macro"],  width, label="F1-Macro",
                   color="#55a868", edgecolor="black", zorder=3)

    # Error bars on accuracy
    errs = np.array([[df_plot["Accuracy"].iloc[i] - df_plot["CI_lo"].iloc[i],
                       df_plot["CI_hi"].iloc[i] - df_plot["Accuracy"].iloc[i]]
                      for i in range(len(df_plot))]).T
    ax.errorbar(x - width/2, df_plot["Accuracy"], yerr=errs,
                fmt="none", ecolor="black", capsize=4, zorder=4)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random Chance (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Defence"], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Dataset Defence Comparison\n(Trained: CIC-MalMem-2022 → Tested: MalVol-25)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "defence_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("  Saved defence_comparison.png")

    # ── Figure: Confusion Matrix Grid ─────────────────────────────────
    key_results = [r for r in results if r["label"] in key_labels]
    n = len(key_results)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    if n == 1:
        axes = [axes]
    cmaps = ["Greys", "Blues", "Greens", "Oranges"]
    for i, r in enumerate(key_results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmaps[i % len(cmaps)], ax=axes[i],
                    xticklabels=["Benign", "Malware"], yticklabels=["Benign", "Malware"])
        short_label = r["label"].split(":")[0]
        axes[i].set_title(f"{short_label}\nAcc={r['accuracy']:.0%}", fontsize=10)
        axes[i].set_ylabel("True" if i == 0 else "")
        axes[i].set_xlabel("Predicted")
    plt.suptitle("Confusion Matrices: Effect of Domain Adaptation on MalVol-25", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "defence_confusion_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("  Saved defence_confusion_grid.png")

    # ── Figure: Drift-Accuracy Correlation ────────────────────────────
    # Show that as we drop MORE drifting features, accuracy changes
    d2_results_all = [r for r in results if r["label"].startswith("D2:")]
    if d2_results_all:
        fig, ax = plt.subplots(figsize=(7, 5))
        n_feats  = [r.get("n_features_kept", 41) for r in d2_results_all]
        accs     = [r["accuracy"] for r in d2_results_all]
        ax.plot(n_feats, accs, "o-", color="#d62728", lw=2, markersize=8, zorder=3)
        for nf, ac in zip(n_feats, accs):
            ax.annotate(f"{ac:.0%}", (nf, ac), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
        ax.axhline(0.5, color="grey", linestyle="--", alpha=0.5, label="Random Chance")
        ax.set_xlabel("Number of Features Retained (after drift filtering)")
        ax.set_ylabel("Cross-Dataset Accuracy (MalVol-25)")
        ax.set_title("Impact of Drift-Aware Feature Selection\non Cross-Dataset Generalisation")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "drift_vs_accuracy.png", dpi=300)
        plt.close()
        logger.success("  Saved drift_vs_accuracy.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("Loading datasets...")
    X_src, y_src, X_tgt, y_tgt, feat_names = load_unified()
    logger.info(f"  Source (CIC):   {X_src.shape[0]:,} × {X_src.shape[1]}")
    logger.info(f"  Target (MalVol): {X_tgt.shape[0]:,} × {X_tgt.shape[1]}")

    # Part 1: Root Cause
    shift_df = part1_feature_shift(X_src, X_tgt, feat_names)

    # Part 2: Defences
    results = part2_defences(X_src, y_src, X_tgt, y_tgt, shift_df, feat_names)

    # Part 3: Figures
    generate_figures(results, shift_df)

    # ── Final Summary Table ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 70)
    for r in results:
        ci = r.get("accuracy_ci", [0, 0])
        print(f"  {r['label']:>50s}  |  Acc={r['accuracy']:.4f}  "
              f"[{ci[0]:.4f}–{ci[1]:.4f}]  F1={r['f1_macro']:.4f}")
    print("=" * 70)

    best = max(results, key=lambda r: r["accuracy"])
    baseline = results[0]
    delta = best["accuracy"] - baseline["accuracy"]
    print(f"\n  Best defence: {best['label']}")
    print(f"  Improvement over baseline: +{delta:.1%} absolute accuracy")
    print(f"  Baseline: {baseline['accuracy']:.1%} → Best: {best['accuracy']:.1%}")
    print("=" * 70 + "\n")
    logger.success("All done. Results in results/unified/")


if __name__ == "__main__":
    main()
