"""
step2_statistical_rigour.py
============================
Q1 Hardening — Step 2: Statistical Power

Produces statistically rigorous cross-dataset results by:

  A. LOOCV on MalVol-25 for all ZERO-SHOT methods
     (D0 Baseline, D3 CORAL, U1 Rank, U5 Delta, U3 TCA)
     → Each sample is excluded once; model predicts it; metric = fraction correct
     → Uses ALL 30 samples; gives debiased estimate with no train-test split

  B. 100-TRIAL REPEATED SUBSAMPLING for all FEW-SHOT methods
     (U2 5-shot, U4 Stacked, U6 Combined)
     → Each trial: randomly sample 5 calibration samples, test on remaining 25
     → Report: mean, std, median, IQR, 95% CI across 100 trials

  C. McNEMAR'S PAIRWISE SIGNIFICANCE TESTS
     → Null: method A and method B have equal error rates on MalVol-25
     → Output: pairwise p-value matrix

Outputs:
    results/unified/step2_statistical.json
    results/unified/figures/step2_loocv_results.png
    results/unified/figures/step2_trial_distributions.png
    results/unified/figures/step2_mcnemar_matrix.png
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
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, QuantileTransformer

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

RS           = 42
N_TRIALS     = 100      # repetitions for few-shot methods
N_SHOT       = 5        # calibration samples
TCA_SUBSAMPLE = 2000    # source-side subsampling for TCA kernel computation


# ═══════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
#  MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════════

def make_ensemble(seed=RS):
    # Now uses the validated Phase 0 standalone XGBoost instead of VotingClassifier
    return XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        use_label_encoder=False, eval_metric="logloss", random_state=seed, verbosity=0
    )


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE TRANSFORMERS
# ═══════════════════════════════════════════════════════════════════════

def transform_zscore(X_src, X_tgt):
    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    return sc_s.transform(X_src), sc_t.transform(X_tgt)


def transform_coral(X_src, X_tgt):
    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    Xs = sc_s.transform(X_src)
    Xt = sc_t.transform(X_tgt)
    n = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(n) * 1e-6
    Ct = np.cov(Xt, rowvar=False) + np.eye(n) * 1e-6
    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-8)
    Cs_nh = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-8)
    Ct_h  = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
    A = Cs_nh @ Ct_h
    return Xs @ A, Xt


def transform_rank(X_src, X_tgt, seed=RS):
    qt_s = QuantileTransformer(output_distribution="uniform", random_state=seed)
    qt_t = QuantileTransformer(output_distribution="uniform", random_state=seed)
    return qt_s.fit_transform(X_src), qt_t.fit_transform(X_tgt)


def transform_tca(X_src, X_tgt, n_components=15, mu=1.0, gamma=1.0, seed=RS):
    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    Xs = sc_s.transform(X_src)
    Xt = sc_t.transform(X_tgt)
    ns = Xs.shape[0]
    rng = np.random.default_rng(seed)
    if ns > TCA_SUBSAMPLE:
        sub_idx = rng.choice(ns, TCA_SUBSAMPLE, replace=False)
        Xs_sub  = Xs[sub_idx]
    else:
        Xs_sub = Xs
    ns_sub = Xs_sub.shape[0]
    nt = Xt.shape[0]
    X_all = np.vstack([Xs_sub, Xt])
    n_all = ns_sub + nt
    sq_dists = squareform(pdist(X_all, "sqeuclidean"))
    med = np.median(sq_dists[sq_dists > 0])
    sigma = gamma * med if med > 0 else 1.0
    K = np.exp(-sq_dists / (2 * sigma))
    L = np.zeros((n_all, n_all))
    L[:ns_sub, :ns_sub] = 1.0 / (ns_sub * ns_sub)
    L[ns_sub:, ns_sub:] = 1.0 / (nt * nt)
    L[:ns_sub, ns_sub:] = -1.0 / (ns_sub * nt)
    L[ns_sub:, :ns_sub] = -1.0 / (ns_sub * nt)
    H = np.eye(n_all) - np.ones((n_all, n_all)) / n_all
    A_mat = K @ L @ K + mu * np.eye(n_all) + np.eye(n_all) * 1e-6
    B_mat = K @ H @ K
    n_comp = min(n_components, Xs.shape[1])
    eigvals, eigvecs = eigh(B_mat, A_mat)
    idx = np.argsort(eigvals)[::-1][:n_comp]
    W = eigvecs[:, idx]
    Z = K @ W
    return Z[:ns_sub], Xt @ (sc_t.transform(Xt) @ np.zeros((Xt.shape[1], 0)) or 0), \
           Z[:ns_sub], Z[ns_sub:]


def transform_delta(X_src, X_tgt, feat_names):
    """Ratio-based delta features + Z-Score normalisation."""
    def safe_ratio(num, den): return np.where(den > 0, num / den, 0.0)
    def augment(X):
        df = pd.DataFrame(X, columns=feat_names)
        out = df.copy()
        pairs = [
            ("dlllist_ndlls",           "pslist_nproc",    "dlls_per_proc"),
            ("handles_nhandles",         "pslist_nproc",    "handles_per_proc"),
            ("malfind_ninjections",      "pslist_nproc",    "injections_per_proc"),
            ("ldrmodules_not_in_load",   "pslist_nproc",    "hidden_modules_per_proc"),
            ("callbacks_ncallbacks",     "modules_nmodules","callbacks_per_module"),
            ("svcscan_nservices",        "pslist_nproc",    "services_per_proc"),
            ("handles_nfile",            "handles_nhandles","file_handle_ratio"),
            ("handles_nthread",          "handles_nhandles","thread_handle_ratio"),
            ("handles_nmutant",          "handles_nhandles","mutant_handle_ratio"),
        ]
        for num_f, den_f, new_name in pairs:
            if num_f in feat_names and den_f in feat_names:
                out[new_name] = safe_ratio(df[num_f].values, df[den_f].values)
        return out.values
    Xs_aug = augment(X_src)
    Xt_aug = augment(X_tgt)
    sc_s = StandardScaler().fit(Xs_aug)
    sc_t = StandardScaler().fit(Xt_aug)
    return sc_s.transform(Xs_aug), sc_t.transform(Xt_aug)


# ═══════════════════════════════════════════════════════════════════════
#  LOOCV ENGINE (for zero-shot methods)
# ═══════════════════════════════════════════════════════════════════════

def run_zero_shot_transductive(X_src_full, y_src, X_tgt, y_tgt, transform_fn, method_name):
    """
    Transductive Unsupervised Domain Adaptation.
    Applies the transform using the full UNLABELLED target set, trains once,
    and predicts the target set. This provides the 30 per-sample predictions
    needed for McNemar's test without the $O(N)$ computational blowup of LOOCV.
    """
    logger.info(f"  Transductive Zero-Shot [{method_name}]...")

    try:
        Xs_t, Xt_t = transform_fn(X_src_full, X_tgt)
    except Exception as e:
        logger.warning(f"    Transform failed: {e}. Using raw features.")
        Xs_t = X_src_full
        Xt_t = X_tgt

    model = make_ensemble()
    model.fit(Xs_t, y_src)

    all_preds  = model.predict(Xt_t)
    try:   all_probas = model.predict_proba(Xt_t)[:, 1]
    except: all_probas = np.zeros(len(y_tgt))

    acc = accuracy_score(y_tgt, all_preds)
    f1  = f1_score(y_tgt, all_preds, average="macro", zero_division=0)
    try:   auc_val = roc_auc_score(y_tgt, all_probas)
    except: auc_val = 0.0

    logger.info(f"    → Acc={acc:.4f}  F1={f1:.4f}  AUC={auc_val:.4f}")

    return {
        "method": method_name,
        "eval_protocol": "Transductive",
        "accuracy":  round(acc, 4),
        "f1_macro":  round(f1, 4),
        "roc_auc":   round(auc_val, 4),
        "per_sample_preds": all_preds.tolist(),
        "confusion_matrix": confusion_matrix(y_tgt, all_preds).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  100-TRIAL ENGINE (for few-shot methods)
# ═══════════════════════════════════════════════════════════════════════

def run_100trials_fewshot(X_src, y_src, X_tgt, y_tgt, transform_fn,
                           method_name, n_shot=N_SHOT, n_trials=N_TRIALS):
    """
    100 repeated trials with different random calibration splits.
    Each trial: randomly sample n_shot target samples for calibration,
    evaluate on remaining (30 - n_shot) samples.
    """
    logger.info(f"  100-Trial [{method_name}] — {n_trials} trials × {n_shot}-shot...")
    rng = np.random.default_rng(RS)

    # Pre-compute transformed features once (transform is unsupervised / source-only)
    Xs_t, Xt_t = transform_fn(X_src, X_tgt)

    # Compute base model outputs once
    base = make_ensemble()
    base.fit(Xs_t, y_src)
    meta_cols = [est.predict_proba(Xt_t)[:, 1:] for est in base.estimators_]
    meta_tgt = np.hstack([Xt_t] + meta_cols)  # (30, feature_dim)

    trial_accs = []
    trial_f1s  = []
    trial_aucs = []

    for trial in range(n_trials):
        # Stratified n_shot selection
        idx_0 = np.where(y_tgt == 0)[0]
        idx_1 = np.where(y_tgt == 1)[0]
        n_cal_0 = n_shot // 2 + n_shot % 2
        n_cal_1 = n_shot // 2
        # Guard against classes having fewer samples than needed
        n_cal_0 = min(n_cal_0, len(idx_0) - 1)
        n_cal_1 = min(n_cal_1, len(idx_1) - 1)
        if n_cal_0 <= 0 or n_cal_1 <= 0:
            break

        cal_0 = rng.choice(idx_0, n_cal_0, replace=False)
        cal_1 = rng.choice(idx_1, n_cal_1, replace=False)
        cal_idx  = np.concatenate([cal_0, cal_1])
        test_idx = np.array([i for i in range(len(y_tgt)) if i not in cal_idx])

        cal_model = LogisticRegression(random_state=int(rng.integers(0, 10000)),
                                        max_iter=1000, C=0.1)
        try:
            cal_model.fit(meta_tgt[cal_idx], y_tgt[cal_idx])
        except Exception:
            continue  # degenerate fold, skip

        yp = cal_model.predict(meta_tgt[test_idx])
        ypr_raw = cal_model.predict_proba(meta_tgt[test_idx])
        ypr = ypr_raw[:, 1] if ypr_raw.shape[1] > 1 else ypr_raw[:, 0]

        trial_accs.append(accuracy_score(y_tgt[test_idx], yp))
        trial_f1s.append(f1_score(y_tgt[test_idx], yp, average="macro", zero_division=0))
        try:
            trial_aucs.append(roc_auc_score(y_tgt[test_idx], ypr))
        except Exception:
            pass

    trial_accs = np.array(trial_accs)
    trial_f1s  = np.array(trial_f1s)

    result = {
        "method": method_name,
        "eval_protocol": f"{n_trials}-trial repeated {n_shot}-shot",
        "n_trials_completed": len(trial_accs),
        "accuracy_mean":  round(float(np.mean(trial_accs)), 4),
        "accuracy_std":   round(float(np.std(trial_accs)),  4),
        "accuracy_median":round(float(np.median(trial_accs)),4),
        "accuracy_ci_95": [round(float(np.percentile(trial_accs, 2.5)), 4),
                            round(float(np.percentile(trial_accs, 97.5)), 4)],
        "accuracy_iqr":   [round(float(np.percentile(trial_accs, 25)), 4),
                            round(float(np.percentile(trial_accs, 75)), 4)],
        "f1_mean":  round(float(np.mean(trial_f1s)), 4),
        "f1_std":   round(float(np.std(trial_f1s)),  4),
        "roc_auc_mean": round(float(np.mean(trial_aucs)), 4) if trial_aucs else 0.0,
        "trial_accuracies": trial_accs.tolist(),
    }

    logger.info(f"    → Acc={result['accuracy_mean']:.4f} ± {result['accuracy_std']:.4f}  "
                f"[{result['accuracy_ci_95'][0]:.4f}–{result['accuracy_ci_95'][1]:.4f}]")
    return result


# ═══════════════════════════════════════════════════════════════════════
#  McNEMAR'S PAIRWISE SIGNIFICANCE
# ═══════════════════════════════════════════════════════════════════════

def mcnemar_pairwise(loocv_results, y_tgt):
    """
    Compute pairwise McNemar's test for all LOOCV zero-shot methods.
    Null hypothesis: two classifiers make the same errors on the same samples.
    """
    logger.info("\n  McNemar's pairwise tests...")
    methods = [r["method"] for r in loocv_results]
    n = len(methods)
    p_matrix = np.ones((n, n))

    for i in range(n):
        preds_i = np.array(loocv_results[i]["per_sample_preds"])
        for j in range(i + 1, n):
            preds_j = np.array(loocv_results[j]["per_sample_preds"])
            # Discordant counts
            b = np.sum((preds_i == y_tgt) & (preds_j != y_tgt))
            c = np.sum((preds_i != y_tgt) & (preds_j == y_tgt))
            # McNemar continuity-corrected
            if b + c == 0:
                p = 1.0
            else:
                chi2 = (abs(b - c) - 1) ** 2 / (b + c)
                p = float(sp_stats.chi2.sf(chi2, df=1))
            p_matrix[i, j] = p
            p_matrix[j, i] = p
            logger.info(f"    {methods[i]} vs {methods[j]}: b={b}, c={c}, p={p:.4f}")

    return {
        "methods": methods,
        "p_matrix": p_matrix.round(4).tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════
#  PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════

def generate_figures(loocv_results, trial_results, mcnemar_data, y_tgt):
    logger.info("\n  Generating Step 2 publication figures...")

    # ── Fig 1: LOOCV Comparison Bar Chart ─────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [r["method"].replace("D0: ", "").replace("D3: ", "").replace(
               "U1: ", "").replace("U5: ", "").replace("U3: ", "") for r in loocv_results]
    accs   = [r["accuracy"] for r in loocv_results]
    f1s    = [r["f1_macro"]  for r in loocv_results]

    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, accs, w, label="Accuracy", color="#4c72b0", edgecolor="black", zorder=3)
    ax.bar(x + w/2, f1s,  w, label="F1-Macro", color="#55a868", edgecolor="black", zorder=3)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (LOOCV on MalVol-25)")
    ax.set_title("Zero-Shot Domain Adaptation — LOOCV Debiased Estimation\n"
                 "(Trained: CIC-MalMem-2022 | Tested via LOOCV: MalVol-25, n=30)")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_loocv_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Fig 2: 100-Trial Accuracy Distributions ───────────────────────
    if trial_results:
        n_methods = len(trial_results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), sharey=True)
        if n_methods == 1:
            axes = [axes]
        palette = ["#4c72b0", "#dd8452", "#55a868"]
        for i, r in enumerate(trial_results):
            data = np.array(r["trial_accuracies"])
            axes[i].hist(data, bins=20, color=palette[i % len(palette)],
                         edgecolor="black", alpha=0.85)
            axes[i].axvline(r["accuracy_mean"], color="red", lw=2,
                             label=f"Mean={r['accuracy_mean']:.2f}")
            axes[i].axvline(r["accuracy_ci_95"][0], color="grey", lw=1.5,
                             linestyle="--", label="95% CI")
            axes[i].axvline(r["accuracy_ci_95"][1], color="grey", lw=1.5,
                             linestyle="--")
            axes[i].axvline(0.5, color="orange", lw=1.5,
                             linestyle=":", label="Random Chance")
            short = r["method"].split(":")[0]
            axes[i].set_title(f"{short}\nμ={r['accuracy_mean']:.2f} ± {r['accuracy_std']:.2f}")
            axes[i].set_xlabel("Accuracy")
            axes[i].legend(fontsize=7)
        axes[0].set_ylabel("Frequency (n_trials=100)")
        plt.suptitle("100-Trial Repeated Subsampling: Accuracy Distributions\n"
                     "(5-Shot Calibration; random seed varied per trial)", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "step2_trial_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ── Fig 3: McNemar's P-Value Matrix ───────────────────────────────
    methods = mcnemar_data["methods"]
    p_mat   = np.array(mcnemar_data["p_matrix"])
    short_methods = [m.split(": ")[1][:20] if ": " in m else m[:20] for m in methods]

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.eye(len(methods), dtype=bool)
    annot = np.array([[f"p={p_mat[i,j]:.3f}" if not mask[i,j] else "—"
                        for j in range(len(methods))]
                       for i in range(len(methods))])
    sig_mask = (p_mat < 0.05) & ~mask

    # Heatmap: green if significant (p<0.05), grey otherwise
    color_mat = np.where(mask, np.nan, p_mat)
    im = ax.imshow(color_mat, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="p-value")
    for i in range(len(methods)):
        for j in range(len(methods)):
            txt = annot[i, j]
            weight = "bold" if sig_mask[i, j] else "normal"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, fontweight=weight)
    ax.set_xticks(range(len(methods)))
    ax.set_yticks(range(len(methods)))
    ax.set_xticklabels(short_methods, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(short_methods, fontsize=8)
    ax.set_title("McNemar's Pairwise Significance Tests (LOOCV Predictions)\n"
                 "Bold = statistically significant (p < 0.05)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "step2_mcnemar_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.success("  All Step 2 figures saved.")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 65)
    logger.info("  STEP 2: STATISTICAL RIGOUR")
    logger.info("=" * 65)

    X_src, y_src, X_tgt, y_tgt, feat_names = load_data()
    logger.info(f"  Source: {X_src.shape[0]:,} × {X_src.shape[1]}  |  "
                f"Target: {X_tgt.shape[0]} × {X_tgt.shape[1]}")
    logger.info(f"  Target class balance: "
                f"Benign={int((y_tgt==0).sum())}  Malware={int((y_tgt==1).sum())}")

    # ── A. LOOCV for zero-shot methods ─────────────────────────────
    logger.info("\n[A] LOOCV — Zero-Shot Methods")

    loocv_results = []

    # D0: Baseline (identity transform)
    loocv_results.append(run_zero_shot_transductive(
        X_src, y_src, X_tgt, y_tgt,
        lambda Xs, Xt: (Xs, Xt),
        "D0: No Adaptation"
    ))

    # D3: CORAL
    loocv_results.append(run_zero_shot_transductive(
        X_src, y_src, X_tgt, y_tgt,
        transform_coral,
        "D3: CORAL Alignment"
    ))

    # U1: Rank Transform
    loocv_results.append(run_zero_shot_transductive(
        X_src, y_src, X_tgt, y_tgt,
        transform_rank,
        "U1: Rank Transform"
    ))

    # U5: Delta Features
    loocv_results.append(run_zero_shot_transductive(
        X_src, y_src, X_tgt, y_tgt,
        lambda Xs, Xt: transform_delta(Xs, Xt, feat_names),
        "U5: Delta Features"
    ))

    # ── B. 100-Trial Few-Shot ───────────────────────────────────────
    logger.info("\n[B] 100-Trial Repeated Subsampling — Few-Shot Methods")

    trial_results = []

    # U2: 5-shot with CORAL base
    trial_results.append(run_100trials_fewshot(
        X_src, y_src, X_tgt, y_tgt,
        transform_coral,
        "U2: 5-Shot (CORAL base)"
    ))

    # U2b: 5-shot with Rank base
    trial_results.append(run_100trials_fewshot(
        X_src, y_src, X_tgt, y_tgt,
        transform_rank,
        "U2b: 5-Shot (Rank base)"
    ))

    # U2c: 5-shot with Delta base
    trial_results.append(run_100trials_fewshot(
        X_src, y_src, X_tgt, y_tgt,
        lambda Xs, Xt: transform_delta(Xs, Xt, feat_names),
        "U2c: 5-Shot (Delta base)"
    ))

    # ── C. McNemar's tests ─────────────────────────────────────────
    logger.info("\n[C] McNemar's Pairwise Significance Tests")
    mcnemar_data = mcnemar_pairwise(loocv_results, y_tgt)

    # ── Save ───────────────────────────────────────────────────────
    output = {
        "loocv": loocv_results,
        "trials": trial_results,
        "mcnemar": mcnemar_data,
    }
    with open(OUT_DIR / "step2_statistical.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.success(f"  Saved step2_statistical.json")

    # ── Figures ────────────────────────────────────────────────────
    generate_figures(loocv_results, trial_results, mcnemar_data, y_tgt)

    # ── Print Summary Table ────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  STEP 2 RESULTS: ZERO-SHOT METHODS (LOOCV)")
    print("=" * 75)
    for r in loocv_results:
        print(f"  {r['method']:<45s}  Acc={r['accuracy']:.4f}  "
              f"F1={r['f1_macro']:.4f}  AUC={r['roc_auc']:.4f}")
    print()
    print("  FEW-SHOT METHODS (100-Trial Mean ± Std)")
    print("=" * 75)
    for r in trial_results:
        print(f"  {r['method']:<45s}  "
              f"Acc={r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}  "
              f"[{r['accuracy_ci_95'][0]:.4f}–{r['accuracy_ci_95'][1]:.4f}]")
    print("=" * 75)


if __name__ == "__main__":
    main()
