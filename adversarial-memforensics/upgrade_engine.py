"""
upgrade_engine.py
=================
Implements all 5 domain adaptation upgrades and produces a comprehensive
comparison table with bootstrap 95% CIs.

    U1: Rank-Based Feature Transformation
    U2: Few-Shot Fine-Tuning (5-shot calibration)
    U3: Transfer Component Analysis (TCA via MMD minimisation)
    U4: Stacked Ensemble with Domain-Aware Meta-Learner
    U5: Delta-Feature Engineering (ratio-based invariants)
    U6: Combined Pipeline (U1+U3+U2 stacked)

Outputs:
    results/unified/upgrade_results.json
    results/unified/figures/upgrade_comparison.png
    results/unified/figures/upgrade_progression.png
    results/unified/figures/upgrade_confusion_grid.png
"""
import sys, json, copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import joblib

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
N_BOOTSTRAP = 1000


# ═══════════════════════════════════════════════════════════════════════
#  UTILITIES
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


def make_ensemble():
    rf   = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS)
    xgb  = XGBClassifier(n_estimators=200, use_label_encoder=False,
                          eval_metric="logloss", random_state=RS, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, random_state=RS, verbosity=-1)
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                             voting="soft", n_jobs=-1)


def bootstrap_ci(y_true, y_pred, metric_fn, n=N_BOOTSTRAP):
    rng = np.random.default_rng(RS)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try: scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except: pass
    if not scores: return 0.0, 0.0, 0.0
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def eval_result(y_true, y_pred, y_proba, label):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    try: auc_val = roc_auc_score(y_true, y_proba)
    except: auc_val = 0.0
    _, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, accuracy_score)
    _, f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred,
                         lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    logger.info(f"  {label:<55s} Acc={acc:.4f} [{acc_lo:.4f}–{acc_hi:.4f}]  "
                f"F1={f1:.4f}  AUC={auc_val:.4f}")
    return {
        "label": label, "accuracy": round(acc, 4),
        "accuracy_ci": [round(acc_lo, 4), round(acc_hi, 4)],
        "f1_macro": round(f1, 4), "f1_ci": [round(f1_lo, 4), round(f1_hi, 4)],
        "roc_auc": round(auc_val, 4), "confusion_matrix": cm,
    }


# ═══════════════════════════════════════════════════════════════════════
#  D0: BASELINE (NO ADAPTATION)
# ═══════════════════════════════════════════════════════════════════════

def run_baseline(X_src, y_src, X_tgt, y_tgt):
    logger.info("\n--- D0: No Adaptation (Baseline) ---")
    model = make_ensemble()
    model.fit(X_src, y_src)
    yp = model.predict(X_tgt)
    ypr = model.predict_proba(X_tgt)[:, 1]
    return eval_result(y_tgt, yp, ypr, "D0: No Adaptation (Baseline)")


# ═══════════════════════════════════════════════════════════════════════
#  D3: CORAL (from previous run, included for completeness)
# ═══════════════════════════════════════════════════════════════════════

def run_coral(X_src, y_src, X_tgt, y_tgt):
    logger.info("\n--- D3: CORAL Alignment ---")
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
    Ct_h = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
    A = Cs_nh @ Ct_h
    Xs_a = Xs @ A
    model = make_ensemble()
    model.fit(Xs_a, y_src)
    yp = model.predict(Xt)
    ypr = model.predict_proba(Xt)[:, 1]
    return eval_result(y_tgt, yp, ypr, "D3: CORAL Alignment")


# ═══════════════════════════════════════════════════════════════════════
#  U1: RANK-BASED FEATURE TRANSFORMATION
# ═══════════════════════════════════════════════════════════════════════

def run_u1_rank(X_src, y_src, X_tgt, y_tgt):
    """
    Convert each feature to its within-dataset percentile rank.
    This makes features distribution-invariant: a high-rank process count
    means "unusually many processes for THIS environment" regardless of
    absolute value differences between VMs.
    """
    logger.info("\n--- U1: Rank-Based Feature Transformation ---")
    qt_src = QuantileTransformer(output_distribution="uniform", random_state=RS)
    qt_tgt = QuantileTransformer(output_distribution="uniform", random_state=RS)
    Xs = qt_src.fit_transform(X_src)
    Xt = qt_tgt.fit_transform(X_tgt)
    model = make_ensemble()
    model.fit(Xs, y_src)
    yp = model.predict(Xt)
    ypr = model.predict_proba(Xt)[:, 1]
    return eval_result(y_tgt, yp, ypr, "U1: Rank Transform (Quantile)")


# ═══════════════════════════════════════════════════════════════════════
#  U5: DELTA-FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════

def engineer_delta_features(X, feat_names):
    """
    Create ratio-based invariant features that measure RELATIVE behaviour
    rather than absolute system state.
    """
    df = pd.DataFrame(X, columns=feat_names)
    new_feats = pd.DataFrame()

    # Raw features passed through
    new_feats = df.copy()

    # Ratio features (guard against division by zero)
    def safe_ratio(num, den):
        return np.where(den > 0, num / den, 0.0)

    if "dlllist_ndlls" in feat_names and "pslist_nproc" in feat_names:
        new_feats["ratio_dlls_per_proc"] = safe_ratio(df["dlllist_ndlls"].values,
                                                       df["pslist_nproc"].values)
    if "handles_nhandles" in feat_names and "pslist_nproc" in feat_names:
        new_feats["ratio_handles_per_proc"] = safe_ratio(df["handles_nhandles"].values,
                                                          df["pslist_nproc"].values)
    if "malfind_ninjections" in feat_names and "pslist_nproc" in feat_names:
        new_feats["ratio_injections_per_proc"] = safe_ratio(df["malfind_ninjections"].values,
                                                             df["pslist_nproc"].values)
    if "ldrmodules_not_in_load" in feat_names and "pslist_nproc" in feat_names:
        new_feats["ratio_hidden_modules_per_proc"] = safe_ratio(
            df["ldrmodules_not_in_load"].values, df["pslist_nproc"].values)
    if "callbacks_ncallbacks" in feat_names and "modules_nmodules" in feat_names:
        new_feats["ratio_callbacks_per_module"] = safe_ratio(
            df["callbacks_ncallbacks"].values, df["modules_nmodules"].values)
    if "svcscan_nservices" in feat_names and "pslist_nproc" in feat_names:
        new_feats["ratio_services_per_proc"] = safe_ratio(
            df["svcscan_nservices"].values, df["pslist_nproc"].values)
    if "handles_nfile" in feat_names and "handles_nhandles" in feat_names:
        new_feats["ratio_file_handles"] = safe_ratio(
            df["handles_nfile"].values, df["handles_nhandles"].values)
    if "handles_nthread" in feat_names and "handles_nhandles" in feat_names:
        new_feats["ratio_thread_handles"] = safe_ratio(
            df["handles_nthread"].values, df["handles_nhandles"].values)
    if "handles_nmutant" in feat_names and "handles_nhandles" in feat_names:
        new_feats["ratio_mutant_handles"] = safe_ratio(
            df["handles_nmutant"].values, df["handles_nhandles"].values)

    return new_feats.values, list(new_feats.columns)


def run_u5_delta(X_src, y_src, X_tgt, y_tgt, feat_names):
    """U5: Delta features + Z-Score normalisation."""
    logger.info("\n--- U5: Delta-Feature Engineering ---")
    Xs, new_names = engineer_delta_features(X_src, feat_names)
    Xt, _         = engineer_delta_features(X_tgt, feat_names)
    logger.info(f"  Engineered {len(new_names)} features ({len(new_names)-len(feat_names)} new ratios)")
    sc_s = StandardScaler().fit(Xs)
    sc_t = StandardScaler().fit(Xt)
    Xs = sc_s.transform(Xs)
    Xt = sc_t.transform(Xt)
    model = make_ensemble()
    model.fit(Xs, y_src)
    yp = model.predict(Xt)
    ypr = model.predict_proba(Xt)[:, 1]
    return eval_result(y_tgt, yp, ypr, "U5: Delta Features + Z-Score")


# ═══════════════════════════════════════════════════════════════════════
#  U3: TRANSFER COMPONENT ANALYSIS (TCA)
# ═══════════════════════════════════════════════════════════════════════

def run_u3_tca(X_src, y_src, X_tgt, y_tgt, n_components=20, mu=1.0, gamma=1.0):
    """
    Transfer Component Analysis (Pan et al., IEEE TNN 2011).
    Learns a shared subspace minimising MMD between source and target.

    Steps:
        1. Construct combined kernel matrix K (RBF)
        2. Build MMD matrix L
        3. Solve generalised eigenproblem: (KLK + μI)W = KHK · Λ · W
        4. Project both domains into the top-k eigenvectors
    """
    logger.info("\n--- U3: Transfer Component Analysis (TCA) ---")

    # Standardise first
    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    Xs = sc_s.transform(X_src)
    Xt = sc_t.transform(X_tgt)

    ns, nt = Xs.shape[0], Xt.shape[0]
    n = ns + nt
    n_components = min(n_components, Xs.shape[1])

    # Subsample source for computational tractability
    rng = np.random.default_rng(RS)
    if ns > 2000:
        sub_idx = rng.choice(ns, 2000, replace=False)
        Xs_sub = Xs[sub_idx]
        ys_sub = y_src[sub_idx]
    else:
        Xs_sub = Xs
        ys_sub = y_src
    ns_sub = Xs_sub.shape[0]

    # Combined data
    X_all = np.vstack([Xs_sub, Xt])
    n_all = ns_sub + nt

    # RBF kernel
    logger.info(f"  Computing RBF kernel ({n_all} × {n_all})...")
    sq_dists = squareform(pdist(X_all, "sqeuclidean"))
    median_dist = np.median(sq_dists[sq_dists > 0])
    sigma = gamma * median_dist if median_dist > 0 else 1.0
    K = np.exp(-sq_dists / (2 * sigma))

    # MMD matrix L
    L = np.zeros((n_all, n_all))
    L[:ns_sub, :ns_sub] = 1.0 / (ns_sub * ns_sub)
    L[ns_sub:, ns_sub:] = 1.0 / (nt * nt)
    L[:ns_sub, ns_sub:] = -1.0 / (ns_sub * nt)
    L[ns_sub:, :ns_sub] = -1.0 / (ns_sub * nt)

    # Centering matrix H
    H = np.eye(n_all) - np.ones((n_all, n_all)) / n_all

    # Solve: (KLK + μI)^{-1} KHK → top eigenvectors
    logger.info(f"  Solving TCA eigenproblem ({n_all} dims, {n_components} components)...")
    A = K @ L @ K + mu * np.eye(n_all)
    B = K @ H @ K

    # Regularise
    A += np.eye(n_all) * 1e-6

    eigvals, eigvecs = eigh(B, A)
    # Take top-k (largest eigenvalues)
    idx = np.argsort(eigvals)[::-1][:n_components]
    W = eigvecs[:, idx]

    # Project
    Z = K @ W  # (n_all, n_components)
    Zs = Z[:ns_sub]
    Zt = Z[ns_sub:]

    logger.info(f"  Projected to {n_components}-dimensional TCA subspace")

    model = make_ensemble()
    model.fit(Zs, ys_sub)
    yp = model.predict(Zt)
    ypr = model.predict_proba(Zt)[:, 1]
    return eval_result(y_tgt, yp, ypr, "U3: TCA (MMD Minimisation)")


# ═══════════════════════════════════════════════════════════════════════
#  U2: FEW-SHOT FINE-TUNING
# ═══════════════════════════════════════════════════════════════════════

def run_u2_fewshot(X_src, y_src, X_tgt, y_tgt, n_shot=5):
    """
    5-shot domain adaptation:
    1. Train base ensemble on source domain (z-score normalised)
    2. Extract probability predictions as meta-features
    3. Fine-tune a calibration layer on n_shot target samples
    4. Evaluate on remaining target samples via Leave-One-Out
    """
    logger.info(f"\n--- U2: {n_shot}-Shot Fine-Tuning ---")

    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    Xs = sc_s.transform(X_src)
    Xt = sc_t.transform(X_tgt)

    # Train base on source
    base = make_ensemble()
    base.fit(Xs, y_src)

    # Get stacked predictions for target
    meta_features_tgt = []
    for est in base.estimators_:
        meta_features_tgt.append(est.predict_proba(Xt)[:, 1:])
    meta_tgt = np.hstack(meta_features_tgt)  # (30, 3)

    # Stratified selection of n_shot calibration samples
    rng = np.random.default_rng(RS)
    cal_idx_0 = rng.choice(np.where(y_tgt == 0)[0], n_shot // 2 + n_shot % 2, replace=False)
    cal_idx_1 = rng.choice(np.where(y_tgt == 1)[0], n_shot // 2, replace=False)
    cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
    test_idx = np.array([i for i in range(len(y_tgt)) if i not in cal_idx])

    logger.info(f"  Calibration: {len(cal_idx)} samples  |  Test: {len(test_idx)} samples")

    # Fine-tune logistic calibration on meta-features
    cal_model = LogisticRegression(random_state=RS, max_iter=1000)
    cal_model.fit(meta_tgt[cal_idx], y_tgt[cal_idx])

    yp = cal_model.predict(meta_tgt[test_idx])
    ypr = cal_model.predict_proba(meta_tgt[test_idx])[:, 1]
    y_test = y_tgt[test_idx]

    return eval_result(y_test, yp, ypr, f"U2: {n_shot}-Shot Fine-Tuning")


# ═══════════════════════════════════════════════════════════════════════
#  U4: STACKED META-LEARNER
# ═══════════════════════════════════════════════════════════════════════

def run_u4_stacked(X_src, y_src, X_tgt, y_tgt, n_shot=5):
    """
    Stacked ensemble with domain-aware meta-learner:
    1. Train RF, XGB, LGBM separately on rank-transformed source data
    2. Extract per-model probability outputs as meta-features
    3. Concatenate raw rank-features + meta-features
    4. Fine-tune a Logistic Regression meta-learner on n_shot target samples
    5. Evaluate on remaining target samples
    """
    logger.info(f"\n--- U4: Stacked Meta-Learner ({n_shot}-shot) ---")

    qt_s = QuantileTransformer(output_distribution="uniform", random_state=RS)
    qt_t = QuantileTransformer(output_distribution="uniform", random_state=RS)
    Xs = qt_s.fit_transform(X_src)
    Xt = qt_t.fit_transform(X_tgt)

    # Train base models on rank-transformed source
    rf   = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RS)
    xgb  = XGBClassifier(n_estimators=200, use_label_encoder=False,
                          eval_metric="logloss", random_state=RS, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=200, random_state=RS, verbosity=-1)
    for m in [rf, xgb, lgbm]:
        m.fit(Xs, y_src)

    # Meta-features on target
    meta_cols = []
    for m in [rf, xgb, lgbm]:
        meta_cols.append(m.predict_proba(Xt)[:, 1:])
    meta_tgt = np.hstack([Xt] + meta_cols)  # rank-features + 3 probability outputs

    # Stratified few-shot split
    rng = np.random.default_rng(RS)
    cal_idx_0 = rng.choice(np.where(y_tgt == 0)[0], n_shot // 2 + n_shot % 2, replace=False)
    cal_idx_1 = rng.choice(np.where(y_tgt == 1)[0], n_shot // 2, replace=False)
    cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
    test_idx = np.array([i for i in range(len(y_tgt)) if i not in cal_idx])

    logger.info(f"  Meta-feature dim: {meta_tgt.shape[1]}  |  "
                f"Cal: {len(cal_idx)}  Test: {len(test_idx)}")

    meta_model = LogisticRegression(random_state=RS, max_iter=1000, C=0.1)
    meta_model.fit(meta_tgt[cal_idx], y_tgt[cal_idx])

    yp = meta_model.predict(meta_tgt[test_idx])
    ypr = meta_model.predict_proba(meta_tgt[test_idx])[:, 1]
    y_test = y_tgt[test_idx]

    return eval_result(y_test, yp, ypr, f"U4: Stacked Meta-Learner ({n_shot}-shot)")


# ═══════════════════════════════════════════════════════════════════════
#  U6: COMBINED PIPELINE (U1 + U5 + U3 best ideas merged)
# ═══════════════════════════════════════════════════════════════════════

def run_u6_combined(X_src, y_src, X_tgt, y_tgt, feat_names, n_shot=5):
    """
    Combined best-of pipeline:
    1. Engineer delta (ratio) features (U5)
    2. Rank-transform all features (U1)
    3. CORAL-align the rank-transformed features (D3)
    4. Fine-tune calibration on n_shot samples (U2)
    """
    logger.info(f"\n--- U6: Combined Pipeline (U5+U1+CORAL+{n_shot}-shot) ---")

    # Step 1: Delta features
    Xs, new_names = engineer_delta_features(X_src, feat_names)
    Xt, _         = engineer_delta_features(X_tgt, feat_names)
    logger.info(f"  Delta features: {Xs.shape[1]} dims")

    # Step 2: Rank transform
    qt_s = QuantileTransformer(output_distribution="uniform", random_state=RS)
    qt_t = QuantileTransformer(output_distribution="uniform", random_state=RS)
    Xs = qt_s.fit_transform(Xs)
    Xt = qt_t.fit_transform(Xt)

    # Step 3: CORAL alignment on rank space
    n_f = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(n_f) * 1e-6
    Ct = np.cov(Xt, rowvar=False) + np.eye(n_f) * 1e-6
    Ds, Vs = np.linalg.eigh(Cs)
    Ds = np.maximum(Ds, 1e-8)
    Cs_nh = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
    Dt, Vt = np.linalg.eigh(Ct)
    Dt = np.maximum(Dt, 1e-8)
    Ct_h = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
    A = Cs_nh @ Ct_h
    Xs = Xs @ A

    # Train ensemble on adapted source
    base = make_ensemble()
    base.fit(Xs, y_src)

    # Step 4: Few-shot calibration
    meta_cols = []
    for est in base.estimators_:
        meta_cols.append(est.predict_proba(Xt)[:, 1:])
    meta_tgt = np.hstack([Xt] + meta_cols)

    rng = np.random.default_rng(RS)
    cal_idx_0 = rng.choice(np.where(y_tgt == 0)[0], n_shot // 2 + n_shot % 2, replace=False)
    cal_idx_1 = rng.choice(np.where(y_tgt == 1)[0], n_shot // 2, replace=False)
    cal_idx = np.concatenate([cal_idx_0, cal_idx_1])
    test_idx = np.array([i for i in range(len(y_tgt)) if i not in cal_idx])

    cal_model = LogisticRegression(random_state=RS, max_iter=1000, C=0.1)
    cal_model.fit(meta_tgt[cal_idx], y_tgt[cal_idx])

    yp = cal_model.predict(meta_tgt[test_idx])
    ypr = cal_model.predict_proba(meta_tgt[test_idx])[:, 1]
    y_test = y_tgt[test_idx]

    return eval_result(y_test, yp, ypr,
                       f"U6: Combined (Delta+Rank+CORAL+{n_shot}-shot)")


# ═══════════════════════════════════════════════════════════════════════
#  PUBLICATION FIGURES
# ═══════════════════════════════════════════════════════════════════════

def generate_figures(all_results):
    logger.info("\n" + "=" * 60)
    logger.info("  GENERATING Q1 PUBLICATION FIGURES")
    logger.info("=" * 60)

    # ── 1. Upgrade Comparison Bar Chart ───────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r["label"].split(": ")[1] if ": " in r["label"] else r["label"]
              for r in all_results]
    accs   = [r["accuracy"] for r in all_results]
    f1s    = [r["f1_macro"] for r in all_results]
    ci_los = [r["accuracy_ci"][0] for r in all_results]
    ci_his = [r["accuracy_ci"][1] for r in all_results]

    x = np.arange(len(labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4c72b0",
                   edgecolor="black", zorder=3)
    bars2 = ax.bar(x + w/2, f1s,  w, label="F1-Macro", color="#55a868",
                   edgecolor="black", zorder=3)

    errs = np.array([[accs[i] - ci_los[i], ci_his[i] - accs[i]]
                      for i in range(len(accs))]).T
    ax.errorbar(x - w/2, accs, yerr=errs, fmt="none", ecolor="black",
                capsize=3, zorder=4)

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Domain Adaptation Upgrades: Cross-Dataset Performance on MalVol-25\n"
                 "(Model trained on CIC-MalMem-2022, 58,596 samples)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "upgrade_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("  Saved upgrade_comparison.png")

    # ── 2. Progression Line Chart ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(accs)), accs, "o-", color="#d62728", lw=2.5, markersize=10,
            label="Accuracy", zorder=3)
    ax.fill_between(range(len(accs)), ci_los, ci_his, alpha=0.15, color="#d62728")
    ax.plot(range(len(f1s)), f1s, "s--", color="#1f77b4", lw=2, markersize=8,
            label="F1-Macro", zorder=3)
    ax.axhline(0.5, color="grey", linestyle=":", alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0.2, 1.05)
    ax.set_title("Progressive Improvement: From Random Chance to Adapted Detection")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "upgrade_progression.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("  Saved upgrade_progression.png")

    # ── 3. Confusion Matrix Grid ──────────────────────────────────────
    n = len(all_results)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
    axes = np.array(axes).flatten()
    palette = ["Greys", "Blues", "Greens", "Oranges", "Purples", "Reds",
               "YlGn", "BuPu"]
    for i, r in enumerate(all_results):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=palette[i % len(palette)],
                    ax=axes[i], xticklabels=["Ben", "Mal"],
                    yticklabels=["Ben", "Mal"])
        short = r["label"].split(": ")[0] if ": " in r["label"] else r["label"][:10]
        axes[i].set_title(f"{short}\nAcc={r['accuracy']:.0%}", fontsize=9)
        axes[i].set_ylabel("True" if i % cols == 0 else "")
        axes[i].set_xlabel("Pred")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Confusion Matrices Across All Adaptation Methods", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "upgrade_confusion_grid.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.success("  Saved upgrade_confusion_grid.png")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    logger.info("Loading datasets...")
    X_src, y_src, X_tgt, y_tgt, feat_names = load_data()
    logger.info(f"  Source: {X_src.shape[0]:,} × {X_src.shape[1]}  |  "
                f"Target: {X_tgt.shape[0]} × {X_tgt.shape[1]}")

    all_results = []

    # === Run all methods ===
    all_results.append(run_baseline(X_src, y_src, X_tgt, y_tgt))
    all_results.append(run_coral(X_src, y_src, X_tgt, y_tgt))
    all_results.append(run_u1_rank(X_src, y_src, X_tgt, y_tgt))
    all_results.append(run_u5_delta(X_src, y_src, X_tgt, y_tgt, feat_names))
    all_results.append(run_u3_tca(X_src, y_src, X_tgt, y_tgt, n_components=15))
    all_results.append(run_u2_fewshot(X_src, y_src, X_tgt, y_tgt, n_shot=5))
    all_results.append(run_u4_stacked(X_src, y_src, X_tgt, y_tgt, n_shot=5))
    all_results.append(run_u6_combined(X_src, y_src, X_tgt, y_tgt, feat_names, n_shot=5))

    # === Save results ===
    with open(OUT_DIR / "upgrade_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.success(f"Results saved to {OUT_DIR / 'upgrade_results.json'}")

    # === Figures ===
    generate_figures(all_results)

    # === Summary Table ===
    print("\n" + "=" * 85)
    print(f"  {'Method':<55s} {'Acc':>6s}  {'95% CI':>16s}  {'F1':>6s}  {'AUC':>6s}")
    print("=" * 85)
    for r in all_results:
        ci = r["accuracy_ci"]
        print(f"  {r['label']:<55s} {r['accuracy']:>5.1%}  "
              f"[{ci[0]:.1%}–{ci[1]:.1%}]  "
              f"{r['f1_macro']:>5.1%}  {r['roc_auc']:>5.3f}")
    print("=" * 85)

    best = max(all_results, key=lambda r: r["accuracy"])
    base = all_results[0]
    delta = best["accuracy"] - base["accuracy"]
    print(f"\n  Best: {best['label']}")
    print(f"  Improvement: {base['accuracy']:.1%} → {best['accuracy']:.1%} (+{delta:.1%})")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    main()
