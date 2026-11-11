"""
train_unified.py
================
Trains the standalone XGBoost Baseline on MalwareMemoryDump.csv (58,596 samples),
evaluates on a held-out test split, then evaluates cross-dataset generalisation
on MalVol-25 (30 samples, zero-shot).

Outputs:
    results/unified/primary_results.json    -- test-set metrics with 95% CI
    results/unified/crossdataset_results.json -- MalVol-25 transfer metrics
    results/models/unified_baseline.joblib  -- saved model
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
import joblib

try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("Install xgboost: pip install xgboost")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

TRAIN_CSV  = ROOT / "data" / "unified" / "train_val_test.csv"
HOLDOUT_CSV = ROOT / "data" / "unified" / "malvol25_holdout.csv"
OUT_DIR    = ROOT / "results" / "unified"
MODEL_PATH = ROOT / "results" / "models" / "unified_baseline.joblib"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def bootstrap_ci(y_true, y_pred, y_proba, metric_fn, n=N_BOOTSTRAP, ci=0.95):
    """Compute bootstrap 95% CI for a metric."""
    scores = []
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            scores.append(metric_fn(y_true[idx], y_pred[idx]))
        except Exception:
            pass
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(np.mean(scores)), float(lo), float(hi)


def evaluate(y_true, y_pred, y_proba, label=""):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    auc  = roc_auc_score(y_true, y_proba[:, 1])

    _, acc_lo, acc_hi = bootstrap_ci(y_true, y_pred, y_proba,
                                     lambda yt, yp: accuracy_score(yt, yp))
    _, f1_lo, f1_hi   = bootstrap_ci(y_true, y_pred, y_proba,
                                     lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0))

    logger.info(f"\n{'='*55}")
    logger.info(f"  {label}")
    logger.info(f"{'='*55}")
    logger.info(f"  Accuracy : {acc:.4f}  95% CI [{acc_lo:.4f}, {acc_hi:.4f}]")
    logger.info(f"  F1-Macro : {f1:.4f}  95% CI [{f1_lo:.4f}, {f1_hi:.4f}]")
    logger.info(f"  ROC-AUC  : {auc:.4f}")
    logger.info(f"\n{classification_report(y_true, y_pred, target_names=['Benign','Malware'], zero_division=0)}")
    logger.info(f"Confusion Matrix:\n{confusion_matrix(y_true, y_pred)}")

    return {
        "n_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "accuracy_ci_95": [round(acc_lo, 4), round(acc_hi, 4)],
        "f1_macro": round(f1, 4),
        "f1_macro_ci_95": [round(f1_lo, 4), round(f1_hi, 4)],
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    # ── Load data ─────────────────────────────────────────────────────
    logger.info("Loading unified training data ...")
    df = pd.read_csv(TRAIN_CSV)
    X = df.drop(columns=["label"]).values.astype(np.float64)
    y = df["label"].values.astype(int)
    logger.info(f"Train pool: {X.shape[0]:,} samples × {X.shape[1]} features")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── 80 / 10 / 10 split ────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.111, stratify=y_trainval, random_state=RANDOM_STATE)
    logger.info(f"Split → Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ── Build model ───────────────────────────────────────────────────
    logger.info("Initializing standalone XGBoost Baseline (n_estimators=100, max_depth=5)...")
    clf = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        use_label_encoder=False, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info("Training model on training split ...")
    clf.fit(X_train, y_train)

    # ── Validate ──────────────────────────────────────────────────────
    val_pred  = clf.predict(X_val)
    val_proba = clf.predict_proba(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    logger.info(f"Validation accuracy: {val_acc:.4f}")

    # ── Primary Test Evaluation ───────────────────────────────────────
    logger.info("Evaluating on held-out test split ...")
    test_pred  = clf.predict(X_test)
    test_proba = clf.predict_proba(X_test)
    primary_results = evaluate(y_test, test_pred, test_proba,
                                label="PRIMARY RESULTS (MalwareMemoryDump test set)")

    # ── 5-fold CV on full train+val pool ─────────────────────────────
    logger.info("Running 5-fold stratified CV ...")
    cv_scores = cross_val_score(clf, X_trainval, y_trainval,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                                 scoring="roc_auc", n_jobs=-1)
    logger.info(f"5-Fold CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    primary_results["cv_auc_mean"] = round(float(cv_scores.mean()), 4)
    primary_results["cv_auc_std"]  = round(float(cv_scores.std()), 4)

    # ── Save primary results ──────────────────────────────────────────
    with open(OUT_DIR / "primary_results.json", "w") as f:
        json.dump(primary_results, f, indent=2)
    logger.success(f"Primary results → {OUT_DIR}/primary_results.json")

    # ── Cross-Dataset: MalVol-25 holdout ─────────────────────────────
    logger.info("\nLoading MalVol-25 cross-dataset holdout ...")
    mv_df   = pd.read_csv(HOLDOUT_CSV)
    X_mv    = mv_df.drop(columns=["label"]).values.astype(np.float64)
    y_mv    = mv_df["label"].values.astype(int)

    mv_pred  = clf.predict(X_mv)
    mv_proba = clf.predict_proba(X_mv)
    cd_results = evaluate(y_mv, mv_pred, mv_proba,
                           label="CROSS-DATASET RESULTS (MalVol-25, zero-shot)")
    cd_results["note"] = ("Model trained on MalwareMemoryDump (58,596 samples). "
                          "Evaluated zero-shot on MalVol-25 (30 samples). "
                          "Performance gap quantifies cross-dataset generalisation challenge.")

    with open(OUT_DIR / "crossdataset_results.json", "w") as f:
        json.dump(cd_results, f, indent=2)
    logger.success(f"Cross-dataset results → {OUT_DIR}/crossdataset_results.json")

    # ── Save model ────────────────────────────────────────────────────
    joblib.dump(clf, MODEL_PATH)
    logger.success(f"Model saved → {MODEL_PATH}")

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL SUMMARY")
    print("="*65)
    print(f"  Training samples : {len(X_train):,}")
    print(f"  Test samples     : {len(X_test):,}")
    print(f"  Features         : {X.shape[1]}")
    print()
    print(f"  PRIMARY TEST SET (MalwareMemoryDump)")
    print(f"    Accuracy : {primary_results['accuracy']:.4f}  "
          f"CI [{primary_results['accuracy_ci_95'][0]:.4f}–{primary_results['accuracy_ci_95'][1]:.4f}]")
    print(f"    F1-Macro : {primary_results['f1_macro']:.4f}  "
          f"CI [{primary_results['f1_macro_ci_95'][0]:.4f}–{primary_results['f1_macro_ci_95'][1]:.4f}]")
    print(f"    ROC-AUC  : {primary_results['roc_auc']:.4f}")
    print(f"    5-Fold CV AUC : {primary_results['cv_auc_mean']:.4f} ± {primary_results['cv_auc_std']:.4f}")
    print()
    print(f"  CROSS-DATASET (MalVol-25, zero-shot)")
    print(f"    Accuracy : {cd_results['accuracy']:.4f}  "
          f"CI [{cd_results['accuracy_ci_95'][0]:.4f}–{cd_results['accuracy_ci_95'][1]:.4f}]")
    print(f"    F1-Macro : {cd_results['f1_macro']:.4f}")
    print(f"    ROC-AUC  : {cd_results['roc_auc']:.4f}")
    print("="*65)


if __name__ == "__main__":
    main()
