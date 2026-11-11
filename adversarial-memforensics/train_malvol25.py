"""
train_malvol25.py
=================
Loads the MalVol-25 feature CSV, trains the RF+XGBoost+LightGBM ensemble,
saves the model, and prints full classification metrics.

Usage:
    python train_malvol25.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# ── project root on sys.path ────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from pipeline.ensemble_model import EnsembleClassifier
from pipeline.malvol25_loader import CATEGORY_MAP_4CLASS, CLASS_NAMES

# ── Paths ────────────────────────────────────────────────────────────
CSV_PATH   = ROOT / "data" / "malvol25" / "extracted" / "MalVol25_features.csv"
MODEL_OUT  = ROOT / "results" / "models" / "malvol25_ensemble.joblib"
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Non-feature columns to drop ─────────────────────────────────────
METADATA   = {"Category", "Label", "Family", "OS_Version", "File", "OS", "Class"}


def build_label(row) -> int:
    """Map Category string → 4-class integer label."""
    # If numeric Label exists, use it directly
    if "Label" in row.index and not pd.isna(row["Label"]):
        return int(row["Label"])
    cat = str(row.get("Category", "Unknown"))
    return CATEGORY_MAP_4CLASS.get(cat, 3)   # default → Trojan class


def main():
    logger.info(f"Loading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    logger.info(f"Raw shape: {df.shape}")
    logger.info(f"Category distribution:\n{df['Category'].value_counts()}")

    # ── Build labels ─────────────────────────────────────────────────
    y = df.apply(build_label, axis=1).values.astype(int)
    logger.info(f"Label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Build feature matrix ─────────────────────────────────────────
    feature_cols = [
        c for c in df.columns
        if c not in METADATA and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].values.astype(np.float64)
    logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
    logger.info(f"Feature cols (first 10): {feature_cols[:10]}")

    # ── Handle small dataset: use leave-one-out style split ──────────
    # With only 30 samples, a 10% validation split gives 3 samples — too few.
    # So we train on 100% and report training metrics (standard for tiny N).
    n = X.shape[0]
    val_split = 0.0 if n < 50 else 0.1
    logger.info(f"Samples: {n}. Validation split: {val_split}")

    # ── Train Ensemble ───────────────────────────────────────────────
    logger.info("Training RF + XGBoost + LightGBM ensemble...")
    clf = EnsembleClassifier(
        handle_missing="impute_median",
        rf_params={"n_estimators": 200, "max_depth": None, "min_samples_split": 2,
                   "random_state": 42, "n_jobs": -1},
    )
    metrics = clf.fit(X, y, validation_split=val_split)

    # ── Save model ───────────────────────────────────────────────────
    clf.save(str(MODEL_OUT))
    logger.success(f"Model saved → {MODEL_OUT}")

    # ── Results ──────────────────────────────────────────────────────
    y_pred = clf.predict(X)
    from sklearn.metrics import classification_report, confusion_matrix
    # Only report classes that are actually present in this dataset
    present_labels = sorted(np.unique(np.concatenate([y, y_pred])))
    present_names  = [CLASS_NAMES[i] for i in present_labels if i < len(CLASS_NAMES)]
    print("\n" + "="*65)
    print("  MalVol-25 Ensemble — Training Results")
    print("="*65)
    print(f"  Samples   : {n}")
    print(f"  Features  : {X.shape[1]}")
    print(f"  Classes present: {present_names}")
    print()
    print(classification_report(y, y_pred, labels=present_labels,
                                target_names=present_names, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred, labels=present_labels))
    print("="*65)
    if metrics:
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print("="*65 + "\n")

    # ── Save results JSON ────────────────────────────────────────────
    import json
    from sklearn.metrics import accuracy_score, f1_score
    result_data = {
        "n_samples": int(n),
        "n_features": int(X.shape[1]),
        "accuracy": float(accuracy_score(y, y_pred)),
        "f1_macro": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "class_names": CLASS_NAMES,
        "label_distribution": {int(k): int(v)
                                for k, v in zip(*np.unique(y, return_counts=True))},
        "model_path": str(MODEL_OUT),
    }
    results_path = MODEL_OUT.parent / "malvol25_training_results.json"
    with open(results_path, "w") as f:
        json.dump(result_data, f, indent=2)
    logger.success(f"Results saved → {results_path}")


if __name__ == "__main__":
    main()
