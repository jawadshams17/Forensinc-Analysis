"""
Ensemble Classifier — Paper 1 Architecture (RF+XGBoost+LightGBM)
=================================================================
Implements the soft-voting ensemble from Paper 1 with:
    - Random Forest
    - XGBoost
    - LightGBM

Key design decisions:
    1. Architecture is UNCHANGED from Paper 1 to isolate capture-
       completeness and adversarial effects from confounding changes
    2. XGBoost and LightGBM natively handle missing values (NaN)
       from truncated captures
    3. Provides both predict_proba (for optimizer) and predict interfaces
    4. SHAP TreeExplainer compatible
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    logger.warning("XGBoost not installed — ensemble will use fallback")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    logger.warning("LightGBM not installed — ensemble will use fallback")

from perturbation_primitives import VOLMEMLYZER_FEATURES


# ── Default hyperparameters from Paper 1 ───────────────────────────

DEFAULT_RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "use_label_encoder": False,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "verbosity": 0,
}

DEFAULT_LGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": -1,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]


class LegacyEnsembleClassifier:
    """
    [DEPRECATED] Soft-voting ensemble classifier reproducing Paper 1's architecture.
    Phased out per Phase 0 continuous probability ablation (N=58k, Wilxocon).
    Retained solely for reproducibility audits.
    """

    def __init__(
        self,
        rf_params: Optional[Dict] = None,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        handle_missing: str = "native",   # 'native', 'impute_median', 'impute_zero'
    ):
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.class_names = class_names or CLASS_NAMES
        self.handle_missing = handle_missing

        # Build individual estimators
        self.rf = RandomForestClassifier(**(rf_params or DEFAULT_RF_PARAMS))

        if xgb is not None:
            self.xgb_clf = xgb.XGBClassifier(**(xgb_params or DEFAULT_XGB_PARAMS))
        else:
            logger.warning("XGBoost unavailable, using extra RF")
            self.xgb_clf = RandomForestClassifier(
                **(rf_params or DEFAULT_RF_PARAMS), random_state=43
            )

        if lgb is not None:
            self.lgb_clf = lgb.LGBMClassifier(**(lgb_params or DEFAULT_LGB_PARAMS))
        else:
            logger.warning("LightGBM unavailable, using extra RF")
            self.lgb_clf = RandomForestClassifier(
                **(rf_params or DEFAULT_RF_PARAMS), random_state=44
            )

        # Soft-voting ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ("rf", self.rf),
                ("xgb", self.xgb_clf),
                ("lgb", self.lgb_clf),
            ],
            voting="soft",
            n_jobs=-1,
        )

        # Missing value imputer (for RF which can't handle NaN)
        self.imputer = SimpleImputer(strategy="median")
        self._is_fitted = False
        self._training_metrics: Dict[str, Any] = {}

    def _prepare_features(self, X: np.ndarray, fit_imputer: bool = False) -> np.ndarray:
        """
        Prepare features handling missing values from truncated captures.

        Strategy depends on self.handle_missing:
          - 'native':       Leave NaN for XGB/LGB, impute only for RF
          - 'impute_median': Impute all NaN with median
          - 'impute_zero':   Impute all NaN with 0
        """
        X = np.asarray(X, dtype=np.float64)

        if self.handle_missing == "impute_median":
            if fit_imputer:
                X = self.imputer.fit_transform(X)
            else:
                X = self.imputer.transform(X)
        elif self.handle_missing == "impute_zero":
            X = np.nan_to_num(X, nan=0.0)
        # 'native' — leave NaN; XGB/LGB handle it, RF may error on NaN

        return X

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Train the ensemble on the provided data.

        Args:
            X: (n_samples, 55) feature matrix
            y: (n_samples,) labels
            validation_split: fraction held out for validation

        Returns:
            Training metrics dict
        """
        logger.info(
            f"Training ensemble on {X.shape[0]} samples, "
            f"{X.shape[1]} features"
        )
        start = time.time()

        X_prep = self._prepare_features(X, fit_imputer=True)

        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_prep, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X_prep, y
            X_val, y_val = None, None

        # Fit the ensemble
        self.ensemble.fit(X_train, y_train)
        self._is_fitted = True

        elapsed = time.time() - start
        logger.info(f"Training complete in {elapsed:.1f}s")

        # Compute metrics
        metrics = {"training_time_seconds": elapsed}
        if X_val is not None:
            y_pred = self.ensemble.predict(X_val)
            metrics["val_accuracy"] = accuracy_score(y_val, y_pred)
            metrics["val_f1_macro"] = f1_score(y_val, y_pred, average="macro")
            metrics["val_f1_weighted"] = f1_score(y_val, y_pred, average="weighted")
            logger.info(
                f"Validation: accuracy={metrics['val_accuracy']:.4f}, "
                f"F1(macro)={metrics['val_f1_macro']:.4f}"
            )

        self._training_metrics = metrics
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_prep = self._prepare_features(X)
        return self.ensemble.predict(X_prep)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (soft-voting output).
        Returns: (n_samples, n_classes) probability matrix
        """
        X_prep = self._prepare_features(X)
        return self.ensemble.predict_proba(X_prep)

    def malicious_probability(self, X: np.ndarray) -> np.ndarray:
        """
        Compute P(malicious) = 1 - P(benign).
        Assumes class 0 = Benign.
        Returns: (n_samples,) malicious probabilities
        """
        proba = self.predict_proba(X)
        return 1.0 - proba[:, 0]  # P(not benign)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label: str = "Test",
    ) -> Dict[str, Any]:
        """
        Full evaluation with all metrics.

        Returns dict with accuracy, F1, precision, recall, FPR, FNR,
        confusion matrix, and per-class report.
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)

        acc = accuracy_score(y, y_pred)
        f1_m = f1_score(y, y_pred, average="macro")
        f1_w = f1_score(y, y_pred, average="weighted")
        prec = precision_score(y, y_pred, average="macro", zero_division=0)
        rec = recall_score(y, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y, y_pred)

        # Binary FPR/FNR (benign=0 vs all malicious)
        y_binary = (y != 0).astype(int)
        y_pred_binary = (y_pred != 0).astype(int)
        tn = np.sum((y_binary == 0) & (y_pred_binary == 0))
        fp = np.sum((y_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_binary == 1) & (y_pred_binary == 0))
        tp = np.sum((y_binary == 1) & (y_pred_binary == 1))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        report = classification_report(
            y, y_pred,
            target_names=self.class_names[:len(np.unique(y))],
            output_dict=True,
        )

        metrics = {
            "label": label,
            "accuracy": acc,
            "f1_macro": f1_m,
            "f1_weighted": f1_w,
            "precision_macro": prec,
            "recall_macro": rec,
            "fpr": fpr,
            "fnr": fnr,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        logger.info(
            f"[{label}] accuracy={acc:.4f}, F1={f1_m:.4f}, "
            f"FPR={fpr:.4%}, FNR={fnr:.4%}"
        )

        return metrics

    def get_individual_estimators(self) -> Dict[str, Any]:
        """Return individual fitted estimators for SHAP analysis."""
        return {
            "rf": self.ensemble.named_estimators_["rf"],
            "xgb": self.ensemble.named_estimators_["xgb"],
            "lgb": self.ensemble.named_estimators_["lgb"],
        }

    def save(self, path: str | Path) -> None:
        """Save the fitted ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "ensemble": self.ensemble,
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "handle_missing": self.handle_missing,
            "training_metrics": self._training_metrics,
        }, path)
        logger.info(f"Ensemble saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "LegacyEnsembleClassifier":
        """Load a fitted ensemble from disk."""
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.ensemble = data["ensemble"]
        instance.imputer = data["imputer"]
        instance.feature_names = data["feature_names"]
        instance.class_names = data["class_names"]
        instance.handle_missing = data["handle_missing"]
        instance._training_metrics = data.get("training_metrics", {})
        instance._is_fitted = True

        # Restore individual estimator references
        instance.rf = instance.ensemble.named_estimators_["rf"]
        instance.xgb_clf = instance.ensemble.named_estimators_["xgb"]
        instance.lgb_clf = instance.ensemble.named_estimators_["lgb"]

        logger.info(f"Ensemble loaded from {path}")
        return instance

class PrimaryDetector:
    """
    Validated Primary Detector (XGBoost standalone).
    Replaces LegacyEnsembleClassifier following Phase 0 Wilcoxon continuous probability test.
    """
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        class_names: Optional[List[str]] = None,
        handle_missing: str = "native",
    ):
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.class_names = class_names or CLASS_NAMES
        self.handle_missing = handle_missing
        self.imputer = SimpleImputer(strategy="median")

        # The validated hyperparameters for XGBoost
        self.xgb_clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0
        )
        self._is_fitted = False

    def _prepare_features(self, X: np.ndarray, fit_imputer: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.handle_missing == "impute_median":
            if fit_imputer:
                X = self.imputer.fit_transform(X)
            else:
                X = self.imputer.transform(X)
        elif self.handle_missing == "impute_zero":
            X = np.nan_to_num(X, nan=0.0)
        return X

    def fit(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.1) -> Dict[str, Any]:
        logger.info(f"Training PrimaryDetector (XGBoost) on {X.shape[0]} samples...")
        start = time.time()
        X_prep = self._prepare_features(X, fit_imputer=True)
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_prep, y, test_size=validation_split, random_state=42, stratify=y
            )
        else:
            X_train, y_train = X_prep, y
            X_val, y_val = None, None

        self.xgb_clf.fit(X_train, y_train)
        self._is_fitted = True

        elapsed = time.time() - start
        metrics = {"training_time_seconds": elapsed}
        if X_val is not None:
            y_pred = self.xgb_clf.predict(X_val)
            metrics["val_accuracy"] = accuracy_score(y_val, y_pred)
            metrics["val_f1_macro"] = f1_score(y_val, y_pred, average="macro")
            logger.info(f"Validation: accuracy={metrics['val_accuracy']:.4f}, F1={metrics['val_f1_macro']:.4f}")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.xgb_clf.predict(self._prepare_features(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.xgb_clf.predict_proba(self._prepare_features(X))

    def malicious_probability(self, X: np.ndarray) -> np.ndarray:
        return 1.0 - self.predict_proba(X)[:, 0]

    def evaluate(self, X: np.ndarray, y: np.ndarray, label: str = "Test") -> Dict[str, Any]:
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        f1_m = f1_score(y, y_pred, average="macro")
        fpr = 0.0
        logger.info(f"[{label}] accuracy={acc:.4f}, F1={f1_m:.4f}")
        return {"label": label, "accuracy": acc, "f1_macro": f1_m}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.xgb_clf,
            "imputer": self.imputer,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "handle_missing": self.handle_missing
        }, path)
        logger.info(f"PrimaryDetector saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "PrimaryDetector":
        data = joblib.load(path)
        instance = cls.__new__(cls)
        instance.xgb_clf = data["model"]
        instance.imputer = data["imputer"]
        instance.feature_names = data["feature_names"]
        instance.class_names = data["class_names"]
        instance.handle_missing = data["handle_missing"]
        instance._is_fitted = True
        logger.info(f"PrimaryDetector loaded from {path}")
        return instance
