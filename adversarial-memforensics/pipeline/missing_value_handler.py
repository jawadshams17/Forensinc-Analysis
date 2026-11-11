"""
Missing Value Handler — NaN Encoding for Truncated Captures
=============================================================
Manages the missing-value semantics that arise from truncated
memory captures, where features cannot be computed from partial dumps.

Key design decision:
    Features from truncated captures are encoded as NaN rather than
    imputed, so that the ensemble's NATIVE missing-value handling is
    exercised rather than masked. This is critical for measuring true
    degradation (RQ1).

    - XGBoost: learns optimal default direction for missing values
    - LightGBM: uses bin for missing values, handles natively
    - Random Forest: does NOT handle NaN; requires imputation or
      a wrapper that provides a default value
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.impute import SimpleImputer

from perturbation_primitives import VOLMEMLYZER_FEATURES, FEATURE_GROUPS


class MissingValueHandler:
    """
    Handles missing values from truncated memory captures.

    Strategies:
        'native':         Leave NaN — XGB/LGB handle natively, RF receives
                          imputed version via a separate transform
        'impute_median':  Replace NaN with training-set median
        'impute_zero':    Replace NaN with 0.0
        'impute_mean':    Replace NaN with training-set mean
        'indicator':      Add binary indicator features for each NaN
    """

    def __init__(
        self,
        strategy: str = "native",
        feature_names: Optional[List[str]] = None,
        add_indicators: bool = False,
    ):
        self.strategy = strategy
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.add_indicators = add_indicators

        self._imputer: Optional[SimpleImputer] = None
        self._training_medians: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X_complete: np.ndarray) -> "MissingValueHandler":
        """
        Fit the handler on complete (non-truncated) training data.
        Learns statistics needed for imputation.

        Args:
            X_complete: (n_samples, n_features) complete feature matrix
        """
        if self.strategy == "impute_median":
            self._imputer = SimpleImputer(strategy="median")
            self._imputer.fit(X_complete)
            self._training_medians = np.nanmedian(X_complete, axis=0)
        elif self.strategy == "impute_mean":
            self._imputer = SimpleImputer(strategy="mean")
            self._imputer.fit(X_complete)
        elif self.strategy == "impute_zero":
            pass  # no fitting needed
        elif self.strategy == "native":
            # Store medians for RF-specific imputation
            self._training_medians = np.nanmedian(X_complete, axis=0)

        self._is_fitted = True
        logger.info(f"MissingValueHandler fitted (strategy='{self.strategy}')")
        return self

    def transform(
        self,
        X: np.ndarray,
        target_model: str = "ensemble",
    ) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Transform features according to the strategy.

        For 'native' strategy with target_model='rf', applies median
        imputation specifically for Random Forest.

        Args:
            X:            (n_samples, n_features) with potential NaN
            target_model: 'ensemble', 'rf', 'xgb', 'lgb'

        Returns:
            Transformed X, optionally with indicator matrix
        """
        X_out = X.copy().astype(np.float64)

        if self.strategy == "native":
            if target_model == "rf" and self._training_medians is not None:
                # RF can't handle NaN — impute with training median
                for j in range(X_out.shape[1]):
                    mask = np.isnan(X_out[:, j])
                    if mask.any():
                        X_out[mask, j] = self._training_medians[j]
            # For XGB/LGB, leave NaN as-is

        elif self.strategy == "impute_median" and self._imputer:
            X_out = self._imputer.transform(X_out)

        elif self.strategy == "impute_mean" and self._imputer:
            X_out = self._imputer.transform(X_out)

        elif self.strategy == "impute_zero":
            X_out = np.nan_to_num(X_out, nan=0.0)

        if self.add_indicators:
            indicators = np.isnan(X).astype(np.float64)
            return X_out, indicators

        return X_out

    # ── Analysis Utilities ─────────────────────────────────────────

    def missing_pattern_report(
        self,
        X: np.ndarray,
        truncation_pct: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Generate a report of which features are missing and their groups.

        Returns DataFrame with:
            feature, group, is_missing, pct_missing_across_samples
        """
        n_samples, n_features = X.shape
        report_rows = []

        for j, name in enumerate(self.feature_names[:n_features]):
            group = name.split(".")[0]
            n_missing = np.sum(np.isnan(X[:, j]))
            report_rows.append({
                "feature": name,
                "group": group,
                "n_missing": n_missing,
                "pct_missing": n_missing / n_samples,
                "is_fully_missing": n_missing == n_samples,
            })

        df = pd.DataFrame(report_rows)
        if truncation_pct is not None:
            df["truncation_pct"] = truncation_pct

        return df

    def group_availability_summary(
        self,
        X: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Summarize feature availability by VolMemLyzer feature group.

        Returns:
            {group_name: {total, available, missing, pct_available}}
        """
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        summary = {}

        for group, features in FEATURE_GROUPS.items():
            total = len(features)
            available = 0
            for feat in features:
                if feat in name_to_idx:
                    idx = name_to_idx[feat]
                    if idx < X.shape[1] and not np.all(np.isnan(X[:, idx])):
                        available += 1

            summary[group] = {
                "total": total,
                "available": available,
                "missing": total - available,
                "pct_available": available / total if total > 0 else 0.0,
            }

        return summary
