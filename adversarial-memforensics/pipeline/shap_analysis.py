"""
SHAP Analysis — Explainability as a Robustness Diagnostic
==========================================================
Implements SHAP-based explainability for the ensemble classifier,
repurposed from a purely explanatory tool into a quantitative
robustness metric (RQ3).

Key capabilities:
    1. Global feature importance via mean |SHAP values|
    2. Per-sample local explanations
    3. Feature-importance rank correlation (Spearman's ρ) across conditions
    4. "Fragile feature" detection (rank change ≥ threshold)
    5. SHAP importance drift heatmaps
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats as scipy_stats

try:
    import shap
except ImportError:
    shap = None
    logger.warning("SHAP not installed — explainability features unavailable")

from perturbation_primitives import VOLMEMLYZER_FEATURES


class SHAPAnalyzer:
    """
    SHAP-based explainability and robustness analysis.

    Computes SHAP values using TreeExplainer (for tree ensembles)
    and provides tools for comparing feature importance across
    different capture conditions (clean, truncated, adversarial).
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        fragile_threshold: int = 10,  # rank change ≥ 10 = fragile
    ):
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.fragile_threshold = fragile_threshold
        self._explainers: Dict[str, Any] = {}
        self._shap_cache: Dict[str, np.ndarray] = {}

    # ── SHAP Value Computation ─────────────────────────────────────

    def compute_shap_values(
        self,
        model: Any,
        X: np.ndarray,
        model_name: str = "ensemble",
        background_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute SHAP values for the given model and data.

        For tree-based models, uses TreeExplainer (exact, fast).
        Falls back to KernelExplainer for non-tree models.

        Args:
            model:              Fitted sklearn/xgb/lgb model
            X:                  (n_samples, n_features) feature matrix
            model_name:         Key for caching
            background_samples: N background samples for KernelExplainer

        Returns:
            SHAP values array: (n_samples, n_features) for binary,
            or (n_samples, n_features, n_classes) for multiclass
        """
        if shap is None:
            raise ImportError("SHAP library is required")

        cache_key = f"{model_name}_{X.shape[0]}"

        try:
            # Try TreeExplainer first (works for RF, XGB, LGB)
            explainer = shap.TreeExplainer(model)
            self._explainers[model_name] = explainer
            logger.info(f"Computing SHAP values with TreeExplainer for {model_name}...")

        except Exception:
            # Fall back to KernelExplainer
            logger.info(f"TreeExplainer failed, using KernelExplainer for {model_name}")
            rng = np.random.default_rng(42)
            bg_idx = rng.choice(X.shape[0], min(background_samples, X.shape[0]), replace=False)
            background = X[bg_idx]
            explainer = shap.KernelExplainer(model.predict_proba, background)
            self._explainers[model_name] = explainer

        shap_values = explainer.shap_values(X)
        self._shap_cache[cache_key] = shap_values

        logger.info(
            f"SHAP values computed: shape={np.array(shap_values).shape}"
        )
        return shap_values

    # ── Global Feature Importance ──────────────────────────────────

    def global_importance(
        self,
        shap_values: np.ndarray,
        method: str = "mean_abs",
    ) -> pd.DataFrame:
        """
        Compute global feature importance from SHAP values.

        Args:
            shap_values: SHAP value array
            method:      'mean_abs' (default) or 'mean_sq'

        Returns:
            DataFrame with columns: feature, importance, rank
        """
        # Handle multiclass (3D array or list of arrays) vs binary (2D array)
        if isinstance(shap_values, list):
            # Average across classes from list
            sv = np.mean([np.abs(v) for v in shap_values], axis=0)
        elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
            # Average across classes from 3D array (n_samples, n_features, n_classes)
            sv = np.mean(np.abs(shap_values), axis=2)
        else:
            sv = np.abs(shap_values)

        if method == "mean_abs":
            importance = np.mean(sv, axis=0)
        elif method == "mean_sq":
            importance = np.mean(sv ** 2, axis=0)
        else:
            importance = np.mean(sv, axis=0)

        # Build DataFrame
        df = pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        })
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)

        return df

    # ── Rank Correlation (RQ3) ─────────────────────────────────────

    def rank_correlation(
        self,
        importance_a: pd.DataFrame,
        importance_b: pd.DataFrame,
        method: str = "spearman",
    ) -> Tuple[float, float]:
        """
        Compute rank correlation between two feature importance rankings.

        Args:
            importance_a: Global importance from condition A
            importance_b: Global importance from condition B
            method:       'spearman' or 'kendall'

        Returns:
            (correlation_coefficient, p_value)
        """
        # Merge on feature name
        merged = importance_a[["feature", "rank"]].merge(
            importance_b[["feature", "rank"]],
            on="feature",
            suffixes=("_a", "_b"),
        )

        if method == "spearman":
            rho, p = scipy_stats.spearmanr(merged["rank_a"], merged["rank_b"])
        elif method == "kendall":
            rho, p = scipy_stats.kendalltau(merged["rank_a"], merged["rank_b"])
        else:
            rho, p = scipy_stats.spearmanr(merged["rank_a"], merged["rank_b"])

        logger.info(f"Rank correlation ({method}): ρ={rho:.4f}, p={p:.6f}")
        return rho, p

    # ── Fragile Feature Detection ──────────────────────────────────

    def detect_fragile_features(
        self,
        importance_baseline: pd.DataFrame,
        importance_stress: pd.DataFrame,
        threshold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Identify features whose importance rank changes by ≥ threshold
        between baseline and stressed conditions.

        Args:
            importance_baseline: Clean-complete importance ranking
            importance_stress:   Truncated or adversarial importance ranking
            threshold:           Minimum rank change to flag (default: self.fragile_threshold)

        Returns:
            DataFrame of fragile features with rank_baseline, rank_stress,
            rank_change, direction
        """
        threshold = threshold or self.fragile_threshold

        merged = importance_baseline[["feature", "rank"]].merge(
            importance_stress[["feature", "rank"]],
            on="feature",
            suffixes=("_baseline", "_stress"),
        )

        merged["rank_change"] = abs(merged["rank_stress"] - merged["rank_baseline"])
        merged["direction"] = np.where(
            merged["rank_stress"] < merged["rank_baseline"],
            "↑ more important",
            "↓ less important",
        )

        fragile = merged[merged["rank_change"] >= threshold].sort_values(
            "rank_change", ascending=False
        )

        logger.info(
            f"Fragile features (Δrank ≥ {threshold}): {len(fragile)}/{len(merged)}"
        )
        return fragile

    # ── Multi-Condition Comparison ─────────────────────────────────

    def compare_conditions(
        self,
        conditions: Dict[str, pd.DataFrame],
        baseline_key: str = "clean_complete",
    ) -> pd.DataFrame:
        """
        Compare feature importance rankings across multiple conditions.

        Args:
            conditions:    {condition_name: importance_dataframe}
            baseline_key:  Key of the baseline condition

        Returns:
            DataFrame with Spearman's ρ for each condition pair
        """
        if baseline_key not in conditions:
            logger.error(f"Baseline '{baseline_key}' not in conditions")
            return pd.DataFrame()

        results = []
        baseline = conditions[baseline_key]
        condition_names = [k for k in conditions if k != baseline_key]

        for name in condition_names:
            rho, p = self.rank_correlation(baseline, conditions[name])
            n_fragile = len(self.detect_fragile_features(baseline, conditions[name]))
            results.append({
                "condition": name,
                "vs_baseline": baseline_key,
                "spearman_rho": rho,
                "p_value": p,
                "significant": p < 0.05,
                "n_fragile_features": n_fragile,
            })

        # Also compute pairwise between non-baseline conditions
        for i, name_a in enumerate(condition_names):
            for name_b in condition_names[i+1:]:
                rho, p = self.rank_correlation(conditions[name_a], conditions[name_b])
                results.append({
                    "condition": f"{name_a} vs {name_b}",
                    "vs_baseline": "pairwise",
                    "spearman_rho": rho,
                    "p_value": p,
                    "significant": p < 0.05,
                    "n_fragile_features": -1,
                })

        return pd.DataFrame(results)

    # ── Persistence ────────────────────────────────────────────────

    def save_shap_values(
        self,
        shap_values: np.ndarray,
        output_path: str | Path,
        condition_label: str = "",
    ) -> None:
        """Save SHAP values to .npy file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, shap_values)
        logger.info(f"SHAP values saved to {output_path}")

    @staticmethod
    def load_shap_values(path: str | Path) -> np.ndarray:
        """Load SHAP values from .npy file."""
        return np.load(path, allow_pickle=True)
