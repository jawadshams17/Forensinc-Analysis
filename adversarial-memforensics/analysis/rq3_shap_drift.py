"""
RQ3 Analysis — SHAP Feature-Importance Drift
==============================================
Measures SHAP global feature-importance rank correlation (Spearman's ρ)
between clean-complete, truncated-clean, and truncated-adversarial
conditions. Detects "fragile features" that become unreliable under
stress.

Tests H3: ρ significantly < 1.0 under combined stress conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from analysis.statistical_tests import permutation_test_spearman, bonferroni_correction
from pipeline.shap_analysis import SHAPAnalyzer
from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor


class SHAPDriftAnalyzer:
    """
    Analyzes SHAP feature-importance stability across conditions.
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        extractor: FeatureExtractor,
        shap_analyzer: SHAPAnalyzer,
        truncation_levels: Optional[List[float]] = None,
        fragile_threshold: int = 10,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.shap = shap_analyzer
        self.truncation_levels = truncation_levels or [1.0, 0.75, 0.50, 0.25, 0.10]
        self.fragile_threshold = fragile_threshold

    def run_analysis(
        self,
        X_clean: np.ndarray,
        X_adversarial: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        max_samples: int = 500,
    ) -> Dict[str, Any]:
        """
        Run the complete SHAP drift analysis.

        Args:
            X_clean:        Complete clean feature matrix
            X_adversarial:  Adversarial feature matrix (optional)
            model:          Model for SHAP computation
            max_samples:    Max samples for SHAP (computational constraint)
        """
        logger.info("=" * 60)
        logger.info("RQ3: SHAP FEATURE-IMPORTANCE DRIFT ANALYSIS")
        logger.info("=" * 60)

        if model is None:
            # Use the RF estimator (fastest for TreeExplainer)
            model = self.classifier.get_individual_estimators()["rf"]

        # Subsample for SHAP computation
        if X_clean.shape[0] > max_samples:
            rng = np.random.default_rng(42)
            idx = rng.choice(X_clean.shape[0], max_samples, replace=False)
            X_subset = X_clean[idx]
            X_adv_subset = X_adversarial[idx] if X_adversarial is not None else None
        else:
            X_subset = X_clean
            X_adv_subset = X_adversarial

        # Compute SHAP importance for each condition
        conditions: Dict[str, pd.DataFrame] = {}

        # 1. Baseline: clean, complete
        logger.info("Computing SHAP for clean-complete condition...")
        sv_baseline = self.shap.compute_shap_values(model, X_subset, "clean_complete")
        imp_baseline = self.shap.global_importance(sv_baseline)
        conditions["clean_complete"] = imp_baseline

        # 2. Truncated-clean conditions
        for pct in self.truncation_levels:
            if pct >= 1.0:
                continue
            label = f"trunc_clean_{int(pct*100)}pct"
            logger.info(f"Computing SHAP for {label}...")
            X_trunc = self.extractor.simulate_truncation(X_subset, pct)
            # Replace NaN with 0 for SHAP computation
            X_trunc_safe = np.nan_to_num(X_trunc, nan=0.0)
            sv = self.shap.compute_shap_values(model, X_trunc_safe, label)
            conditions[label] = self.shap.global_importance(sv)

        # 3. Truncated-adversarial conditions
        if X_adv_subset is not None:
            for pct in self.truncation_levels:
                label = f"trunc_adv_{int(pct*100)}pct"
                logger.info(f"Computing SHAP for {label}...")
                X_trunc = self.extractor.simulate_truncation(X_adv_subset, pct)
                X_trunc_safe = np.nan_to_num(X_trunc, nan=0.0)
                sv = self.shap.compute_shap_values(model, X_trunc_safe, label)
                conditions[label] = self.shap.global_importance(sv)

        # Compare all conditions against baseline
        comparison = self.shap.compare_conditions(conditions, "clean_complete")

        # Permutation tests for each comparison
        p_values = []
        for _, row in comparison.iterrows():
            if row["vs_baseline"] == "clean_complete":
                cond_name = row["condition"]
                if cond_name in conditions:
                    imp = conditions[cond_name]
                    perm_result = permutation_test_spearman(
                        imp_baseline["rank"].values,
                        imp["rank"].values,
                    )
                    p_values.append(perm_result["p_value"])

        # Bonferroni correction
        if p_values:
            corrected = bonferroni_correction(p_values)

        # Fragile features across worst-case condition
        worst_condition = comparison.loc[comparison["spearman_rho"].idxmin()]
        worst_name = worst_condition["condition"]
        if worst_name in conditions:
            fragile = self.shap.detect_fragile_features(
                conditions["clean_complete"],
                conditions[worst_name],
                threshold=self.fragile_threshold,
            )
        else:
            fragile = pd.DataFrame()

        results = {
            "comparison_table": comparison.to_dict(orient="records"),
            "n_conditions": len(conditions),
            "worst_condition": worst_name,
            "worst_rho": float(worst_condition["spearman_rho"]),
            "n_fragile_features": len(fragile),
            "fragile_features": fragile.to_dict(orient="records") if not fragile.empty else [],
            "h3_supported": float(worst_condition["spearman_rho"]) < 0.9,
        }

        logger.info(f"\nWorst Spearman's ρ: {results['worst_rho']:.4f} ({worst_name})")
        logger.info(f"Fragile features: {results['n_fragile_features']}")
        logger.info(f"H3 supported: {results['h3_supported']}")

        return results

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rq3_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"RQ3 results saved to {output_dir}")
