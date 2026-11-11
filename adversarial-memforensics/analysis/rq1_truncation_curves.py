"""
RQ1 Analysis — Truncation Degradation Curves
==============================================
Measures how ensemble accuracy and F1 degrade as a function of
memory-capture truncation percentage, ABSENT any adversarial
manipulation.

Output:
    - Accuracy/F1/Precision/Recall/FPR/FNR vs truncation %
    - 95% bootstrap confidence intervals
    - Friedman test + Nemenyi post-hoc across truncation levels
    - Per-class degradation breakdown
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from analysis.statistical_tests import (
    bootstrap_metric,
    friedman_nemenyi_test,
    compute_effect_size,
)
from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class TruncationAnalyzer:
    """
    Analyzes ensemble performance degradation across truncation levels.

    Protocol:
        1. Load complete dataset
        2. For each truncation level:
           a. Simulate truncation (mask features → NaN)
           b. Run ensemble inference
           c. Compute metrics with bootstrap CIs
        3. Fit degradation curves
        4. Run statistical tests (Friedman + Nemenyi)
        5. Report per-class breakdown
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        extractor: FeatureExtractor,
        truncation_levels: Optional[List[float]] = None,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_seed: int = 42,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.truncation_levels = truncation_levels or [1.0, 0.75, 0.50, 0.25, 0.10]
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.rng = np.random.default_rng(random_seed)

    def run_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run the complete RQ1 analysis.

        Args:
            X: (n_samples, 55) complete feature matrix
            y: (n_samples,) true labels

        Returns:
            Results dict with degradation curves, CIs, and statistical tests
        """
        logger.info("=" * 60)
        logger.info("RQ1: TRUNCATION DEGRADATION ANALYSIS")
        logger.info("=" * 60)

        results = {
            "truncation_levels": self.truncation_levels,
            "metrics_by_level": [],
            "degradation_curves": {},
        }

        # Collect per-level metrics
        accuracy_by_level = []
        f1_by_level = []

        for pct in self.truncation_levels:
            logger.info(f"\n--- Truncation: {pct:.0%} ---")

            # Simulate truncation
            X_trunc = self.extractor.simulate_truncation(X, pct, feature_names)

            # Predict
            y_pred = self.classifier.predict(X_trunc)
            y_proba = self.classifier.predict_proba(X_trunc)

            # Compute metrics
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="macro")
            prec = precision_score(y, y_pred, average="macro", zero_division=0)
            rec = recall_score(y, y_pred, average="macro", zero_division=0)

            # Binary FPR/FNR
            y_bin = (y != 0).astype(int)
            y_pred_bin = (y_pred != 0).astype(int)
            tn = np.sum((y_bin == 0) & (y_pred_bin == 0))
            fp = np.sum((y_bin == 0) & (y_pred_bin == 1))
            fn = np.sum((y_bin == 1) & (y_pred_bin == 0))
            tp = np.sum((y_bin == 1) & (y_pred_bin == 1))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

            # Bootstrap CIs
            acc_ci = bootstrap_metric(
                y, y_pred, accuracy_score,
                n_bootstrap=self.n_bootstrap,
                confidence=self.confidence_level,
                rng=self.rng,
            )
            f1_ci = bootstrap_metric(
                y, y_pred,
                lambda yt, yp: f1_score(yt, yp, average="macro"),
                n_bootstrap=self.n_bootstrap,
                confidence=self.confidence_level,
                rng=self.rng,
            )

            # Per-class F1
            per_class_f1 = f1_score(y, y_pred, average=None)

            level_result = {
                "truncation_pct": pct,
                "accuracy": acc,
                "accuracy_ci": acc_ci,
                "f1_macro": f1,
                "f1_ci": f1_ci,
                "precision_macro": prec,
                "recall_macro": rec,
                "fpr": fpr,
                "fnr": fnr,
                "per_class_f1": per_class_f1.tolist(),
                "n_nan_features": int(np.isnan(X_trunc).any(axis=0).sum()),
            }
            results["metrics_by_level"].append(level_result)

            accuracy_by_level.append(acc)
            f1_by_level.append(f1)

            logger.info(
                f"  Accuracy: {acc:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}]"
            )
            logger.info(
                f"  F1(macro): {f1:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]"
            )
            logger.info(f"  FPR: {fpr:.4%}, FNR: {fnr:.4%}")

        # Store degradation curves
        results["degradation_curves"] = {
            "accuracy": accuracy_by_level,
            "f1_macro": f1_by_level,
        }

        # Friedman test across truncation levels
        if len(self.truncation_levels) >= 3:
            logger.info("\n--- Friedman Test ---")
            # Need per-sample predictions at each level for proper Friedman test
            # Here we use the macro metric as a surrogate
            friedman_result = friedman_nemenyi_test(
                [accuracy_by_level],
                group_names=[f"{p:.0%}" for p in self.truncation_levels],
            )
            results["friedman_test"] = friedman_result

        # Compute degradation rate
        if len(accuracy_by_level) >= 2:
            total_degradation = accuracy_by_level[0] - accuracy_by_level[-1]
            results["total_accuracy_degradation"] = total_degradation
            results["degradation_per_10pct"] = total_degradation / (
                (self.truncation_levels[0] - self.truncation_levels[-1]) * 10
            )

            # Find threshold: at what truncation does accuracy drop below 95%?
            for i, acc in enumerate(accuracy_by_level):
                if acc < 0.95:
                    results["accuracy_below_95_at"] = self.truncation_levels[i]
                    break
            else:
                results["accuracy_below_95_at"] = None

        logger.info(f"\nTotal degradation: {results.get('total_accuracy_degradation', 0):.4f}")
        return results

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        """Save analysis results to JSON and CSV."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        with open(output_dir / "rq1_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        # CSV summary
        df = pd.DataFrame(results["metrics_by_level"])
        df.to_csv(output_dir / "rq1_degradation_table.csv", index=False)

        logger.info(f"RQ1 results saved to {output_dir}")
