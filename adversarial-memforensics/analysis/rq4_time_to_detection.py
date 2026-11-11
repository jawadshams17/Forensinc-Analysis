"""
RQ4 Analysis — Time-to-Correct-Detection (Extended)
=====================================================
Quantifies how quickly a correct detection occurs when using a
streaming/incremental memory analysis approach, measuring detection
latency as a function of capture completeness.

Tests whether early partial captures can yield correct detection
before full acquisition completes (relevant for real-time DFIR).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor


class TimeToDetectionAnalyzer:
    """
    Measures detection latency by evaluating predictions at
    incrementally increasing capture completeness levels.
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        extractor: FeatureExtractor,
        completeness_steps: Optional[List[float]] = None,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.steps = completeness_steps or [
            0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
            0.85, 0.90, 0.95, 1.00,
        ]

    def run_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        For each sample, determine the minimum capture completeness
        at which the ensemble first correctly classifies it.
        """
        logger.info("=" * 60)
        logger.info("RQ4: TIME-TO-CORRECT-DETECTION ANALYSIS")
        logger.info("=" * 60)

        n_samples = X.shape[0]
        first_correct_at = np.full(n_samples, np.nan)
        consistently_correct_from = np.full(n_samples, np.nan)

        per_step_accuracy = []

        for pct in self.steps:
            X_trunc = self.extractor.simulate_truncation(X, pct)
            y_pred = self.classifier.predict(X_trunc)
            correct = y_pred == y
            acc = np.mean(correct)
            per_step_accuracy.append({"completeness": pct, "accuracy": acc})

            for i in range(n_samples):
                if correct[i] and np.isnan(first_correct_at[i]):
                    first_correct_at[i] = pct

            logger.info(f"  Completeness {pct:.0%}: accuracy={acc:.4f}")

        # Determine consistent detection: smallest pct from which all
        # subsequent steps are correct
        for i in range(n_samples):
            for s_idx, pct in enumerate(self.steps):
                X_trunc = self.extractor.simulate_truncation(X[i:i+1], pct)
                pred = self.classifier.predict(X_trunc)[0]
                if pred == y[i]:
                    # Check all subsequent steps
                    all_correct = True
                    for future_pct in self.steps[s_idx:]:
                        X_f = self.extractor.simulate_truncation(X[i:i+1], future_pct)
                        if self.classifier.predict(X_f)[0] != y[i]:
                            all_correct = False
                            break
                    if all_correct:
                        consistently_correct_from[i] = pct
                        break

        valid_first = first_correct_at[~np.isnan(first_correct_at)]
        valid_consistent = consistently_correct_from[~np.isnan(consistently_correct_from)]

        results = {
            "per_step_accuracy": per_step_accuracy,
            "first_correct_detection": {
                "mean_pct": float(np.mean(valid_first)) if len(valid_first) > 0 else None,
                "median_pct": float(np.median(valid_first)) if len(valid_first) > 0 else None,
                "std_pct": float(np.std(valid_first)) if len(valid_first) > 0 else None,
                "n_never_correct": int(np.sum(np.isnan(first_correct_at))),
            },
            "consistent_detection": {
                "mean_pct": float(np.mean(valid_consistent)) if len(valid_consistent) > 0 else None,
                "median_pct": float(np.median(valid_consistent)) if len(valid_consistent) > 0 else None,
                "n_never_consistent": int(np.sum(np.isnan(consistently_correct_from))),
            },
        }

        logger.info(f"\nFirst correct detection (mean): {results['first_correct_detection']['mean_pct']}")
        logger.info(f"Consistent detection (mean): {results['consistent_detection']['mean_pct']}")

        return results

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rq4_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"RQ4 results saved to {output_dir}")
