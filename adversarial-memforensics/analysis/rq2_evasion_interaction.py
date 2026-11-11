"""
RQ2 Analysis — Adversarial Evasion Under Truncation
=====================================================
Tests whether adversarial evasion success rate increases
DISPROPORTIONATELY when perturbations are applied to truncated
captures versus complete ones.

Core metric: Interaction effect Δ
    Δ(t) = evasion_rate(adversarial, trunc=t)
         - evasion_rate(adversarial, trunc=100%)
         - [error_rate(clean, trunc=t) - error_rate(clean, trunc=100%)]

Positive Δ → truncation AMPLIFIES adversarial evasion (supports H2)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from analysis.statistical_tests import bootstrap_metric, compute_effect_size
from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor
from perturbation_primitives import apply_feature_deltas, VOLMEMLYZER_FEATURES
from perturbation_primitives.optimizer import GAMMAOptimizer, OptimizerConfig, EnsembleOracle


class EvasionInteractionAnalyzer:
    """
    Analyzes the interaction between adversarial evasion and capture truncation.
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        extractor: FeatureExtractor,
        truncation_levels: Optional[List[float]] = None,
        n_bootstrap: int = 1000,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.truncation_levels = truncation_levels or [1.0, 0.75, 0.50, 0.25, 0.10]
        self.n_bootstrap = n_bootstrap
        self.rng = np.random.default_rng(42)

    def compute_evasion_rates(
        self,
        X_clean: np.ndarray,
        X_adversarial: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute evasion rates under clean and adversarial conditions
        across all truncation levels.

        Args:
            X_clean:        Complete clean feature vectors
            X_adversarial:  Feature vectors after adversarial perturbation
            y_true:         True labels (malicious samples only)
        """
        results = {"levels": []}

        for pct in self.truncation_levels:
            # Truncate both conditions
            X_c_trunc = self.extractor.simulate_truncation(X_clean, pct)
            X_a_trunc = self.extractor.simulate_truncation(X_adversarial, pct)

            # Predict
            y_pred_clean = self.classifier.predict(X_c_trunc)
            y_pred_adv = self.classifier.predict(X_a_trunc)

            # Evasion = classified as benign when actually malicious
            # (assuming benign=0, malicious=1,2,3)
            mask_malicious = y_true != 0

            clean_errors = np.sum(y_pred_clean[mask_malicious] == 0)
            adv_evasions = np.sum(y_pred_adv[mask_malicious] == 0)
            n_malicious = np.sum(mask_malicious)

            clean_error_rate = clean_errors / n_malicious if n_malicious > 0 else 0.0
            adv_evasion_rate = adv_evasions / n_malicious if n_malicious > 0 else 0.0

            results["levels"].append({
                "truncation_pct": pct,
                "clean_error_rate": clean_error_rate,
                "adv_evasion_rate": adv_evasion_rate,
                "n_malicious": int(n_malicious),
                "n_clean_errors": int(clean_errors),
                "n_adv_evasions": int(adv_evasions),
            })

            logger.info(
                f"Trunc {pct:.0%}: clean_err={clean_error_rate:.4f}, "
                f"adv_evasion={adv_evasion_rate:.4f}"
            )

        # Compute interaction effect Δ
        baseline = results["levels"][0]  # 100% capture
        deltas = []
        for level in results["levels"]:
            delta = (
                level["adv_evasion_rate"] - baseline["adv_evasion_rate"]
                - (level["clean_error_rate"] - baseline["clean_error_rate"])
            )
            level["interaction_delta"] = delta
            deltas.append(delta)

        results["interaction_deltas"] = deltas
        results["max_interaction_delta"] = max(deltas)
        results["h2_supported"] = max(deltas) > 0

        logger.info(f"\nMax interaction Δ: {max(deltas):.4f}")
        logger.info(f"H2 supported: {results['h2_supported']}")

        return results

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rq2_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        df = pd.DataFrame(results["levels"])
        df.to_csv(output_dir / "rq2_evasion_table.csv", index=False)
        logger.info(f"RQ2 results saved to {output_dir}")
