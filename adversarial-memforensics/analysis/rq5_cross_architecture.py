"""
RQ5 Analysis — Cross-Architecture Transferability (Extended)
=============================================================
Tests whether adversarial perturbation sequences optimized against
one ensemble architecture transfer to alternative architectures.

Architectures compared:
    - Original ensemble (RF+XGB+LGB)
    - DNN-based detector (feedforward)
    - Individual estimators (RF-only, XGB-only, LGB-only)

Transfer success = evasion optimized against architecture A
                   also evades architecture B.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from perturbation_primitives import apply_feature_deltas, VOLMEMLYZER_FEATURES
from perturbation_primitives.optimizer import OptimizationResult


class CrossArchitectureAnalyzer:
    """
    Tests adversarial transferability across classifier architectures.
    """

    def __init__(
        self,
        architectures: Dict[str, Callable[[np.ndarray], np.ndarray]],
        feature_names: Optional[List[str]] = None,
    ):
        """
        Args:
            architectures: {name: predict_proba_fn} for each architecture
        """
        self.architectures = architectures
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES

    def evaluate_transfer(
        self,
        X_clean: np.ndarray,
        X_adversarial: np.ndarray,
        y_true: np.ndarray,
        source_architecture: str,
    ) -> Dict[str, Any]:
        """
        Evaluate transfer success of perturbations optimized against
        the source architecture.

        Args:
            X_clean:             Clean feature vectors (malicious samples)
            X_adversarial:       Adversarial feature vectors
            y_true:              True labels
            source_architecture: Name of the architecture the attack was optimized against
        """
        logger.info("=" * 60)
        logger.info("RQ5: CROSS-ARCHITECTURE TRANSFERABILITY")
        logger.info(f"Source: {source_architecture}")
        logger.info("=" * 60)

        mask_malicious = y_true != 0
        X_c = X_clean[mask_malicious]
        X_a = X_adversarial[mask_malicious]
        n_mal = X_c.shape[0]

        results = {"source": source_architecture, "targets": []}

        for arch_name, predict_fn in self.architectures.items():
            # Clean predictions
            proba_clean = predict_fn(X_c)
            pred_clean = np.argmax(proba_clean, axis=1)
            clean_errors = np.sum(pred_clean == 0)

            # Adversarial predictions
            proba_adv = predict_fn(X_a)
            pred_adv = np.argmax(proba_adv, axis=1)
            adv_evasions = np.sum(pred_adv == 0)

            # Evasion rates
            clean_error_rate = clean_errors / n_mal if n_mal > 0 else 0.0
            adv_evasion_rate = adv_evasions / n_mal if n_mal > 0 else 0.0

            # Transfer: how many samples that evade source also evade target?
            is_same_arch = (arch_name == source_architecture)

            # Mean malicious probability shift
            mal_prob_clean = 1.0 - proba_clean[:, 0]  # P(malicious)
            mal_prob_adv = 1.0 - proba_adv[:, 0]
            avg_prob_drop = float(np.mean(mal_prob_clean - mal_prob_adv))

            target_result = {
                "architecture": arch_name,
                "is_source": is_same_arch,
                "clean_error_rate": clean_error_rate,
                "adv_evasion_rate": adv_evasion_rate,
                "n_clean_errors": int(clean_errors),
                "n_adv_evasions": int(adv_evasions),
                "avg_malicious_prob_drop": avg_prob_drop,
                "n_samples": int(n_mal),
            }
            results["targets"].append(target_result)

            logger.info(
                f"  {arch_name}{'*' if is_same_arch else ''}: "
                f"evasion={adv_evasion_rate:.4f} (Δprob={avg_prob_drop:+.4f})"
            )

        # Transfer matrix
        source_evasion = next(
            t["adv_evasion_rate"]
            for t in results["targets"]
            if t["architecture"] == source_architecture
        )
        for t in results["targets"]:
            if t["architecture"] != source_architecture and source_evasion > 0:
                t["transfer_ratio"] = t["adv_evasion_rate"] / source_evasion
            else:
                t["transfer_ratio"] = 1.0 if t["is_source"] else 0.0

        # Overall transfer success
        transfer_ratios = [
            t["transfer_ratio"] for t in results["targets"] if not t["is_source"]
        ]
        results["avg_transfer_ratio"] = float(np.mean(transfer_ratios)) if transfer_ratios else 0.0
        results["h5_supported"] = results["avg_transfer_ratio"] > 0.5

        logger.info(f"\nAverage transfer ratio: {results['avg_transfer_ratio']:.4f}")
        logger.info(f"H5 supported: {results['h5_supported']}")

        return results

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "rq5_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        df = pd.DataFrame(results["targets"])
        df.to_csv(output_dir / "rq5_transfer_table.csv", index=False)
        logger.info(f"RQ5 results saved to {output_dir}")
