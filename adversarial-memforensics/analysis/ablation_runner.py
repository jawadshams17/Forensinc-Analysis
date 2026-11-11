"""
Ablation Runner — Systematic Component Contribution Analysis
=============================================================
Runs ablation studies to quantify the contribution of each
perturbation primitive and truncation factor independently.

Configurations:
    1. Individual primitives only (P1 only, P2 only, P3 only)
    2. Pairwise combinations (P1+P2, P1+P3, P2+P3)
    3. Full combination (P1+P2+P3)
    4. Truncation only (no adversarial)
    5. Adversarial only (no truncation)
    6. Combined (adversarial + truncation)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor
from perturbation_primitives import (
    BasePrimitive,
    PerturbationAction,
    PerturbationSequence,
    apply_feature_deltas,
    VOLMEMLYZER_FEATURES,
)
from perturbation_primitives.child_spawn import BenignChildSpawner
from perturbation_primitives.dll_padding import DLLLoadPadder
from perturbation_primitives.handle_padding import HandleTablePadder
from perturbation_primitives.optimizer import EnsembleOracle


class AblationRunner:
    """
    Runs ablation studies to isolate the contribution of each component.
    """

    def __init__(
        self,
        classifier: EnsembleClassifier,
        extractor: FeatureExtractor,
        truncation_levels: Optional[List[float]] = None,
    ):
        self.classifier = classifier
        self.extractor = extractor
        self.truncation_levels = truncation_levels or [1.0, 0.50, 0.25]

        # Core primitives
        self.primitives: Dict[str, BasePrimitive] = {
            "P1_child_spawn": BenignChildSpawner(),
            "P2_dll_padding": DLLLoadPadder(),
            "P3_handle_padding": HandleTablePadder(),
        }

        # Default parameters per primitive
        self.default_params: Dict[str, Dict] = {
            "P1_child_spawn": {"n_shortlived": 15, "n_longlived": 5},
            "P2_dll_padding": {"n_dlls": 8},
            "P3_handle_padding": {"n_file_handles": 20, "n_reg_handles": 10,
                                  "n_event_handles": 10, "n_mutex_handles": 5},
        }

    def _generate_ablation_configs(self) -> List[Dict[str, Any]]:
        """Generate all ablation configurations."""
        configs = []
        prim_names = list(self.primitives.keys())

        # 1. No perturbation baseline
        configs.append({
            "name": "baseline_no_perturbation",
            "primitives": [],
            "description": "No adversarial perturbation (clean baseline)",
        })

        # 2. Individual primitives
        for name in prim_names:
            configs.append({
                "name": f"single_{name}",
                "primitives": [name],
                "description": f"Only {name}",
            })

        # 3. Pairwise combinations
        for combo in itertools.combinations(prim_names, 2):
            combo_name = "+".join(c.split("_")[0] for c in combo)
            configs.append({
                "name": f"pair_{combo_name}",
                "primitives": list(combo),
                "description": f"Pair: {combo_name}",
            })

        # 4. Full combination
        configs.append({
            "name": "full_P1+P2+P3",
            "primitives": list(prim_names),
            "description": "All three core primitives",
        })

        return configs

    def run_ablation(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run the complete ablation study.

        For each (primitive_combo, truncation_level):
            1. Apply predicted feature deltas from the combo
            2. Simulate truncation
            3. Run inference
            4. Record evasion rate and accuracy
        """
        logger.info("=" * 60)
        logger.info("ABLATION STUDY — COMPONENT CONTRIBUTION ANALYSIS")
        logger.info("=" * 60)

        configs = self._generate_ablation_configs()
        oracle = EnsembleOracle(
            self.classifier.predict_proba,
            self.extractor.feature_names,
        )

        results = []
        mask_malicious = y != 0

        for config in configs:
            # Build perturbation sequence for this config
            if config["primitives"]:
                actions = []
                for prim_name in config["primitives"]:
                    prim = self.primitives[prim_name]
                    params = self.default_params[prim_name]
                    actions.append(PerturbationAction(primitive=prim, params=params))
                seq = PerturbationSequence(actions=actions)
                deltas = seq.predict_combined_deltas()
            else:
                deltas = {}

            for pct in self.truncation_levels:
                # Apply perturbation
                if deltas:
                    X_perturbed = np.array([
                        apply_feature_deltas(x, self.extractor.feature_names, deltas)
                        for x in X
                    ])
                else:
                    X_perturbed = X.copy()

                # Apply truncation
                X_trunc = self.extractor.simulate_truncation(X_perturbed, pct)

                # Predict
                y_pred = self.classifier.predict(X_trunc)

                # Metrics
                accuracy = np.mean(y_pred == y)
                evasion_count = np.sum(y_pred[mask_malicious] == 0)
                evasion_rate = evasion_count / np.sum(mask_malicious) if np.sum(mask_malicious) > 0 else 0.0

                result_row = {
                    "config_name": config["name"],
                    "description": config["description"],
                    "primitives": config["primitives"],
                    "truncation_pct": pct,
                    "accuracy": accuracy,
                    "evasion_rate": evasion_rate,
                    "n_evasions": int(evasion_count),
                    "n_samples": len(y),
                    "n_malicious": int(np.sum(mask_malicious)),
                }
                results.append(result_row)

                logger.info(
                    f"  {config['name']} @ {pct:.0%}: "
                    f"acc={accuracy:.4f}, evasion={evasion_rate:.4f}"
                )

        # Compute contribution (difference vs baseline)
        baseline_results = {
            pct: next(r for r in results
                      if r["config_name"] == "baseline_no_perturbation"
                      and r["truncation_pct"] == pct)
            for pct in self.truncation_levels
        }

        for r in results:
            baseline = baseline_results[r["truncation_pct"]]
            r["accuracy_delta_vs_baseline"] = r["accuracy"] - baseline["accuracy"]
            r["evasion_delta_vs_baseline"] = r["evasion_rate"] - baseline["evasion_rate"]

        output = {
            "configs": [c["name"] for c in configs],
            "truncation_levels": self.truncation_levels,
            "results": results,
        }

        return output

    def save_results(self, results: Dict, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        df = pd.DataFrame(results["results"])
        df.to_csv(output_dir / "ablation_table.csv", index=False)
        logger.info(f"Ablation results saved to {output_dir}")
