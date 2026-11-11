"""
Phase 0 Feasibility Gate — Go/No-Go Evaluation
================================================
Implements the mandatory 3-week feasibility pilot that must satisfy
all go/no-go criteria before the full cloud campaign (Phase 2) is funded.

Three Criteria:
    1. Functionality Preservation — primitives preserve malicious behavior
    2. Feature-Vector Shift Accuracy — measured deltas match predictions
    3. Evasion Efficacy — at least one combination reduces f_θ(x) by ≥ 0.1

If ANY criterion fails → pivot to truncation-only study (RQ1).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from perturbation_primitives import (
    BasePrimitive,
    FeatureDelta,
    PrimitiveResult,
    PerturbationAction,
    PerturbationSequence,
    apply_feature_deltas,
    VOLMEMLYZER_FEATURES,
)
from perturbation_primitives.child_spawn import BenignChildSpawner
from perturbation_primitives.dll_padding import DLLLoadPadder
from perturbation_primitives.handle_padding import HandleTablePadder
from perturbation_primitives.optimizer import (
    EnsembleOracle,
    GAMMAOptimizer,
    OptimizerConfig,
    OptimizationResult,
)


# ── Data Structures ────────────────────────────────────────────────

@dataclass
class FunctionalityTestResult:
    """Result of testing functionality preservation for one combination."""
    primitive_name: str
    malware_category: str
    malware_family: str
    baseline_behaviors: List[str]      # expected behaviors
    post_perturbation_behaviors: List[str]  # observed behaviors
    preserved: bool
    notes: str = ""


@dataclass
class FeatureShiftTestResult:
    """Result of testing feature-vector shift accuracy for one primitive."""
    primitive_name: str
    sample_id: str
    predicted_deltas: Dict[str, Tuple[int, float]]   # feature → (direction, magnitude)
    measured_deltas: Dict[str, Tuple[int, float]]
    direction_matches: Dict[str, bool]
    magnitude_within_tolerance: Dict[str, bool]
    overall_pass: bool
    tolerance: float = 0.5  # 50% tolerance on magnitude


@dataclass
class EvasionTestResult:
    """Result of testing evasion efficacy for one primitive combination."""
    combination_id: str
    primitives_used: List[str]
    original_prob: float
    perturbed_prob: float
    probability_drop: float
    query_budget_used: int
    meets_threshold: bool    # drop ≥ 0.1


@dataclass
class Phase0Report:
    """
    Complete Phase 0 assessment report containing all criterion results
    and the final go/no-go decision.
    """
    # Criterion 1: Functionality Preservation
    criterion_1_results: List[FunctionalityTestResult] = field(default_factory=list)
    criterion_1_pass_ratio: float = 0.0
    criterion_1_passed: bool = False

    # Criterion 2: Feature Shift Accuracy
    criterion_2_results: List[FeatureShiftTestResult] = field(default_factory=list)
    criterion_2_pass_ratio: float = 0.0
    criterion_2_passed: bool = False

    # Criterion 3: Evasion Efficacy
    criterion_3_results: List[EvasionTestResult] = field(default_factory=list)
    criterion_3_best_drop: float = 0.0
    criterion_3_passed: bool = False

    # Overall decision
    go_decision: bool = False
    timestamp: float = field(default_factory=time.time)
    duration_seconds: float = 0.0
    notes: str = ""

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 70,
            "PHASE 0 FEASIBILITY GATE — ASSESSMENT REPORT",
            "=" * 70,
            "",
            f"Timestamp: {time.ctime(self.timestamp)}",
            f"Duration:  {self.duration_seconds:.1f}s",
            "",
            "─── Criterion 1: Functionality Preservation ───",
            f"  Pass ratio: {self.criterion_1_pass_ratio:.1%} "
            f"(threshold: ≥88.9%)",
            f"  Result:     {'✅ PASS' if self.criterion_1_passed else '❌ FAIL'}",
            f"  Tests:      {len(self.criterion_1_results)}",
            "",
            "─── Criterion 2: Feature-Vector Shift Accuracy ───",
            f"  Pass ratio: {self.criterion_2_pass_ratio:.1%} "
            f"(threshold: ≥80.0%)",
            f"  Result:     {'✅ PASS' if self.criterion_2_passed else '❌ FAIL'}",
            f"  Tests:      {len(self.criterion_2_results)}",
            "",
            "─── Criterion 3: Evasion Efficacy ───",
            f"  Best drop:  {self.criterion_3_best_drop:.4f} "
            f"(threshold: ≥0.1)",
            f"  Result:     {'✅ PASS' if self.criterion_3_passed else '❌ FAIL'}",
            f"  Tests:      {len(self.criterion_3_results)}",
            "",
            "═" * 70,
            f"  DECISION: {'🟢 GO — Proceed to Phase 2' if self.go_decision else '🔴 NO-GO — Pivot to truncation-only (RQ1)'}",
            "═" * 70,
        ]
        if self.notes:
            lines.extend(["", f"Notes: {self.notes}"])
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        """Save the report to JSON and text summary."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # JSON (machine-readable)
        json_data = {
            "criterion_1": {
                "pass_ratio": self.criterion_1_pass_ratio,
                "passed": self.criterion_1_passed,
                "n_tests": len(self.criterion_1_results),
            },
            "criterion_2": {
                "pass_ratio": self.criterion_2_pass_ratio,
                "passed": self.criterion_2_passed,
                "n_tests": len(self.criterion_2_results),
            },
            "criterion_3": {
                "best_drop": self.criterion_3_best_drop,
                "passed": self.criterion_3_passed,
                "n_tests": len(self.criterion_3_results),
            },
            "go_decision": self.go_decision,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "notes": self.notes,
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(json_data, f, indent=2)

        # Text summary (human-readable)
        with open(path.with_suffix(".txt"), "w") as f:
            f.write(self.summary())

        logger.info(f"Phase 0 report saved to {path}")


# ── Feasibility Evaluator ──────────────────────────────────────────

class Phase0Evaluator:
    """
    Runs the complete Phase 0 feasibility evaluation.

    This is the MANDATORY gate that must be passed before Phase 2
    (cloud data collection) is funded.
    """

    def __init__(
        self,
        ensemble_predict_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: Optional[List[str]] = None,
        min_pass_ratio_c1: float = 0.889,   # ≥ 8/9
        min_pass_ratio_c2: float = 0.80,
        min_probability_drop_c3: float = 0.1,
        max_query_budget_c3: int = 50,
    ):
        """
        Args:
            ensemble_predict_fn: f_θ predict_proba function
            feature_names:       VolMemLyzer feature names
            min_pass_ratio_c1:   Criterion 1 minimum pass ratio
            min_pass_ratio_c2:   Criterion 2 minimum pass ratio
            min_probability_drop_c3: Criterion 3 minimum probability drop
            max_query_budget_c3: Criterion 3 max query budget
        """
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.oracle = EnsembleOracle(
            predict_proba_fn=ensemble_predict_fn,
            feature_names=self.feature_names,
        )
        self.min_pass_ratio_c1 = min_pass_ratio_c1
        self.min_pass_ratio_c2 = min_pass_ratio_c2
        self.min_probability_drop_c3 = min_probability_drop_c3
        self.max_query_budget_c3 = max_query_budget_c3

        # Initialize core primitives
        self.primitives: List[BasePrimitive] = [
            BenignChildSpawner(),
            DLLLoadPadder(),
            HandleTablePadder(),
        ]

    # ── Criterion 1 ────────────────────────────────────────────────

    def evaluate_criterion_1(
        self,
        functionality_results: List[FunctionalityTestResult],
    ) -> Tuple[float, bool]:
        """
        Evaluate Criterion 1: Functionality Preservation.

        This criterion requires MANUAL dynamic analysis — the evaluator
        accepts pre-computed results from sandbox testing.

        Args:
            functionality_results: List of per-combination test results,
                populated from manual dynamic analysis.

        Returns:
            (pass_ratio, passed)
        """
        if not functionality_results:
            logger.warning("[C1] No functionality test results provided")
            return 0.0, False

        n_passed = sum(1 for r in functionality_results if r.preserved)
        ratio = n_passed / len(functionality_results)
        passed = ratio >= self.min_pass_ratio_c1

        logger.info(
            f"[C1] Functionality preservation: {n_passed}/{len(functionality_results)} "
            f"= {ratio:.1%} ({'PASS' if passed else 'FAIL'})"
        )

        for r in functionality_results:
            status = "✅" if r.preserved else "❌"
            logger.info(
                f"  {status} {r.primitive_name} × {r.malware_family} "
                f"({r.malware_category}): {r.notes}"
            )

        return ratio, passed

    # ── Criterion 2 ────────────────────────────────────────────────

    def evaluate_criterion_2(
        self,
        feature_vectors_before: Dict[str, np.ndarray],
        feature_vectors_after: Dict[str, np.ndarray],
        primitive_used: Dict[str, BasePrimitive],
        params_used: Dict[str, Dict[str, Any]],
        magnitude_tolerance: float = 0.5,
    ) -> Tuple[float, bool, List[FeatureShiftTestResult]]:
        """
        Evaluate Criterion 2: Feature-Vector Shift Accuracy.

        Compares predicted feature deltas against measured deltas.

        Args:
            feature_vectors_before: {sample_id: 55-dim vector before perturbation}
            feature_vectors_after:  {sample_id: 55-dim vector after perturbation}
            primitive_used:         {sample_id: primitive instance used}
            params_used:            {sample_id: params dict used}
            magnitude_tolerance:    Acceptable deviation ratio (0.5 = 50%)

        Returns:
            (pass_ratio, passed, detailed_results)
        """
        results: List[FeatureShiftTestResult] = []

        for sample_id in feature_vectors_before:
            before = feature_vectors_before[sample_id]
            after = feature_vectors_after[sample_id]
            primitive = primitive_used[sample_id]
            params = params_used[sample_id]

            # Get predicted deltas
            predicted = primitive.predict_deltas(**params)
            pred_dict = {d.feature_name: (d.direction, d.magnitude) for d in predicted}

            # Compute measured deltas
            name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
            meas_dict = {}
            dir_matches = {}
            mag_within = {}

            for feat_name, (pred_dir, pred_mag) in pred_dict.items():
                if feat_name in name_to_idx:
                    idx = name_to_idx[feat_name]
                    actual_delta = after[idx] - before[idx]
                    meas_dir = int(np.sign(actual_delta))
                    meas_mag = abs(actual_delta)
                    meas_dict[feat_name] = (meas_dir, meas_mag)

                    # Direction match
                    dir_matches[feat_name] = (meas_dir == pred_dir) or pred_dir == 0

                    # Magnitude within tolerance
                    if pred_mag > 0:
                        mag_ratio = abs(meas_mag - pred_mag) / pred_mag
                        mag_within[feat_name] = mag_ratio <= magnitude_tolerance
                    else:
                        mag_within[feat_name] = meas_mag < 1.0  # small delta OK

            overall_pass = all(dir_matches.values()) if dir_matches else False

            results.append(FeatureShiftTestResult(
                primitive_name=primitive.name,
                sample_id=sample_id,
                predicted_deltas=pred_dict,
                measured_deltas=meas_dict,
                direction_matches=dir_matches,
                magnitude_within_tolerance=mag_within,
                overall_pass=overall_pass,
                tolerance=magnitude_tolerance,
            ))

        n_passed = sum(1 for r in results if r.overall_pass)
        ratio = n_passed / len(results) if results else 0.0
        passed = ratio >= self.min_pass_ratio_c2

        logger.info(
            f"[C2] Feature shift accuracy: {n_passed}/{len(results)} "
            f"= {ratio:.1%} ({'PASS' if passed else 'FAIL'})"
        )

        return ratio, passed, results

    # ── Criterion 3 ────────────────────────────────────────────────

    def evaluate_criterion_3(
        self,
        malware_features: Dict[str, np.ndarray],
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> Tuple[float, bool, List[EvasionTestResult]]:
        """
        Evaluate Criterion 3: Evasion Efficacy.

        Runs the GAMMA optimizer on each malware sample and checks if
        ANY combination can reduce f_θ(x) by ≥ threshold.

        Args:
            malware_features: {sample_id: 55-dim feature vector}
            optimizer_config: Configuration for the optimizer

        Returns:
            (best_drop, passed, detailed_results)
        """
        config = optimizer_config or OptimizerConfig(
            query_budget=self.max_query_budget_c3,
        )
        results: List[EvasionTestResult] = []
        best_drop = 0.0

        for sample_id, x in malware_features.items():
            logger.info(f"[C3] Testing evasion on sample: {sample_id}")

            optimizer = GAMMAOptimizer(config)
            opt_result = optimizer.optimize(x, self.oracle)

            evasion_result = EvasionTestResult(
                combination_id=sample_id,
                primitives_used=[
                    a.primitive.name for a in opt_result.best_sequence.actions
                ],
                original_prob=opt_result.original_prob,
                perturbed_prob=opt_result.final_prob,
                probability_drop=opt_result.probability_drop,
                query_budget_used=opt_result.total_queries,
                meets_threshold=opt_result.probability_drop >= self.min_probability_drop_c3,
            )
            results.append(evasion_result)

            if opt_result.probability_drop > best_drop:
                best_drop = opt_result.probability_drop

            logger.info(
                f"  {sample_id}: {opt_result.original_prob:.4f} → "
                f"{opt_result.final_prob:.4f} (Δ={opt_result.probability_drop:.4f})"
            )

        passed = best_drop >= self.min_probability_drop_c3

        logger.info(
            f"[C3] Evasion efficacy: best_drop={best_drop:.4f} "
            f"(threshold: {self.min_probability_drop_c3}) "
            f"{'PASS' if passed else 'FAIL'}"
        )

        return best_drop, passed, results

    # ── Full Evaluation ────────────────────────────────────────────

    def run_full_evaluation(
        self,
        functionality_results: List[FunctionalityTestResult],
        feature_vectors_before: Dict[str, np.ndarray],
        feature_vectors_after: Dict[str, np.ndarray],
        primitive_used: Dict[str, BasePrimitive],
        params_used: Dict[str, Dict[str, Any]],
        malware_features: Dict[str, np.ndarray],
        optimizer_config: Optional[OptimizerConfig] = None,
    ) -> Phase0Report:
        """
        Run the complete Phase 0 evaluation across all three criteria.

        Returns:
            Phase0Report with go/no-go decision
        """
        start_time = time.time()
        report = Phase0Report(timestamp=start_time)

        logger.info("=" * 70)
        logger.info("PHASE 0 FEASIBILITY GATE — STARTING EVALUATION")
        logger.info("=" * 70)

        # Criterion 1: Functionality Preservation
        c1_ratio, c1_passed = self.evaluate_criterion_1(functionality_results)
        report.criterion_1_results = functionality_results
        report.criterion_1_pass_ratio = c1_ratio
        report.criterion_1_passed = c1_passed

        # Criterion 2: Feature Shift Accuracy
        c2_ratio, c2_passed, c2_results = self.evaluate_criterion_2(
            feature_vectors_before, feature_vectors_after,
            primitive_used, params_used,
        )
        report.criterion_2_results = c2_results
        report.criterion_2_pass_ratio = c2_ratio
        report.criterion_2_passed = c2_passed

        # Criterion 3: Evasion Efficacy
        c3_drop, c3_passed, c3_results = self.evaluate_criterion_3(
            malware_features, optimizer_config,
        )
        report.criterion_3_results = c3_results
        report.criterion_3_best_drop = c3_drop
        report.criterion_3_passed = c3_passed

        # Go/No-Go Decision
        report.go_decision = c1_passed and c2_passed and c3_passed
        report.duration_seconds = time.time() - start_time

        logger.info("\n" + report.summary())

        return report


# ── CLI Entry Point ────────────────────────────────────────────────

def main():
    """CLI entry point for Phase 0 evaluation."""
    import click

    @click.command()
    @click.option("--config", type=click.Path(exists=True), help="Experiment config YAML")
    @click.option("--output-dir", type=click.Path(), default="results/phase0")
    @click.option("--dry-run", is_flag=True, help="Run with synthetic data for testing")
    def cli(config, output_dir, dry_run):
        """Run Phase 0 feasibility gate evaluation."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if dry_run:
            logger.info("DRY RUN MODE — using synthetic data")

            # Create synthetic ensemble for testing
            from sklearn.ensemble import RandomForestClassifier
            rng = np.random.default_rng(42)

            # Synthetic training data
            n = 1000
            X_train = rng.random((n, 55))
            y_train = rng.integers(0, 2, n)
            clf = RandomForestClassifier(n_estimators=10, random_state=42)
            clf.fit(X_train, y_train)

            # Create evaluator
            evaluator = Phase0Evaluator(
                ensemble_predict_fn=clf.predict_proba,
            )

            # Synthetic functionality results
            func_results = []
            categories = ["Ransomware", "Spyware", "Trojan"]
            for prim in evaluator.primitives:
                for cat in categories:
                    func_results.append(FunctionalityTestResult(
                        primitive_name=prim.name,
                        malware_category=cat,
                        malware_family=f"{cat}_family_1",
                        baseline_behaviors=["c2_contact", "persistence"],
                        post_perturbation_behaviors=["c2_contact", "persistence"],
                        preserved=True,
                        notes="Synthetic test — assumed pass",
                    ))

            # Synthetic feature vectors
            n_samples = 5
            X_before = rng.random((n_samples, 55))
            X_after = X_before.copy()
            # Simulate perturbation effects
            X_after[:, 0] += 5   # pslist.nproc
            X_after[:, 5] += 3   # dlllist.ndlls
            X_after[:, 8] += 10  # handles.nhandles

            fv_before = {f"sample_{i}": X_before[i] for i in range(n_samples)}
            fv_after = {f"sample_{i}": X_after[i] for i in range(n_samples)}
            prims = {f"sample_{i}": evaluator.primitives[i % 3] for i in range(n_samples)}
            params = {f"sample_{i}": {} for i in range(n_samples)}

            mal_features = {f"mal_{i}": rng.random(55) for i in range(3)}

            # Run evaluation
            report = evaluator.run_full_evaluation(
                functionality_results=func_results,
                feature_vectors_before=fv_before,
                feature_vectors_after=fv_after,
                primitive_used=prims,
                params_used=params,
                malware_features=mal_features,
            )

            report.save(output_path / "phase0_report")
            click.echo(report.summary())
        else:
            click.echo("Full evaluation requires real data. Use --dry-run for testing.")

    cli()


if __name__ == "__main__":
    main()
