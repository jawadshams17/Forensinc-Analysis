"""
GAMMA-Adapted Black-Box Optimizer for Memory Feature Evasion
=============================================================
Adapts Demetrio et al.'s GAMMA framework from PE-header-space
optimization to runtime memory-feature-space evasion.

The optimizer finds the perturbation sequence δ* that minimizes
the ensemble's malicious-class probability:

    δ* = argmin_{δ ∈ A^k} f_θ(x ⊕ δ) + λ·c(δ)

Where:
    - f_θ(x)  = ensemble's soft-voting malicious probability
    - A       = action set (child_spawn, dll_pad, handle_pad)
    - δ       = sequence of up to k primitive applications
    - c(δ)    = cost penalty (number of primitives in sequence)
    - λ       = cost-evasion tradeoff parameter

Uses genetic/greedy black-box search:
    - Population-based: maintains k candidate sequences
    - Mutation:  add/remove/swap primitives with probability p
    - Crossover: recombine top-performing sequences
    - Selection: keep top-50% by fitness
    - Query-efficient: designed for Q ≤ 50 evaluations

CRITICAL: This runs OFFLINE against a surrogate model, NOT during
the live truncation window. Only the pre-computed δ* is deployed
at runtime (milliseconds to execute).
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from perturbation_primitives import (
    BasePrimitive,
    FeatureDelta,
    PerturbationAction,
    PerturbationSequence,
    apply_feature_deltas,
    VOLMEMLYZER_FEATURES,
)
from perturbation_primitives.child_spawn import BenignChildSpawner
from perturbation_primitives.dll_padding import DLLLoadPadder
from perturbation_primitives.handle_padding import HandleTablePadder


# ── Optimizer Configuration ────────────────────────────────────────

@dataclass
class OptimizerConfig:
    """Configuration for the GAMMA-adapted optimizer."""
    population_size: int = 20
    generations: int = 10
    mutation_probability: float = 0.2
    query_budget: int = 50
    cost_penalty_lambda: float = 0.1
    selection_ratio: float = 0.5
    max_sequence_length: int = 5
    random_seed: int = 42

    # Primitive parameter ranges for random initialization
    child_spawn_n_range: Tuple[int, int] = (5, 30)
    child_spawn_hold_range: Tuple[int, int] = (3, 8)
    dll_pad_n_range: Tuple[int, int] = (3, 15)
    handle_pad_files_range: Tuple[int, int] = (10, 50)
    handle_pad_reg_range: Tuple[int, int] = (5, 25)
    handle_pad_event_range: Tuple[int, int] = (5, 20)


@dataclass
class OptimizationResult:
    """Result of a single optimization run."""
    best_sequence: PerturbationSequence
    best_fitness: float
    original_prob: float           # f_θ(x) before perturbation
    final_prob: float              # f_θ(x ⊕ δ*) after perturbation
    probability_drop: float        # original - final
    evasion_success: bool          # final_prob < 0.5
    total_queries: int
    total_time_seconds: float
    generation_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging/storage."""
        return {
            "best_fitness": self.best_fitness,
            "original_prob": self.original_prob,
            "final_prob": self.final_prob,
            "probability_drop": self.probability_drop,
            "evasion_success": self.evasion_success,
            "total_queries": self.total_queries,
            "total_time_seconds": self.total_time_seconds,
            "sequence_length": self.best_sequence.length,
            "sequence": self.best_sequence.to_dict(),
            "generation_history": self.generation_history,
        }

    def save(self, path: Path) -> None:
        """Save optimization result to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Optimization result saved to {path}")


# ── Classifier Interface ───────────────────────────────────────────

class EnsembleOracle:
    """
    Wrapper for the ensemble classifier that provides a query interface
    for the optimizer. Acts as the "oracle" in the black-box setting.

    In the REAL threat model, the attacker queries a SURROGATE model.
    For experimental purposes, we query the actual model to measure
    true evasion rates.
    """

    def __init__(
        self,
        predict_proba_fn: Callable[[np.ndarray], np.ndarray],
        feature_names: List[str],
        malicious_class_idx: int = 1,
    ):
        """
        Args:
            predict_proba_fn: Function that takes (n_samples, n_features)
                              and returns (n_samples, n_classes) probabilities
            feature_names:    Ordered list of feature names
            malicious_class_idx: Index of the malicious class in output
        """
        self.predict_proba_fn = predict_proba_fn
        self.feature_names = feature_names
        self.malicious_class_idx = malicious_class_idx
        self._query_count = 0

    def query(self, feature_vector: np.ndarray) -> float:
        """
        Query the ensemble for malicious-class probability.
        Returns f_θ(x) ∈ [0, 1].
        """
        self._query_count += 1
        x = feature_vector.reshape(1, -1)
        proba = self.predict_proba_fn(x)
        return float(proba[0, self.malicious_class_idx])

    @property
    def query_count(self) -> int:
        return self._query_count

    def reset_count(self) -> None:
        self._query_count = 0


# ── GAMMA-Adapted Optimizer ────────────────────────────────────────

class GAMMAOptimizer:
    """
    Genetic/greedy black-box optimizer for memory-feature-space evasion.

    Adapts the GAMMA framework (Demetrio et al., 2021) from PE-header
    content injection to runtime memory-feature perturbation.

    Key differences from GAMMA:
      1. Action space is runtime primitives, not file-structure modifications
      2. Feature deltas are PREDICTED (from primitive models), not computed
         from actual file mutation
      3. Cost is measured in number of primitives, not payload bytes
      4. Query budget is much smaller (Q≤50) since we expect transfer
         from surrogate to production model
    """

    def __init__(self, config: OptimizerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

        # Initialize available primitives
        self.primitives: List[BasePrimitive] = [
            BenignChildSpawner(),
            DLLLoadPadder(),
            HandleTablePadder(),
        ]

    def _random_params(self, primitive: BasePrimitive) -> Dict[str, Any]:
        """Generate random parameters for a primitive."""
        c = self.config
        if primitive.name == "benign_child_spawn":
            return {
                "n_shortlived": int(self.rng.integers(*c.child_spawn_n_range)),
                "n_longlived": int(self.rng.integers(*c.child_spawn_hold_range)),
                "hold_time": float(self.rng.uniform(0.05, 0.5)),
            }
        elif primitive.name == "dll_load_padding":
            return {
                "n_dlls": int(self.rng.integers(*c.dll_pad_n_range)),
                "select_random": True,
            }
        elif primitive.name == "handle_table_padding":
            return {
                "n_file_handles": int(self.rng.integers(*c.handle_pad_files_range)),
                "n_reg_handles": int(self.rng.integers(*c.handle_pad_reg_range)),
                "n_event_handles": int(self.rng.integers(*c.handle_pad_event_range)),
                "n_mutex_handles": int(self.rng.integers(2, 10)),
            }
        return {}

    def _random_sequence(self) -> PerturbationSequence:
        """Generate a random perturbation sequence."""
        length = int(self.rng.integers(1, self.config.max_sequence_length + 1))
        actions = []
        for _ in range(length):
            primitive = self.rng.choice(self.primitives)
            params = self._random_params(primitive)
            actions.append(PerturbationAction(primitive=primitive, params=params))
        return PerturbationSequence(actions=actions)

    def _evaluate_sequence(
        self,
        sequence: PerturbationSequence,
        feature_vector: np.ndarray,
        oracle: EnsembleOracle,
    ) -> float:
        """
        Evaluate a perturbation sequence's fitness.

        fitness = -f_θ(x ⊕ δ) - λ·c(δ)

        Higher fitness = lower detection probability + lower cost.
        """
        # Predict combined feature deltas
        deltas = sequence.predict_combined_deltas()

        # Apply deltas to feature vector (the ⊕ operator)
        x_prime = apply_feature_deltas(
            feature_vector, oracle.feature_names, deltas
        )

        # Query the oracle
        mal_prob = oracle.query(x_prime)

        # Compute fitness: minimize detection + minimize cost
        cost = sequence.length * self.config.cost_penalty_lambda
        fitness = -mal_prob - cost

        sequence.fitness = fitness
        sequence.total_cost = cost

        return fitness

    def _crossover(
        self,
        parent1: PerturbationSequence,
        parent2: PerturbationSequence,
    ) -> PerturbationSequence:
        """Single-point crossover between two parent sequences."""
        if parent1.length == 0 or parent2.length == 0:
            return copy.deepcopy(parent1 if parent1.fitness > parent2.fitness else parent2)

        cut1 = int(self.rng.integers(0, parent1.length))
        cut2 = int(self.rng.integers(0, parent2.length))

        child_actions = (
            copy.deepcopy(parent1.actions[:cut1 + 1])
            + copy.deepcopy(parent2.actions[cut2:])
        )

        # Enforce max length
        child_actions = child_actions[: self.config.max_sequence_length]

        return PerturbationSequence(actions=child_actions)

    def _mutate(self, sequence: PerturbationSequence) -> PerturbationSequence:
        """Mutate a sequence: add, remove, or swap one primitive."""
        seq = copy.deepcopy(sequence)

        if self.rng.random() > self.config.mutation_probability:
            return seq  # no mutation

        mutation_type = self.rng.choice(["add", "remove", "swap", "tweak_params"])

        if mutation_type == "add" and seq.length < self.config.max_sequence_length:
            primitive = self.rng.choice(self.primitives)
            params = self._random_params(primitive)
            pos = int(self.rng.integers(0, seq.length + 1))
            seq.actions.insert(pos, PerturbationAction(primitive=primitive, params=params))

        elif mutation_type == "remove" and seq.length > 1:
            pos = int(self.rng.integers(0, seq.length))
            seq.actions.pop(pos)

        elif mutation_type == "swap" and seq.length > 0:
            pos = int(self.rng.integers(0, seq.length))
            primitive = self.rng.choice(self.primitives)
            params = self._random_params(primitive)
            seq.actions[pos] = PerturbationAction(primitive=primitive, params=params)

        elif mutation_type == "tweak_params" and seq.length > 0:
            pos = int(self.rng.integers(0, seq.length))
            action = seq.actions[pos]
            new_params = self._random_params(action.primitive)
            # Blend: 50% old params, 50% new params (numeric only)
            for key in new_params:
                if key in action.params and isinstance(action.params[key], (int, float)):
                    action.params[key] = int(
                        0.5 * action.params[key] + 0.5 * new_params[key]
                    )

        return seq

    def optimize(
        self,
        feature_vector: np.ndarray,
        oracle: EnsembleOracle,
    ) -> OptimizationResult:
        """
        Run the genetic optimizer to find the best evasion sequence.

        Args:
            feature_vector: Original 55-dim feature vector (clean sample)
            oracle:         Ensemble classifier query interface

        Returns:
            OptimizationResult with best sequence and metrics
        """
        start_time = time.time()
        oracle.reset_count()

        # Get baseline malicious probability
        original_prob = oracle.query(feature_vector)
        logger.info(f"[Optimizer] Original malicious probability: {original_prob:.6f}")

        # Initialize population
        population = [self._random_sequence() for _ in range(self.config.population_size)]

        # Evaluate initial population
        for seq in population:
            if oracle.query_count >= self.config.query_budget:
                break
            self._evaluate_sequence(seq, feature_vector, oracle)

        generation_history: List[Dict[str, Any]] = []
        best_overall = max(population, key=lambda s: s.fitness)

        # Evolution loop
        for gen in range(self.config.generations):
            if oracle.query_count >= self.config.query_budget:
                logger.info(f"[Optimizer] Query budget exhausted at generation {gen}")
                break

            # Selection: keep top k
            population.sort(key=lambda s: s.fitness, reverse=True)
            n_select = max(2, int(len(population) * self.config.selection_ratio))
            survivors = population[:n_select]

            # Generate offspring
            offspring = []
            while len(offspring) < self.config.population_size - n_select:
                if oracle.query_count >= self.config.query_budget:
                    break

                # Select parents
                p1 = survivors[int(self.rng.integers(0, len(survivors)))]
                p2 = survivors[int(self.rng.integers(0, len(survivors)))]

                # Crossover + mutation
                child = self._crossover(p1, p2)
                child = self._mutate(child)

                # Evaluate
                self._evaluate_sequence(child, feature_vector, oracle)
                offspring.append(child)

            population = survivors + offspring

            # Track best
            gen_best = max(population, key=lambda s: s.fitness)
            if gen_best.fitness > best_overall.fitness:
                best_overall = copy.deepcopy(gen_best)

            # Compute gen_best's post-perturbation probability
            deltas = gen_best.predict_combined_deltas()
            x_prime = apply_feature_deltas(
                feature_vector, oracle.feature_names, deltas
            )
            gen_best_prob = oracle.query(x_prime)

            gen_info = {
                "generation": gen,
                "best_fitness": gen_best.fitness,
                "best_prob": gen_best_prob,
                "avg_fitness": np.mean([s.fitness for s in population]),
                "population_size": len(population),
                "queries_used": oracle.query_count,
            }
            generation_history.append(gen_info)

            logger.info(
                f"[Optimizer] Gen {gen}: best_fitness={gen_best.fitness:.4f}, "
                f"best_prob={gen_best_prob:.4f}, queries={oracle.query_count}"
            )

        # Final evaluation of best sequence
        deltas = best_overall.predict_combined_deltas()
        x_prime = apply_feature_deltas(
            feature_vector, oracle.feature_names, deltas
        )
        final_prob = oracle.query(x_prime)

        elapsed = time.time() - start_time

        result = OptimizationResult(
            best_sequence=best_overall,
            best_fitness=best_overall.fitness,
            original_prob=original_prob,
            final_prob=final_prob,
            probability_drop=original_prob - final_prob,
            evasion_success=final_prob < 0.5,
            total_queries=oracle.query_count,
            total_time_seconds=elapsed,
            generation_history=generation_history,
        )

        logger.info(
            f"[Optimizer] COMPLETE: {original_prob:.4f} → {final_prob:.4f} "
            f"(Δ={result.probability_drop:.4f}, evasion={'YES' if result.evasion_success else 'NO'}), "
            f"queries={oracle.query_count}, time={elapsed:.1f}s"
        )

        return result


# ── Batch Optimizer ────────────────────────────────────────────────

def optimize_batch(
    feature_vectors: np.ndarray,
    oracle: EnsembleOracle,
    config: Optional[OptimizerConfig] = None,
    labels: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> List[OptimizationResult]:
    """
    Run optimization for a batch of malware feature vectors.

    Args:
        feature_vectors: (n_samples, 55) array of feature vectors
        oracle:          Shared ensemble oracle
        config:          Optimizer configuration
        labels:          Optional sample labels for logging
        output_dir:      Optional directory to save results

    Returns:
        List of OptimizationResult, one per sample
    """
    config = config or OptimizerConfig()
    results: List[OptimizationResult] = []

    for i, x in enumerate(feature_vectors):
        label = labels[i] if labels else f"sample_{i}"
        logger.info(f"[Batch] Optimizing sample {i+1}/{len(feature_vectors)}: {label}")

        optimizer = GAMMAOptimizer(config)
        result = optimizer.optimize(x, oracle)
        results.append(result)

        if output_dir:
            result.save(output_dir / f"opt_result_{label}.json")

    # Summary statistics
    n_evaded = sum(1 for r in results if r.evasion_success)
    avg_drop = np.mean([r.probability_drop for r in results])
    logger.info(
        f"[Batch] SUMMARY: {n_evaded}/{len(results)} evasion success, "
        f"avg probability drop: {avg_drop:.4f}"
    )

    return results
