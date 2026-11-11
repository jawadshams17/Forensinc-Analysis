"""
Unit Tests — Perturbation Primitives
======================================
Tests the core perturbation primitive interfaces, feature delta
predictions, and the GAMMA optimizer mechanics.
"""

from __future__ import annotations

import sys
import unittest
from unittest.mock import patch, MagicMock

import numpy as np

# Add project root to path
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

from perturbation_primitives import (
    BasePrimitive,
    FeatureDelta,
    FeatureGroup,
    PrimitiveResult,
    RiskLevel,
    PerturbationAction,
    PerturbationSequence,
    apply_feature_deltas,
    VOLMEMLYZER_FEATURES,
    FEATURE_GROUPS,
)
from perturbation_primitives.child_spawn import BenignChildSpawner
from perturbation_primitives.dll_padding import DLLLoadPadder
from perturbation_primitives.handle_padding import HandleTablePadder
from perturbation_primitives.malfind_dilution import MalfindDiluter
from perturbation_primitives.optimizer import (
    GAMMAOptimizer,
    OptimizerConfig,
    EnsembleOracle,
    OptimizationResult,
)


class TestFeatureConstants(unittest.TestCase):
    """Tests for feature domain definitions."""

    def test_volmemlyzer_features_count(self):
        """Must have the expected number of VolMemLyzer features."""
        self.assertGreaterEqual(len(VOLMEMLYZER_FEATURES), 50)

    def test_feature_groups_populated(self):
        """Feature groups should map prefix → feature list."""
        expected_groups = {"pslist", "dlllist", "handles", "ldrmodules",
                          "malfind", "modules", "svcscan", "callbacks"}
        self.assertTrue(expected_groups.issubset(set(FEATURE_GROUPS.keys())))

    def test_feature_names_unique(self):
        """All feature names must be unique."""
        self.assertEqual(len(VOLMEMLYZER_FEATURES), len(set(VOLMEMLYZER_FEATURES)))

    def test_feature_group_consistency(self):
        """Every feature should map to its prefix group."""
        for feat in VOLMEMLYZER_FEATURES:
            prefix = feat.split(".")[0]
            self.assertIn(prefix, FEATURE_GROUPS)
            self.assertIn(feat, FEATURE_GROUPS[prefix])


class TestFeatureDelta(unittest.TestCase):
    """Tests for the FeatureDelta dataclass."""

    def test_creation(self):
        delta = FeatureDelta("pslist.nproc", +1, 5.0, 0.85)
        self.assertEqual(delta.feature_name, "pslist.nproc")
        self.assertEqual(delta.direction, +1)
        self.assertEqual(delta.magnitude, 5.0)
        self.assertEqual(delta.confidence, 0.85)

    def test_frozen(self):
        delta = FeatureDelta("pslist.nproc", +1, 5.0, 0.85)
        with self.assertRaises(Exception):
            delta.direction = -1


class TestApplyFeatureDeltas(unittest.TestCase):
    """Tests for the ⊕ operator (apply_feature_deltas)."""

    def test_positive_delta(self):
        x = np.array([10.0, 20.0, 30.0])
        names = ["a.f1", "a.f2", "a.f3"]
        deltas = {"a.f1": FeatureDelta("a.f1", +1, 5.0, 1.0)}
        x_prime = apply_feature_deltas(x, names, deltas)
        self.assertAlmostEqual(x_prime[0], 15.0)
        self.assertAlmostEqual(x_prime[1], 20.0)  # unchanged

    def test_negative_delta_clamped(self):
        """Feature values should never go below 0."""
        x = np.array([2.0])
        names = ["a.f1"]
        deltas = {"a.f1": FeatureDelta("a.f1", -1, 10.0, 1.0)}
        x_prime = apply_feature_deltas(x, names, deltas)
        self.assertAlmostEqual(x_prime[0], 0.0)  # clamped at 0

    def test_nan_preserved(self):
        """NaN features (truncated) should not be modified."""
        x = np.array([np.nan, 10.0])
        names = ["a.f1", "a.f2"]
        deltas = {"a.f1": FeatureDelta("a.f1", +1, 5.0, 1.0)}
        x_prime = apply_feature_deltas(x, names, deltas)
        self.assertTrue(np.isnan(x_prime[0]))

    def test_unknown_feature_ignored(self):
        """Deltas for unknown features should be silently ignored."""
        x = np.array([10.0])
        names = ["a.f1"]
        deltas = {"unknown.feat": FeatureDelta("unknown.feat", +1, 5.0, 1.0)}
        x_prime = apply_feature_deltas(x, names, deltas)
        self.assertAlmostEqual(x_prime[0], 10.0)


class TestBenignChildSpawner(unittest.TestCase):
    """Tests for Primitive 1."""

    def setUp(self):
        self.prim = BenignChildSpawner()

    def test_properties(self):
        self.assertEqual(self.prim.name, "benign_child_spawn")
        self.assertEqual(self.prim.risk_level, RiskLevel.LOW)
        self.assertEqual(self.prim.scope, "core")
        self.assertIn("pslist.nproc", self.prim.target_features)

    def test_predict_deltas(self):
        deltas = self.prim.predict_deltas(n_shortlived=10, n_longlived=5)
        self.assertIsInstance(deltas, list)
        self.assertTrue(all(isinstance(d, FeatureDelta) for d in deltas))

        # nproc should increase
        nproc_delta = next(d for d in deltas if d.feature_name == "pslist.nproc")
        self.assertEqual(nproc_delta.direction, +1)
        self.assertGreater(nproc_delta.magnitude, 0)

    def test_predict_deltas_scaling(self):
        """More processes → larger predicted delta."""
        d_small = self.prim.predict_deltas(n_shortlived=5, n_longlived=2)
        d_large = self.prim.predict_deltas(n_shortlived=30, n_longlived=10)
        nproc_small = next(d for d in d_small if d.feature_name == "pslist.nproc")
        nproc_large = next(d for d in d_large if d.feature_name == "pslist.nproc")
        self.assertGreater(nproc_large.magnitude, nproc_small.magnitude)


class TestDLLLoadPadder(unittest.TestCase):
    """Tests for Primitive 2."""

    def setUp(self):
        self.prim = DLLLoadPadder()

    def test_properties(self):
        self.assertEqual(self.prim.name, "dll_load_padding")
        self.assertEqual(self.prim.risk_level, RiskLevel.LOW_MEDIUM)
        self.assertEqual(self.prim.scope, "core")
        self.assertIn("dlllist.ndlls", self.prim.target_features)

    def test_predict_deltas(self):
        deltas = self.prim.predict_deltas(n_dlls=8)
        ndlls_delta = next(d for d in deltas if d.feature_name == "dlllist.ndlls")
        self.assertEqual(ndlls_delta.direction, +1)
        self.assertAlmostEqual(ndlls_delta.magnitude, 8.0)


class TestHandleTablePadder(unittest.TestCase):
    """Tests for Primitive 3."""

    def setUp(self):
        self.prim = HandleTablePadder()

    def test_properties(self):
        self.assertEqual(self.prim.name, "handle_table_padding")
        self.assertEqual(self.prim.risk_level, RiskLevel.LOW)
        self.assertEqual(self.prim.scope, "core")
        self.assertIn("handles.nhandles", self.prim.target_features)

    def test_predict_deltas_total(self):
        deltas = self.prim.predict_deltas(
            n_file_handles=20, n_reg_handles=10,
            n_event_handles=10, n_mutex_handles=5,
        )
        total_delta = next(d for d in deltas if d.feature_name == "handles.nhandles")
        self.assertEqual(total_delta.direction, +1)
        self.assertAlmostEqual(total_delta.magnitude, 45.0)  # 20+10+10+5


class TestMalfindDiluter(unittest.TestCase):
    """Tests for Primitive 4 (extended)."""

    def setUp(self):
        self.prim = MalfindDiluter()

    def test_properties(self):
        self.assertEqual(self.prim.name, "malfind_dilution")
        self.assertEqual(self.prim.risk_level, RiskLevel.HIGH)
        self.assertEqual(self.prim.scope, "extended")
        self.assertIn("malfind.ninjections", self.prim.target_features)

    def test_predict_deltas(self):
        deltas = self.prim.predict_deltas(n_regions=5, region_size=4096)
        inj_delta = next(d for d in deltas if d.feature_name == "malfind.ninjections")
        self.assertEqual(inj_delta.direction, +1)
        self.assertAlmostEqual(inj_delta.magnitude, 5.0)


class TestPerturbationSequence(unittest.TestCase):
    """Tests for the PerturbationSequence composer."""

    def test_empty_sequence(self):
        seq = PerturbationSequence()
        self.assertEqual(seq.length, 0)
        deltas = seq.predict_combined_deltas()
        self.assertEqual(len(deltas), 0)

    def test_single_action_sequence(self):
        prim = BenignChildSpawner()
        params = {"n_shortlived": 10, "n_longlived": 5}
        seq = PerturbationSequence(
            actions=[PerturbationAction(primitive=prim, params=params)]
        )
        self.assertEqual(seq.length, 1)
        deltas = seq.predict_combined_deltas()
        self.assertIn("pslist.nproc", deltas)

    def test_multi_action_additive(self):
        """Multiple actions should produce additive deltas."""
        p1 = BenignChildSpawner()
        p3 = HandleTablePadder()
        seq = PerturbationSequence(actions=[
            PerturbationAction(primitive=p1, params={"n_shortlived": 10, "n_longlived": 5}),
            PerturbationAction(primitive=p3, params={"n_file_handles": 20, "n_reg_handles": 10,
                                                     "n_event_handles": 10, "n_mutex_handles": 5}),
        ])
        deltas = seq.predict_combined_deltas()
        self.assertIn("pslist.nproc", deltas)
        self.assertIn("handles.nhandles", deltas)

    def test_to_dict(self):
        seq = PerturbationSequence(
            actions=[PerturbationAction(
                primitive=BenignChildSpawner(),
                params={"n_shortlived": 5, "n_longlived": 2},
            )]
        )
        d = seq.to_dict()
        self.assertEqual(d["length"], 1)
        self.assertIn("actions", d)


class TestEnsembleOracle(unittest.TestCase):
    """Tests for the ensemble oracle wrapper."""

    def test_query_count(self):
        def mock_predict(X):
            return np.array([[0.3, 0.7]] * X.shape[0])

        oracle = EnsembleOracle(mock_predict, ["f1", "f2"])
        self.assertEqual(oracle.query_count, 0)

        prob = oracle.query(np.array([1.0, 2.0]))
        self.assertEqual(oracle.query_count, 1)
        self.assertAlmostEqual(prob, 0.7)

        oracle.reset_count()
        self.assertEqual(oracle.query_count, 0)


class TestGAMMAOptimizer(unittest.TestCase):
    """Tests for the GAMMA-adapted optimizer."""

    def test_initialization(self):
        config = OptimizerConfig(population_size=5, generations=2, query_budget=20)
        optimizer = GAMMAOptimizer(config)
        self.assertEqual(len(optimizer.primitives), 3)

    def test_random_sequence_generation(self):
        config = OptimizerConfig(random_seed=42)
        optimizer = GAMMAOptimizer(config)
        seq = optimizer._random_sequence()
        self.assertGreater(seq.length, 0)
        self.assertLessEqual(seq.length, config.max_sequence_length)

    def test_optimize_basic(self):
        """Optimizer should run without errors on a synthetic model."""
        def mock_predict(X):
            # Simple model: high probability if features are large
            p_mal = np.clip(np.mean(X, axis=1) / 100, 0.1, 0.99)
            return np.column_stack([1 - p_mal, p_mal])

        config = OptimizerConfig(
            population_size=5,
            generations=2,
            query_budget=20,
            random_seed=42,
        )
        optimizer = GAMMAOptimizer(config)
        oracle = EnsembleOracle(mock_predict, VOLMEMLYZER_FEATURES[:55])

        x = np.random.default_rng(42).random(55) * 100
        result = optimizer.optimize(x, oracle)

        self.assertIsInstance(result, OptimizationResult)
        self.assertGreater(result.total_queries, 0)
        self.assertLessEqual(result.total_queries, config.query_budget + 5)  # small overhead
        self.assertIsInstance(result.best_sequence, PerturbationSequence)

    def test_crossover_produces_valid_sequence(self):
        config = OptimizerConfig(random_seed=42)
        optimizer = GAMMAOptimizer(config)
        p1 = optimizer._random_sequence()
        p2 = optimizer._random_sequence()
        child = optimizer._crossover(p1, p2)
        self.assertLessEqual(child.length, config.max_sequence_length)

    def test_mutate_preserves_bounds(self):
        config = OptimizerConfig(random_seed=42, mutation_probability=1.0)
        optimizer = GAMMAOptimizer(config)
        seq = optimizer._random_sequence()
        mutated = optimizer._mutate(seq)
        self.assertLessEqual(mutated.length, config.max_sequence_length)


if __name__ == "__main__":
    unittest.main(verbosity=2)
