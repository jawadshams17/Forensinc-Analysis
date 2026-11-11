"""
Integration Test — Pipeline End-to-End
========================================
Tests the complete pipeline from feature extraction through ensemble
inference to SHAP analysis, using synthetic data.
"""

from __future__ import annotations

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.ensemble_model import EnsembleClassifier
from pipeline.feature_extraction import FeatureExtractor
from pipeline.missing_value_handler import MissingValueHandler
from perturbation_primitives import VOLMEMLYZER_FEATURES


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end pipeline tests with synthetic data."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic dataset once for all tests."""
        rng = np.random.default_rng(42)
        n = 500
        n_features = len(VOLMEMLYZER_FEATURES)

        # Generate class-separable synthetic data
        cls.X = np.zeros((n, n_features))
        cls.y = np.zeros(n, dtype=int)

        for i in range(n):
            cls.y[i] = i % 4  # 4 classes
            # Each class gets slightly different feature distributions
            cls.X[i] = rng.normal(loc=50 + cls.y[i] * 10, scale=15, size=n_features)
            cls.X[i] = np.clip(cls.X[i], 0, None)  # non-negative

    def test_ensemble_fit_predict(self):
        """Ensemble should train and predict without errors."""
        clf = EnsembleClassifier(
            rf_params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
            xgb_params=None,
            lgb_params=None,
        )
        metrics = clf.fit(self.X, self.y, validation_split=0.2)
        self.assertIn("val_accuracy", metrics)
        self.assertGreater(metrics["val_accuracy"], 0.0)

        # Predict
        y_pred = clf.predict(self.X[:10])
        self.assertEqual(len(y_pred), 10)

        # Predict proba
        y_proba = clf.predict_proba(self.X[:10])
        self.assertEqual(y_proba.shape, (10, 4))
        np.testing.assert_allclose(y_proba.sum(axis=1), 1.0, atol=1e-6)

    def test_ensemble_evaluate(self):
        """Full evaluation should produce all expected metrics."""
        clf = EnsembleClassifier(
            rf_params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
        )
        clf.fit(self.X, self.y, validation_split=0.0)
        metrics = clf.evaluate(self.X, self.y, label="test")

        required_keys = ["accuracy", "f1_macro", "fpr", "fnr", "confusion_matrix"]
        for key in required_keys:
            self.assertIn(key, metrics)

    def test_ensemble_save_load(self):
        """Model persistence round-trip."""
        clf = EnsembleClassifier(
            rf_params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
        )
        clf.fit(self.X, self.y, validation_split=0.0)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            clf.save(f.name)
            clf2 = EnsembleClassifier.load(f.name)

        y1 = clf.predict(self.X[:5])
        y2 = clf2.predict(self.X[:5])
        np.testing.assert_array_equal(y1, y2)
        Path(f.name).unlink()

    def test_truncation_simulation(self):
        """Truncation should produce NaN for later-acquired groups."""
        extractor = FeatureExtractor()

        # 50% truncation should preserve early groups, mask late ones
        X_50 = extractor.simulate_truncation(self.X[:1], 0.50)
        n_nan = np.isnan(X_50[0]).sum()
        self.assertGreater(n_nan, 0, "50% truncation should mask some features")

        # 100% should preserve all
        X_100 = extractor.simulate_truncation(self.X[:1], 1.0)
        self.assertEqual(np.isnan(X_100[0]).sum(), 0)

        # 10% should mask most features
        X_10 = extractor.simulate_truncation(self.X[:1], 0.10)
        n_nan_10 = np.isnan(X_10[0]).sum()
        self.assertGreater(n_nan_10, n_nan)

    def test_truncation_stats(self):
        """Truncation stats should report feature availability."""
        extractor = FeatureExtractor()
        stats = extractor.compute_truncation_stats(self.X, [1.0, 0.5, 0.1])
        self.assertEqual(len(stats), 3)
        self.assertIn("n_available", stats.columns)
        self.assertIn("n_missing", stats.columns)

    def test_missing_value_handler_native(self):
        """Native handler should leave NaN for XGB/LGB, impute for RF."""
        handler = MissingValueHandler(strategy="native")
        handler.fit(self.X)

        # Add NaN
        X_nan = self.X[:5].copy()
        X_nan[0, 0] = np.nan
        X_nan[1, 3] = np.nan

        # For RF: should impute
        X_rf = handler.transform(X_nan, target_model="rf")
        self.assertFalse(np.isnan(X_rf[0, 0]))

        # For XGB: should keep NaN
        X_xgb = handler.transform(X_nan, target_model="xgb")
        self.assertTrue(np.isnan(X_xgb[0, 0]))

    def test_missing_value_handler_impute(self):
        """Median imputation should remove all NaN."""
        handler = MissingValueHandler(strategy="impute_median")
        handler.fit(self.X)

        X_nan = self.X[:5].copy()
        X_nan[:, 0] = np.nan
        X_out = handler.transform(X_nan)
        self.assertFalse(np.any(np.isnan(X_out)))

    def test_missing_pattern_report(self):
        """Report should identify which features are missing."""
        handler = MissingValueHandler()
        X_nan = self.X[:10].copy()
        X_nan[:, 0] = np.nan  # pslist.nproc
        X_nan[:5, 1] = np.nan  # pslist.nppid

        report = handler.missing_pattern_report(X_nan)
        self.assertGreater(len(report), 0)

        row_nproc = report[report["feature"] == "pslist.nproc"]
        self.assertTrue(row_nproc.iloc[0]["is_fully_missing"])

    def test_pipeline_with_truncation(self):
        """Full pipeline: train on complete → predict on truncated."""
        clf = EnsembleClassifier(
            rf_params={"n_estimators": 10, "random_state": 42, "n_jobs": 1},
            handle_missing="impute_median",
        )
        clf.fit(self.X, self.y, validation_split=0.0)

        extractor = FeatureExtractor()
        X_trunc = extractor.simulate_truncation(self.X, 0.50)

        # Should predict without errors
        y_pred = clf.predict(X_trunc)
        self.assertEqual(len(y_pred), len(self.y))


class TestStatisticalTests(unittest.TestCase):
    """Tests for the statistical utilities."""

    def test_bootstrap_metric(self):
        from analysis.statistical_tests import bootstrap_metric
        from sklearn.metrics import accuracy_score

        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10)
        y_pred = np.array([0, 1, 0, 0, 0, 1, 1, 1] * 10)

        lo, hi = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstrap=100)
        self.assertLess(lo, hi)
        self.assertGreater(lo, 0)
        self.assertLess(hi, 1.1)

    def test_cohens_d(self):
        from analysis.statistical_tests import cohens_d
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([6, 7, 8, 9, 10])
        d = cohens_d(a, b)
        self.assertLess(d, 0)  # a < b → negative d

    def test_cliffs_delta(self):
        from analysis.statistical_tests import cliffs_delta
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        delta, mag = cliffs_delta(a, b)
        self.assertLess(delta, 0)
        self.assertIn(mag, ["negligible", "small", "medium", "large"])

    def test_bonferroni(self):
        from analysis.statistical_tests import bonferroni_correction
        results = bonferroni_correction([0.01, 0.04, 0.06], alpha=0.05)
        self.assertEqual(len(results), 3)
        # 0.01 < 0.05/3 ≈ 0.0167 → significant
        self.assertTrue(results[0]["significant_after_correction"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
