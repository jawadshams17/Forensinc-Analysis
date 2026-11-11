"""
Main Entry Point — Adversarial Memory Forensics
=================================================
Provides CLI commands for each phase of the experiment.

Usage:
    python run.py phase0 --dry-run          # Phase 0 feasibility (synthetic)
    python run.py train --data data.csv     # Train ensemble
    python run.py analyze --rq 1            # Run RQ analysis
    python run.py full-experiment           # Full pipeline dry-run
    
    python run.py malvol25-scan             # Scan dataset directory
    python run.py malvol25-extract          # Extract features with Volatility 3
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import numpy as np
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
logger.add("logs/experiment_{time}.log", level="DEBUG", rotation="50 MB")


@click.group()
@click.option("--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose):
    """Adversarial Memory Forensics — Experiment Runner"""
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")


@cli.command()
@click.option("--dry-run", is_flag=True, default=True, help="Use synthetic data")
@click.option("--output-dir", default="results/phase0", help="Output directory")
def phase0(dry_run, output_dir):
    """Run Phase 0 feasibility gate evaluation."""
    from validation.phase0_feasibility import main as phase0_main
    sys.argv = ["phase0", "--dry-run", "--output-dir", output_dir]
    phase0_main()


@cli.command()
@click.option("--data", type=click.Path(exists=True), help="Training CSV path")
@click.option("--output", default="results/models/ensemble.joblib", help="Model output")
@click.option("--missing-strategy", default="native", type=click.Choice(["native", "impute_median", "impute_zero"]))
def train(data, output, missing_strategy):
    """Train the ensemble classifier."""
    from pipeline.ensemble_model import PrimaryDetector, LegacyEnsembleClassifier
    from pipeline.feature_extraction import FeatureExtractor

    extractor = FeatureExtractor()

    if data:
        X, y, feature_names = extractor.load_dataset(data)
    else:
        logger.info("No data provided — using synthetic dataset for testing")
        rng = np.random.default_rng(42)
        n, d = 1000, 55
        X = rng.random((n, d)) * 100
        y = rng.integers(0, 4, n)
        feature_names = extractor.feature_names

    clf = PrimaryDetector(handle_missing=missing_strategy)
    metrics = clf.fit(X, y, validation_split=0.1)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    clf.save(output)

    click.echo(f"\nTraining complete:")
    click.echo(f"  Accuracy: {metrics.get('val_accuracy', 'N/A')}")
    click.echo(f"  F1:       {metrics.get('val_f1_macro', 'N/A')}")
    click.echo(f"  Model:    {output}")


@cli.command()
@click.option("--rq", type=click.Choice(["1", "2", "3", "4", "5", "all"]), default="all")
@click.option("--model", type=click.Path(exists=True), help="Trained model path")
@click.option("--data", type=click.Path(exists=True), help="Test data CSV")
@click.option("--output-dir", default="results", help="Output directory")
def analyze(rq, model, data, output_dir):
    """Run RQ analysis pipelines."""
    from pipeline.ensemble_model import PrimaryDetector, LegacyEnsembleClassifier
    from pipeline.feature_extraction import FeatureExtractor

    # Load or create model/data
    if model:
        try:
            clf = PrimaryDetector.load(model)
        except Exception:
            logger.info("Failed loading PrimaryDetector, trying legacy load (deprecated).")
            clf = LegacyEnsembleClassifier.load(model)
    else:
        logger.info("No model provided — training synthetic model")
        rng = np.random.default_rng(42)
        X_train = rng.random((500, 55)) * 100
        y_train = rng.integers(0, 4, 500)
        clf = PrimaryDetector(handle_missing="impute_median")
        clf.fit(X_train, y_train, validation_split=0.0)

    extractor = FeatureExtractor()

    if data:
        X, y, _ = extractor.load_dataset(data)
    else:
        rng = np.random.default_rng(99)
        X = rng.random((200, 55)) * 100
        y = rng.integers(0, 4, 200)

    rqs = ["1", "2", "3", "4", "5"] if rq == "all" else [rq]

    for rq_num in rqs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running RQ{rq_num} Analysis")
        logger.info(f"{'='*60}")

        if rq_num == "1":
            from analysis.rq1_truncation_curves import TruncationAnalyzer
            analyzer = TruncationAnalyzer(clf, extractor)
            results = analyzer.run_analysis(X, y)
            analyzer.save_results(results, Path(output_dir) / "rq1")

        elif rq_num == "2":
            from analysis.rq2_evasion_interaction import EvasionInteractionAnalyzer
            from perturbation_primitives import apply_feature_deltas, VOLMEMLYZER_FEATURES
            from perturbation_primitives.child_spawn import BenignChildSpawner
            from perturbation_primitives import PerturbationAction, PerturbationSequence

            # Create adversarial features
            prim = BenignChildSpawner()
            seq = PerturbationSequence(actions=[
                PerturbationAction(primitive=prim, params={"n_shortlived": 15, "n_longlived": 5})
            ])
            deltas = seq.predict_combined_deltas()
            X_adv = np.array([apply_feature_deltas(x, VOLMEMLYZER_FEATURES, deltas) for x in X])

            analyzer = EvasionInteractionAnalyzer(clf, extractor)
            results = analyzer.compute_evasion_rates(X, X_adv, y)
            analyzer.save_results(results, Path(output_dir) / "rq2")

        elif rq_num == "3":
            from analysis.rq3_shap_drift import SHAPDriftAnalyzer
            from pipeline.shap_analysis import SHAPAnalyzer
            shap_analyzer = SHAPAnalyzer()
            analyzer = SHAPDriftAnalyzer(clf, extractor, shap_analyzer)
            try:
                results = analyzer.run_analysis(X, max_samples=50)
                analyzer.save_results(results, Path(output_dir) / "rq3")
            except ImportError:
                logger.warning("SHAP not available — skipping RQ3")

        elif rq_num == "4":
            from analysis.rq4_time_to_detection import TimeToDetectionAnalyzer
            analyzer = TimeToDetectionAnalyzer(clf, extractor)
            results = analyzer.run_analysis(X[:50], y[:50])
            analyzer.save_results(results, Path(output_dir) / "rq4")

        elif rq_num == "5":
            from analysis.rq5_cross_architecture import CrossArchitectureAnalyzer
            estimators = clf.get_individual_estimators()
            architectures = {
                "ensemble": clf.predict_proba,
                "rf_only": lambda X, m=estimators["rf"]: m.predict_proba(np.nan_to_num(X, nan=0.0)),
            }
            analyzer = CrossArchitectureAnalyzer(architectures)
            X_adv = X + np.random.default_rng(42).normal(0, 5, X.shape)
            X_adv = np.clip(X_adv, 0, None)
            results = analyzer.evaluate_transfer(X, X_adv, y, "ensemble")
            analyzer.save_results(results, Path(output_dir) / "rq5")

    click.echo(f"\nAll requested analyses complete. Results in: {output_dir}/")


@cli.command("full-experiment")
@click.option("--output-dir", default="results", help="Output directory")
def full_experiment(output_dir):
    """Run complete experiment pipeline (dry-run with synthetic data)."""
    logger.info("FULL EXPERIMENT DRY RUN")
    logger.info("=" * 60)

    from pipeline.ensemble_model import EnsembleClassifier
    from pipeline.feature_extraction import FeatureExtractor
    from analysis.ablation_runner import AblationRunner

    rng = np.random.default_rng(42)

    # 1. Generate synthetic data
    logger.info("Step 1: Generating synthetic dataset...")
    X = rng.random((300, 55)) * 100
    y = rng.integers(0, 4, 300)

    # 2. Train ensemble
    logger.info("Step 2: Training ensemble...")
    clf = EnsembleClassifier(
        rf_params={"n_estimators": 20, "random_state": 42, "n_jobs": -1},
        handle_missing="impute_median",
    )
    metrics = clf.fit(X, y, validation_split=0.1)
    logger.info(f"  Training accuracy: {metrics.get('val_accuracy', 'N/A'):.4f}")

    # 3. Ablation study
    logger.info("Step 3: Running ablation study...")
    extractor = FeatureExtractor()
    ablation = AblationRunner(clf, extractor, truncation_levels=[1.0, 0.5, 0.25])
    ablation_results = ablation.run_ablation(X, y)
    ablation.save_results(ablation_results, Path(output_dir) / "ablation")

    # 4. Summary
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT DRY RUN COMPLETE")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("=" * 60)

    click.echo("\n✅ Full experiment dry-run complete.")


@cli.command("test")
def run_tests():
    """Run the test suite."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(Path(__file__).parent),
    )
    sys.exit(result.returncode)


@cli.command("malvol25-scan")
@click.option("--dataset-dir", default="data/malvol25/raw", help="Path to raw .mem files")
@click.option("--output-dir", default="data/malvol25/extracted", help="Output directory")
def malvol25_scan(dataset_dir, output_dir):
    """Scan and catalogue the MalVol-25 dataset."""
    from pipeline.malvol25_loader import MalVol25Loader
    loader = MalVol25Loader(dataset_dir, output_dir)
    loader.scan_dataset()
    loader.print_summary()


@cli.command("malvol25-extract")
@click.option("--dataset-dir", default="data/malvol25/raw", help="Path to raw .mem files")
@click.option("--output-dir", default="data/malvol25/extracted", help="Output directory")
@click.option("--max-samples", type=int, default=None, help="Max samples to process")
@click.option("--timeout", type=int, default=600, help="Timeout per Volatility plugin")
def malvol25_extract(dataset_dir, output_dir, max_samples, timeout):
    """Extract VolMemLyzer features from MalVol-25 memory dumps."""
    from pipeline.malvol25_loader import MalVol25Loader
    
    # Needs volatility3
    try:
        import volatility3
    except ImportError:
        logger.error("volatility3 not installed. Please run: pip install volatility3")
        sys.exit(1)
        
    loader = MalVol25Loader(dataset_dir, output_dir)
    # Check if catalogue exists
    if not (Path(output_dir) / "catalogue.json").exists():
        logger.info("Catalogue not found. Scanning dataset first...")
        loader.scan_dataset()
    else:
        loader.load_catalogue()
        
    loader.extract_all_features(max_samples=max_samples, timeout_per_dump=timeout)
    click.echo("\nFeature extraction complete. You can now use the CSVs with 'run.py train'.")


@cli.command("malvol25-pipeline")
@click.option("--zip-path", default=r"f:\FORENSICS\Dataset All Files_0 (1).zip", help="Path to the MalVol-25 ZIP file")
@click.option("--output-dir", default="data/malvol25/extracted", help="Output directory for CSV")
def malvol25_pipeline(zip_path, output_dir):
    """Stream-extract + Volatility feature extraction (disk-efficient, one dump at a time)."""
    try:
        import volatility3
    except ImportError:
        logger.error("volatility3 not installed. Please run: pip install volatility3")
        sys.exit(1)
    from pipeline.extract_pipeline import main as pipeline_main
    import pipeline.extract_pipeline as ep
    ep.ZIP_SRC = Path(zip_path)
    ep.OUT_DIR = Path(output_dir)
    ep.DUMP_DIR = Path(output_dir).parent / "tmp"
    ep.FEATURES_CSV = ep.OUT_DIR / "MalVol25_features.csv"
    pipeline_main()


if __name__ == "__main__":
    cli()
