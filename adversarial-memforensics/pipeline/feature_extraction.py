"""
Feature Extraction — VolMemLyzer Wrapper
=========================================
Wraps the VolMemLyzer feature extraction pipeline to produce the
55-feature vectors from raw memory dumps (.lime, .raw, .dmp).

Handles truncated captures by:
    1. Attempting full extraction on the (partial) dump
    2. Encoding failed features as NaN (not imputed)
    3. Logging which features were successfully computed

This preserves the ensemble's native missing-value handling rather
than masking truncation artifacts.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from perturbation_primitives import VOLMEMLYZER_FEATURES, FEATURE_GROUPS


class FeatureExtractor:
    """
    Extracts VolMemLyzer features from raw memory dumps.

    Supports:
      - Direct extraction via Volatility 3 + VolMemLyzer
      - CSV-based feature loading (from pre-extracted CIC-MalMem-2022)
      - Simulated truncation (masking features based on truncation %)
    """

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        volatility_path: Optional[str] = None,
        volmemlyzer_path: Optional[str] = None,
    ):
        self.feature_names = feature_names or VOLMEMLYZER_FEATURES
        self.volatility_path = volatility_path or "vol3"
        self.volmemlyzer_path = volmemlyzer_path

    # ── CSV-based loading (CIC-MalMem-2022 format) ─────────────────

    def load_dataset(
        self,
        csv_path: str | Path,
        label_column: str = "Category",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load a CSV feature file (CIC-MalMem-2022 or MalVol-25 format).

        Automatically strips non-numeric metadata columns (Category, Label,
        Family, OS_Version, File, OS, Class) so they are never cast to float.

        Returns:
            (X, y, feature_names):
                X: (n_samples, 55) feature matrix
                y: (n_samples,) integer labels
                feature_names: list of feature column names
        """
        logger.info(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)

        # All known metadata / label columns to exclude from the feature matrix
        METADATA_COLS = {
            "Category", "Label", "Family", "OS_Version", "File",
            "OS", "Class", label_column,
        }

        # Use numeric 'Label' column when present (MalVol-25 pipeline writes it)
        if "Label" in df.columns:
            y = df["Label"].values.astype(int)
        elif label_column in df.columns:
            y_raw = df[label_column].values
            label_map = {"Benign": 0, "Ransomware": 1, "Spyware": 2, "Trojan": 3}
            if len(y_raw) > 0 and isinstance(y_raw[0], str):
                y = np.array([label_map.get(lbl, 3) for lbl in y_raw])
            else:
                y = y_raw.astype(int)
        else:
            y = df.iloc[:, -1].values.astype(int)

        # Build purely numeric feature matrix — exclude all metadata columns
        feature_cols = [
            c for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
        X = df[feature_cols].values.astype(np.float64)

        logger.info(
            f"Loaded {X.shape[0]} samples × {X.shape[1]} features, "
            f"{len(np.unique(y))} classes"
        )
        logger.info(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

        return X, y, feature_cols

    # ── Truncation Simulation ──────────────────────────────────────

    def simulate_truncation(
        self,
        X: np.ndarray,
        truncation_pct: float,
        feature_names: Optional[List[str]] = None,
        random_seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate capture truncation by masking features with NaN.

        The masking order is based on feature group acquisition order:
        features from later-acquired groups (netscan, callbacks, svcscan)
        are masked first, simulating the sequential nature of memory
        traversal and the fact that truncated captures lose later-traversed
        data structures first.

        Args:
            X:               (n_samples, n_features) feature matrix
            truncation_pct:  Percentage of capture completed (0.0 to 1.0)
                             1.0 = complete capture, 0.1 = 10% capture
            feature_names:   Feature names for group identification
            random_seed:     Seed for reproducible masking

        Returns:
            X_truncated: feature matrix with NaN for masked features
        """
        if truncation_pct >= 1.0:
            return X.copy()

        feature_names = feature_names or self.feature_names
        X_trunc = X.copy().astype(np.float64)

        # Feature groups ordered by TYPICAL acquisition sequence
        # (earlier groups are acquired first and survive truncation longer)
        acquisition_order = [
            "pslist",      # First pass: process listing
            "dlllist",     # Second pass: DLL enumeration
            "handles",     # Third pass: handle tables
            "ldrmodules",  # Fourth pass: loader modules
            "malfind",     # Fifth pass: injection scanning
            "modules",     # Sixth pass: kernel modules
            "svcscan",     # Seventh pass: service scanning
            "callbacks",   # Eighth pass: callback tables
            "netscan",     # Ninth pass: network connections
        ]

        # Determine which feature groups are fully/partially available
        n_groups = len(acquisition_order)
        groups_available = int(np.ceil(truncation_pct * n_groups))

        # Groups to mask (those that couldn't complete before truncation)
        groups_to_mask = acquisition_order[groups_available:]

        # Additionally, within partially-acquired groups, mask a fraction
        partial_group_idx = groups_available - 1
        if 0 <= partial_group_idx < n_groups:
            partial_group = acquisition_order[partial_group_idx]
            partial_features = FEATURE_GROUPS.get(partial_group, [])
            if partial_features:
                # How much of this group was completed?
                frac_in_group = (truncation_pct * n_groups) - partial_group_idx
                n_available = int(np.ceil(frac_in_group * len(partial_features)))
                features_to_mask_partial = partial_features[n_available:]
                name_to_idx = {name: i for i, name in enumerate(feature_names)}
                for feat in features_to_mask_partial:
                    if feat in name_to_idx:
                        X_trunc[:, name_to_idx[feat]] = np.nan

        # Mask all features from fully-unavailable groups
        name_to_idx = {name: i for i, name in enumerate(feature_names)}
        n_masked = 0
        for group in groups_to_mask:
            group_features = FEATURE_GROUPS.get(group, [])
            for feat in group_features:
                if feat in name_to_idx:
                    X_trunc[:, name_to_idx[feat]] = np.nan
                    n_masked += 1

        total_nans = np.isnan(X_trunc).sum()
        logger.info(
            f"Truncation {truncation_pct:.0%}: masked {n_masked} features "
            f"from groups {groups_to_mask}, total NaN entries: {total_nans}"
        )

        return X_trunc

    # ── Raw Dump Extraction ────────────────────────────────────────

    def extract_from_dump(
        self,
        dump_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Optional[np.ndarray]:
        """
        Extract features from a raw memory dump using Volatility 3.

        This is the REAL extraction path (vs. CSV loading for CIC-MalMem-2022).
        Requires Volatility 3 and VolMemLyzer to be installed.

        Args:
            dump_path:   Path to .lime, .raw, or .dmp memory dump
            output_path: Optional path to save extracted features CSV

        Returns:
            55-dim feature vector, or None if extraction fails
        """
        dump_path = Path(dump_path)
        if not dump_path.exists():
            logger.error(f"Dump file not found: {dump_path}")
            return None

        logger.info(f"Extracting features from {dump_path} ({dump_path.stat().st_size / 1e6:.1f} MB)")

        try:
            # Run VolMemLyzer (assumed to be available as a script or module)
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "python", "-m", "volmemlyzer",
                "--input", str(dump_path),
                "--output", tmp_path,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"VolMemLyzer failed: {result.stderr}")
                # Fall back to partial extraction
                return self._partial_extraction(dump_path)

            # Read extracted features
            df = pd.read_csv(tmp_path)
            features = df.iloc[0].values.astype(np.float64)

            if output_path:
                df.to_csv(output_path, index=False)

            os.unlink(tmp_path)
            return features

        except subprocess.TimeoutExpired:
            logger.error(f"Feature extraction timed out for {dump_path}")
            return None
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def _partial_extraction(self, dump_path: Path) -> np.ndarray:
        """
        Attempt partial feature extraction from a truncated dump.
        Features that cannot be computed are set to NaN.
        """
        features = np.full(len(self.feature_names), np.nan)
        logger.warning(
            f"Partial extraction: returning {len(self.feature_names)} NaN features "
            f"(manual extraction may be needed)"
        )
        return features

    # ── Batch Processing ───────────────────────────────────────────

    def extract_batch(
        self,
        dump_dir: str | Path,
        output_csv: str | Path,
        pattern: str = "*.lime",
    ) -> pd.DataFrame:
        """
        Extract features from all dumps in a directory.

        Args:
            dump_dir:   Directory containing memory dump files
            output_csv: Path to save the combined feature CSV
            pattern:    Glob pattern for dump files

        Returns:
            DataFrame with features and metadata
        """
        dump_dir = Path(dump_dir)
        dumps = sorted(dump_dir.glob(pattern))
        logger.info(f"Processing {len(dumps)} dumps from {dump_dir}")

        rows = []
        for i, dump_path in enumerate(dumps):
            logger.info(f"[{i+1}/{len(dumps)}] {dump_path.name}")
            features = self.extract_from_dump(dump_path)
            if features is not None:
                row = {"filename": dump_path.name}
                for j, name in enumerate(self.feature_names):
                    row[name] = features[j] if j < len(features) else np.nan
                rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        logger.info(f"Batch extraction complete: {len(rows)} samples → {output_csv}")
        return df

    # ── Truncation Statistics ──────────────────────────────────────

    def compute_truncation_stats(
        self,
        X: np.ndarray,
        truncation_levels: List[float],
    ) -> pd.DataFrame:
        """
        Compute statistics about feature availability at each truncation level.

        Returns DataFrame with columns:
            truncation_pct, n_available, n_missing, pct_available, missing_groups
        """
        stats = []
        for pct in truncation_levels:
            X_trunc = self.simulate_truncation(X[:1], pct)  # single sample
            n_available = np.sum(~np.isnan(X_trunc[0]))
            n_missing = np.sum(np.isnan(X_trunc[0]))
            n_total = len(self.feature_names)

            # Identify which groups have missing features
            missing_groups = []
            name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
            for group, features in FEATURE_GROUPS.items():
                n_group_missing = sum(
                    1 for f in features
                    if f in name_to_idx and np.isnan(X_trunc[0, name_to_idx[f]])
                )
                if n_group_missing > 0:
                    missing_groups.append(f"{group}({n_group_missing}/{len(features)})")

            stats.append({
                "truncation_pct": pct,
                "n_available": n_available,
                "n_missing": n_missing,
                "pct_available": n_available / n_total,
                "missing_groups": ", ".join(missing_groups) if missing_groups else "none",
            })

        return pd.DataFrame(stats)
