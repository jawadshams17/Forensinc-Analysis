"""
MalVol-25 Dataset Loader
==========================
Handles loading, cataloguing, and processing the MalVol-25 dataset
(IEEE DataPort DOI: 10.21227/kg5b-nf37).

MalVol-25 provides raw .mem/.raw volatile memory dumps across:
    - Multiple Windows OS versions (Win7, Win10, Win11)
    - Clean (benign) baseline snapshots
    - Infected snapshots with diverse malware families
    - Ransomware, Trojans, Spyware, Worms, Rootkits

This module:
    1. Scans the dataset directory to catalogue all dumps
    2. Parses file naming conventions to extract metadata
    3. Runs Volatility 3 plugins to extract VolMemLyzer-equivalent features
    4. Produces a unified CSV compatible with our ensemble pipeline
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

# Our project's feature definitions
from perturbation_primitives import VOLMEMLYZER_FEATURES, FEATURE_GROUPS


# ── MalVol-25 Naming Convention ───────────────────────────────────
# Files follow patterns like:
#   Win10_Clean_memory.mem
#   Win10_Ransomware_WannaCry_memory.raw
#   Win7_Trojan_Emotet_memory.mem
#   Win11_Spyware_AgentTesla_memory.raw

MALVOL_CATEGORIES = {
    "Clean": "Benign",
    "Ransomware": "Ransomware",
    "Trojan": "Trojan",
    "Spyware": "Spyware",
    "Worm": "Worm",
    "Rootkit": "Rootkit",
    "Adware": "Adware",
    "Backdoor": "Backdoor",
    "Miner": "Miner",
}

# Map MalVol categories to our 4-class scheme (Paper 1)
CATEGORY_MAP_4CLASS = {
    "Benign": 0,
    "Ransomware": 1,
    "Spyware": 2,
    "Trojan": 3,
    # Extended categories mapped to closest match
    "Worm": 3,       # Trojan-like
    "Rootkit": 3,     # Trojan-like
    "Adware": 2,      # Spyware-like
    "Backdoor": 3,    # Trojan-like
    "Miner": 2,       # Spyware-like
}

CLASS_NAMES = ["Benign", "Ransomware", "Spyware", "Trojan"]


@dataclass
class MemoryDumpEntry:
    """Metadata for a single memory dump file."""
    file_path: Path
    file_name: str
    file_size_bytes: int = 0
    os_version: str = ""             # Win7, Win10, Win11
    infection_status: str = ""       # Clean, Infected
    category: str = ""               # Benign, Ransomware, etc.
    malware_family: str = ""         # WannaCry, Emotet, etc.
    label_4class: int = 0            # 0=Benign, 1=Ransom, 2=Spy, 3=Trojan
    file_format: str = ""            # .mem, .raw
    features_extracted: bool = False
    feature_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()
             if k != "feature_vector"}
        d["file_path"] = str(d["file_path"])
        return d


class MalVol25Loader:
    """
    Scans, catalogues, and processes MalVol-25 memory dumps.
    """

    # Volatility 3 plugins used for VolMemLyzer-equivalent extraction
    VOL3_PLUGINS = [
        "windows.pslist.PsList",
        "windows.dlllist.DllList",
        "windows.handles.Handles",
        "windows.ldrmodules.LdrModules",
        "windows.malfind.Malfind",
        "windows.modules.Modules",
        "windows.svcscan.SvcScan",
        "windows.callbacks.Callbacks",
        "windows.netscan.NetScan",
    ]

    def __init__(
        self,
        dataset_dir: str | Path,
        output_dir: str | Path = "data/malvol25",
        volatility_path: str = "vol3",
    ):
        self.dataset_dir = Path(dataset_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.volatility_path = volatility_path

        self.entries: List[MemoryDumpEntry] = []
        self.catalogue_path = self.output_dir / "catalogue.json"

    # ── Scanning & Cataloguing ─────────────────────────────────────

    def scan_dataset(self) -> List[MemoryDumpEntry]:
        """
        Scan the dataset directory for all memory dump files and
        parse metadata from file names and directory structure.
        """
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return []

        extensions = {".mem", ".raw", ".vmem", ".dmp"}
        dump_files = []

        for ext in extensions:
            dump_files.extend(self.dataset_dir.rglob(f"*{ext}"))

        logger.info(f"Found {len(dump_files)} memory dump files in {self.dataset_dir}")

        self.entries = []
        for fp in sorted(dump_files):
            entry = self._parse_dump_metadata(fp)
            self.entries.append(entry)

        # Save catalogue
        self._save_catalogue()

        # Summary
        categories = {}
        for e in self.entries:
            categories[e.category] = categories.get(e.category, 0) + 1

        logger.info(f"Dataset catalogue:")
        for cat, count in sorted(categories.items()):
            logger.info(f"  {cat}: {count} dumps")
        logger.info(f"  Total: {len(self.entries)} dumps")

        return self.entries

    def _parse_dump_metadata(self, file_path: Path) -> MemoryDumpEntry:
        """
        Parse metadata from file path and naming convention.
        Handles multiple naming patterns from MalVol-25.
        """
        name = file_path.stem.lower()
        parts = file_path.parts

        entry = MemoryDumpEntry(
            file_path=file_path,
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size if file_path.exists() else 0,
            file_format=file_path.suffix.lower(),
        )

        # Try to extract OS version from path parts or filename
        for part in list(parts) + [name]:
            part_str = str(part).lower()
            if "win11" in part_str or "windows11" in part_str or "win_11" in part_str:
                entry.os_version = "Win11"
                break
            elif "win10" in part_str or "windows10" in part_str or "win_10" in part_str:
                entry.os_version = "Win10"
                break
            elif "win7" in part_str or "windows7" in part_str or "win_7" in part_str:
                entry.os_version = "Win7"
                break
            elif "win8" in part_str or "windows8" in part_str:
                entry.os_version = "Win8"
                break

        if not entry.os_version:
            entry.os_version = "Unknown"

        # Extract category and infection status
        full_path_str = str(file_path).lower()

        if "clean" in full_path_str or "benign" in full_path_str or "baseline" in full_path_str:
            entry.infection_status = "Clean"
            entry.category = "Benign"
            entry.malware_family = "none"
        else:
            entry.infection_status = "Infected"
            # Try to identify category
            for keyword, category in MALVOL_CATEGORIES.items():
                if keyword.lower() in full_path_str:
                    entry.category = category
                    break
            if not entry.category:
                entry.category = "Unknown"

            # Try to extract malware family name
            # Look in directory names and filename
            family_candidates = []
            for part in parts:
                part_str = str(part)
                # Skip common directory names
                if part_str.lower() not in {"infected", "malware", "memory", "dumps",
                                            "dataset", "data", "raw", "samples"}:
                    if not any(os_name in part_str.lower()
                               for os_name in ["win7", "win10", "win11", "windows"]):
                        family_candidates.append(part_str)

            if family_candidates:
                entry.malware_family = family_candidates[-1]  # deepest directory
            else:
                entry.malware_family = "unknown"

        # Map to 4-class label
        entry.label_4class = CATEGORY_MAP_4CLASS.get(entry.category, 3)

        return entry

    def _save_catalogue(self) -> None:
        """Save the catalogue to JSON."""
        catalogue = {
            "dataset": "MalVol-25",
            "source": "IEEE DataPort (DOI: 10.21227/kg5b-nf37)",
            "n_entries": len(self.entries),
            "entries": [e.to_dict() for e in self.entries],
        }
        with open(self.catalogue_path, "w") as f:
            json.dump(catalogue, f, indent=2)
        logger.info(f"Catalogue saved to {self.catalogue_path}")

    def load_catalogue(self) -> List[MemoryDumpEntry]:
        """Load a previously saved catalogue."""
        if not self.catalogue_path.exists():
            logger.warning("No catalogue found — run scan_dataset() first")
            return []

        with open(self.catalogue_path) as f:
            data = json.load(f)

        self.entries = []
        for e in data["entries"]:
            entry = MemoryDumpEntry(
                file_path=Path(e["file_path"]),
                file_name=e["file_name"],
                file_size_bytes=e.get("file_size_bytes", 0),
                os_version=e.get("os_version", ""),
                infection_status=e.get("infection_status", ""),
                category=e.get("category", ""),
                malware_family=e.get("malware_family", ""),
                label_4class=e.get("label_4class", 0),
                file_format=e.get("file_format", ""),
                features_extracted=e.get("features_extracted", False),
            )
            self.entries.append(entry)

        logger.info(f"Loaded catalogue with {len(self.entries)} entries")
        return self.entries

    # ── Feature Extraction via Volatility 3 ────────────────────────

    def extract_features_single(
        self,
        entry: MemoryDumpEntry,
        timeout: int = 600,
    ) -> Optional[np.ndarray]:
        """
        Extract VolMemLyzer-equivalent features from a single memory dump
        using Volatility 3 plugins.

        Returns: (55,) feature vector or None on failure
        """
        dump_path = entry.file_path
        if not dump_path.exists():
            logger.error(f"Dump not found: {dump_path}")
            return None

        logger.info(f"Extracting features from {dump_path.name}...")

        plugin_results: Dict[str, Any] = {}

        for plugin in self.VOL3_PLUGINS:
            plugin_name = plugin.split(".")[-1]
            try:
                result = self._run_volatility_plugin(
                    dump_path, plugin, timeout=timeout
                )
                plugin_results[plugin_name] = result
                logger.debug(f"  {plugin_name}: {len(result)} rows")
            except Exception as e:
                logger.warning(f"  {plugin_name}: FAILED ({e})")
                plugin_results[plugin_name] = []

        # Convert plugin outputs to 55-feature vector
        features = self._plugins_to_features(plugin_results)
        entry.feature_vector = features.tolist()
        entry.features_extracted = True

        return features

    def _run_volatility_plugin(
        self,
        dump_path: Path,
        plugin: str,
        timeout: int = 600,
    ) -> List[Dict[str, Any]]:
        cmd = [
            sys.executable, str(Path(__file__).parent.parent / "vol_wrapper.py"),
            "-f", str(dump_path),
            "-r", "json",
            plugin,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                return data if isinstance(data, list) else [data]

        except subprocess.TimeoutExpired:
            logger.warning(f"Plugin {plugin} timed out after {timeout}s")
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {plugin}")
        except FileNotFoundError:
            logger.warning(f"Volatility 3 not found — install with: pip install volatility3")

        return []

    def _plugins_to_features(
        self,
        plugin_results: Dict[str, List[Dict]],
    ) -> np.ndarray:
        """
        Convert Volatility 3 plugin outputs into the 55 VolMemLyzer
        feature vector matching our pipeline's expected format.
        """
        features = np.zeros(len(VOLMEMLYZER_FEATURES))
        name_to_idx = {name: i for i, name in enumerate(VOLMEMLYZER_FEATURES)}

        def sget(d, k, default=0):
            val = d.get(k, default)
            return default if val is None else val

        # ── pslist features ────────────────────────────────────────
        pslist = plugin_results.get("PsList", [])
        if pslist:
            n_procs = len(pslist)
            pids = [sget(p, "PID", 0) for p in pslist if isinstance(p, dict)]
            ppids = [sget(p, "PPID", 0) for p in pslist if isinstance(p, dict)]
            threads = [sget(p, "Threads", 0) for p in pslist if isinstance(p, dict)]
            handles = [sget(p, "Handles", 0) for p in pslist if isinstance(p, dict)]

            self._set_feature(features, name_to_idx, "pslist.nproc", n_procs)
            self._set_feature(features, name_to_idx, "pslist.nppid",
                              len(set(ppids)))
            self._set_feature(features, name_to_idx, "pslist.avg_threads",
                              np.mean(threads) if threads else 0)
            self._set_feature(features, name_to_idx, "pslist.nprocs64bit",
                              sum(1 for p in pslist if isinstance(p, dict) and sget(p, "Wow64", False) is False))
            self._set_feature(features, name_to_idx, "pslist.avg_handlers",
                              np.mean(handles) if handles else 0)

        # ── dlllist features ───────────────────────────────────────
        dlllist = plugin_results.get("DllList", [])
        if dlllist:
            n_dlls = len(dlllist)
            dll_sizes = [sget(d, "Size", 0) for d in dlllist if isinstance(d, dict)]
            dll_paths = [str(sget(d, "Path", "")) for d in dlllist if isinstance(d, dict)]

            self._set_feature(features, name_to_idx, "dlllist.ndlls", n_dlls)
            self._set_feature(features, name_to_idx, "dlllist.avg_dlls_per_proc",
                              n_dlls / max(len(pslist), 1))
            self._set_feature(features, name_to_idx, "dlllist.avg_dll_size",
                              np.mean(dll_sizes) if dll_sizes else 0)

            # System vs non-system DLLs
            sys_count = sum(1 for p in dll_paths
                           if "windows\\system32" in p.lower() or "windows\\syswow64" in p.lower())
            self._set_feature(features, name_to_idx, "dlllist.nsysdlls", sys_count)

        # ── handles features ───────────────────────────────────────
        handles_data = plugin_results.get("Handles", [])
        if handles_data:
            n_handles = len(handles_data)
            handle_types = [str(sget(h, "Type", "")) for h in handles_data if isinstance(h, dict)]

            self._set_feature(features, name_to_idx, "handles.nhandles", n_handles)
            self._set_feature(features, name_to_idx, "handles.avg_handles_per_proc",
                              n_handles / max(len(pslist), 1))
            self._set_feature(features, name_to_idx, "handles.nfile",
                              sum(1 for t in handle_types if t == "File"))
            self._set_feature(features, name_to_idx, "handles.nevent",
                              sum(1 for t in handle_types if t == "Event"))
            self._set_feature(features, name_to_idx, "handles.ndesktop",
                              sum(1 for t in handle_types if t == "Desktop"))
            self._set_feature(features, name_to_idx, "handles.nkey",
                              sum(1 for t in handle_types if t == "Key"))
            self._set_feature(features, name_to_idx, "handles.nthread",
                              sum(1 for t in handle_types if t == "Thread"))
            self._set_feature(features, name_to_idx, "handles.ndirectory",
                              sum(1 for t in handle_types if t == "Directory"))
            self._set_feature(features, name_to_idx, "handles.nsemaphore",
                              sum(1 for t in handle_types if t == "Semaphore"))
            self._set_feature(features, name_to_idx, "handles.ntimer",
                              sum(1 for t in handle_types if t == "Timer"))
            self._set_feature(features, name_to_idx, "handles.nsection",
                              sum(1 for t in handle_types if t == "Section"))
            self._set_feature(features, name_to_idx, "handles.nmutant",
                              sum(1 for t in handle_types if t == "Mutant"))

        # ── ldrmodules features ────────────────────────────────────
        ldr = plugin_results.get("LdrModules", [])
        if ldr:
            n_ldr = len(ldr)
            self._set_feature(features, name_to_idx, "ldrmodules.ntotal", n_ldr)

            not_in_load = sum(1 for m in ldr if isinstance(m, dict) and not sget(m, "InLoad", True))
            not_in_init = sum(1 for m in ldr if isinstance(m, dict) and not sget(m, "InInit", True))
            not_in_mem = sum(1 for m in ldr if isinstance(m, dict) and not sget(m, "InMem", True))

            self._set_feature(features, name_to_idx, "ldrmodules.not_in_load", not_in_load)
            self._set_feature(features, name_to_idx, "ldrmodules.not_in_init", not_in_init)
            self._set_feature(features, name_to_idx, "ldrmodules.not_in_mem", not_in_mem)
            self._set_feature(features, name_to_idx, "ldrmodules.not_in_load_avg",
                              not_in_load / max(n_ldr, 1))
            self._set_feature(features, name_to_idx, "ldrmodules.not_in_init_avg",
                              not_in_init / max(n_ldr, 1))
            self._set_feature(features, name_to_idx, "ldrmodules.not_in_mem_avg",
                              not_in_mem / max(n_ldr, 1))

        # ── malfind features ──────────────────────────────────────
        malfind = plugin_results.get("Malfind", [])
        if malfind:
            n_injections = len(malfind)
            self._set_feature(features, name_to_idx, "malfind.ninjections", n_injections)
            self._set_feature(features, name_to_idx, "malfind.commitcharge",
                              sum(sget(m, "CommitCharge", 0) for m in malfind if isinstance(m, dict)))
            self._set_feature(features, name_to_idx, "malfind.protection",
                              sum(1 for m in malfind if isinstance(m, dict)
                                  and "EXECUTE" in str(sget(m, "Protection", ""))))
            self._set_feature(features, name_to_idx, "malfind.uniqueinjections",
                              len(set(sget(m, "PID", 0) for m in malfind if isinstance(m, dict))))

        # ── modules features ──────────────────────────────────────
        modules = plugin_results.get("Modules", [])
        if modules:
            self._set_feature(features, name_to_idx, "modules.nmodules", len(modules))

        # ── svcscan features ──────────────────────────────────────
        svcscan = plugin_results.get("SvcScan", [])
        if svcscan:
            self._set_feature(features, name_to_idx, "svcscan.nservices", len(svcscan))
            n_active = sum(1 for s in svcscan if isinstance(s, dict)
                          and str(sget(s, "State", "")).upper() in ("RUNNING", "SERVICE_RUNNING"))
            self._set_feature(features, name_to_idx, "svcscan.kernel_drivers",
                              sum(1 for s in svcscan if isinstance(s, dict) and "KERNEL" in str(sget(s, "Type", "")).upper()))
            self._set_feature(features, name_to_idx, "svcscan.fs_drivers",
                              sum(1 for s in svcscan if isinstance(s, dict) and "FILE_SYSTEM" in str(sget(s, "Type", "")).upper()))
            self._set_feature(features, name_to_idx, "svcscan.process_services",
                              sum(1 for s in svcscan if isinstance(s, dict) and "PROCESS" in str(sget(s, "Type", "")).upper()))
            self._set_feature(features, name_to_idx, "svcscan.shared_process_services",
                              sum(1 for s in svcscan if isinstance(s, dict) and "SHARED" in str(sget(s, "Type", "")).upper()))
            self._set_feature(features, name_to_idx, "svcscan.interactive_process_services",
                              sum(1 for s in svcscan if isinstance(s, dict) and "INTERACTIVE" in str(sget(s, "Type", "")).upper()))
            self._set_feature(features, name_to_idx, "svcscan.nactive", n_active)

        # ── callbacks features ────────────────────────────────────
        callbacks = plugin_results.get("Callbacks", [])
        if callbacks:
            self._set_feature(features, name_to_idx, "callbacks.ncallbacks", len(callbacks))
            self._set_feature(features, name_to_idx, "callbacks.nanonymous",
                              sum(1 for c in callbacks if isinstance(c, dict)
                                  and "UNKNOWN" in str(sget(c, "Module", "")).upper()))

        # ── netscan features ──────────────────────────────────────
        netscan = plugin_results.get("NetScan", [])
        if netscan:
            self._set_feature(features, name_to_idx, "netscan.nconnections", len(netscan))

        return features

    @staticmethod
    def _set_feature(
        features: np.ndarray,
        name_to_idx: Dict[str, int],
        name: str,
        value: float,
    ) -> None:
        """Safely set a feature value by name."""
        if name in name_to_idx:
            features[name_to_idx[name]] = float(value)

    # ── Batch Processing ──────────────────────────────────────────

    def extract_all_features(
        self,
        max_samples: Optional[int] = None,
        timeout_per_dump: int = 600,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from all catalogued dumps.

        Returns:
            X: (n_samples, 55) feature matrix
            y: (n_samples,) labels
        """
        if not self.entries:
            self.load_catalogue()

        entries = self.entries[:max_samples] if max_samples else self.entries

        logger.info(f"Extracting features from {len(entries)} memory dumps...")

        X_list = []
        y_list = []
        processed = 0

        for entry in entries:
            features = self.extract_features_single(entry, timeout=timeout_per_dump)
            if features is not None:
                X_list.append(features)
                y_list.append(entry.label_4class)
                processed += 1
                logger.info(
                    f"  [{processed}/{len(entries)}] {entry.file_name}: "
                    f"{entry.category} (label={entry.label_4class})"
                )

        X = np.array(X_list)
        y = np.array(y_list)

        # Save to CSV
        self._save_features_csv(X, y, entries[:len(X_list)])

        logger.info(f"\nFeature extraction complete: {X.shape[0]} samples × {X.shape[1]} features")
        return X, y

    def _save_features_csv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        entries: List[MemoryDumpEntry],
    ) -> None:
        """Save extracted features to a CSV file compatible with our pipeline."""
        csv_path = self.output_dir / "MalVol25_features.csv"

        # Build DataFrame
        df = pd.DataFrame(X, columns=VOLMEMLYZER_FEATURES)
        df.insert(0, "Category", [e.category for e in entries])
        df.insert(1, "Label", y)
        df.insert(2, "Family", [e.malware_family for e in entries])
        df.insert(3, "OS_Version", [e.os_version for e in entries])
        df.insert(4, "File", [e.file_name for e in entries])

        df.to_csv(csv_path, index=False)
        logger.info(f"Features CSV saved to {csv_path}")

        # Also save in pipeline-ready format (features only + label column)
        pipeline_csv = self.output_dir / "MalVol25_pipeline.csv"
        df_pipeline = pd.DataFrame(X, columns=VOLMEMLYZER_FEATURES)
        df_pipeline["Class"] = y
        df_pipeline.to_csv(pipeline_csv, index=False)
        logger.info(f"Pipeline CSV saved to {pipeline_csv}")

    # ── Load Pre-Extracted Features ────────────────────────────────

    @staticmethod
    def load_features(
        csv_path: str | Path,
        label_column: str = "Class",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load pre-extracted features from CSV.

        Returns:
            X: feature matrix
            y: labels
            feature_names: column names
        """
        df = pd.read_csv(csv_path)

        y = df[label_column].values
        feature_cols = [c for c in df.columns
                       if c not in {label_column, "Category", "Label", "Family",
                                    "OS_Version", "File"}]
        X = df[feature_cols].values
        feature_names = feature_cols

        logger.info(f"Loaded {X.shape[0]} samples × {X.shape[1]} features from {csv_path}")
        return X, y, feature_names

    # ── Dataset Statistics ─────────────────────────────────────────

    def print_summary(self) -> None:
        """Print a summary of the dataset catalogue."""
        if not self.entries:
            self.load_catalogue()

        print(f"\n{'='*60}")
        print(f"MalVol-25 Dataset Summary")
        print(f"{'='*60}")
        print(f"Total dumps: {len(self.entries)}")
        print(f"Total size:  {sum(e.file_size_bytes for e in self.entries) / 1e9:.2f} GB")

        # By category
        print(f"\nBy Category:")
        cats = {}
        for e in self.entries:
            cats[e.category] = cats.get(e.category, 0) + 1
        for cat, count in sorted(cats.items()):
            print(f"  {cat:15s}: {count:4d}")

        # By OS
        print(f"\nBy OS Version:")
        oses = {}
        for e in self.entries:
            oses[e.os_version] = oses.get(e.os_version, 0) + 1
        for os_ver, count in sorted(oses.items()):
            print(f"  {os_ver:15s}: {count:4d}")

        # By malware family
        print(f"\nBy Malware Family:")
        families = {}
        for e in self.entries:
            if e.malware_family and e.malware_family != "none":
                families[e.malware_family] = families.get(e.malware_family, 0) + 1
        for fam, count in sorted(families.items(), key=lambda x: -x[1])[:15]:
            print(f"  {fam:20s}: {count:4d}")

        # 4-class distribution
        print(f"\n4-Class Distribution:")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            count = sum(1 for e in self.entries if e.label_4class == cls_id)
            print(f"  {cls_name:15s} (class {cls_id}): {count:4d}")

        print(f"\nFeatures extracted: {sum(1 for e in self.entries if e.features_extracted)}/{len(self.entries)}")
        print(f"{'='*60}\n")
