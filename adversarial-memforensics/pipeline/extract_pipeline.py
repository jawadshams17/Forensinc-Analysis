"""
Stream-extract MalVol-25 ZIP one file at a time, run malvol25-scan after each,
accumulate features, then delete the dump to save disk space.

Usage:
    python pipeline/extract_pipeline.py

Writes: data/malvol25/extracted/MalVol25_features.csv
"""

from __future__ import annotations
import csv
import json
import os
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd

# ── project root on path ────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from perturbation_primitives import VOLMEMLYZER_FEATURES
from pipeline.malvol25_loader import (
    MalVol25Loader, MemoryDumpEntry, CATEGORY_MAP_4CLASS, MALVOL_CATEGORIES
)

ZIP_PATH = ROOT / "data" / "tmp_dump.mem"          # temp single-file slot
ZIP_SRC  = pathlib.Path(r"f:\FORENSICS\Dataset All Files_0 (1).zip")
OUT_DIR  = ROOT / "data" / "malvol25" / "extracted"
DUMP_DIR = ROOT / "data" / "malvol25" / "tmp"

DUMP_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = OUT_DIR / "MalVol25_features.csv"


def parse_label(filename: str) -> tuple[str, str, int]:
    """Derive category, infection_status, label from filename."""
    name = filename.lower()
    if "novirus" in name or "no-virus" in name or "no_virus" in name:
        infection = "Clean"
        category  = "Benign"
        label     = 0
    else:
        infection = "Infected"
        category  = "Unknown"
        for kw, cat in MALVOL_CATEGORIES.items():
            if kw.lower() in name:
                category = cat
                break
        label = CATEGORY_MAP_4CLASS.get(category, 3)
    return category, infection, label


def main():
    loader = MalVol25Loader(str(DUMP_DIR), str(OUT_DIR))

    all_rows: list[dict] = []

    with zipfile.ZipFile(ZIP_SRC, "r") as zf:
        file_infos = [i for i in zf.infolist() if not i.filename.endswith("/")]
        total = len(file_infos)
        print(f"Found {total} dump files in ZIP.", flush=True)

        for idx, info in enumerate(file_infos, 1):
            fname = pathlib.PurePosixPath(info.filename).name
            dump_path = DUMP_DIR / fname
            category, infection, label = parse_label(fname)

            print(f"\n[{idx}/{total}] {fname}  ({info.file_size/1e9:.2f} GB)  [{category}]", flush=True)

            # ── extract single dump ──────────────────────────────────
            print("  Extracting...", flush=True)
            with zf.open(info) as src_f, open(dump_path, "wb") as dst_f:
                while chunk := src_f.read(64 * 1024 * 1024):
                    dst_f.write(chunk)

            # ── build entry + extract features ───────────────────────
            entry = MemoryDumpEntry(
                file_path=dump_path,
                file_name=fname,
                file_size_bytes=dump_path.stat().st_size,
                os_version="Win10",
                infection_status=infection,
                category=category,
                malware_family=fname.split("-")[0] if "-" in fname else fname.split(".")[0],
                label_4class=label,
                file_format=dump_path.suffix.lower(),
            )

            try:
                features = loader.extract_features_single(entry, timeout=900)
            except Exception as e:
                print(f"  Feature extraction failed: {e}", flush=True)
                features = None

            # ── save row ─────────────────────────────────────────────
            if features is not None:
                row = dict(zip(VOLMEMLYZER_FEATURES, features.tolist()))
                row["Category"] = category
                row["Label"]    = label
                row["Family"]   = entry.malware_family
                row["OS"]       = entry.os_version
                row["File"]     = fname
                all_rows.append(row)
                print(f"  Features extracted OK. Rows so far: {len(all_rows)}", flush=True)

                # Save incrementally after every file
                df = pd.DataFrame(all_rows)
                df.to_csv(FEATURES_CSV, index=False)
            else:
                print("  Skipped (no features).", flush=True)

            # ── delete dump to reclaim disk space ────────────────────
            dump_path.unlink(missing_ok=True)
            print("  Dump deleted to free disk space.", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"DONE. {len(all_rows)} samples saved to {FEATURES_CSV}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
