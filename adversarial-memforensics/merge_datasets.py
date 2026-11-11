"""
merge_datasets.py
=================
Merges MalwareMemoryDump.csv (58,596 samples) with MalVol-25 (30 samples)
into a unified feature space using the 41 common VolMemLyzer features.

Outputs:
    data/unified/train_val_test.csv   -- MalwareMemoryDump (split-ready)
    data/unified/malvol25_holdout.csv -- MalVol-25 (cross-dataset test)
    data/unified/feature_map.json     -- canonical 41-feature name list
"""
import sys, json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Paths ──────────────────────────────────────────────────────────────
DUMP_CSV    = Path(r"F:\FORENSICS\MalwareMemoryDump.csv")
MALVOL_CSV  = ROOT / "data" / "malvol25" / "extracted" / "MalVol25_features.csv"
OUT_DIR     = ROOT / "data" / "unified"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MalVol-25 non-feature columns ─────────────────────────────────────
MV_METADATA = {"Category", "Label", "Family", "OS_Version", "File", "OS", "Class"}

# ── Load MalwareMemoryDump ─────────────────────────────────────────────
print("Loading MalwareMemoryDump.csv ...")
dump_df = pd.read_csv(DUMP_CSV)

# Label: Benign=0, Malware=1
dump_df["binary_label"] = (dump_df["Label"] == "Malware").astype(int)

# Numeric feature cols only (drop Raw_Type, SubType, Label)
dump_meta = {"Raw_Type", "SubType", "Label", "binary_label"}
dump_feat_cols = [c for c in dump_df.columns
                  if c not in dump_meta and pd.api.types.is_numeric_dtype(dump_df[c])]
print(f"  MalwareMemoryDump features: {len(dump_feat_cols)}")

# ── Load MalVol-25 ─────────────────────────────────────────────────────
print("Loading MalVol25_features.csv ...")
mv_df = pd.read_csv(MALVOL_CSV)

# Normalise dot → underscore in MalVol-25 column names
mv_df.columns = [c.replace(".", "_") for c in mv_df.columns]
MV_METADATA_NORM = {m.replace(".", "_") for m in MV_METADATA}

# Label: use numeric Label col directly; 0=Benign, else Malware
mv_df["binary_label"] = (mv_df["Label"] != 0).astype(int)

mv_feat_cols = [c for c in mv_df.columns
                if c not in MV_METADATA_NORM and pd.api.types.is_numeric_dtype(mv_df[c])
                and c not in {"Label", "binary_label"}]
print(f"  MalVol-25 features (normalised): {len(mv_feat_cols)}")

# ── Find common features ───────────────────────────────────────────────
common_feats = sorted(set(dump_feat_cols) & set(mv_feat_cols))
print(f"\nShared features: {len(common_feats)}")

# ── Build unified MalwareMemoryDump split-ready CSV ────────────────────
dump_out = dump_df[common_feats + ["binary_label"]].copy()
dump_out.columns = common_feats + ["label"]
dump_out.to_csv(OUT_DIR / "train_val_test.csv", index=False)
print(f"  Saved train_val_test.csv: {dump_out.shape}")

# ── Build MalVol-25 holdout CSV ────────────────────────────────────────
mv_out = mv_df[common_feats + ["binary_label"]].copy()
mv_out.columns = common_feats + ["label"]
mv_out.to_csv(OUT_DIR / "malvol25_holdout.csv", index=False)
print(f"  Saved malvol25_holdout.csv: {mv_out.shape}")

# ── Save feature map ───────────────────────────────────────────────────
with open(OUT_DIR / "feature_map.json", "w") as f:
    json.dump({"features": common_feats, "n_features": len(common_feats)}, f, indent=2)
print(f"  Saved feature_map.json  ({len(common_feats)} features)")

# ── Label summary ──────────────────────────────────────────────────────
print("\n=== FINAL DATASET SUMMARY ===")
print(f"Primary (train/val/test) : {len(dump_out):>6,} samples  |  "
      f"Benign={int((dump_out.label==0).sum())}  Malware={int((dump_out.label==1).sum())}")
print(f"Cross-dataset holdout    : {len(mv_out):>6,} samples  |  "
      f"Benign={int((mv_out.label==0).sum())}  Malware={int((mv_out.label==1).sum())}")
print(f"Common feature space     : {len(common_feats):>6} features")
print("Done.")
