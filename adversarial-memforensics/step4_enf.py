"""
step4_enf.py
============
Q1 Hardening — Step 4: Environment-Normalised Features (ENF)

Implements a forensics-aware normalisation technique:
1. Reads the taxonomy from Step 3.
2. For environment-dependent features, drops the absolute scales.
3. For malware-indicative features and mixed features, keeps them
   or normalises them relative to system size.
4. Evaluates ENF using the Transductive Zero-Shot protocol and 100-Trial protocol.

Outputs:
    results/unified/step4_enf.json
"""
import sys, json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
except ImportError:
    raise SystemExit("pip install xgboost lightgbm")

warnings.filterwarnings("ignore")

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "results" / "unified"

TRAIN_CSV   = ROOT / "data" / "unified" / "train_val_test.csv"
HOLDOUT_CSV = ROOT / "data" / "unified" / "malvol25_holdout.csv"
FEAT_MAP    = ROOT / "data" / "unified" / "feature_map.json"
TAXONOMY_F  = OUT_DIR / "step3_taxonomy.json"

RS = 42

def load_data():
    df_src = pd.read_csv(TRAIN_CSV)
    df_tgt = pd.read_csv(HOLDOUT_CSV)
    with open(FEAT_MAP) as f:
        feat_names = json.load(f)["features"]
    X_src = df_src[feat_names].values.astype(np.float64)
    y_src = df_src["label"].values.astype(int)
    X_tgt = df_tgt[feat_names].values.astype(np.float64)
    y_tgt = df_tgt["label"].values.astype(int)
    return X_src, y_src, X_tgt, y_tgt, feat_names, df_src, df_tgt


def make_ensemble():
    rf   = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RS)
    xgb  = XGBClassifier(n_estimators=100, use_label_encoder=False,
                          eval_metric="logloss", random_state=RS, verbosity=0)
    lgbm = LGBMClassifier(n_estimators=100, random_state=RS, verbosity=-1)
    return VotingClassifier(estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm)],
                             voting="soft", n_jobs=-1)


def build_enf_features(df, taxonomy):
    """
    Constructs the ENF representation for a dataset.
    - Drops purely Environment-Dependent absolute count features.
    - Preserves Globally Invariant and Malware-Context features.
    - Adds system-relative ratios (ENF normalisation).
    """
    env_feats = [t["feature"] for t in taxonomy if t["category"] == "Environment-Dependent"]
    mal_feats = [t["feature"] for t in taxonomy if t["category"] != "Environment-Dependent"]
    
    out_df = pd.DataFrame()
    
    # Keep the malware-indicative ones
    for f in mal_feats:
        if f in df.columns:
            out_df[f] = df[f]
            
    # Compute relative ratios using system anchors
    # We use pslist_nproc, modules_nmodules, handles_nhandles as anchors
    # (even though they are Environment-Dependent, they proxy system size)
    def safe_div(num, den):
        return np.where(den > 0, num / den, 0)
        
    anchors = {
        "pslist_nproc": ["dlllist_ndlls", "malfind_ninjections", "ldrmodules_not_in_load", "svcscan_nservices"],
        "modules_nmodules": ["callbacks_ncallbacks"],
        "handles_nhandles": ["handles_nfile", "handles_nthread", "handles_nmutant"]
    }
    
    for anchor, targets in anchors.items():
        if anchor in df.columns:
            den = df[anchor].values
            for num in targets:
                if num in df.columns:
                    out_df[f"{num}_per_{anchor}"] = safe_div(df[num].values, den)
                    
    return out_df.values.astype(np.float64)


def transform_enf_coral(X_src, X_tgt):
    sc_s = StandardScaler().fit(X_src)
    sc_t = StandardScaler().fit(X_tgt)
    Xs = sc_s.transform(X_src)
    Xt = sc_t.transform(X_tgt)
    n = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + np.eye(n) * 1e-6
    Ct = np.cov(Xt, rowvar=False) + np.eye(n) * 1e-6
    try:
        Ds, Vs = np.linalg.eigh(Cs)
        Ds = np.maximum(Ds, 1e-8)
        Cs_nh = Vs @ np.diag(1.0 / np.sqrt(Ds)) @ Vs.T
        Dt, Vt = np.linalg.eigh(Ct)
        Dt = np.maximum(Dt, 1e-8)
        Ct_h  = Vt @ np.diag(np.sqrt(Dt)) @ Vt.T
        A = Cs_nh @ Ct_h
        return Xs @ A, Xt
    except Exception as e:
        logger.warning(f"CORAL failed on ENF features: {e}. Falling back to standard scaling.")
        return Xs, Xt


def main():
    logger.info("=" * 65)
    logger.info("  STEP 4: ENVIRONMENT-NORMALISED FEATURES (ENF)")
    logger.info("=" * 65)

    if not TAXONOMY_F.exists():
        logger.error(f"Taxonomy file {TAXONOMY_F} not found. Run Step 3 first.")
        sys.exit(1)
        
    X_src, y_src, X_tgt, y_tgt, feat_names, df_src, df_tgt = load_data()
    
    with open(TAXONOMY_F) as f:
        tax_data = json.load(f)
    taxonomy = tax_data["taxonomy"]
    
    logger.info("  Building ENF Representations...")
    X_src_enf = build_enf_features(df_src, taxonomy)
    X_tgt_enf = build_enf_features(df_tgt, taxonomy)
    
    logger.info(f"    Original feature dim : {X_src.shape[1]}")
    logger.info(f"    ENF feature dim      : {X_src_enf.shape[1]}")
    
    # ── Transductive Evaluation ──
    logger.info("\n  Evaluating ENF (+ CORAL) [Transductive Zero-Shot]...")
    Xs_t, Xt_t = transform_enf_coral(X_src_enf, X_tgt_enf)
    
    model = make_ensemble()
    model.fit(Xs_t, y_src)
    
    yp = model.predict(Xt_t)
    try: ypr = model.predict_proba(Xt_t)[:, 1]
    except: ypr = np.zeros(len(y_tgt))
    
    acc = accuracy_score(y_tgt, yp)
    f1  = f1_score(y_tgt, yp, average="macro", zero_division=0)
    try: auc = roc_auc_score(y_tgt, ypr)
    except: auc = 0.0
    
    logger.info(f"    → Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    
    res = {
        "method": "ENF + CORAL",
        "accuracy": acc,
        "f1_macro": f1,
        "roc_auc": auc
    }
    
    with open(OUT_DIR / "step4_enf.json", "w") as f:
        json.dump(res, f, indent=2)
        
    logger.success("  Done. Step 4 ENF results saved.")
    
if __name__ == "__main__":
    main()
