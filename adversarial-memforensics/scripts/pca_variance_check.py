import json, sys, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = r"f:\FORENSICS\adversarial-memforensics"

with open(ROOT + r"\data\unified\feature_map.json") as f:
    feat_names = json.load(f)["features"]

print(f"Feature set size: {len(feat_names)}")

# PCA on CIC dataset
df = pd.read_csv(r"F:\FORENSICS\MalwareMemoryDump.csv")
X = df[feat_names].values.astype(np.float64)
X_sc = StandardScaler().fit_transform(X)
pca = PCA().fit(X_sc)
c = np.cumsum(pca.explained_variance_ratio_)

print(f"PC1 variance: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"PC2 variance: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"PC3 variance: {pca.explained_variance_ratio_[2]*100:.2f}%")
print(f"PC1+PC2 cumulative: {c[1]*100:.2f}%")
print(f"PC1+PC2+PC3 cumulative: {c[2]*100:.2f}%")
print(f"PCs for 90% variance: {int(np.argmax(c >= 0.90))+1}")
print(f"PCs for 95% variance: {int(np.argmax(c >= 0.95))+1}")
print(f"PCs for 99% variance: {int(np.argmax(c >= 0.99))+1}")

# MalVol25 feature availability
mv = pd.read_csv(ROOT + r"\data\malvol25\extracted\MalVol25_features.csv")
mv.columns = [c.replace(".", "_") for c in mv.columns]
available = [fv for fv in feat_names if fv in mv.columns]
missing   = [fv for fv in feat_names if fv not in mv.columns]
print(f"\nFull feat_names: {len(feat_names)}")
print(f"Available in MalVol25: {len(available)}")
print(f"Missing from MalVol25: {len(missing)}")
print(f"Missing features: {missing}")
