import json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = r'f:\FORENSICS\adversarial-memforensics'
with open(ROOT + r'\data\unified\feature_map.json') as f:
    feat_names = json.load(f)['features']

mv = pd.read_csv(ROOT + r'\data\malvol25\extracted\MalVol25_features.csv')
mv.columns = [c.replace('.','_') for c in mv.columns]

X_mv = mv[feat_names].values.astype(np.float64)
X_mv_sc = StandardScaler().fit_transform(X_mv)
pca_mv = PCA().fit(X_mv_sc)
c_mv = np.cumsum(pca_mv.explained_variance_ratio_)

print('--- MalVol-25 PCA (N=30, 41 features) ---')
print('PC1 variance:', round(pca_mv.explained_variance_ratio_[0]*100, 2), '%')
print('PC2 variance:', round(pca_mv.explained_variance_ratio_[1]*100, 2), '%')
print('PC3 variance:', round(pca_mv.explained_variance_ratio_[2]*100, 2), '%')
print('PC1+PC2 cumulative:', round(c_mv[1]*100, 2), '%')
print('PC1+PC2+PC3 cumulative:', round(c_mv[2]*100, 2), '%')
print('PCs for 90% variance:', int(np.argmax(c_mv >= 0.90))+1)
print('PCs for 95% variance:', int(np.argmax(c_mv >= 0.95))+1)
print('PCs for 99% variance:', int(np.argmax(c_mv >= 0.99))+1)
