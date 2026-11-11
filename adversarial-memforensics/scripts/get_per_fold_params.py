import json
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier

ROOT = r'f:\FORENSICS\adversarial-memforensics'
with open(ROOT + r'\data\unified\feature_map.json') as f:
    features = json.load(f)['features']

df = pd.read_csv(r'F:\FORENSICS\MalwareMemoryDump.csv')
df['label'] = (df['SubType'].str.strip() != 'Benign').astype(int)

X = df[features].values
y = df['label'].values

SEEDS = [1, 7, 42, 99]
xgb_grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}

print("=== NESTED CV PER-FOLD BEST PARAMS ===")
for rs in SEEDS:
    print(f"\nSeed {rs}:")
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=rs)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=rs)

    for fold_idx, (train_idx, _) in enumerate(outer_cv.split(X, y)):
        X_train, y_train = X[train_idx], y[train_idx]
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=rs, verbosity=0)
        gs = GridSearchCV(xgb, param_grid=xgb_grid, cv=inner_cv, scoring='neg_log_loss', n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"  Fold {fold_idx+1}: {gs.best_params_}")
