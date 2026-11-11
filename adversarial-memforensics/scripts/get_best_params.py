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

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid = {'n_estimators': [50, 100], 'max_depth': [3, 5]}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
gs = GridSearchCV(xgb, param_grid=grid, cv=cv, scoring='neg_log_loss', n_jobs=-1)
gs.fit(X, y)

print('BEST_PARAMS_WINNER:', gs.best_params_)
