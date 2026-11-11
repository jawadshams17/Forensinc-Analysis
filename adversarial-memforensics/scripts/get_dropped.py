import pandas as pd
import json

df1 = pd.read_csv(r'F:\FORENSICS\MalwareMemoryDump.csv')
num_cols = [c for c in df1.columns if pd.api.types.is_numeric_dtype(df1[c])]
with open(r'f:\FORENSICS\adversarial-memforensics\data\unified\feature_map.json') as f:
    f_map = json.load(f)['features']

print('CIC numeric cols:', len(num_cols))
dropped = [c for c in num_cols if c not in f_map]
print('Dropped from CIC:', dropped)
