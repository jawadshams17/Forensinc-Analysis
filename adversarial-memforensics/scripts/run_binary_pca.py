
def plot_pca_tsne_binary(df, features):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    
    RS = 42
    OUT = r'c:/Users/Microsoft/.gemini/antigravity/brain/11a9adb3-edfa-42c0-9245-f3b6f60f73a8/3_pca_tsne.png'
    
    X = StandardScaler().fit_transform(df[features].values.astype(np.float64))
    df['Plot_Label'] = df['binary_label'].map({0: 'Benign', 1: 'Malware'})
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Label': df['Plot_Label']})
    
    idx = np.random.RandomState(RS).choice(X.shape[0], min(5000, X.shape[0]), replace=False)
    tsne = TSNE(n_components=2, random_state=RS)
    X_tsne = tsne.fit_transform(X[idx])
    df_tsne = pd.DataFrame({'Dim1': X_tsne[:, 0], 'Dim2': X_tsne[:, 1], 'Label': df.iloc[idx]['Plot_Label']})
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Label', ax=axes[0], alpha=0.6, s=15, palette=['#2ca02c', '#d62728'])
    axes[0].set_title('PCA: 2D Projection')
    
    sns.scatterplot(data=df_tsne, x='Dim1', y='Dim2', hue='Label', ax=axes[1], alpha=0.6, s=15, palette=['#2ca02c', '#d62728'])
    axes[1].set_title('t-SNE: 2D Projection (N=5000 subsample)')
    
    plt.tight_layout()
    plt.savefig(OUT, dpi=300)
    plt.close()
    print('Saved PCA plotting.')

import json
ROOT = r'f:/FORENSICS/adversarial-memforensics'
with open(ROOT + '/data/unified/feature_map.json') as f:
    fe = json.load(f)['features']
import pandas as pd
df = pd.read_csv(ROOT + '/data/unified/train_val_test.csv')
plot_pca_tsne_binary(df, fe)
