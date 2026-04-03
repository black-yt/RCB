import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances


OUTPUT_DIR = os.path.join('outputs')
FIG_DIR = os.path.join('report', 'images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_data(path='data/adata_RPE.h5ad'):
    adata = sc.read(path)
    return adata


def basic_qc_and_overview(adata):
    """Generate basic overview plots: state composition, phase, age distribution."""
    df = adata.obs.copy()

    plt.figure(figsize=(4, 3))
    sns.countplot(data=df, x='state')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overview_state_counts.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.countplot(data=df, x='phase')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overview_phase_counts.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(4, 3))
    sns.histplot(df['annotated_age'], bins=30, kde=True)
    plt.xlabel('Annotated age')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'overview_age_hist.png'), dpi=300)
    plt.close()


def compute_trajectory_pseudotime(adata):
    """Use annotated_age as a continuous trajectory variable and compute a normalized pseudotime [0,1]."""
    age = adata.obs['annotated_age'].astype(float).values
    age_min, age_max = age.min(), age.max()
    pseudotime = (age - age_min) / (age_max - age_min + 1e-8)
    adata.obs['pseudotime'] = pseudotime
    return adata


def select_trajectory_features(adata, n_top=40):
    """Select features whose expression is strongly and smoothly associated with pseudotime.

    Strategy:
    1. For each feature, compute Spearman correlation with pseudotime.
    2. Compute a smoothness score along pseudotime (1D total-variation penalty).
    3. Combine association and smoothness to rank features.
    """
    from scipy.stats import spearmanr

    X = adata.layers['raw'] if 'raw' in adata.layers else adata.X
    X = X.toarray() if hasattr(X, 'toarray') else X
    pseudotime = adata.obs['pseudotime'].values

    order = np.argsort(pseudotime)
    pt_sorted = pseudotime[order]
    X_sorted = X[order, :]

    corrs = []
    smoothness = []
    for j in range(X_sorted.shape[1]):
        y = X_sorted[:, j]
        if np.allclose(y, y.mean()):
            corrs.append(0.0)
            smoothness.append(0.0)
            continue
        rho, _ = spearmanr(pt_sorted, y)
        if np.isnan(rho):
            rho = 0.0
        corrs.append(abs(rho))
        diffs = np.diff(y)
        tv = np.mean(np.abs(diffs))
        smooth = 1.0 / (1.0 + tv)
        smoothness.append(smooth)

    corrs = np.array(corrs)
    smoothness = np.array(smoothness)
    score = corrs * smoothness

    var_names = np.array(adata.var_names)
    ranking = np.argsort(score)[::-1]
    top_idx = ranking[:n_top]

    feature_table = pd.DataFrame({
        'feature': var_names,
        'corr_abs': corrs,
        'smoothness': smoothness,
        'score': score,
    }).sort_values('score', ascending=False)

    feature_table.to_csv(os.path.join(OUTPUT_DIR, 'trajectory_feature_scores.csv'), index=False)

    adata.uns['trajectory_feature_scores'] = feature_table
    adata.uns['trajectory_top_features'] = var_names[top_idx]
    return adata


def embedding_comparison(adata, n_pcs=20):
    """Compare global embedding with all features vs selected trajectory features using PCA and UMAP."""
    import umap

    X = adata.layers['raw'] if 'raw' in adata.layers else adata.X
    X = X.toarray() if hasattr(X, 'toarray') else X

    # PCA all features
    pca_all = PCA(n_components=2, random_state=0)
    emb_all = pca_all.fit_transform(X)

    # PCA trajectory features
    top_features = adata.uns['trajectory_top_features']
    feature_idx = [list(adata.var_names).index(f) for f in top_features]
    X_top = X[:, feature_idx]
    pca_top = PCA(n_components=2, random_state=0)
    emb_top = pca_top.fit_transform(X_top)

    df_all = pd.DataFrame(emb_all, columns=['PC1', 'PC2'], index=adata.obs_names)
    df_all['pseudotime'] = adata.obs['pseudotime'].values
    df_all['state'] = adata.obs['state'].values

    df_top = pd.DataFrame(emb_top, columns=['PC1', 'PC2'], index=adata.obs_names)
    df_top['pseudotime'] = adata.obs['pseudotime'].values
    df_top['state'] = adata.obs['state'].values

    # Plot colored by pseudotime
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(df_all['PC1'], df_all['PC2'], c=df_all['pseudotime'], cmap='viridis', s=8)
    plt.title('PCA all features')
    plt.colorbar(sc1, label='pseudotime')
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(df_top['PC1'], df_top['PC2'], c=df_top['pseudotime'], cmap='viridis', s=8)
    plt.title('PCA trajectory features')
    plt.colorbar(sc2, label='pseudotime')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'embedding_pca_pseudotime.png'), dpi=300)
    plt.close()

    # Plot colored by state
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='PC1', y='PC2', hue='state', data=df_all, s=8, linewidth=0, legend=False)
    plt.title('PCA all features')
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='PC1', y='PC2', hue='state', data=df_top, s=8, linewidth=0, legend=True)
    plt.title('PCA trajectory features')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'embedding_pca_state.png'), dpi=300)
    plt.close()

    # UMAP embeddings
    reducer = umap.UMAP(n_components=2, random_state=0)
    umap_all = reducer.fit_transform(X)
    umap_top = reducer.fit_transform(X_top)

    df_ua = pd.DataFrame(umap_all, columns=['UMAP1', 'UMAP2'], index=adata.obs_names)
    df_ua['pseudotime'] = adata.obs['pseudotime'].values
    df_ua['state'] = adata.obs['state'].values

    df_ut = pd.DataFrame(umap_top, columns=['UMAP1', 'UMAP2'], index=adata.obs_names)
    df_ut['pseudotime'] = adata.obs['pseudotime'].values
    df_ut['state'] = adata.obs['state'].values

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sc1 = plt.scatter(df_ua['UMAP1'], df_ua['UMAP2'], c=df_ua['pseudotime'], cmap='viridis', s=8)
    plt.title('UMAP all features')
    plt.colorbar(sc1, label='pseudotime')
    plt.subplot(1, 2, 2)
    sc2 = plt.scatter(df_ut['UMAP1'], df_ut['UMAP2'], c=df_ut['pseudotime'], cmap='viridis', s=8)
    plt.title('UMAP trajectory features')
    plt.colorbar(sc2, label='pseudotime')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'embedding_umap_pseudotime.png'), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='state', data=df_ua, s=8, linewidth=0, legend=False)
    plt.title('UMAP all features')
    plt.subplot(1, 2, 2)
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='state', data=df_ut, s=8, linewidth=0, legend=True)
    plt.title('UMAP trajectory features')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'embedding_umap_state.png'), dpi=300)
    plt.close()

    # Quantitative comparison: preservation of pseudotime distances in 2D
    d_pt = pairwise_distances(adata.obs[['pseudotime']], metric='euclidean')
    d_pca_all = pairwise_distances(emb_all, metric='euclidean')
    d_pca_top = pairwise_distances(emb_top, metric='euclidean')

    # Sample pairs for correlation to save memory
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx_i = rng.integers(0, n, size=5000)
    idx_j = rng.integers(0, n, size=5000)

    da = d_pt[idx_i, idx_j]
    dp_all = d_pca_all[idx_i, idx_j]
    dp_top = d_pca_top[idx_i, idx_j]

    corr_all = np.corrcoef(da, dp_all)[0, 1]
    corr_top = np.corrcoef(da, dp_top)[0, 1]

    with open(os.path.join(OUTPUT_DIR, 'trajectory_distance_correlation.txt'), 'w') as f:
        f.write(f'Correlation between pseudotime distances and PCA(all) distances: {corr_all:.3f}\n')
        f.write(f'Correlation between pseudotime distances and PCA(top) distances: {corr_top:.3f}\n')


def feature_trend_plots(adata, n_show=12):
    """Plot expression trends of top features along pseudotime."""
    X = adata.layers['raw'] if 'raw' in adata.layers else adata.X
    X = X.toarray() if hasattr(X, 'toarray') else X

    pt = adata.obs['pseudotime'].values
    order = np.argsort(pt)
    pt_sorted = pt[order]
    X_sorted = X[order, :]

    top_features = list(adata.uns['trajectory_top_features'])[:n_show]
    feature_idx = [list(adata.var_names).index(f) for f in top_features]

    ncols = 4
    nrows = int(np.ceil(len(top_features) / ncols))

    plt.figure(figsize=(3 * ncols, 2.5 * nrows))
    for i, (feat, idx) in enumerate(zip(top_features, feature_idx)):
        y = X_sorted[:, idx]
        # Smooth by rolling mean
        window = max(5, int(0.02 * len(y)))
        y_smooth = pd.Series(y).rolling(window, center=True, min_periods=1).mean().values

        ax = plt.subplot(nrows, ncols, i + 1)
        ax.scatter(pt_sorted, y, s=4, alpha=0.2, color='gray')
        ax.plot(pt_sorted, y_smooth, color='C0')
        ax.set_title(feat, fontsize=8)
        ax.set_xlabel('pseudotime')
        ax.set_ylabel('expression')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'top_feature_trends.png'), dpi=300)
    plt.close()


def main():
    adata = load_data()
    basic_qc_and_overview(adata)
    adata = compute_trajectory_pseudotime(adata)
    adata = select_trajectory_features(adata, n_top=40)
    embedding_comparison(adata)
    feature_trend_plots(adata)

    # Save processed AnnData for reproducibility
    adata.write(os.path.join(OUTPUT_DIR, 'adata_RPE_trajectory_processed.h5ad'))


if __name__ == '__main__':
    main()
