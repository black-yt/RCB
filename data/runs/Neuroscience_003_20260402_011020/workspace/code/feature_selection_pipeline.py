"""
Feature Selection Pipeline for Trajectory-Preserving Protein Imaging Data
==========================================================================
Selects a subset of dynamically expressed molecular features from single-cell
protein imaging data (4i - iterative indirect immunofluorescence imaging) that
best preserves continuous cellular trajectories (cell cycle progression).

Dataset: RPE (retinal pigment epithelium) cells
Features: 241 protein intensity measurements across cellular compartments
Trajectory: annotated_age (continuous, 0-25h cell cycle progression)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import umap
import warnings
warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)

# Paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Neuroscience_003_20260402_011020'
DATA_PATH = os.path.join(WORKSPACE, 'data/adata_RPE.h5ad')
OUTPUTS_PATH = os.path.join(WORKSPACE, 'outputs')
IMAGES_PATH = os.path.join(WORKSPACE, 'report/images')
os.makedirs(OUTPUTS_PATH, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)

print("=" * 60)
print("Feature Selection Pipeline for Trajectory Analysis")
print("=" * 60)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD AND PREPROCESS DATA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] Loading data...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  Shape: {adata.shape}")
print(f"  Features: {adata.n_vars}")
print(f"  Cells: {adata.n_obs}")
print(f"  Obs columns: {list(adata.obs.columns)}")
print(f"  Age range: {adata.obs['annotated_age'].min():.2f} - {adata.obs['annotated_age'].max():.2f}")

# Extract data matrix
X = np.array(adata.X)
feature_names = np.array(adata.var_names)
age = adata.obs['annotated_age'].values
phase = adata.obs['phase'].astype(str).values
state = adata.obs['state'].astype(str).values
batch = adata.obs['batch'].astype(str).values

print(f"\n  Phase distribution: {dict(pd.Series(phase).value_counts())}")
print(f"  State distribution: {dict(pd.Series(state).value_counts())}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE QUALITY ASSESSMENT
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] Feature quality assessment...")

# Check for missing values
nan_fraction = np.isnan(X).mean(axis=0)
print(f"  Features with NaN > 5%: {(nan_fraction > 0.05).sum()}")
print(f"  Max NaN fraction: {nan_fraction.max():.4f}")

# Remove features with > 5% NaN
valid_features = nan_fraction <= 0.05
X = X[:, valid_features]
feature_names = feature_names[valid_features]
print(f"  Features after NaN filter: {X.shape[1]}")

# Fill remaining NaN with column median
for j in range(X.shape[1]):
    col = X[:, j]
    if np.isnan(col).any():
        X[np.isnan(col), j] = np.nanmedian(col)

# ─────────────────────────────────────────────────────────────────────────────
# 3. NORMALIZATION AND BATCH CORRECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Normalization...")

# Clip extreme outliers (beyond 1st/99th percentile per feature)
p01 = np.percentile(X, 1, axis=0)
p99 = np.percentile(X, 99, axis=0)
X_clipped = np.clip(X, p01, p99)

# Z-score normalize per feature
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_clipped)

# Batch-aware normalization: subtract batch mean per feature
batches = np.array(batch)
unique_batches = np.unique(batches)
X_batch_corrected = X_norm.copy()
for b in unique_batches:
    mask = batches == b
    X_batch_corrected[mask] -= X_batch_corrected[mask].mean(axis=0)

print(f"  Normalized data shape: {X_batch_corrected.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE SCORING - MULTIPLE CRITERIA
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Computing feature scores...")

n_features = X_batch_corrected.shape[1]
scores = pd.DataFrame({'feature': feature_names})

# 4a. Variance score (higher = more variable)
scores['variance'] = X_batch_corrected.var(axis=0)

# 4b. Spearman correlation with annotated_age (absolute value)
spearman_corr = np.zeros(n_features)
spearman_pval = np.zeros(n_features)
for j in range(n_features):
    r, p = stats.spearmanr(X_batch_corrected[:, j], age)
    spearman_corr[j] = abs(r)
    spearman_pval[j] = p
scores['spearman_corr'] = spearman_corr
scores['spearman_pval'] = spearman_pval

# 4c. Mutual information with age (captures non-linear relationships)
print("  Computing mutual information...")
mi_scores = mutual_info_regression(X_batch_corrected, age, n_neighbors=5, random_state=42)
scores['mutual_info'] = mi_scores

# 4d. Dynamic range score: max expression difference across cell cycle phases
# Coefficient of variation across phase means
cycling_mask = state == 'cycling'
X_cycling = X_batch_corrected[cycling_mask]
phase_cycling = phase[cycling_mask]
phase_groups = ['G1', 'S', 'G2']  # ordered cell cycle phases for cycling cells
phase_means = np.zeros((len(phase_groups), n_features))
for i, ph in enumerate(phase_groups):
    mask = phase_cycling == ph
    if mask.sum() > 0:
        phase_means[i] = X_cycling[mask].mean(axis=0)
dynamic_range = phase_means.max(axis=0) - phase_means.min(axis=0)
scores['dynamic_range'] = dynamic_range

# 4e. Trajectory smoothness score: correlation of feature with SMOOTH pseudo-age
# Sort cells by age and compute rolling correlation
age_sorted_idx = np.argsort(age)
X_age_sorted = X_batch_corrected[age_sorted_idx]
age_sorted = age[age_sorted_idx]

window = 100
smooth_scores = np.zeros(n_features)
for j in range(n_features):
    # Pearson correlation with local trend (smoothed)
    kernel = np.ones(window) / window
    smoothed = np.convolve(X_age_sorted[:, j], kernel, mode='valid')
    smoothed_age = np.convolve(age_sorted, kernel, mode='valid')
    r, _ = stats.pearsonr(smoothed, smoothed_age)
    smooth_scores[j] = abs(r)
scores['smooth_corr'] = smooth_scores

# 4f. Batch effect score: low batch effect is preferable
batch_effect = np.zeros(n_features)
for j in range(n_features):
    b1 = X_batch_corrected[batches == '1', j]
    b2 = X_batch_corrected[batches == '2', j]
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((b1.std()**2 + b2.std()**2) / 2)
    if pooled_std > 0:
        batch_effect[j] = abs(b1.mean() - b2.mean()) / pooled_std
    else:
        batch_effect[j] = 0
scores['batch_effect'] = batch_effect
scores['batch_score'] = 1 / (1 + batch_effect)  # inverse: low batch = good

print(f"  Feature scores computed for {n_features} features")

# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPOSITE SCORING AND FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] Computing composite scores and selecting features...")

# Normalize each score to [0,1]
def normalize_score(s):
    s = np.array(s, dtype=float)
    rng = s.max() - s.min()
    if rng == 0:
        return np.zeros_like(s)
    return (s - s.min()) / rng

scores['var_norm'] = normalize_score(scores['variance'])
scores['spearman_norm'] = normalize_score(scores['spearman_corr'])
scores['mi_norm'] = normalize_score(scores['mutual_info'])
scores['dynrange_norm'] = normalize_score(scores['dynamic_range'])
scores['smooth_norm'] = normalize_score(scores['smooth_corr'])
scores['batch_norm'] = normalize_score(scores['batch_score'])

# Composite score: weighted combination
# Dynamic expression (corr + MI + dynamic range) weighted more
weights = {
    'spearman_norm': 0.25,
    'mi_norm': 0.25,
    'dynrange_norm': 0.20,
    'smooth_norm': 0.15,
    'var_norm': 0.10,
    'batch_norm': 0.05,
}
scores['composite_score'] = sum(w * scores[k] for k, w in weights.items())

# Rank features
scores = scores.sort_values('composite_score', ascending=False).reset_index(drop=True)
scores['rank'] = np.arange(1, len(scores) + 1)

print(f"\n  Top 30 features by composite score:")
print(scores[['rank', 'feature', 'composite_score', 'spearman_corr', 'mutual_info', 'dynamic_range']].head(30).to_string(index=False))

# Save feature scores
scores.to_csv(os.path.join(OUTPUTS_PATH, 'feature_scores.csv'), index=False)
print(f"\n  Saved feature scores to outputs/feature_scores.csv")

# Select top N features
TOP_N_LIST = [20, 30, 50]
selected_features = {}
for n in TOP_N_LIST:
    selected_features[n] = scores['feature'].values[:n]
    print(f"  Top {n} features selected")

# ─────────────────────────────────────────────────────────────────────────────
# 6. VALIDATE TRAJECTORY PRESERVATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n[6] Validating trajectory preservation...")

def compute_trajectory_metrics(X_sub, age, n_neighbors=15, random_state=42):
    """Compute metrics quantifying how well a feature subset preserves the age trajectory."""
    # 1. kNN overlap: does local neighborhood in feature space match age-proximity?
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_sub)
    _, knn_idx = nbrs.kneighbors(X_sub)

    # Compute age-based kNN
    age_dist = np.abs(age[:, None] - age[None, :])
    age_knn = np.argsort(age_dist, axis=1)[:, 1:n_neighbors+1]

    # Overlap: fraction of kNN that are also age-kNN
    overlap = np.mean([
        len(np.intersect1d(knn_idx[i], age_knn[i])) / n_neighbors
        for i in range(len(age))
    ])

    # 2. Spearman correlation of feature-space distance with age distance
    # Sample 500 pairs for efficiency
    n = len(age)
    np.random.seed(random_state)
    idx = np.random.choice(n, min(500, n), replace=False)
    feat_dist = cdist(X_sub[idx], X_sub[idx], 'euclidean').flatten()
    age_dist_sub = np.abs(age[idx, None] - age[None, idx]).flatten()
    corr, _ = stats.spearmanr(feat_dist, age_dist_sub)
    # Negative because closer in age = closer in feature space (negative dist corr is bad)
    # We want high -corr (cells close in age are close in feature space)

    # 3. Pseudo-time recovery: how well does 1D PCA of features recover age ordering?
    pca1 = PCA(n_components=1).fit_transform(X_sub).flatten()
    pt_corr, _ = stats.spearmanr(pca1, age)

    return {
        'knn_overlap': overlap,
        'dist_corr': -corr,  # higher = closer in feature space when close in age
        'pseudotime_corr': abs(pt_corr)
    }

# All features
print("  Computing metrics for all features...")
metrics_all = compute_trajectory_metrics(X_batch_corrected, age)

# Selected feature subsets
metrics_by_n = {}
for n in TOP_N_LIST:
    feat_idx = [np.where(feature_names == f)[0][0] for f in selected_features[n]]
    X_sub = X_batch_corrected[:, feat_idx]
    metrics_by_n[n] = compute_trajectory_metrics(X_sub, age)
    print(f"  Top {n}: kNN_overlap={metrics_by_n[n]['knn_overlap']:.3f}, "
          f"dist_corr={metrics_by_n[n]['dist_corr']:.3f}, "
          f"pseudotime_corr={metrics_by_n[n]['pseudotime_corr']:.3f}")

print(f"  All ({n_features}): kNN_overlap={metrics_all['knn_overlap']:.3f}, "
      f"dist_corr={metrics_all['dist_corr']:.3f}, "
      f"pseudotime_corr={metrics_all['pseudotime_corr']:.3f}")

# Save metrics
metrics_df = pd.DataFrame({
    'n_features': ['all'] + [str(n) for n in TOP_N_LIST],
    'knn_overlap': [metrics_all['knn_overlap']] + [metrics_by_n[n]['knn_overlap'] for n in TOP_N_LIST],
    'dist_corr': [metrics_all['dist_corr']] + [metrics_by_n[n]['dist_corr'] for n in TOP_N_LIST],
    'pseudotime_corr': [metrics_all['pseudotime_corr']] + [metrics_by_n[n]['pseudotime_corr'] for n in TOP_N_LIST],
})
metrics_df.to_csv(os.path.join(OUTPUTS_PATH, 'trajectory_metrics.csv'), index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 7. UMAP EMBEDDINGS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7] Computing UMAP embeddings...")

reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.3, random_state=42)

# All features
print("  UMAP on all features...")
umap_all = reducer.fit_transform(X_batch_corrected)
np.save(os.path.join(OUTPUTS_PATH, 'umap_all.npy'), umap_all)

# Best subset (top 30)
BEST_N = 30
feat_idx_best = [np.where(feature_names == f)[0][0] for f in selected_features[BEST_N]]
X_best = X_batch_corrected[:, feat_idx_best]
print(f"  UMAP on top {BEST_N} features...")
umap_best = reducer.fit_transform(X_best)
np.save(os.path.join(OUTPUTS_PATH, 'umap_best30.npy'), umap_best)

# Top 20 features
feat_idx_20 = [np.where(feature_names == f)[0][0] for f in selected_features[20]]
X_top20 = X_batch_corrected[:, feat_idx_20]
print("  UMAP on top 20 features...")
umap_top20 = reducer.fit_transform(X_top20)
np.save(os.path.join(OUTPUTS_PATH, 'umap_top20.npy'), umap_top20)

# Top 50 features
feat_idx_50 = [np.where(feature_names == f)[0][0] for f in selected_features[50]]
X_top50 = X_batch_corrected[:, feat_idx_50]
print("  UMAP on top 50 features...")
umap_top50 = reducer.fit_transform(X_top50)
np.save(os.path.join(OUTPUTS_PATH, 'umap_top50.npy'), umap_top50)

print("  All UMAP embeddings computed and saved")

# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE KEY RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[8] Saving results...")

# Save selected feature lists
for n in TOP_N_LIST:
    feat_df = scores[['rank', 'feature', 'composite_score', 'spearman_corr',
                       'mutual_info', 'dynamic_range', 'smooth_corr']].head(n)
    feat_df.to_csv(os.path.join(OUTPUTS_PATH, f'selected_features_top{n}.csv'), index=False)

# Save preprocessed data
np.save(os.path.join(OUTPUTS_PATH, 'X_norm.npy'), X_batch_corrected)
np.save(os.path.join(OUTPUTS_PATH, 'feature_names.npy'), feature_names)
np.save(os.path.join(OUTPUTS_PATH, 'age.npy'), age)
np.save(os.path.join(OUTPUTS_PATH, 'phase.npy'), phase)
np.save(os.path.join(OUTPUTS_PATH, 'state.npy'), state)
np.save(os.path.join(OUTPUTS_PATH, 'batch.npy'), batch)

print("  All data saved to outputs/")
print("\n[DONE] Pipeline completed successfully.")
print(f"  Best subset (top {BEST_N} features) preserves trajectory with:")
print(f"    kNN overlap: {metrics_by_n[BEST_N]['knn_overlap']:.3f}")
print(f"    Distance correlation: {metrics_by_n[BEST_N]['dist_corr']:.3f}")
print(f"    Pseudotime recovery: {metrics_by_n[BEST_N]['pseudotime_corr']:.3f}")
