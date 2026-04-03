"""
Figure Generation for Feature Selection Analysis
=================================================
Generates all publication-quality figures for the research report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Paths
WORKSPACE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Neuroscience_003_20260402_011020'
OUTPUTS_PATH = os.path.join(WORKSPACE, 'outputs')
IMAGES_PATH = os.path.join(WORKSPACE, 'report/images')
os.makedirs(IMAGES_PATH, exist_ok=True)

# Load saved data
print("Loading saved data...")
X_norm = np.load(os.path.join(OUTPUTS_PATH, 'X_norm.npy'))
feature_names = np.load(os.path.join(OUTPUTS_PATH, 'feature_names.npy'), allow_pickle=True)
age = np.load(os.path.join(OUTPUTS_PATH, 'age.npy'))
phase = np.load(os.path.join(OUTPUTS_PATH, 'phase.npy'), allow_pickle=True)
state = np.load(os.path.join(OUTPUTS_PATH, 'state.npy'), allow_pickle=True)
batch = np.load(os.path.join(OUTPUTS_PATH, 'batch.npy'), allow_pickle=True)

umap_all = np.load(os.path.join(OUTPUTS_PATH, 'umap_all.npy'))
umap_best30 = np.load(os.path.join(OUTPUTS_PATH, 'umap_best30.npy'))
umap_top20 = np.load(os.path.join(OUTPUTS_PATH, 'umap_top20.npy'))
umap_top50 = np.load(os.path.join(OUTPUTS_PATH, 'umap_top50.npy'))

scores = pd.read_csv(os.path.join(OUTPUTS_PATH, 'feature_scores.csv'))
metrics_df = pd.read_csv(os.path.join(OUTPUTS_PATH, 'trajectory_metrics.csv'))
top30 = pd.read_csv(os.path.join(OUTPUTS_PATH, 'selected_features_top30.csv'))
top20 = pd.read_csv(os.path.join(OUTPUTS_PATH, 'selected_features_top20.csv'))

# Style
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

# Color palettes
PHASE_COLORS = {'G0': '#7f7f7f', 'G1': '#4477AA', 'S': '#66CCEE', 'G2': '#228833'}
AGE_CMAP = 'plasma'

print("Generating figures...")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Data Overview
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 1: Data overview...")
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 1a: Cell distribution by phase
ax1 = fig.add_subplot(gs[0, 0])
phase_counts = pd.Series(phase).value_counts()
phase_order = ['G0', 'G1', 'S', 'G2']
colors_bar = [PHASE_COLORS[p] for p in phase_order if p in phase_counts]
bars = ax1.bar([p for p in phase_order if p in phase_counts],
               [phase_counts[p] for p in phase_order if p in phase_counts],
               color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, cnt in zip(bars, [phase_counts[p] for p in phase_order if p in phase_counts]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             str(cnt), ha='center', va='bottom', fontsize=9)
ax1.set_xlabel('Cell Cycle Phase')
ax1.set_ylabel('Number of Cells')
ax1.set_title('Cell Distribution by Phase')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# 1b: Age distribution by phase
ax2 = fig.add_subplot(gs[0, 1])
for ph in phase_order:
    mask = phase == ph
    if mask.sum() > 0:
        ax2.hist(age[mask], bins=30, alpha=0.6, label=ph, color=PHASE_COLORS[ph], density=True)
ax2.set_xlabel('Annotated Age (hours)')
ax2.set_ylabel('Density')
ax2.set_title('Age Distribution by Phase')
ax2.legend(fontsize=8)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# 1c: State distribution
ax3 = fig.add_subplot(gs[0, 2])
state_counts = pd.Series(state).value_counts()
state_colors = {'cycling': '#2196F3', 'arrested': '#FF5722', 'nan': '#9E9E9E'}
ax3.pie([state_counts.get(s, 0) for s in ['cycling', 'arrested', 'nan']],
        labels=['Cycling', 'Arrested', 'Unassigned'],
        colors=[state_colors[s] for s in ['cycling', 'arrested', 'nan']],
        autopct='%1.1f%%', startangle=90)
ax3.set_title('Cell State Distribution')

# 1d: Feature variance distribution
ax4 = fig.add_subplot(gs[1, 0])
vars_ = X_norm.var(axis=0)
ax4.hist(vars_, bins=40, color='steelblue', alpha=0.8, edgecolor='white')
ax4.axvline(np.percentile(vars_, 75), color='red', linestyle='--', label='75th pct')
ax4.set_xlabel('Feature Variance (normalized)')
ax4.set_ylabel('Count')
ax4.set_title('Feature Variance Distribution')
ax4.legend(fontsize=8)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# 1e: Batch effect overview - PCA coloured by batch
ax5 = fig.add_subplot(gs[1, 1])
pca = PCA(n_components=2)
pca_proj = pca.fit_transform(X_norm)
for b, col in [('1', '#E91E63'), ('2', '#00BCD4')]:
    mask = batch == b
    ax5.scatter(pca_proj[mask, 0], pca_proj[mask, 1], c=col, s=4, alpha=0.5, label=f'Batch {b}')
ax5.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax5.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax5.set_title('PCA (All Features) – Batch')
ax5.legend(fontsize=8, markerscale=2)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)

# 1f: PCA coloured by age
ax6 = fig.add_subplot(gs[1, 2])
sc = ax6.scatter(pca_proj[:, 0], pca_proj[:, 1], c=age, cmap=AGE_CMAP, s=4, alpha=0.5)
plt.colorbar(sc, ax=ax6, label='Age (h)')
ax6.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax6.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax6.set_title('PCA (All Features) – Age')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

plt.suptitle('Figure 1: Dataset Overview – RPE Cell Protein Imaging Data', fontsize=13, fontweight='bold', y=1.01)
plt.savefig(os.path.join(IMAGES_PATH, 'fig1_data_overview.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig1_data_overview.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Feature Scoring and Selection
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 2: Feature scoring...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 2a: Spearman correlation with age - ranked
ax = axes[0, 0]
top_n_plot = 40
top_scores = scores.nlargest(top_n_plot, 'spearman_corr')
y_pos = np.arange(top_n_plot)
colors = ['#E53935' if i < 20 else '#1E88E5' if i < 40 else '#43A047' for i in range(top_n_plot)]
ax.barh(y_pos, top_scores['spearman_corr'].values, color=colors[::-1][:top_n_plot], height=0.7)
clean_names = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
               for f in top_scores['feature'].values]
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names[::-1], fontsize=7)
ax.set_xlabel('|Spearman ρ| with Age')
ax.set_title('Top 40 Features: Spearman Correlation\nwith Cell Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2b: Mutual information with age - ranked
ax = axes[0, 1]
top_mi = scores.nlargest(top_n_plot, 'mutual_info')
clean_names_mi = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
                  for f in top_mi['feature'].values]
ax.barh(y_pos, top_mi['mutual_info'].values, color=['#7B1FA2' if i < 20 else '#AB47BC' for i in range(top_n_plot)][::-1][:top_n_plot], height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names_mi[::-1], fontsize=7)
ax.set_xlabel('Mutual Information with Age')
ax.set_title('Top 40 Features: Mutual Information\nwith Cell Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2c: Composite score - top features
ax = axes[0, 2]
top_comp = scores.head(40)
clean_names_comp = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
                    for f in top_comp['feature'].values]
colors_comp = ['#F57F17' if i < 20 else '#FFB300' for i in range(40)]
ax.barh(y_pos, top_comp['composite_score'].values, color=colors_comp[::-1][:top_n_plot], height=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(clean_names_comp[::-1], fontsize=7)
ax.set_xlabel('Composite Score')
ax.set_title('Top 40 Features: Composite Score\n(weighted combination)')
ax.axvline(top_comp['composite_score'].values[19], color='red', linestyle='--', alpha=0.7, label='Top 20 cutoff')
ax.legend(fontsize=7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2d: Correlation vs MI scatter
ax = axes[1, 0]
sc = ax.scatter(scores['spearman_corr'], scores['mutual_info'],
                c=scores['composite_score'], cmap='YlOrRd', s=20, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Composite Score')
# Highlight top 20
top20_mask = scores.index < 20
ax.scatter(scores.loc[top20_mask, 'spearman_corr'], scores.loc[top20_mask, 'mutual_info'],
           color='none', edgecolors='black', s=40, linewidths=1.2, label='Top 20')
ax.set_xlabel('|Spearman ρ| with Age')
ax.set_ylabel('Mutual Information with Age')
ax.set_title('Feature Scoring: Correlation vs\nMutual Information')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 2e: Dynamic range by phase
ax = axes[1, 1]
top20_idx = [np.where(feature_names == f)[0][0] for f in top20['feature'].values]
cycling_mask = state == 'cycling'
X_cycling = X_norm[cycling_mask]
phase_cycling = phase[cycling_mask]
phase_order_cyc = ['G1', 'S', 'G2']
top20_names_clean = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
                     for f in top20['feature'].values]

phase_mean_data = []
for ph in phase_order_cyc:
    mask = phase_cycling == ph
    if mask.sum() > 0:
        means = X_cycling[mask][:, top20_idx].mean(axis=0)
        phase_mean_data.append(means)
phase_mean_matrix = np.array(phase_mean_data)  # (3, 20)

im = ax.imshow(phase_mean_matrix, aspect='auto', cmap='RdBu_r', vmin=-1.5, vmax=1.5)
plt.colorbar(im, ax=ax, label='Mean Expression (z-score)')
ax.set_xticks(np.arange(20))
ax.set_xticklabels(top20_names_clean, rotation=90, fontsize=6.5)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(phase_order_cyc)
ax.set_title('Top 20 Features: Mean Expression\nby Cell Cycle Phase (Cycling Cells)')

# 2f: Batch effect for selected vs all features
ax = axes[1, 2]
all_batch = scores['batch_effect'].values
top20_batch = scores.head(20)['batch_effect'].values
top50_batch = scores.head(50)['batch_effect'].values

ax.violinplot([all_batch, top50_batch, top20_batch], positions=[1, 2, 3], showmedians=True)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['All (241)', 'Top 50', 'Top 20'])
ax.set_xlabel('Feature Set')
ax.set_ylabel("Batch Effect (Cohen's d)")
ax.set_title("Batch Effect Distribution\nfor Different Feature Sets")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Figure 2: Feature Scoring and Selection Methodology', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig2_feature_scoring.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig2_feature_scoring.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: UMAP Trajectory Comparison
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 3: UMAP comparison...")
fig, axes = plt.subplots(3, 4, figsize=(18, 13))

umaps = [umap_all, umap_top50, umap_best30, umap_top20]
umap_labels = ['All Features (241)', 'Top 50 Features', 'Top 30 Features', 'Top 20 Features']

for col, (umap_data, label) in enumerate(zip(umaps, umap_labels)):
    # Row 0: Colored by age
    ax = axes[0, col]
    sc = ax.scatter(umap_data[:, 0], umap_data[:, 1], c=age, cmap=AGE_CMAP, s=3, alpha=0.6)
    plt.colorbar(sc, ax=ax, label='Age (h)', shrink=0.8)
    ax.set_title(f'{label}\n(colored by age)', fontsize=9)
    ax.set_xlabel('UMAP1', fontsize=8)
    ax.set_ylabel('UMAP2', fontsize=8)
    ax.tick_params(labelsize=7)

    # Row 1: Colored by phase
    ax = axes[1, col]
    for ph in ['G0', 'G1', 'S', 'G2']:
        mask = phase == ph
        if mask.sum() > 0:
            ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                      c=PHASE_COLORS[ph], s=3, alpha=0.6, label=ph)
    ax.set_title(f'{label}\n(colored by phase)', fontsize=9)
    ax.set_xlabel('UMAP1', fontsize=8)
    ax.set_ylabel('UMAP2', fontsize=8)
    ax.tick_params(labelsize=7)
    if col == 3:
        ax.legend(fontsize=7, markerscale=2, loc='upper right')

    # Row 2: Colored by state
    ax = axes[2, col]
    state_color_map = {'cycling': '#2196F3', 'arrested': '#FF5722', 'nan': '#9E9E9E'}
    for s in ['cycling', 'arrested', 'nan']:
        mask = state == s
        if mask.sum() > 0:
            ax.scatter(umap_data[mask, 0], umap_data[mask, 1],
                      c=state_color_map[s], s=3, alpha=0.6, label=s)
    ax.set_title(f'{label}\n(colored by state)', fontsize=9)
    ax.set_xlabel('UMAP1', fontsize=8)
    ax.set_ylabel('UMAP2', fontsize=8)
    ax.tick_params(labelsize=7)
    if col == 3:
        ax.legend(fontsize=7, markerscale=2, loc='upper right')

plt.suptitle('Figure 3: UMAP Trajectory Visualization — All vs Selected Feature Sets', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig3_umap_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig3_umap_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Trajectory Quality Metrics
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 4: Trajectory metrics...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

n_feat_labels = ['All\n(241)', 'Top 50', 'Top 30', 'Top 20']
n_vals = [241, 50, 30, 20]

# Use corrected metric (sign-flipped)
knn_vals = [metrics_df[metrics_df['n_features']==f]['knn_overlap'].values[0]
            for f in ['all', '50', '30', '20']]
# dist_corr was saved as -corr, so take abs for positive "correlation preserved"
dist_vals = [abs(metrics_df[metrics_df['n_features']==f]['dist_corr'].values[0])
             for f in ['all', '50', '30', '20']]
pt_vals = [metrics_df[metrics_df['n_features']==f]['pseudotime_corr'].values[0]
           for f in ['all', '50', '30', '20']]

colors_bar = ['#9E9E9E', '#66BB6A', '#42A5F5', '#EF5350']

ax = axes[0]
bars = ax.bar(n_feat_labels, knn_vals, color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, knn_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('kNN Overlap Score')
ax.set_title('kNN Neighborhood\nPreservation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(knn_vals) * 1.2)

ax = axes[1]
bars = ax.bar(n_feat_labels, dist_vals, color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, dist_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('|Spearman ρ| (dist vs age dist)')
ax.set_title('Distance-Age\nCorrelation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(dist_vals) * 1.2)

ax = axes[2]
bars = ax.bar(n_feat_labels, pt_vals, color=colors_bar, edgecolor='white', linewidth=0.8)
for bar, v in zip(bars, pt_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f'{v:.3f}',
            ha='center', va='bottom', fontsize=9)
ax.set_ylabel('|Spearman ρ| (PC1 vs age)')
ax.set_title('Pseudotime Recovery\n(PC1 correlation)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_ylim(0, max(pt_vals) * 1.2)

plt.suptitle('Figure 4: Trajectory Quality Metrics for Different Feature Sets', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig4_trajectory_metrics.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig4_trajectory_metrics.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Selected Feature Expression Profiles
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 5: Expression profiles...")
top20_feats = top20['feature'].values
top20_idx = [np.where(feature_names == f)[0][0] for f in top20_feats]
top20_clean = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
               for f in top20_feats]

# Sort cells by age
sort_idx = np.argsort(age)
age_sorted = age[sort_idx]

fig, axes = plt.subplots(4, 5, figsize=(20, 16))

for i, (feat_idx, feat_clean) in enumerate(zip(top20_idx, top20_clean)):
    ax = axes[i // 5, i % 5]
    expr_sorted = X_norm[sort_idx, feat_idx]

    # Scatter with age
    scatter_c = [PHASE_COLORS.get(phase[sort_idx][j], '#888') for j in range(len(sort_idx))]
    ax.scatter(age_sorted, expr_sorted, c=scatter_c, s=3, alpha=0.3)

    # Rolling mean
    window = 80
    kernel = np.ones(window) / window
    smoothed = np.convolve(expr_sorted, kernel, mode='valid')
    smooth_age = np.convolve(age_sorted, np.ones(window)/window, mode='valid')
    ax.plot(smooth_age, smoothed, 'k-', linewidth=1.5, alpha=0.8)

    r, p = stats.spearmanr(age_sorted, expr_sorted)
    ax.set_title(f'{feat_clean}\nρ={r:.2f}', fontsize=8)
    ax.set_xlabel('Age (h)', fontsize=7)
    ax.set_ylabel('Z-score', fontsize=7)
    ax.tick_params(labelsize=6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add legend
legend_patches = [mpatches.Patch(color=PHASE_COLORS[ph], label=ph) for ph in ['G0', 'G1', 'S', 'G2']]
fig.legend(handles=legend_patches, loc='lower right', fontsize=10, ncol=4, title='Phase')

plt.suptitle('Figure 5: Expression Profiles of Top 20 Selected Features Along Cell Age', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig5_expression_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig5_expression_profiles.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Feature Correlation Structure and Subcategories
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 6: Correlation heatmap...")
top30_feats = top30['feature'].values
top30_idx = [np.where(feature_names == f)[0][0] for f in top30_feats]
top30_clean = [f.replace('Int_Med_', '').replace('Int_MeanEdge_', 'ME_').replace('Int_Std_', 'Std_').replace('Int_Intg_', 'Intg_')
               for f in top30_feats]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Correlation matrix of top 30 features
X_top30_mat = X_norm[:, top30_idx]
corr_mat = np.corrcoef(X_top30_mat.T)

ax = axes[0]
mask_upper = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
im = ax.imshow(corr_mat, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Pearson r', shrink=0.8)
ax.set_xticks(np.arange(30))
ax.set_xticklabels(top30_clean, rotation=90, fontsize=6.5)
ax.set_yticks(np.arange(30))
ax.set_yticklabels(top30_clean, fontsize=6.5)
ax.set_title('Pairwise Correlation Heatmap\nTop 30 Selected Features')

# PCA scree plot + cumulative variance
ax = axes[1]
pca_sub = PCA().fit(X_norm[:, top30_idx])
pca_all = PCA().fit(X_norm)

var_sub = np.cumsum(pca_sub.explained_variance_ratio_)
var_all = np.cumsum(pca_all.explained_variance_ratio_[:30])

ax2_twin = ax.twinx()
ax.bar(np.arange(1, 31), pca_sub.explained_variance_ratio_ * 100,
       color='steelblue', alpha=0.7, label='Top 30 features')
ax.set_xlabel('Principal Component')
ax.set_ylabel('Explained Variance (%)')
ax2_twin.plot(np.arange(1, 31), var_sub * 100, 'b-o', markersize=4, label='Cumulative (top 30)')
ax2_twin.plot(np.arange(1, 31), var_all * 100, 'r--o', markersize=4, label='Cumulative (all features)')
ax2_twin.set_ylabel('Cumulative Variance (%)')
ax2_twin.set_ylim(0, 105)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
ax.set_title('PCA Scree: Top 30 Selected Features\nvs All Features')
ax.spines['top'].set_visible(False)

plt.suptitle('Figure 6: Correlation Structure of Selected Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig6_correlation_structure.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig6_correlation_structure.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7: Detailed Trajectory Analysis
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 7: Detailed trajectory analysis...")
fig, axes = plt.subplots(2, 3, figsize=(17, 11))

# 7a: UMAP top 20 with age - large and detailed
ax = axes[0, 0]
sc = ax.scatter(umap_top20[:, 0], umap_top20[:, 1], c=age, cmap='plasma', s=8, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Age (h)', shrink=0.9)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_title('UMAP (Top 20 Features)\nColored by Cell Age')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7b: UMAP top 20 with phase
ax = axes[0, 1]
for ph in ['G0', 'G1', 'S', 'G2']:
    mask = phase == ph
    if mask.sum() > 0:
        ax.scatter(umap_top20[mask, 0], umap_top20[mask, 1],
                  c=PHASE_COLORS[ph], s=8, alpha=0.7, label=ph)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_title('UMAP (Top 20 Features)\nColored by Cell Cycle Phase')
ax.legend(fontsize=9, markerscale=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7c: UMAP top 20 with state
ax = axes[0, 2]
state_color_map = {'cycling': '#2196F3', 'arrested': '#FF5722', 'nan': '#9E9E9E'}
for s in ['cycling', 'arrested', 'nan']:
    mask = state == s
    if mask.sum() > 0:
        label_name = {'cycling': 'Cycling', 'arrested': 'Arrested', 'nan': 'Unassigned'}[s]
        ax.scatter(umap_top20[mask, 0], umap_top20[mask, 1],
                  c=state_color_map[s], s=8, alpha=0.7, label=label_name)
ax.set_xlabel('UMAP1')
ax.set_ylabel('UMAP2')
ax.set_title('UMAP (Top 20 Features)\nColored by Cell State')
ax.legend(fontsize=9, markerscale=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7d: 1D PCA trajectory comparison - top 20 vs all
ax = axes[1, 0]
pca1_all = PCA(n_components=1).fit_transform(X_norm).flatten()
pca1_top20 = PCA(n_components=1).fit_transform(X_norm[:, top20_idx]).flatten()

ax.scatter(age, np.abs(pca1_all), c='gray', s=4, alpha=0.3, label='All features')
ax.scatter(age, np.abs(pca1_top20), c='#EF5350', s=4, alpha=0.3, label='Top 20')

r_all, _ = stats.spearmanr(age, pca1_all)
r_top20, _ = stats.spearmanr(age, pca1_top20)
ax.set_xlabel('Annotated Age (h)')
ax.set_ylabel('|PC1 Score|')
ax.set_title(f'Pseudotime Recovery via PC1\nAll: ρ={abs(r_all):.3f} | Top 20: ρ={abs(r_top20):.3f}')
ax.legend(fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 7e: Feature category breakdown of top 20
ax = axes[1, 1]
def get_category(feat_name):
    if '_nuc' in feat_name: return 'Nuclear'
    elif '_cyto' in feat_name: return 'Cytoplasmic'
    elif '_ring' in feat_name: return 'Ring'
    elif '_cell' in feat_name: return 'Whole Cell'
    else: return 'Other'

def get_measurement(feat_name):
    if 'Int_Med_' in feat_name: return 'Median'
    elif 'Int_MeanEdge_' in feat_name: return 'Mean Edge'
    elif 'Int_Std_' in feat_name: return 'Std Dev'
    elif 'Int_Intg_' in feat_name: return 'Integrated'
    elif 'AreaShape_' in feat_name: return 'Morphology'
    else: return 'Other'

top20_cats = [get_category(f) for f in top20_feats]
cat_counts = pd.Series(top20_cats).value_counts()
colors_cat = {'Nuclear': '#3F51B5', 'Cytoplasmic': '#4CAF50', 'Ring': '#FF9800', 'Whole Cell': '#E91E63', 'Other': '#9E9E9E'}
ax.pie(cat_counts.values, labels=cat_counts.index,
       colors=[colors_cat.get(c, '#9E9E9E') for c in cat_counts.index],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Top 20 Features:\nCellular Compartment Distribution')

# 7f: Protein category breakdown
ax = axes[1, 2]
def get_protein_class(feat_name):
    name = feat_name.split('_')[-2] if len(feat_name.split('_')) > 2 else feat_name
    if name in ['cycA', 'cycB1', 'cycD1', 'cycE']: return 'Cyclins'
    elif name in ['CDK2', 'CDK4', 'CDK6']: return 'CDKs'
    elif name in ['pRB', 'RB', 'E2F1']: return 'RB pathway'
    elif name in ['PCNA', 'Cdt1', 'Skp2']: return 'DNA replic.'
    elif name in ['p21', 'p27', 'p16', 'p53', 'p14ARF']: return 'CKI/p53'
    elif name in ['pH2AX', 'pCHK1']: return 'DNA damage'
    elif 'DNA' in name: return 'DNA content'
    elif 'Area' in name or 'Intg' in name: return 'Morphology'
    else: return 'Signaling'

top20_prot = [get_protein_class(f) for f in top20_feats]
prot_counts = pd.Series(top20_prot).value_counts()
colors_prot = {'Cyclins': '#E53935', 'CDKs': '#F57C00', 'RB pathway': '#7B1FA2',
               'DNA replic.': '#0288D1', 'CKI/p53': '#388E3C', 'DNA damage': '#FBC02D',
               'DNA content': '#00796B', 'Morphology': '#5D4037', 'Signaling': '#616161'}
ax.pie(prot_counts.values, labels=prot_counts.index,
       colors=[colors_prot.get(c, '#9E9E9E') for c in prot_counts.index],
       autopct='%1.1f%%', startangle=90)
ax.set_title('Top 20 Features:\nProtein Functional Class')

plt.suptitle('Figure 7: Detailed Trajectory Analysis with Top 20 Selected Features', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig7_trajectory_detail.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig7_trajectory_detail.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 8: Key Feature Expression Heatmap along Trajectory
# ─────────────────────────────────────────────────────────────────────────────
print("  Figure 8: Heatmap along trajectory...")
# Sort cycling cells by age
cyc_mask = state == 'cycling'
X_cyc = X_norm[cyc_mask]
age_cyc = age[cyc_mask]
phase_cyc = phase[cyc_mask]
sort_cyc = np.argsort(age_cyc)

# Select top 20 features for cycling cells
X_cyc_sorted = X_cyc[sort_cyc][:, top20_idx]
age_cyc_sorted = age_cyc[sort_cyc]
phase_cyc_sorted = phase_cyc[sort_cyc]

# Smooth heatmap using rolling window
window = 50
X_smooth_list = []
for j in range(20):
    kernel = np.ones(window) / window
    smoothed = np.convolve(X_cyc_sorted[:, j], kernel, mode='valid')
    X_smooth_list.append(smoothed)
X_smooth = np.array(X_smooth_list)
age_smooth = np.convolve(age_cyc_sorted, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 6]})

# Top row: phase colorbar
ax_phase = axes[0]
phase_cyc_smooth_idx = np.convolve(np.arange(len(age_cyc_sorted)), np.ones(window)/window, mode='valid').astype(int)
phase_smooth = phase_cyc_sorted[np.round(np.linspace(0, len(age_cyc_sorted)-1, len(age_smooth))).astype(int)]
phase_color_arr = np.array([[int(c[1:3], 16)/255, int(c[3:5], 16)/255, int(c[5:7], 16)/255]
                             for c in [PHASE_COLORS.get(p, '#888888') for p in phase_smooth]])
ax_phase.imshow(phase_color_arr.reshape(1, -1, 3), aspect='auto', interpolation='nearest')
ax_phase.set_xticks([])
ax_phase.set_yticks([0])
ax_phase.set_yticklabels(['Phase'])
# Add legend patches manually
legend_elements = [mpatches.Patch(facecolor=PHASE_COLORS[ph], label=ph) for ph in ['G1', 'S', 'G2']]
ax_phase.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=9)
ax_phase.set_title('Top 20 Features Expression Along Cell Cycle Trajectory (Cycling Cells)')

# Main heatmap
ax_heat = axes[1]
im = ax_heat.imshow(X_smooth, aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2, interpolation='bilinear')
plt.colorbar(im, ax=ax_heat, label='Z-score', shrink=0.8)
ax_heat.set_yticks(np.arange(20))
ax_heat.set_yticklabels(top20_clean, fontsize=8)

# X-axis: age ticks
n_ticks = 6
tick_positions = np.linspace(0, X_smooth.shape[1]-1, n_ticks).astype(int)
tick_ages = [f'{age_smooth[t]:.1f}h' for t in tick_positions]
ax_heat.set_xticks(tick_positions)
ax_heat.set_xticklabels(tick_ages)
ax_heat.set_xlabel('Cell Age (hours)')
ax_heat.set_ylabel('Selected Feature')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_PATH, 'fig8_expression_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("    Saved fig8_expression_heatmap.png")

print("\n[DONE] All figures generated and saved to report/images/")
print("Files:")
for f in sorted(os.listdir(IMAGES_PATH)):
    size = os.path.getsize(os.path.join(IMAGES_PATH, f))
    print(f"  {f} ({size/1024:.1f} KB)")
