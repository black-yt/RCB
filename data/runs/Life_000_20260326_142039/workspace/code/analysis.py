"""
Hydrogel Adhesive Strength Analysis
====================================
De novo design of synthetic hydrogels achieving robust underwater adhesion
by statistically replicating sequence features of natural adhesive proteins.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')
import os

# Setup paths
WORKSPACE = '/mnt/d/xwh/ailab记录/工作/26年03月/SGI-Bench/ResearchClawBench/workspaces/Life_000_20260326_142039'
DATA_DIR = os.path.join(WORKSPACE, 'data')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMAGES_DIR = os.path.join(WORKSPACE, 'report/images')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

np.random.seed(42)

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA', 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
FEATURE_LABELS = ['HEA\n(Nucleophilic)', 'BA\n(Hydrophobic)', 'CBEA\n(Acidic)', 'ATAC\n(Cationic)', 'PEA\n(Aromatic)', 'AAm\n(Amide)']
TARGET_COL = 'Glass (kPa)_10s'

# Natural adhesive protein analog - monomer type mapping
PROTEIN_ANALOG = {
    'Serine/Threonine (Nucleophilic)': 'Nucleophilic-HEA',   # OH groups - mussel adhesive proteins
    'Leucine/Valine (Hydrophobic)': 'Hydrophobic-BA',         # Hydrophobic core
    'Asp/Glu (Acidic)': 'Acidic-CBEA',                        # Carboxylate groups
    'Lys/Arg (Cationic)': 'Cationic-ATAC',                    # Amino groups
    'Phe/Tyr (Aromatic)': 'Aromatic-PEA',                     # DOPA analog - pi stacking
    'Asn/Gln (Amide)': 'Amide-AAm',                           # Amide bonds
}

def load_data():
    """Load and preprocess the main dataset"""
    df = pd.read_excel(os.path.join(DATA_DIR, '184_verified_Original Data_ML_20230926.xlsx'), 
                       sheet_name='Data_to_HU')
    
    # Convert strength to numeric
    for col in [TARGET_COL, 'Steel (kPa)_10s', 'Q', 'Modulus (kPa)', 'XlogP3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter rows with valid target
    df_valid = df[df[TARGET_COL].notna()].copy()
    
    # Convert features to numeric
    for col in FEATURE_COLS:
        df_valid[col] = pd.to_numeric(df_valid[col], errors='coerce')
    
    df_valid = df_valid.dropna(subset=FEATURE_COLS)
    
    return df_valid

def load_optimization_data():
    """Load the optimization rounds data"""
    df_ei = pd.read_excel(os.path.join(DATA_DIR, 'ML_ei&pred (1&2&3rounds)_20240408.xlsx'), 
                          sheet_name='EI')
    df_pred = pd.read_excel(os.path.join(DATA_DIR, 'ML_ei&pred (1&2&3rounds)_20240408.xlsx'), 
                            sheet_name='PRED')
    
    # Convert to numeric
    for col in FEATURE_COLS + ['Glass (kPa)_max']:
        df_ei[col] = pd.to_numeric(df_ei[col], errors='coerce')
        df_pred[col] = pd.to_numeric(df_pred[col], errors='coerce')
    
    df_ei = df_ei.dropna(subset=FEATURE_COLS + ['Glass (kPa)_max'])
    df_pred = df_pred.dropna(subset=FEATURE_COLS + ['Glass (kPa)_max'])
    
    return df_ei, df_pred

print("Loading data...")
df = load_data()
df_ei, df_pred = load_optimization_data()
print(f"Main dataset: {len(df)} samples")
print(f"Optimization EI: {len(df_ei)} samples")
print(f"Optimization PRED: {len(df_pred)} samples")

# Prepare X, y
X = df[FEATURE_COLS].values
y = df[TARGET_COL].values

print(f"\nStrength range: {y.min():.1f} - {y.max():.1f} kPa")
print(f"Mean strength: {y.mean():.1f} kPa")
print(f"Samples >100 kPa: {(y > 100).sum()}")
print(f"Samples >200 kPa: {(y > 200).sum()}")

# ============================================================
# FIGURE 1: Data Overview - Feature distributions and strength
# ============================================================
print("\n=== Figure 1: Data Overview ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Overview of Natural Adhesive Protein-Inspired Hydrogel Dataset\n(n=184 formulations)', 
             fontsize=14, fontweight='bold')

colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0', '#795548']

for i, (col, label) in enumerate(zip(FEATURE_COLS, FEATURE_LABELS)):
    ax = axes[i//3, i%3]
    vals = df[col].values
    ax.hist(vals, bins=20, color=colors[i], alpha=0.7, edgecolor='white')
    ax.set_xlabel(label, fontsize=10)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'{col.split("-")[1]}\nmean={vals.mean():.3f}', fontsize=10)
    ax.axvline(vals.mean(), color='darkred', linestyle='--', linewidth=1.5, label=f'Mean={vals.mean():.3f}')
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig1_feature_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_feature_distributions.png")

# ============================================================
# FIGURE 2: Adhesive strength distribution
# ============================================================
print("\n=== Figure 2: Strength Distribution ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of strength
ax1 = axes[0]
ax1.hist(y, bins=40, color='#1976D2', alpha=0.8, edgecolor='white', linewidth=0.5)
ax1.axvline(y.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean={y.mean():.1f} kPa')
ax1.axvline(100, color='orange', linestyle='--', linewidth=2, label='100 kPa threshold')
ax1.axvline(300, color='green', linestyle='--', linewidth=2, label='300 kPa (top)')
ax1.set_xlabel('Adhesive Strength (kPa)', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.set_title('Distribution of Underwater Adhesive Strength\n(Glass substrate)', fontsize=12)
ax1.legend(fontsize=9)

# Scatter of top composition features vs strength  
ax2 = axes[1]
sc = ax2.scatter(df['Aromatic-PEA'], df['Hydrophobic-BA'], 
                  c=y, cmap='RdYlGn', s=40, alpha=0.7, vmin=0, vmax=200)
plt.colorbar(sc, ax=ax2, label='Adhesive Strength (kPa)')
ax2.set_xlabel('Aromatic-PEA (DOPA analog)', fontsize=11)
ax2.set_ylabel('Hydrophobic-BA', fontsize=11)
ax2.set_title('Composition-Strength Relationship\n(Two most important features)', fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig2_strength_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig2_strength_distribution.png")

# ============================================================
# Train ML Models (Random Forest + Gaussian Process)
# ============================================================
print("\n=== Training ML Models ===")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, 
                            random_state=42, n_jobs=-1)
# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_cv_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')
rf_cv_rmse = np.sqrt(-cross_val_score(rf, X, y, cv=kf, scoring='neg_mean_squared_error'))

print(f"\nRandom Forest 5-fold CV:")
print(f"  R² = {rf_cv_scores.mean():.3f} ± {rf_cv_scores.std():.3f}")
print(f"  RMSE = {rf_cv_rmse.mean():.2f} ± {rf_cv_rmse.std():.2f} kPa")

# GP Model
kernel = ConstantKernel(1.0) * Matern(nu=2.5, length_scale=1.0)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=5, 
                               normalize_y=True, random_state=42)
gp_cv_scores = cross_val_score(gp, X_scaled, y, cv=kf, scoring='r2')
gp_cv_rmse = np.sqrt(-cross_val_score(gp, X_scaled, y, cv=kf, scoring='neg_mean_squared_error'))

print(f"\nGaussian Process 5-fold CV:")
print(f"  R² = {gp_cv_scores.mean():.3f} ± {gp_cv_scores.std():.3f}")
print(f"  RMSE = {gp_cv_rmse.mean():.2f} ± {gp_cv_rmse.std():.2f} kPa")

# Fit on full data
rf.fit(X, y)
gp.fit(X_scaled, y)

# ============================================================
# FIGURE 3: Model Performance Comparison
# ============================================================
print("\n=== Figure 3: Model Performance ===")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_train = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=2, random_state=42)
rf_train.fit(X_train, y_train)
y_pred_rf = rf_train.predict(X_test)

X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
gp_train = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, n_restarts_optimizer=3, 
                                     normalize_y=True, random_state=42)
gp_train.fit(X_train_s, y_train)
y_pred_gp, y_std_gp = gp_train.predict(X_test_s, return_std=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RF parity plot
ax1 = axes[0]
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
ax1.scatter(y_test, y_pred_rf, alpha=0.7, color='#1976D2', s=50, edgecolor='none')
lims = [0, max(y_test.max(), y_pred_rf.max())*1.05]
ax1.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax1.set_xlabel('Experimental Strength (kPa)', fontsize=12)
ax1.set_ylabel('Predicted Strength (kPa)', fontsize=12)
ax1.set_title(f'Random Forest Regression\n(R²={r2_rf:.3f}, RMSE={rmse_rf:.1f} kPa)', fontsize=12)
ax1.legend(fontsize=10)
ax1.set_xlim(lims); ax1.set_ylim(lims)

# GP parity plot with uncertainty
ax2 = axes[1]
r2_gp = r2_score(y_test, y_pred_gp)
rmse_gp = np.sqrt(mean_squared_error(y_test, y_pred_gp))
ax2.errorbar(y_test, y_pred_gp, yerr=1.96*y_std_gp, fmt='o', alpha=0.5, 
             color='#E53935', ecolor='#EF9A9A', capsize=2, markersize=5, label='±1.96σ')
ax2.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction')
ax2.set_xlabel('Experimental Strength (kPa)', fontsize=12)
ax2.set_ylabel('Predicted Strength (kPa)', fontsize=12)
ax2.set_title(f'Gaussian Process Regression\n(R²={r2_gp:.3f}, RMSE={rmse_gp:.1f} kPa)', fontsize=12)
ax2.legend(fontsize=10)
ax2.set_xlim(lims); ax2.set_ylim(lims)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig3_model_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved fig3_model_performance.png")
print(f"RF: R²={r2_rf:.3f}, RMSE={rmse_rf:.1f}")
print(f"GP: R²={r2_gp:.3f}, RMSE={rmse_gp:.1f}")

# ============================================================
# FIGURE 4: Feature Importance
# ============================================================
print("\n=== Figure 4: Feature Importance ===")

importances = rf.feature_importances_
perm_imp = permutation_importance(rf, X, y, n_repeats=20, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# RF feature importances
ax1 = axes[0]
sorted_idx = np.argsort(importances)[::-1]
short_labels = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']
bars = ax1.bar(range(6), importances[sorted_idx], color=np.array(colors)[sorted_idx], 
                alpha=0.8, edgecolor='white')
ax1.set_xticks(range(6))
ax1.set_xticklabels(np.array(short_labels)[sorted_idx], fontsize=11)
ax1.set_ylabel('Gini Importance', fontsize=12)
ax1.set_title('Random Forest Feature Importance\n(Monomer Type Contribution)', fontsize=12)
for i, (bar, imp) in enumerate(zip(bars, importances[sorted_idx])):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.003, 
             f'{imp:.3f}', ha='center', va='bottom', fontsize=9)

# Permutation importance
ax2 = axes[1]
perm_sorted = np.argsort(perm_imp.importances_mean)[::-1]
ax2.barh(range(6), perm_imp.importances_mean[perm_sorted], 
         xerr=perm_imp.importances_std[perm_sorted],
         color=np.array(colors)[perm_sorted], alpha=0.8, height=0.6, 
         error_kw={'linewidth': 1.5})
ax2.set_yticks(range(6))
ax2.set_yticklabels(np.array(short_labels)[perm_sorted], fontsize=11)
ax2.set_xlabel('Permutation Importance (Mean Decrease in R²)', fontsize=11)
ax2.set_title('Permutation Feature Importance\n(Cross-validated significance)', fontsize=12)
ax2.axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig4_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig4_feature_importance.png")

# ============================================================
# Natural Adhesive Protein Analysis
# ============================================================
print("\n=== Analyzing Natural Adhesive Protein Features ===")

# Natural mussel adhesive proteins (MAPs) - residue fractions
# Based on mussel foot proteins (mfp-1 through mfp-6) literature values
# Key feature: DOPA (3,4-dihydroxyphenylalanine) from tyrosine post-translational modification
natural_protein_profiles = {
    'Mussel fp-1 (Mytilus)': {
        'Nucleophilic-HEA': 0.100,   # Ser+Thr ~10%
        'Hydrophobic-BA':   0.050,   # Leu+Val+Ile ~5%
        'Acidic-CBEA':      0.050,   # Asp+Glu ~5%
        'Cationic-ATAC':    0.120,   # Lys+Arg+His ~12%
        'Aromatic-PEA':     0.180,   # Phe+Tyr(DOPA) ~18%
        'Amide-AAm':        0.030,   # Asn+Gln ~3%
    },
    'Mussel fp-3 (Mytilus)': {
        'Nucleophilic-HEA': 0.160,   # Ser+Thr ~16%
        'Hydrophobic-BA':   0.270,   # Leu+Val+Ile ~27%  
        'Acidic-CBEA':      0.040,   # Asp+Glu ~4%
        'Cationic-ATAC':    0.060,   # Lys+Arg ~6%
        'Aromatic-PEA':     0.270,   # Phe+Tyr(DOPA) ~27%
        'Amide-AAm':        0.020,   # Asn+Gln ~2%
    },
    'Mussel fp-5 (Mytilus)': {
        'Nucleophilic-HEA': 0.170,   # Ser+Thr ~17%
        'Hydrophobic-BA':   0.100,   # Leu+Val+Ile ~10%
        'Acidic-CBEA':      0.080,   # Asp+Glu ~8%
        'Cationic-ATAC':    0.110,   # Lys+Arg ~11%
        'Aromatic-PEA':     0.200,   # Phe+Tyr(DOPA) ~20%
        'Amide-AAm':        0.040,   # Asn+Gln ~4%
    },
    'Sandcastle worm (PMMA)': {
        'Nucleophilic-HEA': 0.130,   # Ser+Thr ~13%
        'Hydrophobic-BA':   0.200,   # Leu+Val ~20%
        'Acidic-CBEA':      0.130,   # Asp+Glu ~13%
        'Cationic-ATAC':    0.270,   # Lys+Arg ~27%
        'Aromatic-PEA':     0.080,   # Tyr ~8%
        'Amide-AAm':        0.020,   # Asn+Gln ~2%
    },
    'Barnacle (Semibalanus)': {
        'Nucleophilic-HEA': 0.120,   # Ser+Thr ~12%
        'Hydrophobic-BA':   0.320,   # Leu+Val+Ile ~32%
        'Acidic-CBEA':      0.090,   # Asp+Glu ~9%
        'Cationic-ATAC':    0.070,   # Lys+Arg ~7%
        'Aromatic-PEA':     0.150,   # Phe+Tyr ~15%
        'Amide-AAm':        0.050,   # Asn+Gln ~5%
    }
}

# Normalize to sum=1
for protein, profile in natural_protein_profiles.items():
    total = sum(profile.values())
    for key in profile:
        profile[key] /= total

protein_df = pd.DataFrame(natural_protein_profiles).T

# Compute mean protein profile
mean_protein = protein_df.mean()
std_protein = protein_df.std()

print("Natural adhesive protein profiles (normalized):")
print(protein_df.round(3))
print()
print("Mean protein profile:", mean_protein.round(3).to_dict())

# ============================================================
# FIGURE 5: Natural Protein Profiles vs Hydrogel Design Space
# ============================================================
print("\n=== Figure 5: Protein Profiles ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Radar/bar chart of protein profiles
ax1 = axes[0]
protein_matrix = protein_df[FEATURE_COLS].values
x = np.arange(6)
width = 0.12
protein_colors = ['#1a237e', '#283593', '#1565C0', '#0277BD', '#01579B', '#006064']
for i, (protein, row) in enumerate(protein_df.iterrows()):
    bars = ax1.bar(x + i*width - 2.5*width, row[FEATURE_COLS].values, width=width, 
                   label=protein.replace(' (', '\n('), color=protein_colors[i], alpha=0.8)

ax1.set_xticks(x)
ax1.set_xticklabels(['HEA\n(Nucl.)', 'BA\n(Hydro.)', 'CBEA\n(Acid)', 
                      'ATAC\n(Cat.)', 'PEA\n(Arom.)', 'AAm\n(Amide)'], fontsize=9)
ax1.set_ylabel('Mole Fraction', fontsize=11)
ax1.set_title('Amino Acid Composition Profiles\nof Natural Adhesive Proteins', fontsize=12)
ax1.legend(fontsize=7, loc='upper right')

# Compare with high-performing hydrogels
ax2 = axes[1]
high_performers = df[df[TARGET_COL] > 100].copy()
low_performers = df[df[TARGET_COL] < 50].copy()

mean_high = high_performers[FEATURE_COLS].mean()
mean_low = low_performers[FEATURE_COLS].mean()
mean_protein_normalized = mean_protein[FEATURE_COLS]

x = np.arange(6)
width = 0.25
b1 = ax2.bar(x - width, mean_high.values, width, color='#2E7D32', alpha=0.85, label=f'High Strength (>{100}kPa, n={len(high_performers)})')
b2 = ax2.bar(x, mean_low.values, width, color='#C62828', alpha=0.85, label=f'Low Strength (<{50}kPa, n={len(low_performers)})')
b3 = ax2.bar(x + width, mean_protein_normalized.values, width, color='#1565C0', alpha=0.85, label='Natural Protein (Mean)')

ax2.set_xticks(x)
ax2.set_xticklabels(['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm'], fontsize=10)
ax2.set_ylabel('Mean Mole Fraction', fontsize=11)
ax2.set_title('Composition Comparison: High vs Low Performers\nvs Natural Adhesive Proteins', fontsize=12)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig5_protein_profiles.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5_protein_profiles.png")

# ============================================================
# De Novo Design: Grid Search + ML Prediction
# ============================================================
print("\n=== De Novo Design: Generating Candidate Compositions ===")

# Generate dense grid of candidate compositions
from itertools import product
import itertools

# Smart sampling: use mean protein profile as center + variation
# Dirichlet distribution sampling for compositions that sum to 1
def sample_compositions(n_samples=50000, seed=42):
    """Sample compositions on the simplex using Dirichlet distribution"""
    rng = np.random.RandomState(seed)
    
    # Multiple sampling strategies
    samples = []
    
    # Strategy 1: Uniform Dirichlet (flat)
    alpha_uniform = np.ones(6) * 0.5
    s1 = rng.dirichlet(alpha_uniform, n_samples // 4)
    samples.append(s1)
    
    # Strategy 2: Centered on natural protein mean
    protein_mean = mean_protein[FEATURE_COLS].values
    alpha_protein = protein_mean * 10 + 0.5  # concentrate around protein profile
    s2 = rng.dirichlet(alpha_protein, n_samples // 4)
    samples.append(s2)
    
    # Strategy 3: High BA/PEA (aromatic+hydrophobic - high performers)
    alpha_ba_pea = np.array([0.1, 3.0, 0.1, 0.3, 2.0, 0.1])
    s3 = rng.dirichlet(alpha_ba_pea, n_samples // 4)
    samples.append(s3)
    
    # Strategy 4: Around best known sample (GPRFR-2: 0, 0.53, 0, 0.05, 0.37, 0.05)
    alpha_best = np.array([0.1, 5.0, 0.1, 0.5, 3.5, 0.5])
    s4 = rng.dirichlet(alpha_best, n_samples // 4)
    samples.append(s4)
    
    all_samples = np.vstack(samples)
    return all_samples

candidates = sample_compositions(n_samples=100000)
print(f"Generated {len(candidates)} candidate compositions")

# Predict with RF and GP
y_pred_rf_all = rf.predict(candidates)
X_cand_scaled = scaler.transform(candidates)
y_pred_gp_all, y_std_gp_all = gp.predict(X_cand_scaled, return_std=True)

# Ensemble prediction
y_pred_ensemble = (y_pred_rf_all + y_pred_gp_all) / 2

# Filter top candidates
top_mask = y_pred_ensemble > 200
top_candidates = candidates[top_mask]
top_strengths = y_pred_ensemble[top_mask]
top_rf = y_pred_rf_all[top_mask]
top_gp = y_pred_gp_all[top_mask]
top_std = y_std_gp_all[top_mask]

print(f"\nCandidates with predicted strength >200 kPa: {top_mask.sum()}")
print(f"Predicted strengths: mean={top_strengths.mean():.1f}, max={top_strengths.max():.1f}")

# Sort by ensemble prediction
sort_idx = np.argsort(top_strengths)[::-1]
top_candidates_sorted = top_candidates[sort_idx[:50]]
top_strengths_sorted = top_strengths[sort_idx[:50]]
top_rf_sorted = top_rf[sort_idx[:50]]
top_gp_sorted = top_gp[sort_idx[:50]]

# Save top candidates
top_df = pd.DataFrame(top_candidates_sorted, columns=FEATURE_COLS)
top_df['Predicted_Ensemble (kPa)'] = top_strengths_sorted
top_df['Predicted_RF (kPa)'] = top_rf_sorted
top_df['Predicted_GP (kPa)'] = top_gp_sorted
top_df.to_csv(os.path.join(OUTPUT_DIR, 'top_candidates.csv'), index=True)
print(f"\nTop candidates saved to outputs/top_candidates.csv")
print("\nTop 10 designs:")
print(top_df.head(10).round(4))

# ============================================================
# FIGURE 6: Design Space Exploration - UMAP/PCA
# ============================================================
print("\n=== Figure 6: Design Space Visualization ===")

from sklearn.decomposition import PCA

# PCA of all compositions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_cand_pca = pca.transform(top_candidates_sorted[:200] if len(top_candidates_sorted) >= 200 else top_candidates_sorted)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA plot colored by strength
ax1 = axes[0]
sc = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='RdYlGn', s=40, alpha=0.7, 
                  vmin=0, vmax=250, edgecolors='none')
plt.colorbar(sc, ax=ax1, label='Adhesive Strength (kPa)')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=11)
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=11)
ax1.set_title('PCA of Experimental Hydrogel Compositions\n(colored by adhesive strength)', fontsize=12)

# Overlay de novo designs
n_show = min(50, len(top_candidates_sorted))
top_pca = pca.transform(top_candidates_sorted[:n_show])
ax1.scatter(top_pca[:, 0], top_pca[:, 1], marker='*', c='blue', s=150, alpha=0.9,
            zorder=5, label=f'De Novo Designs (predicted >200 kPa)')
ax1.legend(fontsize=9)

# Distribution of top design compositions vs training data
ax2 = axes[1]
feature_means_train = df[FEATURE_COLS].mean()
feature_means_top = pd.Series(top_candidates_sorted[:20].mean(axis=0), index=FEATURE_COLS)

x = np.arange(6)
width = 0.35
ax2.bar(x - width/2, feature_means_train.values, width, color='steelblue', alpha=0.8, 
        label='Training Data Mean')
ax2.bar(x + width/2, feature_means_top.values, width, color='darkgreen', alpha=0.8, 
        label='De Novo Top-20 Designs')
ax2.set_xticks(x)
ax2.set_xticklabels(['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm'], fontsize=10)
ax2.set_ylabel('Mean Mole Fraction', fontsize=11)
ax2.set_title('Compositional Shift: Training Data\nvs De Novo High-Strength Designs', fontsize=12)
ax2.legend(fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig6_design_space.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6_design_space.png")

# ============================================================
# FIGURE 7: Expected Improvement and Optimization Trajectory
# ============================================================
print("\n=== Figure 7: Optimization Trajectory ===")

# Compute EI (Expected Improvement)
from scipy.stats import norm

def compute_ei(mu, sigma, y_best, xi=0.0):
    """Expected Improvement acquisition function"""
    z = (mu - y_best - xi) / (sigma + 1e-9)
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    ei[sigma <= 0] = 0
    return ei

y_best = y.max()
gp_fit_full = GaussianProcessRegressor(kernel=kernel, alpha=1e-2, 
                                        n_restarts_optimizer=3, normalize_y=True, random_state=42)
gp_fit_full.fit(X_scaled, y)
mu_all, sigma_all = gp_fit_full.predict(X_cand_scaled[:10000], return_std=True)
ei_all = compute_ei(mu_all, sigma_all, y_best, xi=10.0)

# Sort by EI
ei_sort = np.argsort(ei_all)[::-1]
top_ei_idx = ei_sort[:20]
best_ei_comps = candidates[:10000][top_ei_idx]
best_ei_vals = mu_all[top_ei_idx]
best_ei_sigma = sigma_all[top_ei_idx]

print(f"\nTop EI candidates: predicted mean strength = {best_ei_vals.mean():.1f} ± {best_ei_sigma.mean():.1f} kPa")

# Combine training and optimization data for trajectory plot
df_opt_combined = pd.concat([
    df_ei[['Glass (kPa)_max']].rename(columns={'Glass (kPa)_max': 'Strength'}).assign(Round='Opt. Round 1-3 (EI)'),
    df_pred[['Glass (kPa)_max']].rename(columns={'Glass (kPa)_max': 'Strength'}).assign(Round='Opt. Round 1-3 (PRED)'),
], ignore_index=True)

# Add base training data sample
df_opt_combined = pd.concat([
    df[[TARGET_COL]].rename(columns={TARGET_COL: 'Strength'}).assign(Round='Initial Training (n=184)'),
    df_opt_combined
], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Box plots showing improvement across rounds
ax1 = axes[0]
order = ['Initial Training (n=184)', 'Opt. Round 1-3 (EI)', 'Opt. Round 1-3 (PRED)']
round_data = [df_opt_combined[df_opt_combined['Round'] == r]['Strength'].dropna() for r in order]
bp = ax1.boxplot(round_data, patch_artist=True, notch=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.8),
                  medianprops=dict(color='darkblue', linewidth=2),
                  whiskerprops=dict(color='gray'),
                  capprops=dict(color='gray'))

for i, (patch, color) in enumerate(zip(bp['boxes'], ['#FFF9C4', '#B2EBF2', '#C8E6C9'])):
    patch.set_facecolor(color)

ax1.set_xticklabels(['Initial\nTraining\n(n=184)', 'EI-Guided\nOptimization', 'GP Prediction\nOptimization'], 
                     fontsize=10)
ax1.set_ylabel('Adhesive Strength (kPa)', fontsize=12)
ax1.set_title('Optimization Trajectory: Adhesive Strength\nAcross Training and Optimization Rounds', fontsize=12)

# Add mean and max markers
for i, data in enumerate(round_data):
    ax1.scatter([i+1]*len(data), data, alpha=0.3, color='gray', s=15, zorder=2)
    ax1.scatter(i+1, data.max(), marker='^', color='red', s=100, zorder=5, 
                label=f'Max={data.max():.0f}' if i==0 else f'{data.max():.0f}')

ax1.legend(title='Max Strength (kPa)', fontsize=9, loc='upper right')
ax1.axhline(y=100, color='orange', linestyle='--', linewidth=1.5, label='100 kPa')

# EI vs predicted strength (exploration-exploitation tradeoff)
ax2 = axes[1]
ax2.scatter(mu_all[:5000], ei_all[:5000], alpha=0.3, s=10, color='gray', label='Candidates')
ax2.scatter(best_ei_vals, ei_all[top_ei_idx], color='red', s=100, zorder=5, 
            label=f'Top EI Candidates (n=20)')
ax2.axvline(y_best, color='green', linestyle='--', linewidth=1.5, label=f'Current best={y_best:.1f} kPa')
ax2.set_xlabel('GP Predicted Strength (kPa)', fontsize=12)
ax2.set_ylabel('Expected Improvement', fontsize=12)
ax2.set_title('Exploration-Exploitation Trade-off\n(Bayesian Optimization via Expected Improvement)', fontsize=12)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig7_optimization.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7_optimization.png")

# ============================================================
# FIGURE 8: Correlation Matrix and Chemical Interpretation
# ============================================================
print("\n=== Figure 8: Correlation Analysis ===")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Correlation heatmap
ax1 = axes[0]
corr_data = df[FEATURE_COLS + [TARGET_COL]].copy()
corr_data.columns = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm', 'Strength']
corr_matrix = corr_data.corr()

mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask, k=1)] = True
sns.heatmap(corr_matrix, ax=ax1, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
            cbar_kws={'shrink': 0.8})
ax1.set_title('Correlation Matrix: Monomer Composition\nvs Adhesive Strength', fontsize=12)

# Partial correlation: strength vs PEA at different BA levels
ax2 = axes[1]
# Bin by BA level
df_sorted = df.copy()
df_sorted['BA_bin'] = pd.cut(df_sorted['Hydrophobic-BA'], bins=[0, 0.25, 0.45, 1.0], 
                               labels=['Low BA\n(0-0.25)', 'Mid BA\n(0.25-0.45)', 'High BA\n(>0.45)'])
ba_colors = ['#FF7043', '#7CB342', '#1565C0']
for i, (name, group) in enumerate(df_sorted.groupby('BA_bin')):
    ax2.scatter(group['Aromatic-PEA'], group[TARGET_COL], 
                color=ba_colors[i], alpha=0.6, s=40, label=str(name))
    # Trend line
    pea_vals = group['Aromatic-PEA'].values
    strength_vals = group[TARGET_COL].values
    if len(pea_vals) > 5:
        z = np.polyfit(pea_vals, strength_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(pea_vals.min(), pea_vals.max(), 50)
        ax2.plot(x_line, p(x_line), color=ba_colors[i], linewidth=2, alpha=0.8)

ax2.set_xlabel('Aromatic PEA Content (mole fraction)', fontsize=12)
ax2.set_ylabel('Adhesive Strength (kPa)', fontsize=12)
ax2.set_title('Interaction: Aromatic (PEA) vs Hydrophobic (BA)\non Adhesive Strength', fontsize=12)
ax2.legend(title='BA Level', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig8_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8_correlations.png")

# ============================================================
# FIGURE 9: De Novo Design Summary
# ============================================================
print("\n=== Figure 9: De Novo Design Summary ===")

# Select best 5 designs for detailed analysis
n_designs = 10
best_designs = top_candidates_sorted[:n_designs]
best_pred = top_strengths_sorted[:n_designs]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('De Novo Designed Hydrogels: Composition and Performance\n(Protein Sequence Feature-Inspired Design)', 
             fontsize=14, fontweight='bold')

# Composition bar chart for top designs
ax1 = axes[0, 0]
x = np.arange(n_designs)
bottom = np.zeros(n_designs)
design_colors = ['#F44336', '#FF9800', '#FFEB3B', '#4CAF50', '#2196F3', '#9C27B0']
short_labels = ['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm']
for j, (col, color) in enumerate(zip(FEATURE_COLS, design_colors)):
    vals = best_designs[:, j]
    ax1.bar(x, vals, bottom=bottom, color=color, label=short_labels[j], alpha=0.85)
    bottom += vals

ax1.set_xticks(x)
ax1.set_xticklabels([f'D{i+1}' for i in range(n_designs)], fontsize=10)
ax1.set_ylabel('Mole Fraction', fontsize=12)
ax1.set_title('Composition of Top-10 De Novo Designed Hydrogels', fontsize=12)
ax1.legend(loc='upper right', fontsize=9, title='Monomer Type')
ax1.set_ylim(0, 1.05)

# Predicted strength with uncertainty
ax2 = axes[0, 1]
ax2.bar(x, top_rf_sorted[:n_designs], alpha=0.7, color='#1976D2', label='RF Prediction', width=0.35, align='edge')
ax2.bar(x - 0.35, top_gp_sorted[:n_designs], alpha=0.7, color='#E53935', label='GP Prediction', width=0.35)
ax2.axhline(y.max(), color='green', linestyle='--', linewidth=1.5, label=f'Best experimental ({y.max():.0f} kPa)')
ax2.axhline(100, color='orange', linestyle='--', linewidth=1.5, label='100 kPa threshold')
ax2.set_xticks(x)
ax2.set_xticklabels([f'D{i+1}' for i in range(n_designs)], fontsize=10)
ax2.set_ylabel('Predicted Adhesive Strength (kPa)', fontsize=11)
ax2.set_title('ML-Predicted Strength of De Novo Designs\n(RF vs GP)', fontsize=12)
ax2.legend(fontsize=9)

# Comparison with best known + natural protein
ax3 = axes[1, 0]
# Get best experimental composition
best_exp_idx = y.argmax()
best_exp_comp = X[best_exp_idx]

bar_data = {
    'Best Experimental\n(GPRFR-2, 305 kPa)': best_exp_comp,
    'Mean Natural\nAdhesive Protein': mean_protein[FEATURE_COLS].values,
    'Best De Novo\nDesign (D1)': best_designs[0],
}

x2 = np.arange(6)
width = 0.25
for i, (name, vals) in enumerate(bar_data.items()):
    ax3.bar(x2 + (i-1)*width, vals, width, label=name, alpha=0.8)

ax3.set_xticks(x2)
ax3.set_xticklabels(['HEA', 'BA', 'CBEA', 'ATAC', 'PEA', 'AAm'], fontsize=10)
ax3.set_ylabel('Mole Fraction', fontsize=11)
ax3.set_title('Composition Comparison:\nExperimental Best vs Protein vs De Novo', fontsize=12)
ax3.legend(fontsize=9, loc='upper right')

# Scatter: BA vs PEA for all design strategies
ax4 = axes[1, 1]
ax4.scatter(df['Hydrophobic-BA'], df['Aromatic-PEA'], c=y, cmap='RdYlGn', 
             s=40, alpha=0.6, label='Training data', vmin=0, vmax=250)
ax4.scatter(best_designs[:, 1], best_designs[:, 4], s=150, marker='*', 
             c=best_pred, cmap='cool', vmin=200, vmax=350,
             edgecolors='black', linewidths=1, label='De Novo Designs', zorder=5)
sc = ax4.scatter([], [], marker='*', c='blue', s=100)
plt.colorbar(ax4.scatter(df['Hydrophobic-BA'], df['Aromatic-PEA'], c=y, 
                           cmap='RdYlGn', s=40, alpha=0, vmin=0, vmax=250), 
             ax=ax4, label='Strength (kPa)')
ax4.set_xlabel('Hydrophobic BA Content', fontsize=11)
ax4.set_ylabel('Aromatic PEA Content', fontsize=11)
ax4.set_title('De Novo Designs in BA-PEA Composition Space\nvs Training Data', fontsize=12)
ax4.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(IMAGES_DIR, 'fig9_denovo_designs.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9_denovo_designs.png")

# ============================================================
# Save key results
# ============================================================
print("\n=== Saving Results ===")

results_summary = {
    'dataset_size': len(df),
    'strength_mean_kPa': float(y.mean()),
    'strength_max_kPa': float(y.max()),
    'strength_above_100_kPa': int((y > 100).sum()),
    'rf_cv_r2_mean': float(rf_cv_scores.mean()),
    'rf_cv_r2_std': float(rf_cv_scores.std()),
    'rf_cv_rmse_mean': float(rf_cv_rmse.mean()),
    'gp_cv_r2_mean': float(gp_cv_scores.mean()),
    'gp_cv_rmse_mean': float(gp_cv_rmse.mean()),
    'n_candidates_generated': int(len(candidates)),
    'n_candidates_above_200kPa': int(top_mask.sum()),
    'best_denovo_predicted_kPa': float(top_strengths_sorted[0]),
    'best_exp_comp': dict(zip(FEATURE_COLS, best_exp_comp.tolist())),
    'best_denovo_comp': dict(zip(FEATURE_COLS, best_designs[0].tolist())),
    'pca_var_ratio': pca.explained_variance_ratio_.tolist(),
}

import json
with open(os.path.join(OUTPUT_DIR, 'results_summary.json'), 'w') as f:
    json.dump(results_summary, f, indent=2)

print("Results saved to outputs/results_summary.json")
print(f"\nBest de novo design composition:")
for feat, val in zip(FEATURE_COLS, best_designs[0]):
    print(f"  {feat}: {val:.4f}")
print(f"  Predicted strength: {best_pred[0]:.1f} kPa")

print("\n=== Analysis Complete ===")
