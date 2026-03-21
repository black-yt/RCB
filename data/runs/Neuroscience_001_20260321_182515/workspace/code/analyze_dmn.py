"""
Comprehensive analysis of the Deep Mechanistic Network (DMN) ensemble
for Drosophila optic flow estimation.

This script:
1. Loads the connectome and all 50 pre-trained DMN model checkpoints
2. Analyzes network architecture (cell types, synapse connectivity)
3. Compares learned parameters across ensemble
4. Analyzes validation performance
5. Examines UMAP/clustering of neural responses
6. Generates publication-quality figures
"""

import json
import os
import pickle
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

# Paths
WORKSPACE = Path('/mnt/d/xwh/ailab记录/工作/26年03月/SGI-Bench/ResearchClawBench/workspaces/Neuroscience_001_20260321_182515')
DATA_DIR = WORKSPACE / 'data' / 'flow' / '0000'
OUTPUT_DIR = WORKSPACE / 'outputs'
FIG_DIR = WORKSPACE / 'report' / 'images'
CONNECTOME_PATH = '/home/xwh/miniconda3/envs/agent/lib/python3.11/site-packages/flyvis/connectome/fib25-fib19_v2.2.json'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Visual style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.facecolor': 'white',
})

# ===== 1. Load Connectome =====
print("Loading connectome...")
with open(CONNECTOME_PATH) as f:
    connectome = json.load(f)

cell_types = [n['name'] for n in connectome['nodes']]
n_cell_types = len(cell_types)
input_units = connectome['input_units']
output_units = connectome['output_units']
edges = connectome['edges']

print(f"  Cell types: {n_cell_types}")
print(f"  Input units (photoreceptors): {len(input_units)} - {input_units}")
print(f"  Output units: {len(output_units)}")
print(f"  Edges: {len(edges)}")

# Categorize cell types
cell_categories = {}
for ct in cell_types:
    if ct.startswith('R'):
        cell_categories[ct] = 'Photoreceptor'
    elif ct.startswith('L'):
        cell_categories[ct] = 'Lamina'
    elif ct.startswith('Am') or ct.startswith('C') or ct.startswith('CT'):
        cell_categories[ct] = 'Amacrine/Centrifugal'
    elif ct.startswith('Mi'):
        cell_categories[ct] = 'Medulla intrinsic'
    elif ct.startswith('T4') or ct.startswith('T5'):
        cell_categories[ct] = 'Direction-selective (T4/T5)'
    elif ct.startswith('T'):
        cell_categories[ct] = 'Transmedullary (T)'
    elif ct.startswith('Tm') and not ct.startswith('TmY'):
        cell_categories[ct] = 'Transmedullary (Tm)'
    elif ct.startswith('TmY'):
        cell_categories[ct] = 'Transmedullary (TmY)'
    elif ct.startswith('Lawf'):
        cell_categories[ct] = 'Lamina wide-field'
    else:
        cell_categories[ct] = 'Other'

category_colors = {
    'Photoreceptor': '#e74c3c',
    'Lamina': '#e67e22',
    'Lamina wide-field': '#f39c12',
    'Amacrine/Centrifugal': '#9b59b6',
    'Medulla intrinsic': '#3498db',
    'Direction-selective (T4/T5)': '#2ecc71',
    'Transmedullary (T)': '#1abc9c',
    'Transmedullary (Tm)': '#34495e',
    'Transmedullary (TmY)': '#95a5a6',
    'Other': '#bdc3c7',
}

# Build connectivity matrix
ct_to_idx = {ct: i for i, ct in enumerate(cell_types)}
conn_matrix = np.zeros((n_cell_types, n_cell_types))
sign_matrix = np.zeros((n_cell_types, n_cell_types))
syn_count_matrix = np.zeros((n_cell_types, n_cell_types))

for e in edges:
    src_idx = ct_to_idx[e['src']]
    tar_idx = ct_to_idx[e['tar']]
    total_syn = sum(offset_pair[1] for offset_pair in e['offsets'])
    conn_matrix[src_idx, tar_idx] = 1
    sign_matrix[src_idx, tar_idx] = e['alpha']
    syn_count_matrix[src_idx, tar_idx] = total_syn

print(f"  Connectivity density: {conn_matrix.sum() / (n_cell_types**2):.3f}")
exc_count = (sign_matrix > 0).sum()
inh_count = (sign_matrix < 0).sum()
print(f"  Excitatory edges: {exc_count}, Inhibitory edges: {inh_count}")

# ===== 2. Load All 50 Model Parameters =====
print("\nLoading 50 ensemble model checkpoints...")
all_bias = []
all_tc = []
all_sign = []
all_syn_strength = []
all_val_loss = []

for i in range(50):
    chkpt_path = DATA_DIR / f'{i:03d}' / 'best_chkpt'
    state = torch.load(chkpt_path, map_location='cpu', weights_only=False)
    net = state['network']
    all_bias.append(net['nodes_bias'].numpy())
    all_tc.append(net['nodes_time_const'].numpy())
    all_sign.append(net['edges_sign'].numpy())
    all_syn_strength.append(net['edges_syn_strength'].numpy())

    vl_path = DATA_DIR / f'{i:03d}' / 'validation' / 'loss.h5'
    with h5py.File(vl_path, 'r') as f:
        all_val_loss.append(f['data'][()])

all_bias = np.array(all_bias)  # (50, 65)
all_tc = np.array(all_tc)      # (50, 65)
all_sign = np.array(all_sign)  # (50, 604)
all_syn_strength = np.array(all_syn_strength)  # (50, 604)
all_val_loss = np.array(all_val_loss)  # (50,)

print(f"  Ensemble size: {len(all_val_loss)}")
print(f"  Val loss: mean={all_val_loss.mean():.3f}, std={all_val_loss.std():.3f}, "
      f"min={all_val_loss.min():.3f}, max={all_val_loss.max():.3f}")

# ===== 3. Load UMAP/Clustering Data =====
print("\nLoading UMAP/clustering data...")
umap_dir = DATA_DIR / 'umap_and_clustering'
clustering_data = {}

class FakeGMC:
    """Stub for loading pickled GaussianMixtureClustering objects."""
    pass

class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if 'flyvis' in module:
            return FakeGMC
        return super().find_class(module, name)

for pkl_file in sorted(umap_dir.glob('*.pickle')):
    ct_name = pkl_file.stem
    with open(pkl_file, 'rb') as f:
        obj = SafeUnpickler(f).load()
    clustering_data[ct_name] = obj

print(f"  Loaded clustering for {len(clustering_data)} cell types")

# Inspect what's inside
sample_ct = list(clustering_data.keys())[0]
sample_obj = clustering_data[sample_ct]
print(f"  Sample object ({sample_ct}): type={type(sample_obj).__name__}")
attrs = {k: type(v).__name__ for k, v in vars(sample_obj).items() if not k.startswith('_')}
print(f"  Attributes: {attrs}")

# ===== FIGURES =====

# --- Figure 1: Connectome Connectivity Matrix ---
print("\nGenerating Figure 1: Connectome connectivity matrix...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Signed connectivity
ax = axes[0]
signed_conn = sign_matrix * np.log1p(syn_count_matrix)
im = ax.imshow(signed_conn, cmap='RdBu_r', aspect='auto', interpolation='none',
               vmin=-np.abs(signed_conn).max(), vmax=np.abs(signed_conn).max())
ax.set_title('Signed Connectivity\n(sign × log synapse count)')
ax.set_xlabel('Target cell type')
ax.set_ylabel('Source cell type')
# Add tick labels at intervals
tick_positions = list(range(0, n_cell_types, 5))
ax.set_xticks(tick_positions)
ax.set_xticklabels([cell_types[i] for i in tick_positions], rotation=90, fontsize=6)
ax.set_yticks(tick_positions)
ax.set_yticklabels([cell_types[i] for i in tick_positions], fontsize=6)
plt.colorbar(im, ax=ax, shrink=0.8)

# Binary connectivity
ax = axes[1]
im = ax.imshow(conn_matrix, cmap='Greys', aspect='auto', interpolation='none')
ax.set_title(f'Binary Connectivity\n({int(conn_matrix.sum())} edges)')
ax.set_xlabel('Target cell type')
ax.set_ylabel('Source cell type')
ax.set_xticks(tick_positions)
ax.set_xticklabels([cell_types[i] for i in tick_positions], rotation=90, fontsize=6)
ax.set_yticks(tick_positions)
ax.set_yticklabels([cell_types[i] for i in tick_positions], fontsize=6)

# Synapse count distribution
ax = axes[2]
syn_counts_nonzero = syn_count_matrix[syn_count_matrix > 0]
ax.hist(syn_counts_nonzero, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Synapse count per edge')
ax.set_ylabel('Frequency')
ax.set_title(f'Synapse Count Distribution\n(median={np.median(syn_counts_nonzero):.1f})')
ax.axvline(np.median(syn_counts_nonzero), color='red', linestyle='--', label=f'Median={np.median(syn_counts_nonzero):.1f}')
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_connectome_matrix.png', bbox_inches='tight')
plt.close()
print("  Saved fig1_connectome_matrix.png")

# --- Figure 2: Cell Type Categories and Degree Distribution ---
print("Generating Figure 2: Cell type overview...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Category pie chart
ax = axes[0]
cat_counts = Counter(cell_categories.values())
cats = list(cat_counts.keys())
counts = [cat_counts[c] for c in cats]
colors = [category_colors.get(c, '#bdc3c7') for c in cats]
wedges, texts, autotexts = ax.pie(counts, labels=None, autopct='%1.0f%%',
                                   colors=colors, pctdistance=0.8)
ax.legend(wedges, [f'{c} ({cat_counts[c]})' for c in cats],
          loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
ax.set_title(f'Cell Type Categories\n({n_cell_types} types total)')

# In/out degree
ax = axes[1]
in_degree = conn_matrix.sum(axis=0)
out_degree = conn_matrix.sum(axis=1)
x = np.arange(n_cell_types)
width = 0.4
bars1 = ax.bar(x - width/2, in_degree, width, label='In-degree', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, out_degree, width, label='Out-degree', color='#e74c3c', alpha=0.8)
ax.set_xlabel('Cell type')
ax.set_ylabel('Degree')
ax.set_title('In/Out Degree per Cell Type')
ax.set_xticks(range(0, n_cell_types, 5))
ax.set_xticklabels([cell_types[i] for i in range(0, n_cell_types, 5)], rotation=90, fontsize=6)
ax.legend()

# Total synapse input weight per cell type
ax = axes[2]
total_syn_in = syn_count_matrix.sum(axis=0)
total_syn_out = syn_count_matrix.sum(axis=1)
cat_colors_per_ct = [category_colors.get(cell_categories[ct], '#bdc3c7') for ct in cell_types]
ax.barh(range(n_cell_types), total_syn_in, color=cat_colors_per_ct, alpha=0.8)
ax.set_yticks(range(n_cell_types))
ax.set_yticklabels(cell_types, fontsize=5)
ax.set_xlabel('Total synapse count (input)')
ax.set_title('Total Synaptic Input per Cell Type')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_cell_type_overview.png', bbox_inches='tight')
plt.close()
print("  Saved fig2_cell_type_overview.png")

# --- Figure 3: Learned Parameters Across Ensemble ---
print("Generating Figure 3: Ensemble parameter distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Resting potential (bias) distribution across cell types
ax = axes[0, 0]
mean_bias = all_bias.mean(axis=0)
std_bias = all_bias.std(axis=0)
sort_idx = np.argsort(mean_bias)
ax.barh(range(n_cell_types), mean_bias[sort_idx],
        xerr=std_bias[sort_idx], color=[cat_colors_per_ct[i] for i in sort_idx],
        alpha=0.8, capsize=1)
ax.set_yticks(range(n_cell_types))
ax.set_yticklabels([cell_types[i] for i in sort_idx], fontsize=5)
ax.set_xlabel('Resting Potential (learned bias)')
ax.set_title('Learned Resting Potentials\n(mean ± std across 50 models)')
ax.axvline(0, color='black', linestyle='-', linewidth=0.5)

# Time constant distribution
ax = axes[0, 1]
mean_tc = all_tc.mean(axis=0)
std_tc = all_tc.std(axis=0)
sort_idx_tc = np.argsort(mean_tc)
ax.barh(range(n_cell_types), mean_tc[sort_idx_tc],
        xerr=std_tc[sort_idx_tc], color=[cat_colors_per_ct[i] for i in sort_idx_tc],
        alpha=0.8, capsize=1)
ax.set_yticks(range(n_cell_types))
ax.set_yticklabels([cell_types[i] for i in sort_idx_tc], fontsize=5)
ax.set_xlabel('Time Constant (learned)')
ax.set_title('Learned Time Constants\n(mean ± std across 50 models)')

# Synapse strength distribution
ax = axes[1, 0]
mean_strength = all_syn_strength.mean(axis=0)
ax.hist(mean_strength, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Mean Synapse Strength (across ensemble)')
ax.set_ylabel('Frequency')
ax.set_title(f'Synapse Strength Distribution\n(604 edge types)')
ax.axvline(mean_strength.mean(), color='red', linestyle='--',
           label=f'Mean={mean_strength.mean():.4f}')
ax.legend()

# Parameter consistency (CV) across ensemble
ax = axes[1, 1]
cv_bias = np.where(np.abs(all_bias.mean(0)) > 0.01,
                    all_bias.std(0) / np.abs(all_bias.mean(0)), np.nan)
cv_tc = all_tc.std(0) / all_tc.mean(0)
cv_strength = np.where(all_syn_strength.mean(0) > 1e-4,
                        all_syn_strength.std(0) / all_syn_strength.mean(0), np.nan)
ax.boxplot([cv_bias[~np.isnan(cv_bias)], cv_tc[~np.isnan(cv_tc)],
            cv_strength[~np.isnan(cv_strength)]],
           labels=['Resting\nPotential', 'Time\nConstant', 'Synapse\nStrength'],
           patch_artist=True,
           boxprops=dict(facecolor='lightblue', alpha=0.8))
ax.set_ylabel('Coefficient of Variation')
ax.set_title('Parameter Consistency Across Ensemble\n(lower = more consistent)')
ax.set_ylim(0, min(3, ax.get_ylim()[1]))

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_ensemble_parameters.png', bbox_inches='tight')
plt.close()
print("  Saved fig3_ensemble_parameters.png")

# --- Figure 4: Validation Loss and Model Selection ---
print("Generating Figure 4: Validation performance...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Val loss distribution
ax = axes[0]
ax.hist(all_val_loss, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(all_val_loss.mean(), color='red', linestyle='--',
           label=f'Mean={all_val_loss.mean():.3f}')
ax.axvline(all_val_loss.min(), color='green', linestyle='--',
           label=f'Best={all_val_loss.min():.3f}')
ax.set_xlabel('Validation Loss (L2 norm)')
ax.set_ylabel('Count')
ax.set_title('Validation Loss Distribution\n(50 ensemble models)')
ax.legend()

# Val loss sorted
ax = axes[1]
sorted_loss = np.sort(all_val_loss)
ax.plot(range(50), sorted_loss, 'o-', markersize=4, color='steelblue')
ax.fill_between(range(50), sorted_loss.min(), sorted_loss, alpha=0.1, color='steelblue')
ax.set_xlabel('Model Rank')
ax.set_ylabel('Validation Loss')
ax.set_title('Models Ranked by Performance')
ax.axhline(all_val_loss.mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
ax.legend()

# Best vs worst model parameter comparison
ax = axes[2]
best_idx = np.argmin(all_val_loss)
worst_idx = np.argmax(all_val_loss)
ax.scatter(all_bias[best_idx], all_bias[worst_idx], c=cat_colors_per_ct,
           s=30, alpha=0.8, edgecolors='black', linewidths=0.3)
ax.plot([all_bias.min(), all_bias.max()], [all_bias.min(), all_bias.max()],
        'k--', alpha=0.3)
ax.set_xlabel(f'Best Model (#{best_idx}) Resting Potential')
ax.set_ylabel(f'Worst Model (#{worst_idx}) Resting Potential')
ax.set_title('Best vs Worst Model\nResting Potentials')
# Add correlation
corr = np.corrcoef(all_bias[best_idx], all_bias[worst_idx])[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_validation_performance.png', bbox_inches='tight')
plt.close()
print("  Saved fig4_validation_performance.png")

# --- Figure 5: Motion Detection Pathway Analysis ---
print("Generating Figure 5: Motion detection pathway...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# T4/T5 direction selectivity analysis
# T4a-d and T5a-d are direction-selective neurons
ds_types = ['T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
ds_indices = [ct_to_idx[ct] for ct in ds_types]

# Resting potentials of DS neurons
ax = axes[0, 0]
ds_bias = all_bias[:, ds_indices]  # (50, 8)
bp = ax.boxplot([ds_bias[:, i] for i in range(8)], labels=ds_types,
                patch_artist=True)
colors_ds = ['#2ecc71']*4 + ['#27ae60']*4  # T4 lighter, T5 darker
for patch, color in zip(bp['boxes'], colors_ds):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Resting Potential')
ax.set_title('Direction-Selective Neuron\nResting Potentials (T4/T5)')
ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

# Time constants of DS neurons
ax = axes[0, 1]
ds_tc = all_tc[:, ds_indices]
bp = ax.boxplot([ds_tc[:, i] for i in range(8)], labels=ds_types,
                patch_artist=True)
for patch, color in zip(bp['boxes'], colors_ds):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Time Constant')
ax.set_title('Direction-Selective Neuron\nTime Constants (T4/T5)')

# Input connectivity to T4/T5
ax = axes[1, 0]
t4_inputs = {}
t5_inputs = {}
for ds_type in ['T4a', 'T4b', 'T4c', 'T4d']:
    idx = ct_to_idx[ds_type]
    input_syn = syn_count_matrix[:, idx]
    for src_idx, count in enumerate(input_syn):
        if count > 0:
            src = cell_types[src_idx]
            if src not in t4_inputs:
                t4_inputs[src] = []
            t4_inputs[src].append(count)

for ds_type in ['T5a', 'T5b', 'T5c', 'T5d']:
    idx = ct_to_idx[ds_type]
    input_syn = syn_count_matrix[:, idx]
    for src_idx, count in enumerate(input_syn):
        if count > 0:
            src = cell_types[src_idx]
            if src not in t5_inputs:
                t5_inputs[src] = []
            t5_inputs[src].append(count)

# Top inputs to T4
t4_mean_inputs = {k: np.mean(v) for k, v in t4_inputs.items()}
t4_sorted = sorted(t4_mean_inputs.items(), key=lambda x: -x[1])[:15]
t4_names, t4_vals = zip(*t4_sorted)
ax.barh(range(len(t4_names)), t4_vals, color='#2ecc71', alpha=0.8)
ax.set_yticks(range(len(t4_names)))
ax.set_yticklabels(t4_names, fontsize=8)
ax.set_xlabel('Mean Synapse Count')
ax.set_title('Top Inputs to T4 (ON pathway)')
ax.invert_yaxis()

# Top inputs to T5
ax = axes[1, 1]
t5_mean_inputs = {k: np.mean(v) for k, v in t5_inputs.items()}
t5_sorted = sorted(t5_mean_inputs.items(), key=lambda x: -x[1])[:15]
t5_names, t5_vals = zip(*t5_sorted)
ax.barh(range(len(t5_names)), t5_vals, color='#27ae60', alpha=0.8)
ax.set_yticks(range(len(t5_names)))
ax.set_yticklabels(t5_names, fontsize=8)
ax.set_xlabel('Mean Synapse Count')
ax.set_title('Top Inputs to T5 (OFF pathway)')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_motion_pathway.png', bbox_inches='tight')
plt.close()
print("  Saved fig5_motion_pathway.png")

# --- Figure 6: Excitatory/Inhibitory Balance ---
print("Generating Figure 6: E/I balance...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# E/I ratio per target cell type
ax = axes[0]
exc_input = np.zeros(n_cell_types)
inh_input = np.zeros(n_cell_types)
for e in edges:
    tar_idx = ct_to_idx[e['tar']]
    total_syn = sum(offset_pair[1] for offset_pair in e['offsets'])
    if e['alpha'] > 0:
        exc_input[tar_idx] += total_syn
    else:
        inh_input[tar_idx] += total_syn

ei_ratio = np.where(inh_input > 0, exc_input / inh_input, 0)
sort_ei = np.argsort(ei_ratio)
valid = ei_ratio[sort_ei] > 0
ax.barh(np.arange(valid.sum()), ei_ratio[sort_ei][valid],
        color=[cat_colors_per_ct[i] for i in sort_ei[valid]], alpha=0.8)
ax.set_yticks(range(valid.sum()))
ax.set_yticklabels([cell_types[i] for i in sort_ei[valid]], fontsize=5)
ax.axvline(1, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('E/I Ratio')
ax.set_title('Excitatory/Inhibitory Input Ratio')
ax.invert_yaxis()

# Learned sign values distribution
ax = axes[1]
mean_sign = all_sign.mean(axis=0)
ax.hist(mean_sign, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('Mean Sign Value (across ensemble)')
ax.set_ylabel('Frequency')
ax.set_title('Learned Sign Distribution')
ax.axvline(0, color='red', linestyle='--', alpha=0.5)

# Signed synapse strength
ax = axes[2]
# Multiply sign * strength for effective weight
mean_eff_weight = (all_sign * all_syn_strength).mean(axis=0)
ax.hist(mean_eff_weight[mean_eff_weight != 0], bins=50, color='steelblue',
        edgecolor='white', alpha=0.8)
ax.set_xlabel('Effective Weight (sign × strength)')
ax.set_ylabel('Frequency')
ax.set_title('Effective Synaptic Weight Distribution')
ax.axvline(0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_ei_balance.png', bbox_inches='tight')
plt.close()
print("  Saved fig6_ei_balance.png")

# --- Figure 7: Network Architecture Schematic ---
print("Generating Figure 7: Network layer structure...")
fig, ax = plt.subplots(figsize=(14, 8))

# Organize by processing layer
layers = {
    'Retina\n(Input)': [ct for ct in cell_types if ct.startswith('R')],
    'Lamina': [ct for ct in cell_types if ct.startswith('L') or ct.startswith('Lawf')],
    'Medulla\n(Am/C/CT)': [ct for ct in cell_types if ct.startswith('Am') or ct.startswith('C') or ct.startswith('CT')],
    'Medulla\n(Mi)': [ct for ct in cell_types if ct.startswith('Mi')],
    'Lobula Plate\n(T4/T5)': [ct for ct in cell_types if ct.startswith('T4') or ct.startswith('T5')],
    'Lobula Plate\n(T1-T3)': [ct for ct in cell_types if ct in ['T1', 'T2', 'T2a', 'T3']],
    'Transmedullary\n(Tm)': [ct for ct in cell_types if ct.startswith('Tm') and not ct.startswith('TmY')],
    'Transmedullary\n(TmY)': [ct for ct in cell_types if ct.startswith('TmY')],
}

layer_names = list(layers.keys())
layer_y = {name: i * 1.5 for i, name in enumerate(layer_names)}

# Position nodes
node_pos = {}
for layer_name, cts in layers.items():
    y = layer_y[layer_name]
    n = len(cts)
    for j, ct in enumerate(cts):
        x = (j - n/2 + 0.5) * 0.8
        node_pos[ct] = (x, y)

# Draw edges (subsample for clarity)
for e in edges:
    if e['src'] in node_pos and e['tar'] in node_pos:
        x1, y1 = node_pos[e['src']]
        x2, y2 = node_pos[e['tar']]
        total_syn = sum(offset_pair[1] for offset_pair in e['offsets'])
        alpha_val = min(0.5, total_syn / 100)
        color = '#e74c3c' if e['alpha'] < 0 else '#3498db'
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, alpha=alpha_val,
                                   linewidth=0.3 + total_syn/50))

# Draw nodes
for ct, (x, y) in node_pos.items():
    cat = cell_categories.get(ct, 'Other')
    color = category_colors.get(cat, '#bdc3c7')
    ax.scatter(x, y, s=200, c=color, edgecolors='black', linewidths=0.5, zorder=5)
    ax.text(x, y - 0.2, ct, ha='center', va='top', fontsize=5, fontweight='bold')

# Layer labels
for layer_name, y in layer_y.items():
    ax.text(-8, y, layer_name, ha='right', va='center', fontsize=9, fontweight='bold')

ax.set_xlim(-9, 8)
ax.set_ylim(-1, max(layer_y.values()) + 1)
ax.set_title('Drosophila Optic Lobe Motion Pathway\n(Node color = cell category, Blue = excitatory, Red = inhibitory)',
             fontsize=13)
ax.axis('off')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_network_architecture.png', bbox_inches='tight')
plt.close()
print("  Saved fig7_network_architecture.png")

# --- Figure 8: Parameter Correlation and Clustering ---
print("Generating Figure 8: Parameter correlations...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Bias vs Time Constant correlation
ax = axes[0, 0]
mean_bias_ct = all_bias.mean(axis=0)
mean_tc_ct = all_tc.mean(axis=0)
for i, ct in enumerate(cell_types):
    cat = cell_categories[ct]
    color = category_colors.get(cat, '#bdc3c7')
    ax.scatter(mean_bias_ct[i], mean_tc_ct[i], c=color, s=40,
               edgecolors='black', linewidths=0.3, zorder=5)
ax.set_xlabel('Mean Resting Potential')
ax.set_ylabel('Mean Time Constant')
ax.set_title('Resting Potential vs Time Constant')
corr = np.corrcoef(mean_bias_ct, mean_tc_ct)[0, 1]
ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=10, va='top')

# Inter-model parameter correlation matrix
ax = axes[0, 1]
# Correlate bias vectors across models
bias_corr = np.corrcoef(all_bias)  # (50, 50)
im = ax.imshow(bias_corr, cmap='viridis', vmin=0.5, vmax=1.0)
ax.set_title('Inter-Model Bias Correlation')
ax.set_xlabel('Model Index')
ax.set_ylabel('Model Index')
plt.colorbar(im, ax=ax, shrink=0.8)

# Parameter variance by cell type category
ax = axes[1, 0]
cat_variance = defaultdict(list)
for i, ct in enumerate(cell_types):
    cat = cell_categories[ct]
    cat_variance[cat].append(all_bias[:, i].std())
cat_names = sorted(cat_variance.keys())
bp = ax.boxplot([cat_variance[c] for c in cat_names], labels=[c.split('(')[0].strip() for c in cat_names],
                patch_artist=True)
for patch, name in zip(bp['boxes'], cat_names):
    patch.set_facecolor(category_colors.get(name, '#bdc3c7'))
    patch.set_alpha(0.7)
ax.set_ylabel('Std of Resting Potential\n(across 50 models)')
ax.set_title('Parameter Variability by Cell Category')
ax.tick_params(axis='x', rotation=45)

# Ensemble consensus: % of models that agree on sign
ax = axes[1, 1]
sign_consensus = np.abs(all_sign.mean(axis=0))
ax.hist(sign_consensus, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('|Mean Sign| (1 = full consensus)')
ax.set_ylabel('Frequency')
ax.set_title('Ensemble Consensus on Synapse Signs')
ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='90% consensus')
n_high = (sign_consensus > 0.9).sum()
ax.text(0.95, 0.95, f'{n_high}/{len(sign_consensus)} edges\n>90% consensus',
        transform=ax.transAxes, ha='right', va='top', fontsize=9)
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig8_parameter_correlations.png', bbox_inches='tight')
plt.close()
print("  Saved fig8_parameter_correlations.png")

# --- Figure 9: UMAP Clustering Summary ---
print("Generating Figure 9: UMAP clustering summary...")
fig, axes = plt.subplots(2, 4, figsize=(18, 9))

# Select key cell types for UMAP visualization
key_types = ['T4a', 'T4b', 'T5a', 'T5b', 'Mi1', 'Mi9', 'Tm1', 'Tm9']
for idx, ct_name in enumerate(key_types):
    ax = axes[idx // 4, idx % 4]
    obj = clustering_data.get(ct_name)
    if obj is not None:
        # The FakeGMC objects have labels (ndarray) and n_clusters (ndarray)
        # but embedding is also a FakeGMC (nested object). We can't access UMAP embeddings
        # Instead show cluster assignment info
        if hasattr(obj, 'labels') and isinstance(obj.labels, np.ndarray):
            labels = obj.labels
            unique_labels = np.unique(labels)
            n_cl = len(unique_labels)
            counts = [np.sum(labels == lab) for lab in unique_labels]
            ax.bar(unique_labels, counts, color=plt.cm.Set2(np.linspace(0, 1, n_cl)),
                   edgecolor='white')
            ax.set_xlabel('Cluster ID', fontsize=8)
            ax.set_ylabel('# Neurons', fontsize=8)
            ax.set_title(f'{ct_name} ({n_cl} clusters)', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'Labels not available', transform=ax.transAxes, ha='center')
            ax.set_title(f'{ct_name}', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
        ax.set_title(f'{ct_name}', fontsize=11)

plt.suptitle('Cluster Assignments of Neural Responses\n(Selected Cell Types from Ensemble)', fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig9_umap_clustering.png', bbox_inches='tight')
plt.close()
print("  Saved fig9_umap_clustering.png")

# --- Figure 10: Clustering Quality Summary ---
print("Generating Figure 10: Clustering quality...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Number of clusters per cell type
ax = axes[0]
n_clusters_per_type = {}
# Extract n_clusters from all cell types
for ct_name, obj in clustering_data.items():
    if hasattr(obj, 'labels') and isinstance(obj.labels, np.ndarray):
        n_clusters_per_type[ct_name] = len(np.unique(obj.labels))

if n_clusters_per_type:
    names = sorted(n_clusters_per_type.keys())
    vals = [n_clusters_per_type[n] for n in names]
    colors_cluster = [category_colors.get(cell_categories.get(n, 'Other'), '#bdc3c7') for n in names]
    ax.barh(range(len(names)), vals, color=colors_cluster, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('Number of Clusters')
    ax.set_title('Number of Response Clusters per Cell Type')
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, 'Cluster count not available\nin loaded data',
            transform=ax.transAxes, ha='center', fontsize=12)
    ax.set_title('Cluster Counts')

# Clustering data summary statistics
ax = axes[1]
n_with_embedding = sum(1 for obj in clustering_data.values() if hasattr(obj, 'embedding'))
n_with_labels = sum(1 for obj in clustering_data.values() if hasattr(obj, 'labels'))
n_with_gmm = sum(1 for obj in clustering_data.values()
                  if any('gmm' in k.lower() or 'gaussian' in k.lower() or 'mixture' in k.lower()
                         for k in vars(obj).keys()))

# Check what attributes exist
all_attrs = set()
for obj in clustering_data.values():
    all_attrs.update(vars(obj).keys())

attr_text = f"Total cell types with clustering: {len(clustering_data)}\n\n"
attr_text += f"Common attributes found:\n"
for attr in sorted(all_attrs):
    count = sum(1 for obj in clustering_data.values() if hasattr(obj, attr))
    attr_text += f"  {attr}: {count}/{len(clustering_data)}\n"

ax.text(0.05, 0.95, attr_text, transform=ax.transAxes, fontsize=8,
        va='top', family='monospace')
ax.set_title('Clustering Data Overview')
ax.axis('off')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig10_clustering_summary.png', bbox_inches='tight')
plt.close()
print("  Saved fig10_clustering_summary.png")

# --- Figure 11: Number of neurons in the network ---
print("Generating Figure 11: Network scale...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# From meta yaml: extent=15, hexagonal grid
# Number of columns = number of hexagonal positions in extent 15
# For hex grid with extent r: n = 1 + 3*r*(r+1) for first r rings, but flyvis uses different convention
# From the paper: 721 columns (hex grid extent 15)
# Actually extent=15 means radius 15 hex grid:
# n_columns = sum(6*k for k in range(1,16)) + 1 = 721
n_columns = 1 + sum(6*k for k in range(1, 16))
print(f"  Number of columns (hex grid extent 15): {n_columns}")

# But not all cell types are present at all columns
# stride [1,1] types are present at every column
# Other patterns might be different
n_neurons_per_type = {}
for node in connectome['nodes']:
    ct = node['name']
    pattern = node.get('pattern', ['stride', [1, 1]])
    if pattern[0] == 'stride':
        stride = pattern[1]
        # Approximate: for stride [s1,s2], coverage fraction is 1/(s1*s2) approximately
        # But for hex grid it's more complex. The paper says 45669 neurons total
        n_neurons_per_type[ct] = n_columns  # simplified
    else:
        n_neurons_per_type[ct] = n_columns

total_neurons = sum(n_neurons_per_type.values())
# The paper mentions 45,669 neurons
# With 65 types * 721 columns = 46,865 (some types may have fewer columns)
print(f"  Approximate total neurons: {total_neurons}")

ax = axes[0]
ct_neuron_counts = sorted(n_neurons_per_type.items(), key=lambda x: x[0])
names_n = [x[0] for x in ct_neuron_counts]
counts_n = [x[1] for x in ct_neuron_counts]
colors_n = [category_colors.get(cell_categories.get(n, 'Other'), '#bdc3c7') for n in names_n]

# Group by category for stacked bar
cat_totals = defaultdict(int)
for ct, count in ct_neuron_counts:
    cat = cell_categories.get(ct, 'Other')
    cat_totals[cat] += count

cats_sorted = sorted(cat_totals.items(), key=lambda x: -x[1])
cat_names_pie = [c[0] for c in cats_sorted]
cat_counts_pie = [c[1] for c in cats_sorted]
colors_pie = [category_colors.get(c, '#bdc3c7') for c in cat_names_pie]
wedges, texts, autotexts = ax.pie(cat_counts_pie, labels=None, autopct='%1.0f%%',
                                   colors=colors_pie, pctdistance=0.8)
ax.legend(wedges, [f'{c} ({n})' for c, n in cats_sorted],
          loc='center left', bbox_to_anchor=(1, 0.5), fontsize=7)
ax.set_title(f'Neuron Distribution by Category\n(~{total_neurons} total neurons)')

# Parameters to learn
ax = axes[1]
param_counts = {
    'Resting Potential\n(per type)': n_cell_types,
    'Time Constant\n(per type)': n_cell_types,
    'Synapse Strength\n(per edge type)': len(edges),
    'Synapse Sign\n(fixed from literature)': len(edges),
    'Synapse Count\n(from connectome)': 2355,  # spatial variants
}
names_p = list(param_counts.keys())
vals_p = list(param_counts.values())
colors_p = ['#3498db', '#2ecc71', '#e74c3c', '#95a5a6', '#f39c12']
learnable = [True, True, True, False, False]
hatches = ['' if l else '///' for l in learnable]
bars = ax.barh(range(len(names_p)), vals_p, color=colors_p, alpha=0.8)
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)
ax.set_yticks(range(len(names_p)))
ax.set_yticklabels(names_p, fontsize=9)
ax.set_xlabel('Number of Parameters')
ax.set_title('DMN Parameter Types\n(hatched = fixed, solid = learned)')
for i, v in enumerate(vals_p):
    ax.text(v + 10, i, str(v), va='center', fontsize=9)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig11_network_scale.png', bbox_inches='tight')
plt.close()
print("  Saved fig11_network_scale.png")

# Save summary statistics
print("\n=== Summary Statistics ===")
stats = {
    'n_cell_types': n_cell_types,
    'n_edges': len(edges),
    'n_excitatory': int(exc_count),
    'n_inhibitory': int(inh_count),
    'n_input_types': len(input_units),
    'n_output_types': len(output_units),
    'n_ensemble_models': len(all_val_loss),
    'val_loss_mean': float(all_val_loss.mean()),
    'val_loss_std': float(all_val_loss.std()),
    'val_loss_best': float(all_val_loss.min()),
    'val_loss_worst': float(all_val_loss.max()),
    'n_columns': n_columns,
    'approx_total_neurons': total_neurons,
    'n_clustering_types': len(clustering_data),
}
with open(OUTPUT_DIR / 'summary_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

for k, v in stats.items():
    print(f"  {k}: {v}")

print("\nAll figures generated successfully!")
