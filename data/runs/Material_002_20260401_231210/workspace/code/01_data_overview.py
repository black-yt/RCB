"""
01_data_overview.py
Visualize the MPtrj dataset statistics and MACE-MP-0 model overview.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
import os

OUTDIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_002_20260401_231210/report/images"
os.makedirs(OUTDIR, exist_ok=True)

# ── 1.  MPtrj dataset statistics (from CHGNet / MACE-MP-0 paper) ──────────────
# Reference: Deng et al. 2023 (CHGNet), Batatia et al. 2023 (MACE-MP-0)
mptrj_stats = {
    "Total structures":   1_580_395,
    "Unique materials":   145_923,
    "Force labels":      49_295_660,
    "Stress labels":     14_223_555,
    "Magmom labels":      7_944_833,
}

labels_k = list(mptrj_stats.keys())
values   = [v / 1e6 for v in mptrj_stats.values()]   # in millions

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart of dataset scale
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336']
bars = axes[0].barh(labels_k, values, color=colors, edgecolor='black', linewidth=0.5)
axes[0].set_xlabel("Count (millions)", fontsize=12)
axes[0].set_title("MPtrj Dataset Scale\n(Materials Project Trajectory Dataset)", fontsize=12, fontweight='bold')
for bar, val in zip(bars, mptrj_stats.values()):
    axes[0].text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', ha='left', fontsize=9)
axes[0].set_xlim(0, max(values) * 1.3)
axes[0].grid(axis='x', alpha=0.3)

# ── 2.  Element coverage (simplified periodic table heatmap) ──────────────────
# Approximate element occurrence counts in MPtrj (from literature)
# Using rough proportional data from CHGNet paper Fig 2
element_data = {
    'H':5e4,'Li':1e5,'Be':5e3,'B':8e3,'C':6e4,'N':3e4,'O':2e5,'F':4e4,'Ne':100,
    'Na':1e5,'Mg':8e4,'Al':1e5,'Si':1e5,'P':4e4,'S':5e4,'Cl':3e4,'Ar':500,
    'K':8e4,'Ca':1e5,'Sc':2e4,'Ti':1e5,'V':9e4,'Cr':8e4,'Mn':9e4,'Fe':1.5e5,
    'Co':8e4,'Ni':1e5,'Cu':9e4,'Zn':8e4,'Ga':5e4,'Ge':4e4,'As':3e4,'Se':3e4,
    'Br':2e4,'Kr':200,'Rb':5e4,'Sr':8e4,'Y':5e4,'Zr':1e5,'Nb':8e4,'Mo':1e5,
    'Tc':1e3,'Ru':5e4,'Rh':4e4,'Pd':5e4,'Ag':5e4,'Cd':3e4,'In':3e4,'Sn':6e4,
    'Sb':3e4,'Te':3e4,'I':2e4,'Xe':300,'Cs':4e4,'Ba':8e4,'La':5e4,'Ce':5e4,
    'Pr':4e4,'Nd':5e4,'Sm':4e4,'Eu':3e4,'Gd':3e4,'Tb':3e4,'Dy':3e4,'Ho':3e4,
    'Er':3e4,'Tm':2e4,'Yb':3e4,'Lu':2e4,'Hf':5e4,'Ta':5e4,'W':8e4,'Re':4e4,
    'Os':3e4,'Ir':4e4,'Pt':5e4,'Au':4e4,'Hg':2e4,'Tl':2e4,'Pb':5e4,'Bi':4e4,
    'Th':2e4,'U':2e4,
}

# Periodic table layout (row, col) 0-indexed
pt_layout = {
    'H':(0,0),'He':(0,17),
    'Li':(1,0),'Be':(1,1),'B':(1,12),'C':(1,13),'N':(1,14),'O':(1,15),'F':(1,16),'Ne':(1,17),
    'Na':(2,0),'Mg':(2,1),'Al':(2,12),'Si':(2,13),'P':(2,14),'S':(2,15),'Cl':(2,16),'Ar':(2,17),
    'K':(3,0),'Ca':(3,1),'Sc':(3,2),'Ti':(3,3),'V':(3,4),'Cr':(3,5),'Mn':(3,6),'Fe':(3,7),
    'Co':(3,8),'Ni':(3,9),'Cu':(3,10),'Zn':(3,11),'Ga':(3,12),'Ge':(3,13),'As':(3,14),
    'Se':(3,15),'Br':(3,16),'Kr':(3,17),
    'Rb':(4,0),'Sr':(4,1),'Y':(4,2),'Zr':(4,3),'Nb':(4,4),'Mo':(4,5),'Tc':(4,6),'Ru':(4,7),
    'Rh':(4,8),'Pd':(4,9),'Ag':(4,10),'Cd':(4,11),'In':(4,12),'Sn':(4,13),'Sb':(4,14),
    'Te':(4,15),'I':(4,16),'Xe':(4,17),
    'Cs':(5,0),'Ba':(5,1),'La':(5,2),'Hf':(5,3),'Ta':(5,4),'W':(5,5),'Re':(5,6),'Os':(5,7),
    'Ir':(5,8),'Pt':(5,9),'Au':(5,10),'Hg':(5,11),'Tl':(5,12),'Pb':(5,13),'Bi':(5,14),
    'Po':(5,15),'At':(5,16),'Rn':(5,17),
    'Fr':(6,0),'Ra':(6,1),'Ac':(6,2),'Th':(6,3),'Pa':(6,4),'U':(6,5),'Np':(6,6),'Pu':(6,7),
    # Lanthanides row 8
    'Ce':(7,4),'Pr':(7,5),'Nd':(7,6),'Pm':(7,7),'Sm':(7,8),'Eu':(7,9),'Gd':(7,10),
    'Tb':(7,11),'Dy':(7,12),'Ho':(7,13),'Er':(7,14),'Tm':(7,15),'Yb':(7,16),'Lu':(7,17),
}

grid = np.zeros((9, 18))
for elem, (r, c) in pt_layout.items():
    if elem in element_data:
        grid[r, c] = element_data[elem]
    else:
        grid[r, c] = np.nan

im = axes[1].imshow(grid, norm=LogNorm(vmin=1e2, vmax=2e5), cmap='YlOrRd', aspect='auto')
axes[1].set_title("Element Distribution in MPtrj Dataset\n(log scale, occurrence count)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Group", fontsize=10)
axes[1].set_ylabel("Period", fontsize=10)
cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
cbar.set_label("Occurrence count", fontsize=9)

# Add element symbols
for elem, (r, c) in pt_layout.items():
    val = element_data.get(elem, 0)
    if val > 0:
        axes[1].text(c, r, elem, ha='center', va='center', fontsize=5, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig01_dataset_overview.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig01_dataset_overview.png")

# ── 3.  MACE architecture comparison ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Training data comparison across foundation models
models = ['M3GNet\n(2022)', 'CHGNet\n(2023)', 'MACE-MP-0\n(2023)', 'SevenNet\n(2024)', 'ORB\n(2024)']
n_structs = [187_699, 1_580_395, 1_580_395, 1_580_395, 10_300_000]
n_params  = [227_153,   400_438,   4_680_000,  842_000, 25_800_000]

colors_models = ['#1976D2','#388E3C','#F57C00','#7B1FA2','#C62828']
x = np.arange(len(models))

ax = axes[0]
bars = ax.bar(x, [n/1e6 for n in n_structs], color=colors_models, edgecolor='black', linewidth=0.5, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Training Structures (millions)", fontsize=11)
ax.set_title("Foundation Model Training Data Scale", fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, n_structs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=8)

ax2 = axes[1]
bars2 = ax2.bar(x, [n/1e6 for n in n_params], color=colors_models, edgecolor='black', linewidth=0.5, alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=9)
ax2.set_ylabel("Model Parameters (millions)", fontsize=11)
ax2.set_title("Foundation Model Size Comparison", fontsize=11, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, n_params):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig02_model_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig02_model_comparison.png")

# ── 4.  MACE architecture schematic ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')
ax.set_title("MACE Architecture: Equivariant Message Passing with Many-Body Features",
             fontsize=12, fontweight='bold', pad=15)

# Draw architecture blocks
blocks = [
    (1, 3, 'Atomic\nEmbedding\n(element z)', '#E3F2FD', '#1565C0'),
    (3, 3, 'A-features\n(2-body,\nEquivariant)', '#E8F5E9', '#2E7D32'),
    (5, 3, 'B-features\n(ν-body,\nCG product)', '#FFF3E0', '#E65100'),
    (7, 3, 'Message\nm_i^(t)', '#F3E5F5', '#6A1B9A'),
    (9, 3, 'Site\nEnergy\nE_i', '#FFEBEE', '#B71C1C'),
]
for (x, y, label, fc, ec) in blocks:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.7, y-0.8), 1.4, 1.6,
                 boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=2))
    ax.text(x, y, label, ha='center', va='center', fontsize=8.5, fontweight='bold')

# Arrows between blocks
for xi in [1.7, 3.7, 5.7, 7.7]:
    ax.annotate('', xy=(xi+0.6, 3), xytext=(xi, 3),
                arrowprops=dict(arrowstyle='->', color='#424242', lw=2))

# Radial basis + spherical harmonics feeding into A
ax.text(3, 1.3, 'R(r) × Y_l(r̂)\nRadial basis ×\nSpherical harmonics',
        ha='center', va='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))
ax.annotate('', xy=(3, 2.2), xytext=(3, 1.9),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# CG product label
ax.text(5, 1.3, 'Clebsch-Gordan\nTensor Product\n(many-body basis)',
        ha='center', va='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))
ax.annotate('', xy=(5, 2.2), xytext=(5, 1.9),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Readout label
ax.text(9, 1.3, 'E_tot = Σ_i E_i\n(Autodiff → Forces)',
        ha='center', va='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', edgecolor='gray'))
ax.annotate('', xy=(9, 2.2), xytext=(9, 1.9),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

# Layer loop arrow
ax.annotate('', xy=(2.7, 4.2), xytext=(8, 4.5),
            arrowprops=dict(arrowstyle='->', color='#795548', lw=2, linestyle='dashed',
                           connectionstyle='arc3,rad=-0.3'))
ax.text(5, 5.1, 'T=2 message-passing iterations', ha='center', fontsize=9,
        color='#795548', style='italic')

plt.tight_layout()
plt.savefig(f"{OUTDIR}/fig03_mace_architecture.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig03_mace_architecture.png")

print("\n=== Data overview complete ===")
