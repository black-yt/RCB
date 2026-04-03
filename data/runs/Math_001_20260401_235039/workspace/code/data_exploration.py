"""
Data Exploration: VOS Framework Study
Explores the synthetic ill-conditioned Lasso dataset.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path

# Set style
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'lines.linewidth': 1.5,
})
sns.set_style("whitegrid")

# Paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_001_20260401_235039')
DATA_DIR = WORKSPACE / 'data'
OUT_DIR = WORKSPACE / 'outputs'
IMG_DIR = WORKSPACE / 'report' / 'images'

OUT_DIR.mkdir(exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load data
data = np.load(DATA_DIR / 'complex_optimization_data.npy', allow_pickle=True).item()
A = data['A']           # (1000, 2000)
b = data['b']           # (1000,)
x_true = data['x_true'] # (2000,), sparse

m, n = A.shape
nnz = np.count_nonzero(x_true)

print(f"Design matrix A: {m} x {n}")
print(f"Response vector b: {m}")
print(f"True coefficients x_true: {n} dims, {nnz} nonzero ({100*nnz/n:.1f}% sparse)")
print(f"Meta: {data['meta']}")

# Compute singular values (use reduced SVD)
print("Computing SVD...")
U, s, Vt = np.linalg.svd(A, full_matrices=False)
cond = s[0] / s[-1]
print(f"Singular values: max={s[0]:.4f}, min={s[-1]:.4f}, condition number={cond:.2f}")
print(f"Lipschitz constant L = ||A||_2^2 = {s[0]**2:.4f}")

# Compute optimal lambda for Lasso
lam_max = np.max(np.abs(A.T @ b))
lam = 0.1 * lam_max
print(f"lambda_max = {lam_max:.4f}, chosen lambda = {lam:.4f}")

# Save statistics
stats = {
    'm': m, 'n': n, 'nnz': nnz,
    'singular_values': s,
    'condition_number': cond,
    'L': s[0]**2,
    'lam_max': lam_max,
    'lam': lam,
}
np.save(OUT_DIR / 'data_stats.npy', stats, allow_pickle=True)

# ---- Figure 1: Data Overview ----
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.4)

# Panel A: Singular value spectrum
ax1 = fig.add_subplot(gs[0, 0])
ax1.semilogy(s, 'b-', linewidth=1.5, label='Singular values')
ax1.axhline(s[0], color='r', linestyle='--', alpha=0.5, label=f'$\\sigma_{{max}}={s[0]:.2f}$')
ax1.axhline(s[-1], color='g', linestyle='--', alpha=0.5, label=f'$\\sigma_{{min}}={s[-1]:.2f}$')
ax1.set_xlabel('Index')
ax1.set_ylabel('Singular Value (log scale)')
ax1.set_title(f'Singular Value Spectrum\n(Condition $\\kappa = {cond:.1f}$)')
ax1.legend(fontsize=9)

# Panel B: Distribution of singular values
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(s, bins=40, color='steelblue', edgecolor='white', linewidth=0.5)
ax2.set_xlabel('Singular Value')
ax2.set_ylabel('Count')
ax2.set_title('Singular Value Distribution')

# Panel C: True coefficient sparsity
ax3 = fig.add_subplot(gs[0, 2])
colors = ['crimson' if v != 0 else 'lightgray' for v in x_true]
ax3.bar(range(n), np.abs(x_true), color=colors, width=1.0, linewidth=0)
ax3.set_xlabel('Coefficient Index')
ax3.set_ylabel('|x_true|')
ax3.set_title(f'True Sparse Coefficients\n({nnz} nonzero / {n} = {100*nnz/n:.1f}% sparse)')
ax3.set_xlim(0, n)

# Panel D: Response vector b
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(b, bins=40, color='teal', edgecolor='white', linewidth=0.5)
ax4.set_xlabel('Response $b_i$')
ax4.set_ylabel('Count')
ax4.set_title('Response Vector Distribution')

# Panel E: Column norms of A
ax5 = fig.add_subplot(gs[1, 1])
col_norms = np.linalg.norm(A, axis=0)
ax5.hist(col_norms, bins=40, color='darkorange', edgecolor='white', linewidth=0.5)
ax5.set_xlabel('Column Norm $\\|a_j\\|_2$')
ax5.set_ylabel('Count')
ax5.set_title('Column Norms of Design Matrix')

# Panel F: Heatmap of small submatrix
ax6 = fig.add_subplot(gs[1, 2])
sub = A[:50, :100]
im = ax6.imshow(sub, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3)
plt.colorbar(im, ax=ax6)
ax6.set_xlabel('Feature index (0-99)')
ax6.set_ylabel('Sample index (0-49)')
ax6.set_title('Design Matrix Submatrix $A_{1:50, 1:100}$')

fig.suptitle('Dataset Overview: High-Dimensional Lasso Regression\n'
             f'$A \\in \\mathbb{{R}}^{{{m}\\times {n}}}$, '
             f'$b \\in \\mathbb{{R}}^{{{m}}}$, '
             f'Sparsity = {100*nnz/n:.1f}%',
             fontsize=13, fontweight='bold', y=1.01)

plt.savefig(IMG_DIR / 'fig1_data_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig1_data_overview.png")

print("Data exploration complete.")
print(f"Lipschitz constant L = {s[0]**2:.4f}")
print(f"Chosen lambda = {lam:.4f}")
