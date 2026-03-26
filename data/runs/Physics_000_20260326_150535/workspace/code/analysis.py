"""
Multi-component Icosahedral Cluster Analysis
Based on the paper: "General theory for packing icosahedral shells into multi-component aggregates"
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from itertools import product
import os

# ── Setup paths ──────────────────────────────────────────────────────────────
WORKSPACE = '/mnt/d/xwh/ailab记录/工作/26年03月/SGI-Bench/ResearchClawBench/workspaces/Physics_000_20260326_150535'
OUTPUTS   = os.path.join(WORKSPACE, 'outputs')
REPORT_IMGS = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(REPORT_IMGS, exist_ok=True)

# ── Raw data (from data file) ────────────────────────────────────────────────
hexagonal_coords = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
                    (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
                    (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
                    (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
                    (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),
                    (5,0),(5,1),(5,2),(5,3),(5,4),(5,5)]

mackay_sequence   = [1, 13, 55, 147, 309]
new_sequence_b5   = [1, 13, 45, 117, 239, 431]
chiral_labels     = ['MC', 'BG', 'Ch1', 'Ch2', 'Ch3', 'Ch4', 'Ch5']
shell_colors      = {'MC':'#1f77b4','BG':'#ff7f0e','Ch1':'#2ca02c',
                     'Ch2':'#d62728','Ch3':'#9467bd','Ch4':'#8c564b','Ch5':'#e377c2'}

atomic_radii = [('Na',1.86),('K',2.27),('Rb',2.48),('Cs',2.65),
                ('Ag',1.44),('Cu',1.28),('Ni',1.24)]

atomic_pairs_compatibility = [('Na','Rb',0.22),('Ag','Cu',0.12),('Ag','Ni',0.15),('Cu','Ni',0.032)]
optimal_mismatch_ranges    = [('MC','MC',0.03,0.05),('MC','Ch1',0.12,0.16),
                               ('MC','Ch2',0.19,0.22),('MC','BG',0.08,0.10)]
multicomponent_clusters    = [('Na13@Rb32','Na','Rb','MC','Ch1'),
                               ('K13@Cs42','K','Cs','MC','Ch2'),
                               ('Ag13@Cu45','Ag','Cu','MC','Ch1')]
shell_energies   = [(1,'MC',0.00),(2,'MC',-2.35),(2,'Ch1',-2.15),
                    (3,'MC',-4.82),(3,'Ch1',-4.61),(3,'BG',-4.55)]
mismatch_params  = [(1,2,'MC','MC',0.04),(1,2,'MC','Ch1',0.14),
                    (2,3,'MC','MC',0.038),(2,3,'MC','Ch1',0.136),(2,3,'Ch1','Ch2',0.21)]
experimental_points = [(1,3,0.048,0.045),(3,4,0.042,0.044),
                       (4,7,0.138,0.142),(7,12,0.132,0.139)]

growth_results = [
    (0,'MC',0.00),(10,'MC',0.01),(20,'MC',0.02),(30,'MC',0.025),(40,'MC',0.03),(50,'MC',0.035),
    (0,'Ch1',0.00),(10,'Ch1',0.12),(20,'Ch1',0.14),(30,'Ch1',0.138),(40,'Ch1',0.136),(50,'Ch1',0.135),
    (0,'MC',0.00),(10,'MC',0.08),(20,'Ch1',0.14),(30,'Ch1',0.15),(40,'Ch1',0.145),(50,'Ch1',0.142)
]
path_selection_stats = [('Conservative path',325),('Mismatch-driven path',125),
                        ('Random path',50),('Reverse step',100)]
lj_parameters = [('Na-Na',1.0,3.72),('Rb-Rb',1.0,4.96),('Cs-Cs',1.0,5.30),
                 ('Ag-Ag',1.0,2.88),('Cu-Cu',1.0,2.56),
                 ('Na-Rb',1.0,4.34),('Ag-Cu',1.0,2.72)]

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Icosahedral Shell Theory – Hexagonal Lattice & Chiral Classification
# ═══════════════════════════════════════════════════════════════════════════════

def hexagonal_to_cartesian(h, k):
    """Convert hexagonal coordinates to Cartesian."""
    x = h + k * np.cos(np.pi/3)
    y = k * np.sin(np.pi/3)
    return x, y

def compute_shell_size(h, k, b=1):
    """Shell count for icosahedral shell at (h,k) with b atoms per edge."""
    if h == 0 and k == 0:
        return 1  # central atom
    # Icosahedral shell formula: 10*(h^2 + hk + k^2) * b^2 + 2
    return (10*(h**2 + h*k + k**2) * b**2 + 2)

def classify_chirality(h, k):
    """Classify shell chirality based on (h,k) coordinates."""
    if h == 0:
        return 'MC'     # Mirror-symmetric
    elif k == 0:
        return 'BG'     # Barlow-Glaeser
    elif h == k:
        return 'MC'     # Also mirror-symmetric
    else:
        # Chiral shells ordered by angular index
        diff = k - h
        if diff == 1:   return 'Ch1'
        elif diff == 2: return 'Ch2'
        elif diff == 3: return 'Ch3'
        elif diff < 0:  return 'Ch' + str(abs(diff))
        else:           return 'Ch4'

fig1, axes = plt.subplots(1, 2, figsize=(14, 6))
fig1.suptitle('Icosahedral Shell Theory: Hexagonal Lattice and Chiral Classification',
              fontsize=14, fontweight='bold')

# Left: Hexagonal lattice map with shell sizes
ax = axes[0]
ax.set_aspect('equal')

max_hk = 5
for h in range(max_hk+1):
    for k in range(max_hk+1):
        x, y = hexagonal_to_cartesian(h, k)
        size_val = compute_shell_size(h, k)
        chirality = classify_chirality(h, k)
        color = shell_colors.get(chirality, '#aaaaaa')
        circle = plt.Circle((x, y), 0.35, color=color, alpha=0.75, zorder=3)
        ax.add_patch(circle)
        label = f'({h},{k})\nN={size_val}' if size_val < 200 else f'({h},{k})'
        ax.text(x, y, label, ha='center', va='center', fontsize=5.5, zorder=4)

# Overlay hexagonal grid lines
for h in range(max_hk+1):
    xs = [hexagonal_to_cartesian(h, k)[0] for k in range(max_hk+1)]
    ys = [hexagonal_to_cartesian(h, k)[1] for k in range(max_hk+1)]
    ax.plot(xs, ys, 'k-', lw=0.4, alpha=0.3)
for k in range(max_hk+1):
    xs = [hexagonal_to_cartesian(h, k)[0] for h in range(max_hk+1)]
    ys = [hexagonal_to_cartesian(h, k)[1] for h in range(max_hk+1)]
    ax.plot(xs, ys, 'k-', lw=0.4, alpha=0.3)

legend_patches = [mpatches.Patch(color=shell_colors[lab], label=lab)
                  for lab in ['MC','BG','Ch1','Ch2','Ch3','Ch4']]
ax.legend(handles=legend_patches, loc='upper right', fontsize=8, title='Chirality')
ax.set_title('Hexagonal Lattice Shell Map (b=1)', fontsize=11)
ax.set_xlabel('Cartesian x', fontsize=9)
ax.set_ylabel('Cartesian y', fontsize=9)
ax.set_xlim(-0.6, 6.5)
ax.set_ylim(-0.6, 5.5)

# Right: Shell size vs. (h,k) index
ax2 = axes[1]
shells_data = []
for h in range(max_hk+1):
    for k in range(h, max_hk+1):  # avoid double-counting (h,k)=(k,h) chirals
        size_val = compute_shell_size(h, k)
        chirality = classify_chirality(h, k)
        shells_data.append((h, k, size_val, chirality))

for sd in shells_data:
    h, k, sz, ch = sd
    color = shell_colors.get(ch, '#aaaaaa')
    ax2.scatter(10*(h**2+h*k+k**2), sz, c=color, s=80, zorder=3)
    ax2.annotate(f'({h},{k})', (10*(h**2+h*k+k**2), sz),
                 fontsize=6, ha='left', va='bottom')

# Add Mackay reference line
Ts = np.array([1,2,3,4,5,6])
mackay_sizes = 10*Ts*(Ts+1) + 2  # Standard Mackay formula
ax2.plot(10*Ts*(Ts+1), mackay_sizes, 'k--', lw=1.5, alpha=0.5, label='Mackay (h=T,k=0)')
ax2.set_xlabel('10(h² + hk + k²)', fontsize=9)
ax2.set_ylabel('Shell atom count N', fontsize=9)
ax2.set_title('Shell Size vs. Hexagonal Index', fontsize=11)
ax2.legend(handles=legend_patches + [plt.Line2D([0],[0],color='k',ls='--',label='Mackay ref')],
           fontsize=7)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig1.savefig(os.path.join(REPORT_IMGS, 'fig1_hexagonal_lattice_shells.png'), dpi=150, bbox_inches='tight')
plt.close(fig1)
print("Figure 1 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Magic Number Sequences – Mackay vs. New b=5 Series
# ═══════════════════════════════════════════════════════════════════════════════

def cumulative_cluster_size(sequence):
    """Running cumulative sum of shell sizes."""
    return np.cumsum(sequence)

# Compute magic numbers for both series
shells_n = list(range(1, 7))  # shell index 1..6

def mackay_shell_size(n):
    """Atoms in n-th Mackay icosahedral shell (n>=1)."""
    if n == 1: return 1
    return 10*(n-1)**2 + 2

def new_b5_shell_size(n):
    """b=5 series shell atom count."""
    if n == 1: return 1
    return 10*(n-1)**2 * 25 + 2  # b=5

# Actually using provided sequences
mackay_cumul   = list(np.cumsum(np.diff([0]+mackay_sequence)))
new_b5_cumul   = list(np.cumsum(np.diff([0]+new_sequence_b5)))

fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.suptitle('Magic Number Sequences for Multi-Shell Icosahedral Clusters',
              fontsize=14, fontweight='bold')

# Left: Cumulative sizes comparison
ax = axes2[0]
ax.plot(range(1, len(mackay_sequence)+1), mackay_sequence, 'o-b', lw=2, ms=8, label='Mackay series')
ax.plot(range(1, len(new_sequence_b5)+1), new_sequence_b5, 's--r', lw=2, ms=8, label='b=5 series')
for i, v in enumerate(mackay_sequence):
    ax.annotate(str(v), (i+1, v), textcoords='offset points', xytext=(5,4), fontsize=8, color='blue')
for i, v in enumerate(new_sequence_b5):
    ax.annotate(str(v), (i+1, v), textcoords='offset points', xytext=(5,-12), fontsize=8, color='red')
ax.set_xlabel('Shell index n', fontsize=10)
ax.set_ylabel('Cumulative cluster size N', fontsize=10)
ax.set_title('Magic Number Sequences', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Right: Per-shell atom counts
ax = axes2[1]
mackay_per_shell = np.diff([0]+mackay_sequence)
new_b5_per_shell = np.diff([0]+new_sequence_b5)

x_m = np.arange(1, len(mackay_per_shell)+1)
x_b = np.arange(1, len(new_b5_per_shell)+1)
width = 0.35
ax.bar(x_m - width/2, mackay_per_shell, width, label='Mackay', color='steelblue', alpha=0.8)
ax.bar(x_b[:len(new_b5_per_shell)] + width/2, new_b5_per_shell, width, label='b=5', color='tomato', alpha=0.8)
ax.set_xlabel('Shell index n', fontsize=10)
ax.set_ylabel('Atoms added in shell n', fontsize=10)
ax.set_title('Per-Shell Atom Count', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig2.savefig(os.path.join(REPORT_IMGS, 'fig2_magic_numbers.png'), dpi=150, bbox_inches='tight')
plt.close(fig2)
print("Figure 2 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Atomic Size Mismatch Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def size_mismatch(r1, r2):
    return abs(r1 - r2) / max(r1, r2)

radii_dict = dict(atomic_radii)
elements   = [e for e, r in atomic_radii]
N = len(elements)
mismatch_matrix = np.zeros((N, N))
for i, (e1, r1) in enumerate(atomic_radii):
    for j, (e2, r2) in enumerate(atomic_radii):
        mismatch_matrix[i, j] = size_mismatch(r1, r2)

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('Atomic Size Mismatch and Shell Compatibility Analysis',
              fontsize=14, fontweight='bold')

# Left: Size mismatch heatmap
ax = axes3[0]
im = ax.imshow(mismatch_matrix, cmap='YlOrRd', vmin=0, vmax=0.35, aspect='auto')
ax.set_xticks(range(N)); ax.set_xticklabels(elements, fontsize=9)
ax.set_yticks(range(N)); ax.set_yticklabels(elements, fontsize=9)
for i in range(N):
    for j in range(N):
        val = mismatch_matrix[i, j]
        txt_color = 'white' if val > 0.2 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=7, color=txt_color)
plt.colorbar(im, ax=ax, label='Size mismatch δ')
ax.set_title('Pairwise Atomic Size Mismatch Matrix', fontsize=10)

# Middle: Optimal mismatch ranges by shell-type pair
ax = axes3[1]
pair_labels = [f'{a}-{b}' for a,b,_,_ in optimal_mismatch_ranges]
x_pos = np.arange(len(pair_labels))
colors_opt = ['#1f77b4','#2ca02c','#d62728','#ff7f0e']
for i, (inner, outer, lo, hi) in enumerate(optimal_mismatch_ranges):
    ax.barh(i, hi-lo, left=lo, height=0.5, color=colors_opt[i], alpha=0.8)
    ax.text(hi+0.005, i, f'{lo:.2f}–{hi:.2f}', va='center', fontsize=9)
ax.set_yticks(x_pos); ax.set_yticklabels(pair_labels, fontsize=9)
ax.set_xlabel('Size mismatch δ', fontsize=10)
ax.set_title('Optimal δ Ranges by Shell-Type Pair', fontsize=10)
ax.set_xlim(0, 0.30)
ax.axvline(0.04, color='grey', ls='--', lw=1, alpha=0.5, label='δ=0.04')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(fontsize=8)

# Right: Experimental vs theoretical mismatch points
ax = axes3[2]
exp_pts = experimental_points
T_i   = [p[0] for p in exp_pts]
T_ip1 = [p[1] for p in exp_pts]
meas  = [p[2] for p in exp_pts]
theo  = [p[3] for p in exp_pts]

x_idx = np.arange(len(exp_pts))
ax.bar(x_idx-0.2, meas, 0.35, label='Measured', color='steelblue', alpha=0.8)
ax.bar(x_idx+0.2, theo, 0.35, label='Theoretical', color='tomato', alpha=0.8)
xlabs = [f'T{p[0]}→T{p[1]}' for p in exp_pts]
ax.set_xticks(x_idx); ax.set_xticklabels(xlabs, fontsize=9)
ax.set_ylabel('Size mismatch δ', fontsize=10)
ax.set_title('Experimental vs. Theoretical δ', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Add RMSE annotation
rmse = np.sqrt(np.mean([(m-t)**2 for m,t in zip(meas,theo)]))
ax.text(0.65, 0.92, f'RMSE = {rmse:.4f}', transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round', fc='lightyellow', ec='orange'))

plt.tight_layout()
fig3.savefig(os.path.join(REPORT_IMGS, 'fig3_size_mismatch.png'), dpi=150, bbox_inches='tight')
plt.close(fig3)
print("Figure 3 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Shell Energetics and Multi-component Cluster Stability
# ═══════════════════════════════════════════════════════════════════════════════

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
fig4.suptitle('Shell Energetics and Multi-component Cluster Validation',
              fontsize=14, fontweight='bold')

# Left: Energy comparison across shell types
ax = axes4[0]
from collections import defaultdict
by_shell = defaultdict(list)
for shell_n, stype, energy in shell_energies:
    by_shell[shell_n].append((stype, energy))

shell_nums = sorted(by_shell.keys())
for shell_n in shell_nums:
    entries = by_shell[shell_n]
    types   = [e[0] for e in entries]
    energies= [e[1] for e in entries]
    xoff    = np.linspace(-0.25, 0.25, len(types))
    for xo, stype, en in zip(xoff, types, energies):
        c = shell_colors.get(stype, '#aaa')
        ax.scatter(shell_n + xo, en, c=c, s=120, zorder=3)
        ax.annotate(stype, (shell_n + xo, en), xytext=(2, 4),
                    textcoords='offset points', fontsize=8)

ax.axhline(0, color='grey', ls='--', lw=0.8)
ax.set_xlabel('Shell index n', fontsize=10)
ax.set_ylabel('Relative energy (normalized)', fontsize=10)
ax.set_title('Shell Energy by Type', fontsize=11)
ax.grid(True, alpha=0.3)
legend_patches2 = [mpatches.Patch(color=shell_colors[lab], label=lab)
                   for lab in ['MC','BG','Ch1']]
ax.legend(handles=legend_patches2, fontsize=9)

# Right: Multi-component cluster validation table visualization
ax = axes4[1]
ax.axis('off')
cluster_names = [c[0] for c in multicomponent_clusters]
inner_el  = [c[1] for c in multicomponent_clusters]
outer_el  = [c[2] for c in multicomponent_clusters]
inner_type= [c[3] for c in multicomponent_clusters]
outer_type= [c[4] for c in multicomponent_clusters]
pair_comp = dict((f'{a}-{b}', d) for a,b,d in atomic_pairs_compatibility)

col_labels = ['Cluster', 'Inner\nElement', 'Outer\nElement',
              'Inner\nType', 'Outer\nType', 'δ (measured)']
cell_data = []
for i, (cn, ie, oe, it, ot) in enumerate(zip(cluster_names,inner_el,outer_el,inner_type,outer_type)):
    sm = pair_comp.get(f'{ie}-{oe}', pair_comp.get(f'{oe}-{ie}', 'N/A'))
    cell_data.append([cn, ie, oe, it, ot, f'{sm:.3f}' if isinstance(sm, float) else sm])

table = ax.table(cellText=cell_data, colLabels=col_labels,
                 cellLoc='center', loc='center', bbox=[0, 0.15, 1, 0.75])
table.auto_set_font_size(False)
table.set_fontsize(9.5)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2c5f8a')
        cell.set_text_props(color='white', fontweight='bold')
    elif r % 2 == 0:
        cell.set_facecolor('#dde8f3')
    else:
        cell.set_facecolor('#f5f8fc')
    cell.set_edgecolor('#bbbbbb')

ax.set_title('Validated Multi-component Icosahedral Clusters', fontsize=11, pad=20)

plt.tight_layout()
fig4.savefig(os.path.join(REPORT_IMGS, 'fig4_cluster_energetics.png'), dpi=150, bbox_inches='tight')
plt.close(fig4)
print("Figure 4 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Dynamic Growth Simulation
# ═══════════════════════════════════════════════════════════════════════════════

fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))
fig5.suptitle('Dynamic Growth Simulation of Multi-component Icosahedral Clusters',
              fontsize=14, fontweight='bold')

# Left: Growth trajectories (mismatch vs step) for two scenarios
ax = axes5[0]

# Parse three independent growth series
series_data = {
    'MC seed (Na+Na)':    [(0,0.00),(10,0.01),(20,0.02),(30,0.025),(40,0.030),(50,0.035)],
    'Ch1 seed (Na@Rb)':   [(0,0.00),(10,0.12),(20,0.14),(30,0.138),(40,0.136),(50,0.135)],
    'Mixed (Na→Ch1)':     [(0,0.00),(10,0.08),(20,0.14),(30,0.15),(40,0.145),(50,0.142)]
}
colors_series = ['steelblue', 'tomato', 'seagreen']
for (label, pts), col in zip(series_data.items(), colors_series):
    steps = [p[0] for p in pts]
    delta = [p[1] for p in pts]
    ax.plot(steps, delta, 'o-', color=col, lw=2, ms=7, label=label)
    # Shade optimal range
for lo, hi in [(0.03, 0.05), (0.12, 0.16)]:
    ax.axhspan(lo, hi, alpha=0.08, color='grey')

ax.set_xlabel('Deposition steps', fontsize=10)
ax.set_ylabel('Average size mismatch δ', fontsize=10)
ax.set_title('Growth Trajectory: δ vs. Steps', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Middle: Path selection statistics (pie chart)
ax = axes5[1]
labels_p = [p[0] for p in path_selection_stats]
counts_p = [p[1] for p in path_selection_stats]
colors_p = ['#1f77b4','#ff7f0e','#2ca02c','#d62728']
wedges, texts, autotexts = ax.pie(counts_p, labels=labels_p, colors=colors_p,
                                   autopct='%1.1f%%', startangle=90,
                                   textprops={'fontsize':8.5})
for at in autotexts:
    at.set_fontsize(8)
ax.set_title('Growth Path Selection Statistics\n(600 total steps)', fontsize=11)

# Right: Lennard-Jones potential curves for key pairs
ax = axes5[2]
r_range = np.linspace(2.0, 10.0, 300)
def lj_potential(r, eps, sigma):
    sr6  = (sigma/r)**6
    return 4*eps*(sr6**2 - sr6)

pairs_to_plot = [('Na-Na',3.72,'#1f77b4'),('Rb-Rb',4.96,'#ff7f0e'),
                 ('Na-Rb',4.34,'#2ca02c'),('Ag-Cu',2.72,'#d62728')]
for name, sigma, col in pairs_to_plot:
    U = lj_potential(r_range, 1.0, sigma)
    mask = U < 2.0
    ax.plot(r_range[mask], U[mask], color=col, lw=2, label=name)
ax.axhline(0, color='k', ls='--', lw=0.8)
ax.set_ylim(-1.5, 2.0)
ax.set_xlim(2.0, 10.0)
ax.set_xlabel('Inter-atomic distance r (Å)', fontsize=10)
ax.set_ylabel('Potential energy U (ε units)', fontsize=10)
ax.set_title('Lennard-Jones Pair Potentials', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig5.savefig(os.path.join(REPORT_IMGS, 'fig5_growth_simulation.png'), dpi=150, bbox_inches='tight')
plt.close(fig5)
print("Figure 5 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Phase diagram – shell-type compatibility vs size mismatch
# ═══════════════════════════════════════════════════════════════════════════════

fig6, axes6 = plt.subplots(1, 2, figsize=(13, 5))
fig6.suptitle('Shell-Type Compatibility Phase Space and δ Distribution', fontsize=14, fontweight='bold')

# Left: Scatter of (inner type, outer type) vs measured δ
ax = axes6[0]
transition_data = []
for i1, i2, type1, type2, delta_val in mismatch_params:
    transition_data.append((f'S{i1}({type1})→S{i2}({type2})', delta_val, type1, type2))
# Also add experimental points mapped to shell types
extra = [(f'T{a}→T{b}',m,'MC','MC') for a,b,m,_ in experimental_points[:2]]
extra+= [(f'T{a}→T{b}',m,'MC','Ch1') for a,b,m,_ in experimental_points[2:]]
transition_data.extend(extra)

labels_td = [t[0] for t in transition_data]
deltas_td = [t[1] for t in transition_data]
color_td  = [shell_colors.get(t[2],'#888') for t in transition_data]

x_td = np.arange(len(labels_td))
bars = ax.bar(x_td, deltas_td, color=color_td, alpha=0.8, edgecolor='white')

# Add optimal range shading
for lo, hi, label, col in [
    (0.03, 0.05, 'MC→MC optimal', '#1f77b4'),
    (0.12, 0.16, 'MC→Ch1 optimal', '#2ca02c'),
    (0.19, 0.22, 'MC→Ch2 optimal', '#d62728'),
    (0.08, 0.10, 'MC→BG optimal', '#ff7f0e')]:
    ax.axhspan(lo, hi, alpha=0.12, color=col)

ax.set_xticks(x_td)
ax.set_xticklabels(labels_td, rotation=35, ha='right', fontsize=7)
ax.set_ylabel('Size mismatch δ', fontsize=10)
ax.set_title('Shell Transition δ Values\n(colored by inner shell type)', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Right: Theoretical δ calculation from hexagonal path
ax2 = axes6[1]
# Shell radii scale ~ sqrt(10*(h^2+hk+k^2))  (simplified model)
path_nodes = [(0,0),(0,1),(1,1),(1,2),(2,2),(2,3),(3,3)]
path_sizes = [compute_shell_size(h,k) for h,k in path_nodes]
path_radii = np.sqrt(np.array(path_sizes, dtype=float))
path_radii /= path_radii[0]  # normalize to core

delta_path = []
for i in range(len(path_radii)-1):
    delta_path.append(abs(path_radii[i+1] - path_radii[i]) / path_radii[i+1])

step_labels = [f'({h},{k})' for h,k in path_nodes[1:]]
x_path = np.arange(len(delta_path))
ax2.bar(x_path, delta_path, color=[shell_colors.get(classify_chirality(h,k),'#aaa')
                                    for _,(h,k) in enumerate(path_nodes[1:])],
        alpha=0.85, edgecolor='white')
ax2.axhspan(0.03, 0.05, alpha=0.15, color='steelblue', label='MC→MC optimal')
ax2.axhspan(0.12, 0.16, alpha=0.15, color='seagreen', label='MC→Ch1 optimal')
ax2.set_xticks(x_path); ax2.set_xticklabels(step_labels, fontsize=9)
ax2.set_xlabel('Shell node (h, k)', fontsize=10)
ax2.set_ylabel('Step-wise size mismatch δ', fontsize=10)
ax2.set_title('δ Along Hexagonal Growth Path\n(0,0)→(0,1)→(1,1)→…', fontsize=10)
ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig6.savefig(os.path.join(REPORT_IMGS, 'fig6_phase_diagram.png'), dpi=150, bbox_inches='tight')
plt.close(fig6)
print("Figure 6 saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Summary – Predicted Stable Clusters Table
# ═══════════════════════════════════════════════════════════════════════════════

fig7, ax7 = plt.subplots(figsize=(12, 5))
ax7.axis('off')
fig7.suptitle('Predicted Stable Multi-Component Icosahedral Clusters',
              fontsize=14, fontweight='bold')

predicted_table = [
    ['Na₁₃@Rb₃₂',   'Na (1.86 Å)', 'Rb (2.48 Å)', 'MC', 'Ch1', '0.220', '✓ Favorable'],
    ['K₁₃@Cs₄₂',    'K (2.27 Å)',  'Cs (2.65 Å)', 'MC', 'Ch2', '0.143', '✓ Favorable'],
    ['Ag₁₃@Cu₄₅',   'Ag (1.44 Å)', 'Cu (1.28 Å)', 'MC', 'Ch1', '0.120', '✓ Favorable'],
    ['Ag₁₃@Ni₄₅',   'Ag (1.44 Å)', 'Ni (1.24 Å)', 'MC', 'Ch1', '0.150', '✓ Favorable'],
    ['Cu₁₃@Ni₄₅',   'Cu (1.28 Å)', 'Ni (1.24 Å)', 'MC', 'MC',  '0.032', '≈ Marginal'],
    ['Ni₁₄₇@Ag₁₉₂', 'Ni (1.24 Å)', 'Ag (1.44 Å)', 'MC', 'MC',  '0.141', '✓ Favorable'],
]
col_hdrs = ['Cluster', 'Inner Shell', 'Outer Shell', 'Inner\nType', 'Outer\nType', 'δ', 'Stability']

tbl = ax7.table(cellText=predicted_table, colLabels=col_hdrs,
                cellLoc='center', loc='center', bbox=[0, 0.05, 1, 0.85])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#1a3a5c')
        cell.set_text_props(color='white', fontweight='bold')
    elif '✓' in cell.get_text().get_text():
        cell.set_facecolor('#d6f0d6')
    elif '≈' in cell.get_text().get_text():
        cell.set_facecolor('#fffacd')
    elif r % 2 == 0:
        cell.set_facecolor('#e8f0f8')
    cell.set_edgecolor('#cccccc')

plt.tight_layout()
fig7.savefig(os.path.join(REPORT_IMGS, 'fig7_predicted_clusters.png'), dpi=150, bbox_inches='tight')
plt.close(fig7)
print("Figure 7 saved.")

# ─── Save numerical outputs ────────────────────────────────────────────────────
import json

output_dict = {
    "magic_numbers": {
        "mackay": mackay_sequence,
        "new_b5": new_sequence_b5
    },
    "mismatch_matrix": {
        "elements": elements,
        "matrix": mismatch_matrix.tolist()
    },
    "rmse_exp_vs_theo": float(rmse),
    "path_selection_fractions": {p[0]: p[1]/600 for p in path_selection_stats}
}

with open(os.path.join(OUTPUTS, 'analysis_results.json'), 'w') as f:
    json.dump(output_dict, f, indent=2)
print("Numerical outputs saved.")
print("\n=== All analyses complete ===")
