"""
data_exploration.py
Parse and visualize the M-AI-Synth dataset.
"""

import ast
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_001_20260401_224352"
DATA_PATH = os.path.join(WORKSPACE, "data/M-AI-Synth__Materials_AI_Dataset_.txt")
OUT_DIR   = os.path.join(WORKSPACE, "outputs")
FIG_DIR   = os.path.join(WORKSPACE, "report/images")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. Parse raw text ────────────────────────────────────────────────────────
def parse_dataset(path):
    sections = {}
    current_section = None
    arrays = []
    current_arrays = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if current_section is not None:
                    sections[current_section] = current_arrays
                current_section = line.lstrip("# ").strip()
                current_arrays = []
            else:
                try:
                    arr = ast.literal_eval(line)
                    current_arrays.append(np.array(arr, dtype=float))
                except Exception:
                    pass

    if current_section is not None:
        sections[current_section] = current_arrays
    return sections

sections = parse_dataset(DATA_PATH)

print("=== Parsed sections ===")
for name, arrs in sections.items():
    print(f"\n[{name}]")
    for i, a in enumerate(arrs):
        print(f"  Array {i}: shape={a.shape}  min={a.min():.4f}  max={a.max():.4f}  mean={a.mean():.4f}")

# ── 2. Extract sub-datasets ──────────────────────────────────────────────────
# Section 1: property_prediction.py data
prop = sections["文件1: property_prediction.py 数据"]
atom_features   = prop[0]   # shape (100,)   – atomic numbers (all 5 = Boron)
coordinates     = prop[1]   # shape (112,)   – atom positions (grid coordinates)
edge_index_flat = prop[2]   # shape (20,)    – i,j pairs flattened → 10 edges
edge_features   = prop[3]   # shape (97,)    – bond features

# Section 2: structure_generation.py data
struct = sections["文件2: structure_generation.py 数据"]
lattice_a = struct[0]   # shape (101,)
lattice_b = struct[1]   # shape (101,)

# Section 3: autonomous_optimization.py data
opt_raw = sections["文件3: autonomous_optimization.py 数据"]
T_range    = opt_raw[0]   # [200, 500]
t_range    = opt_raw[1]   # [10, 30]
T_target   = opt_raw[2]   # [350]
t_target   = opt_raw[3]   # [20]
lr         = opt_raw[4]   # [0.1]
n_init     = opt_raw[5]   # [10]

print("\n=== Property prediction ===")
print(f"  Atoms:        {len(atom_features)} atoms, element = {np.unique(atom_features.astype(int))}")
print(f"  Coordinates:  {len(coordinates)} values")
print(f"  Edge index:   {len(edge_index_flat)} values → {len(edge_index_flat)//2} edges")
print(f"  Edge feats:   {len(edge_features)} values")

print("\n=== Structure generation ===")
print(f"  Lattice a: {len(lattice_a)} samples, range [{lattice_a.min():.4f}, {lattice_a.max():.4f}]")
print(f"  Lattice b: {len(lattice_b)} samples, range [{lattice_b.min():.4f}, {lattice_b.max():.4f}]")

print("\n=== Optimization ===")
print(f"  T range: {T_range}, target T={T_target}")
print(f"  t range: {t_range}, target t={t_target}")
print(f"  lr={lr}, n_init={n_init}")

# ── 3. Construct crystal graph ───────────────────────────────────────────────
# 100 atoms on a 10×10 grid; coordinates are x-values arranged row-by-row
# We have 112 coordinate values; the first 100 match the 100 atoms,
# the remaining likely correspond to a stretched axis / extra padding.
# Use first 100 as x-coords; generate y-coords from the grid layout.
n_atoms = len(atom_features)

# Reconstruct 2D positions: 10 rows of 10 atoms, origin shifted per row
x_coords = coordinates[:n_atoms]   # use first 100 positions as x-coordinate series
# The pattern in x_coords: 10 atoms per row, each row starts 0.2 higher than prev
# infer y from row index
row_size = 10
y_coords = np.repeat(np.arange(n_atoms // row_size), row_size) * 0.5   # 0.5 Å spacing

# Edge index: reshape from flat [i0,j0,i1,j1,...] to (n_edges,2)
edges = edge_index_flat.astype(int).reshape(-1, 2)

# ── 4. Lattice analysis ──────────────────────────────────────────────────────
# The 7 unique lattice parameter values cycle through both axes
unique_a = np.unique(np.round(lattice_a, 4))
unique_b = np.unique(np.round(lattice_b, 4))
print(f"\n  Unique lattice-a values: {unique_a}")
print(f"  Unique lattice-b values: {unique_b}")

# Compute unit cell volume proxy: V = a * b * c (assume c=a for tetragonal)
volume_proxy = lattice_a * lattice_b   # a*b product (c assumed ~a)

# ── 5. Save parsed data ──────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "atom_features.npy"), atom_features)
np.save(os.path.join(OUT_DIR, "x_coords.npy"), x_coords)
np.save(os.path.join(OUT_DIR, "y_coords.npy"), y_coords)
np.save(os.path.join(OUT_DIR, "edges.npy"), edges)
np.save(os.path.join(OUT_DIR, "edge_features.npy"), edge_features)
np.save(os.path.join(OUT_DIR, "lattice_a.npy"), lattice_a)
np.save(os.path.join(OUT_DIR, "lattice_b.npy"), lattice_b)
np.save(os.path.join(OUT_DIR, "volume_proxy.npy"), volume_proxy)

summary = {
    "n_atoms": int(n_atoms),
    "n_edges": int(len(edges)),
    "element": int(atom_features[0]),
    "lattice_samples": int(len(lattice_a)),
    "unique_a_values": unique_a.tolist(),
    "unique_b_values": unique_b.tolist(),
    "T_range": T_range.tolist(),
    "t_range": t_range.tolist(),
    "T_target": float(T_target[0]),
    "t_target": float(t_target[0]),
}
with open(os.path.join(OUT_DIR, "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nData saved to outputs/")

# ── 6. Data overview figure ───────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.suptitle("M-AI-Synth Dataset Overview", fontsize=16, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# 6a. Crystal graph: atom positions + edges
ax1 = fig.add_subplot(gs[0, :2])
ax1.scatter(x_coords, y_coords, c='steelblue', s=80, zorder=3, label='B atoms')
for e in edges:
    i, j = e[0], e[1]
    if i < n_atoms and j < n_atoms:
        ax1.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]],
                 'gray', linewidth=1, alpha=0.6, zorder=2)
ax1.set_title("Crystal Graph: Boron Layer (10×10 supercell)", fontsize=11)
ax1.set_xlabel("x-coordinate (Å)")
ax1.set_ylabel("y-coordinate (Å)")
ax1.legend(loc='upper left', fontsize=9)
ax1.set_aspect('equal', adjustable='box')
ax1.grid(True, alpha=0.3)

# 6b. Atom feature distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar([5], [100], color='steelblue', width=0.6)
ax2.set_title("Atom Feature Distribution\n(Atomic Number)", fontsize=11)
ax2.set_xlabel("Atomic Number")
ax2.set_ylabel("Count")
ax2.set_xticks([5])
ax2.set_xticklabels(['B (Z=5)'])
ax2.grid(True, alpha=0.3, axis='y')

# 6c. Edge feature distribution
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(edge_features, 'o-', color='darkorange', markersize=4, linewidth=1.5)
ax3.set_title("Bond Feature Sequence\n(97 values)", fontsize=11)
ax3.set_xlabel("Feature Index")
ax3.set_ylabel("Feature Value")
ax3.grid(True, alpha=0.3)

# 6d. Lattice parameter distribution
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(lattice_a, bins=14, color='forestgreen', alpha=0.7, edgecolor='white',
         label='Lattice a')
ax4.hist(lattice_b, bins=14, color='crimson', alpha=0.7, edgecolor='white',
         label='Lattice b')
ax4.axvline(lattice_a.mean(), color='forestgreen', linestyle='--', linewidth=2)
ax4.axvline(lattice_b.mean(), color='crimson',     linestyle='--', linewidth=2)
ax4.set_title("Lattice Parameter Distribution\n(Structure Generation)", fontsize=11)
ax4.set_xlabel("Lattice Parameter (Å)")
ax4.set_ylabel("Count")
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 6e. Lattice a vs b correlation
ax5 = fig.add_subplot(gs[1, 2])
ax5.scatter(lattice_a, lattice_b, c=np.arange(len(lattice_a)),
            cmap='viridis', s=20, alpha=0.6)
ax5.set_title("Lattice a vs b Correlation\n(101 structures)", fontsize=11)
ax5.set_xlabel("Lattice a (Å)")
ax5.set_ylabel("Lattice b (Å)")
ax5.grid(True, alpha=0.3)

# 6f. Lattice parameter evolution
ax6 = fig.add_subplot(gs[2, :2])
ax6.plot(lattice_a, color='forestgreen', linewidth=1.5, label='Lattice a', alpha=0.8)
ax6.plot(lattice_b, color='crimson',     linewidth=1.5, label='Lattice b', alpha=0.8)
ax6.set_title("Lattice Parameter Sequence (101 crystal structures)", fontsize=11)
ax6.set_xlabel("Structure Index")
ax6.set_ylabel("Lattice Parameter (Å)")
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# 6g. Optimization search space
ax7 = fig.add_subplot(gs[2, 2])
T_fine = np.linspace(T_range[0], T_range[1], 200)
t_fine = np.linspace(t_range[0], t_range[1], 200)
TT, tt = np.meshgrid(T_fine, t_fine)
# Mock yield surface: Gaussian peak at target
yield_surf = np.exp(-((TT - T_target[0])**2 / (2*50**2) + (tt - t_target[0])**2 / (2*3**2)))
cs = ax7.contourf(TT, tt, yield_surf, levels=15, cmap='plasma')
ax7.scatter([T_target[0]], [t_target[0]], c='white', s=100, marker='*',
            zorder=5, label=f'Optimum\n(T={T_target[0]}°C, t={t_target[0]}min)')
ax7.set_title("Synthesis Optimization Space", fontsize=11)
ax7.set_xlabel("Temperature (°C)")
ax7.set_ylabel("Reaction Time (min)")
ax7.legend(fontsize=8, loc='upper left')
plt.colorbar(cs, ax=ax7, label='Yield proxy')

fig.savefig(os.path.join(FIG_DIR, "fig1_data_overview.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1 saved: fig1_data_overview.png")
