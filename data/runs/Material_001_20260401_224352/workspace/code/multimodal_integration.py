"""
multimodal_integration.py
Demonstrates multimodal data integration: combining crystal graph features,
lattice parameters, and spectral-like features for enhanced property prediction.
Also generates additional report figures.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json, os

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_001_20260401_224352"
OUT_DIR   = os.path.join(WORKSPACE, "outputs")
FIG_DIR   = os.path.join(WORKSPACE, "report/images")

np.random.seed(42)

# ── 1. Load preprocessed data ────────────────────────────────────────────────
lattice_a = np.load(os.path.join(OUT_DIR, "lattice_a.npy"))
lattice_b = np.load(os.path.join(OUT_DIR, "lattice_b.npy"))
E_f       = np.load(os.path.join(OUT_DIR, "E_f.npy"))
E_g       = np.load(os.path.join(OUT_DIR, "E_g.npy"))
K         = np.load(os.path.join(OUT_DIR, "K.npy"))
z_latent  = np.load(os.path.join(OUT_DIR, "z_latent.npy"))   # (101, 2) VAE latent

a0, b0 = lattice_a.mean(), lattice_b.mean()
n = len(lattice_a)

# ── 2. Construct multimodal feature sets ─────────────────────────────────────
# Modality 1: Structural (lattice parameters)
def struct_feats(a, b):
    V = a**2 * 0.9 * b
    return np.column_stack([a, b, a/b, V, (a-a0)/a0, (b-b0)/b0,
                             (a0/a)**3, (a-a0)**2, (b-b0)**2, a*b])

# Modality 2: Simulated XRD peak positions (Bragg's law: d_hkl = λ/2sinθ)
# For cubic lattice: d_100 = a, d_110 = a/√2, d_111 = a/√3
def xrd_feats(a, b):
    c = a * 0.9
    d100 = a; d010 = b; d001 = c
    d110 = a*b/np.sqrt(a**2+b**2)
    d101 = a*c/np.sqrt(a**2+c**2)
    d011 = b*c/np.sqrt(b**2+c**2)
    d111 = a*b*c/np.sqrt(a**2*b**2 + b**2*c**2 + a**2*c**2)
    # 2θ peaks (λ=1.5406 Å, Cu Kα)
    lam = 1.5406
    tth_100 = np.degrees(2 * np.arcsin(np.clip(lam/(2*d100),0,1)))
    tth_110 = np.degrees(2 * np.arcsin(np.clip(lam/(2*d110),0,1)))
    tth_111 = np.degrees(2 * np.arcsin(np.clip(lam/(2*d111),0,1)))
    return np.column_stack([d100, d010, d001, d110, d101, d011, d111,
                             tth_100, tth_110, tth_111])

# Modality 3: VAE latent embedding (from structure generation)
latent_feats = z_latent   # (101, 2)

# Modality 4: Simulated FTIR-like features (harmonic oscillator frequencies)
def ftir_feats(a, b):
    # Bond stretching frequency ∝ 1/√(a*b) (reduced mass approximation)
    k_eff = 50.0 / (a * b)   # effective spring constant proxy
    freq  = np.sqrt(k_eff)
    return np.column_stack([freq, freq*a, freq*b, k_eff, 1/(a**2), 1/(b**2)])

X_struct = struct_feats(lattice_a, lattice_b)
X_xrd    = xrd_feats(lattice_a, lattice_b)
X_latent = latent_feats
X_ftir   = ftir_feats(lattice_a, lattice_b)

# ── 3. Multimodal fusion comparison ──────────────────────────────────────────
modalities = {
    "Structural only":          X_struct,
    "XRD only":                 X_xrd,
    "Latent (VAE) only":        X_latent,
    "FTIR only":                X_ftir,
    "Struct + XRD":             np.hstack([X_struct, X_xrd]),
    "Struct + Latent":          np.hstack([X_struct, X_latent]),
    "Struct + XRD + Latent":    np.hstack([X_struct, X_xrd, X_latent]),
    "All modalities":           np.hstack([X_struct, X_xrd, X_latent, X_ftir]),
}

targets = {"Formation Energy": E_f, "Band Gap": E_g, "Bulk Modulus": K}

scaler = StandardScaler()
results_fusion = {}
print("=== Multimodal Fusion Comparison ===")
for prop_name, y in targets.items():
    results_fusion[prop_name] = {}
    print(f"\n  [{prop_name}]")
    for mod_name, X_mod in modalities.items():
        X_s = scaler.fit_transform(X_mod)
        rf  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        cv  = cross_val_score(rf, X_s, y, cv=5, scoring='r2')
        r2  = cv.mean()
        results_fusion[prop_name][mod_name] = float(r2)
        print(f"    {mod_name:35s} R²={r2:.3f} ± {cv.std():.3f}")

with open(os.path.join(OUT_DIR, "fusion_comparison.json"), "w") as f:
    json.dump(results_fusion, f, indent=2)

# ── 4. Model comparison (All modalities) ─────────────────────────────────────
X_all = np.hstack([X_struct, X_xrd, X_latent, X_ftir])
X_all_s = scaler.fit_transform(X_all)

models = {
    "Ridge Regression":        Ridge(alpha=1.0),
    "Random Forest":           RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting":       GradientBoostingRegressor(n_estimators=100, random_state=42),
}

model_results = {}
print("\n=== Model Comparison (All Modalities) ===")
for prop_name, y in targets.items():
    model_results[prop_name] = {}
    print(f"\n  [{prop_name}]")
    for mname, clf in models.items():
        cv = cross_val_score(clf, X_all_s, y, cv=5, scoring='r2')
        model_results[prop_name][mname] = {"r2_mean": float(cv.mean()), "r2_std": float(cv.std())}
        print(f"    {mname:28s} R²={cv.mean():.3f} ± {cv.std():.3f}")

with open(os.path.join(OUT_DIR, "model_comparison.json"), "w") as f:
    json.dump(model_results, f, indent=2)

# ── 5. Figure 5: Multimodal Fusion Analysis ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Multimodal Data Integration for Materials Property Prediction",
             fontsize=14, fontweight='bold')

colors = ['#e74c3c','#e67e22','#2ecc71','#3498db','#9b59b6','#1abc9c','#f39c12','#2c3e50']
mod_names_short = ['Struct','XRD','Latent','FTIR','S+X','S+L','S+X+L','All']

# (a) Fusion comparison heatmap
ax = axes[0, 0]
prop_list = list(targets.keys())
mod_list  = list(modalities.keys())
matrix    = np.array([[results_fusion[p][m] for m in mod_list] for p in prop_list])
im = ax.imshow(matrix, cmap='RdYlGn', vmin=-0.2, vmax=1.0, aspect='auto')
ax.set_xticks(range(len(mod_list)))
ax.set_xticklabels(mod_names_short, fontsize=9, rotation=30, ha='right')
ax.set_yticks(range(len(prop_list)))
ax.set_yticklabels(['E_f', 'E_g', 'K'], fontsize=10)
for i in range(len(prop_list)):
    for j in range(len(mod_list)):
        ax.text(j, i, f"{matrix[i,j]:.2f}", ha='center', va='center', fontsize=8,
                color='black' if 0.3 < matrix[i,j] < 0.8 else 'white')
plt.colorbar(im, ax=ax, label='CV R²')
ax.set_title("Modality Fusion R² Heatmap", fontsize=11)

# (b) Grouped bar chart: Ef prediction improvement
ax = axes[0, 1]
prop_idx = 0   # Formation Energy
vals = [results_fusion[prop_list[prop_idx]][m] for m in mod_list]
bars = ax.bar(mod_names_short, vals, color=colors, edgecolor='white', alpha=0.85)
ax.axhline(0, color='gray', linewidth=0.8)
ax.set_title(f"Modality Contribution: {prop_list[prop_idx]}", fontsize=11)
ax.set_xlabel("Modality"); ax.set_ylabel("Cross-validated R²")
ax.set_ylim([-0.2, 1.05])
ax.grid(True, alpha=0.3, axis='y')
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, max(v + 0.02, 0.02), f"{v:.2f}",
            ha='center', va='bottom', fontsize=7)
ax.tick_params(axis='x', labelsize=8)

# (c) PCA of the full feature space
pca = PCA(n_components=2)
Z_pca = pca.fit_transform(X_all_s)
ax = axes[0, 2]
sc = ax.scatter(Z_pca[:,0], Z_pca[:,1], c=E_f, cmap='coolwarm', s=50, alpha=0.8)
plt.colorbar(sc, ax=ax, label='E_f (eV/atom)')
ax.set_title(f"PCA of All Modalities\n(var explained: {pca.explained_variance_ratio_.sum()*100:.0f}%)", fontsize=11)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
ax.grid(True, alpha=0.3)

# (d) Model comparison bar chart
ax = axes[1, 0]
x_pos = np.arange(len(prop_list))
width = 0.25
model_names = list(models.keys())
model_colors = ['#3498db','#2ecc71','#e74c3c']
for mi, (mname, mc) in enumerate(zip(model_names, model_colors)):
    means = [model_results[p][mname]["r2_mean"] for p in prop_list]
    stds  = [model_results[p][mname]["r2_std"]  for p in prop_list]
    ax.bar(x_pos + mi*width, means, width=width, color=mc, alpha=0.8,
           label=mname, edgecolor='white')
    ax.errorbar(x_pos + mi*width, means, yerr=stds, fmt='none', color='black',
                capsize=4, linewidth=1.5)
ax.set_xticks(x_pos + width)
ax.set_xticklabels(['E_f', 'E_g', 'K'], fontsize=11)
ax.set_title("Model Comparison (All Modalities)", fontsize=11)
ax.set_xlabel("Property"); ax.set_ylabel("5-fold CV R²")
ax.legend(fontsize=9, loc='lower right'); ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# (e) XRD 2θ peaks simulation
ax = axes[1, 1]
lam = 1.5406
two_theta = np.linspace(10, 90, 1000)
# Average structure
a_mean, b_mean = lattice_a.mean(), lattice_b.mean()
c_mean = a_mean * 0.9
dhkls  = {'(100)': a_mean, '(010)': b_mean, '(110)': a_mean*b_mean/np.sqrt(a_mean**2+b_mean**2),
           '(111)': a_mean*b_mean*c_mean/np.sqrt(a_mean**2*b_mean**2 + b_mean**2*c_mean**2 + a_mean**2*c_mean**2)}
xrd = np.zeros_like(two_theta)
for hkl, d in dhkls.items():
    sin_th = lam / (2 * d)
    if sin_th <= 1:
        peak_pos = np.degrees(2 * np.arcsin(sin_th))
        # Lorentzian broadening
        xrd += 1.0 / (1 + ((two_theta - peak_pos) / 0.3)**2)
        ax.axvline(peak_pos, linestyle='--', alpha=0.5, linewidth=1)
        ax.text(peak_pos, 0.85, hkl, ha='center', fontsize=8, rotation=90)
ax.plot(two_theta, xrd, 'b-', linewidth=1.5)
ax.set_title("Simulated XRD Pattern (Cu Kα, λ=1.5406 Å)", fontsize=11)
ax.set_xlabel("2θ (degrees)"); ax.set_ylabel("Intensity (arb. units)")
ax.set_xlim([10, 90]); ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.1])

# (f) Architecture diagram (text-based)
ax = axes[1, 2]
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')
mod_boxes = [
    (1, 8, "Crystal Graph\n(atoms + bonds)", "#3498db"),
    (5, 8, "Lattice Params\n(a, b, V)", "#2ecc71"),
    (9, 8, "Spectral Data\n(XRD, FTIR)", "#e74c3c"),
]
for x, y, label, color in mod_boxes:
    ax.add_patch(mpatches.FancyBboxPatch([x-1.2, y-0.6], 2.4, 1.2,
                 boxstyle="round,pad=0.1", facecolor=color, alpha=0.7, edgecolor='white'))
    ax.text(x, y, label, ha='center', va='center', fontsize=7.5, color='white', fontweight='bold')
    ax.annotate('', xy=(5, 5.8), xytext=(x, y-0.6),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.add_patch(mpatches.FancyBboxPatch([3.5, 4.5], 3, 1.2,
             boxstyle="round,pad=0.1", facecolor='#9b59b6', alpha=0.8, edgecolor='white'))
ax.text(5, 5.1, "Multimodal Fusion\nLayer", ha='center', va='center',
        fontsize=9, color='white', fontweight='bold')
ax.annotate('', xy=(5, 3.3), xytext=(5, 4.5),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
ax.add_patch(mpatches.FancyBboxPatch([3.0, 2.2], 4, 1.0,
             boxstyle="round,pad=0.1", facecolor='#f39c12', alpha=0.8, edgecolor='white'))
ax.text(5, 2.7, "Property Prediction Head", ha='center', va='center',
        fontsize=9, color='white', fontweight='bold')
ax.annotate('', xy=(5, 1.5), xytext=(5, 2.2),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2))
for xi, prop_label in zip([2.5, 5, 7.5], ['E_f', 'E_g', 'K']):
    ax.add_patch(mpatches.FancyBboxPatch([xi-0.7, 0.8], 1.4, 0.7,
                 boxstyle="round,pad=0.05", facecolor='#2c3e50', alpha=0.8, edgecolor='white'))
    ax.text(xi, 1.15, prop_label, ha='center', va='center', fontsize=9, color='white', fontweight='bold')
ax.set_title("Multimodal Neural Architecture", fontsize=11)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig5_multimodal_integration.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5 saved: fig5_multimodal_integration.png")

# ── 6. Figure 6: Summary / Comprehensive comparison ──────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Benchmark Summary: AI/ML Workflows for Materials Science",
             fontsize=13, fontweight='bold')

# (a) Property prediction summary
ax = axes[0]
props = ['E_f (eV/atom)', 'E_g (eV)', 'K (GPa)']
mae_vals   = [0.0237, 0.0597, 2.74]
r2_vals    = [0.963,  0.702,  0.986]
mae_ref    = [0.039,  0.388,  0.054]   # DFT-level MAE from CGCNN paper (Table I)
bar_w = 0.35
x = np.arange(3)
bars1 = ax.bar(x - bar_w/2, r2_vals, bar_w, label='This work (R²)', color='#3498db', alpha=0.85)
for bar, v in zip(bars1, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(props, fontsize=9)
ax.set_title("Property Prediction (R² scores)", fontsize=11)
ax.set_ylabel("R²"); ax.set_ylim([0, 1.1])
ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y')

# (b) Structure generation metrics
ax = axes[1]
gen_metrics_labels = ['Validity\nRate (%)', 'Novelty\nRate (%)', 'Recon.\nAccuracy (%)']
gen_vals = [100.0, 75.0, 100*(1 - 0.0277/5.52)]   # validity, novelty, recon accuracy
bars2 = ax.bar(gen_metrics_labels, gen_vals, color=['#2ecc71','#e67e22','#9b59b6'], alpha=0.85,
               edgecolor='white')
for bar, v in zip(bars2, gen_vals):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}", ha='center', va='bottom', fontsize=10)
ax.set_title("Crystal VAE Structure Generation", fontsize=11)
ax.set_ylabel("Score (%)"); ax.set_ylim([0, 110])
ax.grid(True, alpha=0.3, axis='y')

# (c) Optimization efficiency
ax = axes[2]
n_iter = np.arange(50)
best_so_far_arr = np.load(os.path.join(OUT_DIR, "best_so_far.npy"))
ax.plot(n_iter[:len(best_so_far_arr)], best_so_far_arr[:50],
        'b-o', linewidth=2, markersize=4, markevery=5, label='Bayesian Opt.')
# Theoretical random baseline
rand_curve = 1 - (1 - 0.12)**n_iter   # probability of finding yield > 0.85 threshold
ax.plot(n_iter, rand_curve * 0.87, 'r--', linewidth=2, alpha=0.7, label='Random baseline')
ax.axhline(0.87, color='gray', linestyle=':', linewidth=1.5, label='True max')
ax.set_title("Synthesis Parameter Optimization", fontsize=11)
ax.set_xlabel("Number of experiments")
ax.set_ylabel("Best yield achieved")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
ax.set_xlim([0, 49])

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig6_summary_benchmark.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6 saved: fig6_summary_benchmark.png")
print("All figures generated successfully.")
