"""
Visualization Script for Protein Complex Structural Alignment
=============================================================
Generates all figures for the research report.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
from Bio.PDB import PDBParser, DSSP
import json
import os
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260401_200700/data"
OUT_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260401_200700/outputs"
IMG_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Life_002_20260401_200700/report/images"

# Color scheme
COLORS = {
    '7xg4': '#2196F3',   # blue
    '6n40': '#FF5722',   # orange-red
    'protein': '#4CAF50',
    'nucleic': '#9C27B0',
    'highlight': '#FF9800',
}

# Load results
with open(os.path.join(OUT_DIR, "alignment_results.json")) as f:
    results = json.load(f)


def get_ca_coords(chain):
    """Extract Cα coordinates from a chain."""
    coords = []
    for residue in chain.get_residues():
        if residue.id[0] == ' ' and 'CA' in residue:
            coords.append(residue['CA'].get_coord())
    return np.array(coords)


def load_structures():
    parser = PDBParser(QUIET=True)
    s1 = parser.get_structure("7xg4", os.path.join(DATA_DIR, "7xg4.pdb"))
    s2 = parser.get_structure("6n40", os.path.join(DATA_DIR, "6n40.pdb"))
    return s1, s2


# ─── Figure 1: Complex Composition Overview ──────────────────────────────────

def fig_complex_overview(s1, s2):
    """Figure 1: Side-by-side overview of complex compositions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Protein Complex Structural Overview", fontsize=16, fontweight='bold', y=1.02)

    for ax, (struct, info_key, title, subtitle) in zip(axes, [
        (s1, '7xg4', '7xg4', 'Type IV-A CRISPR–Cas Complex\nPseudomonas aeruginosa'),
        (s2, '6n40', '6n40', 'MmpL3 Membrane Transporter\nMycobacterium smegmatis'),
    ]):
        chain_data = results['complex_info'][info_key]['chains']
        chain_ids = list(chain_data.keys())
        n_residues = [chain_data[c]['n_aa_residues'] for c in chain_ids]
        colors = [COLORS['protein'] if chain_data[c]['type'] == 'protein' else COLORS['nucleic']
                  for c in chain_ids]

        bars = ax.bar(range(len(chain_ids)), n_residues, color=colors, edgecolor='white',
                      linewidth=0.5, alpha=0.85, zorder=3)
        ax.set_xticks(range(len(chain_ids)))
        ax.set_xticklabels([f"Chain {c}" for c in chain_ids], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Number of Residues", fontsize=12)
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight='bold')
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, nres in zip(bars, n_residues):
            if nres > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                        str(nres), ha='center', va='bottom', fontsize=8)

        # Add legend for structure type
        prot_patch = mpatches.Patch(color=COLORS['protein'], alpha=0.85, label='Protein chain')
        nuc_patch = mpatches.Patch(color=COLORS['nucleic'], alpha=0.85, label='Nucleic acid chain')
        ax.legend(handles=[prot_patch, nuc_patch], loc='upper right', fontsize=9)

        total = results['complex_info'][info_key]['total_residues']
        n_prot = results['complex_info'][info_key]['n_protein_chains']
        n_nuc = results['complex_info'][info_key]['n_nucleic_chains']
        ax.text(0.02, 0.97, f"Total residues: {total}\nProtein chains: {n_prot}\nNucleic chains: {n_nuc}",
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig1_complex_overview.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_complex_overview.png")


# ─── Figure 2: TM-score Heatmap ──────────────────────────────────────────────

def fig_tm_heatmap():
    """Figure 2: Pairwise TM-score heatmap between 7xg4 chains and 6n40 chains."""
    tm_data = results['pairwise_tm_matrix']
    rmsd_data = results['pairwise_rmsd_matrix']

    chains1 = list(tm_data.keys())
    chains2 = list(tm_data[chains1[0]].keys())

    tm_matrix = np.array([[tm_data[c1][c2] for c2 in chains2] for c1 in chains1])
    rmsd_matrix = np.array([[min(rmsd_data[c1][c2], 50) for c2 in chains2] for c1 in chains1])

    # Chain lengths for annotation
    chain_info1 = results['complex_info']['7xg4']['chains']
    chain_info2 = results['complex_info']['6n40']['chains']

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    # TM-score heatmap
    ax = axes[0]
    cmap_tm = LinearSegmentedColormap.from_list('tm', ['#f7fbff', '#2171b5', '#08306b'])
    im = ax.imshow(tm_matrix, cmap=cmap_tm, vmin=0, vmax=0.3, aspect='auto')
    plt.colorbar(im, ax=ax, label='TM-score', shrink=0.8)
    ax.set_xticks(range(len(chains2)))
    ax.set_xticklabels([f"6n40:{c}\n({chain_info2[c]['n_aa_residues']}aa)" for c in chains2], fontsize=10)
    ax.set_yticks(range(len(chains1)))
    ax.set_yticklabels([f"7xg4:{c} ({chain_info1[c]['n_aa_residues']}aa)" for c in chains1], fontsize=10)
    ax.set_title("Pairwise TM-scores\n(7xg4 chains vs 6n40 chains)", fontsize=12, fontweight='bold')
    # Annotate cells
    for i in range(len(chains1)):
        for j in range(len(chains2)):
            ax.text(j, i, f"{tm_matrix[i, j]:.3f}", ha='center', va='center',
                    fontsize=9, color='white' if tm_matrix[i, j] > 0.15 else 'black',
                    fontweight='bold')

    # Highlight best match
    best_i = np.argmax(tm_matrix[:, 0])
    rect = plt.Rectangle((-.5, best_i - .5), len(chains2), 1,
                          fill=False, edgecolor='#FF9800', linewidth=3)
    ax.add_patch(rect)
    ax.text(len(chains2) - 0.4, best_i, ' Best', va='center', color='#FF9800',
            fontweight='bold', fontsize=9)

    # RMSD heatmap
    ax = axes[1]
    cmap_rmsd = LinearSegmentedColormap.from_list('rmsd', ['#fff5f0', '#fc4e2a', '#800026'])
    im2 = ax.imshow(rmsd_matrix, cmap=cmap_rmsd, aspect='auto')
    plt.colorbar(im2, ax=ax, label='RMSD (Å, capped at 50)', shrink=0.8)
    ax.set_xticks(range(len(chains2)))
    ax.set_xticklabels([f"6n40:{c}\n({chain_info2[c]['n_aa_residues']}aa)" for c in chains2], fontsize=10)
    ax.set_yticks(range(len(chains1)))
    ax.set_yticklabels([f"7xg4:{c} ({chain_info1[c]['n_aa_residues']}aa)" for c in chains1], fontsize=10)
    ax.set_title("Pairwise RMSD (Å)\n(7xg4 chains vs 6n40 chains)", fontsize=12, fontweight='bold')
    for i in range(len(chains1)):
        for j in range(len(chains2)):
            ax.text(j, i, f"{rmsd_matrix[i, j]:.1f}", ha='center', va='center',
                    fontsize=9, color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig2_tm_rmsd_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_tm_rmsd_heatmap.png")


# ─── Figure 3: Chain Length Distribution & Comparison ────────────────────────

def fig_chain_comparison():
    """Figure 3: Chain-level comparison and statistics."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    chain_stats = results['chain_stats']

    # Panel A: Chain length scatter for 7xg4
    ax1 = fig.add_subplot(gs[0, 0])
    chains_7 = chain_stats['7xg4']['chain_lengths']
    labels7 = list(chains_7.keys())
    lengths7 = list(chains_7.values())
    ax1.bar(range(len(labels7)), lengths7, color=COLORS['7xg4'], alpha=0.8, edgecolor='white')
    ax1.set_xticks(range(len(labels7)))
    ax1.set_xticklabels([f"Chain {l}" for l in labels7], rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel("Residues (Cα)", fontsize=11)
    ax1.set_title("7xg4 Chain Lengths", fontsize=11, fontweight='bold')
    ax1.yaxis.grid(True, alpha=0.4)
    ax1.set_axisbelow(True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.axhline(np.mean(lengths7), color='red', linestyle='--', alpha=0.7,
                label=f'Mean: {np.mean(lengths7):.0f}')
    ax1.legend(fontsize=9)

    # Panel B: Chain length for 6n40
    ax2 = fig.add_subplot(gs[0, 1])
    chains_6 = chain_stats['6n40']['chain_lengths']
    labels6 = list(chains_6.keys())
    lengths6 = list(chains_6.values())
    ax2.bar(range(len(labels6)), lengths6, color=COLORS['6n40'], alpha=0.8, edgecolor='white')
    ax2.set_xticks(range(len(labels6)))
    ax2.set_xticklabels([f"Chain {l}" for l in labels6], rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel("Residues (Cα)", fontsize=11)
    ax2.set_title("6n40 Chain Lengths", fontsize=11, fontweight='bold')
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Length ratio (coverage analysis)
    ax3 = fig.add_subplot(gs[0, 2])
    len_6n40 = lengths6[0]  # single chain
    ratios = [min(l / len_6n40, 1.0) for l in lengths7]
    colors_bar = [COLORS['7xg4'] if r < 0.5 else COLORS['highlight'] for r in ratios]
    ax3.barh(range(len(labels7)), ratios, color=colors_bar, alpha=0.8, edgecolor='white')
    ax3.set_yticks(range(len(labels7)))
    ax3.set_yticklabels([f"7xg4:{l}" for l in labels7], fontsize=10)
    ax3.set_xlabel("Coverage fraction relative to 6n40:A", fontsize=10)
    ax3.set_title("Chain Length Coverage\n(7xg4 chains / 6n40:A length)", fontsize=11, fontweight='bold')
    ax3.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='Full coverage')
    ax3.axvline(0.5, color='gray', linestyle=':', alpha=0.5, label='50% coverage')
    ax3.legend(fontsize=8)
    ax3.set_xlim(0, 1.1)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: TM-score vs chain length
    ax4 = fig.add_subplot(gs[1, 0:2])
    tm_vals = [results['pairwise_tm_matrix'][c1]['A'] for c1 in results['pairwise_tm_matrix']]
    chain_ids_prot = list(results['pairwise_tm_matrix'].keys())
    len_7xg4_chains = [chains_7.get(c, 0) for c in chain_ids_prot]

    sc = ax4.scatter(len_7xg4_chains, tm_vals, c=tm_vals, cmap='YlOrRd',
                     s=100, edgecolors='black', linewidths=0.8, zorder=3, vmin=0, vmax=0.3)
    plt.colorbar(sc, ax=ax4, label='TM-score')
    for i, (x, y, lbl) in enumerate(zip(len_7xg4_chains, tm_vals, chain_ids_prot)):
        ax4.annotate(f"7xg4:{lbl}", (x, y), textcoords='offset points', xytext=(5, 3),
                     fontsize=9)
    ax4.set_xlabel("7xg4 Chain Length (residues)", fontsize=11)
    ax4.set_ylabel("TM-score vs 6n40:A", fontsize=11)
    ax4.set_title("TM-score vs Chain Length\n(effect of protein size on TM-score)", fontsize=11, fontweight='bold')
    ax4.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='TM-score = 0.5 (similar fold)')
    ax4.axhline(0.17, color='orange', linestyle='--', alpha=0.7, label='TM-score = 0.17 (random)')
    ax4.legend(fontsize=9)
    ax4.yaxis.grid(True, alpha=0.4)
    ax4.set_axisbelow(True)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Summary statistics
    ax5 = fig.add_subplot(gs[1, 2])
    stats_labels = ['7xg4\nTotal res.', '7xg4\nMean chain', '6n40\nTotal res.', '6n40\nChain A']
    stats_vals = [
        chain_stats['7xg4']['total_residues'],
        chain_stats['7xg4']['mean_length'],
        chain_stats['6n40']['total_residues'],
        chain_stats['6n40']['mean_length'],
    ]
    bar_colors = [COLORS['7xg4'], COLORS['7xg4'], COLORS['6n40'], COLORS['6n40']]
    bars = ax5.bar(range(4), stats_vals, color=bar_colors, alpha=0.8, edgecolor='white')
    ax5.set_xticks(range(4))
    ax5.set_xticklabels(stats_labels, fontsize=9)
    ax5.set_ylabel("Residue Count", fontsize=11)
    ax5.set_title("Complex Statistics", fontsize=11, fontweight='bold')
    for bar, val in zip(bars, stats_vals):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{val:.0f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.yaxis.grid(True, alpha=0.4)
    ax5.set_axisbelow(True)

    fig.suptitle("Chain-Level Structural Comparison: 7xg4 vs 6n40", fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(IMG_DIR, "fig3_chain_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_chain_comparison.png")


# ─── Figure 4: 3D Coordinate Analysis & Superimposition ──────────────────────

def fig_superimposition_analysis(s1, s2):
    """Figure 4: 3D structural superimposition projection."""

    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("7xg4", os.path.join(DATA_DIR, "7xg4.pdb"))
    struct2 = parser.get_structure("6n40", os.path.join(DATA_DIR, "6n40.pdb"))

    # Get all Ca coordinates for both complexes
    all_coords1 = {}
    for chain in struct1[0].get_chains():
        coords = get_ca_coords(chain)
        if len(coords) >= 10:
            all_coords1[chain.id] = coords

    all_coords2 = {}
    for chain in struct2[0].get_chains():
        coords = get_ca_coords(chain)
        if len(coords) >= 10:
            all_coords2[chain.id] = coords

    # Superimpose best chain pair (7xg4:L vs 6n40:A)
    sup_data = results['superimposition_data'][0]
    R = np.array(sup_data['rotation_matrix'])
    t = np.array(sup_data['translation_vector'])
    chain_7xg4_best = sup_data['chain_7xg4']
    chain_6n40_best = sup_data['chain_6n40']

    # Apply rotation to all 7xg4 protein chains
    def apply_rot(coords, R, t):
        return (R @ coords.T).T + t

    # Choose two projections
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("3D Structural Superimposition: 7xg4 vs 6n40\n"
                 f"(Best chain match: 7xg4:{chain_7xg4_best} ↔ 6n40:{chain_6n40_best}, "
                 f"TM-score={sup_data['tm_score']:.4f})",
                 fontsize=13, fontweight='bold')

    projections = [(0, 1, 'XY plane'), (0, 2, 'XZ plane'), (1, 2, 'YZ plane')]
    chain_colors_7xg4 = plt.cm.tab10(np.linspace(0, 1, len(all_coords1)))

    for row, (rotate, label) in enumerate([(False, 'Before Superimposition'),
                                             (True, 'After Superimposition')]):
        for col, (ax, (xi, yi, plane)) in enumerate(zip(axes[row], projections)):
            # Plot 7xg4 chains
            for i, (cid, coords) in enumerate(all_coords1.items()):
                if rotate and cid == chain_7xg4_best:
                    coords_plot = apply_rot(coords, R, t)
                elif rotate:
                    coords_plot = apply_rot(coords, R, t)
                else:
                    coords_plot = coords
                alpha = 0.9 if cid == chain_7xg4_best else 0.5
                lw = 1.5 if cid == chain_7xg4_best else 0.8
                ax.plot(coords_plot[:, xi], coords_plot[:, yi],
                        color=chain_colors_7xg4[i], alpha=alpha, linewidth=lw,
                        label=f"7xg4:{cid}" if col == 0 else None)
                ax.scatter(coords_plot[0, xi], coords_plot[0, yi],
                           color=chain_colors_7xg4[i], s=20, zorder=5)

            # Plot 6n40 chain
            for cid, coords in all_coords2.items():
                ax.plot(coords[:, xi], coords[:, yi],
                        color=COLORS['6n40'], linewidth=2.0, alpha=0.9,
                        label="6n40:A" if col == 0 else None)
                ax.scatter(coords[0, xi], coords[0, yi],
                           color=COLORS['6n40'], s=40, zorder=5, marker='*')

            ax.set_xlabel(['X (Å)', 'X (Å)', 'Y (Å)'][col], fontsize=9)
            ax.set_ylabel(['Y (Å)', 'Z (Å)', 'Z (Å)'][col], fontsize=9)
            ax.set_title(f"{plane}", fontsize=10)
            ax.set_facecolor('#f8f9fa')
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{label}\n" + ax.get_ylabel(), fontsize=9)
            if row == 0 and col == 0:
                ax.legend(loc='upper right', fontsize=7, ncol=2, framealpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig4_superimposition.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_superimposition.png")


# ─── Figure 5: TM-score Interpretation & Benchmarks ─────────────────────────

# ─── Figure 5 (fixed) ────────────────────────────────────────────────────────

def fig_tm_score_interpretation_v2():
    """Figure 5: TM-score distribution and interpretation thresholds (fixed)."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("TM-score Analysis and Interpretation", fontsize=14, fontweight='bold')

    tm_all = [results['pairwise_tm_matrix'][c1]['A']
              for c1 in results['pairwise_tm_matrix']]
    chain_ids = list(results['pairwise_tm_matrix'].keys())
    chain_lengths = [results['chain_stats']['7xg4']['chain_lengths'].get(c, 0) for c in chain_ids]

    # Panel A
    ax1 = axes[0]
    colors_bar = [plt.cm.RdYlGn(t / 0.5) for t in tm_all]
    ax1.barh(range(len(chain_ids)), tm_all, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(range(len(chain_ids)))
    ax1.set_yticklabels([f"7xg4:{c} ({n}aa)" for c, n in zip(chain_ids, chain_lengths)], fontsize=10)
    ax1.axvline(0.5, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Similar fold (≥0.5)')
    ax1.axvline(0.3, color='orange', linestyle='--', linewidth=1.5, alpha=0.8, label='Possible (≥0.3)')
    ax1.axvline(0.17, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Random (~0.17)')
    ax1.set_xlabel("TM-score vs 6n40:A", fontsize=11)
    ax1.set_title("Per-chain TM-scores\n(7xg4 chains vs 6n40:A)", fontsize=11, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xlim(0, 0.35)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B
    ax2 = axes[1]
    x_vals = np.array(chain_lengths)
    y_vals = np.array(tm_all)
    sc = ax2.scatter(x_vals, y_vals, c=y_vals, cmap='RdYlGn', s=120,
                     edgecolors='black', linewidths=0.8, vmin=0, vmax=0.5, zorder=3)
    plt.colorbar(sc, ax=ax2, label='TM-score')
    for x, y, lbl in zip(x_vals, y_vals, chain_ids):
        ax2.annotate(f":{lbl}", (x, y), textcoords='offset points', xytext=(4, 3), fontsize=9)
    ax2.axhline(0.17, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Random')
    ax2.axhline(0.5, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Similar fold')
    ax2.set_xlabel("Chain Length (residues)", fontsize=11)
    ax2.set_ylabel("TM-score vs 6n40:A", fontsize=11)
    ax2.set_title("TM-score vs Chain Length", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 0.35)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    # Panel C: TM-score significance band chart
    ax3 = axes[2]
    bands = [
        ('Globally similar\n(same fold)', 0.5, 1.0, '#4CAF50'),
        ('Possible similarity', 0.3, 0.5, '#8BC34A'),
        ('Ambiguous region', 0.2, 0.3, '#FFC107'),
        ('Random pairs', 0.0, 0.2, '#F44336'),
    ]
    for i, (cat, low, high, color) in enumerate(bands):
        ax3.barh(i, high - low, left=low, height=0.7, color=color, alpha=0.7)
        ax3.text((low + high) / 2, i, cat, ha='center', va='center', fontsize=9, fontweight='bold')

    our_tm = results['complex_tm_score']
    best_chain_tm = max(tm_all)
    ax3.axvline(our_tm, color='navy', linewidth=3, zorder=5, label=f'Complex TM={our_tm:.3f}')
    ax3.axvline(best_chain_tm, color='purple', linewidth=2, linestyle='--', zorder=5,
                label=f'Best chain TM={best_chain_tm:.3f}')
    ax3.set_xlabel("TM-score", fontsize=11)
    ax3.set_title("TM-score Significance\n(thresholds from literature)", fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_xlim(0, 1.0)
    ax3.set_yticks([])
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig5_tm_interpretation.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_tm_interpretation.png")


# ─── Figure 6: Superimposition rotation matrix visualization ─────────────────

def fig_rotation_analysis():
    """Figure 6: Rotation matrix and translation vector visualization."""
    sup = results['superimposition_data'][0]
    R = np.array(sup['rotation_matrix'])
    t = np.array(sup['translation_vector'])
    c1 = np.array(sup['centroid_7xg4'])
    c2 = np.array(sup['centroid_6n40'])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Superimposition Analysis: 7xg4:{sup['chain_7xg4']} → 6n40:{sup['chain_6n40']}\n"
                 f"(TM-score: {sup['tm_score']:.4f}, RMSD: {sup['rmsd']:.3f} Å)",
                 fontsize=13, fontweight='bold')

    # Panel A: Rotation matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(R, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax1, label='Matrix element value', shrink=0.8)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['X', 'Y', 'Z'], fontsize=12, fontweight='bold')
    ax1.set_yticklabels(['X', 'Y', 'Z'], fontsize=12, fontweight='bold')
    ax1.set_title("Rotation Matrix\n(Kabsch algorithm)", fontsize=11, fontweight='bold')
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f"{R[i, j]:.3f}", ha='center', va='center',
                     fontsize=10, color='black')

    # Panel B: Translation vector and centroids
    ax2 = axes[1]
    centers = np.array([c1, c2, c1 + t])
    labels = ['7xg4:L\ncentroid', '6n40:A\ncentroid', '7xg4:L\n(translated)']
    marker_colors = [COLORS['7xg4'], COLORS['6n40'], COLORS['highlight']]

    ax2.bar(range(3), centers[:, 0], label='X', alpha=0.8, color='#E91E63')
    ax2.bar(range(3), centers[:, 1], bottom=0, label='Y', alpha=0.6, color='#2196F3')
    ax2.bar(range(3), centers[:, 2], bottom=0, label='Z', alpha=0.6, color='#4CAF50')
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Coordinate (Å)", fontsize=11)
    ax2.set_title("Structure Centroids\n(X, Y, Z components)", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Translation vector magnitude
    ax3 = axes[2]
    t_magnitude = np.linalg.norm(t)
    dist_centroids = np.linalg.norm(c1 - c2)
    metrics = ['RMSD\n(Å)', 'Translation\nmagnitude (Å)', 'Centroid\ndistance (Å)',
               'TM-score\n(×100 for scale)']
    vals = [sup['rmsd'], t_magnitude, dist_centroids, sup['tm_score'] * 100]
    colors_m = ['#F44336', '#FF9800', '#2196F3', '#4CAF50']
    bars = ax3.bar(range(4), vals, color=colors_m, alpha=0.8, edgecolor='white')
    ax3.set_xticks(range(4))
    ax3.set_xticklabels(metrics, fontsize=9)
    ax3.set_ylabel("Value", fontsize=11)
    ax3.set_title("Alignment Quality Metrics\n(superimposition statistics)", fontsize=11, fontweight='bold')
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.yaxis.grid(True, alpha=0.4)
    ax3.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig6_superimposition_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_superimposition_analysis.png")


# ─── Figure 7: Distance distribution analysis ────────────────────────────────

def fig_distance_analysis(s1, s2):
    """Figure 7: Pairwise distance distribution analysis."""
    parser = PDBParser(QUIET=True)
    struct1 = parser.get_structure("7xg4", os.path.join(DATA_DIR, "7xg4.pdb"))
    struct2 = parser.get_structure("6n40", os.path.join(DATA_DIR, "6n40.pdb"))

    def get_ca_sequential(structure, chain_id):
        chain = structure[0][chain_id]
        coords = get_ca_coords(chain)
        return coords

    # Get consecutive Cα-Cα distances
    def consecutive_distances(coords):
        if len(coords) < 2:
            return np.array([])
        return np.linalg.norm(np.diff(coords, axis=0), axis=1)

    chain_L_coords = get_ca_sequential(struct1, 'L')
    chain_A_coords = get_ca_sequential(struct2, 'A')

    # Also get short chains for comparison
    chain_A_7xg4 = get_ca_sequential(struct1, 'A')
    chain_B_7xg4 = get_ca_sequential(struct1, 'B')

    dists_L = consecutive_distances(chain_L_coords)
    dists_A6 = consecutive_distances(chain_A_coords)
    dists_A7 = consecutive_distances(chain_A_7xg4)
    dists_B7 = consecutive_distances(chain_B_7xg4)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Cα–Cα Distance Distribution Analysis", fontsize=14, fontweight='bold')

    # Panel A: Consecutive Cα distance distribution
    ax1 = axes[0, 0]
    bins = np.linspace(2, 10, 50)
    ax1.hist(dists_L, bins=bins, alpha=0.7, color=COLORS['7xg4'], density=True,
             label=f"7xg4:L (n={len(chain_L_coords)}aa)", edgecolor='white')
    ax1.hist(dists_A6, bins=bins, alpha=0.7, color=COLORS['6n40'], density=True,
             label=f"6n40:A (n={len(chain_A_coords)}aa)", edgecolor='white')
    ax1.axvline(3.8, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                label='Ideal Cα-Cα: 3.8 Å')
    ax1.set_xlabel("Cα–Cα distance (Å)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Consecutive Cα–Cα Distances\n(chain L vs 6n40:A)", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Multiple chain comparison
    ax2 = axes[0, 1]
    for dists, label, color in [
        (dists_A7, "7xg4:A", '#1976D2'),
        (dists_B7, "7xg4:B", '#42A5F5'),
        (dists_L, "7xg4:L", '#0D47A1'),
        (dists_A6, "6n40:A", COLORS['6n40']),
    ]:
        ax2.hist(dists, bins=bins, alpha=0.5, density=True, label=f"{label} ({len(dists)+1}aa)",
                 edgecolor='none', histtype='stepfilled')
    ax2.axvline(3.8, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='3.8 Å ideal')
    ax2.set_xlabel("Cα–Cα distance (Å)", fontsize=11)
    ax2.set_ylabel("Density", fontsize=11)
    ax2.set_title("Cα–Cα Distance Distributions\n(selected chains)", fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Distance matrix (internal structure of chain L of 7xg4)
    ax3 = axes[1, 0]
    # Subsample for visualization
    coords_L_sub = chain_L_coords[::3]  # every 3rd residue
    dist_matrix = np.linalg.norm(
        coords_L_sub[:, None, :] - coords_L_sub[None, :, :], axis=2)
    im = ax3.imshow(dist_matrix, cmap='hot_r', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im, ax=ax3, label='Distance (Å)', shrink=0.8)
    ax3.set_title(f"Intramolecular Distance Matrix\n7xg4:L (every 3rd Cα, {len(coords_L_sub)} pts)",
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel("Residue index (×3)", fontsize=10)
    ax3.set_ylabel("Residue index (×3)", fontsize=10)

    # Panel D: Distance matrix for 6n40:A
    ax4 = axes[1, 1]
    coords_A6_sub = chain_A_coords[::3]
    dist_matrix6 = np.linalg.norm(
        coords_A6_sub[:, None, :] - coords_A6_sub[None, :, :], axis=2)
    im4 = ax4.imshow(dist_matrix6, cmap='hot_r', aspect='auto', vmin=0, vmax=100)
    plt.colorbar(im4, ax=ax4, label='Distance (Å)', shrink=0.8)
    ax4.set_title(f"Intramolecular Distance Matrix\n6n40:A (every 3rd Cα, {len(coords_A6_sub)} pts)",
                  fontsize=11, fontweight='bold')
    ax4.set_xlabel("Residue index (×3)", fontsize=10)
    ax4.set_ylabel("Residue index (×3)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "fig7_distance_analysis.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig7_distance_analysis.png")


# ─── Figure 8: Summary dashboard ─────────────────────────────────────────────

def fig_summary_dashboard():
    """Figure 8: Summary alignment dashboard."""
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.45)

    # Key metrics
    tm_all = [results['pairwise_tm_matrix'][c1]['A']
              for c1 in results['pairwise_tm_matrix']]
    chain_ids = list(results['pairwise_tm_matrix'].keys())
    rmsd_all = [results['pairwise_rmsd_matrix'][c1]['A']
                for c1 in results['pairwise_rmsd_matrix']]
    complex_tm = results['complex_tm_score']
    best_chain_tm = max(tm_all)
    best_chain = chain_ids[np.argmax(tm_all)]
    best_rmsd = rmsd_all[np.argmax(tm_all)]

    # Top row: Key numbers (large text panels)
    metrics = [
        ("Complex TM-score", f"{complex_tm:.4f}", "Weighted average\nacross all chains", '#E3F2FD'),
        ("Best Chain TM", f"{best_chain_tm:.4f}", f"7xg4:{best_chain} ↔ 6n40:A\n(best pair)", '#E8F5E9'),
        ("Best Chain RMSD", f"{best_rmsd:.2f} Å", f"7xg4:{best_chain} ↔ 6n40:A\nafter superimposition", '#FFF3E0'),
        ("7xg4 Chains", "9 protein\n+ 3 nucleic", "Complex composition", '#F3E5F5'),
    ]
    for i, (title, val, subtitle, bg) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(bg)
        ax.text(0.5, 0.65, val, ha='center', va='center', fontsize=20, fontweight='bold',
                transform=ax.transAxes, color='#1A1A2E')
        ax.text(0.5, 0.90, title, ha='center', va='center', fontsize=11, fontweight='bold',
                transform=ax.transAxes, color='#333')
        ax.text(0.5, 0.25, subtitle, ha='center', va='center', fontsize=9,
                transform=ax.transAxes, color='#555', style='italic')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Bottom row panel 1-2: TM-score radar
    ax_bar = fig.add_subplot(gs[1, 0:2])
    x = np.arange(len(chain_ids))
    bars = ax_bar.bar(x, tm_all,
                      color=[plt.cm.RdYlGn(t / 0.5) for t in tm_all],
                      edgecolor='white', linewidth=0.5)
    ax_bar.axhline(0.5, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='Similar fold')
    ax_bar.axhline(0.17, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Random baseline')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f":{c}" for c in chain_ids], fontsize=10)
    ax_bar.set_xlabel("7xg4 chain ID", fontsize=11)
    ax_bar.set_ylabel("TM-score vs 6n40:A", fontsize=11)
    ax_bar.set_title("Per-Chain TM-scores: 7xg4 Chains vs 6n40:A", fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9)
    ax_bar.set_ylim(0, 0.35)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Bottom panel 3: chain composition pie
    ax_pie = fig.add_subplot(gs[1, 2])
    info_7 = results['complex_info']['7xg4']
    chains_7 = info_7['chains']
    prot_res = sum(v['n_aa_residues'] for v in chains_7.values() if v['type'] == 'protein')
    nuc_chains = info_7['n_nucleic_chains']
    labels = ['Protein chains\n(9 chains)', 'Nucleic acid chains\n(3 chains)']
    sizes = [9, 3]
    colors_pie = [COLORS['protein'], COLORS['nucleic']]
    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=labels, colors=colors_pie, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75, textprops={'fontsize': 9})
    ax_pie.set_title("7xg4 Chain Composition", fontsize=11, fontweight='bold')

    # Bottom panel 4: Structural context summary
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.axis('off')
    summary = (
        "KEY FINDINGS\n\n"
        f"• 7xg4 (CRISPR-Cas): 12 chains\n"
        f"  (9 protein + 3 nucleic acid)\n\n"
        f"• 6n40 (MmpL3): 1 chain (726 aa)\n\n"
        f"• Max pairwise TM-score:\n"
        f"  {best_chain_tm:.4f} (7xg4:{best_chain} vs 6n40:A)\n\n"
        f"• Complex TM-score: {complex_tm:.4f}\n\n"
        f"• Interpretation: Structurally\n"
        f"  dissimilar complexes\n"
        f"  (TM < 0.17 expected for random)\n\n"
        f"• Method: TM-align (Kabsch)\n"
        f"  with iterative refinement"
    )
    ax_txt.text(0.05, 0.97, summary, transform=ax_txt.transAxes, va='top', fontsize=9,
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.8))

    fig.suptitle("Structural Alignment Dashboard: 7xg4 vs 6n40\n"
                 "Foldseek-Multimer / TM-align Style Analysis",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(IMG_DIR, "fig8_summary_dashboard.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig8_summary_dashboard.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures...")
    s1, s2 = load_structures()

    print("\n[Figure 1] Complex composition overview...")
    fig_complex_overview(s1, s2)

    print("[Figure 2] TM-score and RMSD heatmaps...")
    fig_tm_heatmap()

    print("[Figure 3] Chain comparison statistics...")
    fig_chain_comparison()

    print("[Figure 4] 3D superimposition projections...")
    fig_superimposition_analysis(s1, s2)

    print("[Figure 5] TM-score interpretation...")
    fig_tm_score_interpretation_v2()

    print("[Figure 6] Rotation matrix analysis...")
    fig_rotation_analysis()

    print("[Figure 7] Distance distribution analysis...")
    fig_distance_analysis(s1, s2)

    print("[Figure 8] Summary dashboard...")
    fig_summary_dashboard()

    print("\nAll figures saved to:", IMG_DIR)


if __name__ == "__main__":
    main()
