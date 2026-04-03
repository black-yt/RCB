"""
XEB (Cross-Entropy Benchmarking) Fidelity Analysis for Random Quantum Circuit Sampling
=======================================================================================
This script:
1. Loads measurement counts and ideal amplitudes for each (N, d, r) instance
2. Computes the XEB fidelity estimate
3. Aggregates over circuit instances (r) to get mean ± std
4. Generates comparison plots for:
   - Fixed N=40, varying depth d
   - Fixed d=12, varying qubit count N
"""

import os
import json
import glob
import ast
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKSPACE = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260402_002341")
DATA_RESULTS = WORKSPACE / "data" / "results"
DATA_AMPLITUDES = WORKSPACE / "data" / "amplitudes"
OUTPUTS = WORKSPACE / "outputs"
REPORT_IMAGES = WORKSPACE / "report" / "images"

OUTPUTS.mkdir(exist_ok=True)
REPORT_IMAGES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def parse_bitstring_key(key: str) -> tuple:
    """Parse a bitstring key which may be a tuple-string or plain bitstring."""
    key = key.strip()
    if key.startswith("("):
        return ast.literal_eval(key)
    else:
        # plain binary string like "0010110..."
        return tuple(int(b) for b in key)


def parse_amplitude(val):
    """Parse amplitude value: may be a string like '(real+imagj)' or a float."""
    if isinstance(val, (int, float)):
        return complex(val)
    s = str(val).strip()
    return complex(s.replace(" ", ""))


def load_counts(filepath: str) -> dict:
    """Load bitstring counts from JSON. Returns {bitstring_tuple: count}."""
    with open(filepath) as f:
        raw = json.load(f)
    return {parse_bitstring_key(k): int(v) for k, v in raw.items()}


def load_amplitudes(filepath: str) -> dict:
    """Load ideal amplitudes from JSON. Returns {bitstring_tuple: probability}."""
    with open(filepath) as f:
        raw = json.load(f)
    result = {}
    for k, v in raw.items():
        key = parse_bitstring_key(k)
        amp = parse_amplitude(v)
        result[key] = abs(amp) ** 2  # ideal probability = |amplitude|^2
    return result


def compute_xeb_fidelity(counts: dict, ideal_probs: dict, N: int) -> float:
    """
    Compute XEB fidelity estimate.

    F_XEB = 2^N * <p_ideal>_measured - 1

    where <p_ideal> is the counts-weighted average of ideal probabilities
    over measured bitstrings.

    Args:
        counts: {bitstring: count}
        ideal_probs: {bitstring: ideal_probability}
        N: number of qubits

    Returns:
        F_XEB (float)
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return np.nan

    # Weighted average of ideal probabilities for measured bitstrings
    weighted_sum = 0.0
    matched = 0
    for bs, cnt in counts.items():
        if bs in ideal_probs:
            weighted_sum += cnt * ideal_probs[bs]
            matched += cnt

    if matched == 0:
        return np.nan

    # Use only the matched shots
    mean_p = weighted_sum / matched
    fidelity = (2 ** N) * mean_p - 1.0
    return fidelity


# ---------------------------------------------------------------------------
# Discover available (N, d) combinations
# ---------------------------------------------------------------------------

def discover_xeb_instances(results_subdir: str, amplitudes_subdir: str):
    """
    Discover all (N, d, r) instances where both counts and amplitudes exist.

    Returns list of dicts with keys: N, d, r, counts_path, amplitudes_path
    """
    results_base = DATA_RESULTS / results_subdir
    amp_base = DATA_AMPLITUDES / amplitudes_subdir

    instances = []
    pattern = str(results_base / "**" / "*_XEB_counts.json")
    for counts_path in sorted(glob.glob(pattern, recursive=True)):
        fname = os.path.basename(counts_path)
        # e.g. N40_d12_r1_XEB_counts.json
        m = re.match(r"N(\d+)_d(\d+)_r(\d+)_XEB_counts\.json", fname)
        if not m:
            continue
        N, d, r = int(m.group(1)), int(m.group(2)), int(m.group(3))

        # Corresponding amplitudes file
        amp_dir = amp_base / f"N{N}_d{d}_XEB"
        amp_path = amp_dir / f"N{N}_d{d}_r{r}_XEB_amplitudes.json"

        if amp_path.exists():
            instances.append({
                "N": N, "d": d, "r": r,
                "counts_path": counts_path,
                "amplitudes_path": str(amp_path),
            })
    return instances


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_instances(instances):
    """
    Compute XEB fidelity for each instance.
    Returns list of dicts with N, d, r, fidelity.
    """
    results = []
    for inst in instances:
        try:
            counts = load_counts(inst["counts_path"])
            ideal_probs = load_amplitudes(inst["amplitudes_path"])
            N = inst["N"]
            fid = compute_xeb_fidelity(counts, ideal_probs, N)
            results.append({"N": N, "d": inst["d"], "r": inst["r"], "fidelity": fid})
        except Exception as e:
            print(f"Error processing {inst['counts_path']}: {e}")
    return results


def aggregate_by_Nd(results):
    """
    Group results by (N, d) and compute mean ± std of fidelity.
    Returns dict: (N,d) -> {"mean": ..., "std": ..., "n": ..., "values": [...]}
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        if not np.isnan(r["fidelity"]):
            groups[(r["N"], r["d"])].append(r["fidelity"])

    agg = {}
    for (N, d), vals in groups.items():
        arr = np.array(vals)
        agg[(N, d)] = {
            "mean": np.mean(arr),
            "std": np.std(arr, ddof=1) if len(arr) > 1 else 0.0,
            "sem": np.std(arr, ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0,
            "n": len(arr),
            "values": arr.tolist(),
        }
    return agg


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

print("=" * 60)
print("XEB Fidelity Analysis")
print("=" * 60)

# --- 1. N=40, varying depth ---
print("\n[1] Discovering N=40 instances (depth scan)...")
inst_N40 = discover_xeb_instances("N40_verification", "N40_verification")
print(f"    Found {len(inst_N40)} instances")
res_N40 = analyze_instances(inst_N40)
agg_N40 = aggregate_by_Nd(res_N40)

# --- 2. N scan at d=12 ---
print("\n[2] Discovering N-scan instances (d=12 fixed)...")
inst_Nscan = discover_xeb_instances("N_scan_depth12", "N_scan_depth12")
print(f"    Found {len(inst_Nscan)} instances")
res_Nscan = analyze_instances(inst_Nscan)
agg_Nscan = aggregate_by_Nd(res_Nscan)

# Save raw results
import json as _json

def make_serializable(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_agg(agg, path):
    data = {str(k): {kk: make_serializable(vv) for kk, vv in v.items()} for k, v in agg.items()}
    with open(path, "w") as f:
        _json.dump(data, f, indent=2)

save_agg(agg_N40, OUTPUTS / "xeb_N40_depth_scan.json")
save_agg(agg_Nscan, OUTPUTS / "xeb_Nscan_d12.json")

# Print summary tables
print("\n--- N=40 depth scan summary ---")
print(f"{'N':>5} {'d':>5} {'instances':>10} {'mean F_XEB':>12} {'std':>10}")
for (N, d), v in sorted(agg_N40.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"{N:>5} {d:>5} {v['n']:>10} {v['mean']:>12.4f} {v['std']:>10.4f}")

print("\n--- N scan at d=12 summary ---")
print(f"{'N':>5} {'d':>5} {'instances':>10} {'mean F_XEB':>12} {'std':>10}")
for (N, d), v in sorted(agg_Nscan.items(), key=lambda x: (x[0][0], x[0][1])):
    print(f"{N:>5} {d:>5} {v['n']:>10} {v['mean']:>12.4f} {v['std']:>10.4f}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# ---- Figure 1: N=40, XEB fidelity vs depth ----
fig1, ax1 = plt.subplots(figsize=(7, 5))

N40_data = {(N, d): v for (N, d), v in agg_N40.items() if N == 40}
if N40_data:
    depths_40 = sorted(set(d for (_, d) in N40_data.keys()))
    means_40 = [N40_data[(40, d)]["mean"] for d in depths_40 if (40, d) in N40_data]
    sems_40 = [N40_data[(40, d)]["sem"] for d in depths_40 if (40, d) in N40_data]
    valid_depths_40 = [d for d in depths_40 if (40, d) in N40_data]

    ax1.errorbar(valid_depths_40, means_40, yerr=sems_40,
                 marker='o', color=COLORS[0], linewidth=2, markersize=7,
                 capsize=4, label='N=40 XEB fidelity (mean ± SEM)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='F=0 (noise floor)')
    ax1.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='F=1 (ideal)')

ax1.set_xlabel("Circuit Depth $d$", fontsize=14)
ax1.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=14)
ax1.set_title("XEB Fidelity vs Circuit Depth (N=40 qubits)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(left=0)
fig1.tight_layout()
fig1.savefig(str(REPORT_IMAGES / "fig1_xeb_vs_depth_N40.png"), dpi=150, bbox_inches='tight')
print("\nSaved fig1_xeb_vs_depth_N40.png")

# ---- Figure 2: d=12, XEB fidelity vs N ----
fig2, ax2 = plt.subplots(figsize=(7, 5))

Nscan_d12 = {(N, d): v for (N, d), v in agg_Nscan.items() if d == 12}
if Nscan_d12:
    Ns = sorted(set(N for (N, _) in Nscan_d12.keys()))
    means_N = [Nscan_d12[(N, 12)]["mean"] for N in Ns if (N, 12) in Nscan_d12]
    sems_N = [Nscan_d12[(N, 12)]["sem"] for N in Ns if (N, 12) in Nscan_d12]
    valid_Ns = [N for N in Ns if (N, 12) in Nscan_d12]

    ax2.errorbar(valid_Ns, means_N, yerr=sems_N,
                 marker='s', color=COLORS[1], linewidth=2, markersize=7,
                 capsize=4, label='d=12 XEB fidelity (mean ± SEM)')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='F=0 (noise floor)')
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='F=1 (ideal)')

ax2.set_xlabel("Number of Qubits $N$", fontsize=14)
ax2.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=14)
ax2.set_title("XEB Fidelity vs Qubit Count (Depth d=12)", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(str(REPORT_IMAGES / "fig2_xeb_vs_N_d12.png"), dpi=150, bbox_inches='tight')
print("Saved fig2_xeb_vs_N_d12.png")


# ---- Figure 3: Both scans combined ----
fig3, axes = plt.subplots(1, 2, figsize=(14, 5))

ax3a = axes[0]
if N40_data and valid_depths_40:
    ax3a.errorbar(valid_depths_40, means_40, yerr=sems_40,
                  marker='o', color=COLORS[0], linewidth=2, markersize=7,
                  capsize=4, label='Experimental F_XEB')
    ax3a.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='F=0')
    ax3a.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='F=1 (ideal)')
ax3a.set_xlabel("Circuit Depth $d$", fontsize=13)
ax3a.set_ylabel("XEB Fidelity", fontsize=13)
ax3a.set_title("(a) Fidelity vs Depth (N=40)", fontsize=13)
ax3a.legend(fontsize=10)
ax3a.grid(True, alpha=0.3)
ax3a.set_xlim(left=0)

ax3b = axes[1]
if Nscan_d12 and valid_Ns:
    ax3b.errorbar(valid_Ns, means_N, yerr=sems_N,
                  marker='s', color=COLORS[1], linewidth=2, markersize=7,
                  capsize=4, label='Experimental F_XEB')
    ax3b.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='F=0')
    ax3b.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='F=1 (ideal)')
ax3b.set_xlabel("Number of Qubits $N$", fontsize=13)
ax3b.set_ylabel("XEB Fidelity", fontsize=13)
ax3b.set_title("(b) Fidelity vs N (d=12)", fontsize=13)
ax3b.legend(fontsize=10)
ax3b.grid(True, alpha=0.3)

fig3.suptitle("XEB Fidelity Analysis of Random Quantum Circuit Sampling", fontsize=14, fontweight='bold')
fig3.tight_layout()
fig3.savefig(str(REPORT_IMAGES / "fig3_combined_xeb.png"), dpi=150, bbox_inches='tight')
print("Saved fig3_combined_xeb.png")


# ---- Figure 4: Distribution of per-instance fidelities (violin/box) ----
# Show spread of fidelity values across circuit instances for N=40

fig4, ax4 = plt.subplots(figsize=(10, 5))

N40_depths_for_violin = sorted([d for (N, d) in N40_data.keys() if N == 40])
violin_data = [N40_data[(40, d)]["values"] for d in N40_depths_for_violin if (40, d) in N40_data]
valid_dv = [d for d in N40_depths_for_violin if (40, d) in N40_data]

if violin_data:
    positions = range(len(valid_dv))
    parts = ax4.violinplot(violin_data, positions=positions, showmedians=True, showmeans=False)
    for pc in parts['bodies']:
        pc.set_facecolor(COLORS[0])
        pc.set_alpha(0.5)
    ax4.set_xticks(list(positions))
    ax4.set_xticklabels([str(d) for d in valid_dv])
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    ax4.set_xlabel("Circuit Depth $d$", fontsize=13)
    ax4.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=13)
    ax4.set_title("Distribution of XEB Fidelity over Circuit Instances (N=40)", fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y')

fig4.tight_layout()
fig4.savefig(str(REPORT_IMAGES / "fig4_fidelity_distribution_N40.png"), dpi=150, bbox_inches='tight')
print("Saved fig4_fidelity_distribution_N40.png")


# ---- Figure 5: Log scale plot of fidelity decay ----
fig5, ax5 = plt.subplots(figsize=(7, 5))

if N40_data and valid_depths_40 and means_40:
    means_arr = np.array(means_40)
    depths_arr = np.array(valid_depths_40)
    # Only plot positive fidelity values on log scale
    pos_mask = means_arr > 0
    if pos_mask.any():
        ax5.semilogy(depths_arr[pos_mask], means_arr[pos_mask], 'o-',
                     color=COLORS[0], linewidth=2, markersize=7, label='Exp. F_XEB')
        # Fit exponential decay: F = exp(-d * lambda)
        if pos_mask.sum() >= 3:
            log_F = np.log(means_arr[pos_mask])
            d_fit = depths_arr[pos_mask]
            # Linear fit: log(F) = log(A) - lambda * d
            coeffs = np.polyfit(d_fit, log_F, 1)
            lambda_eff = -coeffs[0]
            d_range = np.linspace(d_fit.min(), d_fit.max(), 100)
            ax5.semilogy(d_range, np.exp(np.polyval(coeffs, d_range)), '--',
                         color='red', alpha=0.7,
                         label=f'Exp fit: $e^{{-{lambda_eff:.3f} d}}$')
            print(f"\nExponential decay fit (N=40): lambda = {lambda_eff:.4f} per depth layer")

ax5.set_xlabel("Circuit Depth $d$", fontsize=14)
ax5.set_ylabel("XEB Fidelity (log scale)", fontsize=14)
ax5.set_title("Exponential Decay of XEB Fidelity (N=40)", fontsize=14)
ax5.legend(fontsize=11)
ax5.grid(True, alpha=0.3)
fig5.tight_layout()
fig5.savefig(str(REPORT_IMAGES / "fig5_log_fidelity_decay.png"), dpi=150, bbox_inches='tight')
print("Saved fig5_log_fidelity_decay.png")


# ---- Figure 6: Individual instance fidelity scatter for N-scan ----
fig6, ax6 = plt.subplots(figsize=(8, 5))

all_N_d12 = [(r["N"], r["fidelity"]) for r in res_Nscan if r["d"] == 12 and not np.isnan(r["fidelity"])]
if all_N_d12:
    Nvals, fvals = zip(*all_N_d12)
    ax6.scatter(Nvals, fvals, alpha=0.4, s=30, color=COLORS[2], label='Individual instances')

    # Overlay mean
    if valid_Ns and means_N:
        ax6.plot(valid_Ns, means_N, 'k-o', linewidth=2, markersize=8, label='Mean F_XEB')
        # Error bars
        ax6.errorbar(valid_Ns, means_N, yerr=sems_N, fmt='none', color='black', capsize=5)

    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='F=0')
    ax6.set_xlabel("Number of Qubits $N$", fontsize=14)
    ax6.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=14)
    ax6.set_title("XEB Fidelity vs Qubit Count: Individual Instances (d=12)", fontsize=13)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)

fig6.tight_layout()
fig6.savefig(str(REPORT_IMAGES / "fig6_scatter_N_scan.png"), dpi=150, bbox_inches='tight')
print("Saved fig6_scatter_N_scan.png")


# ---- Figure 7: Data overview - number of instances per configuration ----
fig7, axes7 = plt.subplots(1, 2, figsize=(13, 5))

ax7a = axes7[0]
if N40_data:
    n_inst_40 = [N40_data[(40, d)]["n"] for d in valid_depths_40 if (40, d) in N40_data]
    ax7a.bar(valid_depths_40, n_inst_40, color=COLORS[0], alpha=0.7, edgecolor='black')
    ax7a.set_xlabel("Circuit Depth $d$", fontsize=12)
    ax7a.set_ylabel("Number of Circuit Instances", fontsize=12)
    ax7a.set_title("Data Coverage: N=40 Depth Scan", fontsize=12)
    ax7a.grid(True, alpha=0.3, axis='y')

ax7b = axes7[1]
if Nscan_d12:
    n_inst_N = [Nscan_d12[(N, 12)]["n"] for N in valid_Ns if (N, 12) in Nscan_d12]
    ax7b.bar(valid_Ns, n_inst_N, color=COLORS[1], alpha=0.7, edgecolor='black')
    ax7b.set_xlabel("Number of Qubits $N$", fontsize=12)
    ax7b.set_ylabel("Number of Circuit Instances", fontsize=12)
    ax7b.set_title("Data Coverage: N Scan at d=12", fontsize=12)
    ax7b.grid(True, alpha=0.3, axis='y')

fig7.suptitle("Dataset Overview: Number of Circuit Instances", fontsize=13, fontweight='bold')
fig7.tight_layout()
fig7.savefig(str(REPORT_IMAGES / "fig7_data_overview.png"), dpi=150, bbox_inches='tight')
print("Saved fig7_data_overview.png")

print("\n[Done] All figures saved to", str(REPORT_IMAGES))
print("[Done] Aggregated results saved to", str(OUTPUTS))
