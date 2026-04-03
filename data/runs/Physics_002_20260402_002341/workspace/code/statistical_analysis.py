"""
Additional statistical analysis:
- Error propagation and gate-count model
- Comparison with Haar-random baseline expectation
- Bootstrap confidence intervals
- XEB per-sample analysis
"""

import os, json, glob, ast, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

WORKSPACE = Path("/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Physics_002_20260402_002341")
OUTPUTS = WORKSPACE / "outputs"
REPORT_IMAGES = WORKSPACE / "report" / "images"

# Load aggregated results
with open(OUTPUTS / "xeb_N40_depth_scan.json") as f:
    raw_N40 = json.load(f)
with open(OUTPUTS / "xeb_Nscan_d12.json") as f:
    raw_Nscan = json.load(f)

def parse_key(k):
    # "(40, 12)" -> (40, 12)
    return tuple(int(x.strip()) for x in k.strip("()").split(","))

agg_N40 = {parse_key(k): v for k, v in raw_N40.items()}
agg_Nscan = {parse_key(k): v for k, v in raw_Nscan.items()}

COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

# -------------------------------------------------------------------------
# 1. Error propagation model: F ~ exp(-lambda * d) for fixed N
#    lambda = error_rate_per_cycle (can be linked to gate error rates)
# -------------------------------------------------------------------------

N40_data = {(N, d): v for (N, d), v in agg_N40.items() if N == 40}
depths = sorted(d for (N, d) in N40_data if N == 40)
means_arr = np.array([N40_data[(40, d)]["mean"] for d in depths])
sems_arr = np.array([N40_data[(40, d)]["sem"] for d in depths])
n_arr = np.array([N40_data[(40, d)]["n"] for d in depths])

# Fit exponential: F = A * exp(-lambda * d)
# Using log-linear regression on positive values
pos_mask = means_arr > 0.01
d_fit = np.array(depths)[pos_mask]
F_fit = means_arr[pos_mask]
log_F = np.log(F_fit)
slope, intercept, r_value, p_value, std_err = stats.linregress(d_fit, log_F)
lambda_eff = -slope
A = np.exp(intercept)

print("=== Exponential Decay Fit (N=40) ===")
print(f"F(d) = {A:.4f} * exp(-{lambda_eff:.4f} * d)")
print(f"R^2 = {r_value**2:.4f}, p-value = {p_value:.4e}")
print(f"Effective error rate per depth: lambda = {lambda_eff:.4f}")

# -------------------------------------------------------------------------
# 2. Bootstrap 95% CI for mean XEB fidelity
# -------------------------------------------------------------------------

def bootstrap_ci(values, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for mean."""
    values = np.array(values)
    boot_means = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                           for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, 100 * alpha)
    hi = np.percentile(boot_means, 100 * (1 - alpha))
    return lo, hi

np.random.seed(42)

print("\n=== Bootstrap 95% CI (N=40 depth scan) ===")
boot_ci_N40 = {}
for d in depths:
    vals = N40_data[(40, d)]["values"]
    lo, hi = bootstrap_ci(vals)
    boot_ci_N40[d] = (lo, hi)
    print(f"  d={d}: mean={N40_data[(40,d)]['mean']:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={len(vals)}")

print("\n=== Bootstrap 95% CI (N scan, d=12) ===")
Ns = sorted(N for (N, d) in agg_Nscan if d == 12)
boot_ci_Nscan = {}
for N in Ns:
    if (N, 12) not in agg_Nscan:
        continue
    vals = agg_Nscan[(N, 12)]["values"]
    lo, hi = bootstrap_ci(vals)
    boot_ci_Nscan[N] = (lo, hi)
    print(f"  N={N}: mean={agg_Nscan[(N,12)]['mean']:.4f}, 95% CI=[{lo:.4f}, {hi:.4f}], n={len(vals)}")

# -------------------------------------------------------------------------
# 3. Theoretical uniform-distribution baseline
#    Under ideal Haar-random circuit, <2^N * p_ideal> = 2 (due to Porter-Thomas)
#    so F_XEB_ideal = 1.0 (by design of the estimator)
#    Under total noise (uniform random output), F_XEB -> 0
# -------------------------------------------------------------------------

# Expected XEB per-bitstring variance under Porter-Thomas:
# Var[2^N * p] = 2 (for large N) so the std of per-sample F is ~sqrt(2/M) where M is shots
print("\n=== Porter-Thomas variance analysis ===")
for d in depths:
    n_inst = N40_data[(40, d)]["n"]
    # With ~20 bitstrings per instance and 50 instances, M_eff = 1000 per config
    # Theoretical variance ~ 1/sqrt(n_samples * n_instances)
    # Each instance has ~20 bitstrings
    n_samples_total = n_inst * 20  # approximate
    theory_sem = np.sqrt(2) / np.sqrt(n_samples_total)
    print(f"  d={d}: obs SEM={N40_data[(40,d)]['sem']:.4f}, theory SEM≈{theory_sem:.4f}")

# -------------------------------------------------------------------------
# 4. Enhanced figure: Depth scan with bootstrap CI and fit
# -------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ci_lo = np.array([boot_ci_N40[d][0] for d in depths])
ci_hi = np.array([boot_ci_N40[d][1] for d in depths])
depths_arr = np.array(depths)

ax.fill_between(depths_arr, ci_lo, ci_hi, alpha=0.25, color=COLORS[0], label='95% Bootstrap CI')
ax.errorbar(depths_arr, means_arr, yerr=sems_arr,
            marker='o', color=COLORS[0], linewidth=2, markersize=7,
            capsize=4, label='Mean ± SEM')

# Plot fit
d_smooth = np.linspace(depths_arr.min(), depths_arr.max(), 200)
ax.plot(d_smooth, A * np.exp(-lambda_eff * d_smooth), '--',
        color='red', linewidth=2, alpha=0.8,
        label=f'Fit: {A:.2f}·exp(−{lambda_eff:.3f}·d)\nR²={r_value**2:.3f}')

ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax.set_xlabel("Circuit Depth $d$", fontsize=14)
ax.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=14)
ax.set_title("(a) XEB Fidelity vs Depth (N=40)", fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(left=5)

# ---- Panel b: N scan with bootstrap CI ----
ax2 = axes[1]
valid_Ns = sorted(N for N in Ns if N in boot_ci_Nscan)
means_N = np.array([agg_Nscan[(N, 12)]["mean"] for N in valid_Ns])
sems_N = np.array([agg_Nscan[(N, 12)]["sem"] for N in valid_Ns])
ci_lo_N = np.array([boot_ci_Nscan[N][0] for N in valid_Ns])
ci_hi_N = np.array([boot_ci_Nscan[N][1] for N in valid_Ns])
Ns_arr = np.array(valid_Ns)

ax2.fill_between(Ns_arr, ci_lo_N, ci_hi_N, alpha=0.25, color=COLORS[1])
ax2.errorbar(Ns_arr, means_N, yerr=sems_N,
             marker='s', color=COLORS[1], linewidth=2, markersize=7,
             capsize=4, label='Mean ± SEM (95% CI shaded)')
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
ax2.set_xlabel("Number of Qubits $N$", fontsize=14)
ax2.set_ylabel("XEB Fidelity $F_{\\mathrm{XEB}}$", fontsize=14)
ax2.set_title("(b) XEB Fidelity vs Qubit Count (d=12)", fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

fig.suptitle("XEB Fidelity Analysis with Statistical Uncertainties", fontsize=14, fontweight='bold')
fig.tight_layout()
fig.savefig(str(REPORT_IMAGES / "fig8_xeb_with_CI.png"), dpi=150, bbox_inches='tight')
print("\nSaved fig8_xeb_with_CI.png")


# -------------------------------------------------------------------------
# 5. Figure: XEB fidelity with log scale and exponential fit
# -------------------------------------------------------------------------
fig2, ax3 = plt.subplots(figsize=(7, 5))

# Only plot on log scale where positive
ax3.semilogy(depths_arr, np.clip(means_arr, 1e-3, None), 'o-',
             color=COLORS[0], linewidth=2, markersize=8, label='Exp. F_XEB (mean)')
# Error bars
for i, (d_i, m_i, s_i) in enumerate(zip(depths_arr, means_arr, sems_arr)):
    if m_i - s_i > 0:
        ax3.semilogy([d_i, d_i], [m_i - s_i, m_i + s_i], '-', color=COLORS[0], alpha=0.5, linewidth=1)

ax3.plot(d_smooth, A * np.exp(-lambda_eff * d_smooth), 'r--', linewidth=2,
         label=f'Exp. decay fit:\n$F={A:.2f} e^{{-{lambda_eff:.3f} d}}$\n$R^2={r_value**2:.3f}$')
ax3.set_xlabel("Circuit Depth $d$", fontsize=14)
ax3.set_ylabel("XEB Fidelity (log scale)", fontsize=14)
ax3.set_title("Exponential Fidelity Decay (N=40)", fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3, which='both')
ax3.set_xlim(left=5)
fig2.tight_layout()
fig2.savefig(str(REPORT_IMAGES / "fig9_log_fit.png"), dpi=150, bbox_inches='tight')
print("Saved fig9_log_fit.png")


# -------------------------------------------------------------------------
# 6. Summary statistics table
# -------------------------------------------------------------------------
summary = {
    "N40_depth_scan": [
        {"N": 40, "d": int(d),
         "n_instances": N40_data[(40, d)]["n"],
         "mean_F_XEB": round(N40_data[(40, d)]["mean"], 4),
         "std_F_XEB": round(N40_data[(40, d)]["std"], 4),
         "sem_F_XEB": round(N40_data[(40, d)]["sem"], 4),
         "CI_95_lo": round(boot_ci_N40[d][0], 4),
         "CI_95_hi": round(boot_ci_N40[d][1], 4)}
        for d in depths
    ],
    "N_scan_d12": [
        {"N": int(N), "d": 12,
         "n_instances": agg_Nscan[(N, 12)]["n"],
         "mean_F_XEB": round(agg_Nscan[(N, 12)]["mean"], 4),
         "std_F_XEB": round(agg_Nscan[(N, 12)]["std"], 4),
         "sem_F_XEB": round(agg_Nscan[(N, 12)]["sem"], 4),
         "CI_95_lo": round(boot_ci_Nscan[N][0], 4),
         "CI_95_hi": round(boot_ci_Nscan[N][1], 4)}
        for N in valid_Ns
    ],
    "exponential_decay_fit": {
        "N": 40,
        "model": "F(d) = A * exp(-lambda * d)",
        "A": round(float(A), 4),
        "lambda": round(float(lambda_eff), 4),
        "R_squared": round(float(r_value**2), 4),
        "p_value": float(p_value)
    }
}
with open(OUTPUTS / "summary_statistics.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nSaved summary_statistics.json")
print("[Done] Additional analysis complete.")
