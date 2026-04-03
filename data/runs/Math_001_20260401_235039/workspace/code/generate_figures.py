"""
VOS Framework: Generate All Figures
=====================================
Produces publication-quality figures for the research report.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator
import seaborn as sns
from pathlib import Path

WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_001_20260401_235039')
OUT_DIR = WORKSPACE / 'outputs'
IMG_DIR = WORKSPACE / 'report' / 'images'
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Load results
res = np.load(OUT_DIR / 'all_results.npy', allow_pickle=True).item()
data = np.load(WORKSPACE / 'data' / 'complex_optimization_data.npy', allow_pickle=True).item()

# Unpack
obj_ista    = res['obj_ista']
obj_fista   = res['obj_fista']
obj_hb      = res['obj_hb']
obj_admm1   = res['obj_admm1']
obj_admm2   = res['obj_admm2']
obj_admm3   = res['obj_admm3']
obj_rfista  = res['obj_rfista']
f_star      = res['f_star']
L           = res['L']
mu          = res['mu']
kappa       = res['kappa']
lam         = res['lam']
x_star      = res['x_star']
x_true      = res['x_true']
x_fista     = res['x_fista']
x_admm2     = res['x_admm2']
restart_pts = res['restart_points']
admm_results = res['admm_results']
t_ode       = res['t_ode']
X_ode       = res['X_ode']
V_ode       = res['V_ode']
f_ode       = res['f_ode']
E_cont      = res['E_cont']
E_disc      = res['E_disc']
N           = res['N_ITERS']

# Colors
colors = {
    'ISTA': '#E74C3C',
    'FISTA': '#2ECC71',
    'HeavyBall': '#9B59B6',
    'ADMM_rho1': '#F39C12',
    'ADMM_opt': '#3498DB',
    'ADMM_L': '#1ABC9C',
    'RFISTA': '#E91E63',
}

iters = np.arange(1, N + 1)

# ── Figure 2: Main convergence comparison ─────────────────────────────────
print("Generating Figure 2: Convergence Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: absolute objective gap
ax = axes[0]
gap_ista   = np.maximum(obj_ista - f_star, 1e-16)
gap_fista  = np.maximum(obj_fista - f_star, 1e-16)
gap_hb     = np.maximum(obj_hb - f_star, 1e-16)
gap_admm1  = np.maximum(obj_admm1 - f_star, 1e-16)
gap_admm2  = np.maximum(obj_admm2 - f_star, 1e-16)
gap_rfista = np.maximum(obj_rfista - f_star, 1e-16)

ax.semilogy(iters, gap_ista,   color=colors['ISTA'],     linewidth=1.8, label='ISTA / Prox-GD')
ax.semilogy(iters, gap_fista,  color=colors['FISTA'],    linewidth=1.8, label='FISTA (Nesterov AGM)')
ax.semilogy(iters, gap_hb,     color=colors['HeavyBall'],linewidth=1.8, label='Heavy Ball (Polyak)')
ax.semilogy(iters, gap_admm1,  color=colors['ADMM_rho1'],linewidth=1.8, label=r'ADMM ($\rho=1$)')
ax.semilogy(iters, gap_admm2,  color=colors['ADMM_opt'], linewidth=1.8, label=r'ADMM ($\rho=\sqrt{L}$)', linestyle='--')
ax.semilogy(iters, gap_rfista, color=colors['RFISTA'],   linewidth=1.8, label=f'Restarted FISTA (freq={int(res["restart_freq"])})', linestyle='-.')

# Add theoretical rate lines
k_th = iters
C0 = gap_ista[0]
ax.semilogy(k_th[10:], C0 / k_th[10:], 'k:', linewidth=1, alpha=0.5, label=r'$O(1/k)$ rate')
ax.semilogy(k_th[10:], C0 * 4 / k_th[10:]**2, 'k--', linewidth=1, alpha=0.5, label=r'$O(1/k^2)$ rate')

ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap $f(x_k) - f^*$', fontsize=12)
ax.set_title('Convergence Comparison on High-Dimensional Lasso', fontsize=12)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(1, N)
ax.set_ylim(1e-14, None)
ax.grid(True, which='both', alpha=0.3)

# Right: normalized gap (scaled by k^2)
ax = axes[1]
scaled_ista  = iters**2 * gap_ista / gap_ista[0]
scaled_fista = iters**2 * gap_fista / gap_fista[0]
ax.semilogy(iters[:800], scaled_ista[:800],  color=colors['ISTA'],  linewidth=2, label='ISTA: $k^2 (f - f^*)$')
ax.semilogy(iters[:800], scaled_fista[:800], color=colors['FISTA'], linewidth=2, label='FISTA: $k^2 (f - f^*)$')
ax.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Constant (optimal for FISTA)')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Scaled gap $k^2 (f(x_k) - f^*) / (f(x_0) - f^*)$', fontsize=12)
ax.set_title('Scaled Convergence: Verifying $O(1/k^2)$ Rate', fontsize=12)
ax.legend(fontsize=10)
ax.set_xlim(1, 800)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig2_convergence_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig2_convergence_comparison.png")

# ── Figure 3: Continuous-time ODE Trajectory ─────────────────────────────
print("Generating Figure 3: ODE Trajectory...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: 2D trajectory
ax = axes[0]
x_traj = X_ode[0, :]
y_traj = X_ode[1, :]
sc = ax.scatter(x_traj, y_traj, c=t_ode, cmap='plasma', s=5, alpha=0.7, zorder=3)
ax.plot(x_traj, y_traj, 'k-', linewidth=0.5, alpha=0.3, zorder=2)
ax.plot(x_traj[0], y_traj[0], 'go', markersize=8, label='Start', zorder=5)
ax.plot(0, 0, 'r*', markersize=12, label='Optimum $x^*$', zorder=5)
plt.colorbar(sc, ax=ax, label='Time $t$')
ax.set_xlabel('$X_1(t)$', fontsize=12)
ax.set_ylabel('$X_2(t)$', fontsize=12)
ax.set_title('ODE Trajectory: $\\ddot{X} + \\frac{3}{t}\\dot{X} + \\nabla f(X)=0$\n'
             '(2D quadratic: $f=\\frac{1}{2}(L x_1^2 + \\mu x_2^2)$)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel B: f(X(t)) - f* vs t^2 (should be O(1/t^2))
ax = axes[1]
f_gap_ode = np.maximum(f_ode - 0.0, 1e-16)  # f_star=0 for 2D toy
ax.loglog(t_ode[5:], f_gap_ode[5:], 'b-', linewidth=2, label='ODE: $f(X(t)) - f^*$')
# Theoretical bound
t_th = t_ode[5:]
ax.loglog(t_th, f_gap_ode[5] * (t_ode[5]/t_th)**2, 'r--', linewidth=1.5, alpha=0.8,
          label='$O(1/t^2)$ bound')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('$f(X(t)) - f^*$', fontsize=12)
ax.set_title('ODE Convergence Rate\n$f(X(t)) - f^* = O(1/t^2)$', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# Panel C: Lyapunov function E(t)
ax = axes[2]
E_norm = E_cont / E_cont[0]
ax.plot(t_ode, E_norm, 'purple', linewidth=2, label='$E(t)/E(0)$ (Lyapunov)')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('Normalized Lyapunov $E(t)/E(0)$', fontsize=12)
ax.set_title('Lyapunov Function:\n'
             '$E(t) = t^2(f-f^*) + 2\\|X + \\frac{t}{2}\\dot{X} - x^*\\|^2$', fontsize=10)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig3_ode_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig3_ode_trajectory.png")

# ── Figure 4: Lyapunov Functions (discrete + continuous) ─────────────────
print("Generating Figure 4: Lyapunov Functions...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Continuous Lyapunov decay
ax = axes[0]
ax.semilogy(t_ode, np.maximum(E_cont, 1e-16), 'purple', linewidth=2.5,
            label='$E(t) = t^2(f-f^*) + 2\\|X + \\frac{t}{2}\\dot{X}-x^*\\|^2$')
# Upper bound: initial value
ax.axhline(E_cont[0], color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
           label='$E(0) = \\|x_0 - x^*\\|^2$')
ax.set_xlabel('Time $t$', fontsize=12)
ax.set_ylabel('$E(t)$', fontsize=12)
ax.set_title('Continuous-Time Lyapunov Function\n(Non-increasing along ODE trajectories)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', alpha=0.3)

# Right: Discrete Lyapunov for FISTA
ax = axes[1]
k_disc = np.arange(1, len(E_disc) + 1)
ax.semilogy(k_disc, np.maximum(E_disc, 1e-16), 'g-', linewidth=2,
            label='$E_k = t_k^2(f-f^*) + \\frac{1}{2}\\|x_k - x^* + t_k(x_k-x_{k-1})\\|^2$')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('$E_k$', fontsize=12)
ax.set_title('Discrete Lyapunov Function for FISTA\n(Non-increasing $\\Rightarrow$ $O(1/k^2)$ bound)', fontsize=11)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig4_lyapunov.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig4_lyapunov.png")

# ── Figure 5: ADMM Analysis ───────────────────────────────────────────────
print("Generating Figure 5: ADMM Analysis...")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Panel A: ADMM convergence for different rho
ax = axes[0]
cmap = plt.cm.viridis
rho_list = sorted(admm_results.keys())
colors_rho = [cmap(i / len(rho_list)) for i in range(len(rho_list))]
iters_admm = np.arange(1, 501)

for i, rho in enumerate(rho_list):
    gap = np.maximum(admm_results[rho]['obj'] - f_star, 1e-16)
    ax.semilogy(iters_admm, gap, color=colors_rho[i],
                linewidth=1.8, label=f'$\\rho={rho}$')

ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap', fontsize=12)
ax.set_title('ADMM Convergence vs. Penalty $\\rho$', fontsize=11)
ax.legend(fontsize=8, ncol=2)
ax.grid(True, which='both', alpha=0.3)

# Panel B: Final gap vs rho
ax = axes[1]
rho_arr = np.array(rho_list)
final_gaps = np.array([admm_results[r]['obj'][-1] - f_star for r in rho_list])
ax.loglog(rho_arr, np.maximum(final_gaps, 1e-16), 'bo-', linewidth=2, markersize=7)
ax.axvline(x=np.sqrt(L), color='r', linestyle='--', linewidth=1.5,
           label=f'$\\rho^*=\\sqrt{{L}}={np.sqrt(L):.1f}$')
ax.set_xlabel('Penalty parameter $\\rho$', fontsize=12)
ax.set_ylabel('Final objective gap (500 iters)', fontsize=12)
ax.set_title('ADMM Sensitivity to $\\rho$', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

# Panel C: Primal + Dual residuals for optimal rho
ax = axes[2]
pres2 = res['pres_admm2'][:500]
dres2 = res['dres_admm2'][:500]
ax.semilogy(iters_admm[:len(pres2)], np.maximum(pres2, 1e-16),
            'b-', linewidth=2, label='Primal residual $\\|x-z\\|$')
ax.semilogy(iters_admm[:len(dres2)], np.maximum(dres2, 1e-16),
            'r-', linewidth=2, label='Dual residual $\\rho\\|z-z_{k-1}\\|$')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Residual Norm', fontsize=12)
ax.set_title(f'ADMM Residuals ($\\rho=\\sqrt{{L}}={np.sqrt(L):.1f}$)\nVariable/Operator Splitting', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig5_admm_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig5_admm_analysis.png")

# ── Figure 6: VOS Framework -- FISTA vs ADMM vs Restarted ─────────────────
print("Generating Figure 6: VOS Framework Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.semilogy(iters, gap_fista,  color='#2ECC71', linewidth=2.5, label='FISTA (Nesterov AGM)')
ax.semilogy(iters, gap_rfista, color='#E91E63', linewidth=2.5, linestyle='-.', label=f'Restarted FISTA')

# Add ADMM with optimal rho
ax.semilogy(iters, gap_admm2, color='#3498DB', linewidth=2.5, linestyle='--',
            label=r'ADMM ($\rho=\sqrt{L}$)')

# Mark restarts
for rp in restart_pts[:15]:
    ax.axvline(x=rp, color='#E91E63', alpha=0.25, linewidth=0.8)

# Reference rates
k_ref = iters[10:]
ax.semilogy(k_ref, gap_fista[0] * 4 / k_ref**2, 'k--', linewidth=1.0, alpha=0.4,
            label='$O(1/k^2)$')
lin_rate = np.exp(-np.sqrt(mu/L))
ax.semilogy(k_ref, gap_fista[0] * lin_rate**k_ref, 'k:', linewidth=1.0, alpha=0.4,
            label=r'$O(e^{-\sqrt{\mu/L}\cdot k})$')

ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap $f(x_k) - f^*$', fontsize=12)
ax.set_title('VOS Framework: Accelerated Methods\nLasso ($n=2000$, $\\kappa=100$)', fontsize=11)
ax.legend(fontsize=9)
ax.set_xlim(1, N)
ax.grid(True, which='both', alpha=0.3)

# Right: Solution quality (support recovery)
ax = axes[1]
n = len(x_star)
idx = np.arange(n)
true_support = np.where(x_true != 0)[0]
est_support_fista = np.where(np.abs(x_fista) > 1e-4)[0]
est_support_admm = np.where(np.abs(x_admm2) > 1e-4)[0]

# Show first 200 coefficients
mask = idx < 200
ax.stem(idx[mask & (x_true != 0)], x_true[mask & (x_true != 0)],
        linefmt='k-', markerfmt='ko', basefmt='k-',
        label=f'True $x^*$ ({len(true_support)} nonzero)')
ax.scatter(est_support_fista[est_support_fista < 200],
           x_fista[est_support_fista[est_support_fista < 200]],
           color='green', s=40, zorder=5, marker='^', alpha=0.8,
           label=f'FISTA ({len(est_support_fista)} nonzero)')
ax.scatter(est_support_admm[est_support_admm < 200],
           x_admm2[est_support_admm[est_support_admm < 200]],
           color='blue', s=40, zorder=5, marker='x', alpha=0.8, linewidths=1.5,
           label=f'ADMM ({len(est_support_admm)} nonzero)')
ax.set_xlabel('Coefficient Index (first 200)', fontsize=12)
ax.set_ylabel('Coefficient Value', fontsize=12)
ax.set_title('Support Recovery: FISTA vs ADMM\n(True sparsity: 5%)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(-5, 205)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig6_vos_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig6_vos_framework.png")

# ── Figure 7: Theoretical Framework Illustration ─────────────────────────
print("Generating Figure 7: VOS Framework Diagram...")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

# Panel A: log-log convergence showing rates
ax = axes[0, 0]
ax.loglog(iters, gap_ista,  color=colors['ISTA'],  linewidth=2, label='ISTA: $O(1/k)$')
ax.loglog(iters, gap_fista, color=colors['FISTA'], linewidth=2, label='FISTA: $O(1/k^2)$')
ax.loglog(iters, gap_admm2, color=colors['ADMM_opt'], linewidth=2, linestyle='--',
          label=r'ADMM ($\rho^*$): linear')
ax.loglog(iters, gap_rfista, color=colors['RFISTA'], linewidth=2, linestyle='-.',
          label='Restarted FISTA: linear')

# Rate lines in log-log
k_ref = iters
ax.loglog(k_ref[5:], 1e4 / k_ref[5:], 'k-', linewidth=0.8, alpha=0.4)
ax.loglog(k_ref[5:], 1e5 / k_ref[5:]**2, 'k--', linewidth=0.8, alpha=0.4)
ax.text(50, 1e4/50 * 1.5, 'slope $-1$', fontsize=8, alpha=0.6)
ax.text(30, 1e5/30**2 * 1.5, 'slope $-2$', fontsize=8, alpha=0.6)

ax.set_xlabel('$k$ (log scale)', fontsize=12)
ax.set_ylabel('$f(x_k) - f^*$ (log scale)', fontsize=12)
ax.set_title('Log-Log Convergence Rates', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.set_xlim(1, N)

# Panel B: Heavy ball vs FISTA oscillations
ax = axes[0, 1]
window = 400
ax.semilogy(iters[:window], gap_fista[:window], color=colors['FISTA'],
            linewidth=1.8, label='FISTA (Nesterov momentum)')
ax.semilogy(iters[:window], gap_hb[:window], color=colors['HeavyBall'],
            linewidth=1.8, label='Heavy Ball (Polyak constant momentum)', alpha=0.9)
ax.semilogy(iters[:window], gap_ista[:window], color=colors['ISTA'],
            linewidth=1.5, label='ISTA (no momentum)', alpha=0.7)
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap', fontsize=12)
ax.set_title('Momentum Comparison: First 400 Iterations', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# Panel C: Restarted FISTA linear convergence proof
ax = axes[1, 0]
gap_r = np.maximum(obj_rfista - f_star, 1e-16)
ax.semilogy(iters, gap_r, color=colors['RFISTA'], linewidth=2, label='Restarted FISTA')
# Mark restarts
if restart_pts:
    ax.axvline(x=restart_pts[0], color='#E91E63', alpha=0.3, linewidth=1,
               label='Restart events')
    for rp in restart_pts[1:20]:
        ax.axvline(x=rp, color='#E91E63', alpha=0.3, linewidth=1)

# Linear convergence bound
freq = int(res['restart_freq'])
contraction = 0.5  # each restart cycle reduces by ~half
k_check = np.arange(0, N, freq)
bound_pts = [gap_r[0] * contraction**(i) for i in range(len(k_check))]
ax.semilogy(k_check, bound_pts, 'r--', linewidth=1.5, alpha=0.7,
            label='Linear rate $O(e^{-ck})$')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap', fontsize=12)
ax.set_title(f'Restarted FISTA: Linear Convergence\n(Restart period = {freq} = $\\lfloor\\sqrt{{\\kappa}}\\rfloor$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# Panel D: ADMM operator splitting interpretation
ax = axes[1, 1]
iters_admm = np.arange(1, 501)
gap_admm_r1 = np.maximum(admm_results[1.0]['obj'] - f_star, 1e-16)
gap_admm_r5 = np.maximum(admm_results[5.0]['obj'] - f_star, 1e-16)
gap_admm_r10 = np.maximum(admm_results[10.0]['obj'] - f_star, 1e-16)
gap_admm_r20 = np.maximum(admm_results[20.0]['obj'] - f_star, 1e-16)
gap_admm_ropt = np.maximum(admm_results[rho_list[rho_list.index(min(rho_list, key=lambda r: admm_results[r]['obj'][-1]))]]['obj'] - f_star, 1e-16)

ax.semilogy(iters_admm, gap_admm_r1, linewidth=2, label='ADMM $\\rho=1$')
ax.semilogy(iters_admm, gap_admm_r5, linewidth=2, label='ADMM $\\rho=5$')
ax.semilogy(iters_admm, gap_admm_r10, linewidth=2, label='ADMM $\\rho=10$')
ax.semilogy(iters_admm, gap_admm_r20, linewidth=2, label='ADMM $\\rho=20$')
ax.set_xlabel('Iteration $k$', fontsize=12)
ax.set_ylabel('Objective Gap', fontsize=12)
ax.set_title('ADMM: Variable Splitting Convergence\n'
             '($x_{k+1} = (A^TA + \\rho I)^{-1}(A^Tb + \\rho(z_k-u_k))$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig7_theoretical_framework.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig7_theoretical_framework.png")

# ── Figure 8: Solution coefficient comparison ─────────────────────────────
print("Generating Figure 8: Solution Quality...")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

n = len(x_star)
x_fista_sol = res['x_fista']
x_hb_sol = res['x_hb']
x_admm2_sol = res['x_admm2']
x_rfista_sol = res['x_rfista']

solutions = {
    'FISTA': x_fista_sol, 'Heavy Ball': x_hb_sol,
    'ADMM': x_admm2_sol, 'Restarted FISTA': x_rfista_sol
}
sol_colors = ['#2ECC71', '#9B59B6', '#3498DB', '#E91E63']

for idx_ax, (name, x_sol) in enumerate(solutions.items()):
    ax = axes[idx_ax // 2, idx_ax % 2]
    # Show subset
    subset = slice(0, 300)
    idx_range = np.arange(300)

    ax.vlines(idx_range, 0, x_true[subset], colors='black', linewidth=0.8,
              alpha=0.5, label='True $x_{true}$')
    ax.vlines(idx_range, 0, x_sol[subset], colors=sol_colors[idx_ax],
              linewidth=0.8, alpha=0.5)
    ax.scatter(idx_range, x_sol[subset], c=sol_colors[idx_ax], s=8, zorder=4,
               label=f'{name} ($\\|x-x^*\\|={np.linalg.norm(x_sol - x_star):.2f}$)')
    ax.scatter(np.where(x_true[subset] != 0)[0],
               x_true[np.where(x_true[subset] != 0)[0]],
               c='black', s=20, marker='D', zorder=5)

    ax.set_xlabel('Coefficient Index (0-299)', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title(f'{name} Solution\n'
                 f'$\\|x_{{sol}}-x_{{true}}\\|={np.linalg.norm(x_sol-x_true):.3f}$, '
                 f'NNZ={np.sum(np.abs(x_sol)>1e-4)}', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 305)

plt.suptitle('Solution Quality: Sparse Recovery Performance\n'
             '(True solution has 100 nonzero coefficients)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig8_solution_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig8_solution_quality.png")

# ── Figure 9: VOS Unified Framework Summary ───────────────────────────────
print("Generating Figure 9: VOS Framework Summary...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Panel A: Iterations to reach tolerance
tols = np.logspace(-2, -12, 40)
iters_to_tol = {}
for name, gap_arr in [('ISTA', gap_ista), ('FISTA', gap_fista),
                       ('HeavyBall', gap_hb), ('ADMM (opt.)', gap_admm2),
                       ('Restarted FISTA', gap_rfista)]:
    its = []
    for tol in tols:
        crossed = np.where(gap_arr < tol)[0]
        its.append(crossed[0] + 1 if len(crossed) > 0 else N)
    iters_to_tol[name] = np.array(its)

ax = axes[0]
color_map2 = {'ISTA': '#E74C3C', 'FISTA': '#2ECC71', 'HeavyBall': '#9B59B6',
              'ADMM (opt.)': '#3498DB', 'Restarted FISTA': '#E91E63'}
for name, vals in iters_to_tol.items():
    ax.loglog(tols, vals, linewidth=2, label=name, color=color_map2.get(name, 'gray'))
ax.set_xlabel('Tolerance $\\epsilon$', fontsize=12)
ax.set_ylabel('Iterations to reach $f(x_k)-f^* \\leq \\epsilon$', fontsize=12)
ax.set_title('Work Complexity Comparison', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.invert_xaxis()

# Panel B: ODE vs discrete FISTA (1D)
ax = axes[1]
# Discrete FISTA on 1D quadratic
f1d = lambda x: 0.5 * L * x**2
x_k = np.array([1.0])
y_k = x_k.copy()
t_k = 1.0
step = 1.0 / L
f1d_iters_nesterov = []
iters_1d = []
x_k_1d = 1.0
y_k_1d = 1.0
t_k_1d = 1.0
for k_1d in range(200):
    grad_yk = L * y_k_1d
    x_new_1d = y_k_1d - step * grad_yk
    t_new_1d = (1 + np.sqrt(1 + 4*t_k_1d**2)) / 2
    y_k_1d = x_new_1d + ((t_k_1d - 1)/t_new_1d) * (x_new_1d - x_k_1d)
    x_k_1d = x_new_1d
    t_k_1d = t_new_1d
    f1d_iters_nesterov.append(0.5 * L * x_k_1d**2)
    iters_1d.append(k_1d + 1)

f1d_iters_nesterov = np.array(f1d_iters_nesterov)
iters_1d = np.array(iters_1d)

# Map ODE time to iteration index: k ~ t / sqrt(step)
t_ode1 = res['t_ode1']
t_mapped = t_ode1 / np.sqrt(1.0/L)
idx_valid = (t_mapped >= 1) & (t_mapped <= 200)

ax.semilogy(iters_1d, f1d_iters_nesterov, 'g-', linewidth=2.5,
            label='FISTA (discrete, Nesterov 1983)')
ax.semilogy(t_mapped[idx_valid], np.maximum(res['f_ode1'][idx_valid], 1e-16),
            'b--', linewidth=2.5, alpha=0.8, label='ODE $\\ddot{X}+\\frac{3}{t}\\dot{X}+\\nabla f=0$')
ax.semilogy(iters_1d, 0.5*L / iters_1d**2, 'k:', linewidth=1.5, alpha=0.6,
            label='$O(1/k^2)$ theoretical bound')
ax.set_xlabel('Iteration $k$ (or $t / \\sqrt{s}$)', fontsize=12)
ax.set_ylabel('$f(x_k) - f^*$', fontsize=12)
ax.set_title('Discrete–Continuous Equivalence\n(1D quadratic $f=\\frac{L}{2}x^2$)', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)

# Panel C: Speedup ratio FISTA/ISTA
ax = axes[2]
# At each tolerance, compute speedup
speedup_fista_ista = iters_to_tol['ISTA'] / np.maximum(iters_to_tol['FISTA'], 1)
speedup_rfista_ista = iters_to_tol['ISTA'] / np.maximum(iters_to_tol['Restarted FISTA'], 1)
speedup_admm_ista = iters_to_tol['ISTA'] / np.maximum(iters_to_tol['ADMM (opt.)'], 1)

ax.loglog(tols, speedup_fista_ista, 'g-', linewidth=2, label='FISTA / ISTA speedup')
ax.loglog(tols, speedup_rfista_ista, 'r-.', linewidth=2, label='Restarted FISTA / ISTA')
ax.loglog(tols, speedup_admm_ista, 'b--', linewidth=2, label='ADMM (opt.) / ISTA')
ax.axhline(1, color='gray', linewidth=1, linestyle=':')
ax.set_xlabel('Tolerance $\\epsilon$', fontsize=12)
ax.set_ylabel('Speedup Factor', fontsize=12)
ax.set_title('Speedup vs. ISTA Baseline', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, which='both', alpha=0.3)
ax.invert_xaxis()

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig9_vos_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved fig9_vos_summary.png")

print("\nAll figures generated successfully!")
print("Files in report/images/:")
for f in sorted(IMG_DIR.iterdir()):
    print(f"  {f.name}")
