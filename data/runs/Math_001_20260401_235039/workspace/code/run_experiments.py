"""
VOS Framework: Run All Experiments
====================================
Runs all optimization algorithms on the Lasso problem and saves results.
"""

import sys
import numpy as np
from pathlib import Path

WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_001_20260401_235039')
sys.path.insert(0, str(WORKSPACE / 'code'))

from algorithms import (ista, fista, heavy_ball, admm_lasso,
                        fista_restarted, ode_nesterov,
                        lasso_objective, lyapunov_continuous, lyapunov_discrete)

OUT_DIR = WORKSPACE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
data = np.load(WORKSPACE / 'data' / 'complex_optimization_data.npy',
               allow_pickle=True).item()
A = data['A']
b = data['b']
x_true = data['x_true']
m, n = A.shape
AtA = A.T @ A
Atb = A.T @ b

# Problem parameters
U, s, Vt = np.linalg.svd(A, full_matrices=False)
L = s[0] ** 2          # Lipschitz constant of gradient
mu = s[-1] ** 2        # Strong convexity constant (smooth part only)
kappa = L / mu         # Condition number
lam = 0.1 * np.max(np.abs(A.T @ b))  # Regularization parameter

print(f"L = {L:.4f}, mu = {mu:.4f}, kappa = {kappa:.2f}, lambda = {lam:.4f}")

N_ITERS = 1500
np.random.seed(42)
x0 = np.zeros(n)

# ── Compute reference optimal (run FISTA for many iterations) ──────────────
print("Computing f_star via long FISTA run...")
x_opt_ref, obj_ref, _, _ = fista(A, b, lam, L, n_iters=5000, x0=x0)
f_star = obj_ref[-1]
x_star = x_opt_ref
print(f"f_star (approx) = {f_star:.6f}")
print(f"||x_opt - x_true||_2 = {np.linalg.norm(x_opt_ref - x_true):.4f}")

np.save(OUT_DIR / 'reference_solution.npy',
        {'x_star': x_star, 'f_star': f_star}, allow_pickle=True)

# ── Run experiments ────────────────────────────────────────────────────────

print("\n1. Running ISTA...")
x_ista, obj_ista, iter_ista = ista(A, b, lam, L, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_ista[-1]:.6f}, gap: {obj_ista[-1] - f_star:.2e}")

print("2. Running FISTA...")
x_fista, obj_fista, iter_fista, t_fista = fista(A, b, lam, L, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_fista[-1]:.6f}, gap: {obj_fista[-1] - f_star:.2e}")

print("3. Running Heavy Ball...")
x_hb, obj_hb, iter_hb = heavy_ball(A, b, lam, L, mu, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_hb[-1]:.6f}, gap: {obj_hb[-1] - f_star:.2e}")

print("4. Running ADMM (rho=1.0)...")
x_admm1, obj_admm1, iter_admm1, pres1, dres1 = admm_lasso(
    A, b, lam, rho=1.0, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_admm1[-1]:.6f}, gap: {obj_admm1[-1] - f_star:.2e}")

print("5. Running ADMM (rho=sqrt(L))...")
rho_opt = np.sqrt(L)
x_admm2, obj_admm2, iter_admm2, pres2, dres2 = admm_lasso(
    A, b, lam, rho=rho_opt, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_admm2[-1]:.6f}, gap: {obj_admm2[-1] - f_star:.2e}")

print("6. Running ADMM (rho=L)...")
x_admm3, obj_admm3, iter_admm3, pres3, dres3 = admm_lasso(
    A, b, lam, rho=L, n_iters=N_ITERS, x0=x0)
print(f"   Final obj: {obj_admm3[-1]:.6f}, gap: {obj_admm3[-1] - f_star:.2e}")

print("7. Running Restarted FISTA...")
restart_freq = max(10, int(np.sqrt(kappa)))
x_rfista, obj_rfista, iter_rfista, restarts = fista_restarted(
    A, b, lam, L, mu, restart_freq=restart_freq, n_iters=N_ITERS, x0=x0)
print(f"   Restart freq: {restart_freq}, final obj: {obj_rfista[-1]:.6f}, gap: {obj_rfista[-1] - f_star:.2e}")

# ── ODE integration (2D toy problem for visualization) ─────────────────────
print("8. Running ODE integration (2D toy problem)...")
# Use simple 2D quadratic for trajectory visualization
A2 = np.array([[np.sqrt(L), 0], [0, np.sqrt(mu)]])
b2 = np.zeros(2)
lam2 = 0.0  # no regularization for clean ODE viz
f2 = lambda x: 0.5 * (L * x[0]**2 + mu * x[1]**2)
grad2 = lambda x: np.array([L * x[0], mu * x[1]])
x0_2d = np.array([1.0, 1.0])
t_ode, X_ode, V_ode = ode_nesterov(grad2, x0_2d, t_span=(1e-4, 30.0), n_points=1000)
f_ode = np.array([f2(X_ode[:, i]) for i in range(X_ode.shape[1])])
print(f"   ODE integrated over t in [{t_ode[0]:.4f}, {t_ode[-1]:.2f}]")

# ── ODE 1D for direct Nesterov comparison ─────────────────────────────────
print("9. Running 1D ODE + Nesterov comparison...")
# Use quadratic f(x) = 0.5 * L * x^2 in 1D
f1 = lambda x: 0.5 * L * x[0]**2
grad1 = lambda x: np.array([L * x[0]])
x0_1d = np.array([1.0])
t_ode1, X_ode1, V_ode1 = ode_nesterov(grad1, x0_1d, t_span=(1e-4, 20.0), n_points=500)
f_ode1 = np.array([f1(X_ode1[:, i]) for i in range(X_ode1.shape[1])])

# ── Compute Lyapunov functions ─────────────────────────────────────────────
print("10. Computing Lyapunov functions...")

# Continuous: 2D toy
x_star_2d = np.zeros(2)
E_cont = lyapunov_continuous(t_ode, X_ode, V_ode, f_ode, 0.0, x_star_2d)

# Discrete: FISTA Lyapunov
# Need to collect x iterates densely
x_fista_dense, obj_fista_dense, iter_dense, t_dense = fista(
    A, b, lam, L, n_iters=200, x0=x0)
x_prev_list = [np.zeros(n)] + [x for x in iter_dense[:200]]
t_list = list(t_dense[:201])
k_list = list(range(201))
# Approximate E_k
E_disc = []
for i in range(1, min(len(obj_fista_dense), len(x_prev_list)-1)):
    x_k = iter_dense[min(i, len(iter_dense)-1)]
    x_km1 = x_prev_list[max(i-1, 0)]
    tk = t_list[min(i, len(t_list)-1)]
    fk = obj_fista_dense[min(i, len(obj_fista_dense)-1)]
    term1 = tk**2 * max(fk - f_star, 0.0)
    term2 = 0.5 * np.linalg.norm(x_k - x_star + tk * (x_k - x_km1))**2
    E_disc.append(term1 + term2)
E_disc = np.array(E_disc)

# ── ADMM convergence metrics ────────────────────────────────────────────────
print("11. Running ADMM with various rho values for sensitivity study...")
rho_vals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
admm_results = {}
for rho in rho_vals:
    xr, objs, _, pr, dr = admm_lasso(A, b, lam, rho=rho, n_iters=500, x0=x0)
    admm_results[rho] = {'obj': objs, 'primal': pr, 'dual': dr}
    print(f"   rho={rho:.1f}: final gap = {objs[-1] - f_star:.2e}")

# ── Save all results ───────────────────────────────────────────────────────
results = {
    # Objectives
    'obj_ista': obj_ista, 'obj_fista': obj_fista,
    'obj_hb': obj_hb,
    'obj_admm1': obj_admm1, 'obj_admm2': obj_admm2, 'obj_admm3': obj_admm3,
    'obj_rfista': obj_rfista,
    # ADMM residuals
    'pres_admm1': pres1, 'dres_admm1': dres1,
    'pres_admm2': pres2, 'dres_admm2': dres2,
    # Parameters
    'f_star': f_star, 'x_star': x_star,
    'L': L, 'mu': mu, 'kappa': kappa, 'lam': lam,
    'restart_freq': restart_freq, 'restart_points': restarts,
    # ODE
    't_ode': t_ode, 'X_ode': X_ode, 'V_ode': V_ode, 'f_ode': f_ode,
    't_ode1': t_ode1, 'X_ode1': X_ode1, 'f_ode1': f_ode1,
    'E_cont': E_cont,
    'E_disc': E_disc,
    # ADMM sensitivity
    'admm_results': admm_results,
    # Solutions
    'x_ista': x_ista, 'x_fista': x_fista, 'x_admm2': x_admm2,
    'x_hb': x_hb, 'x_rfista': x_rfista,
    'x_true': x_true,
    'N_ITERS': N_ITERS,
}

np.save(OUT_DIR / 'all_results.npy', results, allow_pickle=True)
print(f"\nAll results saved to {OUT_DIR / 'all_results.npy'}")
print("Experiment run complete.")
