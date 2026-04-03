"""
autonomous_optimization.py
Bayesian Optimization for materials synthesis parameters.
Optimizes temperature and reaction time to maximize a simulated yield function.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
import json, os

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Material_001_20260401_224352"
OUT_DIR  = os.path.join(WORKSPACE, "outputs")
FIG_DIR  = os.path.join(WORKSPACE, "report/images")

np.random.seed(42)

# ── 1. Load optimization parameters from dataset ─────────────────────────────
import ast
DATA_PATH = os.path.join(WORKSPACE, "data/M-AI-Synth__Materials_AI_Dataset_.txt")
def parse_dataset(path):
    sections = {}
    current_section = None; current_arrays = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("#"):
                if current_section is not None:
                    sections[current_section] = current_arrays
                current_section = line.lstrip("# ").strip(); current_arrays = []
            else:
                try:
                    arr = ast.literal_eval(line)
                    current_arrays.append(np.array(arr, dtype=float))
                except: pass
    if current_section is not None:
        sections[current_section] = current_arrays
    return sections

sections = parse_dataset(DATA_PATH)
opt_raw  = sections["文件3: autonomous_optimization.py 数据"]
T_bounds = opt_raw[0]  # [200, 500]
t_bounds = opt_raw[1]  # [10, 30]
T_target = float(opt_raw[2][0])   # 350
t_target = float(opt_raw[3][0])   # 20
lr_0     = float(opt_raw[4][0])   # 0.1
n_init   = int(opt_raw[5][0])     # 10

print(f"Optimization bounds: T∈{T_bounds}, t∈{t_bounds}")
print(f"True optimum: T={T_target}°C, t={t_target} min")
print(f"n_init={n_init}, lr={lr_0}")

# ── 2. Define simulated yield function ────────────────────────────────────────
# Physics-inspired multi-modal yield surface:
#   - Primary peak at (T_target, t_target)
#   - Secondary ridge at high temperature (decomposition side-product)
#   - Non-trivial noise structure to simulate experimental scatter

def yield_function(T, t, noise=0.04):
    """Simulated experimental yield (0–1 scale)."""
    rng = np.random.default_rng(int(abs(T * 7 + t * 13)) % 2**31)

    # Main Gaussian peak (desired phase formation)
    peak1 = np.exp(-((T - T_target)**2 / (2*40**2) + (t - t_target)**2 / (2*2.5**2)))

    # Competing secondary reaction (broad ridge at high T)
    peak2 = 0.35 * np.exp(-((T - 460)**2 / (2*30**2) + (t - 25)**2 / (2*4**2)))

    # Arrhenius-like activation: yield drops at low T/t
    activation = 1.0 / (1.0 + np.exp(-(T - 260) / 25.0))
    time_factor = 1.0 / (1.0 + np.exp(-(t - 12)  / 2.0))

    raw = (peak1 * 0.9 + peak2) * activation * time_factor
    return float(np.clip(raw + rng.normal(0, noise), 0, 1))

# Reference: exhaustive grid over the search space
T_grid  = np.linspace(T_bounds[0], T_bounds[1], 100)
t_grid  = np.linspace(t_bounds[0], t_bounds[1], 100)
TT, tt  = np.meshgrid(T_grid, t_grid)
yield_grid = np.vectorize(yield_function)(TT, tt, noise=0.0)

print(f"True optimum in grid: T={T_grid[np.unravel_index(yield_grid.argmax(), yield_grid.shape)[1]]:.1f}°C, "
      f"t={t_grid[np.unravel_index(yield_grid.argmax(), yield_grid.shape)[0]]:.1f} min, "
      f"yield={yield_grid.max():.3f}")

# ── 3. Acquisition function: Expected Improvement ────────────────────────────
def expected_improvement(X_cand, gp, y_best, xi=0.01):
    mu, sigma = gp.predict(X_cand, return_std=True)
    sigma = sigma.reshape(-1, 1)
    mu    = mu.reshape(-1, 1)
    imp   = mu - y_best - xi
    Z     = imp / (sigma + 1e-9)
    ei    = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-10] = 0.0
    return ei.flatten()

def next_query(gp, y_best, bounds):
    """Maximise EI via multi-start L-BFGS-B."""
    best_x, best_ei = None, -np.inf
    rng = np.random.default_rng(0)
    x0s = rng.uniform([bounds[0][0], bounds[1][0]],
                      [bounds[0][1], bounds[1][1]], size=(50, 2))
    for x0 in x0s:
        res = minimize(lambda x: -expected_improvement(x.reshape(1,-1), gp, y_best),
                       x0,
                       bounds=[(bounds[0][0], bounds[0][1]),
                               (bounds[1][0], bounds[1][1])],
                       method='L-BFGS-B')
        ei_val = -res.fun
        if ei_val > best_ei:
            best_ei = ei_val
            best_x  = res.x
    return best_x

# ── 4. Run Bayesian Optimisation ─────────────────────────────────────────────
bounds = [(T_bounds[0], T_bounds[1]),
          (t_bounds[0], t_bounds[1])]

# Initial random exploration (n_init points)
rng_init = np.random.default_rng(42)
T_init   = rng_init.uniform(T_bounds[0], T_bounds[1], n_init)
t_init   = rng_init.uniform(t_bounds[0], t_bounds[1], n_init)
X_obs    = np.column_stack([T_init, t_init])
y_obs    = np.array([yield_function(T, t) for T, t in X_obs])

kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
    length_scale=[30., 3.], length_scale_bounds=[(5,200),(0.5,15)], nu=2.5
) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 0.1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)

N_ITER = 40
best_so_far  = [y_obs.max()]
query_Ts, query_ts = list(T_init), list(t_init)
query_ys     = list(y_obs)

print(f"\nStarting Bayesian Optimisation ({N_ITER} iterations) …")
for it in range(N_ITER):
    gp.fit(X_obs, y_obs)
    y_best  = y_obs.max()
    x_next  = next_query(gp, y_best, bounds)
    y_next  = yield_function(x_next[0], x_next[1])

    X_obs   = np.vstack([X_obs, x_next])
    y_obs   = np.append(y_obs, y_next)
    best_so_far.append(y_obs.max())
    query_Ts.append(x_next[0]); query_ts.append(x_next[1]); query_ys.append(y_next)

    if (it + 1) % 10 == 0:
        best_idx = y_obs.argmax()
        print(f"  Iter {it+1:3d}: best yield={y_obs.max():.4f} "
              f"@ T={X_obs[best_idx,0]:.1f}°C, t={X_obs[best_idx,1]:.1f}min")

best_idx  = y_obs.argmax()
T_opt     = X_obs[best_idx, 0]
t_opt     = X_obs[best_idx, 1]
y_opt     = y_obs[best_idx]
err_T     = abs(T_opt - T_target)
err_t     = abs(t_opt - t_target)
print(f"\nResult:  T={T_opt:.1f}°C  t={t_opt:.1f} min  yield={y_opt:.4f}")
print(f"Error vs. true optimum:  ΔT={err_T:.1f}°C  Δt={err_t:.2f} min")

# ── 5. Comparison: Random baseline ───────────────────────────────────────────
rng_rand = np.random.default_rng(123)
T_rand   = rng_rand.uniform(T_bounds[0], T_bounds[1], n_init + N_ITER)
t_rand   = rng_rand.uniform(t_bounds[0], t_bounds[1], n_init + N_ITER)
y_rand   = np.array([yield_function(T, t) for T, t in zip(T_rand, t_rand)])
best_rand = np.maximum.accumulate(y_rand)

# ── 6. Save results ───────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "X_obs.npy"),       X_obs)
np.save(os.path.join(OUT_DIR, "y_obs.npy"),       y_obs)
np.save(os.path.join(OUT_DIR, "best_so_far.npy"), best_so_far)
np.save(os.path.join(OUT_DIR, "yield_grid.npy"),  yield_grid)

metrics_opt = {
    "n_iterations": N_ITER, "n_init": n_init,
    "best_yield":     float(f"{y_opt:.4f}"),
    "optimal_T":      float(f"{T_opt:.1f}"),
    "optimal_t":      float(f"{t_opt:.2f}"),
    "error_T":        float(f"{err_T:.1f}"),
    "error_t":        float(f"{err_t:.2f}"),
    "convergence_iter": int(np.argmax(best_so_far > 0.85)) if (np.array(best_so_far) > 0.85).any() else -1,
}
with open(os.path.join(OUT_DIR, "optimization_metrics.json"), "w") as f:
    json.dump(metrics_opt, f, indent=2)

# ── 7. Visualisation ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Autonomous Bayesian Optimization of Synthesis Parameters",
             fontsize=14, fontweight='bold')

# (a) True yield surface
ax = axes[0, 0]
cs = ax.contourf(TT, tt, yield_grid, levels=20, cmap='YlOrRd')
ax.scatter(T_init, t_init, c='white', s=60, marker='o', zorder=4,
           edgecolors='gray', linewidths=0.8, label='Initial (random)')
ax.scatter([T_target], [t_target], c='cyan', s=150, marker='*', zorder=6,
           label=f'True optimum ({T_target}°C, {t_target}min)')
plt.colorbar(cs, ax=ax, label='Yield')
ax.set_title("True Yield Surface", fontsize=11)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Time (min)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (b) GP posterior mean after all observations
gp.fit(X_obs, y_obs)
X_flat = np.column_stack([TT.ravel(), tt.ravel()])
mu_flat, _ = gp.predict(X_flat, return_std=True)
mu_grid = mu_flat.reshape(TT.shape)

ax = axes[0, 1]
cs2 = ax.contourf(TT, tt, mu_grid, levels=20, cmap='YlOrRd')
q_Ts = np.array(query_Ts); q_ts = np.array(query_ts); q_ys = np.array(query_ys)
sc  = ax.scatter(q_Ts[n_init:], q_ts[n_init:], c=q_ys[n_init:],
                 cmap='Blues', s=60, zorder=4, edgecolors='navy',
                 linewidths=0.6, vmin=0, vmax=1)
ax.scatter(T_opt, t_opt, c='lime', s=200, marker='*', zorder=6,
           label=f'Found optimum\n({T_opt:.0f}°C, {t_opt:.1f}min, y={y_opt:.3f})')
plt.colorbar(cs2, ax=ax, label='GP mean yield')
ax.set_title("GP Posterior (after 50 experiments)", fontsize=11)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Time (min)")
ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)

# (c) EI acquisition function at last iteration
_, std_flat = gp.predict(X_flat, return_std=True)
ei_flat = expected_improvement(X_flat, gp, y_obs.max())
ei_grid = ei_flat.reshape(TT.shape)
ax = axes[0, 2]
cs3 = ax.contourf(TT, tt, ei_grid, levels=20, cmap='plasma')
ax.scatter(T_opt, t_opt, c='lime', s=150, marker='*', zorder=5,
           label='Current best')
plt.colorbar(cs3, ax=ax, label='Expected Improvement')
ax.set_title("EI Acquisition Function\n(Final Iteration)", fontsize=11)
ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Time (min)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (d) Convergence curve
ax = axes[1, 0]
iters_all = np.arange(len(best_so_far))
ax.plot(iters_all, best_so_far, 'b-o', linewidth=2, markersize=5,
        label='BO best yield', markevery=5)
ax.plot(np.arange(len(best_rand)), best_rand, 'r--', linewidth=2,
        alpha=0.7, label='Random search')
ax.axhline(yield_grid.max(), color='gray', linestyle=':', linewidth=1.5,
           label=f'True max = {yield_grid.max():.3f}')
ax.fill_between(iters_all, best_rand[:len(iters_all)], best_so_far,
                alpha=0.15, color='blue', label='BO advantage')
ax.set_title("Convergence: Bayesian vs. Random Search", fontsize=11)
ax.set_xlabel("Number of experiments")
ax.set_ylabel("Best yield achieved")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (e) Query points over iterations (temperature trace)
ax = axes[1, 1]
iter_idx = np.arange(len(query_Ts))
sc1 = ax.scatter(iter_idx[n_init:], query_Ts[n_init:], c=query_ys[n_init:],
                 cmap='RdYlGn', s=50, vmin=0, vmax=1, zorder=3)
ax.scatter(iter_idx[:n_init], query_Ts[:n_init], c='gray', s=40, marker='s',
           alpha=0.6, zorder=2, label='Initial (random)')
ax.axhline(T_target, color='blue', linestyle='--', linewidth=1.5,
           label=f'Optimal T={T_target}°C')
plt.colorbar(sc1, ax=ax, label='Yield')
ax.set_title("Temperature Queries Over Iterations", fontsize=11)
ax.set_xlabel("Iteration"); ax.set_ylabel("Temperature (°C)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# (f) Yield distribution: BO vs Random
ax = axes[1, 2]
ax.hist(q_ys[n_init:], bins=15, color='steelblue', alpha=0.7,
        edgecolor='white', label=f'BO queries (n={N_ITER})')
ax.hist(y_rand[n_init:], bins=15, color='salmon', alpha=0.7,
        edgecolor='white', label=f'Random (n={N_ITER})')
ax.axvline(np.mean(q_ys[n_init:]), color='blue', linestyle='--', linewidth=2,
           label=f'BO mean={np.mean(q_ys[n_init:]):.3f}')
ax.axvline(np.mean(y_rand[n_init:]), color='red', linestyle='--', linewidth=2,
           label=f'Rand mean={np.mean(y_rand[n_init:]):.3f}')
ax.set_title("Yield Distribution: BO vs Random", fontsize=11)
ax.set_xlabel("Yield"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_bayesian_optimization.png"), dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4 saved: fig4_bayesian_optimization.png")
print("Done.")
