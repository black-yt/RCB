"""
VOS Framework: Algorithms Implementation
========================================
Implements:
  1. Proximal Gradient Descent (ISTA) - baseline
  2. Nesterov's Accelerated Proximal Gradient (FISTA) - O(1/k^2)
  3. Heavy Ball (Polyak momentum) - constant momentum
  4. ADMM for Lasso - operator splitting
  5. ODE integration of continuous-time system (Su, Boyd, Candès 2015)
  6. Restarted FISTA - linear convergence for strongly convex
  7. VOS Unified Framework (ADMM + Nesterov hybrid)

Problem: min_x (1/2)||Ax - b||^2 + lam * ||x||_1  (Lasso)

VOS Framework principle:
  The continuous-time dynamical system Ẍ + (3/t)Ẋ + ∇f(X) = 0
  is discretized to yield Nesterov's AGM. Under operator splitting
  (variable splitting x=z with penalty rho), we derive ADMM.
  Both share the same Lyapunov function structure, proving convergence.
"""

import numpy as np
from scipy.integrate import solve_ivp


def soft_threshold(x, threshold):
    """Proximal operator of lambda * ||.||_1."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def lasso_objective(x, A, b, lam):
    """Lasso objective: (1/2)||Ax - b||^2 + lam * ||x||_1"""
    r = A @ x - b
    return 0.5 * np.dot(r, r) + lam * np.sum(np.abs(x))


def smooth_grad(x, A, b, AtA, Atb):
    """Gradient of smooth part: A^T(Ax - b) = A^T A x - A^T b"""
    return AtA @ x - Atb


# ── 1. ISTA (Proximal Gradient) ────────────────────────────────────────────
def ista(A, b, lam, L, n_iters=2000, x0=None):
    """
    Iterative Shrinkage-Thresholding Algorithm.
    x_{k+1} = prox_{lam/L}( x_k - (1/L) nabla f(x_k) )
    Convergence: f(x_k) - f* <= O(1/k)
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n) if x0 is None else x0.copy()
    step = 1.0 / L
    objectives = []
    iterates = [x.copy()]

    for k in range(n_iters):
        grad = smooth_grad(x, A, b, AtA, Atb)
        x = soft_threshold(x - step * grad, step * lam)
        objectives.append(lasso_objective(x, A, b, lam))
        if k < 200 or k % 10 == 0:
            iterates.append(x.copy())

    return x, np.array(objectives), iterates


# ── 2. FISTA (Nesterov-accelerated ISTA) ──────────────────────────────────
def fista(A, b, lam, L, n_iters=2000, x0=None):
    """
    Fast ISTA (Beck & Teboulle 2009) — Nesterov's acceleration for composite obj.
    x_{k+1} = prox_{lam/L}( y_k - (1/L) nabla f(y_k) )
    y_{k+1} = x_{k+1} + ((t_k-1)/t_{k+1}) (x_{k+1} - x_k)
    where t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
    Convergence: f(x_k) - f* <= O(1/k^2)
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n) if x0 is None else x0.copy()
    y = x.copy()
    t = 1.0
    step = 1.0 / L
    objectives = []
    iterates = [x.copy()]
    t_vals = [t]

    for k in range(n_iters):
        grad = smooth_grad(y, A, b, AtA, Atb)
        x_new = soft_threshold(y - step * grad, step * lam)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        objectives.append(lasso_objective(x, A, b, lam))
        t_vals.append(t)
        if k < 200 or k % 10 == 0:
            iterates.append(x.copy())

    return x, np.array(objectives), iterates, np.array(t_vals)


# ── 3. Heavy Ball (Polyak 1964) ────────────────────────────────────────────
def heavy_ball(A, b, lam, L, mu, n_iters=2000, x0=None):
    """
    Polyak's Heavy Ball with proximal step.
    x_{k+1} = prox_{lam/L}( x_k - alpha * nabla f(x_k) + beta*(x_k - x_{k-1}) )
    Optimal constants for quadratic: alpha = 4/(sqrt(L) + sqrt(mu))^2
    beta = ((sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)))^2
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n) if x0 is None else x0.copy()
    x_prev = x.copy()
    alpha = 4.0 / (np.sqrt(L) + np.sqrt(mu)) ** 2
    beta = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    objectives = []
    iterates = [x.copy()]

    for k in range(n_iters):
        grad = smooth_grad(x, A, b, AtA, Atb)
        x_new = soft_threshold(x - alpha * grad + beta * (x - x_prev), alpha * lam)
        x_prev = x.copy()
        x = x_new
        objectives.append(lasso_objective(x, A, b, lam))
        if k < 200 or k % 10 == 0:
            iterates.append(x.copy())

    return x, np.array(objectives), iterates


# ── 4. ADMM for Lasso ──────────────────────────────────────────────────────
def admm_lasso(A, b, lam, rho=1.0, n_iters=2000, x0=None):
    """
    ADMM for Lasso via variable splitting: min f(x) + g(z) s.t. x = z
      f(x) = (1/2)||Ax - b||^2,  g(z) = lam * ||z||_1
    Updates:
      x_{k+1} = (A^T A + rho I)^{-1} (A^T b + rho (z_k - u_k))
      z_{k+1} = prox_{lam/rho}(x_{k+1} + u_k)
      u_{k+1} = u_k + x_{k+1} - z_{k+1}
    Penalty parameter rho controls convergence speed.
    """
    m, n = A.shape
    x = np.zeros(n) if x0 is None else x0.copy()
    z = x.copy()
    u = np.zeros(n)

    # Precompute: (A^T A + rho I)^{-1} -- use Cholesky for efficiency
    AtA = A.T @ A
    Atb = A.T @ b
    # Cache factorisation
    M = AtA + rho * np.eye(n)
    # Use Cholesky for PD system
    from scipy.linalg import cho_factor, cho_solve
    c, low = cho_factor(M)

    objectives = []
    primal_residuals = []
    dual_residuals = []
    iterates = [x.copy()]

    for k in range(n_iters):
        # x-update
        rhs = Atb + rho * (z - u)
        x = cho_solve((c, low), rhs)
        # z-update
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        # u-update (scaled dual variable)
        u = u + x - z

        primal_res = np.linalg.norm(x - z)
        dual_res = rho * np.linalg.norm(z - z_old)

        objectives.append(lasso_objective(x, A, b, lam))
        primal_residuals.append(primal_res)
        dual_residuals.append(dual_res)
        if k < 200 or k % 10 == 0:
            iterates.append(x.copy())

    return x, np.array(objectives), iterates, np.array(primal_residuals), np.array(dual_residuals)


# ── 5. Restarted FISTA (linear convergence) ───────────────────────────────
def fista_restarted(A, b, lam, L, mu, restart_freq=None, n_iters=2000, x0=None):
    """
    Restarted FISTA (Su, Boyd, Candès 2015, Section 5).
    When f is mu-strongly convex, restarting FISTA every ~sqrt(L/mu) iterations
    gives linear convergence rate ~exp(-sqrt(mu/L)).
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n) if x0 is None else x0.copy()
    y = x.copy()
    t = 1.0
    step = 1.0 / L
    if restart_freq is None:
        restart_freq = max(10, int(np.sqrt(L / mu)))

    objectives = []
    restart_points = []
    iterates = [x.copy()]

    for k in range(n_iters):
        # Restart momentum periodically
        if k > 0 and k % restart_freq == 0:
            y = x.copy()
            t = 1.0
            restart_points.append(k)

        grad = smooth_grad(y, A, b, AtA, Atb)
        x_new = soft_threshold(y - step * grad, step * lam)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        objectives.append(lasso_objective(x, A, b, lam))
        if k < 200 or k % 10 == 0:
            iterates.append(x.copy())

    return x, np.array(objectives), iterates, restart_points


# ── 6. Continuous-time ODE (Su, Boyd, Candès 2015) ─────────────────────────
def ode_nesterov(f_grad, x0, t_span, n_points=500):
    """
    Solve Nesterov's continuous-time ODE:
      Ẍ + (3/t)Ẋ + ∇f(X) = 0
    Reformulate as first-order system:
      [X]   [          V           ]
      [V] = [ -(3/t)V - ∇f(X)     ]
    Initial conditions: X(0) = x0, V(0) = 0 (vanishing initial velocity)
    """
    dim = len(x0)
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    def rhs(t, state):
        X = state[:dim]
        V = state[dim:]
        gradX = f_grad(X)
        dX = V
        dV = -(3.0 / max(t, 1e-8)) * V - gradX
        return np.concatenate([dX, dV])

    state0 = np.concatenate([x0, np.zeros(dim)])
    sol = solve_ivp(rhs, t_span, state0, t_eval=t_eval,
                    method='RK45', rtol=1e-6, atol=1e-9, max_step=0.1)
    X_traj = sol.y[:dim, :]  # shape (dim, n_points)
    V_traj = sol.y[dim:, :]
    return sol.t, X_traj, V_traj


# ── 7. Lyapunov function computation ──────────────────────────────────────
def lyapunov_continuous(t_vals, X_traj, V_traj, f_vals, f_star, x_star):
    """
    Compute the Lyapunov function from Su, Boyd, Candès (2015):
      E(t) = t^2 * (f(X(t)) - f*) + 2 * ||X(t) + (t/2) * V(t) - x*||^2
    This is monotonically non-increasing along trajectories.
    """
    E_vals = []
    for i, t in enumerate(t_vals):
        X = X_traj[:, i]
        V = V_traj[:, i]
        term1 = t ** 2 * (f_vals[i] - f_star)
        term2 = 2.0 * np.linalg.norm(X + (t / 2.0) * V - x_star) ** 2
        E_vals.append(term1 + term2)
    return np.array(E_vals)


def lyapunov_discrete(k_vals, x_vals, t_vals, f_vals, f_star, x_star):
    """
    Discrete analog of the Lyapunov function for FISTA:
      E_k = t_k^2 * (f(x_k) - f*) + (1/2) * ||x_k - x* + t_k*(x_k - x_{k-1})||^2
    """
    E_vals = []
    for i in range(1, len(x_vals)):
        k = k_vals[i]
        x = x_vals[i]
        x_prev = x_vals[i - 1]
        t = t_vals[i]
        term1 = t ** 2 * max(f_vals[i] - f_star, 0.0)
        term2 = 0.5 * np.linalg.norm(x - x_star + t * (x - x_prev)) ** 2
        E_vals.append(term1 + term2)
    return np.array(E_vals)


if __name__ == '__main__':
    print("Algorithm module loaded successfully.")
    print("Available algorithms: ISTA, FISTA, Heavy Ball, ADMM, Restarted FISTA, ODE")
