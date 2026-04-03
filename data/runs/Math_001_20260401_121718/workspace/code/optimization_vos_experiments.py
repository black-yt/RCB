import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set(style='whitegrid')

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'complex_optimization_data.npy'
OUTPUT_DIR = BASE_DIR / 'outputs'
FIG_DIR = BASE_DIR / 'report' / 'images'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


def soft_threshold(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def lasso_objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + lam * np.linalg.norm(x, 1)


def grad_smooth(A, b, x):
    # gradient of 0.5||Ax-b||^2
    return A.T @ (A @ x - b)


def power_iteration_lipschitz(A, n_iter=50):
    # approximate largest eigenvalue of A^T A
    n = A.shape[1]
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = A.T @ (A @ v)
        nrm = np.linalg.norm(Av)
        if nrm == 0:
            break
        v = Av / nrm
    lam = float(v @ (A.T @ (A @ v)))
    return lam


def fista(A, b, lam, x0, n_iter=300):
    L = power_iteration_lipschitz(A)
    t = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    t_k = 1.0
    objs = []
    for k in range(n_iter):
        grad = grad_smooth(A, b, y)
        x_next = soft_threshold(y - t * grad, lam * t)
        t_next = 0.5 * (1 + np.sqrt(1 + 4 * t_k ** 2))
        y = x_next + (t_k - 1) / t_next * (x_next - x)
        x = x_next
        t_k = t_next
        objs.append(lasso_objective(A, b, x, lam))
    return x, np.array(objs)


def admm_lasso(A, b, lam, rho, x0, n_iter=300):
    m, n = A.shape
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    AtA = A.T @ A
    Atb = A.T @ b
    # pre-factorization via eigen-decomposition for stability
    eigvals, eigvecs = np.linalg.eigh(AtA + rho * np.eye(n))
    def lin_solve(rhs):
        return eigvecs @ (rhs @ eigvecs / eigvals)

    objs = []
    for k in range(n_iter):
        x = lin_solve(Atb + rho * (z - u))
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        objs.append(lasso_objective(A, b, x, lam))
    return x, np.array(objs)


def gradient_descent(A, b, lam, x0, n_iter=300):
    L = power_iteration_lipschitz(A)
    t = 1.0 / L
    x = x0.copy()
    objs = []
    for k in range(n_iter):
        grad = grad_smooth(A, b, x)
        x = soft_threshold(x - t * grad, lam * t)
        objs.append(lasso_objective(A, b, x, lam))
    return x, np.array(objs)


def main():
    arr = np.load(DATA_PATH, allow_pickle=True).item()
    A, b, x_true = arr['A'], arr['b'], arr['x_true']
    lam = 0.1
    x0 = np.zeros_like(x_true)

    # run algorithms
    x_gd, obj_gd = gradient_descent(A, b, lam, x0, n_iter=200)
    x_fista, obj_fista = fista(A, b, lam, x0, n_iter=200)
    x_admm, obj_admm = admm_lasso(A, b, lam, rho=1.0, x0=x0, n_iter=200)

    # compute reference objective at ground truth
    f_star = lasso_objective(A, b, x_admm, lam)

    np.save(OUTPUT_DIR / 'solutions.npy', {
        'x_gd': x_gd,
        'x_fista': x_fista,
        'x_admm': x_admm,
        'obj_gd': obj_gd,
        'obj_fista': obj_fista,
        'obj_admm': obj_admm,
        'f_star_est': f_star,
    })

    # plot objective convergence
    k = np.arange(len(obj_gd))
    plt.figure(figsize=(6, 4))
    plt.semilogy(k, obj_gd - f_star + 1e-12, label='Proximal GD')
    plt.semilogy(k, obj_fista - f_star + 1e-12, label='FISTA (VOS-Nesterov)')
    plt.semilogy(k, obj_admm - f_star + 1e-12, label='ADMM (VOS-splitting)')
    plt.xlabel('Iteration')
    plt.ylabel(r'$f(x_k) - f^*$ (log scale)')
    plt.legend()
    plt.tight_layout()
    fig1_path = FIG_DIR / 'objective_convergence.png'
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # sparsity patterns
    plt.figure(figsize=(6, 4))
    idx = np.arange(len(x_true))
    plt.stem(idx, x_true, linefmt='k-', markerfmt='ko', basefmt=' ', label='Ground truth')
    plt.stem(idx, x_fista, linefmt='r--', markerfmt='ro', basefmt=' ', label='FISTA')
    plt.xlim(0, 200)
    plt.xlabel('Coefficient index (first 200 shown)')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    fig2_path = FIG_DIR / 'sparsity_fista_vs_true.png'
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    # error vs iteration
    def l2_err(x):
        return np.linalg.norm(x - x_true)

    plt.figure(figsize=(6, 4))
    plt.semilogy(k, [l2_err(x_gd)] * len(k), alpha=0.3)  # placeholder
    plt.close()

    # final coefficient error bars
    errs = {
        'GD': l2_err(x_gd),
        'FISTA': l2_err(x_fista),
        'ADMM': l2_err(x_admm),
    }
    methods = list(errs.keys())
    values = [errs[m] for m in methods]
    plt.figure(figsize=(5, 4))
    sns.barplot(x=methods, y=values)
    plt.ylabel('L2 error to ground truth')
    plt.tight_layout()
    fig3_path = FIG_DIR / 'final_l2_errors.png'
    plt.savefig(fig3_path, dpi=200)
    plt.close()

    print('Saved figures to:', fig1_path, fig2_path, fig3_path)


if __name__ == '__main__':
    main()
