import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path('.')
DATA_PATH = ROOT / 'data' / 'complex_optimization_data.npy'
OUTPUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lasso_objective(A, b, x, lam):
    r = A @ x - b
    return 0.5 * np.dot(r, r) + lam * np.linalg.norm(x, 1)


def grad_smooth(A, b, x):
    return A.T @ (A @ x - b)


def estimate_lipschitz(A):
    smax = np.linalg.norm(A, 2)
    return smax ** 2


def ista(A, b, lam, x0, L, max_iter):
    x = x0.copy()
    hist = {'objective': [], 'grad_map_norm': [], 'nnz': []}
    step = 1.0 / L
    for _ in range(max_iter):
        x_next = soft_threshold(x - step * grad_smooth(A, b, x), lam * step)
        gmap = (x - x_next) / step
        x = x_next
        hist['objective'].append(lasso_objective(A, b, x, lam))
        hist['grad_map_norm'].append(float(np.linalg.norm(gmap)))
        hist['nnz'].append(int(np.count_nonzero(np.abs(x) > 1e-8)))
    return x, hist


def fista(A, b, lam, x0, L, max_iter, restart=False):
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    step = 1.0 / L
    hist = {'objective': [], 'grad_map_norm': [], 'nnz': [], 'restart_count': 0}
    for _ in range(max_iter):
        x_next = soft_threshold(y - step * grad_smooth(A, b, y), lam * step)
        gmap = (y - x_next) / step
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y_next = x_next + ((t - 1.0) / t_next) * (x_next - x)
        if restart and np.dot(y - x_next, x_next - x) > 0:
            t_next = 1.0
            y_next = x_next.copy()
            hist['restart_count'] += 1
        x, y, t = x_next, y_next, t_next
        hist['objective'].append(lasso_objective(A, b, x, lam))
        hist['grad_map_norm'].append(float(np.linalg.norm(gmap)))
        hist['nnz'].append(int(np.count_nonzero(np.abs(x) > 1e-8)))
    return x, hist


def admm_lasso(A, b, lam, x0, rho, max_iter):
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b
    M = AtA + rho * np.eye(n)
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)
    hist = {'objective': [], 'primal_residual': [], 'dual_residual': [], 'nnz': []}
    for _ in range(max_iter):
        rhs = Atb + rho * (z - u)
        x = np.linalg.solve(M, rhs)
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z
        hist['objective'].append(lasso_objective(A, b, z, lam))
        hist['primal_residual'].append(float(np.linalg.norm(x - z)))
        hist['dual_residual'].append(float(np.linalg.norm(rho * (z - z_old))))
        hist['nnz'].append(int(np.count_nonzero(np.abs(z) > 1e-8)))
    return z, hist


def support_metrics(x_est, x_true, tol=1e-6):
    supp_est = np.abs(x_est) > tol
    supp_true = np.abs(x_true) > tol
    tp = int(np.sum(supp_est & supp_true))
    fp = int(np.sum(supp_est & ~supp_true))
    fn = int(np.sum(~supp_est & supp_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
    }


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    np.random.seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(DATA_PATH, allow_pickle=True).item()
    A = data['A'].astype(float)
    b = data['b'].astype(float)
    x_true = data['x_true'].astype(float)
    n = A.shape[1]
    x0 = np.zeros(n)

    # Problem setup
    Atb = A.T @ b
    lam_max = np.max(np.abs(Atb))
    lam = 0.1 * lam_max
    L = estimate_lipschitz(A)
    eigvals = np.linalg.eigvalsh(A.T @ A)
    positive = eigvals[eigvals > 1e-10]
    cond_est = float(positive.max() / positive.min()) if positive.size else float('inf')

    data_summary = {
        'shape_A': list(A.shape),
        'shape_b': list(b.shape),
        'shape_x_true': list(x_true.shape),
        'meta': data.get('meta', ''),
        'lambda_max': float(lam_max),
        'lambda_used': float(lam),
        'lipschitz_estimate': float(L),
        'condition_estimate_positive_spectrum': cond_est,
        'x_true_nnz': int(np.count_nonzero(np.abs(x_true) > 1e-10)),
        'b_norm': float(np.linalg.norm(b)),
        'A_fro_norm': float(np.linalg.norm(A)),
    }
    save_json(data_summary, OUTPUT_DIR / 'data_summary.json')

    # Reference solve via long restarted FISTA
    x_ref, hist_ref = fista(A, b, lam, x0, L, max_iter=1500, restart=True)
    f_star = float(hist_ref['objective'][-1])

    max_iter = 250
    x_ista, hist_ista = ista(A, b, lam, x0, L, max_iter=max_iter)
    x_fista, hist_fista = fista(A, b, lam, x0, L, max_iter=max_iter, restart=False)
    x_rfista, hist_rfista = fista(A, b, lam, x0, L, max_iter=max_iter, restart=True)
    x_admm, hist_admm = admm_lasso(A, b, lam, x0, rho=1.0, max_iter=max_iter)

    methods = {
        'ISTA': (x_ista, hist_ista),
        'FISTA': (x_fista, hist_fista),
        'Restarted_FISTA': (x_rfista, hist_rfista),
        'ADMM': (x_admm, hist_admm),
        'Reference_Restarted_FISTA': (x_ref, hist_ref),
    }

    metrics = {
        'f_star_reference': f_star,
        'lambda_used': float(lam),
        'methods': {}
    }

    traces = {}
    for name, (x_est, hist) in methods.items():
        obj = np.array(hist['objective'], dtype=float)
        gap = np.maximum(obj - f_star, 1e-16)
        entry = {
            'final_objective': float(obj[-1]),
            'final_gap': float(obj[-1] - f_star),
            'l2_error_to_true': float(np.linalg.norm(x_est - x_true)),
            'l2_error_to_reference': float(np.linalg.norm(x_est - x_ref)),
            'nnz': int(np.count_nonzero(np.abs(x_est) > 1e-8)),
            'objective_monotone_violations': int(np.sum(np.diff(obj) > 1e-10)),
        }
        entry.update(support_metrics(x_est, x_true))
        if name == 'ADMM':
            entry['final_primal_residual'] = float(hist['primal_residual'][-1])
            entry['final_dual_residual'] = float(hist['dual_residual'][-1])
        if 'restart_count' in hist:
            entry['restart_count'] = int(hist['restart_count'])
        metrics['methods'][name] = entry
        traces[f'{name}_objective'] = obj
        traces[f'{name}_gap'] = gap
        for key, values in hist.items():
            if isinstance(values, list):
                traces[f'{name}_{key}'] = np.array(values)

    save_json(metrics, OUTPUT_DIR / 'metrics.json')
    np.savez(OUTPUT_DIR / 'traces.npz', **traces)

    # Figure 1: data overview
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(np.log10(np.maximum(np.linalg.svd(A, compute_uv=False), 1e-12)), bins=30, color='steelblue', edgecolor='black')
    axes[0].set_title('log10 singular values of A')
    axes[0].set_xlabel('log10 sigma')
    axes[0].set_ylabel('count')
    axes[1].plot(x_true[:300], color='darkorange', lw=1.5)
    axes[1].set_title('First 300 entries of x_true')
    axes[1].set_xlabel('index')
    axes[1].set_ylabel('value')
    axes[2].hist(A.flatten()[::50], bins=40, color='seagreen', edgecolor='black')
    axes[2].set_title('Sampled entries of A')
    axes[2].set_xlabel('value')
    axes[2].set_ylabel('count')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'data_overview.png', dpi=200)
    plt.close(fig)

    # Figure 2: convergence comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in ['ISTA', 'FISTA', 'Restarted_FISTA', 'ADMM']:
        ax.semilogy(traces[f'{name}_gap'], label=name.replace('_', ' '), lw=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective gap to reference')
    ax.set_title('Convergence comparison on ill-conditioned Lasso')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'convergence_comparison.png', dpi=200)
    plt.close(fig)

    # Figure 3: ADMM residuals
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(traces['ADMM_primal_residual'], label='Primal residual', lw=2)
    ax.semilogy(traces['ADMM_dual_residual'], label='Dual residual', lw=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual norm')
    ax.set_title('ADMM residual convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'admm_residuals.png', dpi=200)
    plt.close(fig)

    # Figure 4: recovery comparison
    names = ['ISTA', 'FISTA', 'Restarted_FISTA', 'ADMM']
    l2_err = [metrics['methods'][n]['l2_error_to_true'] for n in names]
    recall = [metrics['methods'][n]['recall'] for n in names]
    precision = [metrics['methods'][n]['precision'] for n in names]
    x = np.arange(len(names))
    width = 0.25
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(x - width, l2_err, width=width, label='L2 error to x_true', color='slateblue')
    ax1.set_ylabel('L2 error')
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace('_', ' ') for n in names], rotation=15)
    ax2 = ax1.twinx()
    ax2.bar(x, recall, width=width, label='Recall', color='tomato')
    ax2.bar(x + width, precision, width=width, label='Precision', color='goldenrod')
    ax2.set_ylabel('Support metric')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('Solution quality and support recovery')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'recovery_comparison.png', dpi=200)
    plt.close(fig)

    print('Saved outputs to', OUTPUT_DIR)
    print('Saved figures to', IMG_DIR)


if __name__ == '__main__':
    main()
