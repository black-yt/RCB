#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "complex_optimization_data.npy"
OUTPUTS_DIR = ROOT / "outputs"
IMAGES_DIR = ROOT / "report" / "images"


def soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def lasso_objective(A, b, x, lam):
    n = A.shape[0]
    r = A @ x - b
    return 0.5 * np.dot(r, r) / n + lam * np.sum(np.abs(x))


def smooth_value(A, b, x):
    n = A.shape[0]
    r = A @ x - b
    return 0.5 * np.dot(r, r) / n


def grad_smooth(A, b, x):
    n = A.shape[0]
    return A.T @ (A @ x - b) / n


def power_iteration_lipschitz(A, n_iter=200, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        Av = A @ v
        AtAv = A.T @ Av / A.shape[0]
        norm = np.linalg.norm(AtAv)
        if norm == 0:
            return 0.0
        v = AtAv / norm
    Av = A @ v
    return np.dot(Av, Av) / A.shape[0]


def ista(A, b, lam, L, x0, max_iter=400, tol=1e-10, x_ref=None, f_ref=None):
    step = 1.0 / L
    x = x0.copy()
    hist = {
        "objective": [],
        "objective_gap": [],
        "grad_map_norm": [],
        "distance_to_ref": [],
        "sparsity": [],
    }
    for _ in range(max_iter):
        grad = grad_smooth(A, b, x)
        x_next = soft_threshold(x - step * grad, lam * step)
        obj = lasso_objective(A, b, x_next, lam)
        gmap = (x - x_next) / step
        hist["objective"].append(obj)
        hist["objective_gap"].append(max(obj - f_ref, 0.0) if f_ref is not None else np.nan)
        hist["grad_map_norm"].append(np.linalg.norm(gmap))
        hist["distance_to_ref"].append(np.linalg.norm(x_next - x_ref) if x_ref is not None else np.nan)
        hist["sparsity"].append(int(np.sum(np.abs(x_next) > 1e-8)))
        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break
        x = x_next
    return x, {k: np.asarray(v) for k, v in hist.items()}


def fista(A, b, lam, L, x0, max_iter=400, tol=1e-10, x_ref=None, f_ref=None):
    step = 1.0 / L
    x = x0.copy()
    y = x0.copy()
    t = 1.0
    hist = {
        "objective": [],
        "objective_gap": [],
        "grad_map_norm": [],
        "distance_to_ref": [],
        "sparsity": [],
        "lyapunov": [],
    }
    for k in range(1, max_iter + 1):
        grad = grad_smooth(A, b, y)
        x_next = soft_threshold(y - step * grad, lam * step)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        obj = lasso_objective(A, b, x_next, lam)
        gmap = (y - soft_threshold(y - step * grad_smooth(A, b, y), lam * step)) / step
        gap = max(obj - f_ref, 0.0) if f_ref is not None else np.nan
        dist = np.linalg.norm(x_next - x_ref) if x_ref is not None else np.nan
        lyap = (k + 1) ** 2 * gap + 0.5 * L * (dist ** 2) if x_ref is not None and f_ref is not None else np.nan
        hist["objective"].append(obj)
        hist["objective_gap"].append(gap)
        hist["grad_map_norm"].append(np.linalg.norm(gmap))
        hist["distance_to_ref"].append(dist)
        hist["sparsity"].append(int(np.sum(np.abs(x_next) > 1e-8)))
        hist["lyapunov"].append(lyap)
        if np.linalg.norm(x_next - x) <= tol * max(1.0, np.linalg.norm(x)):
            x = x_next
            break
        x = x_next
        t = t_next
    return x, {k: np.asarray(v) for k, v in hist.items()}


def admm_lasso(A, b, lam, rho, x0, max_iter=400, tol=1e-8, x_ref=None, f_ref=None):
    n, p = A.shape
    x = x0.copy()
    z = x0.copy()
    u = np.zeros_like(x0)

    H = (A.T @ A) / n + rho * np.eye(p)
    rhs_base = A.T @ b / n
    Lchol = np.linalg.cholesky(H)

    def solve_linear(rhs):
        y = np.linalg.solve(Lchol, rhs)
        return np.linalg.solve(Lchol.T, y)

    hist = {
        "objective": [],
        "objective_gap": [],
        "primal_residual": [],
        "dual_residual": [],
        "distance_to_ref": [],
        "sparsity": [],
        "lyapunov": [],
    }

    for _ in range(max_iter):
        x = solve_linear(rhs_base + rho * (z - u))
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        u = u + x - z

        obj = lasso_objective(A, b, z, lam)
        r_norm = np.linalg.norm(x - z)
        s_norm = rho * np.linalg.norm(z - z_old)
        gap = max(obj - f_ref, 0.0) if f_ref is not None else np.nan
        dist = np.linalg.norm(z - x_ref) if x_ref is not None else np.nan
        lyap = gap + rho * r_norm ** 2 + 0.5 * rho * np.linalg.norm(z - z_old) ** 2

        hist["objective"].append(obj)
        hist["objective_gap"].append(gap)
        hist["primal_residual"].append(r_norm)
        hist["dual_residual"].append(s_norm)
        hist["distance_to_ref"].append(dist)
        hist["sparsity"].append(int(np.sum(np.abs(z) > 1e-8)))
        hist["lyapunov"].append(lyap)

        eps_pri = np.sqrt(p) * tol + tol * max(np.linalg.norm(x), np.linalg.norm(z))
        eps_dual = np.sqrt(p) * tol + tol * rho * np.linalg.norm(u)
        if r_norm <= eps_pri and s_norm <= eps_dual:
            break

    return z, {k: np.asarray(v) for k, v in hist.items()}


def support_metrics(x_hat, x_true, threshold=1e-6):
    est = np.abs(x_hat) > threshold
    tru = np.abs(x_true) > threshold
    tp = int(np.sum(est & tru))
    fp = int(np.sum(est & ~tru))
    fn = int(np.sum(~est & tru))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_size": int(np.sum(est)),
    }


def save_data_overview_figures(A, b, x_true, singular_values):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].plot(np.sort(singular_values)[::-1], lw=2)
    axes[0].set_title("Singular value spectrum of A")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular value")

    true_idx = np.flatnonzero(np.abs(x_true) > 1e-12)
    axes[1].stem(true_idx, x_true[true_idx], basefmt=" ", linefmt="C1-", markerfmt="C1o")
    axes[1].set_title("Ground-truth sparse coefficients")
    axes[1].set_xlabel("Coefficient index")
    axes[1].set_ylabel("Value")

    axes[2].hist(b, bins=30, color="C2", alpha=0.85, edgecolor="black")
    axes[2].set_title("Response distribution")
    axes[2].set_xlabel("b")
    axes[2].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "data_overview.png", dpi=200)
    plt.close(fig)


def save_convergence_figure(histories):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for name, hist in histories.items():
        it = np.arange(1, len(hist["objective_gap"]) + 1)
        axes[0, 0].semilogy(it, np.maximum(hist["objective_gap"], 1e-16), label=name, lw=2)
        axes[0, 1].semilogy(it, np.maximum(hist["distance_to_ref"], 1e-16), label=name, lw=2)
        axes[1, 0].plot(it, hist["sparsity"], label=name, lw=2)
        if "lyapunov" in hist:
            axes[1, 1].semilogy(it, np.maximum(hist["lyapunov"], 1e-16), label=name, lw=2)

    axes[0, 0].set_title("Objective gap vs. reference solution")
    axes[0, 1].set_title("Distance to reference solution")
    axes[1, 0].set_title("Active-set size across iterations")
    axes[1, 1].set_title("Task-specific Lyapunov / energy surrogate")

    axes[0, 0].set_xlabel("Iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 1].set_xlabel("Iteration")

    axes[0, 0].set_ylabel("Gap")
    axes[0, 1].set_ylabel("Norm")
    axes[1, 0].set_ylabel("Nonzeros")
    axes[1, 1].set_ylabel("Energy")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "convergence_comparison.png", dpi=200)
    plt.close(fig)


def save_recovery_figure(x_true, solutions):
    fig, axes = plt.subplots(1, len(solutions), figsize=(5 * len(solutions), 4.5), sharex=True, sharey=True)
    if len(solutions) == 1:
        axes = [axes]

    lim = max(np.max(np.abs(x_true)), max(np.max(np.abs(x)) for x in solutions.values())) * 1.05
    for ax, (name, x_hat) in zip(axes, solutions.items()):
        ax.scatter(x_true, x_hat, s=10, alpha=0.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set_title(name)
        ax.set_xlabel("True coefficient")
        ax.set_ylabel("Estimated coefficient")
        ax.grid(alpha=0.3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "coefficient_recovery.png", dpi=200)
    plt.close(fig)


def save_top_coefficients_figure(x_true, solutions, top_k=40):
    idx = np.argsort(np.abs(x_true))[::-1][:top_k]
    order = np.arange(top_k)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(order, x_true[idx], label="True", lw=3, marker="o")
    for name, x_hat in solutions.items():
        ax.plot(order, x_hat[idx], label=name, lw=2, marker=".")
    ax.set_title(f"Top-{top_k} largest true coefficients")
    ax.set_xlabel("Ranked support position")
    ax.set_ylabel("Coefficient value")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "top_coefficients.png", dpi=200)
    plt.close(fig)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    payload = np.load(DATA_PATH, allow_pickle=True).item()
    A = np.asarray(payload["A"], dtype=float)
    b = np.asarray(payload["b"], dtype=float)
    x_true = np.asarray(payload["x_true"], dtype=float)
    meta = payload.get("meta", "")
    n, p = A.shape

    singular_values = np.linalg.svd(A, full_matrices=False, compute_uv=False)
    cond_number = float(singular_values[0] / singular_values[-1])
    L = float((singular_values[0] ** 2) / n)
    L_power = float(power_iteration_lipschitz(A))

    lambda_max = float(np.max(np.abs(A.T @ b)) / n)
    lam = 0.08 * lambda_max

    x0 = np.zeros(p)

    x_ref, ref_hist = fista(A, b, lam, L, x0, max_iter=5000, tol=1e-12)
    f_ref = float(lasso_objective(A, b, x_ref, lam))

    x_ista, hist_ista = ista(A, b, lam, L, x0, max_iter=500, tol=1e-12, x_ref=x_ref, f_ref=f_ref)
    x_fista, hist_fista = fista(A, b, lam, L, x0, max_iter=500, tol=1e-12, x_ref=x_ref, f_ref=f_ref)
    x_admm, hist_admm = admm_lasso(A, b, lam, rho=L / 5.0, x0=x0, max_iter=500, tol=1e-8, x_ref=x_ref, f_ref=f_ref)

    histories = {"ISTA": hist_ista, "FISTA": hist_fista, "ADMM": hist_admm}
    solutions = {"ISTA": x_ista, "FISTA": x_fista, "ADMM": x_admm, "Reference FISTA": x_ref}

    save_data_overview_figures(A, b, x_true, singular_values)
    save_convergence_figure(histories)
    save_recovery_figure(x_true, solutions)
    save_top_coefficients_figure(x_true, solutions)

    np.savez(
        OUTPUTS_DIR / "analysis_histories.npz",
        ista_objective=hist_ista["objective"],
        ista_gap=hist_ista["objective_gap"],
        ista_distance=hist_ista["distance_to_ref"],
        fista_objective=hist_fista["objective"],
        fista_gap=hist_fista["objective_gap"],
        fista_distance=hist_fista["distance_to_ref"],
        fista_lyapunov=hist_fista["lyapunov"],
        admm_objective=hist_admm["objective"],
        admm_gap=hist_admm["objective_gap"],
        admm_distance=hist_admm["distance_to_ref"],
        admm_lyapunov=hist_admm["lyapunov"],
    )
    np.savez(
        OUTPUTS_DIR / "solutions.npz",
        x_true=x_true,
        x_ista=x_ista,
        x_fista=x_fista,
        x_admm=x_admm,
        x_ref=x_ref,
    )

    metrics = {
        "dataset": {
            "meta": str(meta),
            "n_samples": int(n),
            "n_features": int(p),
            "condition_number": cond_number,
            "lipschitz_exact": L,
            "lipschitz_power_iteration": L_power,
            "true_support_size": int(np.sum(np.abs(x_true) > 1e-12)),
        },
        "regularization": {
            "lambda_max": lambda_max,
            "lambda_used": lam,
        },
        "reference_solution": {
            "objective": f_ref,
            "support_metrics_vs_true": support_metrics(x_ref, x_true),
            "relative_error_vs_true": float(np.linalg.norm(x_ref - x_true) / max(np.linalg.norm(x_true), 1e-12)),
        },
        "algorithms": {},
        "generated_files": {
            "outputs": [
                "outputs/analysis_histories.npz",
                "outputs/solutions.npz",
                "outputs/analysis_summary.json",
                "outputs/analysis_summary.txt",
            ],
            "figures": [
                "report/images/data_overview.png",
                "report/images/convergence_comparison.png",
                "report/images/coefficient_recovery.png",
                "report/images/top_coefficients.png",
            ],
        },
    }

    for name, x_hat, hist in [
        ("ISTA", x_ista, hist_ista),
        ("FISTA", x_fista, hist_fista),
        ("ADMM", x_admm, hist_admm),
    ]:
        metrics["algorithms"][name] = {
            "iterations": int(len(hist["objective"])),
            "final_objective": float(hist["objective"][-1]),
            "final_objective_gap": float(hist["objective_gap"][-1]),
            "distance_to_reference": float(np.linalg.norm(x_hat - x_ref)),
            "relative_error_vs_true": float(np.linalg.norm(x_hat - x_true) / max(np.linalg.norm(x_true), 1e-12)),
            "support_metrics_vs_true": support_metrics(x_hat, x_true),
        }
        if name == "ADMM":
            metrics["algorithms"][name]["final_primal_residual"] = float(hist["primal_residual"][-1])
            metrics["algorithms"][name]["final_dual_residual"] = float(hist["dual_residual"][-1])
        else:
            metrics["algorithms"][name]["final_grad_map_norm"] = float(hist["grad_map_norm"][-1])

    with open(OUTPUTS_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "Unified VOS-style Lasso analysis summary",
        "=",
        f"Dataset: {meta}",
        f"Shape: n={n}, p={p}",
        f"Condition number (A): {cond_number:.4f}",
        f"Smooth Lipschitz constant L: {L:.6f}",
        f"Regularization lambda: {lam:.6e} ({lam / lambda_max:.2%} of lambda_max)",
        "",
        "Method comparison:",
    ]
    for name in ["ISTA", "FISTA", "ADMM"]:
        info = metrics["algorithms"][name]
        lines.append(
            f"- {name}: iterations={info['iterations']}, final objective={info['final_objective']:.6e}, "
            f"gap={info['final_objective_gap']:.6e}, rel_err_true={info['relative_error_vs_true']:.6e}, "
            f"support_f1={info['support_metrics_vs_true']['f1']:.4f}"
        )
    lines.extend([
        "",
        "Generated figures:",
        "- report/images/data_overview.png",
        "- report/images/convergence_comparison.png",
        "- report/images/coefficient_recovery.png",
        "- report/images/top_coefficients.png",
    ])
    (OUTPUTS_DIR / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
