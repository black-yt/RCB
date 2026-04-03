import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import linalg
from sklearn.linear_model import ElasticNet


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "complex_optimization_data.npy"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMAGE_DIR = ROOT / "report" / "images"
MPLCONFIGDIR = ROOT / "outputs" / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ElasticNetProblem:
    A: np.ndarray
    b: np.ndarray
    x_true: np.ndarray
    lambda1: float
    mu: float

    def __post_init__(self) -> None:
        self.n, self.p = self.A.shape
        self.Atb_over_n = self.A.T @ self.b / self.n
        self.gram = self.A.T @ self.A / self.n
        self.hessian = self.gram + self.mu * np.eye(self.p)
        singular_values = np.linalg.svd(self.A, compute_uv=False)
        self.singular_values = singular_values
        self.L = singular_values[0] ** 2 / self.n + self.mu
        self.sigma_min_nonzero = singular_values[self.n - 1]

    def grad_smooth(self, x: np.ndarray) -> np.ndarray:
        return self.hessian @ x - self.Atb_over_n

    def smooth_value(self, x: np.ndarray) -> float:
        residual = self.A @ x - self.b
        return 0.5 * np.dot(residual, residual) / self.n + 0.5 * self.mu * np.dot(x, x)

    def objective(self, x: np.ndarray) -> float:
        return self.smooth_value(x) + self.lambda1 * np.linalg.norm(x, 1)


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def ensure_directories() -> None:
    for directory in [OUTPUT_DIR, REPORT_IMAGE_DIR, MPLCONFIGDIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_problem(lambda1: float = 8e-3, mu: float = 1e-2) -> ElasticNetProblem:
    payload = np.load(DATA_PATH, allow_pickle=True).item()
    return ElasticNetProblem(
        A=payload["A"].astype(np.float64),
        b=payload["b"].astype(np.float64),
        x_true=payload["x_true"].astype(np.float64),
        lambda1=lambda1,
        mu=mu,
    )


def compute_support_metrics(x: np.ndarray, x_true: np.ndarray, tol: float = 1e-6) -> dict:
    support = np.abs(x) > tol
    support_true = np.abs(x_true) > tol
    tp = int(np.sum(support & support_true))
    fp = int(np.sum(support & ~support_true))
    fn = int(np.sum(~support & support_true))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "nnz": int(np.sum(support)),
        "true_nnz": int(np.sum(support_true)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def reference_solution(problem: ElasticNetProblem) -> tuple[np.ndarray, dict]:
    alpha = problem.lambda1 + problem.mu
    l1_ratio = problem.lambda1 / alpha
    model = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        fit_intercept=False,
        max_iter=50000,
        tol=1e-10,
        selection="cyclic",
    )
    start = time.perf_counter()
    model.fit(problem.A, problem.b)
    elapsed = time.perf_counter() - start
    x_star = model.coef_.astype(np.float64)
    grad = problem.grad_smooth(x_star)
    kkt_violation = np.max(
        np.maximum(
            np.abs(grad) - problem.lambda1,
            np.where(np.abs(x_star) > 1e-8, np.abs(grad + problem.lambda1 * np.sign(x_star)), 0.0),
        )
    )
    info = {
        "solver": "sklearn.coordinate_descent",
        "runtime_sec": elapsed,
        "objective": float(problem.objective(x_star)),
        "kkt_violation": float(kkt_violation),
        "support": compute_support_metrics(x_star, problem.x_true),
    }
    return x_star, info


def ista(
    problem: ElasticNetProblem,
    x_star: np.ndarray,
    max_iter: int = 1200,
) -> dict:
    x = np.zeros(problem.p)
    step = 1.0 / problem.L
    obj_star = problem.objective(x_star)
    history = {
        "objective": [],
        "gap": [],
        "rel_error": [],
        "lyapunov": [],
        "time_sec": [],
    }
    start = time.perf_counter()
    for _ in range(max_iter):
        x = soft_threshold(x - step * problem.grad_smooth(x), step * problem.lambda1)
        obj = problem.objective(x)
        history["objective"].append(obj)
        history["gap"].append(max(obj - obj_star, 1e-16))
        history["rel_error"].append(np.linalg.norm(x - x_star) / max(np.linalg.norm(x_star), 1e-12))
        history["lyapunov"].append(max(obj - obj_star + 0.5 * problem.mu * np.dot(x - x_star, x - x_star), 1e-16))
        history["time_sec"].append(time.perf_counter() - start)
    history["x_final"] = x
    history["beta"] = 0.0
    return history


def accelerated_proximal_gradient(
    problem: ElasticNetProblem,
    x_star: np.ndarray,
    max_iter: int = 300,
) -> dict:
    x_prev = np.zeros(problem.p)
    x = np.zeros(problem.p)
    y = np.zeros(problem.p)
    step = 1.0 / problem.L
    beta = (math.sqrt(problem.L) - math.sqrt(problem.mu)) / (math.sqrt(problem.L) + math.sqrt(problem.mu))
    obj_star = problem.objective(x_star)
    history = {
        "objective": [],
        "gap": [],
        "rel_error": [],
        "lyapunov": [],
        "time_sec": [],
    }
    start = time.perf_counter()
    for _ in range(max_iter):
        x_next = soft_threshold(y - step * problem.grad_smooth(y), step * problem.lambda1)
        momentum_state = x_next + beta / max(1.0 - beta, 1e-12) * (x_next - x)
        obj = problem.objective(x_next)
        history["objective"].append(obj)
        history["gap"].append(max(obj - obj_star, 1e-16))
        history["rel_error"].append(np.linalg.norm(x_next - x_star) / max(np.linalg.norm(x_star), 1e-12))
        history["lyapunov"].append(
            max(obj - obj_star + 0.5 * problem.mu * np.dot(momentum_state - x_star, momentum_state - x_star), 1e-16)
        )
        history["time_sec"].append(time.perf_counter() - start)
        x_prev, x = x, x_next
        y = x + beta * (x - x_prev)
    history["x_final"] = x
    history["beta"] = beta
    return history


def admm(
    problem: ElasticNetProblem,
    x_star: np.ndarray,
    rho: float = 0.08,
    max_iter: int = 400,
) -> dict:
    x = np.zeros(problem.p)
    z = np.zeros(problem.p)
    u = np.zeros(problem.p)
    y_star = -problem.grad_smooth(x_star)
    obj_star = problem.objective(x_star)
    system_matrix = problem.hessian + rho * np.eye(problem.p)
    chol_factor = linalg.cho_factor(system_matrix, check_finite=False)
    history = {
        "objective": [],
        "gap": [],
        "rel_error": [],
        "lyapunov": [],
        "time_sec": [],
        "primal_residual": [],
        "dual_residual": [],
    }
    start = time.perf_counter()
    for _ in range(max_iter):
        rhs = problem.Atb_over_n + rho * (z - u)
        x = linalg.cho_solve(chol_factor, rhs, check_finite=False)
        z_prev = z.copy()
        z = soft_threshold(x + u, problem.lambda1 / rho)
        u = u + x - z
        dual = rho * u
        obj = problem.objective(z)
        primal_residual = np.linalg.norm(x - z)
        dual_residual = rho * np.linalg.norm(z - z_prev)
        lyapunov = obj - obj_star + 0.5 * rho * np.dot(z - x_star, z - x_star) + 0.5 / rho * np.dot(dual - y_star, dual - y_star)
        history["objective"].append(obj)
        history["gap"].append(max(obj - obj_star, 1e-16))
        history["rel_error"].append(np.linalg.norm(z - x_star) / max(np.linalg.norm(x_star), 1e-12))
        history["lyapunov"].append(max(lyapunov, 1e-16))
        history["time_sec"].append(time.perf_counter() - start)
        history["primal_residual"].append(primal_residual)
        history["dual_residual"].append(dual_residual)
    history["x_final"] = z
    history["rho"] = rho
    return history


def contraction_ratio(sequence: list[float], burn_in: int = 5, floor: float = 1e-12) -> float:
    values = np.asarray(sequence, dtype=float)
    if values.size <= burn_in + 1:
        return float("nan")
    prev = values[burn_in:-1]
    curr = values[burn_in + 1 :]
    mask = (prev > floor) & (curr > floor)
    ratios = curr[mask] / prev[mask]
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        return float("nan")
    return float(np.median(ratios))


def first_below(sequence: list[float], threshold: float) -> int | None:
    values = np.asarray(sequence, dtype=float)
    hits = np.where(values < threshold)[0]
    if hits.size == 0:
        return None
    return int(hits[0] + 1)


def save_histories(results: dict) -> None:
    payload = {}
    for name, result in results.items():
        for key, value in result.items():
            if isinstance(value, list):
                payload[f"{name}_{key}"] = np.asarray(value)
            elif isinstance(value, np.ndarray):
                payload[f"{name}_{key}"] = value
    np.savez_compressed(OUTPUT_DIR / "solver_histories.npz", **payload)


def plot_data_overview(problem: ElasticNetProblem) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(problem.singular_values, lw=2.5, color="#0f766e")
    axes[0].set_title("Singular Values of A")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular value")

    active = np.flatnonzero(np.abs(problem.x_true) > 1e-10)
    inactive = np.setdiff1d(np.arange(problem.p), active)
    axes[1].stem(active, problem.x_true[active], linefmt="#b91c1c", markerfmt=" ", basefmt=" ")
    axes[1].scatter(inactive[: min(400, inactive.size)], np.zeros(min(400, inactive.size)), s=8, alpha=0.2, color="#94a3b8")
    axes[1].set_title("Ground-Truth Sparse Coefficients")
    axes[1].set_xlabel("Coordinate")
    axes[1].set_ylabel("Coefficient value")

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "data_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"ista": "#475569", "apg": "#2563eb", "admm": "#dc2626"}

    for name, result in results.items():
        label = name.upper()
        axes[0].semilogy(result["gap"], lw=2.2, label=label, color=palette[name])
        axes[1].semilogy(result["time_sec"], result["gap"], lw=2.2, label=label, color=palette[name])

    axes[0].set_title("Objective Gap vs Iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$F(x_k) - F(x^*)$")
    axes[0].legend()

    axes[1].set_title("Objective Gap vs Runtime")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel(r"$F(x_k) - F(x^*)$")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "convergence_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lyapunov(results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"apg": "#2563eb", "admm": "#dc2626"}

    for name in ["apg", "admm"]:
        result = results[name]
        axes[0].semilogy(result["lyapunov"], lw=2.2, label=name.upper(), color=palette[name])
        lyapunov = np.asarray(result["lyapunov"])
        ratios = lyapunov[1:] / lyapunov[:-1]
        valid = (lyapunov[:-1] > 1e-12) & (lyapunov[1:] > 1e-12) & np.isfinite(ratios)
        axes[1].plot(np.flatnonzero(valid), ratios[valid], lw=1.8, label=name.upper(), color=palette[name])

    axes[0].set_title("Lyapunov Energy Decay")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Energy")
    axes[0].legend()

    axes[1].set_title("Successive Energy Ratios")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$E_{k+1}/E_k$")
    axes[1].axhline(1.0, color="black", lw=1, ls="--")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "lyapunov_decay.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_validation(problem: ElasticNetProblem, x_star: np.ndarray, results: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    apg_x = results["apg"]["x_final"]
    axes[0].scatter(problem.x_true, x_star, s=20, alpha=0.55, color="#0f766e")
    lim = max(np.max(np.abs(problem.x_true)), np.max(np.abs(x_star))) * 1.05
    axes[0].plot([-lim, lim], [-lim, lim], color="black", ls="--", lw=1)
    axes[0].set_title("Recovered vs Ground-Truth Coefficients")
    axes[0].set_xlabel("Ground truth")
    axes[0].set_ylabel("Reference optimum")

    coords = np.arange(problem.p)
    active = np.abs(problem.x_true) > 1e-10
    axes[1].plot(coords[active], problem.x_true[active], "o", ms=4, label="Truth", color="#0f766e")
    axes[1].plot(coords[active], apg_x[active], "x", ms=4, label="APG", color="#2563eb")
    axes[1].plot(coords[active], results["admm"]["x_final"][active], "+", ms=4, label="ADMM", color="#dc2626")
    axes[1].set_title("Active-Set Coefficient Recovery")
    axes[1].set_xlabel("Active coordinate")
    axes[1].set_ylabel("Coefficient value")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "solution_validation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_admm_residuals(result: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(result["primal_residual"], lw=2.2, label="Primal residual", color="#dc2626")
    ax.semilogy(result["dual_residual"], lw=2.2, label="Dual residual", color="#7c3aed")
    ax.set_title("ADMM Residual Decay")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual norm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "admm_residuals.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize_results(problem: ElasticNetProblem, x_star: np.ndarray, reference_info: dict, results: dict) -> dict:
    summary = {
        "problem": {
            "n": problem.n,
            "p": problem.p,
            "lambda1": problem.lambda1,
            "mu": problem.mu,
            "L": problem.L,
            "condition_number_strong": problem.L / problem.mu,
            "sigma_max": float(problem.singular_values[0]),
            "sigma_min_nonzero": float(problem.sigma_min_nonzero),
            "lambda_max_lasso": float(np.max(np.abs(problem.A.T @ problem.b)) / problem.n),
        },
        "reference": reference_info,
        "algorithms": {},
    }

    for name, result in results.items():
        x_final = result["x_final"]
        algorithm_summary = {
            "iterations": len(result["objective"]),
            "runtime_sec": float(result["time_sec"][-1]),
            "final_objective": float(result["objective"][-1]),
            "final_gap": float(result["gap"][-1]),
            "relative_error": float(result["rel_error"][-1]),
            "iterations_to_gap_1e_8": first_below(result["gap"], 1e-8),
            "iterations_to_gap_1e_12": first_below(result["gap"], 1e-12),
            "support": compute_support_metrics(x_final, problem.x_true),
            "lyapunov_ratio_median": contraction_ratio(result["lyapunov"]),
            "objective_ratio_median": contraction_ratio(result["gap"]),
        }
        if name == "apg":
            algorithm_summary["beta"] = float(result["beta"])
            algorithm_summary["theory_rate"] = float(1.0 - math.sqrt(problem.mu / problem.L))
        if name == "admm":
            algorithm_summary["rho"] = float(result["rho"])
            algorithm_summary["final_primal_residual"] = float(result["primal_residual"][-1])
            algorithm_summary["final_dual_residual"] = float(result["dual_residual"][-1])
        summary["algorithms"][name] = algorithm_summary

    summary["solution_error_vs_truth"] = {
        "mse": float(np.mean((x_star - problem.x_true) ** 2)),
        "relative_l2": float(np.linalg.norm(x_star - problem.x_true) / max(np.linalg.norm(problem.x_true), 1e-12)),
    }
    return summary


def main() -> None:
    ensure_directories()
    sns.set_theme(style="whitegrid", context="talk")
    np.random.seed(0)

    problem = load_problem(lambda1=8e-3, mu=1e-2)
    x_star, reference_info = reference_solution(problem)

    results = {
        "ista": ista(problem, x_star),
        "apg": accelerated_proximal_gradient(problem, x_star),
        "admm": admm(problem, x_star, rho=8e-2),
    }

    plot_data_overview(problem)
    plot_convergence(results)
    plot_lyapunov(results)
    plot_validation(problem, x_star, results)
    plot_admm_residuals(results["admm"])

    summary = summarize_results(problem, x_star, reference_info, results)

    np.save(OUTPUT_DIR / "optimal_solution.npy", x_star)
    np.save(OUTPUT_DIR / "ground_truth.npy", problem.x_true)
    save_histories(results)
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
