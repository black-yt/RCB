from __future__ import annotations

import ast
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Circle
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "M-AI-Synth__Materials_AI_Dataset_.txt"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMAGE_DIR = ROOT / "report" / "images"


@dataclass
class BenchmarkData:
    property_node_tokens: np.ndarray
    property_grid: np.ndarray
    property_edges: np.ndarray
    property_targets: np.ndarray
    structure_x: np.ndarray
    structure_y: np.ndarray
    temperature_bounds: tuple[float, float]
    time_bounds: tuple[float, float]
    initial_temperature: float
    initial_time: float
    learning_rate: float
    optimization_steps: int


class LagFeatureBaseline(BaseEstimator, RegressorMixin):
    def __init__(self, lag_feature: str) -> None:
        self.lag_feature = lag_feature

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_feature].to_numpy()


def ensure_directories() -> None:
    for path in [OUTPUT_DIR, REPORT_IMAGE_DIR, OUTPUT_DIR / "tables"]:
        path.mkdir(parents=True, exist_ok=True)


def load_benchmark_data() -> BenchmarkData:
    arrays = []
    for line in DATA_PATH.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            arrays.append(ast.literal_eval(stripped))

    property_grid = np.array(arrays[1], dtype=float).reshape(13, 9)
    edge_pairs = np.array(arrays[2], dtype=int).reshape(-1, 2)
    return BenchmarkData(
        property_node_tokens=np.array(arrays[0], dtype=float),
        property_grid=property_grid,
        property_edges=edge_pairs,
        property_targets=np.array(arrays[3], dtype=float),
        structure_x=np.array(arrays[4], dtype=float),
        structure_y=np.array(arrays[5], dtype=float),
        temperature_bounds=(float(arrays[6][0]), float(arrays[6][1])),
        time_bounds=(float(arrays[7][0]), float(arrays[7][1])),
        initial_temperature=float(arrays[8][0]),
        initial_time=float(arrays[9][0]),
        learning_rate=float(arrays[10][0]),
        optimization_steps=int(arrays[11][0]),
    )


def rolling_feature(values: np.ndarray, radius: int, reducer: str) -> np.ndarray:
    out = np.zeros_like(values)
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        window = values[lo:hi]
        if reducer == "mean":
            out[idx] = window.mean()
        elif reducer == "std":
            out[idx] = window.std()
        else:
            raise ValueError(reducer)
    return out


def build_property_table(data: BenchmarkData) -> pd.DataFrame:
    flat_grid = data.property_grid.ravel()
    n_samples = len(data.property_targets)
    rows = np.repeat(np.arange(data.property_grid.shape[0]), data.property_grid.shape[1])[:n_samples]
    cols = np.tile(np.arange(data.property_grid.shape[1]), data.property_grid.shape[0])[:n_samples]
    grid_values = flat_grid[:n_samples]

    graph_n_nodes = len(np.unique(data.property_edges))
    graph_n_edges = len(data.property_edges)
    graph_density = 2 * graph_n_edges / (graph_n_nodes * (graph_n_nodes - 1))
    mean_degree = 2 * graph_n_edges / graph_n_nodes

    df = pd.DataFrame(
        {
            "sample_id": np.arange(n_samples),
            "grid_row": rows,
            "grid_col": cols,
            "spectral_signal": grid_values,
            "spectral_signal_sq": grid_values**2,
            "spectral_signal_cu": grid_values**3,
            "rolling_mean_2": rolling_feature(grid_values, radius=2, reducer="mean"),
            "rolling_std_2": rolling_feature(grid_values, radius=2, reducer="std"),
            "sin_position": np.sin(np.arange(n_samples) / 5.0),
            "cos_position": np.cos(np.arange(n_samples) / 5.0),
            "composition_token": data.property_node_tokens.mean(),
            "graph_n_nodes": graph_n_nodes,
            "graph_n_edges": graph_n_edges,
            "graph_density": graph_density,
            "graph_mean_degree": mean_degree,
            "target_property": data.property_targets,
        }
    )
    for lag in [1, 2, 3, 8]:
        df[f"lag_{lag}"] = df["target_property"].shift(lag)
    return df


def evaluate_regressors(property_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    modeling_df = property_df.dropna().reset_index(drop=True)
    multimodal_cols = [
        col
        for col in modeling_df.columns
        if col not in {"target_property", "lag_1", "lag_2", "lag_3", "lag_8", "full_fit_prediction"}
    ]
    history_cols = multimodal_cols + ["lag_1", "lag_2", "lag_3", "lag_8"]

    models = {
        "DummyMean": (multimodal_cols, DummyRegressor(strategy="mean")),
        "Lag8Baseline": (history_cols, LagFeatureBaseline(lag_feature="lag_8")),
        "Ridge": (
            history_cols,
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        ),
        "KNN": (
            history_cols,
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        ),
        "RandomForest": (
            history_cols,
            RandomForestRegressor(
                n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=42
            ),
        ),
    }

    splitter = TimeSeriesSplit(n_splits=5)
    metric_rows = []
    prediction_rows = []

    for name, (feature_cols, model) in models.items():
        X = modeling_df[feature_cols]
        y = modeling_df["target_property"]
        for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            fitted = clone(model)
            fitted.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = fitted.predict(X.iloc[test_idx])
            rmse = float(np.sqrt(mean_squared_error(y.iloc[test_idx], preds)))
            metric_rows.append(
                {
                    "model": name,
                    "fold": fold,
                    "rmse": rmse,
                    "mae": mean_absolute_error(y.iloc[test_idx], preds),
                    "r2": r2_score(y.iloc[test_idx], preds),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "sample_id": modeling_df.iloc[test_idx]["sample_id"].to_numpy(),
                        "actual": y.iloc[test_idx].to_numpy(),
                        "predicted": preds,
                    }
                )
            )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = (
        metrics.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )
    eligible_models = [name for name in summary.index if name not in {"DummyMean", "Lag8Baseline"}]
    best_model = eligible_models[0] if eligible_models else summary.index[0]
    summary.columns = ["_".join(col) for col in summary.columns]
    summary = summary.reset_index()
    return metrics, predictions, best_model


def fit_best_property_model(property_df: pd.DataFrame, best_model_name: str):
    modeling_df = property_df.dropna().reset_index(drop=True)
    multimodal_cols = [
        col
        for col in modeling_df.columns
        if col not in {"target_property", "lag_1", "lag_2", "lag_3", "lag_8", "full_fit_prediction"}
    ]
    history_cols = multimodal_cols + ["lag_1", "lag_2", "lag_3", "lag_8"]
    model_map = {
        "DummyMean": (multimodal_cols, DummyRegressor(strategy="mean")),
        "Lag8Baseline": (history_cols, LagFeatureBaseline(lag_feature="lag_8")),
        "Ridge": (
            history_cols,
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        ),
        "KNN": (
            history_cols,
            Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5))]),
        ),
        "RandomForest": (
            history_cols,
            RandomForestRegressor(
                n_estimators=400, max_depth=6, min_samples_leaf=2, random_state=42
            ),
        ),
    }
    feature_cols, estimator = model_map[best_model_name]
    fitted = clone(estimator)
    X = modeling_df[feature_cols]
    y = modeling_df["target_property"]
    fitted.fit(X, y)
    modeling_df = modeling_df.copy()
    modeling_df["full_fit_prediction"] = fitted.predict(X)
    return fitted, modeling_df


def generate_structure_candidates(data: BenchmarkData) -> tuple[pd.DataFrame, dict[str, float]]:
    training = np.column_stack([data.structure_x, data.structure_y])
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    gmm.fit(training)

    samples, _ = gmm.sample(300)
    log_likelihood = gmm.score_samples(samples)
    seen = {tuple(np.round(point, 4)) for point in training}
    bounds = np.array(
        [
            [training[:, 0].min() - 0.05, training[:, 0].max() + 0.05],
            [training[:, 1].min() - 0.05, training[:, 1].max() + 0.05],
        ]
    )
    selected = []
    for point, _ in sorted(
        zip(samples, log_likelihood), key=lambda item: float(item[1]), reverse=True
    ):
        rounded = tuple(np.round(point, 4))
        if rounded in seen:
            continue
        if not (
            bounds[0, 0] <= point[0] <= bounds[0, 1]
            and bounds[1, 0] <= point[1] <= bounds[1, 1]
        ):
            continue
        min_distance = float(np.min(np.linalg.norm(training - point, axis=1)))
        if min_distance < 0.015:
            continue
        selected.append(
            {
                "lattice_a": float(point[0]),
                "lattice_b": float(point[1]),
                "nearest_training_distance": min_distance,
            }
        )
        if len(selected) == 20:
            break

    candidates = pd.DataFrame(selected)
    stats = {
        "training_unique_points": float(len({tuple(np.round(p, 4)) for p in training})),
        "generated_candidates": float(len(candidates)),
        "mean_nearest_distance": float(candidates["nearest_training_distance"].mean()),
        "training_mean_a": float(training[:, 0].mean()),
        "training_mean_b": float(training[:, 1].mean()),
        "generated_mean_a": float(candidates["lattice_a"].mean()),
        "generated_mean_b": float(candidates["lattice_b"].mean()),
    }
    return candidates, stats


def synthetic_processing_objective(
    temperature: np.ndarray, time_hours: np.ndarray
) -> np.ndarray:
    temperature = np.asarray(temperature, dtype=float)
    time_hours = np.asarray(time_hours, dtype=float)
    peak = np.exp(-(((temperature - 368.0) / 68.0) ** 2 + ((time_hours - 22.0) / 4.8) ** 2))
    texture = 0.12 * np.sin((temperature - 200.0) / 40.0) * np.cos(time_hours / 3.5)
    return peak + texture


def expected_improvement(
    mean: np.ndarray, std: np.ndarray, best: float
) -> np.ndarray:
    std = np.maximum(std, 1e-9)
    improvement = mean - best
    z = improvement / std
    normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / np.sqrt(2.0)))
    return improvement * normal_cdf + std * normal_pdf


def run_bayesian_optimization(data: BenchmarkData) -> tuple[pd.DataFrame, dict[str, float]]:
    temp_lo, temp_hi = data.temperature_bounds
    time_lo, time_hi = data.time_bounds
    points = [
        [data.initial_temperature, data.initial_time],
        [temp_lo, time_lo],
        [temp_hi, time_hi],
        [temp_lo, time_hi],
        [temp_hi, time_lo],
    ]
    observations = [float(synthetic_processing_objective(t, h)) for t, h in points]

    grid_t = np.linspace(temp_lo, temp_hi, 120)
    grid_h = np.linspace(time_lo, time_hi, 120)
    mesh_t, mesh_h = np.meshgrid(grid_t, grid_h)
    candidate_grid = np.column_stack([mesh_t.ravel(), mesh_h.ravel()])

    kernel = ConstantKernel(1.0, (1e-3, 10.0)) * Matern(length_scale=[40.0, 4.0], nu=2.5) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2)
    )

    for _ in range(max(data.optimization_steps - len(points), 0)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=42)
            gp.fit(np.array(points), np.array(observations))
        mean, std = gp.predict(candidate_grid, return_std=True)
        acquisition = expected_improvement(mean, std, best=max(observations))

        candidate_idx = int(np.argmax(acquisition))
        candidate = candidate_grid[candidate_idx]
        if any(np.allclose(candidate, existing) for existing in points):
            break
        points.append(candidate.tolist())
        observations.append(float(synthetic_processing_objective(candidate[0], candidate[1])))

    trace = pd.DataFrame(points, columns=["temperature_C", "time_h"])
    trace["objective"] = observations
    best_idx = int(trace["objective"].idxmax())
    stats = {
        "initial_objective": float(trace.iloc[0]["objective"]),
        "best_objective": float(trace.iloc[best_idx]["objective"]),
        "improvement_pct": float(
            100.0 * (trace.iloc[best_idx]["objective"] - trace.iloc[0]["objective"]) / abs(trace.iloc[0]["objective"])
        ),
        "best_temperature_C": float(trace.iloc[best_idx]["temperature_C"]),
        "best_time_h": float(trace.iloc[best_idx]["time_h"]),
    }
    return trace, stats


def plot_property_overview(data: BenchmarkData, property_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    sns.heatmap(
        data.property_grid,
        cmap="viridis",
        ax=axes[0],
        cbar_kws={"label": "Spectral / structural signal"},
    )
    axes[0].set_title("Serialized 13x9 spectral grid")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    positions = {
        0: (0.0, 1.0),
        1: (-0.95, 0.31),
        2: (-0.59, -0.81),
        3: (0.59, -0.81),
        4: (0.95, 0.31),
    }
    for src, dst in data.property_edges:
        x_vals = [positions[src][0], positions[dst][0]]
        y_vals = [positions[src][1], positions[dst][1]]
        axes[1].plot(x_vals, y_vals, color="#355070", linewidth=1.4, alpha=0.8)
    for node_id, (x_pos, y_pos) in positions.items():
        axes[1].add_patch(Circle((x_pos, y_pos), 0.13, color="#EAAC8B"))
        axes[1].text(x_pos, y_pos, str(node_id), ha="center", va="center", fontsize=11)
    axes[1].set_title("Complete 5-node crystal graph prior")
    axes[1].set_aspect("equal")
    axes[1].axis("off")

    axes[2].plot(
        property_df["sample_id"],
        property_df["target_property"],
        color="#6D597A",
        linewidth=2.0,
        label="Target property",
    )
    axes[2].set_title("Continuous property target sequence")
    axes[2].set_xlabel("Sample index")
    axes[2].set_ylabel("Property value")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "data_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_property_results(
    metrics: pd.DataFrame, predictions: pd.DataFrame, best_model: str
) -> None:
    summary = (
        metrics.groupby("model")[["rmse", "mae", "r2"]]
        .mean()
        .reset_index()
        .sort_values("rmse")
    )
    best_predictions = predictions[predictions["model"] == best_model]

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    axes[0].bar(
        summary["model"],
        summary["rmse"],
        color=sns.color_palette("crest", n_colors=len(summary)),
    )
    axes[0].set_title("Blocked CV error by regressor")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].scatter(
        best_predictions["actual"],
        best_predictions["predicted"],
        color="#2A9D8F",
        s=42,
        alpha=0.8,
    )
    lims = [
        min(best_predictions["actual"].min(), best_predictions["predicted"].min()),
        max(best_predictions["actual"].max(), best_predictions["predicted"].max()),
    ]
    axes[1].plot(lims, lims, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title(f"{best_model} parity plot")
    axes[1].set_xlabel("Observed property")
    axes[1].set_ylabel("Predicted property")

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "property_model_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_generation_results(
    data: BenchmarkData, candidates: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))

    axes[0].plot(data.structure_x, label="Latent dimension A", color="#264653", linewidth=1.8)
    axes[0].plot(data.structure_y, label="Latent dimension B", color="#E76F51", linewidth=1.8)
    axes[0].set_title("Periodic latent structure traces")
    axes[0].set_xlabel("Sequence index")
    axes[0].set_ylabel("Latent value")
    axes[0].legend(frameon=False)

    axes[1].scatter(
        data.structure_x,
        data.structure_y,
        label="Observed motifs",
        color="#1D3557",
        alpha=0.45,
        s=35,
    )
    axes[1].scatter(
        candidates["lattice_a"],
        candidates["lattice_b"],
        label="Generated candidates",
        color="#E63946",
        alpha=0.9,
        s=44,
        marker="x",
    )
    axes[1].set_title("Observed vs generated structure manifold")
    axes[1].set_xlabel("Lattice / latent dimension A")
    axes[1].set_ylabel("Lattice / latent dimension B")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "structure_generation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_optimization_results(trace: pd.DataFrame, data: BenchmarkData) -> None:
    temp_grid = np.linspace(*data.temperature_bounds, 200)
    time_grid = np.linspace(*data.time_bounds, 200)
    mesh_t, mesh_h = np.meshgrid(temp_grid, time_grid)
    surface = synthetic_processing_objective(mesh_t, mesh_h)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
    contour = axes[0].contourf(mesh_t, mesh_h, surface, levels=25, cmap="mako")
    axes[0].plot(trace["temperature_C"], trace["time_h"], color="white", linewidth=1.6)
    axes[0].scatter(trace["temperature_C"], trace["time_h"], c=np.arange(len(trace)), cmap="rocket", s=42)
    axes[0].set_title("Autonomous optimization trajectory")
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("Time (h)")
    fig.colorbar(contour, ax=axes[0], label="Synthetic objective")

    axes[1].plot(trace.index, trace["objective"], marker="o", color="#2A9D8F")
    axes[1].set_title("Objective improvement over iterations")
    axes[1].set_xlabel("Evaluation step")
    axes[1].set_ylabel("Objective")

    fig.tight_layout()
    fig.savefig(REPORT_IMAGE_DIR / "optimization_results.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_outputs(
    property_df: pd.DataFrame,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    candidate_df: pd.DataFrame,
    optimization_trace: pd.DataFrame,
    summary_payload: dict,
) -> None:
    property_df.to_csv(OUTPUT_DIR / "tables" / "property_reconstructed_dataset.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "tables" / "property_cv_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "tables" / "property_cv_predictions.csv", index=False)
    candidate_df.to_csv(OUTPUT_DIR / "tables" / "generated_structure_candidates.csv", index=False)
    optimization_trace.to_csv(OUTPUT_DIR / "tables" / "optimization_trace.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2))


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_directories()
    data = load_benchmark_data()

    property_df = build_property_table(data)
    metrics, predictions, best_model = evaluate_regressors(property_df)
    best_fitted_model, property_df = fit_best_property_model(property_df, best_model)
    summary_metrics = (
        metrics.groupby("model")[["rmse", "mae", "r2"]]
        .agg(["mean", "std"])
        .sort_values(("rmse", "mean"))
    )
    summary_metrics.columns = ["_".join(col) for col in summary_metrics.columns]
    summary_metrics = summary_metrics.reset_index()

    structure_candidates, generation_stats = generate_structure_candidates(data)
    optimization_trace, optimization_stats = run_bayesian_optimization(data)

    plot_property_overview(data, property_df)
    plot_property_results(metrics, predictions, best_model)
    plot_generation_results(data, structure_candidates)
    plot_optimization_results(optimization_trace, data)

    summary_payload = {
        "assumptions": {
            "property_reconstruction": "The property block was reconstructed into 97 pseudo-samples by aligning the first 97 cells of the serialized 13x9 grid with the 97 target values, while treating the 5-node complete graph and constant composition token as shared context.",
            "optimization_objective": "The optimization landscape is a synthetic surrogate used only to validate autonomous optimization code paths because the source file supplies bounds and schedule metadata but no empirical response surface.",
        },
        "property_model_summary": summary_metrics.to_dict(orient="records"),
        "best_property_model": best_model,
        "generation_stats": generation_stats,
        "optimization_stats": optimization_stats,
    }

    save_outputs(
        property_df=property_df,
        metrics=metrics,
        predictions=predictions,
        candidate_df=structure_candidates,
        optimization_trace=optimization_trace,
        summary_payload=summary_payload,
    )

    print(json.dumps(summary_payload, indent=2))
    _ = best_fitted_model


if __name__ == "__main__":
    main()
