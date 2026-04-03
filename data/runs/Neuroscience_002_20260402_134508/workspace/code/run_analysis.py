#!/usr/bin/env python3
"""End-to-end analysis for simulated neuron fragment merge prediction."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve


SEED = 20260402
N_BOOTSTRAPS = 200

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGE_DIR = REPORT_DIR / "images"

TRAIN_PATH = DATA_DIR / "train_simulated.csv"
TEST_PATH = DATA_DIR / "test_simulated.csv"

FEATURE_COLUMNS = [str(i) for i in range(20)]
MORPHOLOGY_COLUMNS = [str(i) for i in range(5)]
INTENSITY_COLUMNS = [str(i) for i in range(5, 10)]
EMBEDDING_COLUMNS = [str(i) for i in range(10, 20)]
ALL_COLUMNS = FEATURE_COLUMNS + ["label", "degradation"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    expected_columns = ALL_COLUMNS
    if list(train_df.columns) != expected_columns or list(test_df.columns) != expected_columns:
        raise ValueError("Unexpected dataset schema.")
    return train_df, test_df


def stratify_key(df: pd.DataFrame) -> pd.Series:
    return df["degradation"].astype(str) + "__" + df["label"].astype(int).astype(str)


def compute_sample_weight(y: pd.Series | np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y).astype(int)
    class_counts = np.bincount(y_arr)
    total = len(y_arr)
    weights = total / (2.0 * class_counts[y_arr])
    return weights


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def select_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    for threshold in thresholds:
        metrics = evaluate_predictions(y_true, y_prob, threshold)
        rows.append(metrics)
    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values(
        by=["f1", "balanced_accuracy", "precision"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best_row["threshold"]), threshold_df


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    metrics = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n_samples, size=n_samples)
        y_boot = y_true[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        prob_boot = y_prob[idx]
        metrics.append(evaluate_predictions(y_boot, prob_boot, threshold))
    boot_df = pd.DataFrame(metrics)
    summary = []
    for column in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "brier"]:
        summary.append(
            {
                "metric": column,
                "mean": boot_df[column].mean(),
                "ci_low": boot_df[column].quantile(0.025),
                "ci_high": boot_df[column].quantile(0.975),
            }
        )
    return pd.DataFrame(summary)


def build_models() -> dict[str, Pipeline | object]:
    numeric_features = FEATURE_COLUMNS
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_features)],
        remainder="drop",
    )

    logistic = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=SEED,
                ),
            ),
        ]
    )

    random_forest = RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=SEED,
    )

    return {
        "LogisticRegression": logistic,
        "RandomForest": random_forest,
    }


def fit_with_sample_weight(model: Pipeline | object, x: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray) -> object:
    if isinstance(model, Pipeline):
        return model.fit(x, y, model__sample_weight=sample_weight)
    return model.fit(x, y, sample_weight=sample_weight)


def plot_class_balance(train_df: pd.DataFrame) -> None:
    balance = (
        train_df.groupby(["degradation", "label"])
        .size()
        .reset_index(name="count")
    )
    balance["label_name"] = balance["label"].map({0.0: "Different neuron", 1.0: "Same neuron"})

    plt.figure(figsize=(9, 5))
    sns.barplot(data=balance, x="degradation", y="count", hue="label_name", palette="Set2")
    plt.xticks(rotation=15)
    plt.ylabel("Sample count")
    plt.xlabel("Degradation type")
    plt.title("Training-set class balance across degradation conditions")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "data_balance.png", dpi=300)
    plt.close()


def plot_group_feature_distributions(train_df: pd.DataFrame) -> None:
    plot_df = pd.DataFrame(
        {
            "Morphology mean": train_df[MORPHOLOGY_COLUMNS].mean(axis=1),
            "Intensity mean": train_df[INTENSITY_COLUMNS].mean(axis=1),
            "Embedding mean": train_df[EMBEDDING_COLUMNS].mean(axis=1),
            "label": train_df["label"].map({0.0: "Different neuron", 1.0: "Same neuron"}),
        }
    )
    sample = plot_df.sample(n=min(30000, len(plot_df)), random_state=SEED)
    melted = sample.melt(id_vars="label", var_name="Feature group", value_name="Mean feature value")

    plt.figure(figsize=(9, 5))
    sns.violinplot(
        data=melted,
        x="Feature group",
        y="Mean feature value",
        hue="label",
        split=True,
        inner="quartile",
        palette="Set2",
    )
    plt.title("Approximate modality-level feature separation")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_group_distributions.png", dpi=300)
    plt.close()


def plot_feature_correlation(train_df: pd.DataFrame) -> None:
    corr = train_df[FEATURE_COLUMNS].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap="vlag", center=0.0, square=True)
    plt.title("Feature correlation structure")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_correlation_heatmap.png", dpi=300)
    plt.close()


def plot_pca_projection(train_df: pd.DataFrame) -> None:
    sample = train_df.sample(n=min(12000, len(train_df)), random_state=SEED).copy()
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(sample[FEATURE_COLUMNS])
    x_pca = PCA(n_components=2, random_state=SEED).fit_transform(x_scaled)
    plot_df = pd.DataFrame(
        {
            "PC1": x_pca[:, 0],
            "PC2": x_pca[:, 1],
            "label": sample["label"].map({0.0: "Different neuron", 1.0: "Same neuron"}),
            "degradation": sample["degradation"].values,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="label",
        alpha=0.45,
        s=18,
        ax=axes[0],
        palette="Set1",
    )
    axes[0].set_title("PCA projection by class")

    sns.scatterplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="degradation",
        alpha=0.45,
        s=18,
        ax=axes[1],
        palette="tab10",
    )
    axes[1].set_title("PCA projection by degradation")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "pca_projection.png", dpi=300)
    plt.close()


def plot_model_curves(curve_data: dict[str, dict[str, np.ndarray]], final_model_name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for model_name, data in curve_data.items():
        axes[0].plot(data["fpr"], data["tpr"], label=model_name, linewidth=2.2 if model_name == final_model_name else 1.6)
        axes[1].plot(data["recall_curve"], data["precision_curve"], label=model_name, linewidth=2.2 if model_name == final_model_name else 1.6)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_title("Validation ROC curves")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")

    axes[1].set_title("Validation precision-recall curves")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")

    for ax in axes:
        ax.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "model_comparison_curves.png", dpi=300)
    plt.close()


def plot_threshold_selection(threshold_df: pd.DataFrame, chosen_threshold: float) -> None:
    plt.figure(figsize=(9, 5))
    for metric_name in ["precision", "recall", "f1", "balanced_accuracy"]:
        plt.plot(threshold_df["threshold"], threshold_df[metric_name], label=metric_name)
    plt.axvline(chosen_threshold, color="black", linestyle="--", linewidth=1.5, label=f"chosen={chosen_threshold:.2f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Metric value")
    plt.title("Validation threshold selection for the final classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "threshold_selection.png", dpi=300)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Test-set calibration of the final model")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "calibration_plot.png", dpi=300)
    plt.close()


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    top = importance_df.sort_values("importance_mean", ascending=False).head(12)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=top, x="importance_mean", y="feature", hue="feature", palette="crest", legend=False)
    plt.xlabel("Model feature importance")
    plt.ylabel("Feature")
    plt.title("Final-model feature importance")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "feature_importance.png", dpi=300)
    plt.close()


def plot_degradation_metrics(metrics_df: pd.DataFrame) -> None:
    melted = metrics_df.melt(
        id_vars="degradation",
        value_vars=["auroc", "auprc", "f1", "balanced_accuracy"],
        var_name="metric",
        value_name="value",
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=melted, x="degradation", y="value", hue="metric", palette="Set2")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15)
    plt.title("Held-out test performance by degradation type")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "degradation_metrics.png", dpi=300)
    plt.close()


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_dirs()

    train_df, test_df = load_data()
    print("Loaded datasets.", flush=True)

    plot_class_balance(train_df)
    plot_group_feature_distributions(train_df)
    plot_feature_correlation(train_df)
    plot_pca_projection(train_df)
    print("Saved data overview figures.", flush=True)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train_df, stratify_key(train_df)))
    dev_df = train_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    x_dev = dev_df[FEATURE_COLUMNS]
    y_dev = dev_df["label"].astype(int).to_numpy()
    x_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"].astype(int).to_numpy()

    sample_weight = compute_sample_weight(y_dev)

    models = build_models()
    model_rows = []
    curve_data = {}
    fitted_models: dict[str, object] = {}

    for model_name, model in models.items():
        print(f"Training {model_name}...", flush=True)
        start = time.time()
        fitted = clone(model)
        fit_with_sample_weight(fitted, x_dev, y_dev, sample_weight)
        elapsed = time.time() - start
        val_prob = fitted.predict_proba(x_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, val_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_prob)
        curve_data[model_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
        }
        metrics = evaluate_predictions(y_val, val_prob, threshold=0.5)
        metrics.update({"model": model_name, "fit_seconds": elapsed})
        model_rows.append(metrics)
        fitted_models[model_name] = fitted
        print(
            f"Finished {model_name}: AUROC={metrics['auroc']:.4f}, AUCPR={metrics['auprc']:.4f}, F1@0.5={metrics['f1']:.4f}",
            flush=True,
        )

    model_comparison = pd.DataFrame(model_rows).sort_values(
        by=["auprc", "auroc", "f1"],
        ascending=[False, False, False],
    )
    model_comparison.to_csv(OUTPUT_DIR / "model_comparison_validation.csv", index=False)

    final_model_name = model_comparison.iloc[0]["model"]
    final_dev_model = fitted_models[final_model_name]
    val_prob = final_dev_model.predict_proba(x_val)[:, 1]
    chosen_threshold, threshold_df = select_threshold(y_val, val_prob)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_sweep_validation.csv", index=False)
    print(f"Selected {final_model_name} with threshold {chosen_threshold:.2f}.", flush=True)

    plot_model_curves(curve_data, final_model_name)
    plot_threshold_selection(threshold_df, chosen_threshold)

    x_train_full = train_df[FEATURE_COLUMNS]
    y_train_full = train_df["label"].astype(int).to_numpy()
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["label"].astype(int).to_numpy()
    full_sample_weight = compute_sample_weight(y_train_full)

    final_model = clone(models[final_model_name])
    fit_with_sample_weight(final_model, x_train_full, y_train_full, full_sample_weight)
    test_prob = final_model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=chosen_threshold)
    test_metrics["model"] = final_model_name
    test_metrics["selected_threshold"] = chosen_threshold
    test_metrics["train_size"] = int(len(train_df))
    test_metrics["test_size"] = int(len(test_df))
    save_json(OUTPUT_DIR / "final_test_metrics.json", test_metrics)

    default_metrics = evaluate_predictions(y_test, test_prob, threshold=0.5)
    save_json(OUTPUT_DIR / "default_threshold_test_metrics.json", default_metrics)
    print(
        f"Test metrics: AUROC={test_metrics['auroc']:.4f}, AUCPR={test_metrics['auprc']:.4f}, F1={test_metrics['f1']:.4f}.",
        flush=True,
    )

    bootstrap_df = bootstrap_confidence_intervals(y_test, test_prob, threshold=chosen_threshold)
    bootstrap_df.to_csv(OUTPUT_DIR / "bootstrap_confidence_intervals.csv", index=False)

    prediction_df = test_df[["label", "degradation"]].copy()
    prediction_df["predicted_probability"] = test_prob
    prediction_df["predicted_label"] = (test_prob >= chosen_threshold).astype(int)
    prediction_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    degradation_rows = []
    for degradation, group in prediction_df.groupby("degradation"):
        y_group = group["label"].astype(int).to_numpy()
        p_group = group["predicted_probability"].to_numpy()
        metrics = evaluate_predictions(y_group, p_group, threshold=chosen_threshold)
        metrics["degradation"] = degradation
        degradation_rows.append(metrics)
    degradation_df = pd.DataFrame(degradation_rows).sort_values("degradation")
    degradation_df.to_csv(OUTPUT_DIR / "degradation_metrics_test.csv", index=False)

    plot_calibration(y_test, test_prob)
    plot_degradation_metrics(degradation_df)

    print("Saving feature importance...", flush=True)
    if hasattr(final_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": final_model.feature_importances_,
                "importance_std": np.zeros(len(FEATURE_COLUMNS)),
            }
        ).sort_values("importance_mean", ascending=False)
    else:
        coef = final_model.named_steps["model"].coef_.ravel()
        importance_df = pd.DataFrame(
            {
                "feature": FEATURE_COLUMNS,
                "importance_mean": np.abs(coef),
                "importance_std": np.zeros(len(FEATURE_COLUMNS)),
            }
        ).sort_values("importance_mean", ascending=False)
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    plot_feature_importance(importance_df)

    dataset_summary = {
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "positive_rate_train": float(train_df["label"].mean()),
        "positive_rate_test": float(test_df["label"].mean()),
        "degradation_counts_train": train_df["degradation"].value_counts().to_dict(),
        "degradation_counts_test": test_df["degradation"].value_counts().to_dict(),
        "selected_model": final_model_name,
    }
    save_json(OUTPUT_DIR / "dataset_summary.json", dataset_summary)
    print("Analysis complete.", flush=True)


if __name__ == "__main__":
    main()
