from __future__ import annotations

import json
import math
import os
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(WORKSPACE / "outputs" / ".mplconfig"))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
TRAIN_FRACTION = 0.7
DATA_DIR = WORKSPACE / "data"
OUTPUT_DIR = WORKSPACE / "outputs"
IMAGE_DIR = WORKSPACE / "report" / "images"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"


def euclidean(ax: pd.Series, ay: pd.Series, bx: pd.Series, by: pd.Series) -> pd.Series:
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def angle(ax: pd.Series, ay: pd.Series, bx: pd.Series, by: pd.Series) -> pd.Series:
    return np.arctan2(by - ay, bx - ax)


def wrapped_angle_delta(a: pd.Series, b: pd.Series) -> pd.Series:
    delta = a - b
    return np.arctan2(np.sin(delta), np.cos(delta))


def bout_lengths(values: pd.Series) -> list[int]:
    lengths: list[int] = []
    current = 0
    for value in values.astype(int):
        if value == 1:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def build_engineered_features(raw: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=raw.index)

    for mouse in (1, 2):
        features[f"body_length_{mouse}"] = euclidean(
            raw[f"Nose_{mouse}_x"],
            raw[f"Nose_{mouse}_y"],
            raw[f"Tail_base_{mouse}_x"],
            raw[f"Tail_base_{mouse}_y"],
        )
        features[f"body_width_{mouse}"] = euclidean(
            raw[f"Lat_left_{mouse}_x"],
            raw[f"Lat_left_{mouse}_y"],
            raw[f"Lat_right_{mouse}_x"],
            raw[f"Lat_right_{mouse}_y"],
        )
        features[f"head_width_{mouse}"] = euclidean(
            raw[f"Ear_left_{mouse}_x"],
            raw[f"Ear_left_{mouse}_y"],
            raw[f"Ear_right_{mouse}_x"],
            raw[f"Ear_right_{mouse}_y"],
        )
        features[f"center_speed_{mouse}"] = euclidean(
            raw[f"Center_{mouse}_x"],
            raw[f"Center_{mouse}_y"],
            raw[f"Center_{mouse}_x"].shift(1),
            raw[f"Center_{mouse}_y"].shift(1),
        ).fillna(0.0)
        features[f"nose_speed_{mouse}"] = euclidean(
            raw[f"Nose_{mouse}_x"],
            raw[f"Nose_{mouse}_y"],
            raw[f"Nose_{mouse}_x"].shift(1),
            raw[f"Nose_{mouse}_y"].shift(1),
        ).fillna(0.0)
        features[f"tail_speed_{mouse}"] = euclidean(
            raw[f"Tail_base_{mouse}_x"],
            raw[f"Tail_base_{mouse}_y"],
            raw[f"Tail_base_{mouse}_x"].shift(1),
            raw[f"Tail_base_{mouse}_y"].shift(1),
        ).fillna(0.0)
        features[f"body_angle_{mouse}"] = angle(
            raw[f"Tail_base_{mouse}_x"],
            raw[f"Tail_base_{mouse}_y"],
            raw[f"Nose_{mouse}_x"],
            raw[f"Nose_{mouse}_y"],
        )
        confidence_cols = [
            f"Nose_{mouse}_p",
            f"Ear_left_{mouse}_p",
            f"Ear_right_{mouse}_p",
            f"Center_{mouse}_p",
            f"Lat_left_{mouse}_p",
            f"Lat_right_{mouse}_p",
            f"Tail_base_{mouse}_p",
            f"Tail_end_{mouse}_p",
        ]
        features[f"mean_pose_conf_{mouse}"] = raw[confidence_cols].mean(axis=1)

    features["nose_to_nose"] = euclidean(
        raw["Nose_1_x"], raw["Nose_1_y"], raw["Nose_2_x"], raw["Nose_2_y"]
    )
    features["center_to_center"] = euclidean(
        raw["Center_1_x"], raw["Center_1_y"], raw["Center_2_x"], raw["Center_2_y"]
    )
    features["nose1_to_center2"] = euclidean(
        raw["Nose_1_x"], raw["Nose_1_y"], raw["Center_2_x"], raw["Center_2_y"]
    )
    features["nose2_to_center1"] = euclidean(
        raw["Nose_2_x"], raw["Nose_2_y"], raw["Center_1_x"], raw["Center_1_y"]
    )
    features["nose1_to_tail2"] = euclidean(
        raw["Nose_1_x"], raw["Nose_1_y"], raw["Tail_base_2_x"], raw["Tail_base_2_y"]
    )
    features["nose2_to_tail1"] = euclidean(
        raw["Nose_2_x"], raw["Nose_2_y"], raw["Tail_base_1_x"], raw["Tail_base_1_y"]
    )
    features["tail_to_tail"] = euclidean(
        raw["Tail_base_1_x"],
        raw["Tail_base_1_y"],
        raw["Tail_base_2_x"],
        raw["Tail_base_2_y"],
    )
    features["center_distance_delta"] = features["center_to_center"].diff().fillna(0.0)
    features["nose_distance_delta"] = features["nose_to_nose"].diff().fillna(0.0)
    features["speed_difference"] = features["center_speed_1"] - features["center_speed_2"]
    features["body_angle_difference"] = wrapped_angle_delta(
        features["body_angle_1"], features["body_angle_2"]
    )

    vector_12 = angle(raw["Center_1_x"], raw["Center_1_y"], raw["Center_2_x"], raw["Center_2_y"])
    vector_21 = angle(raw["Center_2_x"], raw["Center_2_y"], raw["Center_1_x"], raw["Center_1_y"])
    features["facing_error_1_to_2"] = wrapped_angle_delta(features["body_angle_1"], vector_12).abs()
    features["facing_error_2_to_1"] = wrapped_angle_delta(features["body_angle_2"], vector_21).abs()

    temporal_cols = [
        "center_to_center",
        "nose_to_nose",
        "nose1_to_center2",
        "nose2_to_center1",
        "nose1_to_tail2",
        "nose2_to_tail1",
        "center_speed_1",
        "center_speed_2",
        "nose_speed_1",
        "nose_speed_2",
    ]
    for col in temporal_cols:
        for window in (5, 15):
            roll = features[col].rolling(window=window, min_periods=1)
            features[f"{col}_mean_{window}"] = roll.mean()
            features[f"{col}_std_{window}"] = roll.std().fillna(0.0)

    return features.fillna(0.0)


def make_models() -> dict[str, object]:
    return {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=1,
        ),
    }


def summarize_labels(targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in ["Attack", "Sniffing"]:
        bouts = bout_lengths(targets[label])
        rows.append(
            {
                "label": label,
                "positive_frames": int(targets[label].sum()),
                "positive_fraction": float(targets[label].mean()),
                "bout_count": len(bouts),
                "median_bout_length_frames": float(np.median(bouts) if bouts else 0.0),
                "max_bout_length_frames": int(max(bouts) if bouts else 0),
            }
        )
    return pd.DataFrame(rows)


def get_protocol_indices(n_rows: int, y: pd.Series, protocol: str) -> tuple[np.ndarray, np.ndarray]:
    all_indices = np.arange(n_rows)
    if protocol == "random_split":
        train_idx, test_idx = train_test_split(
            all_indices,
            train_size=TRAIN_FRACTION,
            random_state=SEED,
            stratify=y,
        )
        return np.sort(train_idx), np.sort(test_idx)
    if protocol == "temporal_split":
        split_point = int(math.floor(n_rows * TRAIN_FRACTION))
        return all_indices[:split_point], all_indices[split_point:]
    raise ValueError(f"Unknown protocol: {protocol}")


def evaluate_models(features: pd.DataFrame, targets: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    results: list[dict] = []
    artifacts: dict[str, dict] = {}

    for label in ["Attack", "Sniffing"]:
        y = targets[label].astype(int)
        artifacts[label] = {}
        for protocol in ("random_split", "temporal_split"):
            train_idx, test_idx = get_protocol_indices(len(features), y, protocol)
            X_train = features.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            protocol_key = f"{label}__{protocol}"
            artifacts[label][protocol] = {
                "train_index": train_idx,
                "test_index": test_idx,
                "y_test": y_test,
                "models": {},
            }

            for model_name, estimator in make_models().items():
                model = clone(estimator)
                model.fit(X_train, y_train)
                probabilities = model.predict_proba(X_test)[:, 1]
                predictions = (probabilities >= 0.5).astype(int)

                metric_row = {
                    "label": label,
                    "protocol": protocol,
                    "model": model_name,
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "positive_train": int(y_train.sum()),
                    "positive_test": int(y_test.sum()),
                    "average_precision": float(average_precision_score(y_test, probabilities)),
                    "roc_auc": float(roc_auc_score(y_test, probabilities)),
                    "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
                    "precision": float(precision_score(y_test, predictions, zero_division=0)),
                    "recall": float(recall_score(y_test, predictions, zero_division=0)),
                    "f1": float(f1_score(y_test, predictions, zero_division=0)),
                }
                results.append(metric_row)

                report = classification_report(
                    y_test,
                    predictions,
                    output_dict=True,
                    zero_division=0,
                )
                confusion = confusion_matrix(y_test, predictions)
                precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, probabilities)

                prediction_frame = pd.DataFrame(
                    {
                        "frame_index": test_idx,
                        "y_true": y_test.to_numpy(),
                        "y_pred": predictions,
                        "probability": probabilities,
                    }
                )
                prediction_frame.to_csv(
                    OUTPUT_DIR / f"predictions_{label.lower()}_{protocol}_{model_name}.csv",
                    index=False,
                )

                pd.DataFrame(report).to_csv(
                    OUTPUT_DIR / f"classification_report_{label.lower()}_{protocol}_{model_name}.csv"
                )
                pd.DataFrame(
                    confusion,
                    index=["true_0", "true_1"],
                    columns=["pred_0", "pred_1"],
                ).to_csv(
                    OUTPUT_DIR / f"confusion_matrix_{label.lower()}_{protocol}_{model_name}.csv"
                )
                pd.DataFrame(
                    {
                        "precision": precision_curve,
                        "recall": recall_curve,
                        "threshold": np.append(thresholds, np.nan),
                    }
                ).to_csv(
                    OUTPUT_DIR / f"precision_recall_curve_{label.lower()}_{protocol}_{model_name}.csv",
                    index=False,
                )
                joblib.dump(
                    model,
                    OUTPUT_DIR / f"model_{label.lower()}_{protocol}_{model_name}.joblib",
                )

                artifacts[label][protocol]["models"][model_name] = {
                    "estimator": model,
                    "probabilities": probabilities,
                    "predictions": predictions,
                    "confusion": confusion,
                    "classification_report": report,
                    "prediction_frame": prediction_frame,
                }

            rf_model = artifacts[label][protocol]["models"]["random_forest"]["estimator"]
            perm = permutation_importance(
                rf_model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=SEED,
                scoring="average_precision",
                n_jobs=1,
            )
            importance = pd.DataFrame(
                {
                    "feature": features.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                    "forest_importance": rf_model.feature_importances_,
                }
            ).sort_values("importance_mean", ascending=False)
            importance.to_csv(
                OUTPUT_DIR / f"feature_importance_{label.lower()}_{protocol}_random_forest.csv",
                index=False,
            )
            artifacts[label][protocol]["feature_importance"] = importance

    results_df = pd.DataFrame(results).sort_values(
        ["label", "protocol", "average_precision"], ascending=[True, True, False]
    )
    results_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    return results_df, artifacts


def plot_data_overview(targets: pd.DataFrame, label_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[2.2, 1.0])
    frame_index = np.arange(len(targets))

    axes[0].fill_between(frame_index, 0, targets["Attack"], step="pre", alpha=0.7, color="#b22222", label="Attack")
    axes[0].fill_between(frame_index, 1.2, 1.2 + targets["Sniffing"], step="pre", alpha=0.7, color="#1f77b4", label="Sniffing")
    axes[0].set_yticks([0.5, 1.7])
    axes[0].set_yticklabels(["Attack", "Sniffing"])
    axes[0].set_xlabel("Frame")
    axes[0].set_title("Annotated behavior timeline across the full sequence")

    label_summary = label_summary.copy()
    label_summary["negative_frames"] = len(targets) - label_summary["positive_frames"]
    melted = label_summary.melt(
        id_vars="label",
        value_vars=["positive_frames", "negative_frames"],
        var_name="class",
        value_name="frames",
    )
    sns.barplot(data=melted, x="label", y="frames", hue="class", ax=axes[1], palette=["#c94c4c", "#cfd8dc"])
    axes[1].set_title("Class balance")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Frames")

    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "data_overview.png")
    plt.close(fig)


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    subset = metrics[metrics["model"].isin(["logistic_regression", "random_forest"])].copy()
    plot_df = subset.melt(
        id_vars=["label", "protocol", "model"],
        value_vars=["average_precision", "f1"],
        var_name="metric",
        value_name="value",
    )
    g = sns.catplot(
        data=plot_df,
        x="model",
        y="value",
        hue="protocol",
        col="label",
        row="metric",
        kind="bar",
        height=4.2,
        aspect=1.2,
        palette="deep",
        sharey=False,
    )
    g.set_axis_labels("", "Score")
    g.set_titles("{row_name} | {col_name}")
    g.savefig(IMAGE_DIR / "metric_comparison.png")
    plt.close("all")


def plot_precision_recall_curves(artifacts: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=False, sharey=False)
    protocols = ["random_split", "temporal_split"]
    labels = ["Attack", "Sniffing"]

    for row_idx, label in enumerate(labels):
        for col_idx, protocol in enumerate(protocols):
            ax = axes[row_idx, col_idx]
            for model_name, color in [("logistic_regression", "#2a9d8f"), ("random_forest", "#e76f51")]:
                pred_frame = artifacts[label][protocol]["models"][model_name]["prediction_frame"]
                precision, recall, _ = precision_recall_curve(pred_frame["y_true"], pred_frame["probability"])
                ap = average_precision_score(pred_frame["y_true"], pred_frame["probability"])
                ax.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})", color=color, linewidth=2.4)
            baseline = pred_frame["y_true"].mean()
            ax.axhline(baseline, linestyle="--", color="#6c757d", linewidth=1.5, label="prevalence")
            ax.set_title(f"{label} | {protocol}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "precision_recall_curves.png")
    plt.close(fig)


def plot_confusion_matrices(artifacts: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for row_idx, label in enumerate(["Attack", "Sniffing"]):
        for col_idx, protocol in enumerate(["random_split", "temporal_split"]):
            ax = axes[row_idx, col_idx]
            confusion = artifacts[label][protocol]["models"]["random_forest"]["confusion"]
            sns.heatmap(
                confusion,
                annot=True,
                fmt="d",
                cbar=False,
                cmap="Blues",
                ax=ax,
                xticklabels=["Pred 0", "Pred 1"],
                yticklabels=["True 0", "True 1"],
            )
            ax.set_title(f"{label} | {protocol}")
            ax.set_xlabel("")
            ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "confusion_matrices.png")
    plt.close(fig)


def plot_feature_importance(artifacts: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for idx, label in enumerate(["Attack", "Sniffing"]):
        ax = axes[idx]
        importance = (
            artifacts[label]["random_split"]["feature_importance"]
            .head(12)
            .sort_values("importance_mean", ascending=True)
        )
        sns.barplot(
            data=importance,
            x="importance_mean",
            y="feature",
            ax=ax,
            color="#264653",
        )
        ax.set_title(f"{label}: top RF permutation importances")
        ax.set_xlabel("Average precision drop after permutation")
        ax.set_ylabel("")

    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "feature_importance.png")
    plt.close(fig)


def plot_temporal_probability_trace(artifacts: dict) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
    for idx, label in enumerate(["Attack", "Sniffing"]):
        ax = axes[idx]
        pred_frame = artifacts[label]["temporal_split"]["models"]["random_forest"]["prediction_frame"]
        ax.plot(pred_frame["frame_index"], pred_frame["probability"], color="#e76f51", linewidth=1.8, label="RF probability")
        ax.fill_between(
            pred_frame["frame_index"],
            0,
            pred_frame["y_true"],
            step="pre",
            alpha=0.35,
            color="#1d3557",
            label="Ground truth",
        )
        ax.axhline(0.5, linestyle="--", color="#6c757d", linewidth=1.2)
        ax.set_title(f"{label}: temporal holdout probability trace")
        ax.set_ylabel("Probability")
        ax.legend(frameon=True, fontsize=10)
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    fig.savefig(IMAGE_DIR / "temporal_probability_trace.png")
    plt.close(fig)


def summarize_reference_file() -> pd.DataFrame:
    reference = pd.read_csv(DATA_DIR / "Together_1_machine_results_reference.csv")
    rows = []
    for label in ["Attack", "Sniffing"]:
        rows.append(
            {
                "label": label,
                "rows": int(len(reference)),
                "positive_frames": int(reference[label].sum()),
                "positive_fraction": float(reference[label].mean()),
                "mean_probability": float(reference[f"Probability_{label}"].mean()),
            }
        )
    reference_summary = pd.DataFrame(rows)
    reference_summary.to_csv(OUTPUT_DIR / "reference_file_summary.csv", index=False)
    return reference_summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / ".mplconfig").mkdir(exist_ok=True)

    raw_features = pd.read_csv(DATA_DIR / "Together_1_features_extracted.csv")
    targets = pd.read_csv(DATA_DIR / "Together_1_targets_inserted.csv")[["Attack", "Sniffing"]]

    engineered = build_engineered_features(raw_features)
    engineered.to_csv(OUTPUT_DIR / "engineered_features.csv", index=False)

    label_summary = summarize_labels(targets)
    label_summary.to_csv(OUTPUT_DIR / "label_summary.csv", index=False)

    reference_summary = summarize_reference_file()

    metrics, artifacts = evaluate_models(engineered, targets)

    plot_data_overview(targets, label_summary)
    plot_metric_comparison(metrics)
    plot_precision_recall_curves(artifacts)
    plot_confusion_matrices(artifacts)
    plot_feature_importance(artifacts)
    plot_temporal_probability_trace(artifacts)

    summary_payload = {
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "n_frames": int(len(raw_features)),
        "n_engineered_features": int(engineered.shape[1]),
        "label_summary": label_summary.to_dict(orient="records"),
        "reference_summary": reference_summary.to_dict(orient="records"),
        "best_models": (
            metrics.sort_values(["label", "protocol", "average_precision"], ascending=[True, True, False])
            .groupby(["label", "protocol"], as_index=False)
            .first()
            .to_dict(orient="records")
        ),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)


if __name__ == "__main__":
    main()
