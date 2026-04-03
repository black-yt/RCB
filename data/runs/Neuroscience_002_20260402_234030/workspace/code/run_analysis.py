from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
FEATURE_COLS = [str(i) for i in range(20)]
CAT_COLS = ["degradation"]
TARGET_COL = "label"


def ensure_dirs() -> Tuple[Path, Path]:
    outputs_dir = Path("outputs")
    images_dir = Path("report/images")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir, images_dir


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv("data/train_simulated.csv")
    test_df = pd.read_csv("data/test_simulated.csv")
    train_df[TARGET_COL] = train_df[TARGET_COL].astype(int)
    test_df[TARGET_COL] = test_df[TARGET_COL].astype(int)
    return train_df, test_df


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    num_steps: List[Tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(num_steps)
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, FEATURE_COLS),
            ("cat", categorical_pipe, CAT_COLS),
        ]
    )


def build_models(pos_weight: float) -> Dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=True)),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        class_weight={0: 1.0, 1: pos_weight},
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocess", make_preprocessor(scale_numeric=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=6,
                        learning_rate=0.05,
                        max_iter=120,
                        l2_regularization=0.1,
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


def summarize_dataframe(df: pd.DataFrame, name: str) -> Dict[str, object]:
    return {
        "name": name,
        "shape": list(df.shape),
        "positive_rate": float(df[TARGET_COL].mean()),
        "label_counts": {str(k): int(v) for k, v in df[TARGET_COL].value_counts().sort_index().items()},
        "degradation_counts": {str(k): int(v) for k, v in df["degradation"].value_counts().items()},
        "missing_values": int(df.isna().sum().sum()),
    }


def plot_class_and_degradation(train_df: pd.DataFrame, test_df: pd.DataFrame, images_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    label_df = pd.DataFrame(
        {
            "dataset": ["train", "train", "test", "test"],
            "label": [0, 1, 0, 1],
            "count": [
                (train_df[TARGET_COL] == 0).sum(),
                (train_df[TARGET_COL] == 1).sum(),
                (test_df[TARGET_COL] == 0).sum(),
                (test_df[TARGET_COL] == 1).sum(),
            ],
        }
    )
    sns.barplot(data=label_df, x="dataset", y="count", hue="label", ax=axes[0], palette="Blues")
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Count")

    deg_df = pd.concat([train_df.assign(dataset="train"), test_df.assign(dataset="test")], ignore_index=True)
    sns.countplot(data=deg_df, x="degradation", hue="dataset", ax=axes[1])
    axes[1].set_title("Degradation distribution")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(images_dir / "data_overview_class_degradation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_feature_distributions(train_df: pd.DataFrame, images_dir: Path) -> None:
    sample = train_df.sample(n=min(3000, len(train_df)), random_state=SEED)
    selected = FEATURE_COLS[:8]
    long_df = sample.melt(id_vars=[TARGET_COL], value_vars=selected, var_name="feature", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long_df, x="feature", y="value", hue=TARGET_COL, ax=ax, showfliers=False)
    ax.set_title("Feature distributions for selected inputs")
    plt.tight_layout()
    fig.savefig(images_dir / "feature_distributions_selected.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(train_df: pd.DataFrame, images_dir: Path) -> None:
    corr = train_df[FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    plt.tight_layout()
    fig.savefig(images_dir / "feature_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def cross_validate_models(train_df: pd.DataFrame, models: Dict[str, Pipeline], outputs_dir: Path) -> pd.DataFrame:
    X = train_df[FEATURE_COLS + CAT_COLS]
    y = train_df[TARGET_COL].to_numpy()
    strat = train_df["degradation"].astype(str) + "__" + train_df[TARGET_COL].astype(str)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    rows = []
    for model_name, model in models.items():
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, strat), start=1):
            current_model = clone(model)
            current_model.fit(X.iloc[train_idx], y[train_idx])
            y_prob = current_model.predict_proba(X.iloc[val_idx])[:, 1]
            threshold = best_f1_threshold(y[val_idx], y_prob)
            rows.append({"model": model_name, "fold": fold, **evaluate_predictions(y[val_idx], y_prob, threshold)})
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(outputs_dir / "cv_metrics.csv", index=False)
    cv_df.groupby("model").agg(["mean", "std"]).to_csv(outputs_dir / "cv_metrics_summary.csv")
    return cv_df


def plot_cv_performance(cv_df: pd.DataFrame, images_dir: Path) -> None:
    metric_order = ["roc_auc", "average_precision", "balanced_accuracy", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), metric_order):
        sns.boxplot(data=cv_df, x="model", y=metric, ax=ax)
        sns.stripplot(data=cv_df, x="model", y=metric, ax=ax, color="black", alpha=0.6)
        ax.set_title(f"Cross-validation {metric}")
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    fig.savefig(images_dir / "model_comparison_cv.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def choose_best_model(cv_df: pd.DataFrame) -> str:
    summary = cv_df.groupby("model")[["average_precision", "roc_auc", "balanced_accuracy", "f1"]].mean()
    summary = summary.sort_values(["average_precision", "roc_auc", "balanced_accuracy", "f1"], ascending=False)
    return str(summary.index[0])


def fit_best_model(train_df: pd.DataFrame, model: Pipeline) -> Tuple[Pipeline, float, Dict[str, float]]:
    strat = train_df["degradation"].astype(str) + "__" + train_df[TARGET_COL].astype(str)
    train_sub, valid_sub = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=strat)
    model.fit(train_sub[FEATURE_COLS + CAT_COLS], train_sub[TARGET_COL].to_numpy())
    valid_prob = model.predict_proba(valid_sub[FEATURE_COLS + CAT_COLS])[:, 1]
    threshold = best_f1_threshold(valid_sub[TARGET_COL].to_numpy(), valid_prob)
    return model, threshold, evaluate_predictions(valid_sub[TARGET_COL].to_numpy(), valid_prob, threshold)


def evaluate_by_degradation(df: pd.DataFrame, probs: np.ndarray, threshold: float) -> pd.DataFrame:
    rows = []
    for deg, sub in df.groupby("degradation"):
        y_true = sub[TARGET_COL].to_numpy()
        y_prob = probs[sub.index.to_numpy()]
        rows.append({"degradation": deg, "n": int(len(sub)), **evaluate_predictions(y_true, y_prob, threshold)})
    return pd.DataFrame(rows)


def save_predictions(df: pd.DataFrame, probs: np.ndarray, threshold: float, path: Path) -> None:
    out = df.copy()
    out["predicted_probability"] = probs
    out["predicted_label"] = (probs >= threshold).astype(int)
    out.to_csv(path, index=False)


def plot_roc_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, images_dir: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"ROC AUC = {roc_auc_score(y_true, y_prob):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC curve on held-out test set")
    axes[0].legend()
    axes[1].plot(recall, precision, label=f"AP = {average_precision_score(y_true, y_prob):.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall curve on held-out test set")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(images_dir / "test_roc_pr_curves.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float, images_dir: Path) -> None:
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix on test set")
    plt.tight_layout()
    fig.savefig(images_dir / "test_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_degradation_performance(deg_df: pd.DataFrame, images_dir: Path) -> None:
    melted = deg_df.melt(
        id_vars=["degradation", "n"],
        value_vars=["roc_auc", "average_precision", "balanced_accuracy", "f1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="degradation", y="value", hue="metric", ax=ax)
    ax.set_title("Test performance by degradation type")
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(images_dir / "degradation_performance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, images_dir: Path) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(mean_pred, frac_pos, marker="o")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve")
    plt.tight_layout()
    fig.savefig(images_dir / "calibration_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_feature_summary(train_df: pd.DataFrame, outputs_dir: Path, images_dir: Path) -> pd.DataFrame:
    summary_rows = []
    for feature in FEATURE_COLS:
        pos = train_df.loc[train_df[TARGET_COL] == 1, feature]
        neg = train_df.loc[train_df[TARGET_COL] == 0, feature]
        effect = float(pos.mean() - neg.mean())
        pooled = float(np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)) if len(pos) > 1 and len(neg) > 1 else 0.0
        cohen_d = effect / pooled if pooled > 1e-12 else 0.0
        summary_rows.append(
            {
                "feature": feature,
                "positive_mean": float(pos.mean()),
                "negative_mean": float(neg.mean()),
                "mean_difference": effect,
                "cohen_d": float(cohen_d),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("cohen_d", key=np.abs, ascending=False)
    summary_df.to_csv(outputs_dir / "feature_effect_summary.csv", index=False)
    top_df = summary_df.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["crimson" if x < 0 else "steelblue" for x in top_df["cohen_d"]]
    ax.barh(top_df["feature"], top_df["cohen_d"], color=colors)
    ax.set_xlabel("Cohen's d (positive vs negative class)")
    ax.set_title("Top discriminative features by effect size")
    plt.tight_layout()
    fig.savefig(images_dir / "feature_effect_sizes_top15.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return summary_df


def main() -> None:
    sns.set_theme(style="whitegrid")
    outputs_dir, images_dir = ensure_dirs()
    train_df, test_df = load_data()

    with open(outputs_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump({"train": summarize_dataframe(train_df, "train"), "test": summarize_dataframe(test_df, "test")}, f, indent=2)

    plot_class_and_degradation(train_df, test_df, images_dir)
    plot_feature_distributions(train_df, images_dir)
    plot_correlation_heatmap(train_df, images_dir)

    pos_weight = float((train_df[TARGET_COL] == 0).sum() / max((train_df[TARGET_COL] == 1).sum(), 1))
    models = build_models(pos_weight)
    cv_df = cross_validate_models(train_df, models, outputs_dir)
    plot_cv_performance(cv_df, images_dir)
    best_model_name = choose_best_model(cv_df)

    fitted_model, threshold, valid_metrics = fit_best_model(train_df, models[best_model_name])
    fitted_model.fit(train_df[FEATURE_COLS + CAT_COLS], train_df[TARGET_COL].to_numpy())

    y_test = test_df[TARGET_COL].to_numpy()
    test_prob = fitted_model.predict_proba(test_df[FEATURE_COLS + CAT_COLS])[:, 1]
    test_metrics = evaluate_predictions(y_test, test_prob, threshold)

    deg_df = evaluate_by_degradation(test_df, test_prob, threshold)
    deg_df.to_csv(outputs_dir / "degradation_metrics.csv", index=False)
    save_predictions(test_df, test_prob, threshold, outputs_dir / "test_predictions.csv")

    plot_roc_pr_curves(y_test, test_prob, images_dir)
    plot_confusion(y_test, test_prob, threshold, images_dir)
    plot_degradation_performance(deg_df, images_dir)
    plot_calibration(y_test, test_prob, images_dir)
    feature_summary_df = compute_feature_summary(train_df, outputs_dir, images_dir)

    final_report = {
        "best_model": best_model_name,
        "selected_threshold": threshold,
        "validation_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "classification_report": classification_report(y_test, (test_prob >= threshold).astype(int), output_dict=True, zero_division=0),
        "top_effect_features": feature_summary_df.head(10).to_dict(orient="records"),
    }
    with open(outputs_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    with open(outputs_dir / "best_model.pkl", "wb") as f:
        pickle.dump({"model": fitted_model, "threshold": threshold, "best_model_name": best_model_name}, f)

    print(json.dumps({"status": "ok", "best_model": best_model_name, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
