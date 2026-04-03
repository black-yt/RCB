from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
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


RANDOM_STATE = 42
TARGETS = ["Attack", "Sniffing"]
WORKSPACE = Path(__file__).resolve().parents[1]
DATA_DIR = WORKSPACE / "data"
OUTPUTS_DIR = WORKSPACE / "outputs"
IMAGES_DIR = WORKSPACE / "report" / "images"


sns.set_theme(style="whitegrid", context="talk")


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = pd.read_csv(DATA_DIR / "Together_1_features_extracted.csv")
    targets = pd.read_csv(DATA_DIR / "Together_1_targets_inserted.csv")
    reference = pd.read_csv(DATA_DIR / "Together_1_machine_results_reference.csv")
    return features, targets, reference


def align_features_and_targets(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    non_target_columns = [c for c in targets.columns if c not in TARGETS]
    shared = [c for c in features.columns if c in non_target_columns]
    if len(shared) != len(features.columns):
        missing = sorted(set(features.columns) - set(shared))
        raise ValueError(f"Some feature columns are missing in target file: {missing}")

    aligned = targets[shared + TARGETS].copy()
    feature_values = features[shared].copy()

    numeric_shared = feature_values.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_shared:
        max_diff = (feature_values[numeric_shared] - aligned[numeric_shared]).abs().max().max()
    else:
        max_diff = 0.0

    audit = {
        "feature_shape": list(features.shape),
        "target_shape": list(targets.shape),
        "shared_feature_columns": len(shared),
        "max_abs_difference_between_feature_and_target_tables": float(max_diff),
    }
    with open(OUTPUTS_DIR / "alignment_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    return aligned


def save_dataset_overview(df: pd.DataFrame) -> None:
    feature_cols = [c for c in df.columns if c not in TARGETS]
    feature_df = df[feature_cols]

    overview_rows: List[Dict[str, object]] = []
    for target in TARGETS:
        counts = df[target].value_counts().sort_index()
        overview_rows.append(
            {
                "target": target,
                "n_frames": int(len(df)),
                "n_positive": int(counts.get(1, 0)),
                "n_negative": int(counts.get(0, 0)),
                "positive_rate": float(df[target].mean()),
            }
        )

    dataset_summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(df)),
                "n_feature_columns": int(len(feature_cols)),
                "total_missing_values": int(feature_df.isna().sum().sum()),
                "columns_with_missing_values": int((feature_df.isna().sum() > 0).sum()),
            }
        ]
    )
    dataset_summary.to_csv(OUTPUTS_DIR / "dataset_summary.csv", index=False)
    pd.DataFrame(overview_rows).to_csv(OUTPUTS_DIR / "label_summary.csv", index=False)

    missingness = feature_df.isna().mean().sort_values(ascending=False).rename("missing_fraction")
    missingness.to_csv(OUTPUTS_DIR / "feature_missingness.csv", header=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    label_df = pd.DataFrame(overview_rows)
    sns.barplot(data=label_df, x="target", y="positive_rate", ax=axes[0], palette="deep")
    axes[0].set_title("Positive label prevalence")
    axes[0].set_ylabel("Positive fraction")
    axes[0].set_xlabel("")
    axes[0].set_ylim(0, max(0.4, label_df["positive_rate"].max() * 1.2))
    for idx, row in label_df.iterrows():
        axes[0].text(idx, row["positive_rate"] + 0.01, f"{row['n_positive']}/{row['n_frames']}", ha="center", va="bottom", fontsize=10)

    top_missing = missingness.head(15).reset_index()
    top_missing.columns = ["feature", "missing_fraction"]
    sns.barplot(data=top_missing, x="missing_fraction", y="feature", ax=axes[1], palette="mako")
    axes[1].set_title("Top feature missingness")
    axes[1].set_xlabel("Missing fraction")
    axes[1].set_ylabel("")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "data_overview.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    corr_features = feature_df.select_dtypes(include=[np.number]).copy()
    if corr_features.shape[1] > 1:
        variance = corr_features.var().sort_values(ascending=False)
        top_cols = variance.head(min(12, len(variance))).index.tolist()
        corr = corr_features[top_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
        plt.title("Correlation heatmap of high-variance input features")
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "feature_correlation_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close()


def choose_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        precision[:-1] + recall[:-1],
        out=np.zeros_like(thresholds),
        where=(precision[:-1] + recall[:-1]) != 0,
    )
    if len(thresholds) == 0:
        return {"best_threshold": 0.5, "best_f1_from_pr_curve": 0.0}
    best_idx = int(np.nanargmax(f1_scores))
    return {
        "best_threshold": float(thresholds[best_idx]),
        "best_f1_from_pr_curve": float(f1_scores[best_idx]),
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }
    return metrics


def train_and_evaluate(df: pd.DataFrame, reference: pd.DataFrame, target: str) -> Dict[str, object]:
    feature_cols = [c for c in df.columns if c not in TARGETS]
    X = df[feature_cols].copy()
    y = df[target].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=RANDOM_STATE,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    train_prob = pipeline.predict_proba(X_train)[:, 1]
    test_prob = pipeline.predict_proba(X_test)[:, 1]

    threshold_info = choose_threshold(y_train.to_numpy(), train_prob)
    threshold = threshold_info["best_threshold"]

    train_metrics = classification_metrics(y_train.to_numpy(), train_prob, threshold)
    test_metrics = classification_metrics(y_test.to_numpy(), test_prob, threshold)

    metrics_df = pd.DataFrame(
        [
            {"split": "train", "target": target, **train_metrics},
            {"split": "test", "target": target, **test_metrics},
        ]
    )
    metrics_df.to_csv(OUTPUTS_DIR / f"metrics_{target.lower()}.csv", index=False)

    test_pred = (test_prob >= threshold).astype(int)
    pred_df = X_test.copy()
    pred_df[target] = y_test.values
    pred_df[f"predicted_{target}"] = test_pred
    pred_df[f"probability_{target}"] = test_prob
    pred_df.to_csv(OUTPUTS_DIR / f"test_predictions_{target.lower()}.csv", index=False)

    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])
    cm_df.to_csv(OUTPUTS_DIR / f"confusion_matrix_{target.lower()}.csv")

    report = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(OUTPUTS_DIR / f"classification_report_{target.lower()}.csv")

    precision, recall, thresholds = precision_recall_curve(y_test, test_prob)
    pr_df = pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": np.append(thresholds, np.nan),
        }
    )
    pr_df.to_csv(OUTPUTS_DIR / f"precision_recall_{target.lower()}.csv", index=False)

    model = pipeline.named_steps["model"]
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "gini_importance": model.feature_importances_,
        }
    ).sort_values("gini_importance", ascending=False)

    perm = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="average_precision",
        n_jobs=-1,
    )
    perm_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)

    feature_importance = importances.merge(perm_df, on="feature", how="left")
    feature_importance.to_csv(OUTPUTS_DIR / f"feature_importance_{target.lower()}.csv", index=False)

    comparison = {}
    prob_col = f"Probability_{target}"
    label_col = target
    if prob_col in reference.columns and label_col in reference.columns:
        ref = reference[[prob_col, label_col]].dropna().copy()
        ref_y = ref[label_col].astype(int)
        ref_prob = ref[prob_col].astype(float).clip(0, 1)
        ref_pred = (ref_prob >= 0.5).astype(int)
        comparison = {
            "reference_rows": int(len(ref)),
            "reference_positive_rate": float(ref_y.mean()),
            "reference_average_precision": float(average_precision_score(ref_y, ref_prob)) if ref_y.nunique() > 1 else np.nan,
            "reference_roc_auc": float(roc_auc_score(ref_y, ref_prob)) if ref_y.nunique() > 1 else np.nan,
            "reference_precision_at_0_5": float(precision_score(ref_y, ref_pred, zero_division=0)),
            "reference_recall_at_0_5": float(recall_score(ref_y, ref_pred, zero_division=0)),
            "reference_f1_at_0_5": float(f1_score(ref_y, ref_pred, zero_division=0)),
        }

    with open(OUTPUTS_DIR / f"reference_comparison_{target.lower()}.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    plot_target_figures(target, y_test.to_numpy(), test_prob, cm_df, feature_importance)

    return {
        "target": target,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "selected_threshold": float(threshold),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "reference_metrics": comparison,
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }


def plot_target_figures(
    target: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cm_df: pd.DataFrame,
    feature_importance: pd.DataFrame,
) -> None:
    target_lower = target.lower()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0])
    axes[0].set_title(f"{target} confusion matrix")
    axes[0].set_xlabel("Predicted label")
    axes[0].set_ylabel("True label")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    axes[1].plot(recall, precision, linewidth=2.5, label=f"AP = {ap:.3f}")
    positive_rate = y_true.mean()
    axes[1].hlines(positive_rate, 0, 1, colors="gray", linestyles="--", label=f"Baseline = {positive_rate:.3f}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{target} precision-recall curve")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / f"{target_lower}_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    top_features = feature_importance.head(15).iloc[::-1]
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_features, x="permutation_importance_mean", y="feature", palette="viridis")
    plt.title(f"{target} top permutation feature importances")
    plt.xlabel("Mean permutation importance (AP decrease proxy)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{target_lower}_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close()


def save_cross_target_summary(results: List[Dict[str, object]]) -> None:
    rows = []
    for result in results:
        row = {
            "target": result["target"],
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "train_positive_rate": result["train_positive_rate"],
            "test_positive_rate": result["test_positive_rate"],
            "selected_threshold": result["selected_threshold"],
        }
        for k, v in result["test_metrics"].items():
            row[f"test_{k}"] = v
        for k, v in result["train_metrics"].items():
            row[f"train_{k}"] = v
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(OUTPUTS_DIR / "model_summary.csv", index=False)

    score_cols = ["test_average_precision", "test_f1", "test_precision", "test_recall", "test_balanced_accuracy", "test_roc_auc"]
    plot_df = summary_df[["target"] + score_cols].melt(id_vars="target", var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=plot_df, x="metric", y="value", hue="target")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1.05)
    plt.title("Held-out classifier performance summary")
    plt.xlabel("")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "model_performance_summary.png", dpi=200, bbox_inches="tight")
    plt.close()

    with open(OUTPUTS_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main() -> None:
    ensure_dirs()
    features, targets, reference = load_data()
    aligned = align_features_and_targets(features, targets)
    save_dataset_overview(aligned)

    results = []
    for target in TARGETS:
        results.append(train_and_evaluate(aligned, reference, target))

    save_cross_target_summary(results)


if __name__ == "__main__":
    main()
