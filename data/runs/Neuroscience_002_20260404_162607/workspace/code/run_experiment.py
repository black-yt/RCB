import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
N_SPLITS = 3
N_REPEATS = 1
POSITIVE_CLASS = 1

sns.set_theme(style="whitegrid")


def ensure_dirs():
    for path in [Path("outputs"), Path("report/images")]:
        path.mkdir(parents=True, exist_ok=True)


def load_data():
    train = pd.read_csv("data/train_simulated.csv")
    test = pd.read_csv("data/test_simulated.csv")
    feature_cols = [str(i) for i in range(20)]
    target_col = "label"
    cat_cols = ["degradation"]
    return train, test, feature_cols, cat_cols, target_col


def summarize_dataframe(df: pd.DataFrame, name: str):
    summary = {
        "split": name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "label_rate": float(df["label"].mean()),
        "missing_values_total": int(df.isna().sum().sum()),
        "degradation_counts": df["degradation"].value_counts().to_dict(),
    }
    return summary


def make_preprocessor(feature_cols, cat_cols):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, feature_cols),
        ("cat", categorical, cat_cols),
    ])


def get_models(preprocessor):
    return {
        "logistic_regression": Pipeline([
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=40,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("prep", preprocessor),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=120,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]),
    }


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": fp / (fp + tn),
        "fnr": fn / (fn + tp),
    }


def threshold_for_precision(y_true, y_prob, target_precision=0.95):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    precision = precision[:-1]
    recall = recall[:-1]
    if len(thresholds) == 0:
        return 0.5, 0.0
    mask = precision >= target_precision
    if not np.any(mask):
        idx = np.argmax(precision)
        return float(thresholds[idx]), float(recall[idx])
    idx = np.argmax(recall[mask])
    chosen_threshold = thresholds[mask][idx]
    chosen_recall = recall[mask][idx]
    return float(chosen_threshold), float(chosen_recall)


def bootstrap_ci(y_true, y_prob, metric_fn, n_boot=300, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    values = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if len(np.unique(sample_y)) < 2:
            continue
        sample_p = y_prob[idx]
        values.append(metric_fn(sample_y, sample_p))
    return {
        "mean": float(np.mean(values)),
        "low": float(np.percentile(values, 2.5)),
        "high": float(np.percentile(values, 97.5)),
    }


def plot_data_overview(train, test, feature_cols):
    combined = pd.concat([
        train.assign(split="train"),
        test.assign(split="test"),
    ], ignore_index=True)

    plt.figure(figsize=(10, 5))
    counts = combined.groupby(["split", "degradation"]).size().reset_index(name="count")
    sns.barplot(data=counts, x="degradation", y="count", hue="split")
    plt.title("Sample counts by degradation and split")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("report/images/data_overview_degradation.png", dpi=200)
    plt.close()

    label_df = combined.groupby(["split", "degradation"])["label"].mean().reset_index(name="positive_rate")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=label_df, x="degradation", y="positive_rate", hue="split")
    plt.title("Positive merge rate by degradation and split")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig("report/images/data_overview_label_rate.png", dpi=200)
    plt.close()

    corr = train[feature_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Training feature correlation heatmap")
    plt.tight_layout()
    plt.savefig("report/images/feature_correlation_heatmap.png", dpi=200)
    plt.close()

    feature_summary = train[feature_cols].melt(var_name="feature", value_name="value")
    top_var = train[feature_cols].var().sort_values(ascending=False).head(8).index.tolist()
    plt.figure(figsize=(12, 5))
    sns.boxplot(data=train[top_var].melt(var_name="feature", value_name="value"), x="feature", y="value")
    plt.title("Distribution of high-variance features in training set")
    plt.tight_layout()
    plt.savefig("report/images/feature_distributions.png", dpi=200)
    plt.close()


def plot_cv_curves(cv_predictions, y_true):
    plt.figure(figsize=(7, 6))
    for model_name, y_prob in cv_predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUROC={auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Cross-validated ROC curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/cv_roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    prevalence = np.mean(y_true)
    for model_name, y_prob in cv_predictions.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{model_name} (AUPRC={ap:.3f})")
    plt.axhline(prevalence, linestyle="--", color="black", linewidth=1, label=f"prevalence={prevalence:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Cross-validated precision-recall curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/cv_pr_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 6))
    for model_name, y_prob in cv_predictions.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive fraction")
    plt.title("Cross-validated calibration curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("report/images/calibration_curves.png", dpi=200)
    plt.close()


def plot_test_results(metrics_df):
    long_df = metrics_df.melt(id_vars=["model", "split"], value_vars=["auroc", "auprc", "f1", "balanced_accuracy", "mcc"], var_name="metric", value_name="value")
    plt.figure(figsize=(10, 5))
    sns.barplot(data=long_df[long_df["split"] == "test"], x="metric", y="value", hue="model")
    plt.title("Test-set performance across models")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("report/images/test_metric_comparison.png", dpi=200)
    plt.close()


def repeated_cv_predict_proba(model, X, y, cv):
    y = np.asarray(y)
    prob_sum = np.zeros(len(y), dtype=float)
    prob_count = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        fitted_model = clone(model)
        fitted_model.fit(X.iloc[train_idx], y[train_idx])
        fold_prob = fitted_model.predict_proba(X.iloc[test_idx])[:, 1]
        prob_sum[test_idx] += fold_prob
        prob_count[test_idx] += 1

    if np.any(prob_count == 0):
        raise ValueError("Some samples did not receive any out-of-fold predictions.")

    return prob_sum / prob_count


def main():
    ensure_dirs()
    train, test, feature_cols, cat_cols, target_col = load_data()

    train_summary = summarize_dataframe(train, "train")
    test_summary = summarize_dataframe(test, "test")
    with open("outputs/data_summary.json", "w") as f:
        json.dump({"train": train_summary, "test": test_summary}, f, indent=2)

    plot_data_overview(train, test, feature_cols)

    X_train = train[feature_cols + cat_cols]
    y_train = train[target_col].astype(int).to_numpy()
    X_test = test[feature_cols + cat_cols]
    y_test = test[target_col].astype(int).to_numpy()

    preprocessor = make_preprocessor(feature_cols, cat_cols)
    models = get_models(preprocessor)
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_SEED)

    results = []
    cv_predictions = {}
    trained_models = {}
    threshold_summary = []

    for model_name, model in models.items():
        y_cv_prob = repeated_cv_predict_proba(model, X_train, y_train, cv)
        cv_predictions[model_name] = y_cv_prob
        cv_metrics = compute_metrics(y_train, y_cv_prob, threshold=0.5)
        ci_auroc = bootstrap_ci(y_train, y_cv_prob, roc_auc_score)
        ci_auprc = bootstrap_ci(y_train, y_cv_prob, average_precision_score)
        results.append({"model": model_name, "split": "cv_train", **cv_metrics, "auroc_ci_low": ci_auroc["low"], "auroc_ci_high": ci_auroc["high"], "auprc_ci_low": ci_auprc["low"], "auprc_ci_high": ci_auprc["high"]})

        precision_threshold, precision_recall = threshold_for_precision(y_train, y_cv_prob, target_precision=0.95)
        threshold_summary.append({
            "model": model_name,
            "target_precision": 0.95,
            "selected_threshold": precision_threshold,
            "cv_recall_at_target_precision": precision_recall,
        })

        model.fit(X_train, y_train)
        trained_models[model_name] = model
        y_test_prob = model.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, y_test_prob, threshold=0.5)
        results.append({"model": model_name, "split": "test", **test_metrics})

        high_prec_metrics = compute_metrics(y_test, y_test_prob, threshold=precision_threshold)
        results.append({"model": f"{model_name}_high_precision", "split": "test", **high_prec_metrics})

        joblib.dump(model, f"outputs/{model_name}.joblib")

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("outputs/model_metrics.csv", index=False)
    pd.DataFrame(threshold_summary).to_csv("outputs/high_precision_thresholds.csv", index=False)

    plot_cv_curves(cv_predictions, y_train)
    plot_test_results(metrics_df[metrics_df["model"].isin(models.keys())])

    best_model_name = metrics_df[(metrics_df["split"] == "test") & (metrics_df["model"].isin(models.keys()))].sort_values(["auprc", "auroc"], ascending=False).iloc[0]["model"]
    best_model = trained_models[best_model_name]

    perm = permutation_importance(best_model, X_test, y_test, n_repeats=8, random_state=RANDOM_SEED, n_jobs=-1, scoring="average_precision")
    feature_names = feature_cols + [f"degradation_{cat}" for cat in sorted(train['degradation'].unique())]
    perm_df = pd.DataFrame({"feature": feature_names[: len(perm.importances_mean)], "importance_mean": perm.importances_mean, "importance_std": perm.importances_std})
    perm_df = perm_df.sort_values("importance_mean", ascending=False)
    perm_df.to_csv("outputs/permutation_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=perm_df.head(12), x="importance_mean", y="feature", orient="h")
    plt.title(f"Top permutation importances for {best_model_name}")
    plt.xlabel("Decrease in average precision")
    plt.tight_layout()
    plt.savefig("report/images/permutation_importance.png", dpi=200)
    plt.close()

    by_deg_records = []
    y_best = best_model.predict_proba(X_test)[:, 1]
    for deg, subset in test.groupby("degradation"):
        idx = subset.index.to_numpy()
        y_subset = y_test[test.index.get_indexer(idx)]
        p_subset = y_best[test.index.get_indexer(idx)]
        metrics = compute_metrics(y_subset, p_subset, threshold=0.5)
        by_deg_records.append({"degradation": deg, **metrics})
    by_deg_df = pd.DataFrame(by_deg_records)
    by_deg_df.to_csv("outputs/best_model_by_degradation.csv", index=False)

    plt.figure(figsize=(9, 5))
    by_deg_long = by_deg_df.melt(id_vars=["degradation"], value_vars=["auroc", "auprc", "f1", "balanced_accuracy"], var_name="metric", value_name="value")
    sns.barplot(data=by_deg_long, x="metric", y="value", hue="degradation")
    plt.title(f"{best_model_name} performance by degradation on test set")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("report/images/performance_by_degradation.png", dpi=200)
    plt.close()

    summary = {
        "best_model": best_model_name,
        "train_summary": train_summary,
        "test_summary": test_summary,
    }
    with open("outputs/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
