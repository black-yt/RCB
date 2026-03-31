import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

SEED = 42
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
IMG_DIR = Path("report/images")
CODE_DIR = Path("code")

FEATURE_COLS = [
    "Nucleophilic-HEA",
    "Hydrophobic-BA",
    "Acidic-CBEA",
    "Cationic-ATAC",
    "Aromatic-PEA",
    "Amide-AAm",
]
PRIMARY_TARGET = "Glass (kPa)_10s"
SECONDARY_TARGET = "Steel (kPa)_10s"
HIGH_STRENGTH_THRESHOLD_KPA = 100.0


def ensure_dirs():
    for d in [OUTPUT_DIR, IMG_DIR, CODE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def load_excel_sheets(path: Path):
    xl = pd.ExcelFile(path)
    return {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in xl.sheet_names}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in FEATURE_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    feats = out[FEATURE_COLS].astype(float)
    out["feature_sum"] = feats.sum(axis=1)
    out["max_monomer_fraction"] = feats.max(axis=1)
    out["min_monomer_fraction"] = feats.min(axis=1)
    out["hydrophilic_fraction"] = feats[["Nucleophilic-HEA", "Acidic-CBEA", "Cationic-ATAC", "Amide-AAm"]].sum(axis=1)
    out["hydrophobic_aromatic_fraction"] = feats[["Hydrophobic-BA", "Aromatic-PEA"]].sum(axis=1)
    out["charge_balance"] = feats["Cationic-ATAC"] - feats["Acidic-CBEA"]
    out["charge_magnitude"] = feats[["Cationic-ATAC", "Acidic-CBEA"]].sum(axis=1)
    out["nucleophile_aromatic_product"] = feats["Nucleophilic-HEA"] * feats["Aromatic-PEA"]
    out["hydrophobic_cationic_product"] = feats["Hydrophobic-BA"] * feats["Cationic-ATAC"]
    out["composition_entropy"] = -(feats.clip(lower=1e-9) * np.log(feats.clip(lower=1e-9))).sum(axis=1)
    return out


def infer_training_data() -> pd.DataFrame:
    df = pd.read_excel(DATA_DIR / "184_verified_Original Data_ML_20230926.xlsx")
    df = clean_columns(df)
    df = df.dropna(subset=FEATURE_COLS + [PRIMARY_TARGET]).reset_index(drop=True)
    df = add_engineered_features(df)
    df["high_strength"] = (df[PRIMARY_TARGET] >= HIGH_STRENGTH_THRESHOLD_KPA).astype(int)
    return df


def summarize_all_datasets():
    rows = []
    schema = {}
    for fp in sorted(DATA_DIR.glob("*.xlsx")):
        sheets = load_excel_sheets(fp)
        schema[fp.name] = {}
        for sheet, df in sheets.items():
            cdf = clean_columns(df)
            schema[fp.name][sheet] = {
                "n_rows": int(cdf.shape[0]),
                "n_cols": int(cdf.shape[1]),
                "columns": [str(c) for c in cdf.columns],
            }
            numeric = cdf.select_dtypes(include=[np.number])
            rows.append(
                {
                    "file": fp.name,
                    "sheet": sheet,
                    "n_rows": cdf.shape[0],
                    "n_cols": cdf.shape[1],
                    "numeric_cols": numeric.shape[1],
                    "missing_cells": int(cdf.isna().sum().sum()),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "data_overview.csv", index=False)
    with open(OUTPUT_DIR / "schema_summary.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)


def save_training_summary(df: pd.DataFrame):
    desc = df[[PRIMARY_TARGET, SECONDARY_TARGET, *FEATURE_COLS]].describe().T
    desc.to_csv(OUTPUT_DIR / "training_summary.csv")

    issues = {
        "n_samples": int(df.shape[0]),
        "duplicates_all_columns": int(df.duplicated().sum()),
        "duplicates_features_only": int(df.duplicated(subset=FEATURE_COLS).sum()),
        "feature_sum_min": float(df["feature_sum"].min()),
        "feature_sum_max": float(df["feature_sum"].max()),
        "high_strength_count": int(df["high_strength"].sum()),
        "high_strength_rate": float(df["high_strength"].mean()),
    }
    with open(OUTPUT_DIR / "data_quality.json", "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)


def plot_target_distribution(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[PRIMARY_TARGET], kde=True, ax=axes[0], color="#4477AA")
    axes[0].axvline(HIGH_STRENGTH_THRESHOLD_KPA, color="red", linestyle="--", label=">1 MPa")
    axes[0].set_title("Primary target distribution")
    axes[0].set_xlabel("Glass adhesion at 60 s (kPa)")
    axes[0].legend()

    sns.boxplot(data=df[[PRIMARY_TARGET, SECONDARY_TARGET]], ax=axes[1], palette="Set2")
    axes[1].axhline(HIGH_STRENGTH_THRESHOLD_KPA, color="red", linestyle="--")
    axes[1].set_ylabel("Adhesive strength (kPa)")
    axes[1].set_title("Adhesion endpoints")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_correlation(df: pd.DataFrame):
    corr_cols = FEATURE_COLS + [
        PRIMARY_TARGET,
        SECONDARY_TARGET,
        "Modulus (kPa)",
        "Q",
        "XlogP3",
        "charge_balance",
        "composition_entropy",
    ]
    use_cols = [c for c in corr_cols if c in df.columns]
    corr_input = df[use_cols].dropna(axis=1, how="all")
    corr = corr_input.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f", ax=ax)
    ax.set_title("Feature-target correlation matrix")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "feature_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_formulation_projection(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(X)
    pca_df = pd.DataFrame({
        "PC1": coords[:, 0],
        "PC2": coords[:, 1],
        PRIMARY_TARGET: df[PRIMARY_TARGET].values,
        "high_strength": df["high_strength"].map({0: "<1 MPa", 1: ">=1 MPa"}),
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue=PRIMARY_TARGET, style="high_strength", palette="viridis", s=90, ax=ax)
    ax.set_title(f"Formulation PCA (explained variance={pca.explained_variance_ratio_.sum():.2f})")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "formulation_projection.png", bbox_inches="tight")
    plt.close(fig)


def get_feature_matrix(df: pd.DataFrame):
    model_features = FEATURE_COLS + [
        "hydrophilic_fraction",
        "hydrophobic_aromatic_fraction",
        "charge_balance",
        "charge_magnitude",
        "nucleophile_aromatic_product",
        "hydrophobic_cationic_product",
        "composition_entropy",
        "max_monomer_fraction",
    ]
    return df[model_features], model_features


def regression_models():
    numeric_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return {
        "linear": Pipeline([("prep", numeric_processor), ("model", LinearRegression())]),
        "ridge": Pipeline([("prep", numeric_processor), ("model", Ridge(alpha=1.0, random_state=SEED))]),
        "pls": Pipeline([("prep", numeric_processor), ("model", PLSRegression(n_components=4))]),
        "rf": Pipeline([("prep", tree_processor), ("model", RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=SEED))]),
        "gbr": Pipeline([("prep", tree_processor), ("model", GradientBoostingRegressor(random_state=SEED))]),
    }


def classification_models():
    numeric_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    tree_processor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return {
        "logreg": Pipeline([("prep", numeric_processor), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED))]),
        "rf_clf": Pipeline([("prep", tree_processor), ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=SEED))]),
        "gbr_clf": Pipeline([("prep", tree_processor), ("model", GradientBoostingClassifier(random_state=SEED))]),
    }


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(X: pd.DataFrame, y: pd.Series):
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    preds_all = []
    for name, model in regression_models().items():
        fold_metrics = []
        pred_vec = pd.Series(index=y.index, dtype=float)
        for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
            est = clone(model)
            est.fit(X.iloc[tr], y.iloc[tr])
            pred = np.asarray(est.predict(X.iloc[te])).reshape(-1)
            pred_vec.iloc[te] = pred
            rmse = rmse_score(y.iloc[te], pred)
            mae = mean_absolute_error(y.iloc[te], pred)
            r2 = r2_score(y.iloc[te], pred)
            fold_metrics.append((rmse, mae, r2))
            preds_all.append(pd.DataFrame({
                "model": name,
                "fold": fold,
                "index": y.index[te],
                "observed": y.iloc[te].values,
                "predicted": pred,
            }))
        arr = np.array(fold_metrics)
        rows.append({
            "task": "regression",
            "model": name,
            "rmse_mean": arr[:, 0].mean(),
            "rmse_std": arr[:, 0].std(ddof=1),
            "mae_mean": arr[:, 1].mean(),
            "mae_std": arr[:, 1].std(ddof=1),
            "r2_mean": arr[:, 2].mean(),
            "r2_std": arr[:, 2].std(ddof=1),
        })
    return pd.DataFrame(rows), pd.concat(preds_all, ignore_index=True)


def evaluate_classification(X: pd.DataFrame, y: pd.Series):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    rows = []
    preds_all = []
    for name, model in classification_models().items():
        fold_metrics = []
        for fold, (tr, te) in enumerate(cv.split(X, y), start=1):
            est = clone(model)
            est.fit(X.iloc[tr], y.iloc[tr])
            prob = est.predict_proba(X.iloc[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            roc = roc_auc_score(y.iloc[te], prob)
            ap = average_precision_score(y.iloc[te], prob)
            acc = (pred == y.iloc[te].values).mean()
            recall = ((pred == 1) & (y.iloc[te].values == 1)).sum() / max((y.iloc[te].values == 1).sum(), 1)
            precision = ((pred == 1) & (y.iloc[te].values == 1)).sum() / max((pred == 1).sum(), 1)
            fold_metrics.append((roc, ap, acc, recall, precision))
            preds_all.append(pd.DataFrame({
                "task": "classification",
                "model": name,
                "fold": fold,
                "index": y.index[te],
                "observed": y.iloc[te].values,
                "predicted_prob": prob,
                "predicted_label": pred,
            }))
        arr = np.array(fold_metrics)
        rows.append({
            "task": "classification",
            "model": name,
            "roc_auc_mean": arr[:, 0].mean(),
            "roc_auc_std": arr[:, 0].std(ddof=1),
            "average_precision_mean": arr[:, 1].mean(),
            "average_precision_std": arr[:, 1].std(ddof=1),
            "accuracy_mean": arr[:, 2].mean(),
            "accuracy_std": arr[:, 2].std(ddof=1),
            "recall_mean": arr[:, 3].mean(),
            "recall_std": arr[:, 3].std(ddof=1),
            "precision_mean": arr[:, 4].mean(),
            "precision_std": arr[:, 4].std(ddof=1),
        })
    return pd.DataFrame(rows), pd.concat(preds_all, ignore_index=True)


def plot_model_comparison(reg_metrics: pd.DataFrame, clf_metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    reg_sorted = reg_metrics.sort_values("rmse_mean")
    sns.barplot(data=reg_sorted, x="model", y="rmse_mean", ax=axes[0], color="#4477AA")
    axes[0].errorbar(x=np.arange(reg_sorted.shape[0]), y=reg_sorted["rmse_mean"], yerr=reg_sorted["rmse_std"], fmt='none', c='black', capsize=4)
    axes[0].set_title("Regression CV RMSE")
    axes[0].set_ylabel("RMSE (kPa)")

    clf_sorted = clf_metrics.sort_values("roc_auc_mean", ascending=False)
    sns.barplot(data=clf_sorted, x="model", y="roc_auc_mean", ax=axes[1], color="#66AA55")
    axes[1].errorbar(x=np.arange(clf_sorted.shape[0]), y=clf_sorted["roc_auc_mean"], yerr=clf_sorted["roc_auc_std"], fmt='none', c='black', capsize=4)
    axes[1].set_title("High-strength classification ROC-AUC")
    axes[1].set_ylabel("ROC-AUC")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_observed(reg_preds: pd.DataFrame):
    best_model = reg_preds.groupby("model").apply(lambda d: rmse_score(d["observed"], d["predicted"])).sort_values().index[0]
    d = reg_preds[reg_preds["model"] == best_model].copy()
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=d, x="observed", y="predicted", ax=ax, color="#AA3377", s=70)
    lims = [min(d["observed"].min(), d["predicted"].min()), max(d["observed"].max(), d["predicted"].max())]
    ax.plot(lims, lims, linestyle="--", color="black")
    ax.axhline(HIGH_STRENGTH_THRESHOLD_KPA, linestyle=":", color="red")
    ax.axvline(HIGH_STRENGTH_THRESHOLD_KPA, linestyle=":", color="red")
    ax.set_title(f"Predicted vs observed ({best_model})")
    ax.set_xlabel("Observed Glass adhesion at 60 s (kPa)")
    ax.set_ylabel("Cross-validated prediction (kPa)")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "predicted_vs_observed.png", bbox_inches="tight")
    plt.close(fig)
    return best_model


def fit_feature_importance(X: pd.DataFrame, y: pd.Series, model_name: str):
    model = regression_models()[model_name]
    model.fit(X, y)
    perm = permutation_importance(model, X, y, scoring="neg_root_mean_squared_error", n_repeats=30, random_state=SEED)
    imp = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top = imp.head(10).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#CCBB44")
    ax.set_title("Permutation importance (top 10)")
    ax.set_xlabel("Increase in RMSE after permutation")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "high_strength_feature_profile.png", bbox_inches="tight")
    plt.close(fig)


def analyze_optimization(best_reg_model: str, best_clf_model: str, X_train: pd.DataFrame, y_train: pd.Series, y_bin: pd.Series):
    reg = regression_models()[best_reg_model]
    clf = classification_models()[best_clf_model]
    reg.fit(X_train, y_train)
    clf.fit(X_train, y_bin)

    summary_rows = []
    combined_rows = []
    for fname in ["ML_ei&pred_20240213.xlsx", "ML_ei&pred (1&2&3rounds)_20240408.xlsx"]:
        sheets = load_excel_sheets(DATA_DIR / fname)
        for sheet, df in sheets.items():
            df = clean_columns(df)
            if "Glass (kPa)_max" in df.columns:
                df["Glass (kPa)_max"] = pd.to_numeric(df["Glass (kPa)_max"], errors="coerce")
            df = add_engineered_features(df)
            df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
            X_opt = df[X_train.columns]
            df["predicted_glass_60s_kpa"] = np.asarray(reg.predict(X_opt)).reshape(-1)
            df["predicted_high_strength_prob"] = clf.predict_proba(X_opt)[:, 1]
            df["dataset"] = fname
            df["sheet"] = sheet
            summary_rows.append({
                "dataset": fname,
                "sheet": sheet,
                "n_candidates": int(df.shape[0]),
                "predicted_strength_mean": float(df["predicted_glass_60s_kpa"].mean()),
                "predicted_strength_max": float(df["predicted_glass_60s_kpa"].max()),
                "predicted_high_strength_rate_0.5": float((df["predicted_high_strength_prob"] >= 0.5).mean()),
                "observed_max_mean": float(df["Glass (kPa)_max"].mean()) if "Glass (kPa)_max" in df.columns else np.nan,
                "observed_max_over_1MPa_rate": float((df["Glass (kPa)_max"] >= HIGH_STRENGTH_THRESHOLD_KPA).mean()) if "Glass (kPa)_max" in df.columns else np.nan,
            })
            combined_rows.append(df)
    summary = pd.DataFrame(summary_rows)
    combined = pd.concat(combined_rows, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "optimization_summary.csv", index=False)
    combined.to_csv(OUTPUT_DIR / "optimization_scored_candidates.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=combined, x="sheet", y="predicted_glass_60s_kpa", hue="dataset", ax=ax)
    ax.axhline(HIGH_STRENGTH_THRESHOLD_KPA, linestyle="--", color="red")
    ax.set_title("Predicted optimization trajectory by acquisition strategy")
    ax.set_ylabel("Predicted Glass adhesion at 60 s (kPa)")
    fig.tight_layout()
    fig.savefig(IMG_DIR / "optimization_trajectory.png", bbox_inches="tight")
    plt.close(fig)

    tops = combined.sort_values("predicted_glass_60s_kpa", ascending=False).groupby(["dataset", "sheet"]).head(10)
    tops.to_csv(OUTPUT_DIR / "top_optimization_candidates.csv", index=False)


def statistical_tests(df: pd.DataFrame):
    hi = df.loc[df["high_strength"] == 1, FEATURE_COLS]
    lo = df.loc[df["high_strength"] == 0, FEATURE_COLS]
    rows = []
    pvals = []
    for col in FEATURE_COLS:
        stat, p = stats.mannwhitneyu(hi[col], lo[col], alternative="two-sided")
        pvals.append(p)
        rows.append({"feature": col, "u_stat": float(stat), "p_value": float(p), "high_mean": float(hi[col].mean()), "low_mean": float(lo[col].mean())})
    out = pd.DataFrame(rows)
    # Benjamini-Hochberg
    ranked = out["p_value"].rank(method="first").values
    m = len(out)
    out["p_fdr_bh"] = np.minimum(1, out["p_value"] * m / ranked)
    out.sort_values("p_value").to_csv(OUTPUT_DIR / "feature_group_tests.csv", index=False)


def main():
    ensure_dirs()
    summarize_all_datasets()
    df = infer_training_data()
    save_training_summary(df)
    plot_target_distribution(df)
    plot_feature_correlation(df)
    plot_formulation_projection(df)
    statistical_tests(df)

    X, feature_names = get_feature_matrix(df)
    y = df[PRIMARY_TARGET]
    y_bin = df["high_strength"]

    reg_metrics, reg_preds = evaluate_regression(X, y)
    clf_metrics, clf_preds = evaluate_classification(X, y_bin)
    model_metrics = pd.concat([reg_metrics, clf_metrics], ignore_index=True, sort=False)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    reg_preds.to_csv(OUTPUT_DIR / "cv_predictions_regression.csv", index=False)
    clf_preds.to_csv(OUTPUT_DIR / "cv_predictions_classification.csv", index=False)

    plot_model_comparison(reg_metrics, clf_metrics)
    best_reg = plot_predicted_vs_observed(reg_preds)
    best_clf = clf_metrics.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
    fit_feature_importance(X, y, best_reg)
    analyze_optimization(best_reg, best_clf, X, y, y_bin)

    summary = {
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "high_strength_threshold_kpa": HIGH_STRENGTH_THRESHOLD_KPA,
        "best_regression_model": best_reg,
        "best_classification_model": best_clf,
        "n_training_samples": int(df.shape[0]),
        "n_high_strength": int(y_bin.sum()),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
