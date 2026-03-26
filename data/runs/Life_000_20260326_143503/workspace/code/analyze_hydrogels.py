import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True, parents=True)
IMG.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

FEATURES = [
    'Nucleophilic-HEA',
    'Hydrophobic-BA',
    'Acidic-CBEA',
    'Cationic-ATAC',
    'Aromatic-PEA',
    'Amide-AAm',
]
TARGET = 'Glass (kPa)_10s'


def load_training():
    df = pd.read_excel(DATA / '184_verified_Original Data_ML_20230926.xlsx')
    # Clean mixed columns
    for c in ['Tanδ', 'Log_Slope']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def load_optimization():
    xls = pd.ExcelFile(DATA / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx')
    out = {}
    for s in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=s)
        for c in FEATURES + ['Glass (kPa)_max']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['ML'] = df['ML'].ffill()
        out[s] = df.dropna(subset=FEATURES + ['Glass (kPa)_max']).copy()
    return out


def cv_metrics(y, pred):
    rmse = mean_squared_error(y, pred) ** 0.5
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    rho = stats.spearmanr(y, pred, nan_policy='omit').statistic
    pear = stats.pearsonr(y, pred)[0]
    return {'rmse': rmse, 'mae': mae, 'r2': r2, 'spearman': rho, 'pearson': pear}


def evaluate_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    gp_kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(len(FEATURES)), length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e2))

    models = {
        'Linear': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        'Ridge': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0))
        ]),
        'RandomForest': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', RandomForestRegressor(n_estimators=600, min_samples_leaf=2, random_state=42))
        ]),
        'GaussianProcess': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('model', GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, random_state=42, n_restarts_optimizer=3))
        ]),
    }

    rows = []
    preds = {}
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=1)
        preds[name] = pred
        m = cv_metrics(y, pred)
        m['model'] = name
        rows.append(m)
    perf = pd.DataFrame(rows).sort_values('r2', ascending=False)
    return perf, preds, models


def summarize_training(df):
    desc = df[FEATURES + [TARGET, 'Q', 'Phase Seperation', 'Modulus (kPa)', 'XlogP3']].describe().T
    desc.to_csv(OUT / 'training_summary.csv')


def make_figures(df, perf, preds, best_model_name, best_model, opt):
    # Figure 1 target distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df[TARGET], bins=20, kde=True, ax=axes[0], color='#4C72B0')
    axes[0].axvline(1000, ls='--', c='red', lw=2, label='1 MPa target')
    axes[0].set_xlabel('Adhesive strength on glass at 10 s (kPa)')
    axes[0].legend()
    top = df.nlargest(20, TARGET)
    comp = top[FEATURES].mean().sort_values(ascending=False)
    sns.barplot(x=comp.values, y=comp.index, ax=axes[1], color='#55A868')
    axes[1].set_xlabel('Mean composition among top-20 strongest gels')
    axes[1].set_ylabel('Monomer')
    fig.tight_layout()
    fig.savefig(IMG / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2 model comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_perf = perf.melt(id_vars='model', value_vars=['r2', 'rmse', 'mae'], var_name='metric', value_name='value')
    sns.barplot(data=plot_perf, x='metric', y='value', hue='model', ax=ax)
    ax.set_title('10-fold cross-validation performance')
    fig.tight_layout()
    fig.savefig(IMG / 'model_comparison.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3 predicted vs observed for best model
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    y = df[TARGET].values
    pred = preds[best_model_name]
    sns.scatterplot(x=y, y=pred, ax=ax, s=60)
    lo = min(y.min(), pred.min())
    hi = max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], ls='--', c='black')
    ax.set_xlabel('Observed strength (kPa)')
    ax.set_ylabel(f'Cross-validated prediction ({best_model_name})')
    ax.set_title(f'{best_model_name}: observed vs predicted')
    fig.tight_layout()
    fig.savefig(IMG / 'predicted_vs_observed.png', bbox_inches='tight')
    plt.close(fig)

    # Fit best model for importance and design analysis
    X = df[FEATURES]
    y = df[TARGET]
    best_model.fit(X, y)
    pimp = permutation_importance(best_model, X, y, n_repeats=30, random_state=42)
    imp = pd.DataFrame({'feature': FEATURES, 'importance': pimp.importances_mean, 'std': pimp.importances_std}).sort_values('importance', ascending=False)
    imp.to_csv(OUT / 'feature_importance.csv', index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp, x='importance', y='feature', ax=ax, color='#C44E52')
    ax.set_title('Permutation importance for best model')
    fig.tight_layout()
    fig.savefig(IMG / 'feature_importance.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 5 optimization landscape and shift
    pred_df = pd.concat([v.assign(source=k) for k, v in opt.items()], ignore_index=True)
    train_comp = df[FEATURES].copy(); train_comp['dataset'] = 'Initial 184'
    opt_comp = pred_df[FEATURES].copy(); opt_comp['dataset'] = 'Optimization candidates'
    both = pd.concat([train_comp, opt_comp], ignore_index=True)
    long = both.melt(id_vars='dataset', var_name='feature', value_name='fraction')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=long, x='feature', y='fraction', hue='dataset', ax=ax)
    ax.tick_params(axis='x', rotation=30)
    ax.set_title('Composition shift from initial data to optimization candidates')
    fig.tight_layout()
    fig.savefig(IMG / 'composition_shift.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 6 candidate ranking by best model
    pred_df['predicted_by_best'] = best_model.predict(pred_df[FEATURES])
    pred_df['within_1MPa_gap'] = 1000 - pred_df['predicted_by_best']
    topcand = pred_df.sort_values('predicted_by_best', ascending=False).head(20).copy()
    topcand.to_csv(OUT / 'top20_candidates_by_best_model.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=topcand, x='predicted_by_best', y=topcand.index.astype(str), hue='source', dodge=False, ax=ax)
    ax.axvline(1000, ls='--', c='red', label='1 MPa target')
    ax.set_xlabel('Predicted strength by retrained best model (kPa)')
    ax.set_ylabel('Top 20 candidate rank')
    ax.set_title('Top optimization candidates remain below 1 MPa')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(IMG / 'top_candidates.png', bbox_inches='tight')
    plt.close(fig)

    return imp, pred_df, topcand


def additional_analysis(df, pred_df):
    # correlations and extrapolation distances
    corr = df[FEATURES + [TARGET, 'Q', 'Modulus (kPa)', 'XlogP3']].corr(numeric_only=True)
    corr.to_csv(OUT / 'correlation_matrix.csv')

    train_center = df[FEATURES].mean().values
    train_cov = np.cov(df[FEATURES].values.T)
    inv_cov = np.linalg.pinv(train_cov)
    dists = []
    for _, row in pred_df.iterrows():
        x = row[FEATURES].values.astype(float) - train_center
        md = float(np.sqrt(x @ inv_cov @ x.T))
        dists.append(md)
    pred_df['mahalanobis_to_training'] = dists
    pred_df.to_csv(OUT / 'optimization_candidates_scored.csv', index=False)

    # natural-language ready summary tables
    monomer_target_corr = df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET).sort_values(ascending=False)
    monomer_target_corr.rename('pearson_corr').to_csv(OUT / 'monomer_target_correlation.csv')

    return corr, monomer_target_corr, pred_df


def main():
    df = load_training()
    opt = load_optimization()
    summarize_training(df)
    perf, preds, models = evaluate_models(df)
    perf.to_csv(OUT / 'model_performance.csv', index=False)
    best_model_name = perf.iloc[0]['model']
    best_model = models[best_model_name]
    imp, pred_df, topcand = make_figures(df, perf, preds, best_model_name, best_model, opt)
    corr, monomer_corr, pred_df = additional_analysis(df, pred_df)

    # report-ready JSON summary
    summary = {
        'n_training': int(len(df)),
        'max_training_kpa': float(df[TARGET].max()),
        'mean_training_kpa': float(df[TARGET].mean()),
        'top_model': best_model_name,
        'top_model_r2': float(perf.iloc[0]['r2']),
        'top_model_rmse': float(perf.iloc[0]['rmse']),
        'optimization_max_best_model_kpa': float(pred_df['predicted_by_best'].max()),
        'optimization_max_file_value_kpa': float(pred_df['Glass (kPa)_max'].max()),
        'fraction_of_training_above_100_kpa': float((df[TARGET] > 100).mean()),
        'fraction_of_training_above_200_kpa': float((df[TARGET] > 200).mean()),
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('\nModel performance\n', perf.to_string(index=False))
    print('\nTop candidate preview\n', topcand[['source','ML','predicted_by_best','Glass (kPa)_max'] + FEATURES].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
