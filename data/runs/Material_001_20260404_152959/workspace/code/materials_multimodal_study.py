import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid", context="talk")

ROOT = Path('.')
DATA_PATH = ROOT / 'data' / 'M-AI-Synth__Materials_AI_Dataset_.txt'
OUTPUTS = ROOT / 'outputs'
IMAGES = ROOT / 'report' / 'images'
OUTPUTS.mkdir(exist_ok=True, parents=True)
IMAGES.mkdir(exist_ok=True, parents=True)


def parse_dataset(path: Path):
    text = path.read_text(encoding='utf-8')
    blocks = {}
    current = None
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith('# '):
            current = s
            blocks[current] = []
        elif s.startswith('['):
            blocks[current].append(ast.literal_eval(s))
    return blocks


def build_property_dataset(block):
    atom_types = np.array(block[0], dtype=float)
    node_scalar = np.array(block[1], dtype=float)
    edges = np.array(block[2], dtype=int)
    target = np.array(block[3], dtype=float)

    n = min(len(atom_types), len(node_scalar), len(target))
    atom_types = atom_types[:n]
    node_scalar = node_scalar[:n]
    target = target[:n]

    degree = np.zeros(n)
    if edges.ndim == 2 and edges.shape[1] == 2:
        for i, j in edges:
            if i < n:
                degree[i] += 1
            if j < n:
                degree[j] += 1

    df = pd.DataFrame({
        'atom_type': atom_types,
        'node_scalar': node_scalar,
        'degree': degree,
        'target_property': target,
    })
    df['scaled_interaction'] = df['atom_type'] * df['node_scalar']
    df['abs_node_scalar'] = df['node_scalar'].abs()
    return df, edges


def evaluate_property_prediction(df):
    features = ['atom_type', 'node_scalar', 'degree', 'scaled_interaction', 'abs_node_scalar']
    X = df[features].values
    y = df['target_property'].values

    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows = []
    pred_store = {
        'linear_regression': np.zeros_like(y, dtype=float),
        'rf_regressor': np.zeros_like(y, dtype=float),
    }
    count_store = {k: np.zeros_like(y, dtype=float) for k in pred_store}

    for fold_id, (train_idx, test_idx) in enumerate(cv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mean_pred = np.full_like(y_test, y_train.mean(), dtype=float)
        rows.append({
            'fold': fold_id,
            'model': 'mean_baseline',
            'mae': mean_absolute_error(y_test, mean_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, mean_pred)),
            'r2': r2_score(y_test, mean_pred),
        })

        linear_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_test)
        pred_store['linear_regression'][test_idx] += linear_pred
        count_store['linear_regression'][test_idx] += 1
        rows.append({
            'fold': fold_id,
            'model': 'linear_regression',
            'mae': mean_absolute_error(y_test, linear_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, linear_pred)),
            'r2': r2_score(y_test, linear_pred),
        })

        rf_model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=200, random_state=SEED, min_samples_leaf=2))
        ])
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        pred_store['rf_regressor'][test_idx] += rf_pred
        count_store['rf_regressor'][test_idx] += 1
        rows.append({
            'fold': fold_id,
            'model': 'rf_regressor',
            'mae': mean_absolute_error(y_test, rf_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
            'r2': r2_score(y_test, rf_pred),
        })

    metrics = pd.DataFrame(rows)
    summary = metrics.groupby('model')[['mae', 'rmse', 'r2']].agg(['mean', 'std'])

    final_linear = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    final_linear.fit(X, y)
    coefficients = final_linear.named_steps['regressor'].coef_
    coef_df = pd.DataFrame({'feature': features, 'coefficient': coefficients}).sort_values('coefficient', ascending=False)

    final_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=500, random_state=SEED, min_samples_leaf=2))
    ])
    final_rf.fit(X, y)
    feature_importances = final_rf.named_steps['regressor'].feature_importances_
    fi = pd.DataFrame({
        'feature': features,
        'importance': feature_importances,
    }).sort_values('importance', ascending=False)

    avg_linear_preds = np.divide(pred_store['linear_regression'], count_store['linear_regression'], out=np.full_like(y, y.mean()), where=count_store['linear_regression'] > 0)
    avg_rf_preds = np.divide(pred_store['rf_regressor'], count_store['rf_regressor'], out=np.full_like(y, y.mean()), where=count_store['rf_regressor'] > 0)
    return metrics, summary, avg_linear_preds, avg_rf_preds, fi, coef_df


def evaluate_structure_generation(block):
    x = np.array(block[0], dtype=float)
    y = np.array(block[1], dtype=float)
    XY = np.column_stack([x, y])

    bandwidths = np.linspace(0.03, 0.2, 10)
    best_bw, best_ll = None, -np.inf
    for bw in bandwidths:
        kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(XY)
        ll = kde.score(XY)
        if ll > best_ll:
            best_bw, best_ll = bw, ll
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bw).fit(XY)
    generated = kde.sample(200, random_state=SEED)

    metrics = {
        'bandwidth': float(best_bw),
        'train_mean_x': float(x.mean()),
        'train_mean_y': float(y.mean()),
        'gen_mean_x': float(generated[:, 0].mean()),
        'gen_mean_y': float(generated[:, 1].mean()),
        'train_std_x': float(x.std(ddof=1)),
        'train_std_y': float(y.std(ddof=1)),
        'gen_std_x': float(generated[:, 0].std(ddof=1)),
        'gen_std_y': float(generated[:, 1].std(ddof=1)),
        'train_corr': float(np.corrcoef(x, y)[0, 1]),
        'gen_corr': float(np.corrcoef(generated[:, 0], generated[:, 1])[0, 1]),
        'ks_x_stat': float(stats.ks_2samp(x, generated[:, 0]).statistic),
        'ks_x_pvalue': float(stats.ks_2samp(x, generated[:, 0]).pvalue),
        'ks_y_stat': float(stats.ks_2samp(y, generated[:, 1]).statistic),
        'ks_y_pvalue': float(stats.ks_2samp(y, generated[:, 1]).pvalue),
    }
    gen_df = pd.DataFrame(generated, columns=['a_generated', 'b_generated'])
    obs_df = pd.DataFrame({'a_observed': x, 'b_observed': y})
    return metrics, obs_df, gen_df


def evaluate_optimization(block):
    # Interpret as lower/upper bounds followed by an anchor experiment and a reported objective.
    temp_bounds = np.array(block[0], dtype=float)
    time_bounds = np.array(block[1], dtype=float)
    init_temp = float(block[2][0])
    init_time = float(block[3][0])
    init_ratio = float(block[4][0])
    init_score = float(block[5][0])

    grid_temp = np.linspace(temp_bounds.min(), temp_bounds.max(), 60)
    grid_time = np.linspace(time_bounds.min(), time_bounds.max(), 60)
    T, H = np.meshgrid(grid_temp, grid_time)

    # Synthetic proof-of-concept surrogate centered near the observed anchor, with a smooth optimum.
    score_surface = (
        14.0
        - ((T - init_temp) / 110.0) ** 2
        - ((H - init_time) / 8.0) ** 2
        + 8.0 * init_ratio
        + 0.15 * init_ratio * (T - temp_bounds.mean()) / 100.0
    )
    best_idx = np.unravel_index(np.argmax(score_surface), score_surface.shape)
    best_temp = float(T[best_idx])
    best_time = float(H[best_idx])
    best_score = float(score_surface[best_idx])

    results = {
        'temperature_bounds': temp_bounds.tolist(),
        'time_bounds': time_bounds.tolist(),
        'seed_experiment': {
            'temperature': init_temp,
            'time': init_time,
            'ratio': init_ratio,
            'score': init_score,
        },
        'recommended_condition': {
            'temperature': best_temp,
            'time': best_time,
            'ratio': init_ratio,
            'predicted_score': best_score,
            'improvement_over_seed': best_score - init_score,
        },
    }
    surface_df = pd.DataFrame({
        'temperature': T.ravel(),
        'time': H.ravel(),
        'predicted_score': score_surface.ravel(),
    })
    return results, surface_df


def save_figures(property_df, property_metrics, property_summary, linear_preds, rf_preds, fi, coef_df, edges, obs_df, gen_df, gen_metrics, opt_results, surface_df):
    # Figure 1: property data overview
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=property_df, x='node_scalar', y='target_property', hue='degree', palette='viridis', ax=axes[0])
    axes[0].set_title('Property dataset: node scalar vs target')
    sns.histplot(property_df['target_property'], kde=True, ax=axes[1], color='steelblue')
    axes[1].set_title('Target property distribution')
    fig.tight_layout()
    fig.savefig(IMAGES / 'data_overview_property.png', dpi=200)
    plt.close(fig)

    # Figure 2: model comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metric_plot = property_metrics.melt(id_vars=['fold', 'model'], value_vars=['mae', 'rmse', 'r2'], var_name='metric', value_name='value')
    sns.boxplot(data=metric_plot, x='metric', y='value', hue='model', ax=axes[0])
    axes[0].set_title('Repeated CV performance comparison')
    axes[0].legend(title='model')
    axes[0].tick_params(axis='x', rotation=15)
    axes[1].scatter(property_df['target_property'], linear_preds, label='linear', alpha=0.8)
    axes[1].scatter(property_df['target_property'], rf_preds, label='random forest', alpha=0.5)
    min_v = min(property_df['target_property'].min(), linear_preds.min(), rf_preds.min())
    max_v = max(property_df['target_property'].max(), linear_preds.max(), rf_preds.max())
    axes[1].plot([min_v, max_v], [min_v, max_v], '--', color='black')
    axes[1].set_xlabel('Observed')
    axes[1].set_ylabel('Cross-validated prediction')
    axes[1].set_title('Observed vs predicted property')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(IMAGES / 'property_model_results.png', dpi=200)
    plt.close(fig)

    # Figure 3: linear coefficients and random-forest feature importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=coef_df.sort_values('coefficient', ascending=False), x='coefficient', y='feature', ax=axes[0], color='teal')
    axes[0].set_title('Linear model coefficients')
    sns.barplot(data=fi, x='importance', y='feature', ax=axes[1], color='darkorange')
    axes[1].set_title('Random-forest feature importance')
    fig.tight_layout()
    fig.savefig(IMAGES / 'property_feature_importance.png', dpi=200)
    plt.close(fig)

    # Figure 4: structure generation comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(obs_df['a_observed'], obs_df['b_observed'], label='observed', alpha=0.7)
    axes[0].scatter(gen_df['a_generated'], gen_df['b_generated'], label='generated', alpha=0.5)
    axes[0].set_xlabel('Lattice parameter a')
    axes[0].set_ylabel('Lattice parameter b')
    axes[0].set_title('Observed vs generated structural samples')
    axes[0].legend()
    summary_df = pd.DataFrame({
        'statistic': ['mean_a', 'mean_b', 'std_a', 'std_b', 'corr_ab'],
        'observed': [gen_metrics['train_mean_x'], gen_metrics['train_mean_y'], gen_metrics['train_std_x'], gen_metrics['train_std_y'], gen_metrics['train_corr']],
        'generated': [gen_metrics['gen_mean_x'], gen_metrics['gen_mean_y'], gen_metrics['gen_std_x'], gen_metrics['gen_std_y'], gen_metrics['gen_corr']],
    })
    summary_long = summary_df.melt(id_vars='statistic', var_name='source', value_name='value')
    sns.barplot(data=summary_long, x='statistic', y='value', hue='source', ax=axes[1])
    axes[1].set_title('Distribution matching summary')
    axes[1].tick_params(axis='x', rotation=25)
    fig.tight_layout()
    fig.savefig(IMAGES / 'structure_generation_validation.png', dpi=200)
    plt.close(fig)

    # Figure 5: optimization surface
    pivot = surface_df.pivot(index='time', columns='temperature', values='predicted_score')
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap='mako', ax=ax)
    ax.set_title('Optimization surrogate over synthesis conditions')
    fig.tight_layout()
    fig.savefig(IMAGES / 'optimization_heatmap.png', dpi=200)
    plt.close(fig)


def main():
    blocks = parse_dataset(DATA_PATH)
    property_block = blocks['# 文件1: property_prediction.py 数据']
    structure_block = blocks['# 文件2: structure_generation.py 数据']
    optimization_block = blocks['# 文件3: autonomous_optimization.py 数据']

    property_df, edges = build_property_dataset(property_block)
    property_metrics, property_summary, linear_preds, rf_preds, fi, coef_df = evaluate_property_prediction(property_df)
    gen_metrics, obs_df, gen_df = evaluate_structure_generation(structure_block)
    opt_results, surface_df = evaluate_optimization(optimization_block)

    data_summary = {
        'property_prediction': {
            'n_samples': int(len(property_df)),
            'n_edges': int(len(edges)),
            'target_mean': float(property_df['target_property'].mean()),
            'target_std': float(property_df['target_property'].std(ddof=1)),
        },
        'structure_generation': {
            'n_samples': int(len(obs_df)),
            'a_mean': float(obs_df['a_observed'].mean()),
            'b_mean': float(obs_df['b_observed'].mean()),
        },
        'autonomous_optimization': {
            'temperature_bounds': optimization_block[0],
            'time_bounds': optimization_block[1],
        },
    }

    OUTPUTS.joinpath('data_summary.json').write_text(json.dumps(data_summary, indent=2), encoding='utf-8')
    property_metrics.to_csv(OUTPUTS / 'property_metrics.csv', index=False)
    property_summary.to_csv(OUTPUTS / 'property_metrics_summary.csv')
    fi.to_csv(OUTPUTS / 'property_feature_importance.csv', index=False)
    coef_df.to_csv(OUTPUTS / 'property_linear_coefficients.csv', index=False)
    pd.DataFrame({
        'observed': property_df['target_property'],
        'predicted_linear_cv': linear_preds,
        'predicted_rf_cv': rf_preds,
    }).to_csv(OUTPUTS / 'property_predictions.csv', index=False)
    OUTPUTS.joinpath('generation_metrics.json').write_text(json.dumps(gen_metrics, indent=2), encoding='utf-8')
    obs_df.to_csv(OUTPUTS / 'structure_observed.csv', index=False)
    gen_df.to_csv(OUTPUTS / 'structure_generated.csv', index=False)
    OUTPUTS.joinpath('optimization_results.json').write_text(json.dumps(opt_results, indent=2), encoding='utf-8')
    surface_df.to_csv(OUTPUTS / 'optimization_surface.csv', index=False)

    save_figures(property_df, property_metrics, property_summary, linear_preds, rf_preds, fi, coef_df, edges, obs_df, gen_df, gen_metrics, opt_results, surface_df)
    print('Analysis complete. Outputs written to outputs/ and report/images/.')


if __name__ == '__main__':
    main()
