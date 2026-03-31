import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
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
N_BLOCKS = 5
GAP = 20
TARGETS = ["Attack", "Sniffing"]


@dataclass
class Split:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int


ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
REPORT_IMG = ROOT / 'report' / 'images'
CODE_DIR = ROOT / 'code'


for path in [OUTPUTS, REPORT_IMG, CODE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')


def safe_json(obj, path: Path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


features_df = pd.read_csv(DATA_DIR / 'Together_1_features_extracted.csv')
targets_df = pd.read_csv(DATA_DIR / 'Together_1_targets_inserted.csv')
reference_df = pd.read_csv(DATA_DIR / 'Together_1_machine_results_reference.csv')

# Harmonize index column
if features_df.columns[0].startswith('Unnamed') or features_df.columns[0] == '':
    features_df = features_df.rename(columns={features_df.columns[0]: 'frame_index'})
if targets_df.columns[0].startswith('Unnamed') or targets_df.columns[0] == '':
    targets_df = targets_df.rename(columns={targets_df.columns[0]: 'frame_index'})
if reference_df.columns[0].startswith('Unnamed') or reference_df.columns[0] == '':
    reference_df = reference_df.rename(columns={reference_df.columns[0]: 'frame_index'})

assert 'frame_index' in features_df.columns and 'frame_index' in targets_df.columns

merge_cols = ['frame_index']
for t in TARGETS:
    if t not in targets_df.columns:
        raise ValueError(f'Missing target column {t}')

model_df = features_df.merge(targets_df[['frame_index'] + TARGETS], on='frame_index', how='inner', validate='one_to_one')
model_df = model_df.sort_values('frame_index').reset_index(drop=True)

label_summary_rows = []
for t in TARGETS:
    y = model_df[t].astype(int)
    runs = []
    in_run = False
    run_start = None
    for i, val in enumerate(y):
        if val == 1 and not in_run:
            in_run = True
            run_start = i
        elif val == 0 and in_run:
            runs.append(i - run_start)
            in_run = False
    if in_run:
        runs.append(len(y) - run_start)
    label_summary_rows.append({
        'target': t,
        'n_positive_frames': int(y.sum()),
        'prevalence': float(y.mean()),
        'n_positive_bouts': int(len(runs)),
        'mean_bout_len': float(np.mean(runs) if runs else 0),
        'median_bout_len': float(np.median(runs) if runs else 0),
        'max_bout_len': int(max(runs) if runs else 0),
    })
label_summary = pd.DataFrame(label_summary_rows)
label_summary.to_csv(OUTPUTS / 'label_summary.csv', index=False)

# Exclude target/leakage-prone columns
exclude_cols = {'frame_index', *TARGETS}
leakage_terms = ['attack', 'sniff', 'target', 'prob', 'prediction', 'classifier', 'machine', 'result']
excluded_rows = []
feature_cols = []
for col in features_df.columns:
    lower = col.lower()
    if col == 'frame_index':
        excluded_rows.append({'column': col, 'reason': 'identifier'})
        continue
    if any(term in lower for term in leakage_terms):
        excluded_rows.append({'column': col, 'reason': 'leakage_name_filter'})
        continue
    if pd.api.types.is_numeric_dtype(features_df[col]):
        feature_cols.append(col)
    else:
        excluded_rows.append({'column': col, 'reason': 'non_numeric'})

excluded_df = pd.DataFrame(excluded_rows)
excluded_df.to_csv(OUTPUTS / 'excluded_columns.csv', index=False)

feature_summary = pd.DataFrame({
    'column': feature_cols,
    'dtype': [str(features_df[c].dtype) for c in feature_cols],
    'missing_fraction': [float(features_df[c].isna().mean()) for c in feature_cols],
    'n_unique': [int(features_df[c].nunique(dropna=False)) for c in feature_cols],
    'std': [float(features_df[c].std()) for c in feature_cols],
})
feature_summary.to_csv(OUTPUTS / 'feature_summary.csv', index=False)
feature_summary[['column']].to_csv(OUTPUTS / 'feature_manifest.csv', index=False)

model_df[['frame_index'] + feature_cols + TARGETS].to_csv(OUTPUTS / 'model_table.csv', index=False)

# Data audit
alignment_checks = {
    'features_rows': int(len(features_df)),
    'targets_rows': int(len(targets_df)),
    'reference_rows': int(len(reference_df)),
    'merged_rows': int(len(model_df)),
    'features_frame_index_unique': bool(features_df['frame_index'].is_unique),
    'targets_frame_index_unique': bool(targets_df['frame_index'].is_unique),
    'same_frame_index_sequence': bool(features_df['frame_index'].tolist() == targets_df['frame_index'].tolist()),
    'n_features_used': int(len(feature_cols)),
}
safe_json(alignment_checks, OUTPUTS / 'data_audit.json')
with open(OUTPUTS / 'data_audit.md', 'w', encoding='utf-8') as f:
    f.write('# Data audit\n\n')
    for k, v in alignment_checks.items():
        f.write(f'- **{k}**: {v}\n')

# Contiguous blocks
n = len(model_df)
block_size = n // N_BLOCKS
splits: List[Split] = []
block_rows = []
for fold in range(N_BLOCKS):
    test_start = fold * block_size
    test_end = (fold + 1) * block_size if fold < N_BLOCKS - 1 else n
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_start:test_end] = True
    train_mask = ~test_mask
    left_gap = max(0, test_start - GAP)
    right_gap = min(n, test_end + GAP)
    gap_mask = np.zeros(n, dtype=bool)
    gap_mask[left_gap:test_start] = True
    gap_mask[test_end:right_gap] = True
    train_mask[gap_mask] = False
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        continue
    splits.append(Split(fold, train_idx, test_idx, int(train_idx.min()), int(train_idx.max()), test_start, test_end - 1))
    block_rows.append({
        'fold': fold,
        'train_n': len(train_idx),
        'test_n': len(test_idx),
        'train_start_frame': int(model_df.loc[train_idx[0], 'frame_index']),
        'train_end_frame': int(model_df.loc[train_idx[-1], 'frame_index']),
        'test_start_frame': int(model_df.loc[test_start, 'frame_index']),
        'test_end_frame': int(model_df.loc[test_end - 1, 'frame_index']),
        'gap': GAP,
    })
block_manifest = pd.DataFrame(block_rows)
block_manifest.to_csv(OUTPUTS / 'block_manifest.csv', index=False)

# Split diagram
fig, ax = plt.subplots(figsize=(12, 4))
for i, s in enumerate(splits):
    ax.broken_barh([(s.train_idx.min(), len(s.train_idx) // 2)], (i - 0.4, 0.35), facecolors='tab:blue')
    ax.broken_barh([(s.test_start, s.test_end - s.test_start + 1)], (i + 0.05, 0.35), facecolors='tab:orange')
    if s.train_idx.max() > s.test_end:
        ax.broken_barh([(s.test_end + GAP + 1, s.train_idx.max() - (s.test_end + GAP + 1) + 1)], (i - 0.4, 0.35), facecolors='tab:blue')
ax.set_xlabel('Row index')
ax.set_ylabel('Fold')
ax.set_yticks(range(len(splits)))
ax.set_yticklabels([f'Fold {s.fold}' for s in splits])
ax.set_title('Contiguous blocked evaluation with boundary gaps')
fig.tight_layout()
fig.savefig(OUTPUTS / 'split_diagram.png', dpi=200)
fig.savefig(REPORT_IMG / 'split_diagram.png', dpi=200)
plt.close(fig)

X = model_df[feature_cols]

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

models = {
    'dummy_prior': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', DummyClassifier(strategy='prior')),
    ]),
    'logreg_balanced': Pipeline([
        ('prep', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)),
    ]),
    'random_forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            n_jobs=-1,
            class_weight='balanced_subsample',
            random_state=SEED,
        )),
    ]),
}


def compute_metrics(y_true, prob, pred):
    metrics = {
        'average_precision': float(average_precision_score(y_true, prob)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'positive_rate': float(np.mean(y_true)),
    }
    if len(np.unique(y_true)) > 1:
        metrics['roc_auc'] = float(roc_auc_score(y_true, prob))
    else:
        metrics['roc_auc'] = np.nan
    return metrics


all_fold_metrics = []
summary_rows = []
all_predictions = []
pr_store: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
feature_importance_rows = []
bout_rows = []
random_split_rows = []

for target in TARGETS:
    y = model_df[target].astype(int).values
    for model_name, model in models.items():
        fold_probs = []
        fold_truths = []
        fold_metrics_for_summary = []
        for split in splits:
            X_train = X.iloc[split.train_idx]
            X_test = X.iloc[split.test_idx]
            y_train = y[split.train_idx]
            y_test = y[split.test_idx]
            est = clone(model)
            est.fit(X_train, y_train)
            if hasattr(est[-1], 'predict_proba') or hasattr(est, 'predict_proba'):
                prob = est.predict_proba(X_test)[:, 1]
            else:
                scores = est.decision_function(X_test)
                prob = 1 / (1 + np.exp(-scores))
            pred = (prob >= 0.5).astype(int)
            metrics = compute_metrics(y_test, prob, pred)
            metrics.update({'target': target, 'model': model_name, 'fold': split.fold})
            all_fold_metrics.append(metrics)
            fold_metrics_for_summary.append(metrics)
            fold_probs.append(prob)
            fold_truths.append(y_test)
            pred_df = pd.DataFrame({
                'frame_index': model_df.iloc[split.test_idx]['frame_index'].values,
                'target': target,
                'model': model_name,
                'fold': split.fold,
                'y_true': y_test,
                'y_prob': prob,
                'y_pred': pred,
            })
            all_predictions.append(pred_df)

        summary = pd.DataFrame(fold_metrics_for_summary)
        summary_rows.append({
            'target': target,
            'model': model_name,
            'average_precision_mean': summary['average_precision'].mean(),
            'average_precision_sd': summary['average_precision'].std(ddof=1),
            'f1_mean': summary['f1'].mean(),
            'f1_sd': summary['f1'].std(ddof=1),
            'precision_mean': summary['precision'].mean(),
            'recall_mean': summary['recall'].mean(),
            'roc_auc_mean': summary['roc_auc'].mean(),
        })
        pr, rc, _ = precision_recall_curve(np.concatenate(fold_truths), np.concatenate(fold_probs))
        pr_store[(target, model_name)] = (pr, rc)

    # Feature importance from full-data fit for interpretability, not primary evaluation
    rf = clone(models['random_forest'])
    rf.fit(X, y)
    clf = rf.named_steps['clf']
    imp = pd.DataFrame({'feature': feature_cols, 'importance': clf.feature_importances_})
    imp = imp.sort_values('importance', ascending=False).head(20)
    imp['target'] = target
    imp['model'] = 'random_forest'
    feature_importance_rows.append(imp)

    lr = clone(models['logreg_balanced'])
    lr.fit(X, y)
    coef = lr.named_steps['clf'].coef_[0]
    coef_df = pd.DataFrame({'feature': feature_cols, 'importance': np.abs(coef), 'signed_coefficient': coef})
    coef_df = coef_df.sort_values('importance', ascending=False).head(20)
    coef_df['target'] = target
    coef_df['model'] = 'logreg_balanced'
    feature_importance_rows.append(coef_df)

    # Bout-level summary using best AP model
    target_summary = [r for r in summary_rows if r['target'] == target]
    best_model = sorted(target_summary, key=lambda d: d['average_precision_mean'], reverse=True)[0]['model']
    preds = pd.concat([df for df in all_predictions if df['target'].iloc[0] == target and df['model'].iloc[0] == best_model], ignore_index=True)
    preds = preds.sort_values('frame_index').reset_index(drop=True)
    y_true = preds['y_true'].values
    y_pred = preds['y_pred'].values
    def bout_stats(arr):
        bouts = []
        start = None
        for i, v in enumerate(arr):
            if v == 1 and start is None:
                start = i
            elif v == 0 and start is not None:
                bouts.append((start, i - 1, i - start))
                start = None
        if start is not None:
            bouts.append((start, len(arr)-1, len(arr)-start))
        return bouts
    true_bouts = bout_stats(y_true)
    pred_bouts = bout_stats(y_pred)
    bout_rows.append({
        'target': target,
        'best_model': best_model,
        'n_true_bouts': len(true_bouts),
        'mean_true_bout_len': float(np.mean([b[2] for b in true_bouts]) if true_bouts else 0),
        'n_pred_bouts': len(pred_bouts),
        'mean_pred_bout_len': float(np.mean([b[2] for b in pred_bouts]) if pred_bouts else 0),
    })

    # optimistic random split comparison with same best model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=SEED)
    est = clone(models[best_model])
    est.fit(X_train, y_train)
    prob = est.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = compute_metrics(y_test, prob, pred)
    metrics.update({'target': target, 'model': best_model, 'evaluation': 'random_split'})
    random_split_rows.append(metrics)

fold_metrics_df = pd.DataFrame(all_fold_metrics)
fold_metrics_df.to_csv(OUTPUTS / 'baseline_metrics_by_fold.csv', index=False)
summary_df = pd.DataFrame(summary_rows).sort_values(['target', 'average_precision_mean'], ascending=[True, False])
summary_df.to_csv(OUTPUTS / 'baseline_summary.csv', index=False)
predictions_df = pd.concat(all_predictions, ignore_index=True)
for target in TARGETS:
    predictions_df[predictions_df['target'] == target].to_csv(OUTPUTS / f'predictions_{target.lower()}.csv', index=False)

importance_df = pd.concat(feature_importance_rows, ignore_index=True)
importance_df.to_csv(OUTPUTS / 'feature_importance_table.csv', index=False)

pd.DataFrame(bout_rows).to_csv(OUTPUTS / 'bout_level_summary.csv', index=False)
pd.DataFrame(random_split_rows).to_csv(OUTPUTS / 'leakage_comparison.csv', index=False)

# Main result figure: AP barplot
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
for ax, target in zip(axes, TARGETS):
    tmp = summary_df[summary_df['target'] == target].copy()
    sns.barplot(data=tmp, x='model', y='average_precision_mean', ax=ax, palette='deep')
    ax.errorbar(x=np.arange(len(tmp)), y=tmp['average_precision_mean'], yerr=tmp['average_precision_sd'], fmt='none', ecolor='black', capsize=4)
    ax.set_title(f'{target}: average precision across blocked folds')
    ax.set_xlabel('Model')
    ax.set_ylabel('Average precision')
    ax.tick_params(axis='x', rotation=20)
fig.tight_layout()
fig.savefig(REPORT_IMG / 'average_precision_summary.png', dpi=200)
fig.savefig(OUTPUTS / 'average_precision_summary.png', dpi=200)
plt.close(fig)

# PR curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, target in zip(axes, TARGETS):
    for model_name in models:
        precision, recall = pr_store[(target, model_name)]
        ax.plot(recall, precision, label=model_name)
    prevalence = float(model_df[target].mean())
    ax.axhline(prevalence, ls='--', color='grey', label='prevalence')
    ax.set_title(f'{target}: pooled precision-recall curves')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUTPUTS / 'pr_curves.png', dpi=200)
fig.savefig(REPORT_IMG / 'pr_curves.png', dpi=200)
plt.close(fig)

# Confusion matrices for best models by target
best_models = {}
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, target in zip(axes, TARGETS):
    tmp = summary_df[summary_df['target'] == target].sort_values('average_precision_mean', ascending=False)
    best_model = tmp.iloc[0]['model']
    best_models[target] = best_model
    pred_t = predictions_df[(predictions_df['target'] == target) & (predictions_df['model'] == best_model)]
    cm = confusion_matrix(pred_t['y_true'], pred_t['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_title(f'{target}: confusion matrix ({best_model})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
fig.tight_layout()
fig.savefig(OUTPUTS / 'confusion_matrices.png', dpi=200)
fig.savefig(REPORT_IMG / 'confusion_matrices.png', dpi=200)
plt.close(fig)

# Feature importance plots
for target in TARGETS:
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    for ax, model_name in zip(axes, ['random_forest', 'logreg_balanced']):
        tmp = importance_df[(importance_df['target'] == target) & (importance_df['model'] == model_name)].head(15).sort_values('importance')
        sns.barplot(data=tmp, x='importance', y='feature', ax=ax, palette='viridis')
        ax.set_title(f'{target}: top features ({model_name})')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
    fig.tight_layout()
    fig.savefig(REPORT_IMG / f'feature_importance_{target.lower()}.png', dpi=200)
    fig.savefig(OUTPUTS / f'feature_importance_{target.lower()}.png', dpi=200)
    plt.close(fig)

# Data overview plots
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, target in zip(axes.flat[:2], TARGETS):
    sns.lineplot(x=model_df['frame_index'], y=model_df[target], ax=ax)
    ax.set_title(f'{target} labels over time')
    ax.set_xlabel('Frame index')
    ax.set_ylabel('Label')
for ax, col in zip(axes.flat[2:], feature_cols[:2]):
    sns.histplot(model_df[col], bins=50, ax=ax)
    ax.set_title(f'Feature distribution: {col}')
fig.tight_layout()
fig.savefig(REPORT_IMG / 'data_overview.png', dpi=200)
fig.savefig(OUTPUTS / 'data_overview.png', dpi=200)
plt.close(fig)

# Timeline examples for best models
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
window = min(2000, len(model_df))
start = max(0, len(model_df)//2 - window//2)
end = start + window
for ax, target in zip(axes, TARGETS):
    pred_t = predictions_df[(predictions_df['target'] == target) & (predictions_df['model'] == best_models[target])].sort_values('frame_index')
    pred_t = pred_t[(pred_t['frame_index'] >= model_df.loc[start, 'frame_index']) & (pred_t['frame_index'] <= model_df.loc[end-1, 'frame_index'])]
    ax.plot(pred_t['frame_index'], pred_t['y_true'], label='True', linewidth=2)
    ax.plot(pred_t['frame_index'], pred_t['y_prob'], label='Predicted probability', alpha=0.8)
    ax.set_title(f'{target}: timeline example ({best_models[target]})')
    ax.legend()
axes[-1].set_xlabel('Frame index')
fig.tight_layout()
fig.savefig(REPORT_IMG / 'timeline_examples.png', dpi=200)
fig.savefig(OUTPUTS / 'timeline_examples.png', dpi=200)
plt.close(fig)

# Error analysis markdown
with open(OUTPUTS / 'error_analysis.md', 'w', encoding='utf-8') as f:
    f.write('# Error analysis\n\n')
    for target in TARGETS:
        best_model = best_models[target]
        pred_t = predictions_df[(predictions_df['target'] == target) & (predictions_df['model'] == best_model)]
        fp = int(((pred_t['y_true'] == 0) & (pred_t['y_pred'] == 1)).sum())
        fn = int(((pred_t['y_true'] == 1) & (pred_t['y_pred'] == 0)).sum())
        tp = int(((pred_t['y_true'] == 1) & (pred_t['y_pred'] == 1)).sum())
        tn = int(((pred_t['y_true'] == 0) & (pred_t['y_pred'] == 0)).sum())
        f.write(f'## {target}\n')
        f.write(f'- Best blocked-evaluation model: **{best_model}**\n')
        f.write(f'- Confusion counts: TP={tp}, FP={fp}, TN={tn}, FN={fn}\n')
        f.write('- Interpretation: errors should be interpreted with caution because only one sequence is available and neighboring frames are autocorrelated.\n\n')

# Environment/run config
safe_json({
    'seed': SEED,
    'n_blocks': N_BLOCKS,
    'gap': GAP,
    'targets': TARGETS,
    'models': list(models.keys()),
}, OUTPUTS / 'run_config.json')
with open(OUTPUTS / 'environment.txt', 'w', encoding='utf-8') as f:
    f.write(f'Python: {sys.version}\n')
    f.write(f'Platform: {platform.platform()}\n')
    for pkg in ['numpy', 'pandas', 'sklearn', 'matplotlib', 'seaborn']:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        f.write(f'{pkg}: {version}\n')

print('Analysis complete.')
