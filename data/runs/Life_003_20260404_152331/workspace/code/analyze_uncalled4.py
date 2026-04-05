import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORT_IMG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 160
plt.rcParams['savefig.bbox'] = 'tight'
SEED = 42
np.random.seed(SEED)

PORE_FILES = {
    'DNA R9.4.1 6mer': DATA_DIR / 'dna_r9.4.1_400bps_6mer_uncalled4.csv',
    'DNA R10.4.1 9mer': DATA_DIR / 'dna_r10.4.1_400bps_9mer_uncalled4.csv',
    'RNA R9.4.1 5mer': DATA_DIR / 'rna_r9.4.1_70bps_5mer_uncalled4.csv',
    'RNA004 9mer': DATA_DIR / 'rna004_130bps_9mer_uncalled4.csv',
}


def bootstrap_metric(y_true, scores, metric_fn, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        sb = scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        vals.append(metric_fn(yb, sb))
    vals = np.asarray(vals)
    return {
        'mean': float(np.mean(vals)),
        'ci_low': float(np.percentile(vals, 2.5)),
        'ci_high': float(np.percentile(vals, 97.5)),
        'n_boot_valid': int(len(vals)),
    }


def summarize_inputs():
    records = []
    data_summary = {}
    for name, path in PORE_FILES.items():
        df = pd.read_csv(path)
        records.append({
            'dataset': name,
            'rows': len(df),
            'columns': ','.join(df.columns),
            'kmer_length': int(df['kmer'].str.len().iloc[0]),
            'null_rows': int(df.isna().any(axis=1).sum()),
            'duplicate_kmers': int(df['kmer'].duplicated().sum()),
        })
        data_summary[name] = {
            'path': str(path),
            'shape': list(df.shape),
            'columns': list(df.columns),
            'kmer_length': int(df['kmer'].str.len().iloc[0]),
        }

    perf = pd.read_csv(DATA_DIR / 'performance_summary.csv')
    unc = pd.read_csv(DATA_DIR / 'm6a_predictions_uncalled4.csv')
    nano = pd.read_csv(DATA_DIR / 'm6a_predictions_nanopolish.csv')
    labels = pd.read_csv(DATA_DIR / 'm6a_labels.csv')

    for name, df in [('performance_summary', perf), ('m6a_predictions_uncalled4', unc), ('m6a_predictions_nanopolish', nano), ('m6a_labels', labels)]:
        records.append({
            'dataset': name,
            'rows': len(df),
            'columns': ','.join(df.columns),
            'kmer_length': np.nan,
            'null_rows': int(df.isna().any(axis=1).sum()),
            'duplicate_kmers': int(df[df.columns[0]].duplicated().sum()) if len(df.columns) else 0,
        })
        data_summary[name] = {
            'path': str(DATA_DIR / f'{name}.csv') if name != 'performance_summary' else str(DATA_DIR / 'performance_summary.csv'),
            'shape': list(df.shape),
            'columns': list(df.columns),
        }

    common_ids = set(labels.site_id) & set(unc.site_id) & set(nano.site_id)
    data_summary['m6a_join'] = {
        'labels_rows': len(labels),
        'uncalled4_rows': len(unc),
        'nanopolish_rows': len(nano),
        'common_site_ids': len(common_ids),
        'positive_rate': float(labels['label'].mean()),
    }

    pd.DataFrame(records).to_csv(OUTPUT_DIR / 'data_overview.csv', index=False)
    with open(OUTPUT_DIR / 'data_summary.json', 'w') as f:
        json.dump(data_summary, f, indent=2)


def load_pore_data():
    dfs = []
    for name, path in PORE_FILES.items():
        df = pd.read_csv(path)
        df['Chemistry'] = name
        df['kmer_length'] = df['kmer'].str.len()
        for pos in range(df['kmer_length'].iloc[0]):
            df[f'base_{pos+1}'] = df['kmer'].str[pos]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def pore_model_analysis():
    pore_df = load_pore_data()
    summary = pore_df.groupby('Chemistry').agg(
        n_kmers=('kmer', 'size'),
        kmer_length=('kmer_length', 'first'),
        current_mean_mean=('current_mean', 'mean'),
        current_mean_std=('current_mean', 'std'),
        current_mean_min=('current_mean', 'min'),
        current_mean_max=('current_mean', 'max'),
        current_std_mean=('current_std', 'mean'),
        dwell_time_mean=('dwell_time', 'mean'),
        dwell_time_std=('dwell_time', 'std'),
    ).reset_index()
    summary.to_csv(OUTPUT_DIR / 'pore_model_summary.csv', index=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.boxplot(data=pore_df, x='Chemistry', y='current_mean', ax=axes[0], showfliers=False)
    axes[0].tick_params(axis='x', rotation=25)
    axes[0].set_title('Current mean by chemistry')
    sns.boxplot(data=pore_df, x='Chemistry', y='current_std', ax=axes[1], showfliers=False)
    axes[1].tick_params(axis='x', rotation=25)
    axes[1].set_title('Current noise by chemistry')
    sns.boxplot(data=pore_df, x='Chemistry', y='dwell_time', ax=axes[2], showfliers=False)
    axes[2].tick_params(axis='x', rotation=25)
    axes[2].set_title('Dwell time by chemistry')
    fig.savefig(REPORT_IMG_DIR / 'pore_model_overview.png')
    plt.close(fig)

    effects = []
    base_order = list('ACGT')
    for chemistry, df in pore_df.groupby('Chemistry'):
        k = int(df['kmer_length'].iloc[0])
        for pos in range(k):
            grouped = df.groupby(f'base_{pos+1}').agg(
                current_mean=('current_mean', 'mean'),
                current_std=('current_std', 'mean'),
                dwell_time=('dwell_time', 'mean'),
            ).reindex(base_order)
            for base, row in grouped.iterrows():
                effects.append({
                    'Chemistry': chemistry,
                    'position': pos + 1,
                    'base': base,
                    'current_mean_effect': row['current_mean'],
                    'current_std_effect': row['current_std'],
                    'dwell_time_effect': row['dwell_time'],
                })
    effects_df = pd.DataFrame(effects)
    effects_df.to_csv(OUTPUT_DIR / 'substitution_effects.csv', index=False)

    heat_df = effects_df.pivot_table(index=['Chemistry', 'base'], columns='position', values='current_mean_effect')
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heat_df, cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Base-position effects on pore-model current mean')
    fig.savefig(REPORT_IMG_DIR / 'position_effects_heatmap.png')
    plt.close(fig)

    # Chemistry pairwise normalized comparisons at shared position fractions
    pairwise = []
    chem_stats = {}
    for chemistry, df in pore_df.groupby('Chemistry'):
        chem_stats[chemistry] = {
            'current_mean': df['current_mean'].values,
            'current_std': df['current_std'].values,
            'dwell_time': df['dwell_time'].values,
        }
    names = list(chem_stats)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise.append({
                'Chemistry_A': a,
                'Chemistry_B': b,
                'delta_current_mean': float(np.mean(chem_stats[a]['current_mean']) - np.mean(chem_stats[b]['current_mean'])),
                'delta_current_std': float(np.mean(chem_stats[a]['current_std']) - np.mean(chem_stats[b]['current_std'])),
                'delta_dwell_time': float(np.mean(chem_stats[a]['dwell_time']) - np.mean(chem_stats[b]['dwell_time'])),
            })
    pd.DataFrame(pairwise).to_csv(OUTPUT_DIR / 'pore_model_pairwise_deltas.csv', index=False)


def performance_analysis():
    perf = pd.read_csv(DATA_DIR / 'performance_summary.csv')
    baseline = perf[perf['Tool'] == 'Uncalled4'][['Chemistry', 'Time_min', 'FileSize_MB']].rename(
        columns={'Time_min': 'uncalled4_time_min', 'FileSize_MB': 'uncalled4_file_mb'}
    )
    merged = perf.merge(baseline, on='Chemistry', how='left')
    merged['speedup_vs_uncalled4'] = merged['Time_min'] / merged['uncalled4_time_min']
    merged['storage_ratio_vs_uncalled4'] = merged['FileSize_MB'] / merged['uncalled4_file_mb']
    merged['time_saved_min_vs_tool'] = merged['Time_min'] - merged['uncalled4_time_min']
    merged['file_saved_mb_vs_tool'] = merged['FileSize_MB'] - merged['uncalled4_file_mb']
    merged.to_csv(OUTPUT_DIR / 'performance_metrics.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.barplot(data=perf, x='Chemistry', y='Time_min', hue='Tool', ax=axes[0])
    axes[0].set_yscale('log')
    axes[0].set_title('Alignment runtime across chemistries')
    axes[0].tick_params(axis='x', rotation=25)
    sns.barplot(data=perf, x='Chemistry', y='FileSize_MB', hue='Tool', ax=axes[1])
    axes[1].set_yscale('log')
    axes[1].set_title('Alignment output size across chemistries')
    axes[1].tick_params(axis='x', rotation=25)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].legend_.remove()
    axes[1].legend(handles, labels, title='Tool', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.savefig(REPORT_IMG_DIR / 'performance_benchmarks.png')
    plt.close(fig)


def classification_metrics(y_true, scores, threshold=0.5):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        'auroc': float(roc_auc_score(y_true, scores)),
        'auprc': float(average_precision_score(y_true, scores)),
        'accuracy': float(accuracy_score(y_true, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, pred)),
        'precision': float(precision_score(y_true, pred, zero_division=0)),
        'recall': float(recall_score(y_true, pred, zero_division=0)),
        'f1': float(f1_score(y_true, pred, zero_division=0)),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'threshold': threshold,
    }


def m6a_analysis():
    labels = pd.read_csv(DATA_DIR / 'm6a_labels.csv')
    unc = pd.read_csv(DATA_DIR / 'm6a_predictions_uncalled4.csv').rename(columns={'probability': 'uncalled4_probability'})
    nano = pd.read_csv(DATA_DIR / 'm6a_predictions_nanopolish.csv').rename(columns={'probability': 'nanopolish_probability'})

    merged = labels.merge(unc, on='site_id', how='inner').merge(nano, on='site_id', how='inner')
    merged.to_csv(OUTPUT_DIR / 'm6a_joined_predictions.csv', index=False)

    y_true = merged['label'].values
    results = {}
    threshold_rows = []
    score_cols = {
        'Uncalled4': 'uncalled4_probability',
        'Nanopolish': 'nanopolish_probability',
    }

    prevalence = float(np.mean(y_true))
    metrics_rows = []
    for tool, col in score_cols.items():
        scores = merged[col].values
        results[tool] = classification_metrics(y_true, scores, threshold=0.5)
        results[tool]['auroc_bootstrap'] = bootstrap_metric(y_true, scores, roc_auc_score, n_boot=500, seed=SEED)
        results[tool]['auprc_bootstrap'] = bootstrap_metric(y_true, scores, average_precision_score, n_boot=500, seed=SEED + 1)
        metrics_rows.append({
            'tool': tool,
            **results[tool],
            'prevalence': prevalence,
        })
        for threshold in np.linspace(0.05, 0.95, 19):
            row = classification_metrics(y_true, scores, threshold=float(threshold))
            row['tool'] = tool
            threshold_rows.append(row)

    pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / 'm6a_metrics_table.csv', index=False)
    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(OUTPUT_DIR / 'm6a_threshold_metrics.csv', index=False)

    with open(OUTPUT_DIR / 'm6a_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 7))
    for tool, col in score_cols.items():
        precision, recall, _ = precision_recall_curve(y_true, merged[col].values)
        ap = average_precision_score(y_true, merged[col].values)
        ax.plot(recall, precision, label=f'{tool} (AUPRC={ap:.3f})', linewidth=2.5)
    ax.axhline(prevalence, color='gray', linestyle='--', label=f'Prevalence baseline ({prevalence:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('m6A precision-recall comparison')
    ax.legend()
    fig.savefig(REPORT_IMG_DIR / 'm6a_pr_curve.png')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for tool, col in score_cols.items():
        fpr, tpr, _ = roc_curve(y_true, merged[col].values)
        auroc = roc_auc_score(y_true, merged[col].values)
        ax.plot(fpr, tpr, label=f'{tool} (AUROC={auroc:.3f})', linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random baseline')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title('m6A ROC comparison')
    ax.legend()
    fig.savefig(REPORT_IMG_DIR / 'm6a_roc_curve.png')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (tool, col) in zip(axes, score_cols.items()):
        sns.histplot(data=merged, x=col, hue='label', bins=30, stat='density', common_norm=False, ax=ax)
        ax.set_title(f'{tool} score distribution by label')
    fig.savefig(REPORT_IMG_DIR / 'm6a_score_distributions.png')
    plt.close(fig)

    best_f1 = threshold_df.sort_values(['tool', 'f1'], ascending=[True, False]).groupby('tool').head(1)
    best_f1.to_csv(OUTPUT_DIR / 'm6a_best_f1_thresholds.csv', index=False)


def main():
    summarize_inputs()
    pore_model_analysis()
    performance_analysis()
    m6a_analysis()
    print('Analysis complete. Outputs written to outputs/ and report/images/.')


if __name__ == '__main__':
    main()
