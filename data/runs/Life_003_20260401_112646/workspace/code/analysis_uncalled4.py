import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc

sns.set(style="whitegrid")

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'

for d in [OUT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def load_pore_model(fname: str) -> pd.DataFrame:
    path = DATA_DIR / fname
    df = pd.read_csv(path)
    # Standardize column names if needed
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def summarize_pore_models():
    files = [
        'dna_r9.4.1_400bps_6mer_uncalled4.csv',
        'dna_r10.4.1_400bps_9mer_uncalled4.csv',
        'rna_r9.4.1_70bps_5mer_uncalled4.csv',
        'rna004_130bps_9mer_uncalled4.csv',
    ]
    summaries = []
    for f in files:
        df = load_pore_model(f)
        n_kmer = len(df)
        mean_mean = df['mean'].mean() if 'mean' in df.columns else np.nan
        mean_std = df['std'].mean() if 'std' in df.columns else np.nan
        mean_dwell = df['dwell'].mean() if 'dwell' in df.columns else np.nan
        summaries.append({
            'file': f,
            'n_kmer': n_kmer,
            'mean_mean': mean_mean,
            'mean_std': mean_std,
            'mean_dwell': mean_dwell,
        })
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / 'pore_model_summaries.csv', index=False)

    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary_df, x='file', y='mean_mean')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average current mean (pA)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pore_model_mean_current.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.barplot(data=summary_df, x='file', y='mean_dwell')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Average dwell time')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'pore_model_mean_dwell.png', dpi=300)
    plt.close()


def analyze_performance():
    perf_path = DATA_DIR / 'performance_summary.csv'
    perf = pd.read_csv(perf_path)
    perf.columns = [c.strip().lower() for c in perf.columns]
    perf.to_csv(OUT_DIR / 'performance_summary_clean.csv', index=False)

    # Assume columns: chemistry, tool, time_sec, bam_size_mb or similar
    time_cols = [c for c in perf.columns if 'time' in c]
    size_cols = [c for c in perf.columns if 'size' in c or 'bam' in c]

    plt.figure(figsize=(7, 4))
    sns.barplot(data=perf, x='chemistry', y=time_cols[0], hue='tool')
    plt.ylabel('Alignment time (s)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'performance_time.png', dpi=300)
    plt.close()

    if size_cols:
        plt.figure(figsize=(7, 4))
        sns.barplot(data=perf, x='chemistry', y=size_cols[0], hue='tool')
        plt.ylabel('Alignment file size (MB)')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'performance_size.png', dpi=300)
        plt.close()


def evaluate_m6a_predictions():
    labels = pd.read_csv(DATA_DIR / 'm6a_labels.csv')
    u4 = pd.read_csv(DATA_DIR / 'm6a_predictions_uncalled4.csv')
    npol = pd.read_csv(DATA_DIR / 'm6a_predictions_nanopolish.csv')

    # Standardize
    for df in (labels, u4, npol):
        df.columns = [c.strip().lower() for c in df.columns]

    # Assume a common key column, e.g. 'site' or 'id'
    key = [c for c in labels.columns if c not in {'label', 'y', 'y_true', 'truth'}]
    if key:
        on = key
    else:
        labels['idx'] = np.arange(len(labels))
        u4['idx'] = np.arange(len(u4))
        npol['idx'] = np.arange(len(npol))
        on = ['idx']

    # Identify label column
    label_col = None
    for cand in ['label', 'y', 'y_true', 'truth']:
        if cand in labels.columns:
            label_col = cand
            break
    if label_col is None:
        raise ValueError('Could not find label column in m6a_labels.csv')

    # Identify prediction columns
    pred_cols_u4 = [c for c in u4.columns if c not in on]
    pred_cols_npol = [c for c in npol.columns if c not in on]
    if not pred_cols_u4 or not pred_cols_npol:
        raise ValueError('Prediction columns not found in prediction files')

    u4_pred_col = pred_cols_u4[0]
    npol_pred_col = pred_cols_npol[0]

    merged = labels.merge(u4[on + [u4_pred_col]], on=on).merge(
        npol[on + [npol_pred_col]], on=on
    )
    merged.to_csv(OUT_DIR / 'm6a_merged_predictions.csv', index=False)

    # Disambiguate column names after merge
    if u4_pred_col in merged.columns and npol_pred_col in merged.columns:
        u4_col = u4_pred_col
        npol_col = npol_pred_col
    else:
        # Fall back to suffix-based names from pandas merge
        u4_col = u4_pred_col + '_x' if u4_pred_col + '_x' in merged.columns else [c for c in merged.columns if 'probability' in c][0]
        npol_col = u4_pred_col + '_y' if u4_pred_col + '_y' in merged.columns else [c for c in merged.columns if 'probability' in c][-1]

    y_true = merged[label_col].values
    y_u4 = merged[u4_col].values
    y_npol = merged[npol_col].values

    # Precision-recall
    pr_u4 = precision_recall_curve(y_true, y_u4)
    pr_npol = precision_recall_curve(y_true, y_npol)
    ap_u4 = average_precision_score(y_true, y_u4)
    ap_npol = average_precision_score(y_true, y_npol)

    plt.figure(figsize=(5, 5))
    plt.plot(pr_u4[1], pr_u4[0], label=f'Uncalled4 (AP={ap_u4:.3f})')
    plt.plot(pr_npol[1], pr_npol[0], label=f'Nanopolish (AP={ap_npol:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('m6A detection precision-recall')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'm6a_precision_recall.png', dpi=300)
    plt.close()

    # ROC
    fpr_u4, tpr_u4, _ = roc_curve(y_true, y_u4)
    fpr_npol, tpr_npol, _ = roc_curve(y_true, y_npol)
    auc_u4 = auc(fpr_u4, tpr_u4)
    auc_npol = auc(fpr_npol, tpr_npol)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr_u4, tpr_u4, label=f'Uncalled4 (AUC={auc_u4:.3f})')
    plt.plot(fpr_npol, tpr_npol, label=f'Nanopolish (AUC={auc_npol:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('m6A detection ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'm6a_roc.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    summarize_pore_models()
    analyze_performance()
    evaluate_m6a_predictions()
