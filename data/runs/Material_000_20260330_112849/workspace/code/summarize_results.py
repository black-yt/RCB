import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import ensure_dir

sns.set_theme(style='whitegrid')


def copy_image(src, dst):
    import shutil
    if Path(src).exists():
        ensure_dir(str(Path(dst).parent))
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--report_dir', required=True)
    parser.add_argument('--image_dir', required=True)
    args = parser.parse_args()
    ensure_dir(args.report_dir)
    ensure_dir(args.image_dir)
    rows=[]
    for inp in args.inputs:
        summary_csv = Path(inp) / 'summary_metrics.csv'
        if summary_csv.exists():
            df = pd.read_csv(summary_csv)
            df['run'] = Path(inp).name
            rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    out_csv = Path('outputs') / 'final_comparison.csv'
    all_df.to_csv(out_csv, index=False)
    metric_cols = [c for c in all_df.columns if c.startswith('candidate_metrics_') or c.startswith('val_metrics_')]
    agg = all_df.groupby('run')[metric_cols].mean().reset_index()
    agg.to_csv(Path('outputs') / 'final_comparison_mean.csv', index=False)
    focus = ['val_metrics_pr_auc', 'val_metrics_roc_auc', 'candidate_metrics_recall_at_50', 'candidate_metrics_precision_at_50']
    plot_df = agg.melt(id_vars='run', value_vars=[c for c in focus if c in agg.columns], var_name='metric', value_name='value')
    plt.figure(figsize=(8,4))
    sns.barplot(data=plot_df, x='metric', y='value', hue='run')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout(); plt.savefig(Path(args.image_dir) / 'model_comparison.png', dpi=200); plt.close()
    # copy representative images
    preferred = ['candidate_seed0_baseline_pr_curve.png','candidate_seed0_weighted_pr_curve.png','candidate_seed0_focal_pr_curve.png','candidate_seed0_pretrained_pr_curve.png', 'pretrain_loss.png']
    for inp in args.inputs:
        for file in Path(inp).rglob('*.png'):
            name = file.name
            if name in preferred or 'embedding' in name or 'threshold_sweep' in name or 'roc_curve' in name or 'pr_curve' in name:
                copy_image(file, Path(args.image_dir) / name)
    print(agg)


if __name__ == '__main__':
    main()
