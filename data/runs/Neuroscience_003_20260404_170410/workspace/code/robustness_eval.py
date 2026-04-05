import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from trajectory_feature_selection import (
    SEED,
    compute_full_embedding,
    derive_reference_pseudotime,
    evaluate_subset,
    get_subsets,
    prepare_adata,
    score_features,
)


def run_once(adata, frac, seed):
    rs = np.random.RandomState(seed)
    idx = np.arange(adata.n_obs)
    sample_size = int(len(idx) * frac)
    keep = np.sort(rs.choice(idx, size=sample_size, replace=False))
    sub = adata[keep].copy()
    full_work = compute_full_embedding(sub)
    ref_pseudotime = derive_reference_pseudotime(full_work)
    scores = score_features(sub, ref_pseudotime)
    subsets = get_subsets(scores, sizes=(20, 50))
    rows = []
    for name, feats in subsets.items():
        metrics = evaluate_subset(sub, feats, full_work.obsm['X_umap'], ref_pseudotime)
        metrics.pop('umap')
        metrics['subset'] = name
        metrics['method'] = 'full' if name == 'full' else name.split('_top_')[0]
        metrics['seed'] = seed
        metrics['frac_cells'] = frac
        rows.append(metrics)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--n_runs', type=int, default=3)
    parser.add_argument('--frac_cells', type=float, default=0.85)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    adata = prepare_adata(ad.read_h5ad(args.input))

    runs = []
    for i in range(args.n_runs):
        runs.append(run_once(adata, args.frac_cells, SEED + i))
    df = pd.concat(runs, ignore_index=True)
    df.to_csv(outdir / 'robustness_metrics.csv', index=False)

    summary = df.groupby(['method', 'subset', 'n_features']).agg(
        pseudotime_axis_spearman_mean=('pseudotime_axis_spearman', 'mean'),
        pseudotime_axis_spearman_std=('pseudotime_axis_spearman', 'std'),
        age_neighbor_spearman_mean=('age_neighbor_spearman', 'mean'),
        age_neighbor_spearman_std=('age_neighbor_spearman', 'std'),
        age_neighbor_mse_mean=('age_neighbor_mse', 'mean'),
        age_neighbor_mse_std=('age_neighbor_mse', 'std'),
        mean_batch_entropy_mean=('mean_batch_entropy', 'mean'),
        mean_batch_entropy_std=('mean_batch_entropy', 'std'),
    ).reset_index()
    summary.to_csv(outdir / 'robustness_summary.csv', index=False)


if __name__ == '__main__':
    main()
