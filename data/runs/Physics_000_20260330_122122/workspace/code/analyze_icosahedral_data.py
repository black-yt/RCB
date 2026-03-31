import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path('.')
DATA_FILE = ROOT / 'data' / 'Multi-component Icosahedral Reproduction Data.txt'
OUTPUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def safe_eval(value: str):
    try:
        return ast.literal_eval(value)
    except Exception:
        return eval(value, {"__builtins__": {}}, {})


def parse_dataset(path: Path):
    parsed = {}
    with path.open('r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith('##'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                parsed[key.strip()] = safe_eval(value.strip())
    return parsed


def mismatch(inner_radius, outer_radius):
    return abs(outer_radius - inner_radius) / max(inner_radius, outer_radius)


def category_index(labels):
    return {label: i for i, label in enumerate(labels)}


def shell_count_from_label(label: str):
    digits = ''.join(ch for ch in label if ch.isdigit())
    if digits:
        return int(digits)
    base = {'MC': 1, 'BG': 2}
    return base.get(label, np.nan)


def build_atomic_tables(data):
    radii = pd.DataFrame(data['atomic_radii'], columns=['element', 'radius'])
    compatibility = pd.DataFrame(
        data['atomic_pairs_compatibility'],
        columns=['inner_element', 'outer_element', 'reported_mismatch']
    )
    rad = radii.set_index('element')['radius'].to_dict()
    compatibility['computed_mismatch'] = compatibility.apply(
        lambda r: mismatch(rad[r['inner_element']], rad[r['outer_element']]), axis=1
    )
    compatibility['abs_error'] = (compatibility['computed_mismatch'] - compatibility['reported_mismatch']).abs()
    return radii, compatibility


def build_validation_tables(data):
    optimal = pd.DataFrame(
        data['optimal_mismatch_ranges'],
        columns=['inner_category', 'outer_category', 'lower', 'upper']
    )
    optimal['midpoint'] = (optimal['lower'] + optimal['upper']) / 2

    shell_energies = pd.DataFrame(data['shell_energies'], columns=['shell_index', 'category', 'relative_energy'])
    mismatch_params = pd.DataFrame(
        data['mismatch_params'],
        columns=['shell_i', 'shell_j', 'inner_category', 'outer_category', 'theoretical_mismatch']
    )
    experimental = pd.DataFrame(
        data['experimental_points'],
        columns=['T_i', 'T_j', 'measured_sm', 'theoretical_sm']
    )
    experimental['residual'] = experimental['measured_sm'] - experimental['theoretical_sm']
    experimental['abs_residual'] = experimental['residual'].abs()
    return optimal, shell_energies, mismatch_params, experimental


def build_cluster_predictions(data, radii_df, optimal_df, shell_energies_df):
    rad = radii_df.set_index('element')['radius'].to_dict()
    clusters = pd.DataFrame(
        data['multicomponent_clusters'],
        columns=['cluster', 'inner_element', 'outer_element', 'inner_category', 'outer_category']
    )
    clusters['computed_mismatch'] = clusters.apply(
        lambda r: mismatch(rad[r['inner_element']], rad[r['outer_element']]), axis=1
    )

    optimal_map = optimal_df.set_index(['inner_category', 'outer_category'])[['lower', 'upper', 'midpoint']].to_dict('index')
    energy_map = shell_energies_df.set_index(['shell_index', 'category'])['relative_energy'].to_dict()
    cat_idx = category_index(data['chiral_labels'])

    def infer_shell_indices(row):
        inner_idx = 1
        outer_idx = max(2, cat_idx.get(row['outer_category'], 1) + 1)
        return pd.Series({'inner_shell_index': inner_idx, 'outer_shell_index': outer_idx})

    clusters[['inner_shell_index', 'outer_shell_index']] = clusters.apply(infer_shell_indices, axis=1)

    def lookup_range(row):
        info = optimal_map.get((row['inner_category'], row['outer_category']))
        if info is None:
            return pd.Series({'lower': np.nan, 'upper': np.nan, 'target_midpoint': np.nan})
        return pd.Series({'lower': info['lower'], 'upper': info['upper'], 'target_midpoint': info['midpoint']})

    clusters[['lower', 'upper', 'target_midpoint']] = clusters.apply(lookup_range, axis=1)
    clusters['within_optimal_range'] = clusters.apply(
        lambda r: bool(r['lower'] <= r['computed_mismatch'] <= r['upper']) if pd.notna(r['lower']) else False, axis=1
    )
    clusters['mismatch_offset'] = clusters['computed_mismatch'] - clusters['target_midpoint']
    clusters['inner_energy'] = clusters.apply(
        lambda r: energy_map.get((int(r['inner_shell_index']), r['inner_category']), np.nan), axis=1
    )
    clusters['outer_energy'] = clusters.apply(
        lambda r: energy_map.get((int(r['outer_shell_index']), r['outer_category']), np.nan), axis=1
    )
    clusters['estimated_total_energy'] = clusters[['inner_energy', 'outer_energy']].sum(axis=1, min_count=1)
    return clusters


def enumerate_candidate_pairs(data, radii_df, optimal_df, shell_energies_df):
    elements = radii_df['element'].tolist()
    rad = radii_df.set_index('element')['radius'].to_dict()
    labels = data['chiral_labels']
    cat_idx = category_index(labels)
    energy_map = shell_energies_df.groupby('category')['relative_energy'].min().to_dict()

    rows = []
    for _, opt in optimal_df.iterrows():
        for inner in elements:
            for outer in elements:
                if inner == outer:
                    continue
                cm = mismatch(rad[inner], rad[outer])
                target = opt['midpoint']
                score = abs(cm - target)
                energy_bonus = -energy_map.get(opt['outer_category'], 0.0)
                composite_score = score + 0.02 * energy_bonus + 0.002 * cat_idx.get(opt['outer_category'], 0)
                rows.append({
                    'inner_element': inner,
                    'outer_element': outer,
                    'inner_category': opt['inner_category'],
                    'outer_category': opt['outer_category'],
                    'computed_mismatch': cm,
                    'target_midpoint': target,
                    'range_lower': opt['lower'],
                    'range_upper': opt['upper'],
                    'within_range': opt['lower'] <= cm <= opt['upper'],
                    'distance_to_target': score,
                    'energy_bonus': energy_bonus,
                    'composite_score': composite_score,
                })
    candidates = pd.DataFrame(rows)
    candidates = candidates.sort_values(['within_range', 'composite_score', 'distance_to_target'], ascending=[False, True, True])
    return candidates


def build_growth_tables(data):
    growth = pd.DataFrame(data['growth_results'], columns=['step', 'category', 'avg_mismatch'])
    growth['run_id'] = 0
    run_id = 0
    prev_step = -1
    run_ids = []
    for step in growth['step']:
        if step <= prev_step:
            run_id += 1
        run_ids.append(run_id)
        prev_step = step
    growth['run_id'] = run_ids

    path_stats = pd.DataFrame(data['path_selection_stats'], columns=['path_type', 'count'])
    path_stats['fraction'] = path_stats['count'] / path_stats['count'].sum()

    growth_params = pd.DataFrame(data['growth_parameters'], columns=['parameter', 'value'])
    return growth, path_stats, growth_params


def make_figures(radii_df, compatibility_df, clusters_df, candidates_df, experimental_df, growth_df, path_stats_df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    ordered_radii = radii_df.sort_values('radius')
    sns.barplot(data=ordered_radii, x='element', y='radius', hue='element', palette='viridis', legend=False)
    plt.ylabel('Atomic radius (Å)')
    plt.xlabel('Element')
    plt.title('Atomic size hierarchy in the reproduction dataset')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'atomic_radii_overview.png', dpi=300)
    plt.close()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=compatibility_df, x='reported_mismatch', y='computed_mismatch', hue='inner_element', s=140)
    lims = [0, max(compatibility_df[['reported_mismatch', 'computed_mismatch']].max()) * 1.1]
    plt.plot(lims, lims, '--', color='black', linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Reported size mismatch')
    plt.ylabel('Radius-derived size mismatch')
    plt.title('Consistency of reported and radius-derived mismatches')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'reported_vs_computed_mismatch.png', dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    ordered = clusters_df.sort_values('computed_mismatch')
    x = np.arange(len(ordered))
    plt.errorbar(x, ordered['target_midpoint'], 
                 yerr=[ordered['target_midpoint'] - ordered['lower'], ordered['upper'] - ordered['target_midpoint']],
                 fmt='o', capsize=5, label='Optimal mismatch window midpoint ± range')
    plt.scatter(x, ordered['computed_mismatch'], color='crimson', s=110, label='Cluster mismatch from radii')
    plt.xticks(x, ordered['cluster'], rotation=20)
    plt.ylabel('Size mismatch')
    plt.xlabel('Validated cluster')
    plt.title('Validated multicomponent clusters vs optimal mismatch windows')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'validated_clusters_mismatch.png', dpi=300)
    plt.close()

    top_candidates = candidates_df.head(10).copy()
    top_candidates['pair'] = top_candidates['inner_element'] + '@' + top_candidates['outer_element']
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_candidates, y='pair', x='computed_mismatch', hue='outer_category', dodge=False, palette='tab10')
    for i, (_, row) in enumerate(top_candidates.iterrows()):
        plt.plot([row['range_lower'], row['range_upper']], [i, i], color='black', linewidth=2)
    plt.xlabel('Computed mismatch')
    plt.ylabel('Candidate pair')
    plt.title('Top predicted shell pairings and their target mismatch windows')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top_predicted_pairs.png', dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=experimental_df, x='theoretical_sm', y='measured_sm', s=130)
    lims = [0, max(experimental_df[['theoretical_sm', 'measured_sm']].max()) * 1.1]
    plt.plot(lims, lims, '--', color='black', linewidth=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Theoretical mismatch')
    plt.ylabel('Measured mismatch')
    plt.title('Experimental validation of theoretical mismatch predictions')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'experimental_validation.png', dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=growth_df, x='step', y='avg_mismatch', hue='category', style='run_id', markers=True, dashes=False)
    plt.xlabel('Growth step')
    plt.ylabel('Average mismatch')
    plt.title('Growth trajectories across shell categories')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'growth_trajectories.png', dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.barplot(data=path_stats_df, x='path_type', y='fraction', hue='path_type', palette='magma', legend=False)
    plt.ylabel('Fraction of selected paths')
    plt.xlabel('Path type')
    plt.title('Path selection statistics in growth simulations')
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'path_selection_statistics.png', dpi=300)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    data = parse_dataset(DATA_FILE)
    radii_df, compatibility_df = build_atomic_tables(data)
    optimal_df, shell_energies_df, mismatch_params_df, experimental_df = build_validation_tables(data)
    clusters_df = build_cluster_predictions(data, radii_df, optimal_df, shell_energies_df)
    candidates_df = enumerate_candidate_pairs(data, radii_df, optimal_df, shell_energies_df)
    growth_df, path_stats_df, growth_params_df = build_growth_tables(data)

    top_predictions = candidates_df.head(12).copy()
    top_predictions['predicted_cluster_label'] = [
        f"{row.inner_element}13@{row.outer_element}{32 if row.outer_category=='Ch1' else 42 if row.outer_category=='Ch2' else 55}"
        for _, row in top_predictions.iterrows()
    ]

    summary = {
        'n_elements': int(len(radii_df)),
        'n_validated_clusters': int(len(clusters_df)),
        'mean_atomic_pair_abs_error': float(compatibility_df['abs_error'].mean()),
        'experimental_mae': float(experimental_df['abs_residual'].mean()),
        'top_within_range_predictions': int(top_predictions['within_range'].sum()),
        'most_common_growth_category': str(growth_df['category'].mode().iat[0]),
        'dominant_path_type': str(path_stats_df.sort_values('count', ascending=False).iloc[0]['path_type'])
    }

    radii_df.to_csv(OUTPUT_DIR / 'atomic_radii.csv', index=False)
    compatibility_df.to_csv(OUTPUT_DIR / 'atomic_pair_compatibility_analysis.csv', index=False)
    optimal_df.to_csv(OUTPUT_DIR / 'optimal_mismatch_ranges.csv', index=False)
    shell_energies_df.to_csv(OUTPUT_DIR / 'shell_energies.csv', index=False)
    mismatch_params_df.to_csv(OUTPUT_DIR / 'mismatch_params.csv', index=False)
    experimental_df.to_csv(OUTPUT_DIR / 'experimental_validation.csv', index=False)
    clusters_df.to_csv(OUTPUT_DIR / 'validated_clusters_analysis.csv', index=False)
    candidates_df.to_csv(OUTPUT_DIR / 'candidate_shell_pairings_ranked.csv', index=False)
    top_predictions.to_csv(OUTPUT_DIR / 'top_predicted_structures.csv', index=False)
    growth_df.to_csv(OUTPUT_DIR / 'growth_results_annotated.csv', index=False)
    path_stats_df.to_csv(OUTPUT_DIR / 'path_selection_stats.csv', index=False)
    growth_params_df.to_csv(OUTPUT_DIR / 'growth_parameters.csv', index=False)

    make_figures(radii_df, compatibility_df, clusters_df, candidates_df, experimental_df, growth_df, path_stats_df)

    with (OUTPUT_DIR / 'summary_metrics.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
