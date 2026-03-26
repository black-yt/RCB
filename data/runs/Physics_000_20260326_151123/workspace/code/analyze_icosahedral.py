import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / 'data' / 'Multi-component Icosahedral Reproduction Data.txt'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'

sns.set_theme(style='whitegrid', context='talk')


def load_dataset(path: Path):
    namespace = {'__builtins__': {}}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('##'):
            continue
        if '=' not in line:
            continue
        key, value = line.split('=', 1)
        namespace[key.strip()] = eval(value.strip(), {'__builtins__': {}}, namespace)
    return namespace


def build_shell_type_map(data):
    shell_map = {(0, 0): 'MC'}
    coords = data['hexagonal_coords']
    labels = data['chiral_labels']
    for coord in coords:
        if coord == (0, 0):
            continue
        d = max(coord)
        ring_index = min(d, len(labels) - 1)
        shell_map[coord] = labels[ring_index]
    return shell_map


def shell_sequence_path(data):
    coords = data['hexagonal_coords']
    ordered = sorted(coords, key=lambda c: (max(c), c[0] + c[1], c[0], c[1]))
    return ordered


def build_atomic_tables(data):
    radii = dict(data['atomic_radii'])
    pairs = []
    for a, b, mismatch in data['atomic_pairs_compatibility']:
        if radii[a] >= radii[b]:
            core, shell = b, a
        else:
            core, shell = a, b
        calc = radii[shell] / radii[core] - 1.0
        pairs.append({
            'core': core,
            'shell': shell,
            'reported_mismatch': mismatch,
            'radius_ratio_mismatch': calc,
            'abs_error': abs(calc - mismatch),
        })
    return pd.DataFrame(pairs), radii


def compute_cluster_predictions(data, shell_map, radii):
    shell_sizes = {
        'MC': 12,
        'BG': 30,
        'Ch1': 32,
        'Ch2': 42,
        'Ch3': 54,
        'Ch4': 68,
        'Ch5': 84,
    }
    optimal_ranges = {(a, b): ((lo + hi) / 2.0, lo, hi) for a, b, lo, hi in data['optimal_mismatch_ranges']}
    energy_lookup = {}
    for shell_number, shell_type, energy in data['shell_energies']:
        energy_lookup.setdefault(shell_type, {})[shell_number] = energy

    predictions = []
    coords = shell_sequence_path(data)
    for coord in coords[:16]:
        shell_type = shell_map[coord]
        if shell_type == 'MC':
            continue
        target, lo, hi = optimal_ranges.get(('MC', shell_type), (np.nan, np.nan, np.nan))
        for core in radii:
            for shell in radii:
                if core == shell:
                    continue
                mismatch = radii[shell] / radii[core] - 1.0
                if np.isnan(target):
                    score = np.nan
                else:
                    score = abs(mismatch - target)
                total_atoms = 13 + shell_sizes[shell_type]
                predictions.append({
                    'coord': coord,
                    'shell_type': shell_type,
                    'core_element': core,
                    'shell_element': shell,
                    'core_size': 13,
                    'shell_size': shell_sizes[shell_type],
                    'formula': f'{core}13@{shell}{shell_sizes[shell_type]}',
                    'mismatch': mismatch,
                    'target_mismatch': target,
                    'mismatch_error': score,
                    'in_optimal_window': False if np.isnan(target) else (lo <= mismatch <= hi),
                    'estimated_energy': energy_lookup.get(shell_type, {}).get(2, np.nan) - 10 * (0 if np.isnan(score) else score),
                    'path_index': coords.index(coord),
                })
    df = pd.DataFrame(predictions)
    df = df.sort_values(['mismatch_error', 'estimated_energy'], ascending=[True, True]).reset_index(drop=True)
    return df


def growth_simulation(data, shell_map):
    params = dict(data['growth_parameters'])
    weights = dict(data['path_probability_weights'])
    rng = np.random.default_rng(int(params['random_seed']))
    steps = int(params['simulation_steps'])
    delta_opt = float(params['delta_opt'])

    ordered = shell_sequence_path(data)
    shell_targets = {'MC': 0.04, 'BG': 0.09, 'Ch1': 0.14, 'Ch2': 0.205}
    current_idx = 0
    current_type = shell_map[ordered[current_idx]]
    mismatch = 0.0
    trajectory = []

    probs = np.array([
        weights['conservative_step'],
        weights['mismatch_driven_step'],
        weights['random_step'],
        1 - sum(weights.values())
    ])
    labels = np.array(['conservative', 'mismatch_driven', 'random', 'reverse'])

    for step in range(steps + 1):
        trajectory.append({
            'step': step,
            'path_index': current_idx,
            'coord': ordered[current_idx],
            'shell_type': current_type,
            'avg_mismatch': mismatch,
        })
        if step == steps:
            break
        action = rng.choice(labels, p=probs)
        if action in {'conservative', 'mismatch_driven'} and current_idx < len(ordered) - 1:
            nxt = current_idx + 1
            nxt_type = shell_map[ordered[nxt]]
            target = shell_targets.get(nxt_type, shell_targets.get(current_type, delta_opt))
            if action == 'conservative':
                mismatch = mismatch + 0.2 * (target - mismatch)
            else:
                mismatch = mismatch + 0.5 * (target - mismatch)
            current_idx = nxt
            current_type = nxt_type
        elif action == 'random':
            mismatch = max(0.0, mismatch + rng.normal(0, 0.02))
        elif action == 'reverse' and current_idx > 0:
            current_idx -= 1
            current_type = shell_map[ordered[current_idx]]
            mismatch = max(0.0, mismatch - 0.01)

    return pd.DataFrame(trajectory)


def make_figures(data, atomic_df, pred_df, growth_df):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    radii_df = pd.DataFrame(data['atomic_radii'], columns=['element', 'radius'])
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=radii_df, x='element', y='radius', color='#4c78a8')
    ax.set_title('Atomic radii used in reproduction dataset')
    ax.set_ylabel('Atomic radius (Å)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'atomic_radii.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=atomic_df, x='reported_mismatch', y='radius_ratio_mismatch', s=120)
    lims = [0, max(atomic_df['reported_mismatch'].max(), atomic_df['radius_ratio_mismatch'].max()) + 0.05]
    ax.plot(lims, lims, '--', color='gray')
    for _, row in atomic_df.iterrows():
        ax.text(row['reported_mismatch'] + 0.003, row['radius_ratio_mismatch'] + 0.003, f"{row['core']}-{row['shell']}", fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('Reported vs. radius-derived size mismatch')
    ax.set_xlabel('Reported mismatch')
    ax.set_ylabel('Radius-derived mismatch')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'mismatch_validation.png', dpi=200)
    plt.close()

    top = pred_df.head(12).copy()
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=top, y='formula', x='mismatch_error', hue='shell_type', dodge=False, palette='viridis')
    ax.set_title('Top predicted two-shell icosahedral candidates')
    ax.set_xlabel('|mismatch - target|')
    ax.set_ylabel('Candidate cluster')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'top_candidates.png', dpi=200)
    plt.close()

    sampled = growth_df.iloc[::max(1, len(growth_df)//200)].copy()
    plt.figure(figsize=(10, 5))
    ax = sns.lineplot(data=sampled, x='step', y='avg_mismatch', hue='shell_type', palette='tab10')
    ax.set_title('Simulated growth trajectory along shell-sequence path')
    ax.set_ylabel('Average mismatch')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'growth_trajectory.png', dpi=200)
    plt.close()

    exp_df = pd.DataFrame(data['experimental_points'], columns=['Ti', 'Tip1', 'measured', 'theoretical'])
    plt.figure(figsize=(6, 6))
    ax = sns.scatterplot(data=exp_df, x='theoretical', y='measured', s=140, color='#e45756')
    lims = [0, max(exp_df['measured'].max(), exp_df['theoretical'].max()) + 0.02]
    ax.plot(lims, lims, '--', color='gray')
    for _, row in exp_df.iterrows():
        ax.text(row['theoretical'] + 0.002, row['measured'] + 0.002, f"{int(row['Ti'])}->{int(row['Tip1'])}", fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('Experimental vs theoretical mismatch points')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'experimental_comparison.png', dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset(DATA_PATH)
    shell_map = build_shell_type_map(data)
    atomic_df, radii = build_atomic_tables(data)
    pred_df = compute_cluster_predictions(data, shell_map, radii)
    growth_df = growth_simulation(data, shell_map)

    path_df = pd.DataFrame([
        {'path_index': i, 'coord': coord, 'shell_type': shell_map[coord]}
        for i, coord in enumerate(shell_sequence_path(data))
    ])

    atomic_df.to_csv(OUT_DIR / 'atomic_pair_analysis.csv', index=False)
    pred_df.to_csv(OUT_DIR / 'predicted_candidates.csv', index=False)
    growth_df.to_csv(OUT_DIR / 'growth_simulation.csv', index=False)
    path_df.to_csv(OUT_DIR / 'shell_path_sequence.csv', index=False)

    summary = {
        'n_coords': len(data['hexagonal_coords']),
        'n_candidate_pairs': int(len(pred_df)),
        'best_candidate': pred_df.iloc[0]['formula'],
        'best_shell_type': pred_df.iloc[0]['shell_type'],
        'best_mismatch': float(pred_df.iloc[0]['mismatch']),
        'best_error': float(pred_df.iloc[0]['mismatch_error']),
        'growth_final_type': growth_df.iloc[-1]['shell_type'],
        'growth_final_mismatch': float(growth_df.iloc[-1]['avg_mismatch']),
    }
    pd.Series(summary).to_json(OUT_DIR / 'summary.json', indent=2)

    make_figures(data, atomic_df, pred_df, growth_df)
    print('Analysis complete')


if __name__ == '__main__':
    main()
