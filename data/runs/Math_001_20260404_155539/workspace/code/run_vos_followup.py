import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('.')
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'


def main():
    with open(OUT / 'metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    traces = np.load(OUT / 'traces.npz')

    methods = ['ISTA', 'FISTA', 'Restarted_FISTA', 'ADMM']
    checkpoints = [5, 10, 25, 50, 100, 250]
    table = {}
    for m in methods:
        gap = traces[f'{m}_gap']
        row = {}
        for c in checkpoints:
            idx = min(c - 1, len(gap) - 1)
            row[f'gap_at_{c}'] = float(gap[idx])
        thresholds = [1e-1, 1e-2, 1e-3, 1e-4]
        for th in thresholds:
            reached = np.where(gap <= th)[0]
            row[f'iter_to_gap_le_{th}'] = int(reached[0] + 1) if len(reached) else None
        table[m] = row

    with open(OUT / 'early_iteration_summary.json', 'w', encoding='utf-8') as f:
        json.dump(table, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in methods:
        gap = traces[f'{m}_gap'][:100]
        ax.semilogy(np.arange(1, len(gap) + 1), gap, lw=2, label=m.replace('_', ' '))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective gap to reference')
    ax.set_title('Early-iteration convergence (first 100 iterations)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG / 'early_iteration_convergence.png', dpi=200)
    plt.close(fig)

    print('Saved follow-up summary and figure.')


if __name__ == '__main__':
    main()
