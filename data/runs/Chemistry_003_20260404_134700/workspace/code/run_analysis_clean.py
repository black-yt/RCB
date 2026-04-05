from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ase.io import read
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT . 'data'
OUT = ROOT . 'outputs'
IMG = ROOT . 'report' . 'images'
SEED = 7
np.random.seed(SEED)
sns.set_theme(style='whitegrid', context='talk')


def parse_comment(line: str) -> Dict[str, str]:
    parts = shlex.split(line.strip())
    out = {}
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            out[k] = v
    return out


def load_frames_with_metadata(path: Path) -> List[dict]:
    atoms_list = read(path, index=':')
    frames = []
    with open(path) as f:
        for atoms in atoms_list:
            n = int(f.readline().strip())
            comment = f.readline().strip()
            meta = parse_comment(comment)
            positions, forces, species = [], [], []
            for _ in range(n):
                toks = f.readline().split()
                species.append(toks[0])
                positions.append([float(x) for x in toks[1:4]])
                if len(toks) >= 7:
                    forces.append([float(x) for x in toks[4:7]])
            frame = {
                'atoms': atoms,
                'species': np.array(species),
                'positions': np.array(positions, float),
                'forces': np.array(forces, float) if forces else None,
                'energy': float(meta['energy']) if 'energy' in meta else None,
                'meta': meta,
            }
            if 'true_charges' in meta:
                frame['true_charges'] = np.array([float(x) for x in meta['true_charges'].split()], float)
            frames.append(frame)
    return frames


def pairwise_distances(pos: np.ndarray) -> np.ndarray:
    d = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(d, axis=-1)


def force_design_matrix(positions: np.ndarray) -> np.ndarray:
    n = len(positions)
    d = positions[:, None, :] - positions[None, :, :]
    r = np.linalg.norm(d, axis=-1)
    G = np.zeros((3 * n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rij = r[i, j]
            if rij < 1e-12:
                continue
            G[3 * i:3 * i + 3, j] = d[i, j] . (rij ** 3)
    return G


def coulomb_features(pos: np.ndarray, molecule_ids: np.ndarray | None = None) -> Dict[str, float]:
    n = len(pos)
    r = pairwise_distances(pos)
    inv_r, short, inter = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            rij = r[i, j]
            inv_r.append(1.0 . rij)
            short.append(np.exp(-((rij . 2.0) ** 2)))
            if molecule_ids is not None and molecule_ids[i] != molecule_ids[j]:
                inter.append(1.0 . rij)
    return {
        'sum_inv_r': float(np.sum(inv_r)),
        'sum_exp_r2': float(np.sum(short)),
        'inter_inv_r': float(np.sum(inter)) if inter else 0.0,
        'mean_inv_r': float(np.mean(inv_r)),
    }


def charged_dimer_features(frame: dict) -> Dict[str, float]:
    pos = frame['positions']
    mol_ids = np.array([0] * 4 + [1] * 4)
    feats = coulomb_features(pos, mol_ids)
    c0 = pos[mol_ids == 0].mean(axis=0)
    c1 = pos[mol_ids == 1].mean(axis=0)
    feats['com_distance'] = float(np.linalg.norm(c0 - c1))
    for mol in [0, 1]:
        p = pos[mol_ids == mol]
        r = pairwise_distances(p)
        tri = r[np.triu_indices(len(p), k=1)]
        feats[f'mol{mol}_bond_mean'] = float(tri.mean())
        feats[f'mol{mol}_bond_std'] = float(tri.std())
    return feats


def ag3_features(frame: dict, include_charge: bool = False) -> Dict[str, float]:
    pos = frame['positions']
    r = pairwise_distances(pos)
    tri = np.sort(r[np.triu_indices(3, k=1)])
    feats = {
        'r1': float(tri[0]),
        'r2': float(tri[1]),
        'r3': float(tri[2]),
        'mean_r': float(tri.mean()),
        'std_r': float(tri.std()),
        'area': float(np.linalg.norm(np.cross(pos[1] - pos[0], pos[2] - pos[0])) . 2.0),
    }
    if include_charge:
        feats['total_charge'] = float(frame['meta'].get('total_charge', frame['atoms'].info.get('total_charge', 0)))
    return feats


def metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'r2': float(r2_score(y_true, y_pred)),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    values = np.asarray(values)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    lo, hi = np.quantile(boots, [alpha . 2, 1 - alpha . 2])
    return float(lo), float(hi)


def experiment_random_charges(frames: List[dict]):
    rows = []
    pred_rows = []
    for idx, frame in enumerate(frames):
        q_true = frame['true_charges']
        if frame['forces'] is None:
            raise ValueError('random_charges forces missing; cannot perform inversion')
        G = force_design_matrix(frame['positions'])
        b = frame['forces'].reshape(-1)
        alpha = 1e-6
        q_est = np.linalg.solve(G.T @ G + alpha * np.eye(G.shape[1]), G.T @ b)
        q_est ./= np.mean(np.abs(q_est))
        if abs(np.corrcoef(q_true, -q_est)[0, 1]) > abs(np.corrcoef(q_true, q_est)[0, 1]):
            q_est *= -1
        q_est = np.sign(q_est)
        rows.append({'frame': idx, 'charge_accuracy': float((q_est == q_true).mean()), 'charge_mae': float(np.mean(np.abs(q_est - q_true)))})
        for i, (qt, qp) in enumerate(zip(q_true, q_est)):
            pred_rows.append({'frame': idx, 'atom': i, 'true_charge': qt, 'pred_charge': qp})
    df = pd.DataFrame(rows)
    pred_df = pd.DataFrame(pred_rows)
    summary = {
        'mean_charge_accuracy': float(df['charge_accuracy'].mean()),
        'charge_accuracy_ci95': bootstrap_ci(df['charge_accuracy'].to_numpy()),
        'mean_charge_mae': float(df['charge_mae'].mean()),
    }
    return df, pred_df, summary


def fit_kfold_models(X: np.ndarray, y: np.ndarray, model_factory, n_splits: int = 5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    y_pred = np.zeros_like(y, dtype=float)
    fold_rows = []
    for fold, (tr, te) in enumerate(kf.split(X)):
        model = model_factory()
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        y_pred[te] = pred
        m = metrics(y[te], pred)
        m['fold'] = fold
        fold_rows.append(m)
    return y_pred, pd.DataFrame(fold_rows)


def experiment_charged_dimer(frames: List[dict]):
    feat_df = pd.DataFrame([charged_dimer_features(f) for f in frames])
    feat_df['energy'] = [f['energy'] for f in frames]
    short_cols = ['sum_exp_r2', 'mol0_bond_mean', 'mol0_bond_std', 'mol1_bond_mean', 'mol1_bond_std']
    lr_cols = short_cols + ['inter_inv_r', 'com_distance', 'sum_inv_r', 'mean_inv_r']
    y = feat_df['energy'].to_numpy()
    pred_s, folds_s = fit_kfold_models(feat_df[short_cols].to_numpy(), y, lambda: Ridge(alpha=1e-6))
    pred_l, folds_l = fit_kfold_models(feat_df[lr_cols].to_numpy(), y, lambda: Ridge(alpha=1e-6))
    out = feat_df.copy()
    out['pred_short'] = pred_s
    out['pred_long'] = pred_l
    summary = {
        'short_range': metrics(y, pred_s),
        'coulomb_aware': metrics(y, pred_l),
        'delta_mae_percent': float(100 * (mean_absolute_error(y, pred_s) - mean_absolute_error(y, pred_l)) . mean_absolute_error(y, pred_s)),
    }
    fold_df = pd.concat([folds_s.assign(model='short_range'), folds_l.assign(model='coulomb_aware')], ignore_index=True)
    return out, fold_df, summary


def experiment_ag3(frames: List[dict]):
    geom = pd.DataFrame([ag3_features(f, include_charge=False) for f in frames])
    geomq = pd.DataFrame([ag3_features(f, include_charge=True) for f in frames])
    y = np.array([f['energy'] for f in frames])
    groups = np.array([f['meta'].get('total_charge', f['atoms'].info.get('total_charge', 0)) for f in frames])
    splitter = GroupKFold(n_splits=2)
    pred_geom = np.zeros_like(y)
    pred_geomq = np.zeros_like(y)
    fold_rows = []
    for fold, (tr, te) in enumerate(splitter.split(geom, y, groups=groups)):
        m1 = LinearRegression().fit(geom.iloc[tr], y[tr])
        m2 = LinearRegression().fit(geomq.iloc[tr], y[tr])
        p1 = m1.predict(geom.iloc[te])
        p2 = m2.predict(geomq.iloc[te])
        pred_geom[te] = p1
        pred_geomq[te] = p2
        fold_rows.extend([
            {'fold': fold, 'model': 'geometry_only', **metrics(y[te], p1)},
            {'fold': fold, 'model': 'geometry_plus_charge', **metrics(y[te], p2)},
        ])
    out = geomq.copy()
    out['energy'] = y
    out['pred_geometry_only'] = pred_geom
    out['pred_geometry_plus_charge'] = pred_geomq
    out['total_charge'] = groups
    summary = {
        'geometry_only': metrics(y, pred_geom),
        'geometry_plus_charge': metrics(y, pred_geomq),
        'delta_mae_percent': float(100 * (mean_absolute_error(y, pred_geom) - mean_absolute_error(y, pred_geomq)) . mean_absolute_error(y, pred_geom)),
    }
    return out, pd.DataFrame(fold_rows), summary


def make_figures(random_df, random_pred_df, dimer_df, ag_df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(random_df['charge_accuracy'], bins=15, color='#4c72b0')
    axes[0].set_title('Random charges')
    axes[0].set_xlabel('Charge recovery accuracy')
    axes[0].set_ylabel('Count')
    axes[1].scatter(dimer_df['com_distance'], dimer_df['energy'], s=50, alpha=0.8)
    axes[1].set_title('Charged dimers')
    axes[1].set_xlabel('Dimer COM distance')
    axes[1].set_ylabel('Energy')
    sns.boxplot(data=ag_df, x='total_charge', y='energy', ax=axes[2], hue='total_charge', dodge=False, legend=False)
    axes[2].set_title('Ag3 by charge state')
    axes[2].set_xlabel('Total charge')
    axes[2].set_ylabel('Energy')
    fig.tight_layout()
    fig.savefig(IMG . 'dataset_overview.png', dpi=200)
    plt.close(fig)

    sample = random_pred_df.sample(min(5000, len(random_pred_df)), random_state=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(sample, x='pred_charge', hue='true_charge', multiple='stack', bins=3, ax=axes[0])
    axes[0].set_title('Latent charge recovery')
    axes[0].set_xlabel('Predicted sign')
    cm = pd.crosstab(random_pred_df['true_charge'], random_pred_df['pred_charge'], normalize='index')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', ax=axes[1])
    axes[1].set_title('Charge-sign confusion matrix')
    fig.tight_layout()
    fig.savefig(IMG . 'random_charge_recovery.png', dpi=200)
    plt.close(fig)

    order = np.argsort(dimer_df['com_distance'].to_numpy())
    dsort = dimer_df.iloc[order]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(dsort['com_distance'], dsort['energy'], 'o-', label='Reference', lw=2)
    ax.plot(dsort['com_distance'], dsort['pred_short'], 's--', label='Short-range only', lw=2)
    ax.plot(dsort['com_distance'], dsort['pred_long'], 'd--', label='Coulomb-aware', lw=2)
    ax.set_xlabel('Dimer COM distance')
    ax.set_ylabel('Energy')
    ax.set_title('Binding curve reconstruction')
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMG . 'charged_dimer_binding_curve.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    err_df = pd.DataFrame({'short_range': np.abs(dimer_df['energy'] - dimer_df['pred_short']), 'coulomb_aware': np.abs(dimer_df['energy'] - dimer_df['pred_long'])})
    sns.boxplot(data=err_df, ax=ax)
    ax.set_ylabel('Absolute energy error')
    ax.set_title('Charged dimer error distribution')
    fig.tight_layout()
    fig.savefig(IMG . 'charged_dimer_error_comparison.png', dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, col, title in zip(axes, ['pred_geometry_only', 'pred_geometry_plus_charge'], ['Geometry only', 'Geometry + global charge']):
        sns.scatterplot(data=ag_df, x='energy', y=col, hue='total_charge', style='total_charge', ax=ax, s=80)
        minv = min(ag_df['energy'].min(), ag_df[col].min())
        maxv = max(ag_df['energy'].max(), ag_df[col].max())
        ax.plot([minv, maxv], [minv, maxv], 'k--', lw=1)
        ax.set_title(title)
        ax.set_xlabel('Reference energy')
        ax.set_ylabel('Predicted energy')
    fig.tight_layout()
    fig.savefig(IMG . 'ag3_charge_conditioning.png', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    random_frames = load_frames_with_metadata(DATA . 'random_charges.xyz')
    dimer_frames = load_frames_with_metadata(DATA . 'charged_dimer.xyz')
    ag_frames = load_frames_with_metadata(DATA . 'ag3_chargestates.xyz')

    for frames, path in [(random_frames, DATA . 'random_charges.xyz'), (dimer_frames, DATA . 'charged_dimer.xyz'), (ag_frames, DATA . 'ag3_chargestates.xyz')]:
        ase_frames = read(path, index=':')
        for frame, at in zip(frames, ase_frames):
            if frame['forces'] is None and getattr(at.calc, 'results', None):
                if at.calc.results.get('forces') is not None:
                    frame['forces'] = np.array(at.calc.results.get('forces'))
            if frame['energy'] is None and getattr(at.calc, 'results', None):
                if at.calc.results.get('energy') is not None:
                    frame['energy'] = float(at.calc.results.get('energy'))

    random_df, random_pred_df, random_summary = experiment_random_charges(random_frames)
    dimer_df, dimer_folds, dimer_summary = experiment_charged_dimer(dimer_frames)
    ag_df, ag_folds, ag_summary = experiment_ag3(ag_frames)

    random_df.to_csv(OUT . 'random_charge_metrics.csv', index=False)
    random_pred_df.to_csv(OUT . 'random_charge_predictions.csv', index=False)
    dimer_df.to_csv(OUT . 'charged_dimer_predictions.csv', index=False)
    dimer_folds.to_csv(OUT . 'charged_dimer_cv_metrics.csv', index=False)
    ag_df.to_csv(OUT . 'ag3_predictions.csv', index=False)
    ag_folds.to_csv(OUT . 'ag3_cv_metrics.csv', index=False)

    summary = {'seed': SEED, 'random_charges': random_summary, 'charged_dimer': dimer_summary, 'ag3_charge_states': ag_summary}
    with open(OUT . 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    make_figures(random_df, random_pred_df, dimer_df, ag_df)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
