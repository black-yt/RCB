import argparse
import json
import math
import os
import random
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATASETS = {
    'bace': {'path': 'data/bace.csv', 'smiles': 'smiles', 'tasks': ['label']},
    'bbbp': {'path': 'data/bbbp.csv', 'smiles': 'smiles', 'tasks': ['label']},
    'clintox': {'path': 'data/clintox.csv', 'smiles': 'smiles', 'tasks': ['FDA_APPROVED', 'CT_TOX']},
    'hiv': {'path': 'data/hiv.csv', 'smiles': 'smiles', 'tasks': ['label']},
    'muv': {'path': 'data/muv.csv', 'smiles': 'smiles', 'tasks': [
        'MUV-466','MUV-548','MUV-600','MUV-644','MUV-652','MUV-689','MUV-692','MUV-712',
        'MUV-713','MUV-733','MUV-737','MUV-810','MUV-832','MUV-846','MUV-852','MUV-858','MUV-859'
    ]},
}

ATOM_SYMBOLS = ['C','N','O','S','F','P','Cl','Br','I','B','Si','Se','Unknown']
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
CHIRAL_TAGS = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_float(x):
    if pd.isna(x) or x == '':
        return np.nan
    return float(x)


def atom_features(atom: Chem.Atom) -> np.ndarray:
    symbol = atom.GetSymbol()
    symbol_vec = [1.0 if symbol == s else 0.0 for s in ATOM_SYMBOLS[:-1]]
    if sum(symbol_vec) == 0:
        symbol_vec.append(1.0)
    else:
        symbol_vec.append(0.0)
    degree = atom.GetDegree()
    degree_vec = [1.0 if degree == d else 0.0 for d in range(6)] + [1.0 if degree >= 6 else 0.0]
    formal_charge = atom.GetFormalCharge()
    charge_vec = [formal_charge / 3.0]
    num_h = atom.GetTotalNumHs(includeNeighbors=True)
    num_h_vec = [1.0 if num_h == h else 0.0 for h in range(5)] + [1.0 if num_h >= 5 else 0.0]
    hybrid = atom.GetHybridization()
    hybrid_vec = [1.0 if hybrid == h else 0.0 for h in HYBRIDIZATIONS] + [1.0 if hybrid not in HYBRIDIZATIONS else 0.0]
    aromatic = [float(atom.GetIsAromatic())]
    in_ring = [float(atom.IsInRing())]
    mass = [atom.GetMass() / 200.0]
    chiral = atom.GetChiralTag()
    chiral_vec = [1.0 if chiral == c else 0.0 for c in CHIRAL_TAGS]
    return np.array(symbol_vec + degree_vec + charge_vec + num_h_vec + hybrid_vec + aromatic + in_ring + mass + chiral_vec, dtype=np.float32)


def bond_features(bond: Chem.Bond) -> np.ndarray:
    bt = bond.GetBondType()
    bond_vec = [1.0 if bt == b else 0.0 for b in BOND_TYPES] + [1.0 if bt not in BOND_TYPES else 0.0]
    stereo = str(bond.GetStereo())
    stereo_vals = ['STEREONONE', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS', 'STEREOANY']
    stereo_vec = [1.0 if stereo == s else 0.0 for s in stereo_vals]
    return np.array(bond_vec + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ] + stereo_vec, dtype=np.float32)


def shortest_path_distance(mol: Chem.Mol, i: int, j: int) -> int:
    path = Chem.rdmolops.GetShortestPath(mol, i, j)
    if not path:
        return 99
    return len(path) - 1


def noncovalent_edges(mol: Chem.Mol) -> List[Tuple[int, int, np.ndarray]]:
    n = mol.GetNumAtoms()
    heavy_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    extra = []
    pair_count = 0
    for ix, i in enumerate(heavy_atoms):
        ai = mol.GetAtomWithIdx(i)
        for j in heavy_atoms[ix + 1:]:
            if pair_count >= 64:
                return extra
            if mol.GetBondBetweenAtoms(i, j) is not None:
                continue
            aj = mol.GetAtomWithIdx(j)
            sp = shortest_path_distance(mol, i, j)
            if sp < 2 or sp > 3:
                continue
            pair_count += 1
            is_hbond_like = float(ai.GetSymbol() in {'N', 'O', 'S'} and aj.GetSymbol() in {'N', 'O', 'S'})
            is_hydrophobic = float(ai.GetSymbol() in {'C', 'Cl', 'Br', 'I'} and aj.GetSymbol() in {'C', 'Cl', 'Br', 'I'})
            same_ring = float(ai.IsInRing() and aj.IsInRing())
            feat = np.zeros(13, dtype=np.float32)
            feat[0] = sp / 3.0
            feat[-3] = same_ring
            feat[-2] = is_hbond_like
            feat[-1] = is_hydrophobic
            extra.append((i, j, feat))
    return extra


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    x = np.stack([atom_features(atom) for atom in mol.GetAtoms()])
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    for i, j, feat in noncovalent_edges(mol):
        edge_index.extend([[i, j], [j, i]])
        edge_attr.extend([feat, feat])
    if not edge_index:
        edge_index = [[0, 0]]
        edge_attr = [np.zeros(13, dtype=np.float32)]
    return {
        'x': torch.tensor(x, dtype=torch.float32),
        'edge_index': torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        'edge_attr': torch.tensor(np.stack(edge_attr), dtype=torch.float32),
        'num_nodes': x.shape[0],
        'mol_weight': Descriptors.MolWt(Chem.RemoveHs(mol)),
        'num_atoms': Chem.RemoveHs(mol).GetNumAtoms(),
    }


def graph_to_serializable(graph: Dict):
    out = {}
    for k, v in graph.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def serializable_to_graph(graph: Dict):
    out = dict(graph)
    out['x'] = torch.tensor(graph['x'], dtype=torch.float32)
    out['edge_index'] = torch.tensor(graph['edge_index'], dtype=torch.long)
    out['edge_attr'] = torch.tensor(graph['edge_attr'], dtype=torch.float32)
    out['y'] = torch.tensor(graph['y'], dtype=torch.float32)
    out['mask'] = torch.tensor(graph['mask'], dtype=torch.float32)
    return out


def load_dataset(name: str, limit: int = None):
    meta = DATASETS[name]
    tasks = meta['tasks']
    cache_path = f'outputs/cache_{name}.json'
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            cached = json.load(f)
        if limit is not None:
            cached = cached[:limit]
        records = [serializable_to_graph(g) for g in cached]
        return records, tasks
    df = pd.read_csv(meta['path'])
    if limit is not None:
        df = df.head(limit)
    smiles_col = meta['smiles']
    records = []
    for _, row in df.iterrows():
        smiles = row[smiles_col]
        graph = smiles_to_graph(smiles)
        if graph is None:
            continue
        y = np.array([safe_float(row[t]) for t in tasks], dtype=np.float32)
        mask = ~np.isnan(y)
        y[np.isnan(y)] = 0.0
        graph['y'] = torch.tensor(y, dtype=torch.float32)
        graph['mask'] = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
        graph['smiles'] = smiles
        records.append(graph)
    with open(cache_path, 'w') as f:
        json.dump([graph_to_serializable(g) for g in records], f)
    return records, tasks


def collate_graphs(batch: List[Dict]):
    xs, edge_indices, edge_attrs = [], [], []
    batch_idx = []
    ys, masks = [], []
    offset = 0
    for gi, g in enumerate(batch):
        n = g['x'].shape[0]
        xs.append(g['x'])
        edge_indices.append(g['edge_index'] + offset)
        edge_attrs.append(g['edge_attr'])
        batch_idx.append(torch.full((n,), gi, dtype=torch.long))
        ys.append(g['y'])
        masks.append(g['mask'])
        offset += n
    return {
        'x': torch.cat(xs, dim=0),
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_attr': torch.cat(edge_attrs, dim=0),
        'batch': torch.cat(batch_idx, dim=0),
        'y': torch.stack(ys, dim=0),
        'mask': torch.stack(masks, dim=0),
    }


class FourierKANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_freq: int = 4):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.freq = nn.Parameter(torch.randn(in_dim, num_freq) * 0.2)
        self.phase = nn.Parameter(torch.zeros(in_dim, num_freq))
        self.coeff_sin = nn.Parameter(torch.randn(in_dim, num_freq, out_dim) * 0.05)
        self.coeff_cos = nn.Parameter(torch.randn(in_dim, num_freq, out_dim) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        base = self.linear(x)
        angle = x.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.unsqueeze(0)
        sin_part = torch.einsum('bif,ifo->bo', torch.sin(angle), self.coeff_sin)
        cos_part = torch.einsum('bif,ifo->bo', torch.cos(angle), self.coeff_cos)
        return base + sin_part + cos_part + self.bias


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeMessageLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float, use_ka: bool = False):
        super().__init__()
        self.use_ka = use_ka
        block_in = hidden_dim * 2 + edge_dim
        if use_ka:
            self.msg = nn.Sequential(
                FourierKANLayer(block_in, hidden_dim),
                nn.SiLU(),
                FourierKANLayer(hidden_dim, hidden_dim),
            )
            self.upd = nn.Sequential(
                FourierKANLayer(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                FourierKANLayer(hidden_dim, hidden_dim),
            )
        else:
            self.msg = MLP(block_in, hidden_dim, hidden_dim, dropout)
            self.upd = MLP(hidden_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg_in = torch.cat([x[src], x[dst], edge_attr], dim=-1)
        messages = self.msg(msg_in)
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, messages)
        out = self.upd(torch.cat([x, agg], dim=-1))
        out = self.norm(x + self.dropout(F.silu(out)))
        return out


class GraphPredictor(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, num_tasks, num_layers=3, dropout=0.1, use_ka=False):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EdgeMessageLayer(hidden_dim, hidden_dim, dropout, use_ka=use_ka) for _ in range(num_layers)
        ])
        if use_ka:
            self.head = nn.Sequential(
                FourierKANLayer(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, num_tasks),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_tasks),
            )

    def forward(self, batch):
        x = self.node_proj(batch['x'])
        e = self.edge_proj(batch['edge_attr'])
        for layer in self.layers:
            x = layer(x, batch['edge_index'], e)
        graph_sum = scatter_sum(x, batch['batch'])
        graph_mean = scatter_mean(x, batch['batch'])
        graph_repr = torch.cat([graph_sum, graph_mean], dim=-1)
        logits = self.head(graph_repr)
        return logits, graph_repr


def scatter_sum(x, batch_idx):
    out = torch.zeros(batch_idx.max().item() + 1, x.size(-1), device=x.device)
    out.index_add_(0, batch_idx, x)
    return out


def scatter_mean(x, batch_idx):
    out = scatter_sum(x, batch_idx)
    counts = torch.bincount(batch_idx, minlength=out.size(0)).clamp_min(1).unsqueeze(-1).to(x.device)
    return out / counts


def make_splits(dataset: List[Dict], tasks: List[str], seed: int):
    labels = np.array([g['y'][0].item() for g in dataset])
    indices = np.arange(len(dataset))
    stratify = labels if len(np.unique(labels)) > 1 and tasks == ['label'] else None
    tr_idx, te_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=stratify)
    tr_labels = labels[tr_idx]
    stratify_val = tr_labels if stratify is not None else None
    tr_idx, va_idx = train_test_split(tr_idx, test_size=0.2, random_state=seed + 1, stratify=stratify_val)
    return tr_idx.tolist(), va_idx.tolist(), te_idx.tolist()


def batch_iterator(dataset, indices, batch_size, shuffle=False):
    idx = indices.copy()
    if shuffle:
        random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch = [dataset[i] for i in idx[start:start + batch_size]]
        yield collate_graphs(batch)


def compute_metrics(y_true, y_score, mask):
    metrics = {}
    roc_scores, pr_scores = [], []
    for t in range(y_true.shape[1]):
        m = mask[:, t] > 0.5
        yt = y_true[m, t]
        ys = y_score[m, t]
        if len(np.unique(yt)) < 2:
            continue
        roc_scores.append(roc_auc_score(yt, ys))
        pr_scores.append(average_precision_score(yt, ys))
    metrics['roc_auc'] = float(np.mean(roc_scores)) if roc_scores else float('nan')
    metrics['pr_auc'] = float(np.mean(pr_scores)) if pr_scores else float('nan')
    return metrics


def evaluate(model, dataset, indices, batch_size, device):
    model.eval()
    ys, scores, masks, reps = [], [], [], []
    with torch.no_grad():
        for batch in batch_iterator(dataset, indices, batch_size, shuffle=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits, graph_repr = model(batch)
            ys.append(batch['y'].cpu().numpy())
            masks.append(batch['mask'].cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            reps.append(graph_repr.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(scores, axis=0)
    mask = np.concatenate(masks, axis=0)
    metrics = compute_metrics(y_true, y_score, mask)
    return metrics, y_true, y_score, mask, np.concatenate(reps, axis=0)


def train_one(dataset_name, dataset, tasks, model_name, seed, args):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tr_idx, va_idx, te_idx = make_splits(dataset, tasks, seed)
    node_dim = dataset[0]['x'].shape[1]
    edge_dim = dataset[0]['edge_attr'].shape[1]
    model = GraphPredictor(node_dim, edge_dim, args.hidden_dim, len(tasks), num_layers=args.num_layers, dropout=args.dropout, use_ka=(model_name == 'ka_gnn')).to(device)
    pos_weight = []
    y_train = torch.stack([dataset[i]['y'] for i in tr_idx])
    m_train = torch.stack([dataset[i]['mask'] for i in tr_idx])
    for t in range(len(tasks)):
        valid = m_train[:, t] > 0.5
        yt = y_train[valid, t]
        pos = float((yt == 1).sum().item())
        neg = float((yt == 0).sum().item())
        pos_weight.append(neg / max(pos, 1.0))
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = None
    history = []
    patience_left = args.patience
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in batch_iterator(dataset, tr_idx, args.batch_size, shuffle=True):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss_raw = criterion(logits, batch['y']) * batch['mask']
            loss = loss_raw.sum() / batch['mask'].sum().clamp_min(1.0)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.item())
        val_metrics, _, _, _, _ = evaluate(model, dataset, va_idx, args.batch_size, device)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), **val_metrics})
        monitor = val_metrics['roc_auc'] if not math.isnan(val_metrics['roc_auc']) else -1e9
        if best is None or monitor > best['val_roc_auc']:
            best = {
                'epoch': epoch,
                'state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'val_roc_auc': monitor,
                'val_pr_auc': val_metrics['pr_auc'],
            }
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    model.load_state_dict(best['state_dict'])
    train_metrics, _, _, _, _ = evaluate(model, dataset, tr_idx, args.batch_size, device)
    val_metrics, _, _, _, _ = evaluate(model, dataset, va_idx, args.batch_size, device)
    test_metrics, y_true, y_score, mask, reps = evaluate(model, dataset, te_idx, args.batch_size, device)
    runtime = time.time() - start_time
    rep_norm = float(np.linalg.norm(reps, axis=1).mean())
    result = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': seed,
        'train_roc_auc': train_metrics['roc_auc'],
        'train_pr_auc': train_metrics['pr_auc'],
        'val_roc_auc': val_metrics['roc_auc'],
        'val_pr_auc': val_metrics['pr_auc'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_pr_auc': test_metrics['pr_auc'],
        'best_epoch': best['epoch'],
        'runtime_sec': runtime,
        'mean_rep_norm': rep_norm,
        'num_train': len(tr_idx),
        'num_val': len(va_idx),
        'num_test': len(te_idx),
    }
    return result, history, {'y_true': y_true.tolist(), 'y_score': y_score.tolist(), 'mask': mask.tolist()}


def run_eda(selected):
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)
    rows = []
    chem_rows = []
    for name in selected:
        dataset, tasks = load_dataset(name)
        for t_idx, task in enumerate(tasks):
            ys = np.array([g['y'][t_idx].item() for g in dataset])
            ms = np.array([g['mask'][t_idx].item() for g in dataset])
            valid = ms > 0.5
            pos_rate = float(ys[valid].mean()) if valid.sum() else np.nan
            rows.append({
                'dataset': name,
                'task': task,
                'num_molecules': len(dataset),
                'num_labeled': int(valid.sum()),
                'positive_rate': pos_rate,
                'avg_num_atoms': float(np.mean([g['num_atoms'] for g in dataset])),
                'avg_mol_weight': float(np.mean([g['mol_weight'] for g in dataset])),
            })
        for g in dataset[: min(2000, len(dataset))]:
            chem_rows.append({'dataset': name, 'num_atoms': g['num_atoms'], 'mol_weight': g['mol_weight']})
    summary = pd.DataFrame(rows)
    summary.to_csv('outputs/dataset_summary.csv', index=False)
    chem_df = pd.DataFrame(chem_rows)
    chem_df.to_csv('outputs/eda_molecule_stats.csv', index=False)

    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.barplot(data=summary, x='dataset', y='num_molecules', hue='task', ax=axes[0])
    axes[0].set_title('Dataset size')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=summary, x='dataset', y='positive_rate', hue='task', ax=axes[1])
    axes[1].set_title('Positive label rate')
    axes[1].tick_params(axis='x', rotation=45)
    sns.boxplot(data=chem_df, x='dataset', y='num_atoms', ax=axes[2])
    axes[2].set_title('Atom count distribution')
    axes[2].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('report/images/data_overview.png', dpi=200)
    plt.close(fig)


def aggregate_results(results_df: pd.DataFrame):
    agg = results_df.groupby(['dataset', 'model']).agg(
        test_roc_auc_mean=('test_roc_auc', 'mean'),
        test_roc_auc_std=('test_roc_auc', 'std'),
        test_pr_auc_mean=('test_pr_auc', 'mean'),
        test_pr_auc_std=('test_pr_auc', 'std'),
        runtime_sec_mean=('runtime_sec', 'mean'),
        runtime_sec_std=('runtime_sec', 'std'),
        mean_rep_norm=('mean_rep_norm', 'mean'),
        n_runs=('seed', 'count'),
    ).reset_index()
    pivot = agg.pivot(index='dataset', columns='model', values='test_roc_auc_mean')
    if 'ka_gnn' in pivot.columns and 'baseline_gnn' in pivot.columns:
        diff = (pivot['ka_gnn'] - pivot['baseline_gnn']).rename('roc_auc_gain')
        agg = agg.merge(diff, on='dataset', how='left')
    return agg


def run_analysis(selected):
    baseline = pd.read_csv('outputs/results_baseline.csv')
    ka = pd.read_csv('outputs/results_ka.csv')
    results = pd.concat([baseline, ka], ignore_index=True)
    results.to_csv('outputs/all_results.csv', index=False)
    agg = aggregate_results(results)
    agg.to_csv('outputs/final_metrics.csv', index=False)

    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=agg, x='dataset', y='test_roc_auc_mean', hue='model', ax=axes[0])
    axes[0].set_title('Test ROC-AUC comparison')
    axes[0].tick_params(axis='x', rotation=45)
    sns.barplot(data=agg, x='dataset', y='test_pr_auc_mean', hue='model', ax=axes[1])
    axes[1].set_title('Test PR-AUC comparison')
    axes[1].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('report/images/model_comparison.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=agg, x='dataset', y='runtime_sec_mean', hue='model', ax=ax)
    ax.set_title('Runtime comparison')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig('report/images/runtime_comparison.png', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    paired = results.pivot_table(index=['dataset', 'seed'], columns='model', values='test_roc_auc').reset_index()
    if 'baseline_gnn' in paired.columns and 'ka_gnn' in paired.columns:
        for _, row in paired.iterrows():
            ax.plot(['baseline_gnn', 'ka_gnn'], [row['baseline_gnn'], row['ka_gnn']], marker='o', alpha=0.7)
    ax.set_ylabel('Test ROC-AUC')
    ax.set_title('Per-seed validation comparison')
    plt.tight_layout()
    plt.savefig('report/images/validation_plot.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eda', 'train', 'analyze'], required=True)
    parser.add_argument('--model', choices=['baseline_gnn', 'ka_gnn'], default='baseline_gnn')
    parser.add_argument('--datasets', nargs='+', default=['bace', 'bbbp', 'clintox', 'hiv'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('outputs/run_logs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)

    if args.mode == 'eda':
        run_eda(args.datasets)
        return

    if args.mode == 'train':
        all_results = []
        for dataset_name in args.datasets:
            dataset, tasks = load_dataset(dataset_name, limit=args.limit)
            for seed in args.seeds:
                result, history, preds = train_one(dataset_name, dataset, tasks, args.model, seed, args)
                all_results.append(result)
                stem = f"{dataset_name}_{args.model}_seed{seed}"
                with open(f'outputs/run_logs/{stem}_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
                with open(f'outputs/run_logs/{stem}_preds.json', 'w') as f:
                    json.dump(preds, f)
                with open(f'outputs/run_logs/{stem}_summary.json', 'w') as f:
                    json.dump(result, f, indent=2)
                print(json.dumps(result))
        df = pd.DataFrame(all_results)
        out_path = 'outputs/results_baseline.csv' if args.model == 'baseline_gnn' else 'outputs/results_ka.csv'
        df.to_csv(out_path, index=False)
        return

    if args.mode == 'analyze':
        run_analysis(args.datasets)
        return


if __name__ == '__main__':
    main()
