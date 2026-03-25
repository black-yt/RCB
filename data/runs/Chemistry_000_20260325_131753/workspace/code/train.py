"""
Training & Evaluation Pipeline for KA-GNN vs Baseline GNN
"""

import os, sys, json, time, warnings, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from kagnn import KAGNN, BaselineGNN
from featurise import load_dataset, DATASET_SPECS

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Config — tuned for CPU
# ─────────────────────────────────────────────────────────────────────────────
SEED       = 42
BATCH_SIZE = 128
HIDDEN_DIM = 64
N_LAYERS   = 2
N_HARMONICS = 3
DROPOUT    = 0.1
N_EPOCHS   = 40
LR         = 1e-3
WD         = 1e-5
WORKSPACE  = '/mnt/d/xwh/ailab记录/工作/26年03月/SGI-Bench/ResearchClawBench/workspaces/Chemistry_000_20260325_131753'
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'

# Only run these datasets (skip huge ones)
RUN_DATASETS = ['bace', 'bbbp', 'clintox', 'hiv', 'muv']


def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def masked_bce(preds, targets):
    valid = ~torch.isnan(targets)
    if valid.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=preds.device)
    return F.binary_cross_entropy_with_logits(preds[valid], targets[valid])


def compute_aucs(preds_list, labels_list, n_tasks):
    preds  = np.concatenate(preds_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    aucs   = []
    for t in range(n_tasks):
        mask = ~np.isnan(labels[:, t])
        y_t, p_t = labels[mask, t], preds[mask, t]
        if len(np.unique(y_t)) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_t, p_t))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.5


def train_epoch(model, loader, opt, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        opt.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = masked_bce(out, batch.y.view(out.shape))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def eval_epoch(model, loader, device, n_tasks):
    model.eval()
    preds_all, labels_all = [], []
    for batch in loader:
        batch = batch.to(device)
        out   = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        preds_all.append(torch.sigmoid(out).cpu().numpy())
        labels_all.append(batch.y.view(out.shape).cpu().numpy())
    return compute_aucs(preds_all, labels_all, n_tasks)


def run_dataset(ds_name, spec):
    set_seed()
    print(f'\n{"="*60}')
    print(f'  Dataset: {ds_name.upper()}')
    print(f'{"="*60}')

    csv_full = os.path.join(WORKSPACE, spec['path'])
    max_mol  = spec.get('max_mol', None)
    graphs   = load_dataset(csv_full, spec['smiles_col'], spec['label_cols'], max_mol)
    print(f'  Loaded {len(graphs)} valid molecules')

    if len(graphs) < 50:
        print('  Too few molecules — skipping')
        return None

    idx_all   = list(range(len(graphs)))
    idx_train, idx_temp = train_test_split(idx_all, test_size=0.2, random_state=SEED)
    idx_val,   idx_test = train_test_split(idx_temp, test_size=0.5, random_state=SEED)

    train_data = [graphs[i] for i in idx_train]
    val_data   = [graphs[i] for i in idx_val]
    test_data  = [graphs[i] for i in idx_test]

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    node_in = graphs[0].x.shape[1]
    edge_in = graphs[0].edge_attr.shape[1]
    n_tasks = spec['n_tasks']

    print(f'  Split: {len(train_data)} train / {len(val_data)} val / {len(test_data)} test')
    print(f'  Node feat: {node_in}, Edge feat: {edge_in}, Tasks: {n_tasks}')
    print(f'  Device: {DEVICE}')

    results = {}

    for model_name in ['KA-GNN', 'GNN-MLP']:
        set_seed()
        if model_name == 'KA-GNN':
            model = KAGNN(
                node_in_dim=node_in,
                edge_in_dim=edge_in,
                hidden_dim=HIDDEN_DIM,
                n_layers=N_LAYERS,
                n_harmonics=N_HARMONICS,
                n_tasks=n_tasks,
                dropout=DROPOUT,
            ).to(DEVICE)
        else:
            model = BaselineGNN(
                node_in_dim=node_in,
                edge_in_dim=edge_in,
                hidden_dim=HIDDEN_DIM,
                n_layers=N_LAYERS,
                n_tasks=n_tasks,
                dropout=DROPOUT,
            ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'\n  [{model_name}]  Params: {n_params:,}')

        opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

        best_val, best_test, best_ep = 0.0, 0.0, 0
        history = {'train_loss': [], 'val_auc': [], 'test_auc': []}

        t0 = time.time()
        for ep in range(1, N_EPOCHS + 1):
            loss     = train_epoch(model, train_loader, opt, DEVICE)
            val_auc  = eval_epoch(model, val_loader,  DEVICE, n_tasks)
            test_auc = eval_epoch(model, test_loader, DEVICE, n_tasks)
            sched.step()

            history['train_loss'].append(loss)
            history['val_auc'].append(val_auc)
            history['test_auc'].append(test_auc)

            if val_auc > best_val:
                best_val, best_test, best_ep = val_auc, test_auc, ep

            if ep % 10 == 0:
                print(f'    Ep {ep:3d} | loss={loss:.4f} | val={val_auc:.4f} | test={test_auc:.4f} | {time.time()-t0:.0f}s')

        results[model_name] = {
            'best_val_auc':  best_val,
            'best_test_auc': best_test,
            'best_epoch':    best_ep,
            'n_params':      n_params,
            'history':       history,
            'train_time_s':  time.time() - t0,
        }
        print(f'  >> Best test AUC: {best_test:.4f} (ep {best_ep})')

    return results


if __name__ == '__main__':
    print(f'Device: {DEVICE}')
    all_results = {}

    for ds_name in RUN_DATASETS:
        spec = DATASET_SPECS[ds_name]
        res  = run_dataset(ds_name, spec)
        if res:
            all_results[ds_name] = res

    out_path = os.path.join(WORKSPACE, 'outputs', 'training_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def to_serial(obj):
        if isinstance(obj, (np.float32, np.float64, float)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64, int)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serial(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serial(v) for v in obj]
        return obj

    with open(out_path, 'w') as f:
        json.dump(to_serial(all_results), f, indent=2)

    print(f'\n\nResults saved → {out_path}')
    print('\n=== Summary ===')
    for ds, res in all_results.items():
        print(f'  {ds.upper():10s}  KA-GNN: {res["KA-GNN"]["best_test_auc"]:.4f}  |  GNN-MLP: {res["GNN-MLP"]["best_test_auc"]:.4f}')
