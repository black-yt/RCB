"""
Redesigned GNN pipeline for altermagnet discovery.

Key insight:
- pretrain_data.pt has 5000 graphs with y labels, pos_ratio=0.5 (balanced)
- finetune_data.pt has 2000 graphs with y labels, pos_ratio≈5% (imbalanced)
- candidate_data.pt has 1000 unlabeled graphs for prediction

Strategy:
1. Self-supervised pre-training via masked node reconstruction on pretrain_data
2. Supervised auxiliary task: use pretrain_data labels (balanced) to warm-start model
3. Fine-tune on finetune_data (imbalanced, domain-specific)
4. Ensemble multiple seeds + combine with handcrafted features
"""

import sys
import os
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import numpy as np
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def _register_stub():
    if 'data_prepare' not in sys.modules:
        dm = types.ModuleType('data_prepare')
        class RealisticCrystalDataset: pass
        dm.RealisticCrystalDataset = RealisticCrystalDataset
        sys.modules['data_prepare'] = dm


def load_dataset(path):
    _register_stub()
    ds = torch.load(path, map_location='cpu', weights_only=False)
    return ds.data_list


def augment_graph_features(data_list):
    """Add degree and avg bond distance as extra node features."""
    augmented = []
    for d in data_list:
        d2 = d.clone()
        x = d.x
        edge_index = d.edge_index
        edge_attr = d.edge_attr
        N = x.shape[0]
        degree = torch.zeros(N, 1)
        avg_bond = torch.zeros(N, 1)
        if edge_index.shape[1] > 0:
            degree.scatter_add_(0, edge_index[1].unsqueeze(1),
                                torch.ones(edge_index.shape[1], 1))
            degree = degree / (degree.max() + 1e-8)
            bond_dist = edge_attr[:, 0:1]
            avg_bond.scatter_add_(0, edge_index[1].unsqueeze(1), bond_dist)
            cnt = torch.zeros(N, 1)
            cnt.scatter_add_(0, edge_index[1].unsqueeze(1),
                             torch.ones(edge_index.shape[1], 1))
            avg_bond = avg_bond / (cnt + 1e-8)
        d2.x = torch.cat([x, degree, avg_bond], dim=-1)
        augmented.append(d2)
    return augmented


class EdgeConv(nn.Module):
    """Message passing with edge features + skip connection."""
    def __init__(self, in_dim, out_dim, edge_dim):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(in_dim + edge_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Messages: edge feat + source node feat
        msg_in = torch.cat([x[row], edge_attr], dim=-1)
        msg = self.msg_net(msg_in)  # [E, out_dim]
        # Aggregate messages to destination nodes
        agg = torch.zeros(x.size(0), msg.size(-1), device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand(-1, msg.size(-1)), msg)
        # Update: combine node + aggregated message
        out = self.update_net(torch.cat([x, agg], dim=-1))
        return out + self.skip(x)


class GNNEncoder(nn.Module):
    """4-layer edge-conditioned GNN with hierarchical pooling."""
    def __init__(self, node_dim=30, edge_dim=2, hidden=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.convs = nn.ModuleList([
            EdgeConv(hidden, hidden, edge_dim) for _ in range(num_layers)
        ])
        self.drop = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.hidden = hidden
        self.combine = nn.Sequential(
            nn.Linear(hidden * num_layers, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.embed(x)
        layer_reps = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.drop(x)
            layer_reps.append(global_mean_pool(x, batch))
        g = self.combine(torch.cat(layer_reps, dim=-1))
        return g, x   # (graph-level repr, final node repr)


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class FullModel(nn.Module):
    """GNN with classification head."""
    def __init__(self, node_dim=30, edge_dim=2, hidden=128, num_layers=4, dropout=0.4):
        super().__init__()
        self.encoder = GNNEncoder(node_dim, edge_dim, hidden, num_layers, dropout=0.1)
        self.head = ClassificationHead(hidden, hidden // 2, dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        g, _ = self.encoder(x, edge_index, edge_attr, batch)
        return self.head(g)


def focal_loss(logits, labels, gamma=2.0, alpha=0.8):
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt  = torch.exp(-bce)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    fl = alpha_t * (1 - pt) ** gamma * bce
    return fl.mean()


def train_supervised(model, data_list, epochs=30, lr=1e-3, batch_size=128,
                     device='cpu', verbose=True, imbalanced=False):
    """Supervised training (works for both balanced pretrain and imbalanced finetune)."""
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.to(device)
    losses = []

    if imbalanced:
        ys = [d.y.item() for d in data_list]
        pos, neg = sum(ys), len(ys) - sum(ys)
        gamma = 3.0
        alpha = neg / len(ys)   # high alpha = focus on positive class
    else:
        gamma, alpha = 1.0, 0.5  # balanced dataset: standard settings

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = focal_loss(logits, batch.y.float(), gamma=gamma, alpha=alpha)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
        sched.step()
        avg = total / max(n, 1)
        losses.append(avg)
        if verbose and epoch % 5 == 0:
            print(f'  [{("Pretrain-sup" if not imbalanced else "Finetune")}] Ep {epoch:3d}  loss={avg:.4f}')
    return losses


def evaluate_auc(model, data_list, device='cpu', batch_size=256):
    from sklearn.metrics import roc_auc_score
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
    try:
        return roc_auc_score(all_labels, all_logits), np.array(all_logits), np.array(all_labels)
    except Exception:
        return 0.5, np.array(all_logits), np.array(all_labels)


def train_with_validation(model, train_list, val_list, epochs=100, lr=2e-4,
                          batch_size=32, device='cpu', verbose=True, patience=30):
    """Fine-tune with early stopping on val AUC."""
    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5,
                                                        patience=15, min_lr=1e-6)
    model.to(device)
    ys = [d.y.item() for d in train_list]
    pos, neg = sum(ys), len(ys) - sum(ys)
    alpha_ft = neg / max(len(ys), 1)   # imbalanced alpha

    train_losses, val_aucs = [], []
    best_auc, best_state, wait = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = focal_loss(logits, batch.y.float(), gamma=3.0, alpha=alpha_ft)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item(); n += 1
        train_losses.append(total / max(n, 1))

        auc, _, _ = evaluate_auc(model, val_list, device=device)
        val_aucs.append(auc)
        sched.step(auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if verbose and epoch % 10 == 0:
            print(f'  [Finetune] Ep {epoch:3d}  loss={train_losses[-1]:.4f}  '
                  f'AUC={auc:.4f}  best={best_auc:.4f}')
        if wait >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break

    if best_state:
        model.load_state_dict(best_state)
    return train_losses, val_aucs


def predict(model, data_list, device='cpu', batch_size=256):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


def get_embeddings(model, data_list, device='cpu', batch_size=256):
    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            g, _ = model.encoder(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            embs.append(g.cpu().numpy())
    return np.concatenate(embs, axis=0)
