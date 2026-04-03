#!/usr/bin/env python
"""End-to-end experiments for DIDS-MFL on NF-UNSW-NB15-v2 (3D TemporalData).

This script:
1. Loads the TemporalData graph.
2. Performs EDA and saves figures.
3. Constructs train/val/test splits for known/unknown/few-shot attacks.
4. Trains several models:
   - Logistic Regression on aggregated features
   - MLP baseline
   - Graph Neural Network with diffusion-style message passing
5. Evaluates binary and multi-class performance.
6. Saves metrics and representations to outputs/.

Reproducible with fixed random seeds.
"""
import os
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.serialization import add_safe_globals
from torch_geometric.data.temporal import TemporalData
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUTPUT_DIR = BASE_DIR / 'outputs'
REPORT_IMG_DIR = BASE_DIR / 'report' / 'images'


def load_temporal_graph() -> TemporalData:
    add_safe_globals([TemporalData])
    data = torch.load(DATA_PATH, map_location='cpu', weights_only=False)
    assert isinstance(data, TemporalData)
    return data


def derive_node_features_from_msgs(msg: torch.Tensor, src: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Aggregate flow-level msg features to node-level by mean pooling per src node."""
    feat_dim = msg.size(1)
    node_feat = torch.zeros((num_nodes, feat_dim), dtype=msg.dtype)
    counts = torch.zeros((num_nodes, 1), dtype=msg.dtype)
    node_feat.index_add_(0, src, msg)
    ones = torch.ones((msg.size(0), 1), dtype=msg.dtype)
    counts.index_add_(0, src, ones)
    counts[counts == 0] = 1.0
    node_feat = node_feat / counts
    return node_feat


def build_static_graph(data: TemporalData):
    # Use src/dst as edges, make undirected static graph
    edge_index = torch.stack([data.src, data.dst], dim=0)
    edge_index = to_undirected(edge_index)
    num_nodes = int(torch.max(torch.cat([data.src, data.dst])).item()) + 1
    x = derive_node_features_from_msgs(data.msg, data.src, num_nodes)

    # derive node labels based on majority attack label of outgoing flows
    attack = data.attack.long()
    node_labels = torch.full((num_nodes,), -1, dtype=torch.long)
    for node in range(num_nodes):
        mask = data.src == node
        if mask.any():
            labels, counts = torch.unique(attack[mask], return_counts=True)
            node_labels[node] = labels[torch.argmax(counts)]
    # filter unlabeled nodes
    labeled_mask = node_labels >= 0
    labeled_idx = torch.nonzero(labeled_mask, as_tuple=False).view(-1)
    x = x[labeled_idx]
    node_labels = node_labels[labeled_idx]

    # also remap edge_index to compact node indices
    mapping = {int(old): int(i) for i, old in enumerate(labeled_idx.tolist())}
    edge_index_mapped = edge_index.clone()
    for i in range(edge_index_mapped.size(1)):
        edge_index_mapped[0, i] = mapping.get(int(edge_index_mapped[0, i]), -1)
        edge_index_mapped[1, i] = mapping.get(int(edge_index_mapped[1, i]), -1)
    valid_mask = (edge_index_mapped[0] >= 0) & (edge_index_mapped[1] >= 0)
    edge_index_mapped = edge_index_mapped[:, valid_mask]
    return x, node_labels, edge_index_mapped, labeled_idx


def create_known_unknown_splits(labels: torch.Tensor, unknown_ratio: float = 0.2, few_shot_per_class: int = 10,
                                seed: int = 42):
    """Split attack classes into known/unknown; create few-shot within known."""
    set_seed(seed)
    classes = torch.unique(labels).tolist()
    benign_class = int(min(classes))
    attack_classes = [c for c in classes if c != benign_class]
    rng = np.random.default_rng(seed)
    rng.shuffle(attack_classes)
    num_unknown = max(1, int(len(attack_classes) * unknown_ratio))
    unknown_attack = set(attack_classes[:num_unknown])
    known_attack = set(attack_classes[num_unknown:])

    known_mask = torch.isin(labels, torch.tensor(list(known_attack | {benign_class})))
    unknown_mask = torch.isin(labels, torch.tensor(list(unknown_attack)))

    # few-shot: limit samples per known attack class in train
    idx_all = torch.arange(labels.size(0))
    train_idx, val_idx, test_idx = [], [], []

    for c in classes:
        c_mask = labels == c
        c_idx = idx_all[c_mask]
        c_idx = c_idx[torch.randperm(len(c_idx))]
        n = len(c_idx)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        if c in unknown_attack:
            # unknown: use only val/test (no train)
            train_c = []
            val_c = c_idx[:n_val]
            test_c = c_idx[n_val:]
        else:
            train_c = c_idx[:n_train]
            # enforce few-shot for attacks (excluding benign)
            if c != benign_class:
                train_c = train_c[:few_shot_per_class]
            val_c = c_idx[n_train:n_train + n_val]
            test_c = c_idx[n_train + n_val:]
        train_idx.append(train_c)
        val_idx.append(val_c)
        test_idx.append(test_c)

    train_idx = torch.cat([i for i in train_idx if len(i) > 0])
    val_idx = torch.cat([i for i in val_idx if len(i) > 0])
    test_idx = torch.cat([i for i in test_idx if len(i) > 0])

    split = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx,
        'benign_class': benign_class,
        'known_attack_classes': sorted(list(known_attack)),
        'unknown_attack_classes': sorted(list(unknown_attack)),
    }
    return split


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class DiffusionGCN(nn.Module):
    """Simple GCN with residual connections as a proxy for diffusion."""

    def __init__(self, in_dim, hidden_dim, num_classes, num_layers: int = 2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h_res = h
            h = conv(h, edge_index)
            h = torch.relu(h)
            if h.shape == h_res.shape:
                h = h + h_res
        out = self.lin(h)
        return out


def train_torch_model(model, x, y, idx_train, idx_val, edge_index=None, epochs: int = 100, lr: float = 1e-3,
                      weight_decay: float = 1e-4, device: str = 'cpu'):
    model.to(device)
    x = x.to(device)
    y = y.to(device)
    if edge_index is not None:
        edge_index = edge_index.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        if edge_index is None:
            logits = model(x)
        else:
            logits = model(x, edge_index)
        loss = nn.functional.cross_entropy(logits[idx_train], y[idx_train])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            if edge_index is None:
                logits = model(x)
            else:
                logits = model(x, edge_index)
            preds = logits.argmax(dim=-1)
            val_acc = (preds[idx_val] == y[idx_val]).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(name: str, logits: torch.Tensor, y: torch.Tensor, idx_split: Dict[str, torch.Tensor],
                   benign_class: int, output_prefix: str):
    results = {}
    for split_name, idx in idx_split.items():
        idx = idx.cpu()
        y_true = y[idx].cpu().numpy()
        y_pred = logits[idx].argmax(dim=-1).cpu().numpy()

        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # binary mapping
        y_true_bin = (y_true != benign_class).astype(int)
        y_pred_bin = (y_pred != benign_class).astype(int)
        report_bin = classification_report(y_true_bin, y_pred_bin, output_dict=True, zero_division=0)
        cm_bin = confusion_matrix(y_true_bin, y_pred_bin)

        results[split_name] = {
            'multi_class_report': report,
            'multi_class_confusion_matrix': cm.tolist(),
            'binary_report': report_bin,
            'binary_confusion_matrix': cm_bin.tolist(),
        }

        # save confusion matrices as figures
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(cm, annot=False, ax=axes[0], cmap='Blues')
        axes[0].set_title(f'{name} {split_name} Multi-class CM')
        sns.heatmap(cm_bin, annot=True, fmt='d', ax=axes[1], cmap='Reds')
        axes[1].set_title(f'{name} {split_name} Binary CM')
        fig.tight_layout()
        fig_path = REPORT_IMG_DIR / f'{output_prefix}_{split_name}_cm.png'
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)

    out_path = OUTPUT_DIR / f'{output_prefix}_metrics.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    return results


def plot_data_overview(x: torch.Tensor, labels: torch.Tensor, t: torch.Tensor, attack: torch.Tensor):
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    # class distribution
    unique, counts = torch.unique(attack, return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(unique.cpu().numpy(), counts.cpu().numpy())
    ax.set_xlabel('Attack Class ID')
    ax.set_ylabel('Flow Count')
    ax.set_title('Flow-level Attack Class Distribution')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'data_attack_distribution.png', dpi=200)
    plt.close(fig)

    # temporal distribution of flows (simple histogram)
    t_np = t.cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(t_np, bins=50)
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Count of Flows')
    ax.set_title('Temporal Distribution of Flows')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'data_temporal_distribution.png', dpi=200)
    plt.close(fig)

    # node feature t-SNE colored by majority label
    # subsample for speed
    n_samples = min(2000, x.size(0))
    idx = torch.randperm(x.size(0))[:n_samples]
    x_sub = x[idx].cpu().numpy()
    y_sub = labels[idx].cpu().numpy()
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    emb_2d = tsne.fit_transform(x_sub)

    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_sub, s=5, cmap='tab20')
    legend1 = ax.legend(*scatter.legend_elements(num=len(np.unique(y_sub))), title="Classes", loc='best', fontsize=6)
    ax.add_artist(legend1)
    ax.set_title('t-SNE of Node Features (majority attack label)')
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / 'data_tsne_node_features.png', dpi=200)
    plt.close(fig)


def run():
    set_seed(42)
    data = load_temporal_graph()
    x, node_labels, edge_index, labeled_idx = build_static_graph(data)

    # overview plots (using flow-level attack + time, but node-level features)
    plot_data_overview(x, node_labels, data.t, data.attack)

    # train/val/test splits with known/unknown and few-shot
    split = create_known_unknown_splits(node_labels, unknown_ratio=0.3, few_shot_per_class=20, seed=42)
    idx_train = split['train']
    idx_val = split['val']
    idx_test = split['test']

    benign_class = split['benign_class']
    known_attacks = split['known_attack_classes']
    unknown_attacks = split['unknown_attack_classes']

    meta = {
        'num_nodes': int(x.size(0)),
        'feat_dim': int(x.size(1)),
        'num_classes': int(torch.unique(node_labels).numel()),
        'benign_class': benign_class,
        'known_attacks': known_attacks,
        'unknown_attacks': unknown_attacks,
        'num_train': int(idx_train.numel()),
        'num_val': int(idx_val.numel()),
        'num_test': int(idx_test.numel()),
    }
    with open(OUTPUT_DIR / 'data_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Standardize features for non-graph models
    scaler = StandardScaler()
    x_np = x.cpu().numpy()
    x_scaled = scaler.fit_transform(x_np)
    x_scaled = torch.from_numpy(x_scaled).float()

    # Logistic Regression baseline (multi-class)
    clf = LogisticRegression(max_iter=1000, n_jobs=4)
    clf.fit(x_scaled[idx_train].numpy(), node_labels[idx_train].numpy())
    logits_lr = torch.from_numpy(clf.predict_proba(x_scaled.numpy())).float()
    evaluate_model('LogReg', logits_lr, node_labels, {'train': idx_train, 'val': idx_val, 'test': idx_test},
                   benign_class, 'logreg')

    # MLP baseline
    mlp = MLP(in_dim=x.size(1), hidden_dim=128, num_classes=int(torch.unique(node_labels).numel()))
    mlp = train_torch_model(mlp, x_scaled, node_labels, idx_train, idx_val, edge_index=None,
                            epochs=80, lr=1e-3, weight_decay=5e-4)
    mlp.eval()
    with torch.no_grad():
        logits_mlp = mlp(x_scaled)
    evaluate_model('MLP', logits_mlp, node_labels, {'train': idx_train, 'val': idx_val, 'test': idx_test},
                   benign_class, 'mlp')

    # Diffusion GCN (graph-based)
    gcn = DiffusionGCN(in_dim=x.size(1), hidden_dim=128,
                       num_classes=int(torch.unique(node_labels).numel()), num_layers=3)
    gcn = train_torch_model(gcn, x, node_labels, idx_train, idx_val, edge_index=edge_index,
                            epochs=80, lr=1e-3, weight_decay=5e-4)
    gcn.eval()
    with torch.no_grad():
        logits_gcn = gcn(x, edge_index)
    evaluate_model('DiffusionGCN', logits_gcn, node_labels,
                   {'train': idx_train, 'val': idx_val, 'test': idx_test}, benign_class, 'gcn')

    # Save node embeddings from GCN for analysis
    with torch.no_grad():
        h = x
        for conv in gcn.convs:
            h_res = h
            h = conv(h, edge_index)
            h = torch.relu(h)
            if h.shape == h_res.shape:
                h = h + h_res
    torch.save({'embeddings': h.cpu(), 'labels': node_labels.cpu(), 'idx_train': idx_train,
                'idx_val': idx_val, 'idx_test': idx_test}, OUTPUT_DIR / 'gcn_node_embeddings.pt')


if __name__ == '__main__':
    run()
