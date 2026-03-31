import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.manifold import TSNE
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphNorm, global_mean_pool

from common import ensure_dir, load_dataset, save_json, set_seed

sns.set_theme(style='whitegrid')


class Encoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, emb_dim=64, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = GraphNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, emb_dim)
        self.norm2 = GraphNorm(emb_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.norm1(x, batch)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x, batch)
        x = F.relu(x)
        g = global_mean_pool(x, batch)
        return g


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, emb_dim=64, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden_dim, emb_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, 1),
        )

    def forward(self, data):
        z = self.encoder(data.x.float(), data.edge_index, data.batch)
        return self.head(z).view(-1), z


class ProjectionHead(nn.Module):
    def __init__(self, dim=64, proj_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim))

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


def augment_graph(data, drop_prob=0.1, noise_scale=0.01):
    data = data.clone()
    if data.x is not None:
        mask = torch.rand_like(data.x) > drop_prob
        data.x = data.x * mask.float() + noise_scale * torch.randn_like(data.x)
    if data.edge_index.size(1) > 2:
        keep = torch.rand(data.edge_index.size(1), device=data.edge_index.device) > drop_prob
        if keep.sum() >= 2:
            data.edge_index = data.edge_index[:, keep]
            if getattr(data, 'edge_attr', None) is not None:
                data.edge_attr = data.edge_attr[keep]
    return data


def nt_xent(z1, z2, temperature=0.2):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    batch_size = z1.size(0)
    mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    positives = torch.cat([torch.arange(batch_size, 2 * batch_size), torch.arange(0, batch_size)]).to(z.device)
    return F.cross_entropy(sim, positives)


def evaluate(model, loader, device):
    model.eval()
    ys, probs, logits_list, embeds = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, z = model(batch)
            p = torch.sigmoid(logits)
            ys.append(batch.y.view(-1).cpu().numpy())
            probs.append(p.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            embeds.append(z.cpu().numpy())
    y = np.concatenate(ys)
    prob = np.concatenate(probs)
    logits = np.concatenate(logits_list)
    emb = np.concatenate(embeds)
    metrics = {
        'pr_auc': float(average_precision_score(y, prob)),
        'roc_auc': float(roc_auc_score(y, prob)),
    }
    for k in [20, 50, 100]:
        order = np.argsort(-prob)[:k]
        tp = y[order].sum()
        metrics[f'precision_at_{k}'] = float(tp / k)
        metrics[f'recall_at_{k}'] = float(tp / max(1, y.sum()))
        metrics[f'enrichment_at_{k}'] = float((tp / k) / max(1e-8, y.mean()))
    return metrics, y, prob, logits, emb


def plot_curves(y, prob, outdir, prefix):
    ensure_dir(outdir)
    prec, rec, _ = precision_recall_curve(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec)
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR curve: {prefix}')
    plt.tight_layout(); plt.savefig(Path(outdir) / f'{prefix}_pr_curve.png', dpi=200); plt.close()
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr)
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC curve: {prefix}')
    plt.tight_layout(); plt.savefig(Path(outdir) / f'{prefix}_roc_curve.png', dpi=200); plt.close()
    thresholds = np.linspace(0.05, 0.95, 19)
    rows=[]
    for thr in thresholds:
        pred = (prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2*precision*recall / max(1e-8, precision+recall)
        rows.append({'threshold':thr,'precision':precision,'recall':recall,'f1':f1})
    df = pd.DataFrame(rows)
    df.to_csv(Path(outdir) / f'{prefix}_threshold_sweep.csv', index=False)
    plt.figure(figsize=(5,4))
    plt.plot(df['threshold'], df['precision'], label='precision')
    plt.plot(df['threshold'], df['recall'], label='recall')
    plt.plot(df['threshold'], df['f1'], label='f1')
    plt.legend(); plt.xlabel('Threshold'); plt.ylabel('Metric'); plt.title(f'Threshold sweep: {prefix}')
    plt.tight_layout(); plt.savefig(Path(outdir) / f'{prefix}_threshold_sweep.png', dpi=200); plt.close()


def plot_embeddings(emb, y, outdir, prefix):
    if len(emb) < 5:
        return
    tsne = TSNE(n_components=2, init='random', learning_rate='auto', perplexity=min(30, max(5, len(emb)//10)), random_state=0)
    emb2 = tsne.fit_transform(emb)
    df = pd.DataFrame({'x': emb2[:,0], 'y': emb2[:,1], 'label': y})
    plt.figure(figsize=(5,4))
    sns.scatterplot(data=df, x='x', y='y', hue='label', palette='Set1', s=18, alpha=0.8)
    plt.title(f'Embedding t-SNE: {prefix}')
    plt.tight_layout(); plt.savefig(Path(outdir) / f'{prefix}_embedding_tsne.png', dpi=200); plt.close()


def train_supervised(train_list, val_list, candidate_list, mode, outdir, seed, pretrained_state=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    model = Classifier(train_list[0].x.shape[1]).to(device)
    if pretrained_state is not None:
        model.encoder.load_state_dict(pretrained_state)
    train_loader = DataLoader(train_list, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=256)
    cand_loader = DataLoader(candidate_list, batch_size=256)
    pos = sum(int(d.y.item()) for d in train_list)
    neg = len(train_list) - pos
    if mode == 'weighted' or mode == 'pretrained':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(1, pos)], device=device))
    elif mode == 'focal':
        def criterion(logits, target):
            bce = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
            pt = torch.exp(-bce)
            alpha = 0.75
            gamma = 2.0
            return (alpha * (1-pt)**gamma * bce).mean()
    else:
        criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    best_metric = -1
    best_state = None
    for epoch in range(1, 6):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, _ = model(batch)
            loss = criterion(logits, batch.y.view(-1).float())
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        val_metrics, y_val, p_val, _, emb_val = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), **val_metrics})
        if val_metrics['pr_auc'] > best_metric:
            best_metric = val_metrics['pr_auc']
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_cache = (y_val, p_val, emb_val)
    model.load_state_dict(best_state)
    val_metrics, y_val, p_val, _, emb_val = evaluate(model, val_loader, device)
    cand_metrics, y_cand, p_cand, _, emb_cand = evaluate(model, cand_loader, device)
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(Path(outdir) / f'history_seed{seed}.csv', index=False)
    torch.save(best_state, Path(outdir) / f'model_seed{seed}.pt')
    plot_curves(y_val, p_val, outdir, f'val_seed{seed}_{mode}')
    plot_curves(y_cand, p_cand, outdir, f'candidate_seed{seed}_{mode}')
    if seed == 0:
        plot_embeddings(emb_val, y_val, outdir, f'val_seed{seed}_{mode}')
    pred_df = pd.DataFrame({'index': np.arange(len(p_cand)), 'probability': p_cand, 'true_label': y_cand})
    pred_df.sort_values('probability', ascending=False).to_csv(Path(outdir) / f'candidate_predictions_seed{seed}.csv', index=False)
    result = {'seed': seed, 'mode': mode, 'val_metrics': val_metrics, 'candidate_metrics': cand_metrics}
    save_json(result, Path(outdir) / f'metrics_seed{seed}.json')
    return result, best_state


def pretrain_ssl(pretrain_list, outdir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(0)
    encoder = Encoder(pretrain_list[0].x.shape[1]).to(device)
    proj = ProjectionHead(64, 64).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(pretrain_list, batch_size=256, shuffle=True)
    history = []
    for epoch in range(1, 6):
        encoder.train(); proj.train()
        losses=[]
        for batch in loader:
            batch1 = augment_graph(batch)
            batch2 = augment_graph(batch)
            batch1 = batch1.to(device)
            batch2 = batch2.to(device)
            optimizer.zero_grad()
            z1 = encoder(batch1.x.float(), batch1.edge_index, batch1.batch)
            z2 = encoder(batch2.x.float(), batch2.edge_index, batch2.batch)
            p1 = proj(z1); p2 = proj(z2)
            loss = nt_xent(p1, p2)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({'epoch': epoch, 'ssl_loss': float(np.mean(losses))})
    pd.DataFrame(history).to_csv(Path(outdir) / 'pretrain_history.csv', index=False)
    torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, Path(outdir) / 'encoder.pt')
    plt.figure(figsize=(5,4))
    plt.plot([h['epoch'] for h in history], [h['ssl_loss'] for h in history])
    plt.xlabel('Epoch'); plt.ylabel('SSL loss'); plt.title('Pretraining loss')
    plt.tight_layout(); plt.savefig(Path(outdir) / 'pretrain_loss.png', dpi=200); plt.close()
    return {k: v.cpu() for k, v in encoder.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['baseline', 'weighted', 'focal', 'pretrained'], required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    ensure_dir(args.outdir)
    pretrain_list = load_dataset('data/pretrain_data.pt')
    finetune_list = load_dataset('data/finetune_data.pt')
    candidate_list = load_dataset('data/candidate_data.pt')
    pretrained_state = None
    if args.mode == 'pretrained':
        ssl_dir = Path(args.outdir) / 'ssl'
        ensure_dir(ssl_dir)
        pretrained_state = pretrain_ssl(pretrain_list, ssl_dir)
    split_files = sorted(Path('outputs/splits').glob('split_seed*.json'))[:args.seeds]
    all_results = []
    for split_file in split_files:
        with open(split_file) as f:
            split = json.load(f)
        seed = int(split['seed'])
        train_list = [finetune_list[i] for i in split['train_idx']]
        val_list = [finetune_list[i] for i in split['val_idx']]
        result, _ = train_supervised(train_list, val_list, candidate_list, args.mode, args.outdir, seed, pretrained_state)
        all_results.append(result)
    rows = []
    for r in all_results:
        row = {'seed': r['seed'], 'mode': r['mode']}
        for prefix in ['val_metrics', 'candidate_metrics']:
            for k, v in r[prefix].items():
                row[f'{prefix}_{k}'] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.outdir) / 'summary_metrics.csv', index=False)
    summary = df.drop(columns=['seed', 'mode']).agg(['mean', 'std']).to_dict()
    save_json({'mode': args.mode, 'per_seed': rows, 'aggregate': summary}, Path(args.outdir) / 'summary_metrics.json')
    print(df)


if __name__ == '__main__':
    main()
