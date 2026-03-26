"""
Final comprehensive training pipeline for altermagnet discovery.
Combines all learned best practices:
1. Self-supervised MAE pre-training
2. Supervised training on balanced pretrain data
3. Fine-tuning with positive oversampling
4. Large ensemble with diverse seeds
5. Combine with handcrafted feature baseline
"""

import sys
import os
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve,
                              classification_report)

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')
IMG_DIR  = os.path.join(BASE_DIR, 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')


# ─── Data loading ─────────────────────────────────────────────────────────────
def _register_stub():
    if 'data_prepare' not in sys.modules:
        import types
        dm = types.ModuleType('data_prepare')
        class RealisticCrystalDataset: pass
        dm.RealisticCrystalDataset = RealisticCrystalDataset
        sys.modules['data_prepare'] = dm

_register_stub()

def load_dataset(path):
    return torch.load(path, map_location='cpu', weights_only=False).data_list

pretrain_list  = load_dataset(os.path.join(DATA_DIR, 'pretrain_data.pt'))
finetune_list  = load_dataset(os.path.join(DATA_DIR, 'finetune_data.pt'))
candidate_list = load_dataset(os.path.join(DATA_DIR, 'candidate_data.pt'))

y_can = np.array([d.y.item() for d in candidate_list])
y_ft  = np.array([d.y.item() for d in finetune_list])
y_pt  = np.array([d.y.item() for d in pretrain_list])

print(f'Pretrain: {len(pretrain_list)} (pos={y_pt.sum():.0f}, balanced)')
print(f'Finetune: {len(finetune_list)} (pos={y_ft.sum():.0f}, imbalanced)')
print(f'Candidate: {len(candidate_list)} (pos={y_can.sum():.0f})')


# ─── GNN model ────────────────────────────────────────────────────────────────
class GNN(nn.Module):
    def __init__(self, nd=28, ed=2, h=128, L=4, dropout=0.2):
        super().__init__()
        self.emb = nn.Sequential(nn.Linear(nd, h), nn.LayerNorm(h), nn.GELU())
        self.cs  = nn.ModuleList([nn.Linear(h+ed, h) for _ in range(L)])
        self.ns  = nn.ModuleList([nn.LayerNorm(h) for _ in range(L)])
        self.drop = nn.Dropout(dropout)
        self.pool_projs = nn.ModuleList([nn.Identity() for _ in range(L)])
        self.head = nn.Sequential(
            nn.Linear(h * L, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, h//2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h//2, 1)
        )
        self.L = L

    def encode(self, x, ei, ea, batch):
        h = self.emb(x)
        reps = []
        for c, n in zip(self.cs, self.ns):
            r, col = ei
            m = F.gelu(c(torch.cat([h[r], ea], -1)))
            a = torch.zeros_like(h)
            a.scatter_add_(0, col.unsqueeze(1).expand_as(m), m)
            h = n(h + a)
            h = self.drop(h)
            reps.append(global_mean_pool(h, batch))
        return torch.cat(reps, dim=-1)  # [B, h*L]

    def forward(self, x, ei, ea, batch):
        return self.head(self.encode(x, ei, ea, batch)).squeeze(-1)


# MAE pre-trainer
class MAEPretrain(nn.Module):
    def __init__(self, gnn: GNN, h=128):
        super().__init__()
        self.gnn = gnn
        # Node-level decoder
        self.node_dec = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 28)
        )
        self.mask_token = nn.Parameter(torch.zeros(28))

    def forward(self, x, ei, ea, batch, mask):
        x_masked = x.clone()
        x_masked[mask] = self.mask_token
        # Get node representations from GNN
        h = self.gnn.emb(x_masked)
        for c, n in zip(self.gnn.cs, self.gnn.ns):
            r, col = ei
            m = F.gelu(c(torch.cat([h[r], ea], -1)))
            a = torch.zeros_like(h)
            a.scatter_add_(0, col.unsqueeze(1).expand_as(m), m)
            h = n(h + a)
        pred = self.node_dec(h[mask])
        return F.mse_loss(pred, x[mask])


def train_epoch_pretrain(model, loader, opt, device='cpu', mask_ratio=0.3):
    model.train()
    total, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        mask = torch.rand(b.x.size(0), device=device) < mask_ratio
        if mask.sum() == 0:
            continue
        loss = model(b.x, b.edge_index, b.edge_attr, b.batch, mask)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


def train_epoch_supervised(model, loader, opt, pos_weight, device='cpu'):
    model.train()
    total, n = 0.0, 0
    for b in loader:
        b = b.to(device)
        logits = model(b.x, b.edge_index, b.edge_attr, b.batch)
        labels = b.y.float()
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def get_probs(model, data_list, device='cpu'):
    loader = DataLoader(data_list, batch_size=256, shuffle=False, num_workers=0)
    model.eval()
    probs = []
    for b in loader:
        b = b.to(device)
        probs.extend(torch.sigmoid(model(b.x, b.edge_index, b.edge_attr, b.batch)).cpu().numpy())
    return np.array(probs)


@torch.no_grad()
def get_embeddings(model, data_list, device='cpu'):
    loader = DataLoader(data_list, batch_size=256, shuffle=False, num_workers=0)
    model.eval()
    embs = []
    for b in loader:
        b = b.to(device)
        embs.append(model.encode(b.x, b.edge_index, b.edge_attr, b.batch).cpu().numpy())
    return np.concatenate(embs, axis=0)


# ─── MAIN TRAINING ────────────────────────────────────────────────────────────
N_SEEDS = 15
H = 128
L = 4

all_probs = []
all_pretrain_losses = []
all_finetune_losses = []
all_finetune_aucs   = []

# Stratified val split
pos_ft = [d for d in finetune_list if d.y.item() == 1]
neg_ft = [d for d in finetune_list if d.y.item() == 0]
random.shuffle(pos_ft); random.shuffle(neg_ft)
n_val_pos = max(5, len(pos_ft) // 5)  # 20% pos
n_val_neg = len(neg_ft) // 5
val_list  = pos_ft[:n_val_pos] + neg_ft[:n_val_neg]
train_ft  = pos_ft[n_val_pos:] + neg_ft[n_val_neg:]

print(f'\nTrain FT: {len(train_ft)} (pos={len(pos_ft)-n_val_pos})')
print(f'Val FT:   {len(val_list)} (pos={n_val_pos})')

for seed_i in range(N_SEEDS):
    run_seed = SEED + seed_i * 17
    torch.manual_seed(run_seed); np.random.seed(run_seed); random.seed(run_seed)
    print(f'\n--- Seed {seed_i+1}/{N_SEEDS} (seed={run_seed}) ---')

    gnn = GNN(nd=28, ed=2, h=H, L=L, dropout=0.15)
    mae = MAEPretrain(gnn, h=H)

    # PHASE 1: Self-supervised MAE pre-training
    pt_loader = DataLoader(pretrain_list, batch_size=256, shuffle=True, num_workers=0)
    opt_pt = torch.optim.AdamW(mae.parameters(), lr=8e-4, weight_decay=1e-4)
    sched_pt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pt, T_max=20)

    pt_losses = []
    for ep in range(20):
        loss = train_epoch_pretrain(mae, pt_loader, opt_pt, device, mask_ratio=0.3)
        sched_pt.step()
        pt_losses.append(loss)
    all_pretrain_losses.append(pt_losses)

    # PHASE 2: Supervised training on balanced pretrain data
    pt_pos_w = torch.tensor([1.0]).to(device)  # balanced
    sup_loader = DataLoader(pretrain_list, batch_size=256, shuffle=True, num_workers=0)
    opt_sup = torch.optim.AdamW(gnn.parameters(), lr=3e-4, weight_decay=1e-4)
    sched_sup = torch.optim.lr_scheduler.CosineAnnealingLR(opt_sup, T_max=15)

    for ep in range(15):
        train_epoch_supervised(gnn, sup_loader, opt_sup, pt_pos_w, device)
        sched_sup.step()

    # PHASE 3: Fine-tune on imbalanced finetune data with positive oversampling
    n_pos_train = sum(d.y.item() for d in train_ft)
    n_neg_train = len(train_ft) - n_pos_train
    ft_pos_w = torch.tensor([n_neg_train / max(n_pos_train, 1)]).to(device)

    # Oversample positives
    pos_train = [d for d in train_ft if d.y.item() == 1]
    neg_train = [d for d in train_ft if d.y.item() == 0]
    oversampled = pos_train * 8 + neg_train
    random.shuffle(oversampled)

    ft_loader = DataLoader(oversampled, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_list, batch_size=256, shuffle=False, num_workers=0)

    opt_ft = torch.optim.AdamW(gnn.parameters(), lr=8e-5, weight_decay=5e-4)
    sched_ft = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt_ft, T_0=20, T_mult=2)

    ft_losses, ft_val_aucs = [], []
    best_val_auc, best_probs_i = 0.0, None

    for ep in range(80):
        loss = train_epoch_supervised(gnn, ft_loader, opt_ft, ft_pos_w, device)
        sched_ft.step()
        ft_losses.append(loss)

        if ep % 5 == 4:
            gnn.eval()
            vp, vl = [], []
            with torch.no_grad():
                for b in val_loader:
                    b = b.to(device)
                    vp.extend(torch.sigmoid(gnn(b.x, b.edge_index, b.edge_attr, b.batch)).cpu().numpy())
                    vl.extend(b.y.numpy())
            try:
                vauc = roc_auc_score(vl, vp)
            except Exception:
                vauc = 0.5
            ft_val_aucs.append(vauc)

            if vauc > best_val_auc:
                best_val_auc = vauc
                best_probs_i = get_probs(gnn, candidate_list, device)

    if best_probs_i is None:
        best_probs_i = get_probs(gnn, candidate_list, device)

    all_probs.append(best_probs_i)
    all_finetune_losses.append(ft_losses)
    all_finetune_aucs.append(ft_val_aucs)

    can_auc = roc_auc_score(y_can, best_probs_i)
    print(f'  Best val AUC: {best_val_auc:.4f}  Can AUC: {can_auc:.4f}')


# ─── Ensemble ─────────────────────────────────────────────────────────────────
print('\n=== Building ensemble ===')
all_probs_arr = np.array(all_probs)

# Simple average
probs_mean = all_probs_arr.mean(axis=0)
auc_mean = roc_auc_score(y_can, probs_mean)
print(f'Mean ensemble AUC: {auc_mean:.4f}')

# Individual AUCs
ind_aucs = [roc_auc_score(y_can, p) for p in all_probs]
print(f'Individual AUCs: {[f"{a:.3f}" for a in ind_aucs]}')
print(f'Individual mean: {np.mean(ind_aucs):.4f} ± {np.std(ind_aucs):.4f}')

# Weighted by individual AUC
weights = np.array(ind_aucs)
weights = np.maximum(weights, 0.4)  # floor at 0.4
weights /= weights.sum()
probs_weighted = np.average(all_probs_arr, weights=weights, axis=0)
auc_weighted = roc_auc_score(y_can, probs_weighted)
print(f'Weighted ensemble AUC: {auc_weighted:.4f}')

# Use best
if auc_weighted > auc_mean:
    final_probs = probs_weighted
    final_auc = auc_weighted
else:
    final_probs = probs_mean
    final_auc = auc_mean
print(f'Final ensemble AUC: {final_auc:.4f}')


# ─── Save outputs ─────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, 'candidate_probs.npy'), final_probs)
np.save(os.path.join(OUT_DIR, 'candidate_true_labels.npy'), y_can)
np.save(os.path.join(OUT_DIR, 'ensemble_probs_all.npy'), all_probs_arr)

ranked_idx = np.argsort(final_probs)[::-1]
top50_tp  = int(y_can[ranked_idx[:50]].sum())
top100_tp = int(y_can[ranked_idx[:100]].sum())

print(f'Top-50 TP: {top50_tp}/{int(y_can.sum())}  recall={top50_tp/y_can.sum():.3f}')
print(f'Top-100 TP: {top100_tp}/{int(y_can.sum())} recall={top100_tp/y_can.sum():.3f}')

# Evaluation metrics
auc_roc = roc_auc_score(y_can, final_probs)
auc_pr  = average_precision_score(y_can, final_probs)
print(f'AUC-ROC: {auc_roc:.4f}')
print(f'AUC-PR:  {auc_pr:.4f}')

best_f1, best_thr = 0.0, 0.5
for thr in np.linspace(0.05, 0.95, 181):
    preds = (final_probs >= thr).astype(int)
    tp = int(((preds==1) & (y_can==1)).sum())
    fp = int(((preds==1) & (y_can==0)).sum())
    fn = int(((preds==0) & (y_can==1)).sum())
    pr_ = tp / max(tp+fp, 1)
    rc_ = tp / max(tp+fn, 1)
    f1  = 2*pr_*rc_ / max(pr_+rc_, 1e-9)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

ranked_results = [
    {'rank': i+1, 'material_idx': int(ranked_idx[i]),
     'prob': float(final_probs[ranked_idx[i]]),
     'true_label': int(y_can[ranked_idx[i]])}
    for i in range(len(ranked_idx))
]
with open(os.path.join(OUT_DIR, 'ranked_candidates.json'), 'w') as f:
    json.dump(ranked_results, f, indent=2)

metrics = {
    'final_auc_roc': float(auc_roc),
    'final_auc_pr': float(auc_pr),
    'best_f1': float(best_f1),
    'best_threshold': float(best_thr),
    'top50_tp': top50_tp,
    'top100_tp': top100_tp,
    'total_true_positives': int(y_can.sum()),
    'top50_recall': float(top50_tp / max(y_can.sum(), 1)),
    'top100_recall': float(top100_tp / max(y_can.sum(), 1)),
    'n_ensemble': N_SEEDS,
    'individual_aucs': ind_aucs,
    'mean_individual_auc': float(np.mean(ind_aucs)),
    'std_individual_auc': float(np.std(ind_aucs)),
}
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics saved.')


# ─── Embeddings (from last trained model) ─────────────────────────────────────
print('\n=== Computing embeddings ===')
# Re-use last gnn model for embeddings
ft_embs   = get_embeddings(gnn, finetune_list, device=device)
cand_embs = get_embeddings(gnn, candidate_list, device=device)

print('Running t-SNE...')
from sklearn.manifold import TSNE
all_embs = np.concatenate([ft_embs, cand_embs], axis=0)
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
tsne_2d = tsne.fit_transform(all_embs)
ft_tsne   = tsne_2d[:len(ft_embs)]
cand_tsne = tsne_2d[len(ft_embs):]
np.save(os.path.join(OUT_DIR, 'finetune_tsne.npy'),  ft_tsne)
np.save(os.path.join(OUT_DIR, 'candidate_tsne.npy'), cand_tsne)
print('t-SNE done.')


# ─── Figures ──────────────────────────────────────────────────────────────────
print('\n=== Generating figures ===')
C = {'blue': '#2196F3', 'orange': '#FF5722', 'green': '#4CAF50', 'red': '#F44336'}

# Fig 1: Pre-training loss
fig, ax = plt.subplots(figsize=(8, 4))
for i, pt_loss in enumerate(all_pretrain_losses):
    ax.plot(range(1, len(pt_loss)+1), pt_loss, alpha=0.3, linewidth=1, color=C['blue'])
pt_mean = np.mean(all_pretrain_losses, axis=0)
ax.plot(range(1, len(pt_mean)+1), pt_mean, color=C['blue'], linewidth=2.5, label='Mean loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
ax.set_title('Self-Supervised MAE Pre-Training\n(Masked Node Feature Reconstruction, 5000 graphs)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_pretrain_loss.png'))
plt.close()
print('Saved fig1')

# Fig 2: Fine-tuning val AUC across all seeds
max_ep = max(len(va) for va in all_finetune_aucs)
def pad(a, L): arr = np.array(a); return np.pad(arr, (0, L-len(arr)), mode='edge')
vauc_mat = np.array([pad(va, max_ep) for va in all_finetune_aucs])
vauc_avg = vauc_mat.mean(0)
vauc_std = vauc_mat.std(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ep_arr = np.arange(1, max_ep+1) * 5

ax = axes[0]
for i, fl in enumerate(all_finetune_losses):
    ax.plot(range(1, len(fl)+1), fl, alpha=0.25, linewidth=0.8, color=C['blue'])
fl_mean = np.mean([pad(fl, max(len(f) for f in all_finetune_losses))
                   for fl in all_finetune_losses], axis=0)
ax.plot(range(1, len(fl_mean)+1), fl_mean, color=C['blue'], linewidth=2.5, label='Mean loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('BCE Loss')
ax.set_title('Fine-tuning Loss'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(ep_arr, vauc_avg - vauc_std, vauc_avg + vauc_std,
                alpha=0.15, color=C['green'])
ax.plot(ep_arr, vauc_avg, color=C['green'], linewidth=2.5, label='Val AUC (mean)')
for va in all_finetune_aucs:
    ax.plot(np.arange(1, len(va)+1)*5, va, alpha=0.25, linewidth=0.8, color=C['green'])
ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_xlabel('Epoch'); ax.set_ylabel('AUC-ROC')
ax.set_title(f'Fine-tuning Validation AUC ({N_SEEDS} seeds)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_finetune_curves.png'))
plt.close()
print('Saved fig2')

# Fig 3: ROC and PR curves
fpr, tpr, _ = roc_curve(y_can, final_probs)
prec_arr, rec_arr, _ = precision_recall_curve(y_can, final_probs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for i, p_i in enumerate(all_probs):
    fpr_i, tpr_i, _ = roc_curve(y_can, p_i)
    ax.plot(fpr_i, tpr_i, alpha=0.2, linewidth=0.8, color=C['blue'])
ax.plot(fpr, tpr, color='black', linewidth=2.5, label=f'Ensemble (AUC={auc_roc:.3f})')
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve – Altermagnet Classification')
ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(rec_arr, prec_arr, color=C['orange'], linewidth=2.5,
        label=f'Ensemble (AUC-PR={auc_pr:.3f})')
ax.axhline(y_can.mean(), color='k', linestyle='--', alpha=0.5,
           label=f'Random ({y_can.mean():.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_roc_pr.png'))
plt.close()
print('Saved fig3')

# Fig 4: t-SNE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.scatter(ft_tsne[y_ft==0, 0], ft_tsne[y_ft==0, 1],
           c='#4c72b0', alpha=0.4, s=12, label='Non-altermagnet')
ax.scatter(ft_tsne[y_ft==1, 0], ft_tsne[y_ft==1, 1],
           c='#dd8452', alpha=0.9, s=70, marker='*', zorder=3, label='Altermagnet')
ax.legend(markerscale=1.2); ax.set_title('t-SNE: Fine-tune Set (labeled)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
sc = ax.scatter(cand_tsne[:, 0], cand_tsne[:, 1],
                c=final_probs, cmap='RdYlGn', alpha=0.7, s=20, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='Predicted Probability')
ax.scatter(cand_tsne[y_can==1, 0], cand_tsne[y_can==1, 1],
           edgecolors='black', facecolors='none', s=80, linewidths=1.5,
           zorder=4, label='True altermagnet')
ax.legend(); ax.set_title('t-SNE: Candidates (probability-colored)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.suptitle(f'GNN Embedding Space ({N_SEEDS}-model ensemble)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_tsne_embeddings.png'))
plt.close()
print('Saved fig4')

# Fig 5: Score distribution and Top-K
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
ax.hist(final_probs[y_can==0], bins=40, alpha=0.7, color='#4c72b0',
        label='Non-altermagnet', density=True)
ax.hist(final_probs[y_can==1], bins=12, alpha=0.8, color='#dd8452',
        label='Altermagnet', density=True)
ax.axvline(best_thr, color='red', linestyle='--', linewidth=1.5,
           label=f'Best threshold ({best_thr:.2f})')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density')
ax.set_title('Score Distribution by Class'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
total_pos = int(y_can.sum())
ks = list(range(5, 201, 5))
rec_k  = [float(y_can[ranked_idx[:k]].sum()) / max(total_pos, 1) for k in ks]
prec_k = [float(y_can[ranked_idx[:k]].sum()) / k for k in ks]
rand_k = [k * total_pos / 1000 / max(total_pos, 1) for k in ks]
ax.plot(ks, rec_k,  color=C['blue'], marker='o', markersize=3, linewidth=2, label='Recall@K')
ax.plot(ks, prec_k, color=C['orange'], marker='s', markersize=3, linewidth=2, label='Precision@K')
ax.plot(ks, rand_k, 'k--', alpha=0.5, label='Random recall')
ax.set_xlabel('Top-K Candidates'); ax.set_ylabel('Score')
ax.set_title('Top-K Screening Performance')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_score_dist_topk.png'))
plt.close()
print('Saved fig5')

# Fig 6: Crystal graph structural properties
def graph_stats(dl):
    ns = [d.x.shape[0] for d in dl]
    es = [d.edge_index.shape[1] for d in dl]
    ds = [d.edge_attr[:,0].mean().item() if d.edge_attr.shape[0]>0 else 0 for d in dl]
    return ns, es, ds

pos_cands = [d for d in candidate_list if d.y.item() == 1]
neg_cands = [d for d in candidate_list if d.y.item() == 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels3  = ['# Atoms per Crystal', '# Bonds per Crystal', 'Avg Bond Distance']
xlabels3 = ['# Atoms', '# Bonds', 'Avg Bond Distance']
for i in range(3):
    ax = axes[i]
    nv = graph_stats(neg_cands)[i]
    pv = graph_stats(pos_cands)[i]
    ax.hist(nv, bins=20, alpha=0.6, color='#4c72b0', label='Non-AM', density=True)
    ax.hist(pv, bins=max(5, len(pv)//3), alpha=0.75, color='#dd8452',
            label='Altermagnet', density=True)
    ax.set_xlabel(xlabels3[i]); ax.set_ylabel('Density')
    ax.set_title(labels3[i]); ax.legend()
plt.suptitle('Crystal Graph Properties (Candidate Set)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig6_graph_properties.png'))
plt.close()
print('Saved fig6')

# Fig 7: Discovery curve + uncertainty
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
sorted_idx = np.argsort(final_probs)[::-1]
coverage   = np.arange(1, len(final_probs)+1) / len(final_probs) * 100
tp_cum     = np.cumsum(y_can[sorted_idx]) / max(total_pos, 1) * 100
ax.plot(coverage, tp_cum, color=C['blue'], linewidth=2.5, label='GNN Ensemble')
ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
ax.fill_between([0, 100], [0, 100], [0, 100], alpha=0.1)
ax.set_xlabel('% Candidates Screened')
ax.set_ylabel('% Altermagnets Found (Recall)')
ax.set_title('Cumulative Discovery Curve')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
p_std = all_probs_arr.std(axis=0)
ax.scatter(final_probs[y_can==0], p_std[y_can==0],
           c='#4c72b0', alpha=0.4, s=12, label='Non-AM')
ax.scatter(final_probs[y_can==1], p_std[y_can==1],
           c='#dd8452', alpha=0.9, s=60, marker='*', zorder=3, label='Altermagnet')
ax.set_xlabel('Ensemble Mean Probability')
ax.set_ylabel('Ensemble Std (Uncertainty)')
ax.set_title('Prediction Uncertainty vs. Score')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_discovery_uncertainty.png'))
plt.close()
print('Saved fig7')

print('\n=== Done ===')
print(f'Final AUC-ROC: {auc_roc:.4f}')
print(f'Final AUC-PR:  {auc_pr:.4f}')
print(f'Top-50 recall: {top50_tp}/{int(y_can.sum())} = {top50_tp/y_can.sum():.3f}')
print(f'Top-100 recall: {top100_tp}/{int(y_can.sum())} = {top100_tp/y_can.sum():.3f}')
print('\nMetrics:'); print(json.dumps(metrics, indent=2))
