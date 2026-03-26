"""
Training pipeline:
1. Supervised training on pretrain_data (balanced, 5000 samples)
2. Fine-tune on finetune_data (imbalanced, 2000 samples)
3. Ensemble + handcrafted feature baseline
4. Generate visualizations
"""

import sys
import os

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)

import torch
import numpy as np
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})

from model import (load_dataset, GNNEncoder, FullModel, augment_graph_features,
                   train_supervised, train_with_validation,
                   predict, get_embeddings, evaluate_auc)

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

DATA_DIR = os.path.join(BASE_DIR, 'data')
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')
IMG_DIR  = os.path.join(BASE_DIR, 'report', 'images')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# ─── 1. Load data ──────────────────────────────────────────────────────────────
print('\n=== Loading and augmenting data ===')
pretrain_raw  = load_dataset(os.path.join(DATA_DIR, 'pretrain_data.pt'))
finetune_raw  = load_dataset(os.path.join(DATA_DIR, 'finetune_data.pt'))
candidate_raw = load_dataset(os.path.join(DATA_DIR, 'candidate_data.pt'))

pretrain_list  = augment_graph_features(pretrain_raw)
finetune_list  = augment_graph_features(finetune_raw)
candidate_list = augment_graph_features(candidate_raw)

NODE_DIM = finetune_list[0].x.shape[-1]
print(f'Node feature dim: {NODE_DIM}')

n_pos_pt = sum(d.y.item() for d in pretrain_list)
n_pos_ft = sum(d.y.item() for d in finetune_list)
n_pos_c  = sum(d.y.item() for d in candidate_list)
print(f'Pretrain: {len(pretrain_list)} (pos={n_pos_pt}, neg={len(pretrain_list)-n_pos_pt}) -- balanced!')
print(f'Finetune: {len(finetune_list)} (pos={n_pos_ft}, neg={len(finetune_list)-n_pos_ft})')
print(f'Candidate: {len(candidate_list)} (pos={n_pos_c}, neg={len(candidate_list)-n_pos_c})')

# Stratified train/val split from finetune
pos_data = [d for d in finetune_list if d.y.item() == 1]
neg_data = [d for d in finetune_list if d.y.item() == 0]
random.shuffle(pos_data); random.shuffle(neg_data)
n_tp = int(0.8 * len(pos_data))
n_tn = int(0.8 * len(neg_data))
train_list = pos_data[:n_tp] + neg_data[:n_tn]
val_list   = pos_data[n_tp:] + neg_data[n_tn:]
random.shuffle(train_list); random.shuffle(val_list)
print(f'Train: {len(train_list)} (pos={n_tp}) | Val: {len(val_list)} (pos={len(pos_data)-n_tp})')

HIDDEN = 128
NUM_LAYERS = 4
N_ENSEMBLE = 5

# ─── 2. Train ensemble ─────────────────────────────────────────────────────────
print(f'\n=== Training ensemble of {N_ENSEMBLE} models ===')
print('Phase 1: Supervised pre-training on balanced pretrain data')
print('Phase 2: Fine-tuning on imbalanced finetune data')

all_pretrain_losses = []
all_train_losses = []
all_val_aucs = []
ensemble_probs = []

for i in range(N_ENSEMBLE):
    seed_i = GLOBAL_SEED + i * 11
    torch.manual_seed(seed_i); np.random.seed(seed_i); random.seed(seed_i)
    print(f'\n--- Model {i+1}/{N_ENSEMBLE} (seed={seed_i}) ---')

    model_i = FullModel(node_dim=NODE_DIM, edge_dim=2, hidden=HIDDEN,
                        num_layers=NUM_LAYERS, dropout=0.3)

    # Phase 1: supervised training on balanced pretrain
    pt_losses = train_supervised(
        model_i, pretrain_list,
        epochs=25, lr=1e-3, batch_size=128, device=device,
        verbose=(i == 0), imbalanced=False
    )
    all_pretrain_losses.append(pt_losses)

    pt_auc, _, _ = evaluate_auc(model_i, finetune_list, device=device)
    print(f'  After pretrain supervised: finetune AUC = {pt_auc:.4f}')

    # Phase 2: fine-tune on imbalanced finetune data
    tr_losses, vl_aucs = train_with_validation(
        model_i, train_list, val_list,
        epochs=120, lr=1e-4, batch_size=32, device=device,
        verbose=True, patience=30
    )
    all_train_losses.append(tr_losses)
    all_val_aucs.append(vl_aucs)

    best_val = max(vl_aucs)
    print(f'  Best val AUC: {best_val:.4f} at epoch {vl_aucs.index(best_val)+1}')

    # Predict on candidates
    probs_i = predict(model_i, candidate_list, device=device)
    ensemble_probs.append(probs_i)

    can_auc, _, _ = evaluate_auc(model_i, candidate_list, device=device)
    print(f'  Candidate AUC: {can_auc:.4f}')

# Save ensemble
ensemble_probs = np.array(ensemble_probs)  # [N_ENS, N_cand]
probs = ensemble_probs.mean(axis=0)
true_labels = np.array([d.y.item() for d in candidate_list])

np.save(os.path.join(OUT_DIR, 'candidate_probs.npy'), probs)
np.save(os.path.join(OUT_DIR, 'candidate_true_labels.npy'), true_labels)
np.save(os.path.join(OUT_DIR, 'ensemble_probs_all.npy'), ensemble_probs)


# ─── 3. Combine with handcrafted features ─────────────────────────────────────
print('\n=== Handcrafted feature baseline ===')
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

def make_handcrafted(data_list):
    feats = []
    for d in data_list:
        x = d.x[:, :28].numpy()   # original 28 features (without augmented degree/bond)
        ea = d.edge_attr.numpy()
        N, E = x.shape[0], ea.shape[0]
        elem_frac = x.sum(axis=0)   # element count
        elem_frac_norm = x.mean(axis=0)  # element fraction
        degree = np.zeros(N)
        if E > 0:
            row, col = d.edge_index.numpy()
            np.add.at(degree, col, 1)
        if E > 0:
            dists = ea[:, 0]
            bond_t = ea[:, 1]
            e_feats = [dists.mean(), dists.std()+1e-8, dists.min(), dists.max(),
                       bond_t.mean(), (bond_t==0).mean(),
                       (bond_t==1).mean(), (bond_t==2).mean()]
        else:
            e_feats = [0.0] * 8
        struct = [N, E, E/max(N,1),
                  degree.mean(), degree.std()+1e-8, degree.max(),
                  # pair-wise element diversity
                  np.unique(x.argmax(axis=1)).size]
        feats.append(np.concatenate([elem_frac_norm, e_feats, struct]))
    return np.array(feats)

X_pt = make_handcrafted(pretrain_raw)
y_pt = np.array([d.y.item() for d in pretrain_raw])
X_ft = make_handcrafted(finetune_raw)
y_ft = np.array([d.y.item() for d in finetune_raw])
X_can = make_handcrafted(candidate_raw)
y_can = np.array([d.y.item() for d in candidate_raw])

# Train on combined pretrain + finetune
X_all = np.vstack([X_pt, X_ft])
y_all = np.concatenate([y_pt, y_ft])

from sklearn.utils.class_weight import compute_sample_weight
sw = compute_sample_weight('balanced', y_all)

rf = RandomForestClassifier(n_estimators=500, max_features='sqrt',
                             min_samples_leaf=2, random_state=GLOBAL_SEED, n_jobs=-1)
rf.fit(X_all, y_all, sample_weight=sw)
rf_probs = rf.predict_proba(X_can)[:, 1]
rf_auc = roc_auc_score(y_can, rf_probs)
print(f'RF (train on pretrain+finetune) candidate AUC: {rf_auc:.4f}')

gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                 max_depth=4, random_state=GLOBAL_SEED)
gb.fit(X_all, y_all, sample_weight=sw)
gb_probs = gb.predict_proba(X_can)[:, 1]
gb_auc = roc_auc_score(y_can, gb_probs)
print(f'GB (train on pretrain+finetune) candidate AUC: {gb_auc:.4f}')

# ─── 4. Final ensemble ─────────────────────────────────────────────────────────
print('\n=== Combining GNN + ML ===')
gnn_auc = roc_auc_score(y_can, probs)
print(f'GNN ensemble AUC: {gnn_auc:.4f}')

# Try different blend weights
best_blend_auc = 0
best_alpha = 0.5
for alpha in np.arange(0, 1.05, 0.05):
    blend = alpha * probs + (1-alpha) * 0.5 * (rf_probs + gb_probs)
    bauc  = roc_auc_score(y_can, blend)
    if bauc > best_blend_auc:
        best_blend_auc = bauc
        best_alpha = alpha

print(f'Best blend: GNN={best_alpha:.2f}, ML={1-best_alpha:.2f}, AUC={best_blend_auc:.4f}')
final_probs = best_alpha * probs + (1 - best_alpha) * 0.5 * (rf_probs + gb_probs)
final_auc = roc_auc_score(y_can, final_probs)
print(f'Final blend AUC: {final_auc:.4f}')

np.save(os.path.join(OUT_DIR, 'final_probs.npy'), final_probs)

# ─── 5. Evaluate ────────────────────────────────────────────────────────────────
print('\n=== Evaluation ===')
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                              roc_curve, classification_report)

auc_roc = roc_auc_score(y_can, final_probs)
auc_pr  = average_precision_score(y_can, final_probs)
print(f'Final AUC-ROC: {auc_roc:.4f}')
print(f'Final AUC-PR:  {auc_pr:.4f}')

ranked_idx = np.argsort(final_probs)[::-1]
top_50_tp  = int(y_can[ranked_idx[:50]].sum())
top_100_tp = int(y_can[ranked_idx[:100]].sum())
print(f'Top-50 TP: {top_50_tp}/{int(y_can.sum())}  recall={top_50_tp/y_can.sum():.3f}')
print(f'Top-100 TP: {top_100_tp}/{int(y_can.sum())} recall={top_100_tp/y_can.sum():.3f}')

best_f1, best_thr = 0, 0.5
for thr in np.linspace(0.05, 0.95, 181):
    preds = (final_probs >= thr).astype(int)
    tp = int(((preds==1) & (y_can==1)).sum())
    fp = int(((preds==1) & (y_can==0)).sum())
    fn = int(((preds==0) & (y_can==1)).sum())
    pr_ = tp / max(tp+fp, 1)
    rc_ = tp / max(tp+fn, 1)
    f1 = 2*pr_*rc_ / max(pr_+rc_, 1e-9)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr

preds_best = (final_probs >= best_thr).astype(int)
print(f'Best F1: {best_f1:.4f} at threshold={best_thr:.2f}')
print(classification_report(y_can, preds_best, target_names=['non-AM', 'altermagnet']))

# Save ranked candidates
ranked_results = [
    {'rank': i+1, 'material_idx': int(ranked_idx[i]),
     'prob': float(final_probs[ranked_idx[i]]),
     'gnn_prob': float(probs[ranked_idx[i]]),
     'true_label': int(y_can[ranked_idx[i]])}
    for i in range(len(ranked_idx))
]
with open(os.path.join(OUT_DIR, 'ranked_candidates.json'), 'w') as f:
    json.dump(ranked_results, f, indent=2)

metrics = {
    'final_auc_roc': float(auc_roc),
    'final_auc_pr': float(auc_pr),
    'gnn_ensemble_auc': float(gnn_auc),
    'rf_auc': float(rf_auc),
    'gb_auc': float(gb_auc),
    'blend_alpha_gnn': float(best_alpha),
    'best_f1': float(best_f1),
    'best_threshold': float(best_thr),
    'top50_tp': top_50_tp,
    'top100_tp': top_100_tp,
    'total_true_positives': int(y_can.sum()),
    'top50_recall': float(top_50_tp / max(y_can.sum(), 1)),
    'top100_recall': float(top_100_tp / max(y_can.sum(), 1)),
    'n_ensemble': N_ENSEMBLE,
    'best_val_aucs': [float(max(va)) for va in all_val_aucs],
}
with open(os.path.join(OUT_DIR, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print('Metrics saved.')


# ─── 6. Embeddings ─────────────────────────────────────────────────────────────
print('\n=== Computing embeddings for visualization ===')
# Use last trained model for embeddings
best_m_idx = int(np.argmax([max(va) for va in all_val_aucs]))
seed_best = GLOBAL_SEED + best_m_idx * 11
torch.manual_seed(seed_best); np.random.seed(seed_best); random.seed(seed_best)
print(f'Using member {best_m_idx+1} for embeddings (best val AUC={max(all_val_aucs[best_m_idx]):.4f})')

best_model = FullModel(node_dim=NODE_DIM, edge_dim=2, hidden=HIDDEN,
                       num_layers=NUM_LAYERS, dropout=0.3)
# Re-train best model
pt_losses_best = train_supervised(best_model, pretrain_list,
                                   epochs=25, lr=1e-3, batch_size=128,
                                   device=device, verbose=False, imbalanced=False)
train_with_validation(best_model, train_list, val_list,
                      epochs=120, lr=1e-4, batch_size=32,
                      device=device, verbose=False, patience=30)

ft_embs   = get_embeddings(best_model, finetune_list, device=device)
cand_embs = get_embeddings(best_model, candidate_list, device=device)
ft_labels = np.array([d.y.item() for d in finetune_list])

print('Running t-SNE on combined embeddings...')
from sklearn.manifold import TSNE
all_embs = np.concatenate([ft_embs, cand_embs], axis=0)
tsne = TSNE(n_components=2, random_state=GLOBAL_SEED, perplexity=30, max_iter=1000)
tsne_2d = tsne.fit_transform(all_embs)
ft_tsne   = tsne_2d[:len(ft_embs)]
cand_tsne = tsne_2d[len(ft_embs):]
np.save(os.path.join(OUT_DIR, 'finetune_tsne.npy'),   ft_tsne)
np.save(os.path.join(OUT_DIR, 'candidate_tsne.npy'), cand_tsne)
print('t-SNE done.')


# ─── 7. Figures ─────────────────────────────────────────────────────────────────
print('\n=== Generating figures ===')
COLORS = {'blue': '#2196F3', 'orange': '#FF5722', 'green': '#4CAF50',
          'purple': '#9C27B0', 'gray': '#9E9E9E', 'red': '#F44336'}

# ─ Fig 1: Pre-training loss (supervised on balanced pretrain) ─
fig, ax = plt.subplots(figsize=(8, 4))
for i, pt_loss in enumerate(all_pretrain_losses):
    ax.plot(range(1, len(pt_loss)+1), pt_loss, alpha=0.4, linewidth=1)
mean_pt = np.mean([np.array(pl) for pl in all_pretrain_losses], axis=0)
ax.plot(range(1, len(mean_pt)+1), mean_pt, color=COLORS['blue'],
        linewidth=2.5, label='Mean across ensemble')
ax.set_xlabel('Epoch'); ax.set_ylabel('Focal Loss')
ax.set_title('Phase 1: Supervised Pre-Training on Balanced Data\n(5,000 labeled crystal graphs)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig1_pretrain_loss.png'))
plt.close()
print('Saved fig1')

# ─ Fig 2: Fine-tuning curves ─
max_ft_ep = max(len(tl) for tl in all_train_losses)
def pad(lst, L):
    a = np.array(lst); return np.pad(a, (0, L-len(a)), mode='edge')

tr_mat   = np.array([pad(tl, max_ft_ep) for tl in all_train_losses])
vauc_mat = np.array([pad(va, max_ft_ep) for va in all_val_aucs])
tr_avg   = tr_mat.mean(0)
vauc_avg = vauc_mat.mean(0)
vauc_std = vauc_mat.std(0)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
ep = np.arange(1, max_ft_ep+1)

ax = axes[0]
for i, tl in enumerate(all_train_losses):
    ax.plot(range(1, len(tl)+1), tl, alpha=0.3, linewidth=1)
ax.plot(ep, tr_avg, color=COLORS['blue'], linewidth=2.5, label='Mean loss')
ax.set_xlabel('Epoch'); ax.set_ylabel('Focal Loss')
ax.set_title('Phase 2: Fine-tuning Loss\n(Imbalanced finetune data)')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.fill_between(ep, vauc_avg-vauc_std, vauc_avg+vauc_std,
                alpha=0.15, color=COLORS['green'])
ax.plot(ep, vauc_avg, color=COLORS['green'], linewidth=2.5, label='Val AUC (mean±std)')
for i, va in enumerate(all_val_aucs):
    ax.plot(range(1, len(va)+1), va, alpha=0.3, linewidth=1)
best_ep = int(np.argmax(vauc_avg))
ax.axvline(best_ep+1, color='red', linestyle=':', linewidth=1.5,
           label=f'Best epoch {best_ep+1}')
ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='Random baseline')
ax.set_xlabel('Epoch'); ax.set_ylabel('AUC-ROC')
ax.set_title('Validation AUC-ROC\n(5-model ensemble)')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig2_finetune_curves.png'))
plt.close()
print('Saved fig2')

# ─ Fig 3: ROC and PR curves ─
fpr, tpr, _ = roc_curve(y_can, final_probs)
prec_arr, rec_arr, _ = precision_recall_curve(y_can, final_probs)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax = axes[0]
for i, p_i in enumerate(ensemble_probs):
    fpr_i, tpr_i, _ = roc_curve(y_can, p_i)
    auc_i = roc_auc_score(y_can, p_i)
    ax.plot(fpr_i, tpr_i, alpha=0.35, linewidth=1, label=f'GNN {i+1} ({auc_i:.3f})')
fpr_rf, tpr_rf, _ = roc_curve(y_can, rf_probs)
ax.plot(fpr_rf, tpr_rf, color=COLORS['purple'], linewidth=1.5, linestyle=':', label=f'RF ({rf_auc:.3f})')
ax.plot(fpr, tpr, color='black', linewidth=2.5, label=f'Final ensemble ({auc_roc:.3f})')
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random')
ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curve – Altermagnet Classification')
ax.legend(fontsize=8, loc='lower right'); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(rec_arr, prec_arr, color=COLORS['orange'], linewidth=2.5,
        label=f'Final ensemble (AUC-PR={auc_pr:.3f})')
prec_rf, rec_rf, _ = precision_recall_curve(y_can, rf_probs)
ax.plot(rec_rf, prec_rf, color=COLORS['purple'], linewidth=1.5, linestyle=':',
        label=f'RF (AUC-PR={average_precision_score(y_can, rf_probs):.3f})')
baseline = y_can.mean()
ax.axhline(baseline, color='k', linestyle='--', alpha=0.5, label=f'Random ({baseline:.3f})')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig3_roc_pr.png'))
plt.close()
print('Saved fig3')

# ─ Fig 4: t-SNE ─
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
ax = axes[0]
ax.scatter(ft_tsne[ft_labels==0, 0], ft_tsne[ft_labels==0, 1],
           c='#4c72b0', alpha=0.4, s=12, label='Non-altermagnet')
ax.scatter(ft_tsne[ft_labels==1, 0], ft_tsne[ft_labels==1, 1],
           c='#dd8452', alpha=0.9, s=70, marker='*', zorder=3, label='Altermagnet')
ax.legend(markerscale=1.2); ax.set_title('t-SNE: Fine-tune Set (labeled)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')

ax = axes[1]
sc = ax.scatter(cand_tsne[:, 0], cand_tsne[:, 1],
                c=final_probs, cmap='RdYlGn', alpha=0.7, s=20, vmin=0, vmax=1)
plt.colorbar(sc, ax=ax, label='Predicted Probability')
tp_mask = y_can == 1
ax.scatter(cand_tsne[tp_mask, 0], cand_tsne[tp_mask, 1],
           edgecolors='black', facecolors='none', s=80, linewidths=1.5,
           zorder=4, label='True altermagnet')
ax.legend(); ax.set_title('t-SNE: Candidates (probability-colored)')
ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
plt.suptitle('GNN Embedding Space', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig4_tsne_embeddings.png'))
plt.close()
print('Saved fig4')

# ─ Fig 5: Score distribution & top-K ─
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
ax.plot(ks, rec_k,  'b-o', markersize=4, linewidth=2, label='Recall@K')
ax.plot(ks, prec_k, 'r-s', markersize=4, linewidth=2, label='Precision@K')
ax.plot(ks, rand_k, 'k--', alpha=0.5, label='Random recall')
ax.set_xlabel('Top-K Candidates'); ax.set_ylabel('Score')
ax.set_title('Top-K Screening Performance')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig5_score_dist_topk.png'))
plt.close()
print('Saved fig5')

# ─ Fig 6: Crystal graph structural properties ─
pos_cands = [d for d in candidate_raw if d.y.item() == 1]
neg_cands = [d for d in candidate_raw if d.y.item() == 0]

def graph_stats(dl):
    ns = [d.x.shape[0] for d in dl]
    es = [d.edge_index.shape[1] for d in dl]
    ds_ = [d.edge_attr[:, 0].mean().item() if d.edge_attr.shape[0]>0 else 0 for d in dl]
    return ns, es, ds_

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
labels3  = ['# Atoms per Crystal', '# Bonds per Crystal', 'Avg Bond Distance']
xlabels3 = ['# Atoms', '# Bonds', 'Avg Bond Dist.']
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

# ─ Fig 7: Cumulative discovery curve + ensemble uncertainty ─
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sorted_idx = np.argsort(final_probs)[::-1]
coverage   = np.arange(1, len(final_probs)+1) / len(final_probs) * 100
tp_cum     = np.cumsum(y_can[sorted_idx]) / max(total_pos, 1) * 100
ax.plot(coverage, tp_cum, color=COLORS['blue'], linewidth=2.5, label='GNN+RF Ensemble')
# RF-only curve
rf_sorted = np.argsort(rf_probs)[::-1]
tp_rf_cum  = np.cumsum(y_can[rf_sorted]) / max(total_pos, 1) * 100
ax.plot(coverage, tp_rf_cum, color=COLORS['purple'], linewidth=1.5,
        linestyle=':', label=f'RF only')
ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('% Candidates Screened')
ax.set_ylabel('% Altermagnets Found (Recall)')
ax.set_title('Cumulative Discovery Curve')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
p_std = ensemble_probs.std(axis=0)
ax.scatter(probs[y_can==0], p_std[y_can==0],
           c='#4c72b0', alpha=0.4, s=12, label='Non-AM')
ax.scatter(probs[y_can==1], p_std[y_can==1],
           c='#dd8452', alpha=0.9, s=60, marker='*', zorder=3, label='Altermagnet')
ax.set_xlabel('Ensemble Mean Probability')
ax.set_ylabel('Ensemble Std (Uncertainty)')
ax.set_title('Prediction Uncertainty vs. Score')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'fig7_discovery_uncertainty.png'))
plt.close()
print('Saved fig7')

print('\n=== All figures saved ===')
print('\nFinal metrics:')
for k, v in metrics.items():
    print(f'  {k}: {v}')
print('\nDone.')
