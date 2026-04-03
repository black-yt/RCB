"""
DIDS-MFL: Disentangled Intrusion Detection System with Multi-scale Feature Learning
Efficient implementation using vectorized operations
"""
import sys
sys.path.insert(0, '/mnt/shared-storage-user/yetianlin/ResearchClawBench/.venv/lib/python3.13/site-packages')

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, accuracy_score,
                              precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import defaultdict
import json
import copy
import warnings
warnings.filterwarnings('ignore')

torch.manual_seed(42)
np.random.seed(42)

# Paths
WORKSPACE = Path('/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Information_003_20260401_224112')
DATA_PATH = WORKSPACE / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUTPUTS = WORKSPACE / 'outputs'
IMAGES = WORKSPACE / 'report' / 'images'

ATTACK_NAMES_REMAPPED = {
    0: 'Normal', 1: 'Exploits', 2: 'Recon', 3: 'DoS', 4: 'Generic',
    5: 'Shellcode', 6: 'Fuzzers', 7: 'Analysis', 8: 'Backdoors', 9: 'Worms'
}
CLASS_NAMES = [ATTACK_NAMES_REMAPPED[i] for i in range(10)]

# ===========================================================================
# 1. DATA LOADING
# ===========================================================================
print("="*60)
print("Step 1: Loading Data")
print("="*60)

from torch_geometric.data.temporal import TemporalData
data = torch.load(str(DATA_PATH), map_location='cpu', weights_only=False)

X = data.msg.numpy().astype(np.float32)
y_binary = data.label.numpy()
y_multi = data.attack.numpy()
t = data.t.numpy()
src = data.src.numpy()
dst = data.dst.numpy()
dt = data.dt.numpy()

# Sort temporally
sort_idx = np.argsort(t)
X, y_binary, y_multi = X[sort_idx], y_binary[sort_idx], y_multi[sort_idx]
t, src, dst, dt = t[sort_idx], src[sort_idx], dst[sort_idx], dt[sort_idx]

# Remap attack labels: 2->0(Normal), others 1-9
attack_remap = {2: 0, 0: 1, 1: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
y_mc = np.array([attack_remap[a] for a in y_multi])

print(f"Samples: {len(X)}, Features: {X.shape[1]}")

# Normalize
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
print("Data loaded and normalized.")

# ===========================================================================
# 2. STATISTICAL DISENTANGLEMENT (efficient PCA-based)
# ===========================================================================
print("\n"+"="*60)
print("Step 2: Statistical Disentanglement")
print("="*60)

from sklearn.decomposition import PCA

# PCA decomposition
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(X_norm)

# Compute discriminability of each PC
# Discriminability = variance of attacks / variance of normals
var_attack = np.var(X_pca[y_binary == 1], axis=0) + 1e-8
var_normal = np.var(X_pca[y_binary == 0], axis=0) + 1e-8
mean_attack = np.mean(X_pca[y_binary == 1], axis=0)
mean_normal = np.mean(X_pca[y_binary == 0], axis=0)

# Fisher's discriminant ratio
fdr = (mean_attack - mean_normal)**2 / (var_attack + var_normal)
disc_order = np.argsort(-fdr)

# Split PCs into attack-discriminative vs normal subspace
n_disc = 10
attack_pcs = disc_order[:n_disc]   # top discriminative
normal_pcs = disc_order[n_disc:]   # residual

X_attack_subspace = np.zeros_like(X_pca)
X_normal_subspace = np.zeros_like(X_pca)
X_attack_subspace[:, attack_pcs] = X_pca[:, attack_pcs]
X_normal_subspace[:, normal_pcs] = X_pca[:, normal_pcs]

# Reconstruct in original space for interpretability
X_attack_feat = pca.inverse_transform(X_attack_subspace)
X_normal_feat = pca.inverse_transform(X_normal_subspace)

print(f"FDR scores (top 5): {fdr[disc_order[:5]]}")
print(f"Attack subspace dims: {n_disc}, Normal subspace dims: {20-n_disc}")
np.save(OUTPUTS/'disc_scores.npy', fdr)

# ===========================================================================
# 3. DYNAMIC GRAPH DIFFUSION (efficient vectorized)
# ===========================================================================
print("\n"+"="*60)
print("Step 3: Dynamic Graph Diffusion")
print("="*60)

def efficient_graph_diffusion(X, src, dst, window=1000, max_neighbors=15):
    """
    Efficient graph diffusion: for each node (IP), aggregate features of
    recent flows involving the same IP using vectorized sliding window.
    """
    n, d = X.shape
    X_graph = X.copy()

    # Build src-indexed and dst-indexed feature aggregations
    # Using a sliding window over sorted-by-time flows

    # For each unique source IP, compute running mean of features
    print("  Building source IP feature aggregations...")
    src_agg = np.zeros_like(X)
    dst_agg = np.zeros_like(X)

    # Vectorized: for each flow, find nearby flows by same src/dst
    # Use a hash-based approach with small windows
    step = max(1, window // 10)

    for start in range(0, n, step):
        end = min(n, start + window)
        src_chunk = src[start:end]
        dst_chunk = dst[start:end]
        X_chunk = X[start:end]

        # For each unique src IP in this chunk
        unique_srcs = np.unique(src_chunk)
        for s in unique_srcs:
            mask = src_chunk == s
            if mask.sum() > 1:
                mean_feat = X_chunk[mask].mean(axis=0)
                # Assign to all flows with this src in this window
                global_mask = np.where((src[start:end] == s))[0] + start
                src_agg[global_mask] = mean_feat

        # For each unique dst IP
        unique_dsts = np.unique(dst_chunk)
        for d_ip in unique_dsts:
            mask = dst_chunk == d_ip
            if mask.sum() > 1:
                mean_feat = X_chunk[mask].mean(axis=0)
                global_mask = np.where((dst[start:end] == d_ip))[0] + start
                dst_agg[global_mask] = mean_feat

        if start % (step * 50) == 0:
            print(f"    Progress: {start}/{n}")

    # Diffused: self + src_neighbors + dst_neighbors
    X_graph = 0.5 * X + 0.3 * src_agg + 0.2 * dst_agg
    print(f"  Graph diffusion complete. Shape: {X_graph.shape}")
    return X_graph

# Use attack-enhanced features for diffusion
X_enhanced = np.concatenate([X_norm, X_attack_feat[:, :10]], axis=1)
print("Running graph diffusion...")
X_graph = efficient_graph_diffusion(X_enhanced, src, dst, window=2000)
np.save(OUTPUTS/'X_graph_diffused.npy', X_graph[:, :20])  # Save first 20 dims
print(f"Graph diffused shape: {X_graph.shape}")

# ===========================================================================
# 4. MULTI-SCALE TEMPORAL FEATURE EXTRACTION
# ===========================================================================
print("\n"+"="*60)
print("Step 4: Multi-Scale Temporal Features")
print("="*60)

def compute_rolling_stats(X, window_size):
    """Compute rolling mean and std for each feature."""
    n, d = X.shape
    roll_mean = np.zeros((n, d), dtype=np.float32)
    roll_std = np.zeros((n, d), dtype=np.float32)

    cumsum = np.cumsum(X, axis=0)
    cumsum2 = np.cumsum(X**2, axis=0)

    for i in range(n):
        start = max(0, i - window_size)
        end = i + 1
        cnt = end - start

        if start == 0:
            s = cumsum[i]
            s2 = cumsum2[i]
        else:
            s = cumsum[i] - cumsum[start-1]
            s2 = cumsum2[i] - cumsum2[start-1]

        roll_mean[i] = s / cnt
        roll_std[i] = np.sqrt(np.maximum(0, s2/cnt - (s/cnt)**2))

    return roll_mean, roll_std

print("Computing multi-scale rolling statistics...")
# Use a subset of features for efficiency
X_for_scale = X_norm[:, :10]  # top 10 features

# Fine scale (100 flows)
print("  Fine scale (100 flows)...")
mean_fine, std_fine = compute_rolling_stats(X_for_scale, 100)

# Medium scale (500 flows)
print("  Medium scale (500 flows)...")
mean_med, std_med = compute_rolling_stats(X_for_scale, 500)

# Coarse scale (2000 flows)
print("  Coarse scale (2000 flows)...")
mean_coarse, std_coarse = compute_rolling_stats(X_for_scale, 2000)

print("Multi-scale features computed.")

# ===========================================================================
# 5. FINAL REPRESENTATION: DIDS-MFL
# ===========================================================================
print("\n"+"="*60)
print("Step 5: Building Final DIDS-MFL Representation")
print("="*60)

X_dids_mfl = np.concatenate([
    X_norm[:, :15],           # (15) Raw normalized features
    X_attack_feat[:, :8],     # (8) Attack-discriminative subspace
    X_normal_feat[:, :7],     # (7) Normal behavior subspace
    X_graph[:, :10],          # (10) Graph-diffused representation
    mean_fine,                 # (10) Fine-scale temporal mean
    std_fine,                  # (10) Fine-scale temporal std
    mean_med,                  # (10) Medium-scale temporal mean
    std_med,                   # (10) Medium-scale temporal std
    mean_coarse,               # (10) Coarse-scale temporal mean
    std_coarse,                # (10) Coarse-scale temporal std
], axis=1)

print(f"DIDS-MFL final feature dimension: {X_dids_mfl.shape[1]}")
np.save(OUTPUTS/'X_dids_mfl.npy', X_dids_mfl)

# Baseline: only raw features
X_baseline = X_norm  # 40 features, no disentanglement

# ===========================================================================
# 6. CLASSIFICATION EXPERIMENTS
# ===========================================================================
print("\n"+"="*60)
print("Step 6: Classification Experiments")
print("="*60)

# Temporal split 80/20
n_total = len(X_dids_mfl)
n_train = int(n_total * 0.8)

X_tr = X_dids_mfl[:n_train]
X_te = X_dids_mfl[n_train:]
X_tr_base = X_baseline[:n_train]
X_te_base = X_baseline[n_train:]
y_bin_tr, y_bin_te = y_binary[:n_train], y_binary[n_train:]
y_mc_tr, y_mc_te = y_mc[:n_train], y_mc[n_train:]

print(f"Train: {n_train} | Test: {n_total-n_train}")
all_results = {}

# ===== A: Binary Classification =====
print("\n--- A: Binary Classification ---")

def run_binary(X_tr, y_tr, X_te, y_te, label=''):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]
    return {
        'accuracy': float(accuracy_score(y_te, y_pred)),
        'f1': float(f1_score(y_te, y_pred)),
        'precision': float(precision_score(y_te, y_pred, zero_division=0)),
        'recall': float(recall_score(y_te, y_pred, zero_division=0)),
        'auc': float(roc_auc_score(y_te, y_prob)),
        'y_pred': y_pred, 'y_prob': y_prob, 'model': clf
    }

res_bin_dids = run_binary(X_tr, y_bin_tr, X_te, y_bin_te, 'DIDS-MFL')
res_bin_base = run_binary(X_tr_base, y_bin_tr, X_te_base, y_bin_te, 'Baseline')

print(f"DIDS-MFL: Acc={res_bin_dids['accuracy']:.4f}, F1={res_bin_dids['f1']:.4f}, AUC={res_bin_dids['auc']:.4f}")
print(f"Baseline: Acc={res_bin_base['accuracy']:.4f}, F1={res_bin_base['f1']:.4f}, AUC={res_bin_base['auc']:.4f}")

all_results['binary'] = {
    'dids_mfl': {k: v for k, v in res_bin_dids.items() if k not in ['y_pred', 'y_prob', 'model']},
    'baseline': {k: v for k, v in res_bin_base.items() if k not in ['y_pred', 'y_prob', 'model']}
}

# ===== B: Multi-class Classification =====
print("\n--- B: Multi-class Classification ---")

def run_multiclass(X_tr, y_tr, X_te, y_te):
    # Filter to classes seen in training
    train_classes = set(np.unique(y_tr))
    test_mask = np.array([y in train_classes for y in y_te])

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te[test_mask])
    y_true = y_te[test_mask]

    return {
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'y_pred': y_pred, 'y_true': y_true, 'model': clf
    }

res_mc_dids = run_multiclass(X_tr, y_mc_tr, X_te, y_mc_te)
res_mc_base = run_multiclass(X_tr_base, y_mc_tr, X_te_base, y_mc_te)

print(f"DIDS-MFL: F1_macro={res_mc_dids['f1_macro']:.4f}, F1_weighted={res_mc_dids['f1_weighted']:.4f}")
print(f"Baseline: F1_macro={res_mc_base['f1_macro']:.4f}, F1_weighted={res_mc_base['f1_weighted']:.4f}")

all_results['multiclass'] = {
    'dids_mfl': {k: v for k, v in res_mc_dids.items() if k not in ['y_pred', 'y_true', 'model']},
    'baseline': {k: v for k, v in res_mc_base.items() if k not in ['y_pred', 'y_true', 'model']}
}

np.save(OUTPUTS/'cm_true.npy', res_mc_dids['y_true'])
np.save(OUTPUTS/'cm_pred.npy', res_mc_dids['y_pred'])

# ===== C: Unknown Attack Detection =====
print("\n--- C: Unknown Attack Detection ---")

# Train on 6/10 classes (5 attack + normal), test on held-out 4 attack types
KNOWN_CLASSES = [0, 1, 2, 3, 4, 6]    # Normal + 5 attack types
UNKNOWN_CLASSES = [5, 7, 8, 9]         # 4 unknown attack types

# Training data: only known classes
known_tr_mask = np.array([y in KNOWN_CLASSES for y in y_mc_tr])
X_unk_tr = X_tr[known_tr_mask]
y_unk_tr_bin = (y_mc_tr[known_tr_mask] != 0).astype(int)

X_unk_tr_b = X_tr_base[known_tr_mask]

def run_unknown(X_tr, y_tr, X_te, y_te_mc, y_te_bin, label=''):
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=20)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    # Overall binary metrics
    f1_all = float(f1_score(y_te_bin, y_pred, zero_division=0))
    auc_all = float(roc_auc_score(y_te_bin, y_prob))

    # Metrics specifically on unknown attack classes vs normal
    unk_mask = np.array([(y in UNKNOWN_CLASSES or y == 0) for y in y_te_mc])
    if unk_mask.sum() > 0:
        y_unk_true = (y_te_mc[unk_mask] != 0).astype(int)
        y_unk_pred = y_pred[unk_mask]
        f1_unk = float(f1_score(y_unk_true, y_unk_pred, zero_division=0))
        det_rate = float(recall_score(y_unk_true, y_unk_pred, zero_division=0))
    else:
        f1_unk = 0.0
        det_rate = 0.0

    return {'f1_all': f1_all, 'auc': auc_all, 'f1_unknown': f1_unk,
            'detection_rate': det_rate, 'model': clf}

res_unk_dids = run_unknown(X_unk_tr, y_unk_tr_bin, X_te, y_mc_te, y_bin_te, 'DIDS-MFL')
res_unk_base = run_unknown(X_unk_tr_b, y_unk_tr_bin, X_te_base, y_mc_te, y_bin_te, 'Baseline')

print(f"DIDS-MFL: F1_all={res_unk_dids['f1_all']:.4f}, AUC={res_unk_dids['auc']:.4f}, F1_unknown={res_unk_dids['f1_unknown']:.4f}")
print(f"Baseline: F1_all={res_unk_base['f1_all']:.4f}, AUC={res_unk_base['auc']:.4f}, F1_unknown={res_unk_base['f1_unknown']:.4f}")

all_results['unknown'] = {
    'dids_mfl': {k: v for k, v in res_unk_dids.items() if k != 'model'},
    'baseline': {k: v for k, v in res_unk_base.items() if k != 'model'}
}

# ===== D: Few-shot Detection =====
print("\n--- D: Few-Shot Attack Detection ---")

def run_few_shot(X_tr, y_mc_tr, X_te, y_mc_te, y_bin_te, n_shots, n_normal=5000):
    attack_classes = [c for c in np.unique(y_mc_tr) if c != 0]
    normal_idx = np.where(y_mc_tr == 0)[0]
    normal_sel = normal_idx[:n_normal]

    train_idx = list(normal_sel)
    for cls in attack_classes:
        cls_idx = np.where(y_mc_tr == cls)[0]
        if len(cls_idx) >= n_shots:
            sel = np.random.choice(cls_idx, n_shots, replace=False)
        elif len(cls_idx) > 0:
            sel = cls_idx
        else:
            continue
        train_idx.extend(sel)

    X_shot = X_tr[train_idx]
    y_shot = (y_mc_tr[train_idx] != 0).astype(int)

    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_shot, y_shot)
    y_pred = clf.predict(X_te)
    y_te_bin = (y_mc_te != 0).astype(int)

    return {
        'f1': float(f1_score(y_te_bin, y_pred, zero_division=0)),
        'accuracy': float(accuracy_score(y_te_bin, y_pred))
    }

shot_values = [1, 5, 10, 20, 50, 100]
fs_dids_results = {}
fs_base_results = {}

for n_shots in shot_values:
    r_d = run_few_shot(X_tr, y_mc_tr, X_te, y_mc_te, y_bin_te, n_shots)
    r_b = run_few_shot(X_tr_base, y_mc_tr, X_te_base, y_mc_te, y_bin_te, n_shots)
    fs_dids_results[n_shots] = r_d
    fs_base_results[n_shots] = r_b
    print(f"  {n_shots:>3}-shot: DIDS-MFL F1={r_d['f1']:.4f}, Baseline F1={r_b['f1']:.4f}")

all_results['few_shot'] = {
    'dids_mfl': {str(k): v for k, v in fs_dids_results.items()},
    'baseline': {str(k): v for k, v in fs_base_results.items()}
}

# ===== E: Per-class F1 Analysis =====
print("\n--- E: Per-class Analysis ---")
# Classification report for DIDS-MFL multi-class
cr = classification_report(
    res_mc_dids['y_true'], res_mc_dids['y_pred'],
    target_names=CLASS_NAMES,
    output_dict=True, zero_division=0
)
print(classification_report(
    res_mc_dids['y_true'], res_mc_dids['y_pred'],
    target_names=CLASS_NAMES, zero_division=0
))

with open(OUTPUTS/'classification_report.json', 'w') as f:
    json.dump(cr, f, indent=2)

# Save all results
with open(OUTPUTS/'all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nAll experiments complete!")
print(f"Results saved to {OUTPUTS/'all_results.json'}")

print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"Binary Classification:")
print(f"  DIDS-MFL: F1={all_results['binary']['dids_mfl']['f1']:.4f}, AUC={all_results['binary']['dids_mfl']['auc']:.4f}")
print(f"  Baseline: F1={all_results['binary']['baseline']['f1']:.4f}, AUC={all_results['binary']['baseline']['auc']:.4f}")
print(f"Multi-class Classification:")
print(f"  DIDS-MFL: F1_macro={all_results['multiclass']['dids_mfl']['f1_macro']:.4f}")
print(f"  Baseline: F1_macro={all_results['multiclass']['baseline']['f1_macro']:.4f}")
print(f"Unknown Attack Detection:")
print(f"  DIDS-MFL: F1_unk={all_results['unknown']['dids_mfl']['f1_unknown']:.4f}")
print(f"  Baseline: F1_unk={all_results['unknown']['baseline']['f1_unknown']:.4f}")
