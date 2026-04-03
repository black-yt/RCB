"""
Feature importance, threshold analysis, score distribution, and additional plots.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, precision_recall_curve)

DATA_DIR = Path('../data')
OUT_DIR  = Path('../outputs')
IMG_DIR  = Path('../report/images')

train = pd.read_csv(DATA_DIR / 'train_simulated.csv')
test  = pd.read_csv(DATA_DIR / 'test_simulated.csv')
feat_cols = [str(i) for i in range(20)]
feat_labels = [f'F{i}' for i in range(20)]

X_train = train[feat_cols].values
y_train = train['label'].values.astype(int)
X_test  = test[feat_cols].values
y_test  = test['label'].values.astype(int)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Retrain MLP (best model) ──────────────────────────────────────────────
print("Re-training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu',
                    alpha=1e-3, max_iter=200, random_state=42, early_stopping=True)
mlp.fit(X_train_s, y_train)
mlp_proba = mlp.predict_proba(X_test_s)[:, 1]

# ── Retrain RF for feature importance ────────────────────────────────────
print("Re-training RF for feature importance...")
rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced',
                             n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]

# ── Permutation importance (on MLP, small subset for speed) ──────────────
print("Computing permutation importance on MLP...")
idx_sub = np.random.RandomState(42).choice(len(X_test_s), 5000, replace=False)
perm_result = permutation_importance(
    mlp, X_test_s[idx_sub], y_test[idx_sub],
    n_repeats=10, random_state=42, scoring='roc_auc', n_jobs=-1)
perm_means = perm_result.importances_mean
perm_stds  = perm_result.importances_std

# ── Figure 9: Feature importance (RF Gini + MLP permutation) ─────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

order_rf   = np.argsort(rf.feature_importances_)[::-1]
order_perm = np.argsort(perm_means)[::-1]

axes[0].bar(range(20), rf.feature_importances_[order_rf],
            color=sns.color_palette('viridis', 20))
axes[0].set_xticks(range(20))
axes[0].set_xticklabels([feat_labels[i] for i in order_rf], rotation=45, ha='right')
axes[0].set_ylabel('Gini Importance')
axes[0].set_title('Random Forest: Gini Feature Importance')
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(range(20), perm_means[order_perm],
            yerr=perm_stds[order_perm],
            color=sns.color_palette('plasma', 20), capsize=3)
axes[1].set_xticks(range(20))
axes[1].set_xticklabels([feat_labels[i] for i in order_perm], rotation=45, ha='right')
axes[1].set_ylabel('AUROC decrease')
axes[1].set_title('MLP: Permutation Feature Importance')
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Feature Importance Analysis', fontsize=14)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig9_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig9")

# ── Figure 10: Score distributions ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for lbl, clr, nm in [(0, '#4C72B0', 'No Merge'), (1, '#DD8452', 'Merge')]:
    vals = mlp_proba[y_test == lbl]
    axes[0].hist(vals, bins=100, alpha=0.6, color=clr, label=nm, density=True)

axes[0].set_xlabel('Predicted merge probability')
axes[0].set_ylabel('Density')
axes[0].set_title('MLP Score Distribution by Class')
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_yscale('log')

# Threshold vs F1 / Precision / Recall
thresholds = np.linspace(0.01, 0.99, 200)
prec_arr, rec_arr, f1_arr = [], [], []
for t in thresholds:
    pred_t = (mlp_proba >= t).astype(int)
    from sklearn.metrics import precision_score, recall_score
    prec_arr.append(precision_score(y_test, pred_t, zero_division=0))
    rec_arr.append(recall_score(y_test, pred_t, zero_division=0))
    f1_arr.append(f1_score(y_test, pred_t, zero_division=0))

axes[1].plot(thresholds, prec_arr, label='Precision', color='#4C72B0')
axes[1].plot(thresholds, rec_arr,  label='Recall',    color='#DD8452')
axes[1].plot(thresholds, f1_arr,   label='F1',        color='#55A868', lw=2)
best_t = thresholds[np.argmax(f1_arr)]
axes[1].axvline(best_t, color='gray', ls='--', lw=1, label=f'Best t={best_t:.3f}')
axes[1].set_xlabel('Decision threshold')
axes[1].set_ylabel('Score')
axes[1].set_title('MLP: Threshold vs. Metrics')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig10_score_distribution_threshold.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig10")

# ── Figure 11: Learning curve (MLP loss curve proxy via partial_fit) ──────
# Use training loss from MLPClassifier
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(mlp.loss_curve_, color='#4C72B0', lw=2, label='Training loss')
if hasattr(mlp, 'validation_fraction') and mlp.early_stopping:
    # validation scores are stored in best_validation_score_; we can't get curve
    pass
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss (cross-entropy)')
ax.set_title('MLP Training Loss Curve')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig11_mlp_loss_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11")

# ── Figure 12: Per-degradation score distributions ───────────────────────
degs = test['degradation'].unique()
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, deg in zip(axes, sorted(degs)):
    mask = test['degradation'].values == deg
    proba_d = mlp_proba[mask]
    y_d     = y_test[mask]
    for lbl, clr, nm in [(0, '#4C72B0', 'No Merge'), (1, '#DD8452', 'Merge')]:
        ax.hist(proba_d[y_d == lbl], bins=60, alpha=0.6, color=clr,
                label=nm, density=True)
    auroc_d = roc_auc_score(y_d, proba_d)
    ax.set_title(f'{deg}  (AUROC={auroc_d:.4f})', fontsize=10)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density (log)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('MLP Score Distributions by Degradation Type', fontsize=13)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig12_degradation_score_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig12")

# ── Figure 13: t-SNE of feature space (sampled) ───────────────────────────
print("Running t-SNE...")
from sklearn.manifold import TSNE
idx_tsne = np.random.RandomState(0).choice(len(X_test_s), 3000, replace=False)
Xs = X_test_s[idx_tsne]
ys = y_test[idx_tsne]
Xt2d = TSNE(n_components=2, perplexity=30, random_state=42,
            n_jobs=-1).fit_transform(Xs)
fig, ax = plt.subplots(figsize=(8, 6))
for lbl, clr, nm in [(0, '#4C72B0', 'No Merge'), (1, '#DD8452', 'Merge')]:
    mask = ys == lbl
    ax.scatter(Xt2d[mask, 0], Xt2d[mask, 1], c=clr, s=4, alpha=0.5, label=nm)
ax.set_title('t-SNE Embedding of Feature Space (test set, n=3000)')
ax.legend(markerscale=3)
ax.set_xlabel('t-SNE dim 1')
ax.set_ylabel('t-SNE dim 2')
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig13_tsne.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig13")

# ── Save permutation importances ─────────────────────────────────────────
perm_df = pd.DataFrame({'feature': feat_labels, 'importance_mean': perm_means,
                         'importance_std': perm_stds})
perm_df.to_csv(OUT_DIR / 'permutation_importance.csv', index=False)
print("All done.")
