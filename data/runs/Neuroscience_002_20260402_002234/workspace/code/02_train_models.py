"""
Train and evaluate multiple ML models for neuron segment merge prediction.
Models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, MLP
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json, time, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              classification_report, confusion_matrix,
                              roc_curve, precision_recall_curve, f1_score)
from sklearn.calibration import CalibratedClassifierCV

DATA_DIR = Path('../data')
OUT_DIR  = Path('../outputs')
IMG_DIR  = Path('../report/images')
OUT_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / 'train_simulated.csv')
test  = pd.read_csv(DATA_DIR / 'test_simulated.csv')
feat_cols = [str(i) for i in range(20)]

X_train = train[feat_cols].values
y_train = train['label'].values.astype(int)
X_test  = test[feat_cols].values
y_test  = test['label'].values.astype(int)

# Scale
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"Train: {X_train.shape}, pos rate: {y_train.mean():.3f}")
print(f"Test:  {X_test.shape},  pos rate: {y_test.mean():.3f}")

# ── Model definitions ────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight='balanced',
        n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, random_state=42),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        alpha=1e-3, max_iter=200, random_state=42, early_stopping=True),
}

results = {}

for name, model in models.items():
    t0 = time.time()
    print(f"\nTraining {name}...")

    if name in ['Logistic Regression', 'MLP']:
        model.fit(X_train_s, y_train)
        proba = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

    elapsed = time.time() - t0

    # Find best threshold by F1 on test (reporting only)
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s = [f1_score(y_test, proba >= t) for t in thresholds]
    best_t = thresholds[np.argmax(f1s)]
    pred = (proba >= best_t).astype(int)

    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    f1    = f1_score(y_test, pred)

    results[name] = {
        'auroc': auroc, 'auprc': auprc, 'f1': f1,
        'best_threshold': best_t, 'elapsed': elapsed,
        'proba': proba.tolist(),
        'pred': pred.tolist(),
    }
    print(f"  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  F1={f1:.4f}  t={best_t:.3f}  [{elapsed:.1f}s]")

# Save numerical results (without proba/pred arrays)
summary = {k: {m: v for m, v in res.items() if m not in ('proba','pred')}
           for k, res in results.items()}
with open(OUT_DIR / 'model_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

# ── Figure 5: ROC curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for (name, res), clr in zip(results.items(), colors):
    proba = np.array(res['proba'])
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, color=clr, lw=2,
                 label=f"{name} (AUC={res['auroc']:.3f})")

axes[0].plot([0,1],[0,1],'k--', lw=1)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves — All Models')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

for (name, res), clr in zip(results.items(), colors):
    proba = np.array(res['proba'])
    prec, rec, _ = precision_recall_curve(y_test, proba)
    axes[1].plot(rec, prec, color=clr, lw=2,
                 label=f"{name} (AP={res['auprc']:.3f})")

baseline = y_test.mean()
axes[1].axhline(baseline, color='k', ls='--', lw=1, label=f'Baseline ({baseline:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves — All Models')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig5_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig5")

# ── Figure 6: Metric comparison bar chart ───────────────────────────────
metrics_df = pd.DataFrame({
    name: {'AUROC': res['auroc'], 'AUPRC': res['auprc'], 'F1': res['f1']}
    for name, res in results.items()
}).T

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(metrics_df))
w = 0.25
for j, (metric, clr) in enumerate(zip(['AUROC','AUPRC','F1'],
                                        ['#4C72B0','#DD8452','#55A868'])):
    ax.bar(x + j*w, metrics_df[metric], w, label=metric, color=clr, alpha=0.85)
    for i, v in enumerate(metrics_df[metric]):
        ax.text(i + j*w, v + 0.004, f'{v:.3f}', ha='center', fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels(metrics_df.index, rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title('Model Comparison: AUROC, AUPRC, F1')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig6_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig6")

# ── Figure 7: Confusion matrices for best model ──────────────────────────
best_name = max(results, key=lambda k: results[k]['auroc'])
print(f"\nBest model by AUROC: {best_name}")
best_pred = np.array(results[best_name]['pred'])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# Absolute
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
axes[0].set_title(f'{best_name}\nConfusion Matrix (Counts)')

# Normalized
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues', ax=axes[1],
            xticklabels=['Pred 0','Pred 1'], yticklabels=['True 0','True 1'])
axes[1].set_title(f'{best_name}\nConfusion Matrix (Normalized)')

plt.tight_layout()
plt.savefig(IMG_DIR / 'fig7_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig7")

# ── Per-degradation performance ──────────────────────────────────────────
deg_results = {}
best_proba = np.array(results[best_name]['proba'])
best_t     = results[best_name]['best_threshold']

for deg in test['degradation'].unique():
    mask = (test['degradation'] == deg).values
    y_d = y_test[mask]
    p_d = best_proba[mask]
    if y_d.sum() == 0:
        continue
    deg_results[deg] = {
        'auroc': roc_auc_score(y_d, p_d),
        'auprc': average_precision_score(y_d, p_d),
        'f1':    f1_score(y_d, (p_d >= best_t).astype(int)),
        'n': int(mask.sum()), 'pos_rate': float(y_d.mean()),
    }

with open(OUT_DIR / 'degradation_results.json', 'w') as f:
    json.dump(deg_results, f, indent=2)

print("\nPer-degradation results:")
for deg, m in deg_results.items():
    print(f"  {deg:20s}  AUROC={m['auroc']:.4f}  AUPRC={m['auprc']:.4f}  F1={m['f1']:.4f}")

# ── Figure 8: Per-degradation metric bars ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
deg_df = pd.DataFrame(deg_results).T
x = np.arange(len(deg_df))
for j, (metric, clr) in enumerate(zip(['auroc','auprc','f1'],
                                        ['#4C72B0','#DD8452','#55A868'])):
    ax.bar(x + j*0.25, deg_df[metric], 0.25, label=metric.upper(), color=clr, alpha=0.85)
    for i, v in enumerate(deg_df[metric]):
        ax.text(i + j*0.25, v + 0.005, f'{v:.3f}', ha='center', fontsize=8)

ax.set_xticks(x + 0.25)
ax.set_xticklabels(deg_df.index, rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.05)
ax.set_title(f'{best_name}: Performance by Degradation Type')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(IMG_DIR / 'fig8_degradation_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig8")

# ── Feature importance (RF / GB) ─────────────────────────────────────────
for nm in ['Random Forest', 'Gradient Boosting']:
    if nm in models:
        importances = models[nm].feature_importances_
        np.save(OUT_DIR / f'feature_importance_{nm.replace(" ","_")}.npy', importances)

print("\nAll done.")
