import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from torch.serialization import add_safe_globals
from torch_geometric.data.temporal import TemporalData
from torch.utils.data import DataLoader, TensorDataset


ATTACK_NAMES = {
    0: 'Analysis',
    1: 'Backdoor',
    2: 'Benign',
    3: 'DoS',
    4: 'Exploits',
    5: 'Fuzzers',
    6: 'Generic',
    7: 'Reconnaissance',
    8: 'Shellcode',
    9: 'Worms',
}


@dataclass
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    attack_test: np.ndarray
    feature_names: list
    meta_test: pd.DataFrame


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs():
    os.makedirs('outputs', exist_ok=True)
    os.makedirs('report/images', exist_ok=True)


def load_data():
    add_safe_globals([TemporalData])
    data = torch.load('data/NF-UNSW-NB15-v2_3d.pt', map_location='cpu', weights_only=False)
    return data


def build_dataframe(data: TemporalData) -> pd.DataFrame:
    msg = data.msg.numpy().astype(np.float32)
    cols = [f'f{i}' for i in range(msg.shape[1])]
    df = pd.DataFrame(msg, columns=cols)
    df['src'] = data.src.numpy().astype(np.int64)
    df['dst'] = data.dst.numpy().astype(np.int64)
    df['t'] = data.t.numpy().astype(np.int64)
    df['label'] = data.label.numpy().astype(np.int64)
    df['attack'] = data.attack.numpy().astype(np.int64)
    return df


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values('t').reset_index(drop=True)
    out['src_degree'] = out.groupby('src')['src'].transform('count')
    out['dst_degree'] = out.groupby('dst')['dst'].transform('count')
    out['src_prev_t'] = out.groupby('src')['t'].shift(1)
    out['dst_prev_t'] = out.groupby('dst')['t'].shift(1)
    out['src_gap'] = (out['t'] - out['src_prev_t']).fillna(0)
    out['dst_gap'] = (out['t'] - out['dst_prev_t']).fillna(0)
    out['pair_count'] = out.groupby(['src', 'dst'])['src'].cumcount() + 1
    out['src_label_rate'] = out.groupby('src')['label'].transform('mean')
    out['dst_label_rate'] = out.groupby('dst')['label'].transform('mean')
    base_features = [c for c in out.columns if c.startswith('f')]
    windows = [3, 10]
    for w in windows:
        rolled = out[base_features].rolling(window=w, min_periods=1).mean().add_prefix(f'roll{w}_')
        out = pd.concat([out, rolled], axis=1)
    for c in ['src_degree', 'dst_degree', 'src_gap', 'dst_gap', 'pair_count']:
        out[c] = np.log1p(out[c].astype(float))
    out = out.drop(columns=['src_prev_t', 'dst_prev_t'])
    return out


def temporal_split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df, task='binary') -> SplitData:
    exclude = {'label', 'attack'}
    feature_names = [c for c in train_df.columns if c not in exclude]
    X_train = train_df[feature_names].to_numpy(dtype=np.float32)
    X_val = val_df[feature_names].to_numpy(dtype=np.float32)
    X_test = test_df[feature_names].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    if task == 'binary':
        y_train = train_df['label'].to_numpy(dtype=np.int64)
        y_val = val_df['label'].to_numpy(dtype=np.int64)
        y_test = test_df['label'].to_numpy(dtype=np.int64)
    else:
        y_train = train_df['attack'].to_numpy(dtype=np.int64)
        y_val = val_df['attack'].to_numpy(dtype=np.int64)
        y_test = test_df['attack'].to_numpy(dtype=np.int64)
    return SplitData(X_train, y_train, X_val, y_val, X_test, y_test, test_df['attack'].to_numpy(dtype=np.int64), feature_names, test_df[['t', 'label', 'attack']].copy())


def save_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def plot_data_overview(df: pd.DataFrame):
    sns.set_theme(style='whitegrid')
    attack_counts = df['attack'].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    sns.barplot(x=[ATTACK_NAMES.get(i, str(i)) for i in attack_counts.index], y=attack_counts.values, color='steelblue')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Count')
    plt.title('Attack class distribution')
    plt.tight_layout()
    plt.savefig('report/images/attack_distribution.png', dpi=200)
    plt.close()

    hourly = df.groupby('t').size().rolling(180, min_periods=1).mean()
    plt.figure(figsize=(10, 4))
    plt.plot(hourly.index, hourly.values, lw=1.2)
    plt.xlabel('Time (s of day)')
    plt.ylabel('Smoothed event count')
    plt.title('Temporal traffic intensity')
    plt.tight_layout()
    plt.savefig('report/images/temporal_intensity.png', dpi=200)
    plt.close()

    sample = df.sample(min(4000, len(df)), random_state=42)
    corr = sample[[c for c in df.columns if c.startswith('f')][:15]].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Feature correlation heatmap (first 15 features)')
    plt.tight_layout()
    plt.savefig('report/images/feature_correlation.png', dpi=200)
    plt.close()


def binary_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, y_prob)),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
    }


def multiclass_metrics(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }


def plot_confusion(y_true, y_pred, labels, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_pr(y_true, y_prob, prefix):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label='ROC')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Binary ROC curve')
    plt.tight_layout()
    plt.savefig(f'report/images/{prefix}_roc.png', dpi=200)
    plt.close()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Binary precision-recall curve')
    plt.tight_layout()
    plt.savefig(f'report/images/{prefix}_pr.png', dpi=200)
    plt.close()


def fit_logreg_binary(split: SplitData):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1, random_state=42)
    clf.fit(split.X_train, split.y_train)
    y_prob = clf.predict_proba(split.X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = binary_metrics(split.y_test, y_pred, y_prob)
    return clf, metrics, y_pred, y_prob


def fit_logreg_multiclass(split: SplitData):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=1, random_state=42)
    clf.fit(split.X_train, split.y_train)
    y_pred = clf.predict(split.X_test)
    metrics = multiclass_metrics(split.y_test, y_pred)
    return clf, metrics, y_pred


class DIDSNet(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=128, disent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.z_stat = nn.Linear(hidden, disent_dim)
        self.z_dyn = nn.Linear(hidden, disent_dim)
        self.z_fuse = nn.Linear(hidden, disent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(disent_dim * 3, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        h = self.enc(x)
        z_stat = self.z_stat(h)
        z_dyn = self.z_dyn(h)
        z_fuse = self.z_fuse(h)
        z = torch.cat([z_stat, z_dyn, z_fuse], dim=1)
        logits = self.classifier(z)
        return logits, z_stat, z_dyn, z_fuse, z


def correlation_penalty(a, b):
    a = a - a.mean(dim=0, keepdim=True)
    b = b - b.mean(dim=0, keepdim=True)
    cov = (a.T @ b) / max(1, a.shape[0] - 1)
    return cov.pow(2).mean()


def contrastive_fusion_loss(z, y, margin=1.0):
    if len(z) > 2048:
        idx = torch.randperm(len(z), device=z.device)[:2048]
        z = z[idx]
        y = y[idx]
    dist = torch.cdist(z, z)
    same = (y[:, None] == y[None, :]).float()
    pos = (dist * same).sum() / (same.sum() + 1e-6)
    neg_mask = 1.0 - same
    neg = (F.relu(margin - dist) * neg_mask).sum() / (neg_mask.sum() + 1e-6)
    return pos + neg


def train_torch_model(split: SplitData, num_classes: int, epochs=12, batch_size=512, lr=1e-3):
    device = torch.device('cpu')
    model = DIDSNet(split.X_train.shape[1], num_classes).to(device)
    counts = np.bincount(split.y_train, minlength=num_classes)
    class_w = counts.sum() / np.maximum(counts, 1)
    class_w = torch.tensor(class_w, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_ds = TensorDataset(torch.tensor(split.X_train, dtype=torch.float32), torch.tensor(split.y_train, dtype=torch.long))
    val_x = torch.tensor(split.X_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(split.y_val, dtype=torch.long, device=device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    best_state = None
    best_val = -1
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits, z1, z2, z3, z = model(xb)
            ce = F.cross_entropy(logits, yb, weight=class_w)
            disent = correlation_penalty(z1, z2) + correlation_penalty(z1, z3) + correlation_penalty(z2, z3)
            fuse = contrastive_fusion_loss(z, yb)
            loss = ce + 0.05 * disent + 0.02 * fuse
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            v_logits, *_ = model(val_x)
            v_pred = v_logits.argmax(dim=1).cpu().numpy()
            if num_classes == 2:
                score = f1_score(split.y_val, v_pred, zero_division=0)
            else:
                score = f1_score(split.y_val, v_pred, average='macro', zero_division=0)
        history.append({'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'val_score': float(score)})
        if score > best_val:
            best_val = score
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def evaluate_torch_binary(model, split: SplitData):
    x = torch.tensor(split.X_test, dtype=torch.float32)
    with torch.no_grad():
        logits, z1, z2, z3, z = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(dim=1).numpy()
    return binary_metrics(split.y_test, pred, prob), pred, prob, z.numpy()


def evaluate_torch_multiclass(model, split: SplitData):
    x = torch.tensor(split.X_test, dtype=torch.float32)
    with torch.no_grad():
        logits, z1, z2, z3, z = model(x)
        pred = logits.argmax(dim=1).numpy()
    return multiclass_metrics(split.y_test, pred), pred, z.numpy()


def open_set_eval(df_features: pd.DataFrame):
    unknown_cls = 9
    df = df_features.copy()
    train_df, val_df, test_df = temporal_split(df)
    train_df = train_df[train_df['attack'] != unknown_cls].copy()
    val_df = val_df[val_df['attack'] != unknown_cls].copy()
    known_mask = test_df['attack'] != unknown_cls
    test_known = test_df[known_mask].copy()
    test_unknown = test_df[~known_mask].copy()
    split = prepare_features(train_df, val_df, pd.concat([test_known, test_unknown], axis=0), task='multiclass')
    model, _ = train_torch_model(split, num_classes=10, epochs=6)
    x = torch.tensor(split.X_train, dtype=torch.float32)
    with torch.no_grad():
        _, _, _, _, z_train = model(x)
    centers = []
    y_train = split.y_train
    for c in sorted(np.unique(y_train)):
        centers.append(z_train[y_train == c].mean(dim=0).numpy())
    centers = np.stack(centers)
    test_x = torch.tensor(split.X_test, dtype=torch.float32)
    with torch.no_grad():
        logits, _, _, _, z_test = model(test_x)
        pred = logits.argmax(dim=1).numpy()
    dists = np.linalg.norm(z_test.numpy()[:, None, :] - centers[None, :, :], axis=2)
    score = dists.min(axis=1)
    threshold = np.percentile(score[: len(test_known)], 95)
    is_unknown_true = np.concatenate([np.zeros(len(test_known)), np.ones(len(test_unknown))]).astype(int)
    is_unknown_pred = (score > threshold).astype(int)
    metrics = {
        'unknown_recall': float(recall_score(is_unknown_true, is_unknown_pred, zero_division=0)),
        'unknown_precision': float(precision_score(is_unknown_true, is_unknown_pred, zero_division=0)),
        'unknown_f1': float(f1_score(is_unknown_true, is_unknown_pred, zero_division=0)),
        'threshold': float(threshold),
        'unknown_support': int(is_unknown_true.sum()),
    }
    return metrics


def few_shot_eval(df_features: pd.DataFrame):
    few_classes = [0, 1, 9]
    df = df_features.copy()
    train_df, val_df, test_df = temporal_split(df)
    sampled_parts = []
    for cls, grp in train_df.groupby('attack'):
        if cls in few_classes:
            sampled_parts.append(grp.sample(min(20, len(grp)), random_state=42))
        else:
            sampled_parts.append(grp)
    train_fs = pd.concat(sampled_parts).sort_values('t')
    split = prepare_features(train_fs, val_df, test_df, task='multiclass')
    lr_clf, lr_metrics, lr_pred = fit_logreg_multiclass(split)
    dids, _ = train_torch_model(split, num_classes=10, epochs=6)
    dids_metrics, dids_pred, _ = evaluate_torch_multiclass(dids, split)
    cls_rows = []
    for cls in few_classes:
        true_mask = split.y_test == cls
        if true_mask.sum() == 0:
            continue
        cls_rows.append({
            'class_id': int(cls),
            'class_name': ATTACK_NAMES.get(cls, str(cls)),
            'support': int(true_mask.sum()),
            'logreg_f1': float(f1_score(split.y_test == cls, lr_pred == cls, zero_division=0)),
            'dids_f1': float(f1_score(split.y_test == cls, dids_pred == cls, zero_division=0)),
        })
    return {'logreg': lr_metrics, 'dids': dids_metrics, 'per_class': cls_rows}


def representation_plot(emb, labels, path):
    n = min(3000, len(emb))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(emb), size=n, replace=False)
    emb_s = emb[idx]
    lab_s = labels[idx]
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(emb_s)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=[ATTACK_NAMES.get(int(i), str(i)) for i in lab_s], s=12, linewidth=0, alpha=0.75, legend=False)
    plt.title('DIDS representation PCA on test flows')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def feature_importance_plot(clf, feature_names, path, topk=15):
    coef = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim == 2 else np.abs(clf.coef_)
    order = np.argsort(coef)[-topk:]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=coef[order], y=np.array(feature_names)[order], color='darkorange')
    plt.title('Top logistic-regression feature importances')
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run(mode='full'):
    ensure_dirs()
    set_seed(42)
    data = load_data()
    df = build_dataframe(data)
    df_feat = add_context_features(df)

    eda_summary = {
        'num_events': int(len(df_feat)),
        'num_features_before_context': int(data.msg.shape[1]),
        'num_features_after_context': int(len([c for c in df_feat.columns if c not in ['label', 'attack']])),
        'binary_counts': {str(i): int(v) for i, v in df_feat['label'].value_counts().sort_index().items()},
        'attack_counts': {ATTACK_NAMES.get(int(i), str(i)): int(v) for i, v in df_feat['attack'].value_counts().sort_index().items()},
        'time_range': [int(df_feat['t'].min()), int(df_feat['t'].max())],
        'nan_features': int(df_feat.isna().sum().sum()),
    }
    save_json('outputs/eda_summary.json', eda_summary)
    plot_data_overview(df_feat)
    if mode == 'eda':
        return

    train_df, val_df, test_df = temporal_split(df_feat)

    # Binary
    split_bin = prepare_features(train_df, val_df, test_df, task='binary')
    lr_bin, lr_bin_metrics, lr_bin_pred, lr_bin_prob = fit_logreg_binary(split_bin)
    dids_bin, bin_hist = train_torch_model(split_bin, num_classes=2, epochs=8)
    dids_bin_metrics, dids_bin_pred, dids_bin_prob, dids_emb = evaluate_torch_binary(dids_bin, split_bin)
    plot_confusion(split_bin.y_test, lr_bin_pred, [0, 1], 'report/images/binary_confusion_logreg.png', 'Binary confusion matrix: Logistic regression')
    plot_confusion(split_bin.y_test, dids_bin_pred, [0, 1], 'report/images/binary_confusion_dids.png', 'Binary confusion matrix: DIDS-inspired')
    plot_roc_pr(split_bin.y_test, dids_bin_prob, 'binary_dids')
    feature_importance_plot(lr_bin, split_bin.feature_names, 'report/images/logreg_feature_importance.png')

    # Multi-class
    split_multi = prepare_features(train_df, val_df, test_df, task='multiclass')
    lr_multi, lr_multi_metrics, lr_multi_pred = fit_logreg_multiclass(split_multi)
    dids_multi, multi_hist = train_torch_model(split_multi, num_classes=10, epochs=8)
    dids_multi_metrics, dids_multi_pred, dids_multi_emb = evaluate_torch_multiclass(dids_multi, split_multi)
    labels = list(range(10))
    plot_confusion(split_multi.y_test, lr_multi_pred, labels, 'report/images/multiclass_confusion_logreg.png', 'Multiclass confusion matrix: Logistic regression')
    plot_confusion(split_multi.y_test, dids_multi_pred, labels, 'report/images/multiclass_confusion_dids.png', 'Multiclass confusion matrix: DIDS-inspired')
    representation_plot(dids_multi_emb, split_multi.y_test, 'report/images/dids_representation_pca.png')

    report_df = pd.DataFrame(classification_report(split_multi.y_test, dids_multi_pred, output_dict=True, zero_division=0)).T
    report_df.to_csv('outputs/dids_multiclass_classification_report.csv')
    pd.DataFrame(bin_hist).to_csv('outputs/dids_binary_history.csv', index=False)
    pd.DataFrame(multi_hist).to_csv('outputs/dids_multiclass_history.csv', index=False)

    # Open-set / unknown
    open_metrics = open_set_eval(df_feat)

    # Few-shot
    few_metrics = few_shot_eval(df_feat)

    summary = {
        'binary': {
            'logreg': lr_bin_metrics,
            'dids_mfl': dids_bin_metrics,
        },
        'multiclass': {
            'logreg': lr_multi_metrics,
            'dids_mfl': dids_multi_metrics,
        },
        'open_set_unknown': open_metrics,
        'few_shot': few_metrics,
    }
    save_json('outputs/metrics_summary.json', summary)

    comparison_rows = [
        {'task': 'binary_f1', 'logreg': lr_bin_metrics['f1'], 'dids_mfl': dids_bin_metrics['f1']},
        {'task': 'binary_pr_auc', 'logreg': lr_bin_metrics['pr_auc'], 'dids_mfl': dids_bin_metrics['pr_auc']},
        {'task': 'multiclass_macro_f1', 'logreg': lr_multi_metrics['macro_f1'], 'dids_mfl': dids_multi_metrics['macro_f1']},
        {'task': 'multiclass_weighted_f1', 'logreg': lr_multi_metrics['weighted_f1'], 'dids_mfl': dids_multi_metrics['weighted_f1']},
    ]
    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv('outputs/model_comparison.csv', index=False)
    plt.figure(figsize=(8, 4))
    comp_m = comp_df.melt(id_vars='task', var_name='model', value_name='score')
    sns.barplot(data=comp_m, x='task', y='score', hue='model')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha='right')
    plt.title('Baseline vs DIDS-inspired comparison')
    plt.tight_layout()
    plt.savefig('report/images/model_comparison.png', dpi=200)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['eda', 'full'], default='full')
    args = parser.parse_args()
    run(args.mode)
