#!/usr/bin/env python3
import json
import math
import os
import pickle
import random
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

WORKSPACE = Path(__file__).resolve().parents[1]
DATA_PATH = WORKSPACE / 'data' / 'NF-UNSW-NB15-v2_3d.pt'
OUTPUTS = WORKSPACE / 'outputs'
IMAGES = WORKSPACE / 'report' / 'images'
CODE = WORKSPACE / 'code'


@dataclass
class TemporalDataLite:
    src: np.ndarray
    dst: np.ndarray
    t: np.ndarray
    msg: np.ndarray
    src_layer: np.ndarray
    dst_layer: np.ndarray
    dt: np.ndarray
    label: np.ndarray
    attack: np.ndarray


class GlobalStorage:
    pass


class TemporalData:
    pass


class StorageRef:
    def __init__(self, storage_type, key, location, size):
        self.storage_type = storage_type
        self.key = key
        self.location = location
        self.size = size


class TorchPTUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, zip_file, prefix):
        super().__init__(file_obj)
        self.zip_file = zip_file
        self.prefix = prefix

    def persistent_load(self, pid):
        # Expected tuple: ('storage', storage_type, key, location, size)
        if isinstance(pid, tuple) and len(pid) == 5 and pid[0] == 'storage':
            _, storage_type, key, location, size = pid
            return StorageRef(storage_type, key, location, size)
        raise pickle.UnpicklingError(f'Unsupported persistent id: {pid!r}')

    def find_class(self, module, name):
        if module == 'torch_geometric.data.temporal' and name == 'TemporalData':
            return TemporalData
        if module == 'torch_geometric.data.storage' and name == 'GlobalStorage':
            return GlobalStorage
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return self._rebuild_tensor_v2
        if module == 'collections' and name == 'OrderedDict':
            from collections import OrderedDict
            return OrderedDict
        if module == 'torch' and name in {'LongStorage', 'FloatStorage', 'DoubleStorage', 'IntStorage', 'BoolStorage'}:
            return name
        raise pickle.UnpicklingError(f'Unsupported global: {module}.{name}')

    def _storage_dtype(self, storage_type):
        mapping = {
            'LongStorage': np.int64,
            'FloatStorage': np.float32,
            'DoubleStorage': np.float64,
            'IntStorage': np.int32,
            'BoolStorage': np.bool_,
        }
        if storage_type not in mapping:
            raise ValueError(f'Unsupported storage type: {storage_type}')
        return mapping[storage_type]

    def _rebuild_tensor_v2(self, storage_ref, storage_offset, size, stride, requires_grad, backward_hooks):
        dtype = self._storage_dtype(storage_ref.storage_type)
        raw_path = f'{self.prefix}/data/{storage_ref.key}'
        raw = self.zip_file.read(raw_path)
        arr = np.frombuffer(raw, dtype=dtype, count=storage_ref.size)

        size = tuple(int(x) for x in size)
        stride = tuple(int(x) for x in stride)
        storage_offset = int(storage_offset)

        if len(size) == 1:
            out = arr[storage_offset: storage_offset + size[0]].copy()
            return out

        # Handle contiguous tensors directly; otherwise fall back to as_strided.
        expected_contiguous = []
        running = 1
        for dim in reversed(size):
            expected_contiguous.append(running)
            running *= dim
        expected_contiguous = tuple(reversed(expected_contiguous))

        if stride == expected_contiguous:
            count = int(np.prod(size))
            out = arr[storage_offset: storage_offset + count].reshape(size).copy()
            return out

        byte_strides = tuple(s * arr.dtype.itemsize for s in stride)
        base = arr[storage_offset:]
        out = np.lib.stride_tricks.as_strided(base, shape=size, strides=byte_strides).copy()
        return out


def ensure_dirs():
    for p in [OUTPUTS, IMAGES, CODE]:
        p.mkdir(parents=True, exist_ok=True)


def load_temporal_data(path: Path) -> TemporalDataLite:
    try:
        import torch  # type: ignore
        obj = torch.load(path, map_location='cpu')
        return TemporalDataLite(
            src=np.asarray(obj.src.cpu()),
            dst=np.asarray(obj.dst.cpu()),
            t=np.asarray(obj.t.cpu()),
            msg=np.asarray(obj.msg.cpu()),
            src_layer=np.asarray(obj.src_layer.cpu()),
            dst_layer=np.asarray(obj.dst_layer.cpu()),
            dt=np.asarray(obj.dt.cpu()),
            label=np.asarray(obj.label.cpu()),
            attack=np.asarray(obj.attack.cpu()),
        )
    except Exception:
        pass

    with zipfile.ZipFile(path, 'r') as zf:
        names = zf.namelist()
        prefix = sorted({n.split('/')[0] for n in names if '/' in n})[0]
        with zf.open(f'{prefix}/data.pkl', 'r') as fh:
            obj = TorchPTUnpickler(fh, zf, prefix).load()

    mapping = obj._store._mapping
    return TemporalDataLite(
        src=np.asarray(mapping['src']),
        dst=np.asarray(mapping['dst']),
        t=np.asarray(mapping['t']),
        msg=np.asarray(mapping['msg']),
        src_layer=np.asarray(mapping['src_layer']),
        dst_layer=np.asarray(mapping['dst_layer']),
        dt=np.asarray(mapping['dt']),
        label=np.asarray(mapping['label']),
        attack=np.asarray(mapping['attack']),
    )


def prepare_dataframe(td: TemporalDataLite) -> pd.DataFrame:
    msg = np.asarray(td.msg)
    cols = [f'f_{i:02d}' for i in range(msg.shape[1])]
    df = pd.DataFrame(msg, columns=cols)
    df['src'] = td.src.astype(np.int64)
    df['dst'] = td.dst.astype(np.int64)
    df['t'] = td.t.astype(np.int64)
    df['src_layer'] = td.src_layer.astype(np.int64)
    df['dst_layer'] = td.dst_layer.astype(np.int64)
    df['dt'] = td.dt.astype(np.float64)
    df['label'] = td.label.astype(np.int64)
    df['attack'] = td.attack.astype(np.int64)

    # Lightweight dynamic-topological features aligned with the task.
    df['same_node'] = (df['src'] == df['dst']).astype(int)
    df['same_layer'] = (df['src_layer'] == df['dst_layer']).astype(int)
    df['layer_gap'] = np.abs(df['src_layer'] - df['dst_layer'])
    df['node_gap'] = np.abs(df['src'] - df['dst'])
    df['temporal_rank'] = np.linspace(0.0, 1.0, len(df), endpoint=False)

    src_deg = pd.Series(df['src']).value_counts()
    dst_deg = pd.Series(df['dst']).value_counts()
    df['src_out_degree'] = df['src'].map(src_deg).astype(float)
    df['dst_in_degree'] = df['dst'].map(dst_deg).astype(float)
    df['pair_count'] = list(zip(df['src'], df['dst']))
    pair_counts = Counter(df['pair_count'])
    df['pair_count'] = df['pair_count'].map(pair_counts).astype(float)

    for c in ['src_out_degree', 'dst_in_degree', 'pair_count', 'node_gap', 'dt']:
        df[f'log1p_{c}'] = np.log1p(np.abs(df[c].astype(float)))

    return df


def chronological_split(df: pd.DataFrame, train=0.6, val=0.2):
    n = len(df)
    i1 = int(n * train)
    i2 = int(n * (train + val))
    return df.iloc[:i1].copy(), df.iloc[i1:i2].copy(), df.iloc[i2:].copy()


def feature_columns(df: pd.DataFrame):
    exclude = {'label', 'attack'}
    return [c for c in df.columns if c not in exclude and c != 'pair_count']


def macro_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    wf = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'macro_precision': float(p),
        'macro_recall': float(r),
        'macro_f1': float(f),
        'weighted_f1': float(wf),
    }


def train_binary(train_df, test_df, features):
    X_train = train_df[features]
    y_train = train_df['label']
    X_test = test_df[features]
    y_test = test_df['label']

    models = {
        'logreg': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1200, class_weight='balanced', random_state=SEED)),
        ]),
        'rf': Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(
                n_estimators=200,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                n_jobs=-1,
                random_state=SEED,
            )),
        ]),
    }

    results = {}
    best_name, best_f1, best_pred = None, -1, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = macro_metrics(y_test, pred)
        results[name] = metrics
        if metrics['macro_f1'] > best_f1:
            best_name, best_f1, best_pred = name, metrics['macro_f1'], pred
    return results, best_name, y_test.to_numpy(), np.asarray(best_pred)


def train_multiclass(train_df, test_df, features):
    train_df = train_df[train_df['attack'] != 0].copy()
    test_df = test_df[test_df['attack'] != 0].copy()
    X_train = train_df[features]
    y_train = train_df['attack']
    X_test = test_df[features]
    y_test = test_df['attack']

    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1500, class_weight='balanced', random_state=SEED, multi_class='auto')),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return macro_metrics(y_test, pred), y_test.to_numpy(), np.asarray(pred), model


def unknown_attack_protocol(train_df, test_df, features):
    attacks = sorted(a for a in train_df['attack'].unique() if a != 0)
    rows = []
    for held_out in attacks:
        local_train = train_df[train_df['attack'] != held_out].copy()
        local_test = test_df[(test_df['attack'] == held_out) | (test_df['label'] == 0)].copy()
        if local_test.empty or local_train['label'].nunique() < 2:
            continue
        model = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1200, class_weight='balanced', random_state=SEED)),
        ])
        model.fit(local_train[features], local_train['label'])
        pred = model.predict(local_test[features])
        metrics = macro_metrics(local_test['label'], pred)
        rows.append({
            'held_out_attack': int(held_out),
            'n_test': int(len(local_test)),
            **metrics,
        })
    return pd.DataFrame(rows).sort_values('macro_f1', ascending=False)


def few_shot_protocol(train_df, test_df, features, shots=(1, 5, 10)):
    attack_train = train_df[train_df['attack'] != 0].copy()
    attack_test = test_df[test_df['attack'] != 0].copy()
    scaler = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    X_train = scaler.fit_transform(attack_train[features])
    X_test = scaler.transform(attack_test[features])
    y_train = attack_train['attack'].to_numpy()
    y_test = attack_test['attack'].to_numpy()

    results = []
    for k in shots:
        centroids = {}
        for cls in sorted(np.unique(y_train)):
            idx = np.where(y_train == cls)[0]
            if len(idx) == 0:
                continue
            pick = idx[: min(k, len(idx))]
            centroids[int(cls)] = X_train[pick].mean(axis=0)
        classes = sorted(centroids)
        if not classes:
            continue
        centroid_matrix = np.vstack([centroids[c] for c in classes])
        dists = ((X_test[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
        pred = np.array([classes[i] for i in np.argmin(dists, axis=1)])
        results.append({'shots': int(k), **macro_metrics(y_test, pred)})
    return pd.DataFrame(results)


def save_json(path: Path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def plot_data_overview(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.countplot(x='label', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Binary label distribution')
    axes[0, 0].set_xlabel('Label (0=benign, 1=attack)')

    attack_counts = df['attack'].value_counts().sort_values(ascending=False).head(12)
    sns.barplot(x=attack_counts.index.astype(str), y=attack_counts.values, ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_title('Top attack classes')
    axes[0, 1].set_xlabel('Attack id')
    axes[0, 1].tick_params(axis='x', rotation=45)

    tvals = df['t'].astype(float)
    axes[1, 0].plot(np.linspace(0, 1, len(tvals)), np.sort(tvals), color='darkorange', lw=1)
    axes[1, 0].set_title('Temporal progression of events')
    axes[1, 0].set_xlabel('Normalized event index')
    axes[1, 0].set_ylabel('Timestamp')

    sns.histplot(np.log1p(df['pair_count']), bins=40, ax=axes[1, 1], color='seagreen')
    axes[1, 1].set_title('Repeated src-dst interaction intensity')
    axes[1, 1].set_xlabel('log1p(pair count)')

    plt.tight_layout()
    fig.savefig(IMAGES / 'data_overview.png', dpi=180)
    plt.close(fig)


def plot_pca_embedding(df: pd.DataFrame, features):
    sample = df.sample(n=min(5000, len(df)), random_state=SEED).copy()
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    X = pipe.fit_transform(sample[features])
    emb = PCA(n_components=2, random_state=SEED).fit_transform(X)
    sample['pc1'] = emb[:, 0]
    sample['pc2'] = emb[:, 1]
    sample['attack_or_benign'] = np.where(sample['label'] == 0, 'benign', 'attack')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=sample, x='pc1', y='pc2', hue='attack_or_benign', s=10, alpha=0.55, ax=axes[0])
    axes[0].set_title('PCA embedding: benign vs attack')

    subset = sample[sample['attack'] != 0].copy()
    top_attacks = subset['attack'].value_counts().head(6).index
    subset['attack_group'] = subset['attack'].where(subset['attack'].isin(top_attacks), -1).astype(int).astype(str)
    sns.scatterplot(data=subset, x='pc1', y='pc2', hue='attack_group', s=10, alpha=0.6, ax=axes[1])
    axes[1].set_title('PCA embedding: top attack groups')

    plt.tight_layout()
    fig.savefig(IMAGES / 'feature_embedding.png', dpi=180)
    plt.close(fig)


def plot_binary_results(results):
    rows = []
    for model, metrics in results.items():
        for k, v in metrics.items():
            rows.append({'model': model, 'metric': k, 'value': v})
    rdf = pd.DataFrame(rows)
    order = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1']
    rdf['metric'] = pd.Categorical(rdf['metric'], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=rdf, x='metric', y='value', hue='model', ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title('Binary intrusion detection performance')
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    fig.savefig(IMAGES / 'binary_results.png', dpi=180)
    plt.close(fig)


def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    fig.savefig(IMAGES / filename, dpi=180)
    plt.close(fig)


def plot_unknown_results(df):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    tmp = df.sort_values('macro_f1', ascending=False)
    sns.barplot(data=tmp, x='held_out_attack', y='macro_f1', ax=ax, color='indianred')
    ax.set_ylim(0, 1)
    ax.set_title('Unknown-attack detection by held-out attack class')
    ax.set_xlabel('Held-out attack id')
    plt.tight_layout()
    fig.savefig(IMAGES / 'unknown_attack_results.png', dpi=180)
    plt.close(fig)


def plot_few_shot_results(df):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in ['accuracy', 'macro_f1', 'weighted_f1']:
        ax.plot(df['shots'], df[metric], marker='o', label=metric)
    ax.set_ylim(0, 1)
    ax.set_xticks(df['shots'])
    ax.set_title('Few-shot attack classification performance')
    ax.set_xlabel('Shots per class')
    ax.set_ylabel('Score')
    ax.legend()
    plt.tight_layout()
    fig.savefig(IMAGES / 'few_shot_results.png', dpi=180)
    plt.close(fig)


def main():
    ensure_dirs()
    sns.set_theme(style='whitegrid')

    td = load_temporal_data(DATA_PATH)
    df = prepare_dataframe(td)
    train_df, val_df, test_df = chronological_split(df)
    trainval_df = pd.concat([train_df, val_df], axis=0).reset_index(drop=True)
    feats = feature_columns(df)

    dataset_summary = {
        'n_events': int(len(df)),
        'n_features_msg': int(td.msg.shape[1]),
        'n_unique_src': int(df['src'].nunique()),
        'n_unique_dst': int(df['dst'].nunique()),
        'label_distribution': {str(k): int(v) for k, v in df['label'].value_counts().sort_index().items()},
        'attack_distribution_top10': {str(k): int(v) for k, v in df['attack'].value_counts().head(10).items()},
        'time_range': [int(df['t'].min()), int(df['t'].max())],
        'feature_columns_used': feats,
        'split_sizes': {
            'train': int(len(train_df)),
            'val': int(len(val_df)),
            'test': int(len(test_df)),
        },
    }
    save_json(OUTPUTS / 'dataset_summary.json', dataset_summary)

    # Binary detection over train+val -> test.
    binary_results, best_binary_name, yb_true, yb_pred = train_binary(trainval_df, test_df, feats)
    save_json(OUTPUTS / 'binary_metrics.json', binary_results)
    pd.DataFrame({'y_true': yb_true, 'y_pred': yb_pred}).to_csv(OUTPUTS / 'binary_predictions.csv', index=False)

    # Multi-class attack typing.
    multiclass_metrics, ym_true, ym_pred, _ = train_multiclass(trainval_df, test_df, feats)
    save_json(OUTPUTS / 'multiclass_metrics.json', multiclass_metrics)
    pd.DataFrame({'y_true': ym_true, 'y_pred': ym_pred}).to_csv(OUTPUTS / 'multiclass_predictions.csv', index=False)
    with open(OUTPUTS / 'multiclass_report.txt', 'w', encoding='utf-8') as f:
        f.write(classification_report(ym_true, ym_pred, zero_division=0))

    # Unknown and few-shot protocols.
    unknown_df = unknown_attack_protocol(trainval_df, test_df, feats)
    unknown_df.to_csv(OUTPUTS / 'unknown_attack_results.csv', index=False)

    fewshot_df = few_shot_protocol(trainval_df, test_df, feats)
    fewshot_df.to_csv(OUTPUTS / 'few_shot_results.csv', index=False)

    # Figures.
    plot_data_overview(df)
    plot_pca_embedding(df, feats)
    plot_binary_results(binary_results)
    plot_confusion(yb_true, yb_pred, f'Binary confusion matrix ({best_binary_name})', 'binary_confusion.png')
    plot_confusion(ym_true, ym_pred, 'Multi-class attack confusion matrix', 'multiclass_confusion.png')
    plot_unknown_results(unknown_df)
    plot_few_shot_results(fewshot_df)

    analysis_summary = {
        'binary_best_model': best_binary_name,
        'binary_best_macro_f1': float(binary_results[best_binary_name]['macro_f1']),
        'multiclass_macro_f1': float(multiclass_metrics['macro_f1']),
        'unknown_attack_macro_f1_mean': float(unknown_df['macro_f1'].mean()) if not unknown_df.empty else None,
        'few_shot_macro_f1': {
            str(int(r['shots'])): float(r['macro_f1']) for _, r in fewshot_df.iterrows()
        },
        'generated_figures': [
            'report/images/data_overview.png',
            'report/images/feature_embedding.png',
            'report/images/binary_results.png',
            'report/images/binary_confusion.png',
            'report/images/multiclass_confusion.png',
            'report/images/unknown_attack_results.png',
            'report/images/few_shot_results.png',
        ],
    }
    save_json(OUTPUTS / 'analysis_summary.json', analysis_summary)

    readme = f'''This directory contains task-specific analysis outputs generated by code/run_analysis.py.

Files:
- dataset_summary.json: dataset shape, split sizes, and label distributions.
- binary_metrics.json / binary_predictions.csv: benign-vs-attack results.
- multiclass_metrics.json / multiclass_predictions.csv / multiclass_report.txt: attack-type classification results.
- unknown_attack_results.csv: leave-one-attack-out unknown-attack binary detection benchmark.
- few_shot_results.csv: prototype-based few-shot attack classification benchmark.
- analysis_summary.json: compact summary of the main findings and figure paths.
'''
    (OUTPUTS / 'README.txt').write_text(readme, encoding='utf-8')

    print('Analysis completed.')
    print(json.dumps(analysis_summary, indent=2))


if __name__ == '__main__':
    main()
