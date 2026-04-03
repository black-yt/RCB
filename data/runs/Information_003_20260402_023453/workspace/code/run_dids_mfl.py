import json
import math
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 20260402
BENIGN_ATTACK_ID = 2
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
N_CLUSTERS = 4
MAX_PCA_COMPONENTS = 4
ICA_COMPONENTS = 8
UNKNOWN_LABEL = -1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dirs() -> None:
    for path in [
        Path("outputs"),
        Path("report/images"),
        Path("outputs/mplconfig"),
    ]:
        path.mkdir(parents=True, exist_ok=True)


def load_temporal_data() -> dict:
    obj = torch.load("data/NF-UNSW-NB15-v2_3d.pt", map_location="cpu", weights_only=False)
    order = np.argsort(obj.t.numpy(), kind="stable")
    data = {
        "src": obj.src.numpy()[order].astype(np.int64),
        "dst": obj.dst.numpy()[order].astype(np.int64),
        "t": obj.t.numpy()[order].astype(np.int64),
        "dt": obj.dt.numpy()[order].astype(np.float32),
        "msg": obj.msg.numpy()[order].astype(np.float32),
        "label": obj.label.numpy()[order].astype(np.int64),
        "attack": obj.attack.numpy()[order].astype(np.int64),
        "num_nodes": int(obj.num_nodes),
    }
    return data


def split_indices(n_rows: int) -> dict:
    n_train = int(n_rows * TRAIN_FRAC)
    n_val = int(n_rows * VAL_FRAC)
    idx = np.arange(n_rows)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def temporal_features(t: np.ndarray, dt: np.ndarray) -> np.ndarray:
    t_scaled = t.astype(np.float32) / 86400.0
    hour_angle = 2.0 * np.pi * t_scaled
    return np.column_stack(
        [
            t_scaled,
            dt.astype(np.float32),
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(2.0 * hour_angle),
            np.cos(2.0 * hour_angle),
        ]
    ).astype(np.float32)


class StatisticalDisentangler:
    def __init__(self, n_clusters: int = N_CLUSTERS):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.bin_discretizer = KBinsDiscretizer(
            n_bins=10, encode="ordinal", strategy="quantile", subsample=50000
        )
        self.feature_groups = []
        self.pcas = []
        self.ica = None
        self.ica_fallback = None

    def _normalized_mi(self, x_binned: np.ndarray) -> np.ndarray:
        n_features = x_binned.shape[1]
        out = np.zeros((n_features, n_features), dtype=np.float32)
        for i in range(n_features):
            xi = x_binned[:, i]
            hx = np.maximum(pd.Series(xi).value_counts(normalize=True).values, 1e-12)
            hxi = -(hx * np.log(hx)).sum()
            for j in range(i, n_features):
                xj = x_binned[:, j]
                hy = np.maximum(pd.Series(xj).value_counts(normalize=True).values, 1e-12)
                hyj = -(hy * np.log(hy)).sum()
                joint = pd.crosstab(xi, xj, normalize=True).to_numpy()
                joint = joint[joint > 0]
                hxy = -(joint * np.log(joint)).sum()
                mi = max(hxi + hyj - hxy, 0.0)
                denom = max(min(hxi, hyj), 1e-6)
                nmi = mi / denom
                out[i, j] = nmi
                out[j, i] = nmi
        np.fill_diagonal(out, 1.0)
        return out

    def fit(self, x_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        x_bins = self.bin_discretizer.fit_transform(x_train).astype(np.int32)

        corr = np.abs(np.corrcoef(x_scaled, rowvar=False))
        np.nan_to_num(corr, copy=False)
        mi = self._normalized_mi(x_bins)
        redundancy = np.clip(0.7 * corr + 0.3 * mi, 0.0, 1.0)
        distance = 1.0 - redundancy
        np.fill_diagonal(distance, 0.0)

        clusterer = AgglomerativeClustering(
            n_clusters=self.n_clusters, metric="precomputed", linkage="average"
        )
        labels = clusterer.fit_predict(distance)

        self.feature_groups = []
        self.pcas = []
        for cluster_id in range(self.n_clusters):
            cols = np.where(labels == cluster_id)[0]
            if len(cols) == 0:
                continue
            x_group = x_scaled[:, cols]
            n_components = min(MAX_PCA_COMPONENTS, len(cols), x_group.shape[0] - 1)
            pca_full = PCA(n_components=n_components, random_state=SEED)
            pca_full.fit(x_group)
            cumulative = np.cumsum(pca_full.explained_variance_ratio_)
            keep = int(np.searchsorted(cumulative, 0.95) + 1)
            keep = max(1, min(keep, n_components))
            pca = PCA(n_components=keep, random_state=SEED)
            pca.fit(x_group)
            self.feature_groups.append(cols)
            self.pcas.append(pca)

        ica_components = min(ICA_COMPONENTS, x_scaled.shape[1] - 1, x_scaled.shape[0] - 1)
        try:
            self.ica = FastICA(
                n_components=ica_components,
                random_state=SEED,
                whiten="unit-variance",
                max_iter=2000,
            )
            self.ica.fit(x_scaled)
        except Exception:
            self.ica = None
            self.ica_fallback = PCA(n_components=ica_components, random_state=SEED)
            self.ica_fallback.fit(x_scaled)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x_scaled = self.scaler.transform(x)
        outputs = []
        for cols, pca in zip(self.feature_groups, self.pcas):
            outputs.append(pca.transform(x_scaled[:, cols]))
        if self.ica is not None:
            outputs.append(self.ica.transform(x_scaled))
        else:
            outputs.append(self.ica_fallback.transform(x_scaled))
        return np.hstack(outputs).astype(np.float32)


class GraphDiffusionFeaturizer:
    def __init__(self, num_nodes: int):
        self.num_nodes = num_nodes
        self.node_count = None
        self.node_degree = None
        self.diff1 = None
        self.diff2 = None
        self.directed_codes = None
        self.directed_counts = None
        self.undirected_codes = None
        self.undirected_counts = None
        self.node_hour_codes = None
        self.node_hour_counts = None

    def _lookup_counts(self, sorted_codes: np.ndarray, counts: np.ndarray, query_codes: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(sorted_codes, query_codes)
        hit = idx < len(sorted_codes)
        out = np.zeros(len(query_codes), dtype=np.float32)
        matched = np.zeros(len(query_codes), dtype=bool)
        matched[hit] = sorted_codes[idx[hit]] == query_codes[hit]
        out[matched] = counts[idx[matched]].astype(np.float32)
        return out

    def fit(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> None:
        endpoint_ids = np.concatenate([src, dst])
        self.node_count = np.bincount(endpoint_ids, minlength=self.num_nodes).astype(np.float32)

        row = np.concatenate([src, dst])
        col = np.concatenate([dst, src])
        data = np.ones(len(row), dtype=np.float32)
        adj = sparse.coo_matrix((data, (row, col)), shape=(self.num_nodes, self.num_nodes)).tocsr()
        self.node_degree = np.array(adj.sum(axis=1)).ravel().astype(np.float32)
        degree_safe = np.maximum(self.node_degree, 1.0)

        activity = np.log1p(self.node_count)
        self.diff1 = adj.dot(activity / degree_safe).astype(np.float32)
        self.diff2 = adj.dot(self.diff1 / degree_safe).astype(np.float32)

        directed_codes = src.astype(np.uint64) * np.uint64(self.num_nodes) + dst.astype(np.uint64)
        self.directed_codes, self.directed_counts = np.unique(directed_codes, return_counts=True)

        u = np.minimum(src, dst).astype(np.uint64)
        v = np.maximum(src, dst).astype(np.uint64)
        undirected_codes = u * np.uint64(self.num_nodes) + v
        self.undirected_codes, self.undirected_counts = np.unique(undirected_codes, return_counts=True)

        hour = (t // 3600).astype(np.uint64)
        node_hour_codes = np.concatenate(
            [
                src.astype(np.uint64) * np.uint64(24) + hour,
                dst.astype(np.uint64) * np.uint64(24) + hour,
            ]
        )
        self.node_hour_codes, self.node_hour_counts = np.unique(node_hour_codes, return_counts=True)

    def transform(self, src: np.ndarray, dst: np.ndarray, t: np.ndarray) -> np.ndarray:
        src = src.astype(np.int64)
        dst = dst.astype(np.int64)
        hour = (t // 3600).astype(np.uint64)

        directed_codes = src.astype(np.uint64) * np.uint64(self.num_nodes) + dst.astype(np.uint64)
        u = np.minimum(src, dst).astype(np.uint64)
        v = np.maximum(src, dst).astype(np.uint64)
        undirected_codes = u * np.uint64(self.num_nodes) + v
        src_hour_codes = src.astype(np.uint64) * np.uint64(24) + hour
        dst_hour_codes = dst.astype(np.uint64) * np.uint64(24) + hour

        src_count = self.node_count[src]
        dst_count = self.node_count[dst]
        src_degree = self.node_degree[src]
        dst_degree = self.node_degree[dst]
        src_diff1 = self.diff1[src]
        dst_diff1 = self.diff1[dst]
        src_diff2 = self.diff2[src]
        dst_diff2 = self.diff2[dst]

        directed_count = self._lookup_counts(self.directed_codes, self.directed_counts, directed_codes)
        undirected_count = self._lookup_counts(self.undirected_codes, self.undirected_counts, undirected_codes)
        src_hour_count = self._lookup_counts(self.node_hour_codes, self.node_hour_counts, src_hour_codes)
        dst_hour_count = self._lookup_counts(self.node_hour_codes, self.node_hour_counts, dst_hour_codes)

        features = np.column_stack(
            [
                np.log1p(src_count),
                np.log1p(dst_count),
                np.log1p(src_degree),
                np.log1p(dst_degree),
                src_diff1,
                dst_diff1,
                src_diff2,
                dst_diff2,
                np.log1p(directed_count),
                np.log1p(undirected_count),
                np.log1p(src_hour_count),
                np.log1p(dst_hour_count),
                np.abs(src_diff1 - dst_diff1),
                np.sqrt((src_count + 1.0) * (dst_count + 1.0)),
            ]
        )
        return features.astype(np.float32)


@dataclass
class FeatureBundle:
    raw_time_train: np.ndarray
    raw_time_val: np.ndarray
    raw_time_test: np.ndarray
    disent_train: np.ndarray
    disent_val: np.ndarray
    disent_test: np.ndarray
    graph_train: np.ndarray
    graph_val: np.ndarray
    graph_test: np.ndarray
    full_train: np.ndarray
    full_val: np.ndarray
    full_test: np.ndarray


def build_feature_bundle(data: dict, idx: dict, train_mask: np.ndarray | None = None) -> FeatureBundle:
    train_idx = idx["train"]
    val_idx = idx["val"]
    test_idx = idx["test"]

    if train_mask is None:
        train_mask = np.ones(len(train_idx), dtype=bool)

    train_fit_idx = train_idx[train_mask]
    msg = data["msg"]
    t = data["t"]
    dt = data["dt"]
    src = data["src"]
    dst = data["dst"]

    time_all = temporal_features(t, dt)
    raw_time = np.hstack([msg, time_all]).astype(np.float32)

    disentangler = StatisticalDisentangler()
    disentangler.fit(msg[train_fit_idx])
    disent_all = disentangler.transform(msg)

    graph_featurizer = GraphDiffusionFeaturizer(num_nodes=data["num_nodes"])
    graph_featurizer.fit(src[train_fit_idx], dst[train_fit_idx], t[train_fit_idx])
    graph_all = graph_featurizer.transform(src, dst, t)

    scaler = StandardScaler()
    full_train_fit = np.hstack([raw_time[train_fit_idx], disent_all[train_fit_idx], graph_all[train_fit_idx]])
    scaler.fit(full_train_fit)

    full_all = scaler.transform(np.hstack([raw_time, disent_all, graph_all])).astype(np.float32)
    disent_scaler = StandardScaler()
    disent_scaler.fit(disent_all[train_fit_idx])
    disent_all = disent_scaler.transform(disent_all).astype(np.float32)

    return FeatureBundle(
        raw_time_train=raw_time[train_idx],
        raw_time_val=raw_time[val_idx],
        raw_time_test=raw_time[test_idx],
        disent_train=disent_all[train_idx],
        disent_val=disent_all[val_idx],
        disent_test=disent_all[test_idx],
        graph_train=graph_all[train_idx],
        graph_val=graph_all[val_idx],
        graph_test=graph_all[test_idx],
        full_train=full_all[train_idx],
        full_val=full_all[val_idx],
        full_test=full_all[test_idx],
    )


def inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    counts = pd.Series(y).value_counts()
    weights = y.astype(np.float64).copy()
    for cls, cnt in counts.items():
        weights[y == cls] = 1.0 / cnt
    weights *= len(y) / weights.sum()
    return weights


def make_hgb_classifier(multiclass: bool = False) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.06,
        max_depth=6,
        max_iter=250,
        min_samples_leaf=40,
        l2_regularization=1e-3,
        early_stopping=True,
        random_state=SEED,
    )


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
    }


def fit_predict_binary(x_train, y_train, x_val, x_test) -> tuple[dict, np.ndarray]:
    clf = make_hgb_classifier(multiclass=False)
    weights = inverse_frequency_weights(y_train)
    clf.fit(x_train, y_train, sample_weight=weights)
    prob_test = clf.predict_proba(x_test)[:, 1]
    prob_val = clf.predict_proba(x_val)[:, 1]
    return {"model": clf, "val_prob": prob_val}, prob_test


def fit_predict_multiclass(x_train, y_train, x_val, x_test) -> tuple[dict, np.ndarray, np.ndarray]:
    clf = make_hgb_classifier(multiclass=True)
    weights = inverse_frequency_weights(y_train)
    clf.fit(x_train, y_train, sample_weight=weights)
    pred_test = clf.predict(x_test)
    prob_test = clf.predict_proba(x_test)
    pred_val = clf.predict(x_val)
    return {"model": clf, "val_pred": pred_val}, pred_test, prob_test


def plot_attack_distribution(data: dict) -> None:
    df = pd.DataFrame({"attack": data["attack"], "label": data["label"]})
    counts = df["attack"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, color="#3B82F6", ax=ax)
    ax.set_title("Attack-ID Distribution")
    ax.set_xlabel("Attack ID")
    ax.set_ylabel("Flow Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig("report/images/attack_distribution.png", dpi=200)
    plt.close(fig)


def plot_hourly_profile(data: dict) -> None:
    hour = data["t"] // 3600
    df = pd.DataFrame({"hour": hour, "label": data["label"]})
    profile = df.groupby(["hour", "label"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=profile, x="hour", y="count", hue="label", marker="o", ax=ax)
    ax.set_title("Hourly Traffic Profile")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Flow Count")
    ax.legend(title="Binary Label", labels=["Benign", "Attack"])
    fig.tight_layout()
    fig.savefig("report/images/hourly_profile.png", dpi=200)
    plt.close(fig)


def plot_embedding_separation(raw_time_test: np.ndarray, full_test: np.ndarray, y_test: np.ndarray) -> None:
    sample = np.arange(len(y_test))
    if len(sample) > 5000:
        rng = np.random.default_rng(SEED)
        sample = np.sort(rng.choice(sample, 5000, replace=False))
    raw_2d = PCA(n_components=2, random_state=SEED).fit_transform(raw_time_test[sample])
    full_2d = PCA(n_components=2, random_state=SEED).fit_transform(full_test[sample])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, emb, title in zip(
        axes,
        [raw_2d, full_2d],
        ["Raw Features Projection", "Disentangled Fusion Projection"],
    ):
        sns.scatterplot(
            x=emb[:, 0],
            y=emb[:, 1],
            hue=y_test[sample],
            palette={0: "#10B981", 1: "#EF4444"},
            alpha=0.5,
            s=12,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].legend(handles=handles, labels=["Benign", "Attack"], title="Label")
    fig.tight_layout()
    fig.savefig("report/images/embedding_separation.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title("Closed-set Multi-class Confusion Matrix")
    ax.set_xlabel("Predicted Attack ID")
    ax.set_ylabel("True Attack ID")
    fig.tight_layout()
    fig.savefig("report/images/multiclass_confusion.png", dpi=200)
    plt.close(fig)


def plot_comparison(ablation_df: pd.DataFrame) -> None:
    melted = ablation_df.melt(id_vars="variant", value_vars=["binary_f1", "binary_roc_auc", "multiclass_macro_f1"])
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.barplot(data=melted, x="variant", y="value", hue="variable", ax=ax)
    ax.set_title("Ablation Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig("report/images/ablation_comparison.png", dpi=200)
    plt.close(fig)


def plot_unknown_results(unknown_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=unknown_df, x="held_out_attack", y="unknown_auroc", color="#F97316", ax=ax)
    ax.set_title("Unknown Attack Detection by Held-out Class")
    ax.set_xlabel("Held-out Attack ID")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig("report/images/unknown_attack_auroc.png", dpi=200)
    plt.close(fig)


def plot_few_shot_results(few_shot_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=few_shot_df, x="shots", y="few_shot_macro_f1", hue="method", marker="o", ax=ax)
    ax.set_title("Few-shot Recognition Performance")
    ax.set_xlabel("Support Shots per Rare Attack")
    ax.set_ylabel("Macro-F1 on Rare Attack Classes")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig("report/images/few_shot_curve.png", dpi=200)
    plt.close(fig)


def cosine_scores(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    p_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-8)
    return x_norm @ p_norm.T


def euclidean_scores(x: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    dists = np.sqrt(((x[:, None, :] - prototypes[None, :, :]) ** 2).sum(axis=2))
    scale = np.median(dists) + 1e-6
    return -dists / scale


def prototype_predict(
    x_query: np.ndarray,
    prototype_blocks: list[np.ndarray],
    class_order: np.ndarray,
    mode: str = "cosine",
) -> np.ndarray:
    scores = np.zeros((x_query.shape[0], len(class_order)), dtype=np.float32)
    if mode == "cosine":
        for x_block, p_block in prototype_blocks:
            scores += cosine_scores(x_block, p_block)
    elif mode == "bi_similarity":
        for x_block, p_block in prototype_blocks:
            scores += 0.5 * cosine_scores(x_block, p_block) + 0.5 * euclidean_scores(x_block, p_block)
    else:
        raise ValueError(mode)
    return class_order[scores.argmax(axis=1)]


def compute_class_prototypes(x: np.ndarray, y: np.ndarray, class_order: np.ndarray) -> np.ndarray:
    return np.vstack([x[y == cls].mean(axis=0) for cls in class_order]).astype(np.float32)


def run_unknown_attack_evaluation(data: dict, idx: dict) -> pd.DataFrame:
    attack_train = data["attack"][idx["train"]]
    attack_val = data["attack"][idx["val"]]
    attack_test = data["attack"][idx["test"]]
    held_out_ids = sorted(int(x) for x in np.unique(data["attack"]) if x != BENIGN_ATTACK_ID)

    rows = []
    for held_out in held_out_ids:
        train_mask = attack_train != held_out
        feature_bundle = build_feature_bundle(data, idx, train_mask=train_mask)

        x_train = feature_bundle.full_train[train_mask]
        y_train = attack_train[train_mask]
        x_val = feature_bundle.full_val[attack_val != held_out]
        y_val = attack_val[attack_val != held_out]
        x_test = feature_bundle.full_test
        y_test = attack_test

        clf = make_hgb_classifier(multiclass=True)
        clf.fit(x_train, y_train, sample_weight=inverse_frequency_weights(y_train))
        val_prob = clf.predict_proba(x_val)
        test_prob = clf.predict_proba(x_test)

        threshold = np.quantile(val_prob.max(axis=1), 0.05)
        known_pred = clf.classes_[test_prob.argmax(axis=1)]
        open_pred = np.where(test_prob.max(axis=1) >= threshold, known_pred, UNKNOWN_LABEL)
        open_true = np.where(y_test == held_out, UNKNOWN_LABEL, y_test)

        unknown_true = (y_test == held_out).astype(int)
        unknown_score = 1.0 - test_prob.max(axis=1)
        rows.append(
            {
                "held_out_attack": held_out,
                "unknown_auroc": float(roc_auc_score(unknown_true, unknown_score)),
                "unknown_ap": float(average_precision_score(unknown_true, unknown_score)),
                "unknown_f1": float(
                    f1_score(unknown_true, (open_pred == UNKNOWN_LABEL).astype(int), zero_division=0)
                ),
                "open_macro_f1": float(f1_score(open_true, open_pred, average="macro", zero_division=0)),
                "known_subset_accuracy": float(
                    accuracy_score(open_true[open_true != UNKNOWN_LABEL], open_pred[open_true != UNKNOWN_LABEL])
                ),
            }
        )
    return pd.DataFrame(rows)


def run_few_shot_evaluation(data: dict, idx: dict, feature_bundle: FeatureBundle) -> pd.DataFrame:
    y_train = data["attack"][idx["train"]]
    y_test = data["attack"][idx["test"]]
    counts = pd.Series(y_train).value_counts().sort_index()
    rare_classes = [int(cls) for cls, cnt in counts.items() if cls != BENIGN_ATTACK_ID and cnt < 1500]
    base_classes = [int(cls) for cls in sorted(counts.index) if cls not in rare_classes]

    raw_scaler = StandardScaler()
    raw_scaler.fit(feature_bundle.raw_time_train)
    raw_train = raw_scaler.transform(feature_bundle.raw_time_train)
    raw_test = raw_scaler.transform(feature_bundle.raw_time_test)
    disent_train = feature_bundle.full_train[:, feature_bundle.raw_time_train.shape[1] : -feature_bundle.graph_train.shape[1]]
    disent_test = feature_bundle.full_test[:, feature_bundle.raw_time_test.shape[1] : -feature_bundle.graph_test.shape[1]]
    graph_scaler = StandardScaler()
    graph_scaler.fit(feature_bundle.graph_train)
    graph_train = graph_scaler.transform(feature_bundle.graph_train)
    graph_test = graph_scaler.transform(feature_bundle.graph_test)

    few_test_mask = np.isin(y_test, rare_classes)
    rows = []
    rng = np.random.default_rng(SEED)
    for shots in [1, 5, 10]:
        usable = True
        for cls in rare_classes:
            cls_idx = np.where(y_train == cls)[0]
            if len(cls_idx) < shots:
                usable = False
                break
        if not usable:
            continue

        prototype_classes = np.array(base_classes + rare_classes, dtype=np.int64)
        true_few = y_test[few_test_mask]
        raw_scores = []
        bi_scores = []

        for _ in range(20):
            support_idx_by_class = {}
            for cls in rare_classes:
                cls_idx = np.where(y_train == cls)[0]
                support_idx_by_class[cls] = rng.choice(cls_idx, size=shots, replace=False)

            raw_proto = []
            disent_proto = []
            graph_proto = []
            for cls in prototype_classes:
                if cls in rare_classes:
                    cls_support = support_idx_by_class[cls]
                    raw_proto.append(raw_train[cls_support].mean(axis=0))
                    disent_proto.append(disent_train[cls_support].mean(axis=0))
                    graph_proto.append(graph_train[cls_support].mean(axis=0))
                else:
                    cls_idx = np.where(y_train == cls)[0]
                    raw_proto.append(raw_train[cls_idx].mean(axis=0))
                    disent_proto.append(disent_train[cls_idx].mean(axis=0))
                    graph_proto.append(graph_train[cls_idx].mean(axis=0))

            raw_proto = np.vstack(raw_proto).astype(np.float32)
            disent_proto = np.vstack(disent_proto).astype(np.float32)
            graph_proto = np.vstack(graph_proto).astype(np.float32)

            pred_raw = prototype_predict(
                raw_test[few_test_mask],
                [(raw_test[few_test_mask], raw_proto)],
                prototype_classes,
                mode="cosine",
            )
            pred_bi = prototype_predict(
                feature_bundle.full_test[few_test_mask],
                [
                    (raw_test[few_test_mask], raw_proto),
                    (disent_test[few_test_mask], disent_proto),
                    (graph_test[few_test_mask], graph_proto),
                ],
                prototype_classes,
                mode="bi_similarity",
            )
            raw_scores.append(f1_score(true_few, pred_raw, average="macro", zero_division=0))
            bi_scores.append(f1_score(true_few, pred_bi, average="macro", zero_division=0))

        rows.append(
            {
                "shots": shots,
                "method": "Raw cosine prototype",
                "few_shot_macro_f1": float(np.mean(raw_scores)),
                "few_shot_macro_f1_std": float(np.std(raw_scores)),
            }
        )
        rows.append(
            {
                "shots": shots,
                "method": "DIDS-MFL bi-similarity",
                "few_shot_macro_f1": float(np.mean(bi_scores)),
                "few_shot_macro_f1_std": float(np.std(bi_scores)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    set_seed(SEED)
    ensure_dirs()
    sns.set_theme(style="whitegrid", palette="colorblind")

    data = load_temporal_data()
    idx = split_indices(len(data["attack"]))
    feature_bundle = build_feature_bundle(data, idx)

    y_train_bin = data["label"][idx["train"]]
    y_val_bin = data["label"][idx["val"]]
    y_test_bin = data["label"][idx["test"]]
    y_train_attack = data["attack"][idx["train"]]
    y_val_attack = data["attack"][idx["val"]]
    y_test_attack = data["attack"][idx["test"]]

    variants = {
        "Raw+Time": (
            feature_bundle.raw_time_train,
            feature_bundle.raw_time_val,
            feature_bundle.raw_time_test,
        ),
        "Raw+Time+Disent": (
            np.hstack([feature_bundle.raw_time_train, feature_bundle.disent_train]),
            np.hstack([feature_bundle.raw_time_val, feature_bundle.disent_val]),
            np.hstack([feature_bundle.raw_time_test, feature_bundle.disent_test]),
        ),
        "Raw+Time+Graph": (
            np.hstack([feature_bundle.raw_time_train, feature_bundle.graph_train]),
            np.hstack([feature_bundle.raw_time_val, feature_bundle.graph_val]),
            np.hstack([feature_bundle.raw_time_test, feature_bundle.graph_test]),
        ),
        "Full DIDS-MFL": (
            feature_bundle.full_train,
            feature_bundle.full_val,
            feature_bundle.full_test,
        ),
    }

    ablation_rows = []
    binary_results = {}
    multiclass_results = {}
    best_multiclass_pred = None
    best_variant = None
    best_macro_f1 = -1.0

    for name, (x_train, x_val, x_test) in variants.items():
        _, prob_test = fit_predict_binary(x_train, y_train_bin, x_val, x_test)
        bin_metrics = binary_metrics(y_test_bin, prob_test)
        binary_results[name] = bin_metrics

        _, pred_test, _ = fit_predict_multiclass(x_train, y_train_attack, x_val, x_test)
        mc_metrics = multiclass_metrics(y_test_attack, pred_test)
        multiclass_results[name] = mc_metrics

        ablation_rows.append(
            {
                "variant": name,
                "binary_f1": bin_metrics["f1"],
                "binary_roc_auc": bin_metrics["roc_auc"],
                "multiclass_macro_f1": mc_metrics["macro_f1"],
            }
        )
        if mc_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = mc_metrics["macro_f1"]
            best_multiclass_pred = pred_test
            best_variant = name

    unknown_df = run_unknown_attack_evaluation(data, idx)
    few_shot_df = run_few_shot_evaluation(data, idx, feature_bundle)
    ablation_df = pd.DataFrame(ablation_rows)

    plot_attack_distribution(data)
    plot_hourly_profile(data)
    plot_embedding_separation(feature_bundle.raw_time_test, feature_bundle.full_test, y_test_bin)
    plot_confusion(y_test_attack, best_multiclass_pred)
    plot_comparison(ablation_df)
    plot_unknown_results(unknown_df)
    plot_few_shot_results(few_shot_df)

    summary = {
        "data_summary": {
            "num_flows": int(len(data["attack"])),
            "num_nodes": int(data["num_nodes"]),
            "num_features": int(data["msg"].shape[1]),
            "train_size": int(len(idx["train"])),
            "val_size": int(len(idx["val"])),
            "test_size": int(len(idx["test"])),
            "binary_counts": pd.Series(data["label"]).value_counts().sort_index().to_dict(),
            "attack_counts": pd.Series(data["attack"]).value_counts().sort_index().to_dict(),
        },
        "binary_results": binary_results,
        "multiclass_results": multiclass_results,
        "best_multiclass_variant": best_variant,
        "unknown_attack_results_mean": unknown_df.mean(numeric_only=True).to_dict(),
        "few_shot_results": few_shot_df.to_dict(orient="records"),
    }

    Path("outputs/summary_metrics.json").write_text(json.dumps(summary, indent=2))
    ablation_df.to_csv("outputs/ablation_metrics.csv", index=False)
    unknown_df.to_csv("outputs/unknown_attack_metrics.csv", index=False)
    few_shot_df.to_csv("outputs/few_shot_metrics.csv", index=False)
    Path("outputs/multiclass_classification_report.txt").write_text(
        classification_report(y_test_attack, best_multiclass_pred, digits=4, zero_division=0)
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
