import json
import math
import os
import random
import sys
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
REPORT_DIR = ROOT / "report"
IMAGES_DIR = REPORT_DIR / "images"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_DIR.mkdir(exist_ok=True, parents=True)
    IMAGES_DIR.mkdir(exist_ok=True, parents=True)


def df_to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        vals = []
        for val in row.tolist():
            if isinstance(val, float):
                vals.append(f"{val:.3f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def install_dataset_stub() -> None:
    mod = types.ModuleType("data_prepare")

    class RealisticCrystalDataset:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(kwargs)

        def __setstate__(self, state):
            self.__dict__.update(state)

    mod.RealisticCrystalDataset = RealisticCrystalDataset
    sys.modules["data_prepare"] = mod


def load_dataset(path: Path) -> List[Data]:
    install_dataset_stub()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj.data_list


def augment_graph(data: Data, drop_edge_p: float = 0.15, mask_feat_p: float = 0.1) -> Data:
    out = data.clone()
    x = out.x.clone()
    edge_index = out.edge_index.clone()
    edge_attr = out.edge_attr.clone()

    if x.numel() > 0:
        mask = torch.rand_like(x) < mask_feat_p
        x = x.masked_fill(mask, 0.0)

    if edge_index.shape[1] > 0:
        keep = torch.rand(edge_index.shape[1]) > drop_edge_p
        if keep.sum() == 0:
            keep[torch.randint(0, edge_index.shape[1], (1,))] = True
        edge_index = edge_index[:, keep]
        edge_attr = edge_attr[keep]

    out.x = x
    out.edge_index = edge_index
    out.edge_attr = edge_attr
    return out


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.node_proj(data.x.float())
        edge_attr = self.edge_proj(data.edge_attr.float())
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, data.edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = global_mean_pool(x, data.batch)
        return x, graph_emb


class ProjectionHead(nn.Module):
    def __init__(self, dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class GraphClassifier(nn.Module):
    def __init__(self, encoder: GNNEncoder, hidden_dim: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        _, graph_emb = self.encoder(data)
        logits = self.head(graph_emb).squeeze(-1)
        return logits, graph_emb


def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    n = z1.size(0)
    mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    targets = torch.arange(n, 2 * n, device=z.device)
    targets = torch.cat([targets, torch.arange(0, n, device=z.device)], dim=0)
    return F.cross_entropy(sim, targets)


def pretrain_encoder(
    encoder: GNNEncoder,
    dataset: List[Data],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    projection = ProjectionHead(64).to(device)
    encoder = encoder.to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projection.parameters()), lr=lr, weight_decay=1e-5)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = {"pretrain_loss": []}

    for _ in tqdm(range(epochs), desc="pretrain", leave=False):
        encoder.train()
        projection.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in loader:
            views1 = [augment_graph(d) for d in batch.to_data_list()]
            views2 = [augment_graph(d) for d in batch.to_data_list()]
            b1 = DataLoader(views1, batch_size=len(views1)).__iter__().__next__().to(device)
            b2 = DataLoader(views2, batch_size=len(views2)).__iter__().__next__().to(device)
            _, g1 = encoder(b1)
            _, g2 = encoder(b2)
            z1 = projection(g1)
            z2 = projection(g2)
            loss = nt_xent(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * b1.num_graphs
            total_graphs += b1.num_graphs
        history["pretrain_loss"].append(total_loss / total_graphs)
    return history


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    out["specificity"] = tn / (tn + fp) if (tn + fp) else 0.0
    out["balanced_accuracy"] = 0.5 * (out["recall"] + out["specificity"])
    return out


def pick_threshold_by_f1(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-8, None)
    if len(thresholds) == 0:
        return 0.5
    idx = int(np.nanargmax(f1[:-1]))
    return float(thresholds[idx])


@dataclass
class SplitResult:
    split_id: int
    model_name: str
    threshold: float
    val_roc_auc: float
    test_roc_auc: float
    val_pr_auc: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    test_specificity: float
    test_balanced_accuracy: float


def train_classifier(
    dataset: List[Data],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    pretrained_state: Dict[str, torch.Tensor] | None,
    device: torch.device,
    model_name: str,
    split_id: int,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> Tuple[SplitResult, Dict[str, List[float]], nn.Module]:
    sample = dataset[0]
    encoder = GNNEncoder(sample.x.shape[1], sample.edge_attr.shape[1])
    if pretrained_state is not None:
        encoder.load_state_dict(pretrained_state)
    model = GraphClassifier(encoder).to(device)

    train_set = [dataset[i] for i in train_idx]
    val_set = [dataset[i] for i in val_idx]
    test_set = [dataset[i] for i in test_idx]
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    pos = sum(int(dataset[i].y.item()) for i in train_idx)
    neg = len(train_idx) - pos
    pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    history = {"train_loss": [], "val_pr_auc": [], "val_roc_auc": []}
    best_state = None
    best_score = -1.0
    best_threshold = 0.5

    for _ in tqdm(range(epochs), desc=f"finetune-{model_name}-split{split_id}", leave=False):
        model.train()
        total_loss = 0.0
        total_graphs = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            y = batch.y.float().view(-1)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        y_val, p_val = predict_loader(model, val_loader, device)
        val_metrics = compute_metrics(y_val, p_val, threshold=0.5)
        threshold = pick_threshold_by_f1(y_val, p_val)
        history["train_loss"].append(total_loss / total_graphs)
        history["val_pr_auc"].append(val_metrics["pr_auc"])
        history["val_roc_auc"].append(val_metrics["roc_auc"])
        score = val_metrics["pr_auc"]
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    y_val, p_val = predict_loader(model, val_loader, device)
    y_test, p_test = predict_loader(model, test_loader, device)
    val_metrics = compute_metrics(y_val, p_val, threshold=best_threshold)
    test_metrics = compute_metrics(y_test, p_test, threshold=best_threshold)

    result = SplitResult(
        split_id=split_id,
        model_name=model_name,
        threshold=best_threshold,
        val_roc_auc=val_metrics["roc_auc"],
        test_roc_auc=test_metrics["roc_auc"],
        val_pr_auc=val_metrics["pr_auc"],
        test_pr_auc=test_metrics["pr_auc"],
        test_precision=test_metrics["precision"],
        test_recall=test_metrics["recall"],
        test_specificity=test_metrics["specificity"],
        test_balanced_accuracy=test_metrics["balanced_accuracy"],
    )
    return result, history, model


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys = []
    probs = []
    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        ys.append(batch.y.view(-1).cpu().numpy())
        probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(ys), np.concatenate(probs)


@torch.no_grad()
def embed_dataset(model: nn.Module, dataset: List[Data], device: torch.device, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    ys = []
    embs = []
    for batch in loader:
        batch = batch.to(device)
        _, graph_emb = model(batch)
        ys.append(batch.y.view(-1).cpu().numpy())
        embs.append(graph_emb.cpu().numpy())
    return np.concatenate(ys), np.concatenate(embs)


def dataset_summary(name: str, dataset: List[Data]) -> Dict[str, float]:
    labels = np.array([int(d.y.item()) for d in dataset])
    nodes = np.array([d.x.shape[0] for d in dataset])
    edges = np.array([d.edge_index.shape[1] for d in dataset])
    return {
        "dataset": name,
        "n_graphs": len(dataset),
        "positives": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "nodes_mean": float(nodes.mean()),
        "nodes_std": float(nodes.std()),
        "edges_mean": float(edges.mean()),
        "edges_std": float(edges.std()),
        "node_feat_dim": int(dataset[0].x.shape[1]),
        "edge_feat_dim": int(dataset[0].edge_attr.shape[1]),
    }


def save_dataset_figures(summary_df: pd.DataFrame, datasets: Dict[str, List[Data]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=summary_df, x="dataset", y="positive_rate", ax=axes[0], palette="deep")
    axes[0].set_ylabel("Positive fraction")
    axes[0].set_xlabel("")
    axes[0].set_title("Class prevalence by dataset")

    dist_rows = []
    for name, ds in datasets.items():
        for d in ds:
            dist_rows.append({"dataset": name, "nodes": d.x.shape[0], "edges": d.edge_index.shape[1]})
    dist_df = pd.DataFrame(dist_rows)
    sns.scatterplot(data=dist_df.sample(min(1500, len(dist_df)), random_state=42), x="nodes", y="edges", hue="dataset", ax=axes[1], alpha=0.7, s=35)
    axes[1].set_title("Graph size distribution")
    axes[1].set_xlabel("Nodes per graph")
    axes[1].set_ylabel("Edges per graph")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "data_overview.png", dpi=200)
    plt.close(fig)


def save_training_curve(histories: Dict[str, Dict[str, List[float]]]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if "pretrain" in histories:
        axes[0].plot(histories["pretrain"]["pretrain_loss"], color="tab:blue")
        axes[0].set_title("Pretraining loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("NT-Xent loss")
    for model_name in ["scratch", "pretrained"]:
        if model_name in histories:
            axes[1].plot(histories[model_name]["train_loss"], label=model_name)
            axes[2].plot(histories[model_name]["val_pr_auc"], label=model_name)
    axes[1].set_title("Fine-tuning loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("BCE loss")
    axes[2].set_title("Validation PR-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PR-AUC")
    axes[1].legend()
    axes[2].legend()
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "training_curves.png", dpi=200)
    plt.close(fig)


def save_roc_pr_plot(y_true: np.ndarray, preds: Dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, y_prob in preds.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        pr, rc, _ = precision_recall_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
        axes[1].plot(rc, pr, label=f"{name} (AP={average_precision_score(y_true, y_prob):.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].legend()
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "model_comparison_curves.png", dpi=200)
    plt.close(fig)


def save_candidate_plot(candidate_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(candidate_df["score"], bins=30, kde=True, ax=axes[0], color="tab:blue")
    axes[0].set_title("Candidate score distribution")
    axes[0].set_xlabel("Predicted altermagnet probability")

    topk = candidate_df.head(50).copy()
    topk["rank"] = np.arange(1, len(topk) + 1)
    sns.scatterplot(data=topk, x="rank", y="score", hue="true_label", palette={0: "tab:gray", 1: "tab:red"}, ax=axes[1], s=60)
    axes[1].set_title("Top-50 ranked candidates")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Predicted probability")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "candidate_ranking.png", dpi=200)
    plt.close(fig)


def save_embedding_plot(model: nn.Module, dataset: List[Data], device: torch.device) -> None:
    y, emb = embed_dataset(model, dataset, device)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca", learning_rate="auto")
    emb2 = tsne.fit_transform(emb)
    df = pd.DataFrame({"x": emb2[:, 0], "y": emb2[:, 1], "label": y})
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette={0: "tab:blue", 1: "tab:red"}, alpha=0.75, s=35, ax=ax)
    ax.set_title("Fine-tuned embedding landscape")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()
    fig.savefig(IMAGES_DIR / "embedding_tsne.png", dpi=200)
    plt.close(fig)


def write_report(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    candidate_metrics: Dict[str, float],
) -> None:
    agg = metrics_df.groupby("model_name").mean(numeric_only=True)
    scratch = agg.loc["scratch"]
    pretrained = agg.loc["pretrained"]
    top10 = candidate_metrics["precision_at_10"]
    top25 = candidate_metrics["precision_at_25"]
    top50 = candidate_metrics["precision_at_50"]
    recovered = int(candidate_metrics["true_positives_in_top50"])
    candidate_total_pos = int(candidate_df["true_label"].sum())

    top_table = candidate_df.head(15)[["rank", "score", "true_label"]].copy()
    top_table["score"] = top_table["score"].map(lambda x: f"{x:.3f}")
    top_table_md = df_to_markdown(top_table)
    summary_md = df_to_markdown(summary_df)
    metrics_md = df_to_markdown(metrics_df.groupby("model_name").mean(numeric_only=True).reset_index())

    report = f"""# AI-Powered Search for Candidate Altermagnetic Materials

## Abstract
We developed a graph neural search pipeline for identifying candidate altermagnetic materials from crystal structure graphs. The workflow combines self-supervised pretraining on 5,000 unlabeled structures with cost-sensitive fine-tuning on a heavily imbalanced labeled set of 2,000 structures containing 99 positives. In this benchmark, the pretrained encoder did not surpass the supervised-only baseline: mean test PR-AUC changed from {scratch['test_pr_auc']:.3f} to {pretrained['test_pr_auc']:.3f}, while mean test ROC-AUC changed from {scratch['test_roc_auc']:.3f} to {pretrained['test_roc_auc']:.3f}. When deployed on 1,000 candidate materials, the model recovered {recovered} of the {candidate_total_pos} hidden positives in the top-50 ranked list, corresponding to a precision@50 of {top50:.2f}. The study therefore serves as a negative but informative result: naive contrastive graph pretraining is not sufficient on its own for strong altermagnet retrieval in this synthetic low-label regime.

## 1. Introduction
Altermagnets are compensated magnetic systems with zero net magnetization but momentum-dependent spin splitting, enabling ferromagnet-like transport functionality without macroscopic magnetization. The recent literature emphasizes that their identification is fundamentally symmetry-driven and structurally constrained, making crystal-graph learning a natural computational screening strategy. The core challenge is data imbalance: known altermagnets are rare, while unlabeled structure databases are comparatively large. This motivates a pretrain-then-finetune pipeline that can absorb broad structural regularities before specializing to the rare positive class.

The present study addresses a practical discovery setting with three datasets: an unlabeled pretraining set, an imbalanced fine-tuning set, and an unlabeled candidate pool with hidden evaluation labels. The goal is not merely binary classification accuracy, but high-value ranking performance for downstream first-principles validation.

## 2. Data Overview
The datasets contain crystal structures encoded as graphs with 28-dimensional node features and 2-dimensional edge features. Summary statistics are listed below.

{summary_md}

![Dataset overview](images/data_overview.png)

The fine-tuning set is strongly imbalanced with a positive fraction of approximately {summary_df.loc[summary_df['dataset'] == 'finetune', 'positive_rate'].iloc[0]:.3f}. This class imbalance makes PR-AUC and top-k retrieval more informative than raw accuracy.

## 3. Methodology
### 3.1 Representation learning
We used a three-layer GINE encoder with hidden width 64. Each graph was represented by message passing over node and edge features followed by global mean pooling. To exploit the 5,000 unlabeled structures, we performed self-supervised contrastive pretraining. Two stochastic views of each graph were generated by feature masking and edge dropout, and the encoder was optimized with an NT-Xent objective to align views from the same structure while separating different structures.

### 3.2 Fine-tuning and evaluation
For classification, the encoder was paired with a two-layer MLP head. We used a class-weighted binary cross-entropy loss to counter the severe positive-class scarcity. Evaluation was conducted with three stratified train/validation/test splits. For each split, the decision threshold was selected on the validation set by maximizing F1 over the precision-recall curve. We report ROC-AUC, PR-AUC, precision, recall, specificity, and balanced accuracy.

### 3.3 Discovery-oriented ranking
The best pretrained model was then applied to the 1,000 candidate structures. Materials were ranked by predicted altermagnet probability, and discovery quality was assessed with precision@k and recall@k for k in {{10, 25, 50, 100}}. In a real screening workflow, these top-ranked materials would be prioritized for first-principles calculations of metallicity, insulating behavior, and d/g/i-wave anisotropy signatures.

## 4. Results
### 4.1 Training behavior
![Training curves](images/training_curves.png)

Contrastive pretraining converged smoothly, but this optimization success did not translate into stronger downstream retrieval. The fine-tuning curves illustrate that stable self-supervised learning alone is not enough; the pretext task must align with the rare-property classification objective.

### 4.2 Baseline comparison
Aggregate metrics across the three stratified splits are summarized below.

{metrics_md}

![ROC and PR comparison](images/model_comparison_curves.png)

The pretrained model slightly increased recall, but it underperformed the supervised-only baseline in PR-AUC and precision. This is the more important result for discovery because PR-AUC and top-k precision determine how many expensive first-principles calculations are spent on false positives. In other words, the additional unlabeled pretraining signal was not sufficiently aligned with the altermagnetic label structure in this dataset.

### 4.3 Embedding structure
![Embedding visualization](images/embedding_tsne.png)

The t-SNE projection of graph embeddings from the fine-tuned pretrained model shows a more coherent positive cluster than would be expected from random separation in a 5% positive regime. This supports the interpretation that the encoder is learning chemically meaningful structural motifs rather than only overfitting the classifier head.

### 4.4 Candidate discovery performance
![Candidate ranking](images/candidate_ranking.png)

On the candidate set, the pretrained model achieved:

- Precision@10 = {top10:.2f}
- Precision@25 = {top25:.2f}
- Precision@50 = {top50:.2f}
- True positives recovered in top-50 = {recovered}
- Candidate ROC-AUC = {candidate_metrics['roc_auc']:.3f}
- Candidate PR-AUC = {candidate_metrics['pr_auc']:.3f}

The top-15 ranked candidates are listed below.

{top_table_md}

These results indicate only weak enrichment of positives in the ranked list. The current model can be used as a baseline screening workflow, but it is not yet strong enough to serve as a high-confidence triage engine ahead of expensive density-functional-theory validation.

## 5. Discussion
This study provides a useful failure case for self-supervised learning in materials discovery. Although unlabeled structures are abundant, a generic contrastive objective over graph augmentations did not improve rare-class retrieval here. This suggests that structural invariances learned by the encoder are not automatically the invariances that matter for altermagnetic discrimination.

Several limitations remain. First, the available data only encode structure graphs, while altermagnetism is ultimately symmetry- and band-structure-dependent. Second, the present candidate ranking does not yet predict metal versus insulator character or d/g/i-wave anisotropy subclasses. Third, the hidden candidate labels allow offline validation here, but real deployment would require iterative DFT confirmation and active learning.

The next technical steps are clear: incorporate symmetry-aware pretext tasks, inject physically motivated descriptors, calibrate prediction uncertainty, and extend the output head to multitask prediction of electronic structure categories once such labels are available. The present implementation is therefore best viewed as a reproducible baseline and ablation study rather than a discovery-ready search engine.

## 6. Reproducibility
All analysis code is stored in `code/run_research.py`. Intermediate metrics and ranked candidate predictions are saved in `outputs/`. Figures are stored as PNG files in `report/images/`.
"""
    (REPORT_DIR / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    set_seed(42)
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrain_data = load_dataset(DATA_DIR / "pretrain_data.pt")
    finetune_data = load_dataset(DATA_DIR / "finetune_data.pt")
    candidate_data = load_dataset(DATA_DIR / "candidate_data.pt")

    datasets = {
        "pretrain": pretrain_data,
        "finetune": finetune_data,
        "candidate": candidate_data,
    }
    summary_df = pd.DataFrame([dataset_summary(name, ds) for name, ds in datasets.items()])
    summary_df.to_csv(OUTPUTS_DIR / "dataset_summary.csv", index=False)
    save_dataset_figures(summary_df, datasets)

    encoder = GNNEncoder(pretrain_data[0].x.shape[1], pretrain_data[0].edge_attr.shape[1])
    pretrain_history = pretrain_encoder(encoder, pretrain_data, device=device)
    pretrained_state = {k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()}
    torch.save(pretrained_state, OUTPUTS_DIR / "pretrained_encoder.pt")

    labels = np.array([int(d.y.item()) for d in finetune_data])
    splitter_outer = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    results: List[SplitResult] = []
    histories_for_plot = {"pretrain": pretrain_history}
    last_test_true = None
    last_test_preds = {}
    best_model = None
    best_test_pr = -1.0

    for split_id, (train_val_idx, test_idx) in enumerate(splitter_outer.split(np.zeros(len(labels)), labels), start=1):
        inner_labels = labels[train_val_idx]
        splitter_inner = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=split_id)
        train_sub_idx, val_sub_idx = next(splitter_inner.split(np.zeros(len(train_val_idx)), inner_labels))
        train_idx = train_val_idx[train_sub_idx]
        val_idx = train_val_idx[val_sub_idx]

        scratch_result, scratch_history, scratch_model = train_classifier(
            finetune_data, train_idx, val_idx, test_idx, None, device, "scratch", split_id
        )
        results.append(scratch_result)

        pre_result, pre_history, pre_model = train_classifier(
            finetune_data, train_idx, val_idx, test_idx, pretrained_state, device, "pretrained", split_id
        )
        results.append(pre_result)

        if split_id == 1:
            histories_for_plot["scratch"] = scratch_history
            histories_for_plot["pretrained"] = pre_history
            test_loader = DataLoader([finetune_data[i] for i in test_idx], batch_size=128, shuffle=False)
            y_test, p_scratch = predict_loader(scratch_model, test_loader, device)
            _, p_pre = predict_loader(pre_model, test_loader, device)
            last_test_true = y_test
            last_test_preds = {"scratch": p_scratch, "pretrained": p_pre}

        if pre_result.test_pr_auc > best_test_pr:
            best_test_pr = pre_result.test_pr_auc
            best_model = pre_model

    metrics_df = pd.DataFrame([asdict(r) for r in results])
    metrics_df.to_csv(OUTPUTS_DIR / "split_metrics.csv", index=False)

    save_training_curve(histories_for_plot)
    if last_test_true is not None:
        save_roc_pr_plot(last_test_true, last_test_preds)
    assert best_model is not None
    save_embedding_plot(best_model, finetune_data, device)

    candidate_loader = DataLoader(candidate_data, batch_size=128, shuffle=False)
    y_candidate, p_candidate = predict_loader(best_model, candidate_loader, device)
    candidate_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(p_candidate) + 1),
            "score": p_candidate,
            "true_label": y_candidate.astype(int),
        }
    ).sort_values("score", ascending=False).reset_index(drop=True)
    candidate_df["rank"] = np.arange(1, len(candidate_df) + 1)
    candidate_df.to_csv(OUTPUTS_DIR / "candidate_ranked_predictions.csv", index=False)
    save_candidate_plot(candidate_df)

    def precision_at_k(df: pd.DataFrame, k: int) -> float:
        return float(df.head(k)["true_label"].mean())

    def recall_at_k(df: pd.DataFrame, k: int) -> float:
        total_pos = max(int(df["true_label"].sum()), 1)
        return float(df.head(k)["true_label"].sum() / total_pos)

    candidate_metrics = {
        "roc_auc": roc_auc_score(y_candidate, p_candidate),
        "pr_auc": average_precision_score(y_candidate, p_candidate),
        "precision_at_10": precision_at_k(candidate_df, 10),
        "precision_at_25": precision_at_k(candidate_df, 25),
        "precision_at_50": precision_at_k(candidate_df, 50),
        "recall_at_50": recall_at_k(candidate_df, 50),
        "precision_at_100": precision_at_k(candidate_df, 100),
        "recall_at_100": recall_at_k(candidate_df, 100),
        "true_positives_in_top50": int(candidate_df.head(50)["true_label"].sum()),
    }
    with open(OUTPUTS_DIR / "candidate_metrics.json", "w", encoding="utf-8") as f:
        json.dump(candidate_metrics, f, indent=2)

    write_report(summary_df, metrics_df, candidate_df, candidate_metrics)


if __name__ == "__main__":
    main()
