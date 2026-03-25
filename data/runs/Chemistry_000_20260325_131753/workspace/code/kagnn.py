"""
Kolmogorov-Arnold Graph Neural Networks (KA-GNN) — Efficient Implementation
============================================================================
The FourierKAN layer is reformulated as a linear transform on explicit
Fourier feature maps, enabling efficient GPU/CPU execution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
import numpy as np
import math


# ─────────────────────────────────────────────────────────────────────────────
# Efficient Fourier-KAN Layer
# ─────────────────────────────────────────────────────────────────────────────

class FourierKANLayer(nn.Module):
    """
    Fourier-based KAN layer.

    Equivalent to: for each input x_j, compute k·x_j for k=1..K,
    apply cos and sin, then apply a linear transform over the resulting
    2*K*in_features expanded representation.

    This is algebraically identical to the per-edge formulation but
    uses standard torch.linear for efficiency.
    """

    def __init__(self, in_features: int, out_features: int, n_harmonics: int = 3):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.n_harmonics  = n_harmonics

        fourier_dim = 2 * n_harmonics * in_features
        self.weight = nn.Parameter(torch.empty(out_features, fourier_dim))
        self.bias   = nn.Parameter(torch.zeros(out_features))
        # DC (constant) term
        self.dc_weight = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.dc_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., in_features)  →  (..., out_features)"""
        # Harmonics k = 1 … K
        k = torch.arange(1, self.n_harmonics + 1, device=x.device, dtype=x.dtype)

        # x: (..., in) → (..., in, K)
        xk = x.unsqueeze(-1) * k.view(*([1] * len(x.shape)), -1)

        # Fourier features: concat [cos, sin] → (..., 2*K*in)
        fourier = torch.cat([xk.cos(), xk.sin()], dim=-1)  # (..., in, 2*K)
        fourier = fourier.flatten(-2)  # (..., 2*K*in)

        # Linear transform + DC term
        return F.linear(fourier, self.weight, self.bias) + F.linear(x, self.dc_weight)


# ─────────────────────────────────────────────────────────────────────────────
# KAN residual block
# ─────────────────────────────────────────────────────────────────────────────

class KANBlock(nn.Module):
    """FourierKAN + residual + LayerNorm."""

    def __init__(self, dim: int, n_harmonics: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kan  = FourierKANLayer(dim, dim, n_harmonics)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.norm(x + self.drop(self.kan(x)))


# ─────────────────────────────────────────────────────────────────────────────
# KA-GNN Message Passing Layer
# ─────────────────────────────────────────────────────────────────────────────

class KAGNNConv(MessagePassing):
    """
    KA-GNN message-passing layer.

    Message: m_ij = KAN([h_i || h_j || e_ij])
    Update:  h_i' = KAN(h_i + agg(m_ij))
    """

    def __init__(self, h_dim: int, e_dim: int, n_harmonics: int = 3, dropout: float = 0.1):
        super().__init__(aggr='add')

        msg_in = 2 * h_dim + e_dim
        self.msg_kan  = FourierKANLayer(msg_in, h_dim, n_harmonics)
        self.msg_norm = nn.LayerNorm(h_dim)
        self.upd      = KANBlock(h_dim, n_harmonics, dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_norm(F.silu(self.msg_kan(inp)))

    def update(self, aggr_out, x):
        return self.upd(x + aggr_out)


# ─────────────────────────────────────────────────────────────────────────────
# Full KA-GNN
# ─────────────────────────────────────────────────────────────────────────────

class KAGNN(nn.Module):
    """Kolmogorov-Arnold Graph Neural Network."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_harmonics: int = 3,
        n_tasks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList([
            KAGNNConv(hidden_dim, hidden_dim, n_harmonics, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.pool = global_mean_pool

        self.head = nn.Sequential(
            FourierKANLayer(hidden_dim, hidden_dim // 2, n_harmonics),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x.float())
        e = self.edge_embed(edge_attr.float())

        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, e))

        g = self.pool(x, batch)
        return self.head(g)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline GNN (MLP-based)
# ─────────────────────────────────────────────────────────────────────────────

class MLPGNNConv(MessagePassing):
    """Standard MLP message-passing baseline."""

    def __init__(self, h_dim: int, e_dim: int, dropout: float = 0.1):
        super().__init__(aggr='add')

        msg_in = 2 * h_dim + e_dim
        self.msg_net = nn.Sequential(
            nn.Linear(msg_in, h_dim),
            nn.SiLU(),
            nn.LayerNorm(h_dim),
        )
        self.upd_net = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.LayerNorm(h_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        inp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.msg_net(inp)

    def update(self, aggr_out, x):
        return self.drop(self.upd_net(x + aggr_out)) + x


class BaselineGNN(nn.Module):
    """Baseline GNN with MLP transformations (same depth and width as KAGNN)."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        n_layers: int = 2,
        n_tasks: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.convs = nn.ModuleList([
            MLPGNNConv(hidden_dim, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])

        self.pool = global_mean_pool

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_tasks),
        )

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_embed(x.float())
        e = self.edge_embed(edge_attr.float())
        for conv, norm in zip(self.convs, self.norms):
            x = norm(conv(x, edge_index, e))
        g = self.pool(x, batch)
        return self.head(g)
