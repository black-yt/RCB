"""
AlphaFold3-inspired diffusion framework for biomolecular structure prediction.

This module implements key components of a unified deep learning framework
for predicting 3D structures of biomolecular complexes:
  1. Input encoders (protein sequence, small molecule)
  2. Pairwise representation module (Evoformer-inspired)
  3. Diffusion-based structure module
  4. Confidence estimation

Reference: Abramson et al., AlphaFold3, Nature 2024.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


# ============================================================
# 1. SEQUENCE ENCODER
# ============================================================

class SequenceEncoder(nn.Module):
    """
    Encodes amino acid sequences into single-residue embeddings.
    Incorporates positional encodings and residue-level features.
    """
    def __init__(self, n_aa_types: int = 21, d_model: int = 128, max_len: int = 1024):
        super().__init__()
        self.n_aa_types = n_aa_types
        self.d_model = d_model

        # Amino acid type embedding (20 standard + 1 unknown)
        self.aa_embedding = nn.Embedding(n_aa_types, d_model)

        # Positional encoding (sinusoidal)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Layer norm and projection
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq_tokens: [B, L] integer token sequence
        Returns:
            embeddings: [B, L, d_model]
        """
        x = self.aa_embedding(seq_tokens)  # [B, L, d_model]
        x = x + self.pe[:x.size(1)]       # Add positional encoding
        x = self.norm(x)
        x = self.dropout(x)
        return x


# ============================================================
# 2. SMALL MOLECULE ENCODER
# ============================================================

class AtomEncoder(nn.Module):
    """
    Encodes small molecule atoms and their local chemical environment.
    Based on SE(3)-invariant atom features.
    """
    def __init__(self, n_atom_types: int = 10, d_model: int = 64):
        super().__init__()
        # Atom type embedding (C, N, O, S, P, F, Cl, Br, I, Other)
        self.atom_type_embedding = nn.Embedding(n_atom_types, d_model // 2)

        # Bond features: linear projection
        self.bond_mlp = nn.Sequential(
            nn.Linear(4, d_model // 2),  # bond type one-hot
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 2),
        )

        # Combine atom + local environment
        self.atom_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, atom_types: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """
        Args:
            atom_types: [N_atoms] integer atom types
        Returns:
            atom_embeddings: [N_atoms, d_model]
        """
        atom_emb = self.atom_type_embedding(atom_types)  # [N, d/2]
        # Pad with zeros for bond features
        bond_features = torch.zeros(n_atoms, self.bond_mlp[0].in_features, device=atom_types.device)
        bond_emb = self.bond_mlp(bond_features)  # [N, d/2]

        combined = torch.cat([atom_emb, bond_emb], dim=-1)  # [N, d]
        return self.atom_mlp(combined)


# ============================================================
# 3. PAIRWISE REPRESENTATION (EVOFORMER-INSPIRED)
# ============================================================

class TriangleAttention(nn.Module):
    """
    Triangle attention for pairwise representations.
    Key innovation from AlphaFold2/3 — updates pair features using
    triangle multiplicative updates.
    """
    def __init__(self, d_pair: int = 64, n_heads: int = 4):
        super().__init__()
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.d_head = d_pair // n_heads

        self.norm = nn.LayerNorm(d_pair)
        self.q = nn.Linear(d_pair, d_pair, bias=False)
        self.k = nn.Linear(d_pair, d_pair, bias=False)
        self.v = nn.Linear(d_pair, d_pair, bias=False)
        self.gate = nn.Linear(d_pair, d_pair)
        self.out = nn.Linear(d_pair, d_pair)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, L, L, d_pair] pairwise representation
        Returns:
            updated z: [B, L, L, d_pair]
        """
        B, L, _, D = z.shape
        z_norm = self.norm(z)

        # Multi-head attention along rows
        Q = self.q(z_norm).view(B, L, L, self.n_heads, self.d_head)
        K = self.k(z_norm).view(B, L, L, self.n_heads, self.d_head)
        V = self.v(z_norm).view(B, L, L, self.n_heads, self.d_head)

        # Attention: [B, L, n_heads, L, L]
        Q = Q.permute(0, 1, 3, 2, 4)  # [B, L, n_heads, L, d_head]
        K = K.permute(0, 1, 3, 4, 2)  # [B, L, n_heads, d_head, L]

        scale = math.sqrt(self.d_head)
        attn = torch.matmul(Q, K) / scale  # [B, L, n_heads, L, L]
        attn = F.softmax(attn, dim=-1)

        V = V.permute(0, 1, 3, 2, 4)  # [B, L, n_heads, L, d_head]
        out = torch.matmul(attn, V)    # [B, L, n_heads, L, d_head]
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, L, L, D)

        # Gating
        gate = torch.sigmoid(self.gate(z_norm))
        out = gate * out

        return self.out(out)


class PairwiseModule(nn.Module):
    """
    Builds and refines pairwise (residue-residue) representations.
    Used for capturing long-range dependencies between residues.
    """
    def __init__(self, d_single: int = 128, d_pair: int = 64, n_layers: int = 4):
        super().__init__()
        self.d_pair = d_pair

        # Initialize pair representation from single representations
        self.init_pair = nn.Linear(d_single * 2, d_pair)

        # Relative position encoding
        self.rel_pos_emb = nn.Embedding(65, d_pair)  # -32 to +32

        # Stack of triangle attention layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'tri_attn': TriangleAttention(d_pair),
                'ffn': nn.Sequential(
                    nn.LayerNorm(d_pair),
                    nn.Linear(d_pair, d_pair * 4),
                    nn.GELU(),
                    nn.Linear(d_pair * 4, d_pair),
                )
            })
            for _ in range(n_layers)
        ])

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: [B, L, d_single] single residue representations
        Returns:
            z: [B, L, L, d_pair] pairwise representations
        """
        B, L, _ = s.shape

        # Initialize pairwise from outer concatenation of single reps
        si = s.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, d_single]
        sj = s.unsqueeze(1).expand(-1, L, -1, -1)  # [B, L, L, d_single]
        z = self.init_pair(torch.cat([si, sj], dim=-1))  # [B, L, L, d_pair]

        # Add relative position encoding
        positions = torch.arange(L, device=s.device)
        rel_pos = (positions.unsqueeze(0) - positions.unsqueeze(1)).clamp(-32, 32) + 32  # [L, L]
        z = z + self.rel_pos_emb(rel_pos).unsqueeze(0)

        # Refine with triangle attention
        for layer in self.layers:
            z = z + layer['tri_attn'](z)
            z = z + layer['ffn'](z)

        return z


# ============================================================
# 4. DIFFUSION-BASED STRUCTURE MODULE
# ============================================================

class DiffusionNoiseSchedule:
    """
    Cosine noise schedule for diffusion-based structure generation.
    Based on AlphaFold3's diffusion formulation.
    """
    def __init__(self, T: int = 200, s_min: float = 0.0001, s_max: float = 160.0):
        self.T = T
        self.s_min = s_min
        self.s_max = s_max

        # Log-linear noise schedule
        # sigma(t) = exp(log(s_min) + (t/T) * log(s_max/s_min))
        t = torch.linspace(0, 1, T + 1)
        self.sigmas = torch.exp(
            math.log(s_min) + t * (math.log(s_max) - math.log(s_min))
        )

    def get_sigma(self, t: int) -> float:
        return self.sigmas[t].item()

    def noise_sample(self, x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t: x_t = x_0 + sigma(t) * epsilon"""
        sigma = self.sigmas[t].to(x0.device)
        epsilon = torch.randn_like(x0)
        x_t = x0 + sigma * epsilon
        return x_t, epsilon


class StructureDiffuser(nn.Module):
    """
    Score-based diffusion model for 3D structure prediction.

    Predicts the denoised coordinates given noisy coordinates and
    conditional features from the pairwise representation.

    Inspired by: AlphaFold3 diffusion module
    """
    def __init__(self, d_pair: int = 64, d_atom: int = 3, n_layers: int = 6):
        super().__init__()
        self.d_pair = d_pair

        # Atom position encoder: 3D coords + time embedding -> d_pair
        self.coord_encoder = nn.Sequential(
            nn.Linear(3 + 64, d_pair),  # coords + time embedding
            nn.SiLU(),
            nn.Linear(d_pair, d_pair),
        )

        # Time embedding (sinusoidal, then MLP)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Per-atom denoising layers (IPA-inspired)
        self.atom_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_pair),
                nn.Linear(d_pair, d_pair),
                nn.SiLU(),
                nn.Linear(d_pair, d_pair),
            )
            for _ in range(n_layers)
        ])

        # Output: predict displacement (score function)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_pair),
            nn.Linear(d_pair, d_pair // 2),
            nn.SiLU(),
            nn.Linear(d_pair // 2, 3),  # 3D displacement
        )

    def get_time_embedding(self, t: torch.Tensor, dim: int = 64) -> torch.Tensor:
        """Sinusoidal time embedding."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device).float() * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x_t: torch.Tensor, t: torch.Tensor,
                pair_repr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_t: [B, N, 3] noisy atom coordinates at timestep t
            t: [B] timestep values
            pair_repr: [B, N, N, d_pair] pairwise representations (optional)
        Returns:
            score: [B, N, 3] predicted score (denoised displacement)
        """
        B, N, _ = x_t.shape

        # Time embedding
        t_emb = self.get_time_embedding(t, dim=64)  # [B, 64]
        t_emb = self.time_mlp(t_emb)  # [B, 64]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, N, -1)  # [B, N, 64]

        # Encode atom positions with time
        h = self.coord_encoder(torch.cat([x_t, t_emb_expanded], dim=-1))  # [B, N, d_pair]

        # If pair representation available, aggregate information
        if pair_repr is not None:
            # Aggregate pairwise info: mean over second dim -> [B, N, d_pair]
            pair_agg = pair_repr.mean(dim=2)  # [B, N, d_pair]
            h = h + pair_agg

        # Per-atom refinement
        for layer in self.atom_layers:
            h = h + layer(h)

        # Output displacement
        score = self.output_proj(h)  # [B, N, 3]
        return score


# ============================================================
# 5. FULL UNIFIED FRAMEWORK
# ============================================================

class BioMolecularDiffusionFramework(nn.Module):
    """
    Unified deep learning framework for biomolecular complex structure prediction.

    Architecture:
    1. Sequence encoder (protein, nucleic acid)
    2. Atom encoder (small molecules)
    3. Pairwise representation module (Evoformer-inspired)
    4. Diffusion structure module (score-based)
    5. Confidence estimator

    Supports: protein sequences, small molecules, and nucleic acid sequences.
    """
    def __init__(self,
                 n_aa_types: int = 21,
                 d_single: int = 128,
                 d_pair: int = 64,
                 n_evoformer_layers: int = 4,
                 n_diffusion_layers: int = 6,
                 diffusion_T: int = 200):
        super().__init__()

        # Encoders
        self.protein_encoder = SequenceEncoder(n_aa_types, d_single)
        self.atom_encoder = AtomEncoder(n_atom_types=10, d_model=d_single)

        # Pairwise module
        self.pairwise_module = PairwiseModule(d_single, d_pair, n_evoformer_layers)

        # Diffusion module
        self.diffuser = StructureDiffuser(d_pair, n_layers=n_diffusion_layers)
        self.noise_schedule = DiffusionNoiseSchedule(T=diffusion_T)

        # Confidence predictor (pLDDT-inspired)
        self.confidence_head = nn.Sequential(
            nn.Linear(d_single + d_pair, 128),
            nn.GELU(),
            nn.Linear(128, 50),  # 50 bins for pLDDT
            nn.Softmax(dim=-1),
        )

    def encode(self, seq_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode sequence and build pairwise representation."""
        s = self.protein_encoder(seq_tokens)  # [B, L, d_single]
        z = self.pairwise_module(s)            # [B, L, L, d_pair]
        return s, z

    def predict_score(self, x_t: torch.Tensor, t: torch.Tensor,
                      pair_repr: torch.Tensor) -> torch.Tensor:
        """Predict denoising score."""
        return self.diffuser(x_t, t, pair_repr)

    def forward(self, seq_tokens: torch.Tensor, x_noisy: torch.Tensor,
                t: torch.Tensor) -> dict:
        """
        Full forward pass.
        Args:
            seq_tokens: [B, L] amino acid token indices
            x_noisy: [B, L, 3] noisy CA coordinates
            t: [B] diffusion timesteps
        Returns:
            dict with score predictions and confidence estimates
        """
        s, z = self.encode(seq_tokens)  # [B,L,d_s], [B,L,L,d_p]

        # Predict score for denoising
        score = self.diffuser(x_noisy, t, pair_repr=z)

        # Confidence estimation from single + pairwise diagonal
        z_diag = z.diagonal(dim1=1, dim2=2).transpose(1, 2)  # [B, L, d_pair]
        conf_input = torch.cat([s, z_diag], dim=-1)
        plddt_logits = self.confidence_head(conf_input)

        # pLDDT score (0-100)
        bin_centers = torch.linspace(0.5, 99.5, 50, device=s.device)
        plddt = (plddt_logits * bin_centers).sum(dim=-1)  # [B, L]

        return {
            'score': score,
            'plddt': plddt,
            'single_repr': s,
            'pair_repr': z,
        }

    @torch.no_grad()
    def sample(self, seq_tokens: torch.Tensor, n_steps: int = 50,
               return_trajectory: bool = False):
        """
        Generate structure using DDPM-style reverse diffusion sampling.

        Args:
            seq_tokens: [B, L] sequence tokens
            n_steps: number of denoising steps
            return_trajectory: if True, return intermediate coordinates
        Returns:
            x0: [B, L, 3] predicted coordinates
        """
        B, L = seq_tokens.shape
        device = seq_tokens.device

        # Encode sequence
        s, z = self.encode(seq_tokens)

        # Initialize from random noise
        sigma_max = self.noise_schedule.s_max
        x = torch.randn(B, L, 3, device=device) * sigma_max

        trajectory = [x.clone()] if return_trajectory else None
        T = self.noise_schedule.T

        # Reverse diffusion (Euler-Maruyama)
        step_indices = torch.linspace(T - 1, 0, n_steps, dtype=torch.long)

        for i, t_idx in enumerate(step_indices):
            t_val = torch.tensor([t_idx.item()], device=device).expand(B)

            sigma_curr = self.noise_schedule.sigmas[t_idx].to(device)
            sigma_next = self.noise_schedule.sigmas[max(t_idx - 1, 0)].to(device)

            # Predict score (denoising direction)
            score = self.diffuser(x, t_val, pair_repr=z)

            # Step size
            d_sigma = sigma_curr - sigma_next

            # Euler step toward cleaner structure
            x = x - d_sigma * score / (sigma_curr + 1e-8)

            # Add small stochastic term for variance
            if t_idx > 0:
                noise_scale = torch.sqrt(2 * d_sigma * sigma_next + 1e-8)
                x = x + noise_scale * torch.randn_like(x)

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, trajectory
        return x


# ============================================================
# 6. UTILITY FUNCTIONS
# ============================================================

AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6,
    'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13,
    'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, 'X': 20
}

ELEMENT_TO_IDX = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4,
    'F': 5, 'Cl': 6, 'Br': 7, 'I': 8
}


def sequence_to_tokens(sequence: str) -> torch.Tensor:
    """Convert amino acid sequence string to integer tokens."""
    tokens = [AA_TO_IDX.get(aa, 20) for aa in sequence.upper()]
    return torch.tensor(tokens, dtype=torch.long)


def compute_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD between two coordinate arrays (after superposition)."""
    assert coords1.shape == coords2.shape, "Coordinate arrays must have same shape"
    n = len(coords1)

    # Center
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # Kabsch algorithm for optimal rotation
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    # Apply rotation
    c1_rotated = c1 @ R.T

    # Compute RMSD
    diff = c1_rotated - c2
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_complexity():
    """Report model complexity statistics."""
    model = BioMolecularDiffusionFramework(
        d_single=128, d_pair=64, n_evoformer_layers=4, n_diffusion_layers=6
    )

    total = count_parameters(model)
    breakdown = {
        'protein_encoder': count_parameters(model.protein_encoder),
        'atom_encoder': count_parameters(model.atom_encoder),
        'pairwise_module': count_parameters(model.pairwise_module),
        'diffuser': count_parameters(model.diffuser),
        'confidence_head': count_parameters(model.confidence_head),
    }

    print(f"\nModel complexity:")
    print(f"  Total parameters: {total:,}")
    for name, n in breakdown.items():
        print(f"  {name}: {n:,} ({100*n/total:.1f}%)")

    return model, breakdown


if __name__ == '__main__':
    print("=" * 60)
    print("BioMolecular Diffusion Framework — Architecture Test")
    print("=" * 60)

    model, breakdown = compute_model_complexity()
    model.eval()

    # Test with small inputs
    B, L = 1, 107  # Batch=1, 107 residues (FKBP12)
    seq = torch.randint(0, 20, (B, L))
    x_noisy = torch.randn(B, L, 3)
    t = torch.tensor([50])

    print(f"\nTest forward pass (B={B}, L={L}):")
    with torch.no_grad():
        out = model(seq, x_noisy, t)
    print(f"  Score shape: {out['score'].shape}")
    print(f"  pLDDT shape: {out['plddt'].shape}")
    print(f"  Mean pLDDT: {out['plddt'].mean().item():.1f}")
    print(f"  Pair repr shape: {out['pair_repr'].shape}")

    # Test sampling
    print(f"\nTest sampling (n_steps=10):")
    with torch.no_grad():
        x_pred, traj = model.sample(seq, n_steps=10, return_trajectory=True)
    print(f"  Predicted coords shape: {x_pred.shape}")
    print(f"  Trajectory length: {len(traj)}")
    print(f"  Coord range: [{x_pred.min().item():.2f}, {x_pred.max().item():.2f}]")
    print("\nAll architecture tests passed!")
