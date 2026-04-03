"""
Fast LES model using vectorized PyTorch operations.
Optimized for 128-atom systems with batch processing.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def rbf_expansion(distances, n_rbf=16, r_max=8.0):
    """Vectorized RBF expansion. distances: tensor of shape (*)"""
    centers = torch.linspace(0.5, r_max, n_rbf, device=distances.device)
    eta = 0.5 * r_max / n_rbf
    d = distances.unsqueeze(-1)
    rbf = torch.exp(-((d - centers) ** 2) / (2 * eta ** 2))
    # Smooth cutoff
    x = (distances / r_max).clamp(max=1.0)
    smooth = (0.5 * (1.0 + torch.cos(np.pi * x))).unsqueeze(-1)
    return rbf * smooth


def compute_features_fast(positions, cutoff, n_rbf):
    """
    Vectorized feature computation.
    positions: (N, 3) tensor
    Returns: (N, n_rbf) features
    """
    N = positions.shape[0]
    # Full pairwise distances
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    # Mask: within cutoff, exclude self
    mask = (dist > 1e-6) & (dist < cutoff)  # (N, N)

    # Compute RBF for all pairs
    d_flat = dist[mask]  # (n_pairs,)
    rbf_flat = rbf_expansion(d_flat, n_rbf=n_rbf, r_max=cutoff)  # (n_pairs, n_rbf)

    # Scatter sum: for each atom i, sum over neighbors
    row_idx = mask.nonzero(as_tuple=True)[0]  # which atom i
    features = torch.zeros(N, n_rbf, device=positions.device)
    features.scatter_add_(0, row_idx.unsqueeze(-1).expand(-1, n_rbf), rbf_flat)
    return features


def compute_coulomb_energy_fast(positions, charges):
    """
    Vectorized Coulomb energy computation.
    positions: (N, 3), charges: (N,)
    Returns: scalar energy
    """
    N = positions.shape[0]
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (N, N)
    # Upper triangle only, no self
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=positions.device), diagonal=1)
    r_ij = dist[mask]
    q_outer = charges.unsqueeze(1) * charges.unsqueeze(0)  # (N, N)
    qq_ij = q_outer[mask]
    return (qq_ij / r_ij).sum()


class FastLESModel(nn.Module):
    """Fast vectorized LES model."""

    def __init__(self, n_rbf=16, n_hidden=64, cutoff=6.0):
        super().__init__()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        # Charge network
        self.charge_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )
        # Short-range energy network
        self.sr_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, positions, total_charge=0.0):
        """
        positions: (N, 3)
        total_charge: scalar
        Returns: energy scalar, latent_charges (N,)
        """
        N = positions.shape[0]
        features = compute_features_fast(positions, self.cutoff, self.n_rbf)

        # Latent charges with constraint
        raw_q = self.charge_net(features).squeeze(-1)
        correction = (total_charge - raw_q.sum()) / N
        latent_q = raw_q + correction

        # Long-range Coulomb
        E_lr = compute_coulomb_energy_fast(positions, latent_q)

        # Short-range
        E_sr = self.sr_net(features).squeeze(-1).sum()

        return E_lr + E_sr, latent_q


class FastSRModel(nn.Module):
    """Fast short-range-only model."""

    def __init__(self, n_rbf=16, n_hidden=64, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, n_hidden), nn.SiLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, positions, total_charge=0.0):
        features = compute_features_fast(positions, self.cutoff, self.n_rbf)
        return self.net(features).squeeze(-1).sum(), None


def train_fast(model, positions_list, energies, forces_list=None,
               total_charges=None, n_epochs=200, lr=5e-4,
               e_weight=1.0, f_weight=0.1, verbose=True, batch_size=10):
    """
    Efficient training loop with optional mini-batching.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    if total_charges is None:
        total_charges = [0.0] * len(positions_list)

    energies_t = torch.tensor(energies, dtype=torch.float32)
    e_mean = energies_t.mean().item()
    e_std = (energies_t.std() + 1e-8).item()

    losses = {'total': [], 'energy': [], 'force': []}

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        e_loss_total = 0.0
        f_loss_total = 0.0

        idx_perm = np.random.permutation(len(positions_list))

        for idx in idx_perm:
            pos = torch.tensor(positions_list[idx], dtype=torch.float32)
            e_ref = energies_t[idx]
            tc = float(total_charges[idx])

            optimizer.zero_grad()

            if forces_list is not None and forces_list[idx] is not None:
                pos.requires_grad_(True)
                e_pred, q = model(pos, tc)
                # Compute forces
                forces_pred = -torch.autograd.grad(e_pred, pos, create_graph=False)[0]
                f_ref = torch.tensor(forces_list[idx], dtype=torch.float32)
                f_loss = ((forces_pred - f_ref) ** 2).mean()

                # Recompute without grad for energy
                pos_ng = pos.detach()
                e_pred_ng, _ = model(pos_ng, tc)
                e_loss = ((e_pred_ng - e_ref) ** 2) / (e_std ** 2)
                loss = e_weight * e_loss + f_weight * f_loss
                f_loss_total += f_loss.item()
            else:
                e_pred, q = model(pos, tc)
                e_loss = ((e_pred - e_ref) ** 2) / (e_std ** 2)
                loss = e_weight * e_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            e_loss_total += e_loss.item()

        scheduler.step()

        n = len(positions_list)
        losses['total'].append(total_loss / n)
        losses['energy'].append(e_loss_total / n)
        losses['force'].append(f_loss_total / n)

        if verbose and (epoch + 1) % 25 == 0:
            print(f'  Epoch {epoch+1:4d}: loss={total_loss/n:.4f}, e_loss={e_loss_total/n:.4f}')

    return losses


def evaluate_fast(model, positions_list, energies_ref, total_charges=None, return_charges=False):
    """Evaluate model, return metrics."""
    model.eval()
    if total_charges is None:
        total_charges = [0.0] * len(positions_list)

    preds, all_q = [], []
    with torch.no_grad():
        for i, pos in enumerate(positions_list):
            pos_t = torch.tensor(pos, dtype=torch.float32)
            e, q = model(pos_t, float(total_charges[i]))
            preds.append(e.item())
            if q is not None:
                all_q.append(q.numpy())

    preds = np.array(preds)
    refs = np.array(energies_ref)
    mae = np.mean(np.abs(preds - refs))
    rmse = np.sqrt(np.mean((preds - refs) ** 2))
    var = np.var(refs)
    r2 = 1 - np.mean((preds - refs) ** 2) / (var + 1e-10)

    res = {'energies_pred': preds, 'mae': mae, 'rmse': rmse, 'r2': r2}
    if return_charges:
        res['latent_charges'] = all_q
    return res
