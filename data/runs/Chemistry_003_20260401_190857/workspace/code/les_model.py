"""
Latent Ewald Summation (LES) Model Implementation.

Implements a machine-learning interatomic potential that:
1. Computes local atomic features from short-range neighborhood
2. Predicts "latent charges" per atom via neural network
3. Constrains latent charges to sum to total system charge
4. Computes long-range electrostatic energy from latent charges
5. Computes short-range energy correction via neural network

This allows the model to capture long-range electrostatics without
explicit charge equilibration or charge supervision.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class RadialBasisLayer(nn.Module):
    """Learnable RBF expansion of pairwise distances."""

    def __init__(self, n_rbf=16, r_min=0.5, r_max=8.0):
        super().__init__()
        centers = torch.linspace(r_min, r_max, n_rbf)
        self.register_buffer('centers', centers)
        # Adaptive width
        eta = 0.5 * (r_max - r_min) / n_rbf
        self.register_buffer('eta', torch.tensor(eta))
        self.r_max = r_max

    def forward(self, distances):
        """
        Args:
            distances: (N,) tensor of distances
        Returns:
            rbf: (N, n_rbf) RBF features
        """
        d = distances.unsqueeze(-1)  # (N, 1)
        rbf = torch.exp(-((d - self.centers) ** 2) / (2 * self.eta ** 2))
        # Smooth cutoff envelope
        x = distances / self.r_max
        smooth = torch.where(x < 1.0,
                             0.5 * (1.0 + torch.cos(np.pi * x)),
                             torch.zeros_like(x))
        rbf = rbf * smooth.unsqueeze(-1)
        return rbf


class AtomicNetwork(nn.Module):
    """Neural network mapping local features to per-atom scalar output."""

    def __init__(self, n_input, n_hidden=64, n_output=1, n_layers=2):
        super().__init__()
        layers = []
        in_size = n_input
        for _ in range(n_layers):
            layers.extend([nn.Linear(in_size, n_hidden), nn.SiLU()])
            in_size = n_hidden
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ShortRangeModel(nn.Module):
    """
    Short-range-only baseline model.
    Predicts per-atom energy from local RBF features.
    Total energy is sum of atomic energies.
    """

    def __init__(self, n_rbf=16, n_hidden=64, cutoff=5.0):
        super().__init__()
        self.cutoff = cutoff
        self.rbf = RadialBasisLayer(n_rbf=n_rbf, r_max=cutoff)
        self.atomic_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)

    def compute_features(self, positions):
        """
        Compute per-atom features by summing RBF features over neighbors.
        Args:
            positions: (N, 3) tensor
        Returns:
            features: (N, n_rbf) tensor
        """
        n = positions.shape[0]
        # Pairwise distances
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        # Mask: within cutoff, exclude self
        mask = (dist > 0) & (dist < self.cutoff)

        features = torch.zeros(n, self.rbf.centers.shape[0],
                               device=positions.device)
        for i in range(n):
            nbr_dists = dist[i][mask[i]]
            if nbr_dists.numel() > 0:
                rbf_feats = self.rbf(nbr_dists)  # (n_nbrs, n_rbf)
                features[i] = rbf_feats.sum(dim=0)
        return features

    def forward(self, positions, total_charge=0.0):
        features = self.compute_features(positions)
        atomic_e = self.atomic_net(features).squeeze(-1)  # (N,)
        return atomic_e.sum()

    def predict_batch(self, positions_list, total_charges=None):
        """Predict energies for a batch of configurations."""
        energies = []
        for i, pos in enumerate(positions_list):
            pos_t = torch.tensor(pos, dtype=torch.float32)
            with torch.no_grad():
                e = self.forward(pos_t).item()
            energies.append(e)
        return np.array(energies)


class LESModel(nn.Module):
    """
    Latent Ewald Summation model.

    Combines:
    1. Short-range energy from neural network on local features
    2. Long-range Coulomb energy from predicted latent charges

    The latent charges are predicted from local atomic environments
    and constrained to sum to the system's total charge.
    The model discovers charges implicitly from energy supervision.
    """

    def __init__(self, n_rbf=16, n_hidden=64, cutoff=5.0, sr_cutoff=None):
        super().__init__()
        self.cutoff = cutoff
        self.sr_cutoff = sr_cutoff if sr_cutoff is not None else cutoff
        self.rbf = RadialBasisLayer(n_rbf=n_rbf, r_max=cutoff)
        # Network to predict latent charge per atom
        self.charge_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)
        # Network to predict short-range atomic energy
        self.energy_net = AtomicNetwork(n_rbf, n_hidden=n_hidden, n_output=1)

    def compute_features(self, positions):
        """Compute per-atom features from local neighborhood."""
        n = positions.shape[0]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)
        dist = torch.norm(diff, dim=-1)
        mask = (dist > 0) & (dist < self.cutoff)

        features = torch.zeros(n, self.rbf.centers.shape[0],
                               device=positions.device)
        for i in range(n):
            nbr_dists = dist[i][mask[i]]
            if nbr_dists.numel() > 0:
                rbf_feats = self.rbf(nbr_dists)
                features[i] = rbf_feats.sum(dim=0)
        return features

    def predict_latent_charges(self, positions, total_charge=0.0):
        """
        Predict latent charges from atomic positions.
        Charges are constrained to sum to total_charge.
        """
        features = self.compute_features(positions)
        raw_charges = self.charge_net(features).squeeze(-1)  # (N,)

        # Constrain: shift charges so they sum to total_charge
        n = positions.shape[0]
        correction = (total_charge - raw_charges.sum()) / n
        latent_charges = raw_charges + correction
        return latent_charges

    def compute_coulomb_energy(self, positions, charges):
        """
        Compute direct Coulomb energy: E = sum_{i<j} q_i * q_j / r_ij
        """
        n = positions.shape[0]
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        # Mask upper triangle, exclude self
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool,
                                     device=positions.device), diagonal=1)
        r_ij = dist[mask]
        q_i = charges.unsqueeze(1).expand(n, n)
        q_j = charges.unsqueeze(0).expand(n, n)
        qq_ij = (q_i * q_j)[mask]
        return (qq_ij / r_ij).sum()

    def forward(self, positions, total_charge=0.0):
        """
        Compute total energy = E_coulomb(latent charges) + E_short_range

        Args:
            positions: (N, 3) atomic positions tensor
            total_charge: scalar total system charge

        Returns:
            total_energy: scalar energy
            latent_charges: (N,) tensor of predicted charges
        """
        features = self.compute_features(positions)

        # Predict latent charges
        raw_charges = self.charge_net(features).squeeze(-1)
        n = positions.shape[0]
        correction = (total_charge - raw_charges.sum()) / n
        latent_charges = raw_charges + correction

        # Long-range Coulomb energy
        E_lr = self.compute_coulomb_energy(positions, latent_charges)

        # Short-range energy
        E_sr = self.energy_net(features).squeeze(-1).sum()

        return E_lr + E_sr, latent_charges

    def predict_batch(self, positions_list, total_charges=None, return_charges=False):
        """Predict energies (and optionally charges) for a batch of configs."""
        energies = []
        all_charges = []
        if total_charges is None:
            total_charges = [0.0] * len(positions_list)

        for i, pos in enumerate(positions_list):
            pos_t = torch.tensor(pos, dtype=torch.float32)
            with torch.no_grad():
                e, q = self.forward(pos_t, total_charge=float(total_charges[i]))
            energies.append(e.item())
            all_charges.append(q.detach().numpy())

        if return_charges:
            return np.array(energies), all_charges
        return np.array(energies)


def train_model(model, positions_list, energies, forces_list=None,
                total_charges=None, n_epochs=200, lr=1e-3,
                energy_weight=1.0, force_weight=1.0, verbose=True):
    """
    Train a model (ShortRangeModel or LESModel) on energy (and optionally force) data.

    Args:
        model: PyTorch model
        positions_list: list of (N, 3) numpy arrays
        energies: array of reference energies
        forces_list: optional list of (N, 3) force arrays
        total_charges: list of total charges per config (default: all 0)
        n_epochs: number of training epochs
        lr: learning rate
        energy_weight: weight for energy loss
        force_weight: weight for force loss
        verbose: print training progress

    Returns:
        losses: list of (energy_loss, force_loss) per epoch
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    if total_charges is None:
        total_charges = [0.0] * len(positions_list)

    energies_t = torch.tensor(energies, dtype=torch.float32)
    # Normalize energies
    e_mean = energies_t.mean()
    e_std = energies_t.std() + 1e-8

    losses = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        e_loss_total = 0.0
        f_loss_total = 0.0

        indices = np.random.permutation(len(positions_list))
        for idx in indices:
            pos = torch.tensor(positions_list[idx], dtype=torch.float32,
                               requires_grad=(forces_list is not None))
            optimizer.zero_grad()

            if isinstance(model, LESModel):
                e_pred, _ = model(pos, total_charge=float(total_charges[idx]))
            else:
                e_pred = model(pos)

            # Energy loss (normalized)
            e_loss = ((e_pred - energies_t[idx]) ** 2) / (e_std ** 2)
            loss = energy_weight * e_loss

            # Force loss (if available)
            if forces_list is not None and forces_list[idx] is not None:
                pos_grad = torch.tensor(positions_list[idx], dtype=torch.float32,
                                        requires_grad=True)
                if isinstance(model, LESModel):
                    e_pred_grad, _ = model(pos_grad, total_charge=float(total_charges[idx]))
                else:
                    e_pred_grad = model(pos_grad)

                forces_pred = -torch.autograd.grad(
                    e_pred_grad, pos_grad, create_graph=True
                )[0]

                f_ref = torch.tensor(forces_list[idx], dtype=torch.float32)
                f_loss = ((forces_pred - f_ref) ** 2).mean()
                loss = loss + force_weight * f_loss
                f_loss_total += f_loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            e_loss_total += e_loss.item()

        avg_loss = total_loss / len(positions_list)
        avg_e_loss = e_loss_total / len(positions_list)
        scheduler.step(avg_loss)

        losses.append({'total': avg_loss, 'energy': avg_e_loss, 'force': f_loss_total / len(positions_list)})

        if verbose and (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1:4d}: loss={avg_loss:.4f}, e_loss={avg_e_loss:.4f}')

    return losses


def evaluate_model(model, positions_list, energies_ref, total_charges=None, return_charges=False):
    """
    Evaluate model predictions and return metrics.
    """
    model.eval()
    if total_charges is None:
        total_charges = [0.0] * len(positions_list)

    energies_pred = []
    all_charges = []

    with torch.no_grad():
        for i, pos in enumerate(positions_list):
            pos_t = torch.tensor(pos, dtype=torch.float32)
            if isinstance(model, LESModel):
                e, q = model(pos_t, total_charge=float(total_charges[i]))
                all_charges.append(q.numpy())
            else:
                e = model(pos_t)
            energies_pred.append(e.item())

    energies_pred = np.array(energies_pred)
    energies_ref = np.array(energies_ref)

    mae = np.mean(np.abs(energies_pred - energies_ref))
    rmse = np.sqrt(np.mean((energies_pred - energies_ref) ** 2))
    r2 = 1 - np.sum((energies_pred - energies_ref) ** 2) / np.sum((energies_ref - energies_ref.mean()) ** 2)

    result = {
        'energies_pred': energies_pred,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    if return_charges:
        result['latent_charges'] = all_charges
    return result
