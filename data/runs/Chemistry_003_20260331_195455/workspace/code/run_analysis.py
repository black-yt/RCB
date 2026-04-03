import os
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from ase.io import read
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="talk", style="whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FIG_DIR = "report/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_xyz(path):
    return read(path, index=":")


class RandomChargesDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures
        self.n_atoms = len(structures[0])
        charges = structures[0].info.get("true_charges")
        if charges is None:
            raise RuntimeError("true_charges field missing in random_charges.xyz")
        self.q_true = np.asarray(charges, dtype=float)

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        at = self.structures[idx]
        pos = at.get_positions().astype(np.float32)
        energy = float(at.info.get("energy", 0.0))
        forces = at.get_forces() if at.has("forces") else np.zeros_like(pos)
        return {
            "pos": pos,
            "energy": np.array([energy], dtype=np.float32),
            "forces": forces.astype(np.float32),
        }


def coulomb_energy_forces(pos, q, epsilon=1.0, detach=False):
    dtype = torch.get_default_dtype()
    pos = pos.to(dtype)
    q = q.to(dtype)
    N = pos.shape[0]
    rij = pos[None, :, :] - pos[:, None, :]
    dist = torch.linalg.norm(rij, dim=-1) + torch.eye(N) * 1e-6
    mask = ~torch.eye(N, dtype=bool)
    qq = q[:, None] * q[None, :]
    inv_r = torch.zeros_like(dist)
    inv_r[mask] = 1.0 / dist[mask]
    energy = 0.5 * (qq * inv_r).sum() / epsilon
    inv_r3 = torch.zeros_like(dist)
    inv_r3[mask] = inv_r[mask] ** 3
    fij = qq[..., None] / epsilon * inv_r3[..., None] * rij
    forces = fij.sum(dim=1)
    if detach:
        energy_out = energy.detach().cpu().numpy().astype(np.float32)
        forces_out = forces.detach().cpu().numpy().astype(np.float32)
        return energy_out, forces_out
    return energy, forces


class LatentChargeModel(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(n_atoms))

    def forward(self, pos):
        B, N, _ = pos.shape
        q = self.q
        device = pos.device
        energy = []
        forces = []
        for b in range(B):
            e, f = coulomb_energy_forces(pos[b], q)
            energy.append(e)
            forces.append(f)
        energy = torch.stack(energy)
        forces = torch.stack(forces)
        return energy, forces


class ChargedDimerDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        at = self.structures[idx]
        pos = at.get_positions().astype(np.float32)
        energy = float(at.info.get("energy", 0.0))
        return {"pos": pos, "energy": np.array([energy], dtype=np.float32)}


class Ag3ChargeStatesDataset(Dataset):
    def __init__(self, structures):
        self.structures = structures

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        at = self.structures[idx]
        pos = at.get_positions().astype(np.float32)
        energy = float(at.info.get("energy", 0.0))
        charge_state = float(at.info.get("total_charge", 0.0))
        return {
            "pos": pos,
            "energy": np.array([energy], dtype=np.float32),
            "charge": np.array([charge_state], dtype=np.float32),
        }


def plot_random_charges_overview(structures):
    at0 = structures[0]
    pos = at0.get_positions()
    charges = np.asarray(at0.info["true_charges"], dtype=float)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=charges, cmap="coolwarm", s=20)
    fig.colorbar(sc, ax=ax, label="True charge (e)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Random charges configuration (frame 0)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "random_charges_overview.png"), dpi=200)
    plt.close(fig)


def train_latent_charge_model(random_structures, n_epochs=50, lr=0.05):
    dataset = RandomChargesDataset(random_structures)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    n_atoms = dataset.n_atoms
    model = LatentChargeModel(n_atoms).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch in loader:
            pos = batch["pos"].to(device)
            energy_true = batch["energy"].to(device).squeeze(-1)
            energy_pred, _ = model(pos)
            loss = ((energy_pred - energy_true) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * pos.shape[0]
        epoch_loss /= len(dataset)
        losses.append(epoch_loss)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss (energy)")
    ax.set_title("Latent-charge model training on random_charges")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "latent_charge_training_loss.png"), dpi=200)
    plt.close(fig)

    q_est = model.q.detach().cpu().numpy()
    q_true = dataset.q_true
    np.save(os.path.join(OUTPUT_DIR, "latent_q_est.npy"), q_est)
    np.save(os.path.join(OUTPUT_DIR, "latent_q_true.npy"), q_true)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(q_true, q_est, s=20)
    ax.plot([-1, 1], [-1, 1], "k--", lw=1)
    ax.set_xlabel("True charge (e)")
    ax.set_ylabel("Recovered latent charge (a.u.)")
    ax.set_title("Charge recovery on random_charges")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "latent_charge_scatter.png"), dpi=200)
    plt.close(fig)


def compute_dimer_separation(at):
    pos = at.get_positions()
    n = len(pos)
    half = n // 2
    com1 = pos[:half].mean(axis=0)
    com2 = pos[half:].mean(axis=0)
    return np.linalg.norm(com1 - com2)


def analyze_charged_dimer(structures):
    separations = []
    energies = []
    for at in structures:
        separations.append(compute_dimer_separation(at))
        energies.append(float(at.info.get("energy", 0.0)))
    separations = np.array(separations)
    energies = np.array(energies)
    order = np.argsort(separations)
    separations = separations[order]
    energies = energies[order]
    np.savez(
        os.path.join(OUTPUT_DIR, "charged_dimer_curve.npz"), r=separations, E=energies
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(separations, energies, "o-", ms=3)
    ax.set_xlabel("Inter-dimer separation (Å)")
    ax.set_ylabel("Total energy (a.u.)")
    ax.set_title("Charged dimer binding curve (reference)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "charged_dimer_binding_curve.png"), dpi=200)
    plt.close(fig)


class Ag3LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1 + 1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, bond_length, charge):
        x = torch.cat([bond_length, charge], dim=-1)
        return self.fc(x)


def ag3_features(structures):
    bond_lengths = []
    energies = []
    charges = []
    for at in structures:
        pos = at.get_positions()
        dists = []
        for i in range(3):
            for j in range(i + 1, 3):
                dists.append(np.linalg.norm(pos[i] - pos[j]))
        bond_lengths.append(np.mean(dists))
        energies.append(float(at.info.get("energy", 0.0)))
        charges.append(float(at.info.get("total_charge", 0.0)))
    return np.array(bond_lengths), np.array(energies), np.array(charges)


def analyze_ag3_charge_states(structures):
    r, E, Q = ag3_features(structures)
    np.savez(os.path.join(OUTPUT_DIR, "ag3_features.npz"), r=r, E=E, Q=Q)

    fig, ax = plt.subplots(figsize=(6, 4))
    for q in np.unique(Q):
        m = Q == q
        ax.scatter(r[m], E[m], label=f"Q={q:+.0f}", s=15)
    ax.set_xlabel("Mean Ag–Ag bond length (Å)")
    ax.set_ylabel("Total energy (a.u.)")
    ax.set_title("Ag3 potential energy surfaces by charge state")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "ag3_pes_scatter.png"), dpi=200)
    plt.close(fig)

    r_t = torch.from_numpy(r[:, None]).float()
    Q_t = torch.from_numpy(Q[:, None]).float()
    E_t = torch.from_numpy(E[:, None]).float()

    model = Ag3LinearModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for epoch in range(500):
        pred = model(r_t, Q_t)
        loss = ((pred - E_t) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    E_pred = model(r_t, Q_t).detach().numpy().squeeze()

    model2 = Ag3LinearModel()
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
    for epoch in range(500):
        pred2 = model2(r_t, torch.zeros_like(Q_t))
        loss2 = ((pred2 - E_t) ** 2).mean()
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
    E_pred2 = model2(r_t, torch.zeros_like(Q_t)).detach().numpy().squeeze()

    rmse_with = float(np.sqrt(np.mean((E_pred - E) ** 2)))
    rmse_without = float(np.sqrt(np.mean((E_pred2 - E) ** 2)))
    with open(os.path.join(OUTPUT_DIR, "ag3_rmse.json"), "w") as f:
        json.dump(
            {
                "rmse_with_charge": rmse_with,
                "rmse_without_charge": rmse_without,
            },
            f,
            indent=2,
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    order = np.argsort(r)
    ax.plot(r[order], E[order], "k.", label="Reference", ms=4)
    ax.plot(r[order], E_pred[order], "-", label="Model with charge")
    ax.plot(r[order], E_pred2[order], "--", label="Model without charge")
    ax.set_xlabel("Mean Ag–Ag bond length (Å)")
    ax.set_ylabel("Energy (a.u.)")
    ax.set_title("Effect of global charge embedding for Ag3")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, "ag3_charge_embedding_comparison.png"), dpi=200
    )
    plt.close(fig)


def main():
    random_structures = load_xyz(os.path.join(DATA_DIR, "random_charges.xyz"))
    charged_dimer_structures = load_xyz(os.path.join(DATA_DIR, "charged_dimer.xyz"))
    ag3_structures = load_xyz(os.path.join(DATA_DIR, "ag3_chargestates.xyz"))

    plot_random_charges_overview(random_structures)
    train_latent_charge_model(random_structures, n_epochs=50, lr=0.05)
    analyze_charged_dimer(charged_dimer_structures)
    analyze_ag3_charge_states(ag3_structures)


if __name__ == "__main__":
    main()
