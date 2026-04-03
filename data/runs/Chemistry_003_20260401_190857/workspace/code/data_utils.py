"""
Data loading utilities for LES analysis.
Parses extXYZ files for random_charges, charged_dimer, and ag3_chargestates datasets.
"""
import numpy as np
import re


def parse_extxyz(filename):
    """Parse extended XYZ file into list of frame dicts."""
    frames = []
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                n_atoms = int(line.strip())
            except ValueError:
                break
            comment = f.readline()

            # Parse key-value pairs from comment line
            info = {}
            # energy
            m = re.search(r'energy=([^\s]+)', comment)
            if m:
                info['energy'] = float(m.group(1))
            # charge_state
            m = re.search(r'charge_state=([^\s]+)', comment)
            if m:
                info['charge_state'] = int(m.group(1))
            # total_charge
            m = re.search(r'total_charge=([^\s]+)', comment)
            if m:
                info['total_charge'] = int(m.group(1))
            # true_charges
            m = re.search(r'true_charges="([^"]+)"', comment)
            if m:
                info['true_charges'] = np.array([float(x) for x in m.group(1).split()])

            # Parse atoms
            symbols, positions, forces = [], [], []
            has_forces = 'forces:R:3' in comment
            for _ in range(n_atoms):
                parts = f.readline().split()
                symbols.append(parts[0])
                positions.append([float(x) for x in parts[1:4]])
                if has_forces and len(parts) >= 7:
                    forces.append([float(x) for x in parts[4:7]])

            frame = {
                'symbols': symbols,
                'positions': np.array(positions, dtype=np.float64),
                **info
            }
            if forces:
                frame['forces'] = np.array(forces, dtype=np.float64)

            frames.append(frame)
    return frames


def compute_coulomb_energy(positions, charges, cutoff=None):
    """
    Compute total electrostatic energy E = sum_{i<j} q_i * q_j / r_ij
    (in dimensionless units, k_e = 1)
    """
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if cutoff is None or r < cutoff:
                E += charges[i] * charges[j] / r
    return E


def compute_repulsive_energy(positions, sigma=2.0, epsilon=1.0):
    """
    Compute repulsive LJ energy E = sum_{i<j} epsilon * (sigma/r_ij)^12
    """
    n = len(positions)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            E += epsilon * (sigma / r) ** 12
    return E


def compute_total_energy_rc(positions, charges, sigma=2.0, epsilon=1.0):
    """
    Compute total energy for random charges system:
    E = E_Coulomb + E_repulsive
    Uses vectorized computation for efficiency.
    """
    n = len(positions)
    pos = np.array(positions)
    # Vectorized pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=-1)
    # Avoid self-interaction
    np.fill_diagonal(dist_sq, np.inf)
    dist = np.sqrt(dist_sq)

    # Upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_ij = dist[mask]
    q_i = charges[:, np.newaxis]
    q_j = charges[np.newaxis, :]
    qq = (q_i * q_j)[mask]

    E_coulomb = np.sum(qq / r_ij)
    E_rep = np.sum(epsilon * (sigma / r_ij) ** 12)
    return E_coulomb + E_rep, E_coulomb, E_rep


def compute_coulomb_forces(positions, charges):
    """
    Compute forces from Coulomb potential: F_i = -dE/dr_i
    F_i = sum_{j!=i} q_i * q_j * (r_i - r_j) / |r_i - r_j|^3
    """
    n = len(positions)
    pos = np.array(positions)
    forces = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = pos[i] - pos[j]
            r = np.linalg.norm(r_vec)
            forces[i] += charges[i] * charges[j] * r_vec / r ** 3
    return forces


def pairwise_distances_vectorized(positions):
    """
    Compute all pairwise distances efficiently using numpy.
    Returns (n, n) distance matrix.
    """
    pos = np.array(positions)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist


def get_radial_basis(distances, n_rbf=16, r_min=0.5, r_max=8.0):
    """
    Compute radial basis function (RBF) features for distances.
    Uses Gaussian RBFs centered on a grid from r_min to r_max.

    Args:
        distances: array of distances
        n_rbf: number of RBF centers
        r_min, r_max: range for RBF centers

    Returns:
        RBF features array of shape (len(distances), n_rbf)
    """
    centers = np.linspace(r_min, r_max, n_rbf)
    # Adaptive width
    eta = 0.5 * (r_max - r_min) / n_rbf
    d = np.array(distances)[:, np.newaxis]  # (N, 1)
    rbf = np.exp(-((d - centers) ** 2) / (2 * eta ** 2))
    return rbf


def compute_local_features(positions, cutoff=5.0, n_rbf=16):
    """
    Compute local atomic features by summing RBF-expanded distances over neighbors.

    Args:
        positions: (N, 3) atomic positions
        cutoff: neighbor cutoff distance
        n_rbf: number of RBF basis functions

    Returns:
        features: (N, n_rbf) local feature matrix
    """
    n = len(positions)
    dist_mat = pairwise_distances_vectorized(positions)
    features = np.zeros((n, n_rbf))

    for i in range(n):
        # Find neighbors within cutoff (exclude self)
        neighbor_dists = dist_mat[i]
        mask = (neighbor_dists > 0) & (neighbor_dists < cutoff)
        nbr_dists = neighbor_dists[mask]

        if len(nbr_dists) > 0:
            # Smooth cutoff function
            smooth = 0.5 * (1 + np.cos(np.pi * nbr_dists / cutoff))
            rbf = get_radial_basis(nbr_dists, n_rbf=n_rbf, r_max=cutoff)
            features[i] = np.sum(rbf * smooth[:, np.newaxis], axis=0)

    return features


def get_dimer_separation(positions, n_atoms_per_monomer=4):
    """
    Compute center-of-mass separation for a dimer.
    """
    com1 = positions[:n_atoms_per_monomer].mean(axis=0)
    com2 = positions[n_atoms_per_monomer:].mean(axis=0)
    return np.linalg.norm(com1 - com2)


if __name__ == '__main__':
    # Quick test
    BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857/data'
    rc = parse_extxyz(f'{BASE}/random_charges.xyz')
    print(f'random_charges: {len(rc)} frames, {len(rc[0]["positions"])} atoms')
    print(f'  true charges unique: {np.unique(rc[0]["true_charges"])}')

    cd = parse_extxyz(f'{BASE}/charged_dimer.xyz')
    print(f'charged_dimer: {len(cd)} frames, {len(cd[0]["positions"])} atoms')
    print(f'  energy range: {min(f["energy"] for f in cd):.4f} to {max(f["energy"] for f in cd):.4f}')

    ag = parse_extxyz(f'{BASE}/ag3_chargestates.xyz')
    print(f'ag3: {len(ag)} frames, {len(ag[0]["positions"])} atoms')
    print(f'  charge states: {sorted(set(f["charge_state"] for f in ag))}')
