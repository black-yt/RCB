"""
Complete LES Analysis - All Three Datasets
Optimized for speed: no force computation, energy-only training,
pre-computed features and Coulomb matrices.
"""
import numpy as np
import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from data_utils import parse_extxyz, compute_total_energy_rc, get_dimer_separation, pairwise_distances_vectorized
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import matplotlib.pyplot as plt
import time

BASE = '/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_003_20260401_190857'
DATA_DIR = f'{BASE}/data'
OUT_DIR = f'{BASE}/outputs'
IMG_DIR = f'{BASE}/report/images'

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def rbf_features(pos, cutoff, n_rbf=16):
    """Compute RBF-summed local features. pos: (N,3) numpy array → (N,n_rbf)."""
    N = len(pos)
    centers = np.linspace(0.5, cutoff, n_rbf)
    eta = 0.5 * cutoff / n_rbf
    features = np.zeros((N, n_rbf))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r = np.linalg.norm(pos[i] - pos[j])
            if r < cutoff:
                smooth = 0.5 * (1 + np.cos(np.pi * r / cutoff))
                rbf = np.exp(-((r - centers) ** 2) / (2 * eta ** 2))
                features[i] += smooth * rbf
    return features


def rbf_features_vectorized(pos, cutoff, n_rbf=16):
    """Vectorized RBF features for a single config. pos: (N,3) → (N,n_rbf)."""
    N = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N,N,3)
    dist = np.sqrt(np.sum(diff**2, axis=-1))  # (N,N)
    np.fill_diagonal(dist, np.inf)

    centers = np.linspace(0.5, cutoff, n_rbf)
    eta = 0.5 * cutoff / n_rbf

    # For each pair within cutoff
    mask = dist < cutoff
    dist_masked = dist.copy()
    dist_masked[~mask] = 0.0

    features = np.zeros((N, n_rbf))
    for i in range(N):
        d_i = dist_masked[i][mask[i]]  # (n_nbr,)
        if len(d_i) == 0:
            continue
        smooth = 0.5 * (1 + np.cos(np.pi * d_i / cutoff))
        rbf = np.exp(-((d_i[:, None] - centers[None, :]) ** 2) / (2 * eta ** 2))
        features[i] = (smooth[:, None] * rbf).sum(axis=0)
    return features


def coulomb_matrix(pos):
    """Compute (N,N) Coulomb matrix: C_ij = 1/r_ij (off-diagonal), 0 diagonal."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    with np.errstate(divide='ignore'):
        C = np.where(dist > 1e-10, 1.0 / dist, 0.0)
    np.fill_diagonal(C, 0.0)
    return C


# ─── NEURAL NETWORK MODELS ───────────────────────────────────────────────────

class SRModel(nn.Module):
    """Short-range model: features → per-atom energy."""
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, features_t):
        return self.net(features_t).sum()


class LESModel(nn.Module):
    """LES model: features → latent charges → Coulomb + SR energy."""
    def __init__(self, n_rbf=16, n_hidden=32):
        super().__init__()
        self.charge_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
        self.sr_net = nn.Sequential(
            nn.Linear(n_rbf, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, features_t, C_t, lj_energy, total_charge=0.0):
        """
        features_t: (N, n_rbf)
        C_t: (N, N) Coulomb matrix (precomputed)
        lj_energy: scalar
        """
        N = features_t.shape[0]
        raw_q = self.charge_net(features_t).squeeze(-1)
        # Charge constraint
        correction = (total_charge - raw_q.sum()) / N
        q = raw_q + correction
        # Coulomb energy E = 0.5 * q^T C q
        E_lr = 0.5 * (q @ (C_t @ q))
        E_sr = self.sr_net(features_t).sum()
        return E_lr + E_sr + lj_energy, q


class CSEmbedModel(nn.Module):
    """Model with explicit charge state embedding (for Ag3)."""
    def __init__(self, n_rbf=16, n_hidden=32, embed_dim=8):
        super().__init__()
        self.embed = nn.Embedding(2, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(n_rbf + embed_dim, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, n_hidden), nn.Tanh(),
            nn.Linear(n_hidden, 1)
        )
    def forward(self, features_t, charge_state=1):
        cs_idx = torch.tensor(0 if charge_state < 0 else 1)
        embed = self.embed(cs_idx).unsqueeze(0).expand(features_t.shape[0], -1)
        x = torch.cat([features_t, embed], dim=-1)
        return self.net(x).sum(), None


# ─── TRAINING (BATCHED - one backward per epoch for speed) ───────────────────

def _make_sched(opt, n_epochs):
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)


def train_sr_batch(model, d_list, energies, n_epochs=200, lr=1e-3, verbose=True):
    """Batch training for SRModel: one forward-backward per epoch."""
    features_stack = torch.stack([d['features'] for d in d_list])  # (N, Natom, nrbf)
    N, Natom, nrbf = features_stack.shape
    e_ref = torch.tensor(energies, dtype=torch.float32)
    e_std = e_ref.std().item() + 1e-8
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_sched(opt, n_epochs)
    losses = []
    for epoch in range(n_epochs):
        model.train(); opt.zero_grad()
        e_pred = model.net(features_stack.reshape(-1, nrbf)).reshape(N, Natom).sum(1)
        loss = ((e_pred - e_ref) ** 2).mean() / (e_std ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f'    Epoch {epoch+1:4d}: loss={loss.item():.6f}', flush=True)
    return losses


def train_les_batch(model, d_list, energies, n_epochs=200, lr=1e-3, verbose=True):
    """Batch training for LESModel: stacks features+Coulomb, uses bmm."""
    features_stack = torch.stack([d['features'] for d in d_list])   # (N, Natom, nrbf)
    C_stack = torch.stack([d['C'] for d in d_list])                  # (N, Natom, Natom)
    lj_arr = torch.tensor([d.get('lj', 0.0) for d in d_list], dtype=torch.float32)
    Q_arr = torch.tensor([d.get('total_charge', 0.0) for d in d_list], dtype=torch.float32)
    N, Natom, nrbf = features_stack.shape
    e_ref = torch.tensor(energies, dtype=torch.float32)
    e_std = e_ref.std().item() + 1e-8
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_sched(opt, n_epochs)
    losses = []
    for epoch in range(n_epochs):
        model.train(); opt.zero_grad()
        raw_q = model.charge_net(features_stack.reshape(-1, nrbf)).reshape(N, Natom)
        correction = (Q_arr - raw_q.sum(1)) / Natom
        q = raw_q + correction.unsqueeze(1)                           # (N, Natom)
        Cq = torch.bmm(C_stack, q.unsqueeze(2)).squeeze(2)           # (N, Natom)
        E_lr = 0.5 * (q * Cq).sum(1)                                 # (N,)
        E_sr = model.sr_net(features_stack.reshape(-1, nrbf)).reshape(N, Natom).sum(1)
        e_pred = E_lr + E_sr + lj_arr
        loss = ((e_pred - e_ref) ** 2).mean() / (e_std ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f'    Epoch {epoch+1:4d}: loss={loss.item():.6f}', flush=True)
    return losses


def train_cs_batch(model, d_list, energies, n_epochs=200, lr=1e-3, verbose=True):
    """Batch training for CSEmbedModel: batch embedding lookup + net."""
    features_stack = torch.stack([d['features'] for d in d_list])   # (N, Natom, nrbf)
    cs_idx = torch.tensor([0 if d.get('charge_state', 1) < 0 else 1 for d in d_list])
    N, Natom, nrbf = features_stack.shape
    embed_dim = model.embed.embedding_dim
    e_ref = torch.tensor(energies, dtype=torch.float32)
    e_std = e_ref.std().item() + 1e-8
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = _make_sched(opt, n_epochs)
    losses = []
    for epoch in range(n_epochs):
        model.train(); opt.zero_grad()
        embed = model.embed(cs_idx)                                   # (N, embed_dim)
        embed_exp = embed.unsqueeze(1).expand(-1, Natom, -1)          # (N, Natom, embed_dim)
        x = torch.cat([features_stack, embed_exp], dim=-1)           # (N, Natom, nrbf+embed)
        e_pred = model.net(x.reshape(-1, nrbf + embed_dim)).reshape(N, Natom).sum(1)
        loss = ((e_pred - e_ref) ** 2).mean() / (e_std ** 2)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        if verbose and (epoch + 1) % 50 == 0:
            print(f'    Epoch {epoch+1:4d}: loss={loss.item():.6f}', flush=True)
    return losses


def forward_from_data(model, dat):
    """Dispatch forward based on model type (used in evaluate_model)."""
    if isinstance(model, LESModel):
        e, _ = model(dat['features'], dat['C'], dat.get('lj', 0.0), dat.get('total_charge', 0.0))
        return e
    elif isinstance(model, CSEmbedModel):
        e, _ = model(dat['features'], dat.get('charge_state', 1))
        return e
    else:  # SRModel
        return model(dat['features'])


def evaluate_model(model, data_list, energies, return_charges=False):
    """Evaluate model on data_list."""
    model.eval()
    preds, all_q = [], []
    with torch.no_grad():
        for dat in data_list:
            if isinstance(model, LESModel):
                e, q = model(dat['features'], dat['C'], dat.get('lj', 0.0), dat.get('total_charge', 0.0))
                if return_charges:
                    all_q.append(q.numpy().copy())
            elif isinstance(model, CSEmbedModel):
                e, _ = model(dat['features'], dat.get('charge_state', 1))
            else:
                e = model(dat['features'])
            preds.append(e.item())

    preds = np.array(preds)
    refs = np.array(energies)
    mae = np.mean(np.abs(preds - refs))
    rmse = np.sqrt(np.mean((preds - refs)**2))
    r2 = 1 - np.mean((preds-refs)**2) / (np.var(refs) + 1e-10)
    res = {'energies_pred': preds, 'mae': mae, 'rmse': rmse, 'r2': r2}
    if return_charges:
        res['latent_charges'] = all_q
    return res


# ─── ANALYSIS 1: RANDOM CHARGES ─────────────────────────────────────────────

def run_random_charges():
    print("\n" + "="*60)
    print("ANALYSIS 1: Random Charges Benchmark")
    print("="*60)

    frames = parse_extxyz(f'{DATA_DIR}/random_charges.xyz')
    N = len(frames)
    print(f"  {N} configs of {len(frames[0]['positions'])} atoms")

    # Compute reference energies
    print("\n  Computing Coulomb+LJ energies...")
    sigma, eps = 2.0, 1.0
    energies, e_coul, e_rep = [], [], []
    for fr in frames:
        e, ec, er = compute_total_energy_rc(fr['positions'], fr['true_charges'], sigma, eps)
        energies.append(e); e_coul.append(ec); e_rep.append(er)
    energies = np.array(energies); e_coul = np.array(e_coul); e_rep = np.array(e_rep)
    print(f"  Energy range: {energies.min():.1f} to {energies.max():.1f}")

    # Pre-compute features and Coulomb matrices
    print("\n  Pre-computing features and Coulomb matrices...")
    t0 = time.time()
    all_data = []
    for i, fr in enumerate(frames):
        pos = fr['positions']
        feats = rbf_features_vectorized(pos, cutoff=6.0, n_rbf=16)
        C = coulomb_matrix(pos)
        lj_e = float(e_rep[i])
        all_data.append({
            'features': torch.tensor(feats, dtype=torch.float32),
            'C': torch.tensor(C, dtype=torch.float32),
            'lj': lj_e,
            'total_charge': 0.0
        })
        if (i+1) % 25 == 0:
            print(f'    {i+1}/{N} done ({time.time()-t0:.1f}s)')
    print(f"  Precomputation: {time.time()-t0:.1f}s")

    # Train/test split
    idx = np.random.permutation(N)
    n_tr = 70
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    d_tr = [all_data[i] for i in tr_idx]
    d_te = [all_data[i] for i in te_idx]
    e_tr, e_te = energies[tr_idx], energies[te_idx]
    tc_tr = [frames[i]['true_charges'] for i in tr_idx]

    # Train SR model
    print("\n  Training SR model...")
    sr = SRModel(n_rbf=16, n_hidden=32)
    sr_losses = train_sr_batch(sr, d_tr, e_tr, n_epochs=300, lr=1e-3, verbose=True)
    sr_tr_res = evaluate_model(sr, d_tr, e_tr)
    sr_te_res = evaluate_model(sr, d_te, e_te)
    print(f"  SR Train MAE={sr_tr_res['mae']:.3f}, R²={sr_tr_res['r2']:.4f}")
    print(f"  SR Test  MAE={sr_te_res['mae']:.3f}, R²={sr_te_res['r2']:.4f}")

    # Train LES model
    print("\n  Training LES model...")
    les = LESModel(n_rbf=16, n_hidden=32)
    les_losses = train_les_batch(les, d_tr, e_tr, n_epochs=300, lr=1e-3, verbose=True)
    les_tr_res = evaluate_model(les, d_tr, e_tr, return_charges=True)
    les_te_res = evaluate_model(les, d_te, e_te, return_charges=True)
    print(f"  LES Train MAE={les_tr_res['mae']:.3f}, R²={les_tr_res['r2']:.4f}")
    print(f"  LES Test  MAE={les_te_res['mae']:.3f}, R²={les_te_res['r2']:.4f}")

    # Save results
    np.save(f'{OUT_DIR}/rc_results.npy', {
        'sr_test': sr_te_res, 'les_test': les_te_res,
        'sr_train': sr_tr_res, 'les_train': les_tr_res,
        'sr_losses': sr_losses, 'les_losses': les_losses,
        'energies': energies, 'e_coul': e_coul, 'e_rep': e_rep,
        'true_charges_train': tc_tr
    }, allow_pickle=True)

    # Figures
    print("\n  Generating figures...")
    _fig_rc_overview(frames, e_coul, e_rep)
    _fig_rc_parity(sr_te_res, les_te_res, e_te)
    _fig_rc_charges(les_tr_res, tc_tr)
    _fig_rc_training(sr_losses, les_losses)

    print(f"\n  RESULTS:")
    print(f"  SR  MAE={sr_te_res['mae']:.3f}, R²={sr_te_res['r2']:.4f}")
    print(f"  LES MAE={les_te_res['mae']:.3f}, R²={les_te_res['r2']:.4f}")
    return sr_te_res, les_te_res, e_coul, e_rep, les_tr_res, tc_tr


# ─── ANALYSIS 2: CHARGED DIMER ───────────────────────────────────────────────

def run_charged_dimer():
    print("\n" + "="*60)
    print("ANALYSIS 2: Charged Dimer Binding Curves")
    print("="*60)

    frames = parse_extxyz(f'{DATA_DIR}/charged_dimer.xyz')
    positions_list = [fr['positions'] for fr in frames]
    energies = np.array([fr['energy'] for fr in frames])
    separations = np.array([get_dimer_separation(fr['positions'], 4) for fr in frames])
    print(f"  {len(frames)} configs; sep {separations.min():.2f}–{separations.max():.2f} Å")

    # Pre-compute features (cutoff 4.5Å - within each monomer, not across)
    print("\n  Pre-computing features...")
    SR_CUTOFF = 4.0  # short-range cutoff - doesn't bridge dimer gap at large sep
    LES_CUTOFF = 4.0  # LES uses same local cutoff for features, but Coulomb is global
    all_feats = [rbf_features_vectorized(fr['positions'], SR_CUTOFF, n_rbf=16) for fr in frames]
    all_C = [coulomb_matrix(fr['positions']) for fr in frames]

    d_all_sr = [{'features': torch.tensor(f, dtype=torch.float32)} for f in all_feats]
    d_all_les = [{'features': torch.tensor(f, dtype=torch.float32),
                  'C': torch.tensor(c, dtype=torch.float32), 'lj': 0.0}
                 for f, c in zip(all_feats, all_C)]

    # Train SR
    print("\n  Training SR model (energy only)...")
    sr = SRModel(n_rbf=16, n_hidden=32)
    sr_l = train_sr_batch(sr, d_all_sr, energies, n_epochs=300, lr=1e-3, verbose=True)
    sr_res = evaluate_model(sr, d_all_sr, energies)
    print(f"  SR  MAE={sr_res['mae']:.4f}, R²={sr_res['r2']:.4f}")

    # Train LES
    print("\n  Training LES model...")
    les = LESModel(n_rbf=16, n_hidden=32)
    les_l = train_les_batch(les, d_all_les, energies, n_epochs=300, lr=1e-3, verbose=True)
    les_res = evaluate_model(les, d_all_les, energies, return_charges=True)
    print(f"  LES MAE={les_res['mae']:.4f}, R²={les_res['r2']:.4f}")

    # Save
    np.save(f'{OUT_DIR}/dimer_results.npy', {
        'sr': sr_res, 'les': les_res, 'separations': separations,
        'energies': energies, 'sr_losses': sr_l, 'les_losses': les_l
    }, allow_pickle=True)

    # Figures
    print("\n  Generating figures...")
    _fig_dimer(frames, separations, energies, sr_res, les_res)

    print(f"\n  RESULTS:")
    print(f"  SR  MAE={sr_res['mae']:.4f} eV, R²={sr_res['r2']:.4f}")
    print(f"  LES MAE={les_res['mae']:.4f} eV, R²={les_res['r2']:.4f}")
    return sr_res, les_res, separations, energies


# ─── ANALYSIS 3: AG3 CHARGE STATES ──────────────────────────────────────────

def run_ag3():
    print("\n" + "="*60)
    print("ANALYSIS 3: Ag3 Charge State Discrimination")
    print("="*60)

    frames = parse_extxyz(f'{DATA_DIR}/ag3_chargestates.xyz')
    positions_list = [fr['positions'] for fr in frames]
    energies = np.array([fr['energy'] for fr in frames])
    charge_states = np.array([fr['charge_state'] for fr in frames])
    print(f"  {len(frames)} configs: +1: {(charge_states==1).sum()}, -1: {(charge_states==-1).sum()}")

    # Pre-compute
    print("\n  Pre-computing features and Coulomb matrices...")
    all_feats = [rbf_features_vectorized(fr['positions'], 5.0, 16) for fr in frames]
    all_C = [coulomb_matrix(fr['positions']) for fr in frames]

    # Data dicts for three models
    d_sr = [{'features': torch.tensor(f, dtype=torch.float32)} for f in all_feats]
    d_cs = [{'features': torch.tensor(f, dtype=torch.float32), 'charge_state': int(cs)}
            for f, cs in zip(all_feats, charge_states)]
    d_les = [{'features': torch.tensor(f, dtype=torch.float32),
              'C': torch.tensor(c, dtype=torch.float32), 'lj': 0.0,
              'total_charge': float(cs)}
             for f, c, cs in zip(all_feats, all_C, charge_states)]

    # Train/test split (stratified)
    from sklearn.model_selection import train_test_split as tts
    tr_i, te_i = tts(np.arange(len(frames)), test_size=0.25, stratify=charge_states, random_state=42)

    def split(d): return [d[i] for i in tr_i], [d[i] for i in te_i]
    d_sr_tr, d_sr_te = split(d_sr)
    d_cs_tr, d_cs_te = split(d_cs)
    d_les_tr, d_les_te = split(d_les)
    e_tr, e_te = energies[tr_i], energies[te_i]
    cs_te = charge_states[te_i]
    cs_tr = charge_states[tr_i]

    # SR model
    print("\n  Training SR model (no charge state)...")
    sr = SRModel(n_rbf=16, n_hidden=32)
    sr_l = train_sr_batch(sr, d_sr_tr, e_tr, n_epochs=400, lr=1e-3, verbose=True)
    sr_tr_r = evaluate_model(sr, d_sr_tr, e_tr)
    sr_te_r = evaluate_model(sr, d_sr_te, e_te)
    print(f"  SR  Train MAE={sr_tr_r['mae']:.4f}, Test MAE={sr_te_r['mae']:.4f}")

    # CS embed model
    print("\n  Training CS Embedding model...")
    cs_m = CSEmbedModel(n_rbf=16, n_hidden=32)
    cs_l = train_cs_batch(cs_m, d_cs_tr, e_tr, n_epochs=400, lr=1e-3, verbose=True)
    cs_tr_r = evaluate_model(cs_m, d_cs_tr, e_tr)
    cs_te_r = evaluate_model(cs_m, d_cs_te, e_te)
    print(f"  CS  Train MAE={cs_tr_r['mae']:.4f}, Test MAE={cs_te_r['mae']:.4f}")

    # LES model
    print("\n  Training LES model...")
    les = LESModel(n_rbf=16, n_hidden=32)
    les_l = train_les_batch(les, d_les_tr, e_tr, n_epochs=400, lr=1e-3, verbose=True)
    les_tr_r = evaluate_model(les, d_les_tr, e_tr, return_charges=True)
    les_te_r = evaluate_model(les, d_les_te, e_te, return_charges=True)
    print(f"  LES Train MAE={les_tr_r['mae']:.4f}, Test MAE={les_te_r['mae']:.4f}")

    # Save
    np.save(f'{OUT_DIR}/ag3_results.npy', {
        'sr_test': sr_te_r, 'cs_test': cs_te_r, 'les_test': les_te_r,
        'sr_train': sr_tr_r, 'cs_train': cs_tr_r, 'les_train': les_tr_r,
        'charge_states_test': cs_te, 'charge_states_train': cs_tr,
        'e_te': e_te, 'sr_losses': sr_l, 'cs_losses': cs_l, 'les_losses': les_l
    }, allow_pickle=True)

    # Figures
    print("\n  Generating figures...")
    _fig_ag3_overview(frames, energies, charge_states)
    _fig_ag3_parity(sr_te_r, cs_te_r, les_te_r, e_te, cs_te)
    _fig_ag3_latent(les_tr_r, cs_tr)
    _fig_ag3_training(sr_l, cs_l, les_l)
    _fig_ag3_bar(sr_te_r, cs_te_r, les_te_r)

    print(f"\n  RESULTS:")
    for name, res in [('SR', sr_te_r), ('CS', cs_te_r), ('LES', les_te_r)]:
        print(f"  {name}: MAE={res['mae']:.4f} eV, R²={res['r2']:.4f}")

    return sr_te_r, cs_te_r, les_te_r


# ─── FIGURE FUNCTIONS ────────────────────────────────────────────────────────

def _fig_rc_overview(frames, e_coul, e_rep):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # (a) structure
    ax = axes[0]
    pos = frames[0]['positions']
    tc = frames[0]['true_charges']
    ax.scatter(pos[tc > 0, 0], pos[tc > 0, 1], c='crimson', s=25, alpha=0.6, label='+1e')
    ax.scatter(pos[tc < 0, 0], pos[tc < 0, 1], c='royalblue', s=25, alpha=0.6, label='−1e')
    ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_title('(a) Random Charges Config\n(128 atoms, x-y projection)')
    ax.legend(fontsize=9); ax.set_aspect('equal')

    # (b) pairwise distance distribution for like/unlike charges
    ax = axes[1]
    pos = frames[0]['positions']; tc = frames[0]['true_charges']
    dm = pairwise_distances_vectorized(pos)
    mask_up = np.triu(np.ones_like(dm, dtype=bool), k=1)
    r_all = dm[mask_up]
    qq = (tc[:, None] * tc[None, :])[mask_up]
    ax.hist(r_all[qq > 0], bins=40, alpha=0.6, color='crimson', density=True, label='Same charge')
    ax.hist(r_all[qq < 0], bins=40, alpha=0.6, color='royalblue', density=True, label='Opp. charge')
    ax.set_xlabel('Pair Distance r (Å)'); ax.set_ylabel('Density')
    ax.set_title('(b) Pairwise Distance Distribution\nby Charge Product Sign'); ax.legend()

    # (c) energy distributions
    ax = axes[2]
    ax.hist(e_coul, bins=20, alpha=0.7, color='purple', density=True, label='Coulomb')
    ax.hist(e_rep, bins=20, alpha=0.7, color='darkorange', density=True, label='LJ Repulsion')
    ax.set_xlabel('Energy (a.u.)'); ax.set_ylabel('Density')
    ax.set_title('(c) Energy Components\n(100 configurations)'); ax.legend()

    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig01_rc_overview.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig01_rc_overview.png')


def _fig_rc_parity(sr_te, les_te, e_true):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, res, title, color in zip(axes, [sr_te, les_te],
                                      ['Short-Range Model', 'LES Model'],
                                      ['steelblue', 'tomato']):
        ep = res['energies_pred']
        lim = [min(e_true.min(), ep.min())-3, max(e_true.max(), ep.max())+3]
        ax.scatter(e_true, ep, c=color, s=60, alpha=0.8, edgecolors='k', lw=0.3, zorder=5)
        ax.plot(lim, lim, 'k--', lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel('Reference Energy (a.u.)'); ax.set_ylabel('Predicted Energy (a.u.)')
        ax.set_title(title)
        ax.text(0.05, 0.95, f'MAE = {res["mae"]:.2f}\nR² = {res["r2"]:.4f}',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    plt.suptitle('Random Charges: Energy Parity (Test Set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig02_rc_parity.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig02_rc_parity.png')


def _fig_rc_charges(les_tr, true_charges_train):
    if not les_tr.get('latent_charges'):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    all_lq = np.concatenate(les_tr['latent_charges'])
    all_tq = np.concatenate(true_charges_train)
    qp = all_lq[all_tq > 0]; qm = all_lq[all_tq < 0]

    ax = axes[0]
    ax.hist(qp, bins=50, alpha=0.65, color='crimson', density=True, label=f'True +1e (n={len(qp)})')
    ax.hist(qm, bins=50, alpha=0.65, color='royalblue', density=True, label=f'True −1e (n={len(qm)})')
    d = abs(qp.mean()-qm.mean())/(0.5*(qp.std()+qm.std())+1e-8)
    ax.set_xlabel('LES Latent Charge (a.u.)'); ax.set_ylabel('Density')
    ax.set_title('(a) Latent Charge Distribution\nGrouped by True Charge')
    ax.text(0.05, 0.95, f"d' = {d:.2f} (separation)", transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))
    ax.legend()

    ax = axes[1]
    ns = min(300, len(qp))
    ip = np.random.choice(len(qp), ns, replace=False)
    im = np.random.choice(len(qm), ns, replace=False)
    ax.scatter(np.ones(ns)+np.random.randn(ns)*0.05, qp[ip], c='crimson', alpha=0.15, s=5)
    ax.scatter(-np.ones(ns)+np.random.randn(ns)*0.05, qm[im], c='royalblue', alpha=0.15, s=5)
    ax.errorbar([1], [qp.mean()], yerr=[qp.std()], fmt='rs', ms=12, capsize=8, lw=2)
    ax.errorbar([-1], [qm.mean()], yerr=[qm.std()], fmt='b^', ms=12, capsize=8, lw=2)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_xticks([-1, 1]); ax.set_xticklabels(['True −1e', 'True +1e'])
    ax.set_ylabel('LES Latent Charge (a.u.)')
    ax.set_title('(b) Latent Charge per Group\n(points = atoms, mean ± std shown)')

    plt.suptitle('LES Charge Recovery: Trained on Energies Only, No Charge Labels', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig03_rc_latent_charges.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig03_rc_latent_charges.png')


def _fig_rc_training(sr_l, les_l):
    fig, ax = plt.subplots(figsize=(8, 5))
    e = np.arange(1, len(sr_l)+1)
    ax.semilogy(e, sr_l, 'b-', lw=2, label='Short-Range Model')
    ax.semilogy(e, les_l, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)'); ax.grid(True, alpha=0.3)
    ax.set_title('Random Charges: Training Curves'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig04_rc_training.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig04_rc_training.png')


def _fig_dimer(frames, seps, energies, sr_res, les_res):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    sort_i = np.argsort(seps)
    s = seps[sort_i]; e_r = energies[sort_i]
    e_sr = sr_res['energies_pred'][sort_i]; e_les = les_res['energies_pred'][sort_i]

    # Panel 1: Binding curve
    ax = axes[0]
    ax.plot(s, e_r, 'ko-', ms=4, lw=2, label='Reference')
    ax.plot(s, e_sr, 'b^--', ms=5, lw=2, label=f'SR (MAE={sr_res["mae"]:.3f})')
    ax.plot(s, e_les, 'r*-', ms=6, lw=2, label=f'LES (MAE={les_res["mae"]:.3f})')
    ax.axvline(x=4.0, color='gray', ls=':', lw=2, label='SR cutoff')
    ax.set_xlabel('Separation (Å)'); ax.set_ylabel('Energy (eV)')
    ax.set_title('(a) Binding Energy Curve'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2: Absolute error vs separation
    ax = axes[1]
    ax.plot(s, np.abs(e_sr-e_r), 'b^--', ms=5, lw=2, label='SR error')
    ax.plot(s, np.abs(e_les-e_r), 'r*-', ms=6, lw=2, label='LES error')
    ax.axvline(x=4.0, color='gray', ls=':', lw=2)
    ax.set_xlabel('Separation (Å)'); ax.set_ylabel('|Error| (eV)')
    ax.set_title('(b) Prediction Error vs. Separation'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel 3: Parity
    ax = axes[2]
    lim = [energies.min()-0.05, energies.max()+0.05]
    ax.scatter(energies, sr_res['energies_pred'], c='steelblue', s=30, alpha=0.7, label=f'SR R²={sr_res["r2"]:.4f}')
    ax.scatter(energies, les_res['energies_pred'], c='tomato', s=30, alpha=0.7, marker='^', label=f'LES R²={les_res["r2"]:.4f}')
    ax.plot(lim, lim, 'k--', lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
    ax.set_xlabel('Reference Energy (eV)'); ax.set_ylabel('Predicted Energy (eV)')
    ax.set_title('(c) Energy Parity'); ax.legend(fontsize=9)

    plt.suptitle('Charged Dimer: LES vs. Short-Range Model', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig05_dimer_analysis.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig05_dimer_analysis.png')


def _fig_ag3_overview(frames, energies, charge_states):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    # (a)
    ax = axes[0]
    pos = frames[0]['positions']
    for i in range(3):
        for j in range(i+1, 3):
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], 'k-', lw=2.5, zorder=3)
    ax.scatter(pos[:,0], pos[:,1], c='silver', s=400, edgecolors='gray', lw=2, zorder=5)
    ax.set_title('(a) Ag₃ Cluster Geometry'); ax.set_xlabel('x (Å)'); ax.set_ylabel('y (Å)')
    ax.set_aspect('equal'); m = 1.5
    ax.set_xlim(pos[:,0].min()-m, pos[:,0].max()+m); ax.set_ylim(pos[:,1].min()-m, pos[:,1].max()+m)
    # (b)
    ax = axes[1]
    for cs, c in [(1,'crimson'),(-1,'royalblue')]:
        ax.hist(energies[charge_states==cs], bins=12, alpha=0.65, color=c, density=True, label=f'q={cs:+d}e')
    ax.set_xlabel('Energy (eV)'); ax.set_ylabel('Density')
    ax.set_title('(b) Energy Distribution by Charge State'); ax.legend()
    # (c) bond lengths
    ax = axes[2]
    for cs, c, m in [(1,'crimson','^'),(-1,'royalblue','o')]:
        mean_b, mean_e = [], []
        for fr in [f for i, f in enumerate(frames) if charge_states[i]==cs]:
            dm = pairwise_distances_vectorized(fr['positions'])
            mean_b.append(np.mean([dm[i,j] for i in range(3) for j in range(i+1,3)]))
        mean_e = energies[charge_states==cs]
        ax.scatter(mean_b, mean_e, c=c, s=20, alpha=0.7, marker=m, label=f'q={cs:+d}e')
    ax.set_xlabel('Mean Bond Length (Å)'); ax.set_ylabel('Energy (eV)')
    ax.set_title('(c) Energy vs. Bond Length'); ax.legend()
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig06_ag3_overview.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig06_ag3_overview.png')


def _fig_ag3_parity(sr_te, cs_te, les_te, e_ref, charge_states):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, res, c) in zip(axes, [
        ('SR Model\n(no charge state)', sr_te, 'steelblue'),
        ('CS Embed Model', cs_te, 'seagreen'),
        ('LES Model', les_te, 'tomato')
    ]):
        ep = res['energies_pred']
        lim = [min(e_ref.min(), ep.min())-0.1, max(e_ref.max(), ep.max())+0.1]
        for cs, mrkr, col in [(1, '^', 'red'), (-1, 'o', 'blue')]:
            m = charge_states == cs
            ax.scatter(e_ref[m], ep[m], marker=mrkr, c=col, s=50, alpha=0.7, edgecolors='k', lw=0.3, label=f'q={cs:+d}e')
        ax.plot(lim, lim, 'k--', lw=1.5)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect('equal')
        ax.set_xlabel('Reference (eV)'); ax.set_ylabel('Predicted (eV)')
        ax.set_title(f'{title}\nMAE={res["mae"]:.4f}, R²={res["r2"]:.4f}')
        ax.legend(fontsize=9)
    plt.suptitle('Ag₃ PES: Model Comparison (Test Set)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig07_ag3_parity.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig07_ag3_parity.png')


def _fig_ag3_latent(les_tr, charge_states_train):
    if not les_tr.get('latent_charges'):
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    lqs = les_tr['latent_charges']
    cs = np.array(charge_states_train)

    ax = axes[0]
    for state, color in [(1,'crimson'),(-1,'royalblue')]:
        m = cs == state
        qs = np.array([lqs[i] for i in np.where(m)[0]])
        for ai in range(3):
            off = 0.12 if state == 1 else -0.12
            ax.errorbar(ai+off, qs[:,ai].mean(), yerr=qs[:,ai].std(),
                       fmt='^' if state==1 else 'o', color=color, ms=10, capsize=6,
                       label=f'q={state:+d}e' if ai==0 else '')
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['Ag1','Ag2','Ag3'])
    ax.set_ylabel('Latent Charge (a.u.)'); ax.legend()
    ax.set_title('(a) Per-Atom Latent Charges\nby True Charge State')
    ax.axhline(0, color='gray', ls='--', lw=1); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for state, color in [(1,'crimson'),(-1,'royalblue')]:
        m = cs == state
        tq = np.array([np.sum(lqs[i]) for i in np.where(m)[0]])
        ax.hist(tq, bins=10, alpha=0.65, color=color, density=True,
                label=f'q={state:+d}e (mean={tq.mean():.3f})')
    ax.set_xlabel('Sum of Latent Charges (a.u.)'); ax.set_ylabel('Density')
    ax.set_title('(b) Total Latent Charge Distribution'); ax.legend()

    plt.suptitle('Ag₃ LES Latent Charge Analysis', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig08_ag3_latent.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig08_ag3_latent.png')


def _fig_ag3_training(sr_l, cs_l, les_l):
    fig, ax = plt.subplots(figsize=(8, 5))
    e = np.arange(1, len(sr_l)+1)
    ax.semilogy(e, sr_l, 'b-', lw=2, label='SR Model (no CS)')
    ax.semilogy(e, cs_l, 'g-', lw=2, label='CS Embedding')
    ax.semilogy(e, les_l, 'r-', lw=2, label='LES Model')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log scale)')
    ax.set_title('Ag₃ Training Curves'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig09_ag3_training.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig09_ag3_training.png')


def _fig_ag3_bar(sr_te, cs_te, les_te):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    models = ['SR Model\n(no CS)', 'CS Embed', 'LES Model']
    res_list = [sr_te, cs_te, les_te]
    colors = ['steelblue', 'seagreen', 'tomato']

    ax = axes[0]
    maes = [r['mae'] for r in res_list]
    bars = ax.bar(models, maes, color=colors, alpha=0.8, edgecolor='k', lw=1)
    for b, v in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, v+0.001, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('MAE (eV)'); ax.set_title('(a) Energy MAE Comparison (Test Set)')

    ax = axes[1]
    r2s = [r['r2'] for r in res_list]
    bars = ax.bar(models, r2s, color=colors, alpha=0.8, edgecolor='k', lw=1)
    ax.set_ylim(max(0, min(r2s)-0.05), 1.02)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x()+b.get_width()/2, v+0.002, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('R² Score'); ax.set_title('(b) R² Score Comparison (Test Set)')

    plt.suptitle('Ag₃ Model Comparison Summary', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig10_ag3_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig10_ag3_comparison.png')


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    torch.manual_seed(42)

    t_start = time.time()
    print("Starting complete LES analysis...")

    # Run all three analyses
    rc_sr, rc_les, e_coul, e_rep, les_tr_res, tc_tr = run_random_charges()
    d_sr, d_les, seps, d_energies = run_charged_dimer()
    a_sr, a_cs, a_les = run_ag3()

    # Final summary figure
    _fig_summary(rc_sr, rc_les, d_sr, d_les, a_sr, a_cs, a_les)

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"COMPLETE ANALYSIS DONE in {t_total:.0f}s")
    print("─"*60)
    print("RANDOM CHARGES (128 atoms, 100 configs)")
    print(f"  SR  MAE={rc_sr['mae']:.3f}, R²={rc_sr['r2']:.4f}")
    print(f"  LES MAE={rc_les['mae']:.3f}, R²={rc_les['r2']:.4f}")
    print("CHARGED DIMER (8 atoms, 60 configs)")
    print(f"  SR  MAE={d_sr['mae']:.4f} eV, R²={d_sr['r2']:.4f}")
    print(f"  LES MAE={d_les['mae']:.4f} eV, R²={d_les['r2']:.4f}")
    print("Ag3 CHARGE STATES (3 atoms, 60 configs)")
    print(f"  SR  MAE={a_sr['mae']:.4f} eV, R²={a_sr['r2']:.4f}")
    print(f"  CS  MAE={a_cs['mae']:.4f} eV, R²={a_cs['r2']:.4f}")
    print(f"  LES MAE={a_les['mae']:.4f} eV, R²={a_les['r2']:.4f}")
    print("─"*60)
    print(f"All figures saved to {IMG_DIR}")


def _fig_summary(rc_sr, rc_les, d_sr, d_les, a_sr, a_cs, a_les):
    """Overall comparison figure across all datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Random charges
    ax = axes[0]
    models = ['SR', 'LES']
    maes = [rc_sr['mae'], rc_les['mae']]
    colors = ['steelblue', 'tomato']
    bars = ax.bar(models, maes, color=colors, alpha=0.8, edgecolor='k', lw=1, width=0.5)
    for b, v in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f'{v:.1f}', ha='center', fontweight='bold')
    ax.set_ylabel('MAE (a.u.)'); ax.set_title('(a) Random Charges\n(128 atoms, energy MAE)')

    # (b) Dimer
    ax = axes[1]
    models = ['SR', 'LES']
    maes = [d_sr['mae'], d_les['mae']]
    bars = ax.bar(models, maes, color=colors, alpha=0.8, edgecolor='k', lw=1, width=0.5)
    for b, v in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f'{v:.4f}', ha='center', fontweight='bold')
    ax.set_ylabel('MAE (eV)'); ax.set_title('(b) Charged Dimer\n(8 atoms, energy MAE)')

    # (c) Ag3
    ax = axes[2]
    models = ['SR', 'CS Embed', 'LES']
    maes = [a_sr['mae'], a_cs['mae'], a_les['mae']]
    cols = ['steelblue', 'seagreen', 'tomato']
    bars = ax.bar(models, maes, color=cols, alpha=0.8, edgecolor='k', lw=1)
    for b, v in zip(bars, maes):
        ax.text(b.get_x()+b.get_width()/2, v*1.02, f'{v:.4f}', ha='center', fontweight='bold')
    ax.set_ylabel('MAE (eV)'); ax.set_title('(c) Ag₃ Charge States\n(3 atoms, energy MAE)')

    plt.suptitle('LES Model Comparison Across All Three Benchmark Datasets', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{IMG_DIR}/fig00_summary.png', dpi=150, bbox_inches='tight')
    plt.close(); print('  Saved: fig00_summary.png')


if __name__ == '__main__':
    main()
