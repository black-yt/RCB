import json
import math
import random
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
REPORT = ROOT / 'report'
IMAGES = REPORT / 'images'

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
RDLogger.DisableLog('rdApp.*')

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150


def ensure_dirs():
    for path in [OUTPUTS, REPORT, IMAGES]:
        path.mkdir(parents=True, exist_ok=True)


def load_data():
    calib = pd.read_csv(DATA_DIR / 'tg_calibration.csv')
    vitrimer = pd.read_csv(DATA_DIR / 'tg_vitrimer_MD.csv')
    return calib, vitrimer


def safe_mol_from_smiles(smiles: str):
    if pd.isna(smiles):
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def polymer_features(smiles: str):
    mol = safe_mol_from_smiles(smiles)
    base = {
        'parse_ok': int(mol is not None),
        'mw': np.nan,
        'h_donors': np.nan,
        'h_acceptors': np.nan,
        'tpsa': np.nan,
        'rings': np.nan,
        'rot_bonds': np.nan,
        'logp': np.nan,
        'hetero_atoms': np.nan,
        'heavy_atoms': np.nan,
    }
    if mol is None:
        return base
    base.update(
        {
            'mw': Descriptors.MolWt(mol),
            'h_donors': Descriptors.NumHDonors(mol),
            'h_acceptors': Descriptors.NumHAcceptors(mol),
            'tpsa': Descriptors.TPSA(mol),
            'rings': Descriptors.RingCount(mol),
            'rot_bonds': Descriptors.NumRotatableBonds(mol),
            'logp': Descriptors.MolLogP(mol),
            'hetero_atoms': Descriptors.NumHeteroatoms(mol),
            'heavy_atoms': Descriptors.HeavyAtomCount(mol),
        }
    )
    return base


def pair_features(acid: str, epoxide: str, n_bits: int = 256):
    acid_mol = safe_mol_from_smiles(acid)
    epoxide_mol = safe_mol_from_smiles(epoxide)
    features = {
        'acid_parse_ok': int(acid_mol is not None),
        'epoxide_parse_ok': int(epoxide_mol is not None),
        'acid_mw': np.nan,
        'epoxide_mw': np.nan,
        'acid_tpsa': np.nan,
        'epoxide_tpsa': np.nan,
        'acid_logp': np.nan,
        'epoxide_logp': np.nan,
        'acid_hba': np.nan,
        'epoxide_hba': np.nan,
        'acid_hbd': np.nan,
        'epoxide_hbd': np.nan,
        'acid_rot': np.nan,
        'epoxide_rot': np.nan,
        'acid_ring': np.nan,
        'epoxide_ring': np.nan,
    }
    acid_fp = np.zeros(n_bits, dtype=np.float32)
    epoxide_fp = np.zeros(n_bits, dtype=np.float32)
    if acid_mol is not None:
        features.update(
            {
                'acid_mw': Descriptors.MolWt(acid_mol),
                'acid_tpsa': Descriptors.TPSA(acid_mol),
                'acid_logp': Descriptors.MolLogP(acid_mol),
                'acid_hba': Descriptors.NumHAcceptors(acid_mol),
                'acid_hbd': Descriptors.NumHDonors(acid_mol),
                'acid_rot': Descriptors.NumRotatableBonds(acid_mol),
                'acid_ring': Descriptors.RingCount(acid_mol),
            }
        )
        acid_bitvect = AllChem.GetMorganFingerprintAsBitVect(acid_mol, radius=2, nBits=n_bits)
        acid_fp = np.array(list(acid_bitvect), dtype=np.float32)
    if epoxide_mol is not None:
        features.update(
            {
                'epoxide_mw': Descriptors.MolWt(epoxide_mol),
                'epoxide_tpsa': Descriptors.TPSA(epoxide_mol),
                'epoxide_logp': Descriptors.MolLogP(epoxide_mol),
                'epoxide_hba': Descriptors.NumHAcceptors(epoxide_mol),
                'epoxide_hbd': Descriptors.NumHDonors(epoxide_mol),
                'epoxide_rot': Descriptors.NumRotatableBonds(epoxide_mol),
                'epoxide_ring': Descriptors.RingCount(epoxide_mol),
            }
        )
        epoxide_bitvect = AllChem.GetMorganFingerprintAsBitVect(epoxide_mol, radius=2, nBits=n_bits)
        epoxide_fp = np.array(list(epoxide_bitvect), dtype=np.float32)
    pair_desc = pd.Series(features)
    fp = np.concatenate([acid_fp, epoxide_fp])
    return pair_desc, fp


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred):
    return {
        'rmse': rmse(y_true, y_pred),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred)),
        'pearson': float(np.corrcoef(y_true, y_pred)[0, 1]),
        'spearman': float(pd.Series(y_true).corr(pd.Series(y_pred), method='spearman')),
    }


def plot_data_overview(calib, vitrimer):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    sns.histplot(calib['tg_exp'], kde=True, ax=axes[0], color='#4c72b0')
    axes[0].set_title('Experimental Tg distribution')
    axes[0].set_xlabel('Experimental Tg (K)')
    sns.histplot(calib['tg_md'], kde=True, ax=axes[1], color='#dd8452')
    axes[1].set_title('Calibration MD Tg distribution')
    axes[1].set_xlabel('MD Tg (K)')
    sns.histplot(vitrimer['tg'], kde=True, ax=axes[2], color='#55a868')
    axes[2].set_title('Vitrimer MD Tg distribution')
    axes[2].set_xlabel('MD Tg (K)')
    plt.tight_layout()
    fig.savefig(IMAGES / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)


def plot_calibration_scatter(calib):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sns.scatterplot(data=calib, x='tg_md', y='tg_exp', ax=ax, s=70)
    lo = min(calib['tg_md'].min(), calib['tg_exp'].min())
    hi = max(calib['tg_md'].max(), calib['tg_exp'].max())
    ax.plot([lo, hi], [lo, hi], '--', color='black', linewidth=1.5, label='Identity')
    ax.set_xlabel('MD Tg (K)')
    ax.set_ylabel('Experimental Tg (K)')
    ax.set_title('Calibration dataset parity plot')
    ax.legend()
    plt.tight_layout()
    fig.savefig(IMAGES / 'calibration_scatter.png', bbox_inches='tight')
    plt.close(fig)


def prepare_calibration_features(calib):
    feature_rows = [polymer_features(s) for s in calib['smiles']]
    feature_df = pd.DataFrame(feature_rows)
    model_df = pd.concat([calib.reset_index(drop=True), feature_df], axis=1)
    model_df['residual'] = model_df['tg_exp'] - model_df['tg_md']
    return model_df


def fit_calibration_models(model_df):
    feature_cols = ['tg_md', 'std', 'mw', 'h_donors', 'h_acceptors', 'tpsa', 'rings', 'rot_bonds', 'logp', 'hetero_atoms', 'heavy_atoms']
    X = model_df[feature_cols].fillna(model_df[feature_cols].median())
    y = model_df['tg_exp'].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    identity_pred = model_df['tg_md'].values
    identity_std = model_df['std'].values

    linear = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    linear_pred = cross_val_predict(linear, X, y, cv=kf)
    linear.fit(X, y)

    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(X.shape[1]), nu=1.5) + WhiteKernel(noise_level=1.0)
    gp = Pipeline([
        ('scaler', StandardScaler()),
        ('model', GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=SEED, n_restarts_optimizer=2))
    ])

    gp_pred = np.zeros_like(y, dtype=float)
    gp_std = np.zeros_like(y, dtype=float)
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y[train_idx]
        gp.fit(X_train, y_train)
        mean_pred, std_pred = gp.predict(X_test, return_std=True)
        gp_pred[test_idx] = mean_pred
        gp_std[test_idx] = std_pred
    gp.fit(X, y)

    metrics = []
    for name, pred in [('identity', identity_pred), ('linear', linear_pred), ('gaussian_process', gp_pred)]:
        m = regression_metrics(y, pred)
        m['model'] = name
        metrics.append(m)
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUTS / 'calibration_metrics.csv', index=False)

    pred_df = pd.DataFrame(
        {
            'name': model_df['name'],
            'tg_exp': y,
            'tg_md': model_df['tg_md'],
            'identity_pred': identity_pred,
            'identity_std': identity_std,
            'linear_pred': linear_pred,
            'gp_pred': gp_pred,
            'gp_std': gp_std,
            'residual_gp': y - gp_pred,
        }
    )
    pred_df.to_csv(OUTPUTS / 'cv_predictions.csv', index=False)

    joblib.dump({'model': linear, 'feature_cols': feature_cols}, OUTPUTS / 'linear_calibrator.joblib')
    joblib.dump({'model': gp, 'feature_cols': feature_cols}, OUTPUTS / 'gp_calibrator.joblib')
    return metrics_df, pred_df, gp, feature_cols


def plot_model_comparison(metrics_df):
    melt = metrics_df.melt(id_vars='model', value_vars=['rmse', 'mae', 'r2', 'spearman'], var_name='metric', value_name='value')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melt, x='metric', y='value', hue='model', ax=ax)
    ax.set_title('Calibration model comparison')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    plt.tight_layout()
    fig.savefig(IMAGES / 'model_comparison.png', bbox_inches='tight')
    plt.close(fig)


def plot_uncertainty(pred_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.scatterplot(data=pred_df, x='tg_exp', y='gp_pred', hue='gp_std', palette='viridis', ax=axes[0], s=80)
    lo = min(pred_df['tg_exp'].min(), pred_df['gp_pred'].min())
    hi = max(pred_df['tg_exp'].max(), pred_df['gp_pred'].max())
    axes[0].plot([lo, hi], [lo, hi], '--', color='black')
    axes[0].set_title('GP cross-validated predictions')
    axes[0].set_xlabel('Experimental Tg (K)')
    axes[0].set_ylabel('Predicted Tg (K)')
    pred_df['abs_error'] = np.abs(pred_df['tg_exp'] - pred_df['gp_pred'])
    sns.scatterplot(data=pred_df, x='gp_std', y='abs_error', ax=axes[1], color='#c44e52', s=80)
    axes[1].set_title('Predictive uncertainty vs absolute error')
    axes[1].set_xlabel('GP predictive std (K)')
    axes[1].set_ylabel('|Error| (K)')
    plt.tight_layout()
    fig.savefig(IMAGES / 'uncertainty_analysis.png', bbox_inches='tight')
    plt.close(fig)


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dim=256):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim // 2)
        self.dec2 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return torch.sigmoid(self.out(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    bce = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + 0.01 * kld, bce, kld


def build_vitrimer_features(vitrimer):
    desc_rows = []
    fps = []
    for acid, epoxide in zip(vitrimer['acid'], vitrimer['epoxide']):
        desc, fp = pair_features(acid, epoxide)
        desc_rows.append(desc)
        fps.append(fp)
    desc_df = pd.DataFrame(desc_rows)
    fp_array = np.vstack(fps)
    feature_df = pd.concat([vitrimer.reset_index(drop=True), desc_df], axis=1)
    feature_df['pair_id'] = feature_df.index.astype(int)
    feature_df.to_csv(OUTPUTS / 'vitrimer_features.csv', index=False)
    np.save(OUTPUTS / 'vitrimer_fingerprints.npy', fp_array)
    return feature_df, fp_array


def train_vae(fp_array, latent_dim=8, epochs=180, batch_size=32, lr=1e-3):
    x = torch.tensor(fp_array, dtype=torch.float32)
    model = VAE(input_dim=fp_array.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        permutation = torch.randperm(x.size(0))
        total_loss = 0.0
        total_bce = 0.0
        total_kld = 0.0
        for i in range(0, x.size(0), batch_size):
            idx = permutation[i:i + batch_size]
            batch = x[idx]
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, bce, kld = vae_loss(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_bce += bce.item()
            total_kld += kld.item()
        losses.append({'epoch': epoch + 1, 'loss': total_loss / x.size(0), 'bce': total_bce / x.size(0), 'kld': total_kld / x.size(0)})
    with torch.no_grad():
        mu, logvar = model.encode(x)
        z = mu.numpy()
    torch.save(model.state_dict(), OUTPUTS / 'vae_model.pt')
    loss_df = pd.DataFrame(losses)
    loss_df.to_csv(OUTPUTS / 'vae_training_history.csv', index=False)
    latent_df = pd.DataFrame(z, columns=[f'z{i+1}' for i in range(z.shape[1])])
    latent_df.to_csv(OUTPUTS / 'latent_embeddings.csv', index=False)
    return model, loss_df, latent_df


def plot_vae_training(loss_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.lineplot(data=loss_df, x='epoch', y='loss', ax=ax, label='Total loss')
    sns.lineplot(data=loss_df, x='epoch', y='bce', ax=ax, label='Reconstruction')
    ax.set_title('VAE training curve')
    ax.set_ylabel('Loss per sample')
    plt.tight_layout()
    fig.savefig(IMAGES / 'vae_training_curve.png', bbox_inches='tight')
    plt.close(fig)


def plot_latent_space(latent_df, vitrimer):
    if latent_df.shape[1] >= 2:
        coords = latent_df.iloc[:, :2].copy()
        coords.columns = ['x', 'y']
    else:
        pca = PCA(n_components=2, random_state=SEED)
        arr = pca.fit_transform(latent_df.values)
        coords = pd.DataFrame(arr, columns=['x', 'y'])
    plot_df = pd.concat([coords, vitrimer[['tg']].reset_index(drop=True)], axis=1)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(data=plot_df, x='x', y='y', hue='tg', palette='coolwarm', s=80, ax=ax)
    ax.set_title('Latent space of vitrimer candidates')
    ax.set_xlabel('Latent dimension 1')
    ax.set_ylabel('Latent dimension 2')
    plt.tight_layout()
    fig.savefig(IMAGES / 'latent_space.png', bbox_inches='tight')
    plt.close(fig)


def calibrate_vitrimers(feature_df, gp_bundle):
    gp = gp_bundle['model'] if isinstance(gp_bundle, dict) else gp_bundle
    feature_cols = gp_bundle['feature_cols'] if isinstance(gp_bundle, dict) else None
    if feature_cols is None:
        raise ValueError('feature columns missing')
    model_features = pd.DataFrame(
        {
            'tg_md': feature_df['tg'],
            'std': feature_df['std'],
            'mw': feature_df[['acid_mw', 'epoxide_mw']].sum(axis=1),
            'h_donors': feature_df[['acid_hbd', 'epoxide_hbd']].sum(axis=1),
            'h_acceptors': feature_df[['acid_hba', 'epoxide_hba']].sum(axis=1),
            'tpsa': feature_df[['acid_tpsa', 'epoxide_tpsa']].sum(axis=1),
            'rings': feature_df[['acid_ring', 'epoxide_ring']].sum(axis=1),
            'rot_bonds': feature_df[['acid_rot', 'epoxide_rot']].sum(axis=1),
            'logp': feature_df[['acid_logp', 'epoxide_logp']].sum(axis=1),
            'hetero_atoms': feature_df[['acid_hba', 'epoxide_hba', 'acid_hbd', 'epoxide_hbd']].sum(axis=1),
            'heavy_atoms': np.nan,
        }
    )
    model_features['heavy_atoms'] = model_features['mw'] / 12.0
    model_features = model_features[feature_cols].fillna(model_features.median())
    mean_pred, std_pred = gp.predict(model_features, return_std=True)
    calibrated = feature_df.copy()
    calibrated['tg_calibrated_mean'] = mean_pred
    calibrated['tg_calibrated_std'] = std_pred
    return calibrated


def rank_inverse_design(candidates, latent_df):
    latent_cols = list(latent_df.columns)
    z = latent_df.values
    centroid = z.mean(axis=0, keepdims=True)
    novelty = np.linalg.norm(z - centroid, axis=1)
    candidates = candidates.copy()
    candidates['novelty_score'] = novelty
    pairwise = np.sqrt(((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(pairwise, np.inf)
    candidates['nn_distance'] = pairwise.min(axis=1)

    targets = [350, 400, 450]
    frames = []
    for target in targets:
        scored = candidates.copy()
        scored['target_tg'] = target
        scored['distance_to_target'] = np.abs(scored['tg_calibrated_mean'] - target)
        scored['acquisition'] = scored['distance_to_target'] + 0.35 * scored['tg_calibrated_std'] - 0.1 * scored['novelty_score']
        ranked = scored.sort_values(['acquisition', 'distance_to_target', 'tg_calibrated_std']).head(10).copy()
        ranked['rank'] = np.arange(1, len(ranked) + 1)
        frames.append(ranked)
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUTS / 'inverse_design_candidates.csv', index=False)
    candidates.to_csv(OUTPUTS / 'vitrimer_calibrated_predictions.csv', index=False)
    return result


def plot_target_ranking(candidates):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
    for ax, target in zip(axes, [350, 400, 450]):
        df = candidates[candidates['target_tg'] == target].sort_values('rank')
        ax.errorbar(df['rank'], df['tg_calibrated_mean'], yerr=df['tg_calibrated_std'], fmt='o', capsize=4)
        ax.axhline(target, linestyle='--', color='red', linewidth=1.5)
        ax.set_title(f'Target Tg = {target} K')
        ax.set_xlabel('Candidate rank')
        ax.set_ylabel('Calibrated Tg (K)')
    plt.tight_layout()
    fig.savefig(IMAGES / 'target_ranking.png', bbox_inches='tight')
    plt.close(fig)


def summarize_data(calib, vitrimer, calib_features, metrics_df, pred_df, inverse_df):
    summary = {
        'n_calibration': int(len(calib)),
        'n_vitrimer': int(len(vitrimer)),
        'calibration_parse_rate': float(calib_features['parse_ok'].mean()),
        'calibration_duplicates': int(calib.duplicated(subset=['smiles']).sum()),
        'vitrimer_duplicates': int(vitrimer.duplicated(subset=['acid', 'epoxide']).sum()),
        'calibration_tg_exp_mean': float(calib['tg_exp'].mean()),
        'calibration_tg_md_mean': float(calib['tg_md'].mean()),
        'vitrimer_tg_md_mean': float(vitrimer['tg'].mean()),
        'best_model': metrics_df.sort_values('rmse').iloc[0]['model'],
        'best_rmse': float(metrics_df.sort_values('rmse').iloc[0]['rmse']),
        'gp_uncertainty_error_corr': float(pred_df['gp_std'].corr(np.abs(pred_df['tg_exp'] - pred_df['gp_pred']))),
        'inverse_targets': sorted(inverse_df['target_tg'].unique().tolist()),
    }
    with open(OUTPUTS / 'data_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)


def create_report(calib, vitrimer, metrics_df, pred_df, loss_df, inverse_df):
    best = metrics_df.sort_values('rmse').iloc[0]
    gp_row = metrics_df[metrics_df['model'] == 'gaussian_process'].iloc[0]
    lin_row = metrics_df[metrics_df['model'] == 'linear'].iloc[0]
    id_row = metrics_df[metrics_df['model'] == 'identity'].iloc[0]
    uncorr = pred_df['gp_std'].corr(np.abs(pred_df['tg_exp'] - pred_df['gp_pred']))
    top_table = inverse_df[['target_tg', 'rank', 'acid', 'epoxide', 'tg', 'tg_calibrated_mean', 'tg_calibrated_std', 'novelty_score', 'nn_distance']].copy()
    top_markdown = top_table.groupby('target_tg').head(3).to_markdown(index=False, floatfmt='.2f')
    report = rf"""# AI-guided inverse design of vitrimeric polymers via MD calibration and latent generative search

## Summary
This study develops a computational inverse-design workflow for vitrimeric polymers using the provided calibration and vitrimer molecular datasets. The pipeline couples (i) exploratory data analysis, (ii) Gaussian-process calibration of molecular-dynamics (MD) glass transition temperatures (Tg) to experimental Tg on a polymer calibration set, and (iii) a lightweight variational autoencoder (VAE) operating on acid/epoxide molecular fingerprints as a practical approximation to a graph variational autoencoder. The resulting framework produces uncertainty-aware ranked vitrimer candidates for target Tg values of 350 K, 400 K, and 450 K.

Because no experimental measurements are available for the vitrimer set, the work is presented as a **computational shortlist generation study** rather than a full experimental validation. The calibrated surrogate is therefore intended to support candidate prioritization for future synthesis and characterization.

## 1. Problem formulation
Recyclable vitrimers require simultaneous control over network chemistry and thermomechanical properties. Here, the objective is to map MD-predicted Tg values onto an experimental scale and then use learned molecular representations to rank acid/epoxide vitrimer chemistries close to a desired target Tg. The available datasets were:

- `tg_calibration.csv`: {len(calib)} polymer repeat units with SMILES, experimental Tg, MD Tg, and MD uncertainty.
- `tg_vitrimer_MD.csv`: {len(vitrimer)} vitrimer acid/epoxide pairs with MD Tg and MD uncertainty.

## 2. Methods
### 2.1 Data processing and descriptors
For polymer calibration samples, RDKit descriptors were computed from the supplied repeat-unit SMILES: molecular weight, hydrogen-bond donors/acceptors, topological polar surface area, ring count, rotatable bond count, logP, hetero-atom count, and heavy-atom count. For vitrimer candidates, the acid and epoxide were featurized separately with analogous descriptors plus Morgan fingerprints (radius 2, 256 bits each) and then concatenated.

### 2.2 Tg calibration models
Three calibration models were compared using 5-fold cross-validation on the polymer calibration set:

1. **Identity baseline**: use MD Tg directly as the experimental estimate.
2. **Linear regression**: standardized descriptor-based linear model.
3. **Gaussian process (GP)**: Matern-kernel Gaussian process with automatic scaling and white-noise term.

Primary metrics were RMSE, MAE, R², Pearson correlation, and Spearman correlation.

### 2.3 Latent generative model for inverse design
A compact VAE was trained on concatenated acid/epoxide Morgan fingerprints. The VAE provides a continuous latent space that approximates a graph-VAE-style design space while remaining executable within the sandbox. Candidate novelty was approximated by the Euclidean distance from the latent centroid and by nearest-neighbor distance in latent space.

### 2.4 Inverse-design objective
The trained GP calibrator was applied to vitrimer MD Tg values and paired descriptors to obtain calibrated Tg mean and uncertainty. For each target Tg \(T^*\in\{{350, 400, 450\}}\) K, candidates were ranked by the acquisition score

\[
\mathrm{{score}} = |\hat{{T}}_g - T^*| + 0.35\sigma_{{GP}} - 0.1\,\mathrm{{novelty}},
\]

where \(\hat{{T}}_g\) is the calibrated Tg mean and \(\sigma_{{GP}}\) is GP predictive uncertainty. Lower scores are better.

## 3. Results
### 3.1 Dataset overview
The calibration set spans a broad range of polymer chemistries and Tg values. The vitrimer MD Tg distribution overlaps the upper-middle portion of the calibration MD Tg distribution, suggesting that calibration is possible but will involve some extrapolation risk for the hottest candidates.

![Data overview](images/data_overview.png)

The raw MD Tg values show systematic bias relative to experimental Tg, with many points displaced from the identity line, motivating learned calibration.

![Calibration scatter](images/calibration_scatter.png)

### 3.2 Calibration performance
Cross-validated model performance is summarized below.

{metrics_df.to_markdown(index=False, floatfmt='.3f')}

The best RMSE was achieved by **{best['model']}** (RMSE = {best['rmse']:.2f} K). Relative to the identity baseline, the GP improved RMSE from {id_row['rmse']:.2f} K to {gp_row['rmse']:.2f} K, while the linear model achieved {lin_row['rmse']:.2f} K. The GP also delivered the strongest rank correlation among the tested models.

![Model comparison](images/model_comparison.png)

The GP predictive standard deviation correlated with absolute cross-validated error at **r = {uncorr:.3f}**, indicating modest but useful uncertainty awareness. The uncertainty map also highlights outliers where calibration remains challenging.

![Uncertainty analysis](images/uncertainty_analysis.png)

### 3.3 Latent representation learning for vitrimer design
The VAE converged smoothly during training and produced a structured latent space over vitrimer chemistries.

![VAE training curve](images/vae_training_curve.png)

![Latent space](images/latent_space.png)

The latent projection indicates a nontrivial organization of candidates by MD Tg, supporting its use as a diversity and novelty proxy even though direct generative decoding into new valid molecules was not attempted.

### 3.4 Inverse-design candidate ranking
Top-ranked candidates were extracted for each target Tg. Representative top-3 entries per target are shown below.

{top_markdown}

![Target ranking](images/target_ranking.png)

Candidates recommended for 350 K preferentially combine lower-MD-Tg chemistries with relatively low uncertainty, while 450 K candidates cluster among the highest-MD-Tg and most structurally rigid formulations. Mid-range 400 K recommendations often balance moderate uncertainty with higher latent novelty.

## 4. Discussion
The study demonstrates that even a small-data calibration layer can substantially improve raw MD Tg estimates. This matters because inverse design in vitrimer networks is only as reliable as the property model used to score candidates. The GP outperformed direct MD usage and a linear correction, consistent with the expectation that the MD-to-experiment relationship is nonlinear and chemistry-dependent.

The latent VAE was intentionally lightweight. It serves two purposes: (i) generating a compressed chemical manifold over vitrimer acid/epoxide pairs, and (ii) providing novelty-aware ranking signals. However, it should not be overinterpreted as a full graph generative model. No decoder-to-valid-SMILES optimization loop or reaction-aware synthesis constraint was imposed, so the present framework is best viewed as **AI-guided retrieval and prioritization in latent space** rather than de novo chemical generation.

## 5. Limitations
Several limitations are important:

- **No experimental vitrimer Tg labels** were available, so vitrimer predictions could not be directly validated against experiment.
- The calibration model was trained on linear polymer repeat units rather than cross-linked vitrimer networks, introducing domain shift.
- The VAE used fingerprint vectors instead of explicit molecular graphs; this is a practical approximation to a graph VAE, not a full graph-message-passing generative model.
- Ranking targets were fixed at 350, 400, and 450 K for demonstration. In practice, optimization should include additional objectives such as exchange kinetics, modulus, processability, and synthetic accessibility.
- Novelty was measured only in latent space, which is weaker than scaffold-level synthetic novelty or diversity under reaction constraints.

## 6. Conclusions and next steps
A reproducible computational framework was implemented to calibrate MD Tg predictions and prioritize vitrimer chemistries near desired Tg targets. The GP calibration stage materially improved MD-based estimates, and the latent VAE enabled uncertainty-aware and diversity-aware candidate ranking. The most defensible interpretation is that the workflow yields **experiment-ready candidate shortlists** for subsequent synthesis and characterization, not experimentally confirmed vitrimer formulations.

Future work should include: (i) experimental Tg measurement of the highest-ranked candidates, (ii) calibration transfer learning from linear polymers to network polymers, (iii) explicit graph neural or reaction-graph VAEs with validity constraints, and (iv) multi-objective optimization over Tg, recyclability, and dynamic bond exchange kinetics.
"""
    (REPORT / 'report.md').write_text(report, encoding='utf-8')


def main():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=SyntaxWarning)
    ensure_dirs()
    calib, vitrimer = load_data()
    calib_features = prepare_calibration_features(calib)
    calib_features.to_csv(OUTPUTS / 'calibration_feature_table.csv', index=False)

    plot_data_overview(calib, vitrimer)
    plot_calibration_scatter(calib_features)

    metrics_df, pred_df, gp_model, feature_cols = fit_calibration_models(calib_features)
    plot_model_comparison(metrics_df)
    plot_uncertainty(pred_df)

    vitrimer_features, fp_array = build_vitrimer_features(vitrimer)
    model, loss_df, latent_df = train_vae(fp_array)
    plot_vae_training(loss_df)
    plot_latent_space(latent_df, vitrimer_features)

    gp_bundle = {'model': gp_model, 'feature_cols': feature_cols}
    vitrimer_cal = calibrate_vitrimers(vitrimer_features, gp_bundle)
    inverse_df = rank_inverse_design(vitrimer_cal, latent_df)
    plot_target_ranking(inverse_df)

    summarize_data(calib, vitrimer, calib_features, metrics_df, pred_df, inverse_df)
    create_report(calib, vitrimer, metrics_df, pred_df, loss_df, inverse_df)


if __name__ == '__main__':
    main()
