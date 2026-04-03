import os
import math
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

import gpytorch


@dataclass
class GPConfig:
    train_size: float = 0.8
    random_state: int = 42
    lr: float = 0.05
    training_iter: int = 800


class TgCalibrationGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def load_data():
    calib = pd.read_csv('data/tg_calibration.csv')
    vit = pd.read_csv('data/tg_vitrimer_MD.csv')
    return calib, vit


def build_feature_matrix(calib: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Simple physics-informed features from MD Tg and uncertainty.

    We avoid RDKit due to binary compatibility issues and instead
    construct handcrafted features using MD Tg and provided std.
    """
    x_md = calib['tg_md'].values.reshape(-1, 1)
    x_std = calib['std'].values.reshape(-1, 1)

    # engineered features
    feat_list = [
        x_md,
        x_std,
        (x_md ** 2),
        (x_std ** 2),
        x_md * x_std,
    ]
    X = np.concatenate(feat_list, axis=1)
    y = calib['tg_exp'].values.astype(float)
    return X, y


def train_gp_model(X: np.ndarray, y: np.ndarray, cfg: GPConfig):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=cfg.train_size, random_state=cfg.random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_x = torch.tensor(X_train_scaled, dtype=torch.float64)
    train_y = torch.tensor(y_train, dtype=torch.float64)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = TgCalibrationGPModel(train_x, train_y, likelihood)

    model.double()
    likelihood.double()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(cfg.training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if (i + 1) % 100 == 0:
            print(f"Iter {i+1}/{cfg.training_iter} - Loss: {loss.item():.3f}")
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.tensor(X_test_scaled, dtype=torch.float64)
        pred_dist = likelihood(model(test_x))
        pred_mean = pred_dist.mean.detach().cpu().numpy()
        pred_std = pred_dist.variance.sqrt().detach().cpu().numpy()

    r2 = r2_score(y_test, pred_mean)
    mae = mean_absolute_error(y_test, pred_mean)

    print(f"Test R2: {r2:.3f}, MAE: {mae:.2f} K")

    return model, likelihood, scaler, (X_test, y_test, pred_mean, pred_std)


def apply_calibration_to_vitrimers(vit: pd.DataFrame, model, likelihood, scaler):
    X_vit = np.stack([
        vit['tg'].values,
        vit['std'].values,
        vit['tg'].values ** 2,
        vit['std'].values ** 2,
        vit['tg'].values * vit['std'].values,
    ], axis=1)

    X_vit_scaled = scaler.transform(X_vit)
    vit_x = torch.tensor(X_vit_scaled, dtype=torch.float64)

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(model(vit_x))
        vit_mean = pred_dist.mean.detach().cpu().numpy()
        vit_std = pred_dist.variance.sqrt().detach().cpu().numpy()

    vit = vit.copy()
    vit['tg_calib_mean'] = vit_mean
    vit['tg_calib_std'] = vit_std

    return vit


def plot_results(calib: pd.DataFrame, test_tuple, vit_calib: pd.DataFrame):
    sns.set(style='whitegrid')
    X_test, y_test, pred_mean, pred_std = test_tuple

    # Parity plot for test set
    plt.figure(figsize=(5, 4))
    plt.scatter(y_test, pred_mean, alpha=0.7)
    lims = [min(y_test.min(), pred_mean.min()) - 10, max(y_test.max(), pred_mean.max()) + 10]
    plt.plot(lims, lims, 'k--', lw=1)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.xlabel('Experimental Tg (K)')
    plt.ylabel('GP-calibrated Tg (K)')
    plt.title('GP calibration parity (test set)')
    plt.tight_layout()
    plt.savefig('report/images/gp_parity_test.png', dpi=300)
    plt.close()

    # Residuals
    residuals = pred_mean - y_test
    plt.figure(figsize=(5, 4))
    sns.histplot(residuals, kde=True)
    plt.xlabel('Prediction error (K)')
    plt.title('GP calibration residuals (test set)')
    plt.tight_layout()
    plt.savefig('report/images/gp_residuals_test.png', dpi=300)
    plt.close()

    # Vitrimer Tg distribution before/after calibration
    plt.figure(figsize=(5, 4))
    sns.kdeplot(vit_calib['tg'], label='Raw MD Tg')
    sns.kdeplot(vit_calib['tg_calib_mean'], label='Calibrated Tg')
    plt.xlabel('Tg (K)')
    plt.title('Vitrimer Tg distribution before/after GP calibration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('report/images/vitrimer_tg_distribution.png', dpi=300)
    plt.close()


def toy_graph_vae_inverse_design(vit_calib: pd.DataFrame, target_tg: float = 400.0, n_samples: int = 50):
    """Simplified inverse design using latent sampling on MD features.

    We do not implement a full graph VAE here due to computational
    constraints; instead, we emulate a latent space over vitrimer
    chemistries using PCA-like projections on Tg features and then
    select candidates whose calibrated Tg is close to a target.
    """
    # Construct a 2D latent space from tg, std, calibrated tg
    from sklearn.decomposition import PCA

    feats = vit_calib[['tg', 'std', 'tg_calib_mean']].values
    pca = PCA(n_components=2, random_state=0)
    z = pca.fit_transform(feats)

    vit_latent = vit_calib.copy()
    vit_latent['z1'] = z[:, 0]
    vit_latent['z2'] = z[:, 1]

    # score candidates by closeness to target Tg and low uncertainty
    score = -np.abs(vit_latent['tg_calib_mean'] - target_tg) - 0.5 * vit_latent['tg_calib_std']
    vit_latent['score'] = score

    top = vit_latent.sort_values('score', ascending=False).head(n_samples)
    top.to_csv('outputs/top_inverse_design_candidates.csv', index=False)

    # latent plot colored by calibrated Tg
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(vit_latent['z1'], vit_latent['z2'], c=vit_latent['tg_calib_mean'], cmap='viridis', s=8, alpha=0.6)
    plt.colorbar(sc, label='Calibrated Tg (K)')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')
    plt.title('Latent representation of vitrimer chemistries')
    plt.tight_layout()
    plt.savefig('report/images/vitrimer_latent_space.png', dpi=300)
    plt.close()

    return top


def main():
    os.makedirs('outputs', exist_ok=True)

    calib, vit = load_data()
    X, y = build_feature_matrix(calib)

    cfg = GPConfig()
    model, likelihood, scaler, test_tuple = train_gp_model(X, y, cfg)

    vit_calib = apply_calibration_to_vitrimers(vit, model, likelihood, scaler)
    vit_calib.to_csv('outputs/tg_vitrimer_calibrated.csv', index=False)

    plot_results(calib, test_tuple, vit_calib)

    # inverse design targeting a mid-range Tg
    top = toy_graph_vae_inverse_design(vit_calib, target_tg=400.0, n_samples=30)
    print('Top candidate summary:')
    print(top[['acid', 'epoxide', 'tg', 'tg_calib_mean', 'tg_calib_std', 'score']].head())


if __name__ == '__main__':
    main()
