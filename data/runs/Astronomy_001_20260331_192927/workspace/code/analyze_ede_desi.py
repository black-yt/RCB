import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_IMG_DIR = BASE_DIR / 'report' / 'images'
OUTPUT_DIR = BASE_DIR / 'outputs'
DATA_FILE = BASE_DIR / 'data' / 'DESI_EDE_Repro_Data.txt'

REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the structured "python" data file by executing it in a sandbox dict
context = {}
with open(DATA_FILE, 'r') as f:
    code = f.read()
exec(compile(code, str(DATA_FILE), 'exec'), context)

lcdm = context['lcdm_params']
ede = context['ede_params']
w0wa = context['w0wa_params']

# Helper to build arrays for a given set of parameters
core_params = ['omega_m', 'H0', 'sigma8']

models = ['LCDM', 'EDE', 'w0wA']
param_dicts = [lcdm, ede, w0wa]

means = {p: [] for p in core_params}
errs = {p: [] for p in core_params}
for p in core_params:
    for d in param_dicts:
        m, s = d[p]
        means[p].append(m)
        errs[p].append(s)

# Plot 1: comparison bar plots for Omega_m, H0, sigma8
sns.set(context='talk', style='whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, p in enumerate(core_params):
    ax = axes[i]
    x = np.arange(len(models))
    ax.errorbar(x, means[p], yerr=errs[p], fmt='o', capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title(p)
    ax.set_xlim(-0.5, len(models)-0.5)

axes[0].set_ylabel('Parameter value')
fig.suptitle('Key cosmological parameters for different models (CMB+DESI)')
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(REPORT_IMG_DIR / 'params_comparison.png', dpi=200)
plt.close(fig)

# Plot 2: EDE parameter posterior means with 1-sigma
ede_extra_params = ['f_EDE', 'log10_ac']
fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(ede_extra_params))
ede_means = [ede[p][0] for p in ede_extra_params]
ede_errs = [ede[p][1] for p in ede_extra_params]
ax.errorbar(x, ede_means, yerr=ede_errs, fmt='o', capsize=4, color='C1')
ax.set_xticks(x)
ax.set_xticklabels([r'$f_\mathrm{EDE}$', r'$\log_{10} a_c$'])
ax.set_ylabel('Value')
ax.set_title('EDE model parameters (CMB+DESI)')
fig.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'ede_params.png', dpi=200)
plt.close(fig)

# Load BAO and SNe points
z_dvrd, dvrd, dvrd_err = np.array(context['desi_dvrd_points']).T
z_fap, fap, fap_err = np.array(context['desi_fap_points']).T
z_sne, mu, mu_err = np.array(context['sne_mu_points']).T

# Plot 3: DESI BAO DV/r_d residuals
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(z_dvrd, dvrd, yerr=dvrd_err, fmt='o', capsize=3)
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Redshift z')
ax.set_ylabel(r'$\Delta(D_V/r_d)$ relative to fiducial')
ax.set_title('DESI DR2 BAO distance residuals')
fig.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'desi_dvrd_residuals.png', dpi=200)
plt.close(fig)

# Plot 4: DESI BAO F_AP residuals
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(z_fap, fap, yerr=fap_err, fmt='s', capsize=3, color='C2')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Redshift z')
ax.set_ylabel(r'$\Delta F_\mathrm{AP}$ relative to fiducial')
ax.set_title('DESI DR2 BAO Alcock–Paczynski residuals')
fig.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'desi_fap_residuals.png', dpi=200)
plt.close(fig)

# Plot 5: Union3 SNe distance modulus residuals
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(z_sne, mu, yerr=mu_err, fmt='^', capsize=3, color='C3')
ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Redshift z')
ax.set_ylabel('Distance modulus residual $\\Delta \\mu$ (mag)')
ax.set_title('Union3 SNe distance modulus residuals')
fig.tight_layout()
fig.savefig(REPORT_IMG_DIR / 'sne_mu_residuals.png', dpi=200)
plt.close(fig)

# Simple derived comparison: tension in H0 between models (in sigma units)
H0_lcdm, H0_lcdm_err = lcdm['H0']
H0_ede, H0_ede_err = ede['H0']
H0_w0wa, H0_w0wa_err = w0wa['H0']

# Compare LCDM vs EDE, LCDM vs w0wa
lcdm_ede_tension = abs(H0_ede - H0_lcdm) / np.sqrt(H0_ede_err**2 + H0_lcdm_err**2)
lcdm_w0wa_tension = abs(H0_w0wa - H0_lcdm) / np.sqrt(H0_w0wa_err**2 + H0_lcdm_err**2)

with open(OUTPUT_DIR / 'summary.txt', 'w') as f:
    f.write('H0 LCDM = {:.2f} +/- {:.2f}\n'.format(H0_lcdm, H0_lcdm_err))
    f.write('H0 EDE   = {:.2f} +/- {:.2f}\n'.format(H0_ede, H0_ede_err))
    f.write('H0 w0wa  = {:.2f} +/- {:.2f}\n'.format(H0_w0wa, H0_w0wa_err))
    f.write('\n')
    f.write('LCDM vs EDE H0 tension: {:.2f} sigma\n'.format(lcdm_ede_tension))
    f.write('LCDM vs w0wa H0 tension: {:.2f} sigma\n'.format(lcdm_w0wa_tension))

print('Analysis complete. Figures saved to', REPORT_IMG_DIR)
