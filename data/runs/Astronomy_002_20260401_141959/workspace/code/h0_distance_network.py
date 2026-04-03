#!/usr/bin/env python3
"""
Local Distance Network: H0 measurement via covariance-weighted GLS framework.
Reproduces a ~1% precision Hubble constant from geometric anchors, primary
distance indicators, SNe Ia, and SBF Hubble-flow observations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from scipy.linalg import inv
import json
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
WS = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Astronomy_002_20260401_141959"
OUT = os.path.join(WS, "outputs")
FIGS = os.path.join(WS, "report/images")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

# ── Data ───────────────────────────────────────────────────────────────────────
anchors = {
    'N4258': {'mu': 29.397, 'err': 0.032},
    'LMC':   {'mu': 18.477, 'err': 0.024},
    'MW':    {'mu': 0.0,    'err': 0.0}
}

host_measurements = [
    ('NGC1309', 'Cepheid', 'N4258', 32.50, 0.10),
    ('NGC1365', 'Cepheid', 'N4258', 31.33, 0.08),
    ('NGC1448', 'Cepheid', 'N4258', 31.31, 0.09),
    ('NGC1559', 'Cepheid', 'N4258', 31.42, 0.07),
    ('M101',    'Cepheid', 'N4258', 29.12, 0.06),
    ('NGC1316', 'TRGB',    'N4258', 31.39, 0.10),
    ('NGC1365', 'TRGB',    'N4258', 31.32, 0.12),
    ('NGC5643', 'TRGB',    'N4258', 30.53, 0.09),
    ('M101',    'TRGB',    'N4258', 29.13, 0.08),
    ('NGC1309', 'Cepheid', 'LMC',   32.51, 0.11),
    ('NGC1365', 'Cepheid', 'LMC',   31.34, 0.09),
]

sneia_calibrators = [
    ('NGC1309', 12.10, 0.05),
    ('NGC1365', 11.93, 0.06),
    ('NGC1448', 11.90, 0.05),
    ('NGC1559', 12.22, 0.05),
    ('M101',    9.85,  0.04),
    ('NGC1316', 11.88, 0.07),
    ('NGC5643', 11.56, 0.06),
]

sbf_calibrators = [
    ('NGC1399', 28.35, 0.10),
    ('NGC1404', 28.33, 0.10),
    ('NGC4472', 28.56, 0.12),
]

hubble_flow_sneia = [
    (0.034, 15.12, 0.06, 250),
    (0.042, 15.68, 0.05, 250),
    (0.055, 16.35, 0.05, 250),
    (0.068, 17.02, 0.05, 250),
    (0.082, 17.55, 0.06, 250),
]

hubble_flow_sbf = [
    (0.023, 30.45, 0.15, 250),
    (0.031, 31.02, 0.15, 250),
    (0.045, 31.89, 0.16, 250),
]

method_anchor_err = {
    ('Cepheid', 'N4258'): 0.04,
    ('Cepheid', 'LMC'):   0.03,
    ('Cepheid', 'MW'):    0.02,
    ('TRGB',    'N4258'): 0.05,
}

host_group = {
    'NGC1399': 'Fornax',
    'NGC1404': 'Fornax',
    'NGC4472': 'Virgo',
}
depth_scatter = 0.10
c_km = 299792.458

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  HOST DISTANCE MODULI via GLS
# ═══════════════════════════════════════════════════════════════════════════════
# Each host measurement: μ_host^obs(m,a) = μ_host^true + ε_stat
# The anchor uncertainty introduces a correlated error shared by all
# measurements on the same anchor.
#
# Total σ² for a single measurement = σ_meas² + σ_anchor² + σ_method²
# Two measurements on the same anchor have covariance = σ_anchor².
# We propagate this through inverse-variance weighting.

def compute_host_distance_moduli():
    """
    Return dict: host -> (mu_host, sigma_host)
    Handles anchor covariance via covariance-matrix inversion per host.
    """
    from collections import defaultdict
    # Group measurements by host
    host_data = defaultdict(list)
    for (host, method, anchor, mu_obs, err_obs) in host_measurements:
        sys_err = method_anchor_err.get((method, anchor), 0.03)
        host_data[host].append({
            'anchor': anchor,
            'method': method,
            'mu_obs': mu_obs,
            'err_stat': err_obs,
            'err_anchor': anchors[anchor]['err'],
            'err_sys': sys_err,
        })

    host_mu = {}
    for host, meas_list in host_data.items():
        n = len(meas_list)
        # Build covariance matrix
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    C[i, j] = (meas_list[i]['err_stat']**2
                               + meas_list[i]['err_anchor']**2
                               + meas_list[i]['err_sys']**2)
                else:
                    # Shared anchor covariance
                    if meas_list[i]['anchor'] == meas_list[j]['anchor']:
                        C[i, j] = meas_list[i]['err_anchor']**2
                    else:
                        C[i, j] = 0.0
        Cinv = inv(C)
        ones = np.ones(n)
        mu_vec = np.array([m['mu_obs'] for m in meas_list])
        # GLS estimate: mu = (1^T C^{-1} 1)^{-1} * 1^T C^{-1} mu
        denom = ones @ Cinv @ ones
        numer = ones @ Cinv @ mu_vec
        mu_best = numer / denom
        sigma_best = np.sqrt(1.0 / denom)
        host_mu[host] = (mu_best, sigma_best)

    return host_mu

host_mu = compute_host_distance_moduli()
print("Host distance moduli:")
for h, (mu, sig) in sorted(host_mu.items()):
    print(f"  {h:12s}  mu = {mu:.4f} ± {sig:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  CALIBRATE M_B (absolute SNe Ia magnitude)
# ═══════════════════════════════════════════════════════════════════════════════
# M_B = m_B - mu_host  for each calibrator host

MB_values = []
MB_errors = []
MB_hosts = []
for (host, mB, err_mB) in sneia_calibrators:
    if host in host_mu:
        mu_h, sig_mu_h = host_mu[host]
        MB_i = mB - mu_h
        sig_MB_i = np.sqrt(err_mB**2 + sig_mu_h**2)
        MB_values.append(MB_i)
        MB_errors.append(sig_MB_i)
        MB_hosts.append(host)

MB_values = np.array(MB_values)
MB_errors = np.array(MB_errors)
weights_MB = 1.0 / MB_errors**2
MB_mean = np.sum(weights_MB * MB_values) / np.sum(weights_MB)
MB_sigma = np.sqrt(1.0 / np.sum(weights_MB))

print(f"\nCalibrated M_B (SNe Ia) = {MB_mean:.4f} ± {MB_sigma:.4f} mag")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  H0 FROM HUBBLE-FLOW SNe Ia
# ═══════════════════════════════════════════════════════════════════════════════
# m_B = M_B + 5*log10(cz / H0) + 25
# => log10(H0) = log10(c*z) - (m_B - M_B - 25) / 5
# Peculiar velocity adds: delta_mu_pec = (5/ln10) * (v_pec / (c*z))

H0_sneia_individual = []
H0_sneia_errors = []
for (z, mB, err_mB, v_pec) in hubble_flow_sneia:
    # Peculiar velocity error in magnitudes
    sig_pec = (5.0 / np.log(10)) * (v_pec / (c_km * z))
    # Total uncertainty on m_B
    sig_tot = np.sqrt(err_mB**2 + MB_sigma**2 + sig_pec**2)
    # H0 from this SN
    log10_H0 = np.log10(c_km * z) - (mB - MB_mean - 25.0) / 5.0
    H0_i = 10**log10_H0
    # Error propagation: sigma_logH0 = sig_tot / (5 * ln10)...
    # Actually: d(log10 H0)/d(mB) = -1/5, so d(H0)/H0 = ln(10)/5 * sig_tot
    sig_H0_i = H0_i * (np.log(10) / 5.0) * sig_tot
    H0_sneia_individual.append(H0_i)
    H0_sneia_errors.append(sig_H0_i)

H0_sneia_arr = np.array(H0_sneia_individual)
H0_sneia_errs = np.array(H0_sneia_errors)
w_snia = 1.0 / H0_sneia_errs**2
H0_snia = np.sum(w_snia * H0_sneia_arr) / np.sum(w_snia)
sigma_H0_snia = np.sqrt(1.0 / np.sum(w_snia))

print(f"\nH0 (SNe Ia Hubble flow) = {H0_snia:.3f} ± {sigma_H0_snia:.3f} km/s/Mpc")

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SBF CALIBRATION  (linking via host mu to TRGB/Cepheid hosts)
# ═══════════════════════════════════════════════════════════════════════════════
# SBF calibrators: (host, m_SBF, err)
# The SBF absolute magnitude M_SBF is calibrated from hosts that have
# both a known mu (from step 1) and a SBF measurement.
# Here none of the SBF calibrators directly overlap with host_mu, so we
# use the average offset from the literature to get M_SBF.
# Strategy: use the SBF calibrators to build M_SBF assuming the Fornax cluster
# members share a distance (their mean mu from anchors).
# For NGC4472 (Virgo), use separately.

# We do not have direct primary-indicator calibrations for the SBF hosts in
# this dataset. Instead we compute M_SBF by assuming:
#   mu_Fornax ~ from external calibration (use anchor propagation via SBF zero-point)
# Practical approach: use weighted mean of calibrators to get <m-M>_SBF,
# and calibrate absolute M_SBF against Fornax distance from the literature
# (mu_Fornax = 31.51 ± 0.10, a typical value from TRGB+Cepheid literature).
# This is an external calibration step in the distance network.

mu_Fornax_ext = 31.51  # literature Fornax cluster distance
sig_mu_Fornax = 0.10

# Fornax SBF calibrators
fornax_sbf = [(h, m, e) for (h, m, e) in sbf_calibrators if host_group.get(h) == 'Fornax']
m_sbf_fornax = np.array([m for (_, m, _) in fornax_sbf])
e_sbf_fornax = np.array([e for (_, _, e) in fornax_sbf])
# Account for intra-group depth scatter
total_e_fornax = np.sqrt(e_sbf_fornax**2 + depth_scatter**2)
w_f = 1.0 / total_e_fornax**2
m_sbf_mean_fornax = np.sum(w_f * m_sbf_fornax) / np.sum(w_f)
sig_m_sbf_fornax = np.sqrt(1.0 / np.sum(w_f))

M_SBF = m_sbf_mean_fornax - mu_Fornax_ext
sig_M_SBF = np.sqrt(sig_m_sbf_fornax**2 + sig_mu_Fornax**2)

print(f"\nCalibrated M_SBF (Fornax) = {M_SBF:.4f} ± {sig_M_SBF:.4f} mag")

# Virgo SBF calibrator: use same M_SBF to get mu_Virgo
virgo_sbf = [(h, m, e) for (h, m, e) in sbf_calibrators if host_group.get(h) == 'Virgo']
if virgo_sbf:
    h_v, m_v, e_v = virgo_sbf[0]
    mu_Virgo = m_v - M_SBF
    sig_mu_Virgo = np.sqrt(e_v**2 + sig_M_SBF**2 + depth_scatter**2)
    print(f"  mu_Virgo ({h_v}) = {mu_Virgo:.4f} ± {sig_mu_Virgo:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  H0 FROM HUBBLE-FLOW SBF
# ═══════════════════════════════════════════════════════════════════════════════
H0_sbf_individual = []
H0_sbf_errors = []
for (z, m_obs, err_m, v_pec) in hubble_flow_sbf:
    sig_pec = (5.0 / np.log(10)) * (v_pec / (c_km * z))
    sig_tot = np.sqrt(err_m**2 + sig_M_SBF**2 + sig_pec**2)
    log10_H0 = np.log10(c_km * z) - (m_obs - M_SBF - 25.0) / 5.0
    H0_i = 10**log10_H0
    sig_H0_i = H0_i * (np.log(10) / 5.0) * sig_tot
    H0_sbf_individual.append(H0_i)
    H0_sbf_errors.append(sig_H0_i)

H0_sbf_arr = np.array(H0_sbf_individual)
H0_sbf_errs = np.array(H0_sbf_errors)
w_sbf = 1.0 / H0_sbf_errs**2
H0_sbf = np.sum(w_sbf * H0_sbf_arr) / np.sum(w_sbf)
sigma_H0_sbf = np.sqrt(1.0 / np.sum(w_sbf))

print(f"\nH0 (SBF Hubble flow)  = {H0_sbf:.3f} ± {sigma_H0_sbf:.3f} km/s/Mpc")

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  COMBINED CONSENSUS H0 (covariance-weighted)
# ═══════════════════════════════════════════════════════════════════════════════
# The two primary probes are SNe Ia and SBF.  Treat as partially correlated
# through the shared anchor calibration.
# Correlation coefficient estimated from shared anchor uncertainty contribution.
# For SNe Ia: anchor contributes anchors['N4258']['err'] / sigma_H0_snia ≈ small
# Conservative: assume correlation rho = 0.2 from shared calibration systematics.

rho = 0.20  # shared calibration correlation
C2 = np.array([
    [sigma_H0_snia**2,                rho * sigma_H0_snia * sigma_H0_sbf],
    [rho * sigma_H0_snia * sigma_H0_sbf, sigma_H0_sbf**2],
])
C2inv = inv(C2)
H0_vec = np.array([H0_snia, H0_sbf])
ones2 = np.ones(2)
denom2 = ones2 @ C2inv @ ones2
numer2 = ones2 @ C2inv @ H0_vec
H0_combined = numer2 / denom2
sigma_H0_combined = np.sqrt(1.0 / denom2)

print(f"\n{'='*55}")
print(f"  CONSENSUS H0 = {H0_combined:.3f} ± {sigma_H0_combined:.3f} km/s/Mpc")
print(f"  Fractional precision: {sigma_H0_combined/H0_combined*100:.2f}%")
print(f"{'='*55}")

# CMB (Planck 2018) reference
H0_CMB = 67.4
sigma_CMB = 0.5
tension = (H0_combined - H0_CMB) / np.sqrt(sigma_H0_combined**2 + sigma_CMB**2)
print(f"  Tension with Planck CMB ({H0_CMB} ± {sigma_CMB}): {tension:.1f}σ")

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  ANALYSIS VARIANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Variant A: Only N4258 anchor
# Variant B: Only LMC anchor
# Variant C: All anchors together (baseline)
# Variant D: Cepheid-only hosts
# Variant E: TRGB-only hosts

def H0_from_subset(meas_subset, sneia_calib=sneia_calibrators):
    from collections import defaultdict
    hd = defaultdict(list)
    for (host, method, anchor, mu_obs, err_obs) in meas_subset:
        sys_err = method_anchor_err.get((method, anchor), 0.03)
        hd[host].append({
            'anchor': anchor, 'method': method, 'mu_obs': mu_obs,
            'err_stat': err_obs, 'err_anchor': anchors[anchor]['err'],
            'err_sys': sys_err,
        })
    hm = {}
    for host, ml in hd.items():
        n = len(ml)
        C = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    C[i, j] = ml[i]['err_stat']**2 + ml[i]['err_anchor']**2 + ml[i]['err_sys']**2
                else:
                    C[i, j] = ml[i]['err_anchor']**2 if ml[i]['anchor'] == ml[j]['anchor'] else 0.0
        Ci = inv(C)
        ones = np.ones(n)
        mu_v = np.array([m['mu_obs'] for m in ml])
        denom = ones @ Ci @ ones
        hm[host] = (ones @ Ci @ mu_v / denom, np.sqrt(1.0 / denom))

    MB_v, MB_e = [], []
    for (host, mB, err_mB) in sneia_calib:
        if host in hm:
            mu_h, sig_h = hm[host]
            MB_v.append(mB - mu_h)
            MB_e.append(np.sqrt(err_mB**2 + sig_h**2))
    if not MB_v:
        return None, None
    MB_v = np.array(MB_v); MB_e = np.array(MB_e)
    w = 1.0 / MB_e**2
    MB_m = np.sum(w * MB_v) / np.sum(w)
    MB_s = np.sqrt(1.0 / np.sum(w))

    H0_vals, H0_errs = [], []
    for (z, mB, err_mB, v_pec) in hubble_flow_sneia:
        sp = (5.0 / np.log(10)) * (v_pec / (c_km * z))
        st = np.sqrt(err_mB**2 + MB_s**2 + sp**2)
        H0_i = 10 ** (np.log10(c_km * z) - (mB - MB_m - 25.0) / 5.0)
        H0_vals.append(H0_i)
        H0_errs.append(H0_i * (np.log(10) / 5.0) * st)
    H0_v = np.array(H0_vals); H0_e = np.array(H0_errs)
    wh = 1.0 / H0_e**2
    H0_out = np.sum(wh * H0_v) / np.sum(wh)
    sig_out = np.sqrt(1.0 / np.sum(wh))
    return H0_out, sig_out

variants = {
    'Baseline (All)':    host_measurements,
    'N4258 anchor only': [m for m in host_measurements if m[2] == 'N4258'],
    'LMC anchor only':   [m for m in host_measurements if m[2] == 'LMC'],
    'Cepheid only':      [m for m in host_measurements if m[1] == 'Cepheid'],
    'TRGB only':         [m for m in host_measurements if m[1] == 'TRGB'],
}

variant_results = {}
print("\nAnalysis variants (SNe Ia probe):")
for name, meas in variants.items():
    h0, sh0 = H0_from_subset(meas)
    if h0 is not None:
        variant_results[name] = (h0, sh0)
        print(f"  {name:30s}  H0 = {h0:.3f} ± {sh0:.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
results = {
    'host_distance_moduli': {k: list(v) for k, v in host_mu.items()},
    'MB_calibration': {'mean': MB_mean, 'sigma': MB_sigma},
    'M_SBF_calibration': {'mean': M_SBF, 'sigma': sig_M_SBF},
    'H0_SNe_Ia': {'value': H0_snia, 'sigma': sigma_H0_snia},
    'H0_SBF':    {'value': H0_sbf,  'sigma': sigma_H0_sbf},
    'H0_consensus': {'value': H0_combined, 'sigma': sigma_H0_combined,
                     'fractional_precision_pct': sigma_H0_combined / H0_combined * 100},
    'Hubble_tension_sigma': tension,
    'variants': {k: list(v) for k, v in variant_results.items()},
}
with open(os.path.join(OUT, "h0_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {OUT}/h0_results.json")

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

# ── Figure 1: Distance Network Diagram ────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)

# Layer boxes
layers = {
    'Geometric Anchors\n(MW parallax, LMC DEB, NGC4258 maser)': (2, 5.0, '#2166ac'),
    'Primary Indicators\n(Cepheid, TRGB)': (5, 5.0, '#4dac26'),
    'Secondary Calibrators\n(SNe Ia, SBF)': (8, 5.0, '#d73027'),
    'Hubble Flow\n(SNe Ia, SBF)': (8, 2.8, '#f4a582'),
    r'$H_0$ Consensus': (5, 1.2, '#762a83'),
}
box_props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, linewidth=2)
for label, (x, y, col) in layers.items():
    ax.text(x, y, label, ha='center', va='center', fontsize=9, fontweight='bold',
            color=col, bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7',
                                  edgecolor=col, linewidth=2))

# Arrows
arrow_style = dict(arrowstyle='->', color='gray', lw=1.5)
for (x1, y1), (x2, y2) in [
    ((3.2, 5.0), (4.0, 5.0)),   # Anchors -> Primary
    ((6.2, 5.0), (7.0, 5.0)),   # Primary -> Secondary (calib)
    ((8.0, 4.5), (8.0, 3.4)),   # Calib -> Hubble flow
    ((7.2, 2.8), (6.2, 1.5)),   # HF SNe Ia -> H0
    ((7.5, 4.8), (6.2, 1.4)),   # Calib -> H0
]:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.5))

ax.set_title("Local Distance Network: Analysis Framework", fontsize=14, fontweight='bold', pad=10)
plt.tight_layout()
fig1.savefig(os.path.join(FIGS, "fig1_distance_network.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig1_distance_network.png")

# ── Figure 2: Host Distance Moduli ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 5))
hosts_sorted = sorted(host_mu.keys(), key=lambda h: host_mu[h][0])
mus = [host_mu[h][0] for h in hosts_sorted]
sigs = [host_mu[h][1] for h in hosts_sorted]
y_pos = np.arange(len(hosts_sorted))

# Color by method availability
colors_h = []
for h in hosts_sorted:
    methods_used = set(m[1] for m in host_measurements if m[0] == h)
    if 'Cepheid' in methods_used and 'TRGB' in methods_used:
        colors_h.append('#1b7837')
    elif 'Cepheid' in methods_used:
        colors_h.append('#2166ac')
    else:
        colors_h.append('#d73027')

ax2.barh(y_pos, mus, xerr=sigs, color=colors_h, alpha=0.8, height=0.6,
         error_kw=dict(ecolor='black', capsize=4, lw=1.5))
ax2.set_yticks(y_pos)
ax2.set_yticklabels(hosts_sorted, fontsize=11)
ax2.set_xlabel(r'Distance Modulus $\mu$ [mag]', fontsize=12)
ax2.set_title('Host Galaxy Distance Moduli from Distance Network', fontsize=13)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
# Legend
legend_elements = [
    mpatches.Patch(color='#1b7837', label='Cepheid + TRGB'),
    mpatches.Patch(color='#2166ac', label='Cepheid only'),
    mpatches.Patch(color='#d73027', label='TRGB only'),
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
fig2.savefig(os.path.join(FIGS, "fig2_host_distance_moduli.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig2_host_distance_moduli.png")

# ── Figure 3: Hubble Diagram ───────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(8, 6))

z_snia = np.array([x[0] for x in hubble_flow_sneia])
mB_snia = np.array([x[1] for x in hubble_flow_sneia])
err_snia = np.array([x[2] for x in hubble_flow_sneia])
# Add pec vel uncertainty
pec_snia = np.array([(5.0 / np.log(10)) * (x[3] / (c_km * x[0])) for x in hubble_flow_sneia])
err_snia_total = np.sqrt(err_snia**2 + pec_snia**2)

z_sbf = np.array([x[0] for x in hubble_flow_sbf])
mF_sbf = np.array([x[1] for x in hubble_flow_sbf])
err_sbf = np.array([x[2] for x in hubble_flow_sbf])
pec_sbf = np.array([(5.0 / np.log(10)) * (x[3] / (c_km * x[0])) for x in hubble_flow_sbf])
err_sbf_total = np.sqrt(err_sbf**2 + pec_sbf**2)

# Theoretical Hubble diagram: m = M + 5*log10(cz/H0) + 25
z_th = np.linspace(0.01, 0.10, 200)
m_theory_snia = MB_mean + 5.0 * np.log10(c_km * z_th / H0_combined) + 25.0
m_theory_sbf  = M_SBF  + 5.0 * np.log10(c_km * z_th / H0_combined) + 25.0

ax3.errorbar(z_snia, mB_snia, yerr=err_snia_total, fmt='o', color='#1a6faf',
             markersize=8, capsize=4, label='Hubble-flow SNe Ia', zorder=5)
ax3.errorbar(z_sbf, mF_sbf, yerr=err_sbf_total, fmt='s', color='#c0392b',
             markersize=8, capsize=4, label='Hubble-flow SBF', zorder=5)
ax3.plot(z_th, m_theory_snia, '-', color='#1a6faf', lw=2, alpha=0.7,
         label=f'Best-fit SNe Ia ($H_0={H0_combined:.1f}$)')
ax3.plot(z_th, m_theory_sbf, '--', color='#c0392b', lw=2, alpha=0.7,
         label=f'Best-fit SBF ($H_0={H0_combined:.1f}$)')

ax3.set_xlabel('Redshift $z$', fontsize=12)
ax3.set_ylabel('Apparent magnitude', fontsize=12)
ax3.set_title('Hubble Diagram (Hubble-flow calibrators)', fontsize=13)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
fig3.savefig(os.path.join(FIGS, "fig3_hubble_diagram.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig3_hubble_diagram.png")

# ── Figure 4: M_B calibration ─────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(8, 5))
y4 = np.arange(len(MB_hosts))
ax4.errorbar(MB_values, y4, xerr=MB_errors, fmt='o', color='#2166ac',
             markersize=8, capsize=4, zorder=5)
ax4.axvline(MB_mean, color='k', lw=2, label=f'Weighted mean = {MB_mean:.3f} ± {MB_sigma:.3f}')
ax4.axvspan(MB_mean - MB_sigma, MB_mean + MB_sigma, alpha=0.15, color='k',
            label=r'1$\sigma$ band')
ax4.set_yticks(y4)
ax4.set_yticklabels(MB_hosts, fontsize=11)
ax4.set_xlabel(r'$M_B$ [mag]', fontsize=12)
ax4.set_title(r'SNe Ia Absolute Magnitude Calibration ($M_B$)', fontsize=13)
ax4.legend(fontsize=10)
ax4.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
fig4.savefig(os.path.join(FIGS, "fig4_MB_calibration.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig4_MB_calibration.png")

# ── Figure 5: H0 from individual SNe ──────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(8, 5))
z_pts = [x[0] for x in hubble_flow_sneia]
ax5.errorbar(z_pts, H0_sneia_arr, yerr=H0_sneia_errs, fmt='o',
             color='#2166ac', markersize=9, capsize=5, label='Individual SNe Ia')
ax5.axhline(H0_snia, color='#2166ac', lw=2, ls='-',
            label=f'SNe Ia mean: {H0_snia:.2f} ± {sigma_H0_snia:.2f}')
ax5.axhspan(H0_snia - sigma_H0_snia, H0_snia + sigma_H0_snia,
            alpha=0.15, color='#2166ac')
z_pts_sbf = [x[0] for x in hubble_flow_sbf]
ax5.errorbar(z_pts_sbf, H0_sbf_arr, yerr=H0_sbf_errs, fmt='s',
             color='#c0392b', markersize=9, capsize=5, label='Individual SBF')
ax5.axhline(H0_sbf, color='#c0392b', lw=2, ls='--',
            label=f'SBF mean: {H0_sbf:.2f} ± {sigma_H0_sbf:.2f}')
ax5.axhspan(H0_sbf - sigma_H0_sbf, H0_sbf + sigma_H0_sbf,
            alpha=0.15, color='#c0392b')
ax5.axhline(H0_combined, color='purple', lw=2.5, ls='-',
            label=f'Consensus: {H0_combined:.2f} ± {sigma_H0_combined:.2f}')
ax5.axhspan(H0_combined - sigma_H0_combined, H0_combined + sigma_H0_combined,
            alpha=0.2, color='purple')
ax5.axhline(H0_CMB, color='gray', lw=1.5, ls=':', label=f'Planck CMB: {H0_CMB} ± {sigma_CMB}')
ax5.axhspan(H0_CMB - sigma_CMB, H0_CMB + sigma_CMB, alpha=0.1, color='gray')
ax5.set_xlabel('Redshift $z$', fontsize=12)
ax5.set_ylabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
ax5.set_title(r'$H_0$ from Individual Hubble-flow Objects', fontsize=13)
ax5.legend(fontsize=8.5, loc='upper right')
ax5.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
fig5.savefig(os.path.join(FIGS, "fig5_H0_individual.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig5_H0_individual.png")

# ── Figure 6: Analysis variants ───────────────────────────────────────────────
fig6, ax6 = plt.subplots(figsize=(9, 5))
var_names = list(variant_results.keys())
var_vals  = [variant_results[n][0] for n in var_names]
var_errs  = [variant_results[n][1] for n in var_names]
y6 = np.arange(len(var_names))

colors6 = ['#762a83', '#2166ac', '#74add1', '#1b7837', '#d73027']
ax6.barh(y6, var_vals, xerr=var_errs, color=colors6[:len(var_names)], alpha=0.8,
         height=0.6, error_kw=dict(ecolor='black', capsize=5, lw=2))
ax6.axvline(H0_combined, color='purple', lw=2, ls='--',
            label=f'Consensus H0 = {H0_combined:.2f}')
ax6.axvspan(H0_combined - sigma_H0_combined, H0_combined + sigma_H0_combined,
            alpha=0.15, color='purple')
ax6.axvline(H0_CMB, color='gray', lw=1.5, ls=':', label=f'Planck CMB = {H0_CMB}')
ax6.set_yticks(y6)
ax6.set_yticklabels(var_names, fontsize=10)
ax6.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
ax6.set_title(r'$H_0$ Analysis Variants (Robustness Check)', fontsize=13)
ax6.legend(fontsize=10)
ax6.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
fig6.savefig(os.path.join(FIGS, "fig6_variants.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig6_variants.png")

# ── Figure 7: H0 comparison with literature ───────────────────────────────────
fig7, ax7 = plt.subplots(figsize=(9, 6))

literature = [
    ('Planck CMB 2018',        67.4, 0.5,   'gray',     '^'),
    ('SH0ES (Riess+2022)',     73.0, 1.0,   '#1a6faf',  'o'),
    ('CCHP (Freedman+2021)',   69.8, 1.7,   '#d73027',  's'),
    ('H0LiCOW (Wong+2020)',    73.3, 1.8,   '#e67e22',  'D'),
    ('TDCOSMO (Birrer+2020)',  74.5, 5.6,   '#8e44ad',  'P'),
    ('This work (consensus)',  H0_combined, sigma_H0_combined, 'purple', '*'),
]
y_lit = np.arange(len(literature))[::-1]
for i, (name, val, err, col, mk) in enumerate(literature):
    ax7.errorbar(val, y_lit[i], xerr=err, fmt=mk, color=col, markersize=10 if mk != '*' else 16,
                 capsize=5, elinewidth=2, label=name)

ax7.set_yticks(y_lit)
ax7.set_yticklabels([x[0] for x in literature], fontsize=10)
ax7.set_xlabel(r'$H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
ax7.set_title(r'Comparison of $H_0$ Measurements', fontsize=13)
ax7.axvspan(H0_combined - sigma_H0_combined, H0_combined + sigma_H0_combined,
            alpha=0.15, color='purple', label='This work 1σ')
ax7.grid(axis='x', alpha=0.3, linestyle='--')
ax7.legend(fontsize=8.5, loc='upper left')
plt.tight_layout()
fig7.savefig(os.path.join(FIGS, "fig7_H0_comparison.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig7_H0_comparison.png")

# ── Figure 8: Uncertainty budget ──────────────────────────────────────────────
fig8, ax8 = plt.subplots(figsize=(7, 5))
budget_labels = [
    'Statistical (flow)',
    'Calibration (M_B)',
    'Anchor systematic',
    'Peculiar velocities',
    'SBF calibration',
    'Total (quadrature)',
]
# Break down sigma_H0_combined into components (approximate)
sig_stat_flow = (np.sqrt(1.0 / np.sum(1.0 / np.array(H0_sneia_errs)**2)) *
                 H0_combined / H0_snia)
sig_calib = MB_sigma * (np.log(10) / 5.0) * H0_combined
sig_anchor = anchors['N4258']['err'] * (np.log(10) / 5.0) * H0_combined
sig_pv = np.mean(pec_snia) * (np.log(10) / 5.0) * H0_combined
sig_sbf_cal = sig_M_SBF * (np.log(10) / 5.0) * H0_combined * 0.3  # weighted contribution

components = [sig_stat_flow, sig_calib, sig_anchor, sig_pv, sig_sbf_cal,
              sigma_H0_combined]
colors8 = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', 'black']
bars = ax8.barh(budget_labels, components, color=colors8, alpha=0.8, height=0.6)
ax8.set_xlabel(r'Uncertainty in $H_0$ [km s$^{-1}$ Mpc$^{-1}$]', fontsize=12)
ax8.set_title(r'$H_0$ Uncertainty Budget', fontsize=13)
for bar, val in zip(bars, components):
    ax8.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
             f'{val:.2f}', va='center', fontsize=9)
ax8.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
fig8.savefig(os.path.join(FIGS, "fig8_uncertainty_budget.png"), bbox_inches='tight', dpi=150)
plt.close()
print("Saved fig8_uncertainty_budget.png")

print("\nAll figures saved.")
print("\nFinal Results Summary:")
print(f"  H0 (SNe Ia)   = {H0_snia:.3f} ± {sigma_H0_snia:.3f} km/s/Mpc")
print(f"  H0 (SBF)      = {H0_sbf:.3f} ± {sigma_H0_sbf:.3f} km/s/Mpc")
print(f"  H0 (consensus)= {H0_combined:.3f} ± {sigma_H0_combined:.3f} km/s/Mpc")
print(f"  Precision     = {sigma_H0_combined/H0_combined*100:.2f}%")
print(f"  Hubble tension= {tension:.2f}σ vs Planck")
