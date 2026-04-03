import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple

# -----------------------------
# Load minimal dataset (Python literal file)
# -----------------------------

def load_minimal_dataset(path: str):
    """Executes the minimal dataset file in an isolated namespace
    and returns the defined variables as a dict.
    """
    ns: Dict[str, object] = {}
    with open(path, 'r') as f:
        code = f.read()
    exec(code, {}, ns)
    return ns


# -----------------------------
# Basic utilities
# -----------------------------

C_KM_S = 299792.458  # km/s


def mu_to_distance_mpc(mu: float) -> float:
    """Distance modulus to distance in Mpc."""
    return 10 ** ((mu - 25.0) / 5.0)


def distance_mpc_to_mu(d_mpc: float) -> float:
    return 5 * np.log10(d_mpc) + 25.0


# -----------------------------
# Anchors and host distances
# -----------------------------

@dataclass
class Anchor:
    name: str
    mu: float
    sigma: float


@dataclass
class HostMeasurement:
    host: str
    method: str
    anchor: str
    mu_meas: float
    sigma_meas: float


def build_host_distance_estimates(anchors: Dict[str, Anchor],
                                  host_measurements: List[HostMeasurement],
                                  method_anchor_err: Dict[Tuple[str, str], float]):
    """Combine multiple measurements for each host using inverse-variance weighting.

    We treat the anchor modulus as an external parameter and propagate
    method+anchor calibration uncertainties.
    Returns a dict host -> (mu_host, sigma_host).
    """
    by_host: Dict[str, List[Tuple[float, float]]] = {}
    for m in host_measurements:
        cal_err = method_anchor_err.get((m.method, m.anchor), 0.0)
        sigma_tot = np.sqrt(m.sigma_meas**2 + cal_err**2)
        by_host.setdefault(m.host, []).append((m.mu_meas, sigma_tot))

    host_mu: Dict[str, Tuple[float, float]] = {}
    for host, lst in by_host.items():
        mus = np.array([x[0] for x in lst])
        sig = np.array([x[1] for x in lst])
        w = 1.0 / sig**2
        mu_comb = np.sum(w * mus) / np.sum(w)
        sigma_comb = np.sqrt(1.0 / np.sum(w))
        host_mu[host] = (mu_comb, sigma_comb)
    return host_mu


# -----------------------------
# SN Ia calibration and H0 fit
# -----------------------------

@dataclass
class SNCalibrator:
    host: str
    mB: float
    sigma_mB: float


@dataclass
class HubbleFlowSN:
    z: float
    mB: float
    sigma_mB: float
    sigma_vpec: float


def calibrate_snia_Mb(calibrators: List[SNCalibrator],
                      host_mu: Dict[str, Tuple[float, float]]):
    """Infer the absolute magnitude M_B of SNe Ia from calibrator hosts.

    m_B = mu_host + M_B  =>  M_B = m_B - mu_host.
    We propagate both host distance and apparent magnitude errors.
    Returns (M_B, sigma_M_B).
    """
    Mb_vals = []
    sig_vals = []
    for cal in calibrators:
        if cal.host not in host_mu:
            continue
        mu_h, sig_mu = host_mu[cal.host]
        Mb_i = cal.mB - mu_h
        sigma_i = np.sqrt(cal.sigma_mB**2 + sig_mu**2)
        Mb_vals.append(Mb_i)
        sig_vals.append(sigma_i)

    Mb_vals = np.array(Mb_vals)
    sig_vals = np.array(sig_vals)
    w = 1.0 / sig_vals**2
    Mb = np.sum(w * Mb_vals) / np.sum(w)
    sigma_Mb = np.sqrt(1.0 / np.sum(w))
    return Mb, sigma_Mb


def fit_H0_from_snia(hflow: List[HubbleFlowSN], Mb: float):
    """Fit H0 using a simple Hubble-law approximation in the nearby universe.

    For each SN, distance modulus is
        mu = m_B - M_B.
    Convert to luminosity distance d_L, and equate cz = H0 d_L (in km/s, Mpc).
    We ignore cosmological curvature at these low redshifts.

    We perform a weighted least-squares fit in velocity-distance space,
    taking into account photometric and peculiar-velocity uncertainties.
    """
    d_list = []
    sigma_d_list = []
    v_list = []
    sigma_v_list = []

    for sn in hflow:
        mu = sn.mB - Mb
        # propagate only photometric uncertainty; peculiar velocity goes into velocity error
        # assume 0.1 mag intrinsic SN Ia scatter
        sigma_mu_phot = np.sqrt(sn.sigma_mB**2 + 0.10**2)
        d_mpc = mu_to_distance_mpc(mu)
        # derivative d d / d mu = (ln(10)/5) * d
        sigma_d = (np.log(10) / 5.0) * d_mpc * sigma_mu_phot

        v = C_KM_S * sn.z
        sigma_v = sn.sigma_vpec

        d_list.append(d_mpc)
        sigma_d_list.append(sigma_d)
        v_list.append(v)
        sigma_v_list.append(sigma_v)

    d = np.array(d_list)
    sig_d = np.array(sigma_d_list)
    v = np.array(v_list)
    sig_v = np.array(sigma_v_list)

    # model: v = H0 * d
    # variance in v from both sides: sigma_v^2 + (H0^2 * sig_d^2).
    # For simplicity, start with ignoring sig_d when defining weights,
    # then iterate once using the best-fit H0.

    w = 1.0 / (sig_v**2)
    H0_init = np.sum(w * d * v) / np.sum(w * d**2)

    # update weights including distance uncertainty
    w2 = 1.0 / (sig_v**2 + (H0_init**2) * sig_d**2)
    H0 = np.sum(w2 * d * v) / np.sum(w2 * d**2)
    sigma_H0 = np.sqrt(1.0 / np.sum(w2 * d**2))
    return H0, sigma_H0, d, v, sig_d, sig_v


# -----------------------------
# SBF calibration and H0 (optional cross-check)
# -----------------------------

@dataclass
class SBFGal:
    host: str
    mF110W: float
    sigma_m: float


@dataclass
class HubbleFlowSBF:
    z: float
    mF110W: float
    sigma_m: float
    sigma_vpec: float


def analyze_sbf(sbf_calibrators: List[SBFGal],
                 hubble_flow_sbf: List[HubbleFlowSBF],
                 host_group: Dict[str, str]):
    """Very simplified SBF H0 using group-averaged zero point.

    This is included as a secondary cross-check, not a full treatment of
    the depth/cluster covariance used in the real Distance Network.
    We assume calibrators define an average absolute magnitude M_SBF.
    """
    # Assume a fiducial distance modulus for Fornax and Virgo to derive M_SBF.
    # These are rough and illustrative.
    group_mu = {
        'Fornax': 31.5,
        'Virgo': 31.1,
    }

    Ms = []
    sigs = []
    for gal in sbf_calibrators:
        g = host_group[gal.host]
        mu_g = group_mu[g]
        M = gal.mF110W - mu_g
        sigs.append(gal.sigma_m)
        Ms.append(M)
    Ms = np.array(Ms)
    sigs = np.array(sigs)
    w = 1.0 / sigs**2
    M_sbf = np.sum(w * Ms) / np.sum(w)
    sigma_Msbf = np.sqrt(1.0 / np.sum(w))

    # Hubble-flow SBF galaxies
    d_list = []
    sig_d_list = []
    v_list = []
    sig_v_list = []

    for g in hubble_flow_sbf:
        mu = g.mF110W - M_sbf
        sigma_mu = np.sqrt(g.sigma_m**2 + sigma_Msbf**2)
        d = mu_to_distance_mpc(mu)
        sigma_d = (np.log(10) / 5.0) * d * sigma_mu
        v = C_KM_S * g.z
        sigma_v = g.sigma_vpec
        d_list.append(d)
        sig_d_list.append(sigma_d)
        v_list.append(v)
        sig_v_list.append(sigma_v)

    d = np.array(d_list)
    sig_d = np.array(sig_d_list)
    v = np.array(v_list)
    sig_v = np.array(sig_v_list)

    w = 1.0 / (sig_v**2)
    H0_init = np.sum(w * d * v) / np.sum(w * d**2)
    w2 = 1.0 / (sig_v**2 + (H0_init**2) * sig_d**2)
    H0 = np.sum(w2 * d * v) / np.sum(w2 * d**2)
    sigma_H0 = np.sqrt(1.0 / np.sum(w2 * d**2))
    return H0, sigma_H0, d, v, sig_d, sig_v


# -----------------------------
# Visualization helpers
# -----------------------------


def plot_host_distances(host_mu: Dict[str, Tuple[float, float]], outpath: str):
    hosts = list(host_mu.keys())
    mus = np.array([host_mu[h][0] for h in hosts])
    sig = np.array([host_mu[h][1] for h in hosts])
    order = np.argsort(mus)
    hosts = [hosts[i] for i in order]
    mus = mus[order]
    sig = sig[order]

    plt.figure(figsize=(7, 4))
    plt.errorbar(mus, np.arange(len(hosts)), xerr=sig, fmt='o', color='C0')
    plt.yticks(np.arange(len(hosts)), hosts)
    plt.xlabel("Distance modulus mu (mag)")
    plt.title("Host distance estimates (combined)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_hubble_diagram(d, v, sig_d, sig_v, H0, outpath: str, label: str):
    plt.figure(figsize=(6, 5))
    plt.errorbar(d, v, xerr=sig_d, yerr=sig_v, fmt='o', label=label)
    d_line = np.linspace(0, max(d)*1.1, 100)
    plt.plot(d_line, H0 * d_line, 'k--', label=fr"Fit: $H_0={H0:.1f}$ km s$^{{-1}}$ Mpc$^{{-1}}$")
    plt.xlabel("Distance (Mpc)")
    plt.ylabel("Recession velocity (km/s)")
    plt.legend()
    plt.title("Hubble diagram")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -----------------------------
# Main analysis pipeline
# -----------------------------


def main():
    base = '.'
    data_path = base + '/data/H0DN_MinimalDataset.txt'
    ns = {}
    with open(data_path, 'r') as f:
        code = f.read()
    exec(code, {}, ns)

    anchors_raw = ns['anchors']
    host_meas_raw = ns['host_measurements']
    sneia_cal_raw = ns['sneia_calibrators']
    sbf_cal_raw = ns['sbf_calibrators']
    hubble_sneia_raw = ns['hubble_flow_sneia']
    hubble_sbf_raw = ns['hubble_flow_sbf']
    method_anchor_err = ns['method_anchor_err']

    anchors = {k: Anchor(k, v['mu'], v['err']) for k, v in anchors_raw.items()}
    host_measurements = [HostMeasurement(*t) for t in host_meas_raw]
    sneia_calibrators = [SNCalibrator(*t) for t in sneia_cal_raw]
    hubble_sneia = [HubbleFlowSN(*t) for t in hubble_sneia_raw]

    # SBF structures
    sbf_cals = [SBFGal(*t) for t in sbf_cal_raw]
    hubble_sbf = [HubbleFlowSBF(*t) for t in hubble_sbf_raw]

    # Host distances
    host_mu = build_host_distance_estimates(anchors, host_measurements, method_anchor_err)

    # SN Ia calibration
    Mb, sigma_Mb = calibrate_snia_Mb(sneia_calibrators, host_mu)

    # H0 from SNe Ia
    H0_sn, sigma_H0_sn, d_sn, v_sn, sigd_sn, sigv_sn = fit_H0_from_snia(hubble_sneia, Mb)

    # H0 from SBF cross-check
    host_group = ns['host_group']
    H0_sbf, sigma_H0_sbf, d_sbf, v_sbf, sigd_sbf, sigv_sbf = analyze_sbf(sbf_cals, hubble_sbf, host_group)

    # Save key numerical results
    import json, os
    os.makedirs('outputs', exist_ok=True)
    results = {
        'Mb_SNIa': Mb,
        'sigma_Mb_SNIa': sigma_Mb,
        'H0_SNIa': H0_sn,
        'sigma_H0_SNIa': sigma_H0_sn,
        'H0_SBF': H0_sbf,
        'sigma_H0_SBF': sigma_H0_sbf,
    }
    with open('outputs/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Figures
    os.makedirs('report/images', exist_ok=True)
    plot_host_distances(host_mu, 'report/images/host_distances.png')
    plot_hubble_diagram(d_sn, v_sn, sigd_sn, sigv_sn, H0_sn,
                        'report/images/hubble_snia.png', 'SNe Ia')
    plot_hubble_diagram(d_sbf, v_sbf, sigd_sbf, sigv_sbf, H0_sbf,
                        'report/images/hubble_sbf.png', 'SBF')

    # Also save simple table of H0 results
    with open('outputs/h0_summary.txt', 'w') as f:
        f.write(f"H0 from SNe Ia: {H0_sn:.2f} +- {sigma_H0_sn:.2f} km/s/Mpc\n")
        f.write(f"H0 from SBF : {H0_sbf:.2f} +- {sigma_H0_sbf:.2f} km/s/Mpc\n")


if __name__ == '__main__':
    main()
