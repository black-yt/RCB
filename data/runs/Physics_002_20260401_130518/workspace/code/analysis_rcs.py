import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
AMPLITUDES_DIR = DATA_DIR / "amplitudes"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = BASE_DIR / "report" / "images"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_N_d_r_from_filename(name: str):
    # Handles patterns like N40_d10_r10_XEB_counts.json or N40_d12_r1_Transport_1QRB_counts.json
    base = os.path.basename(name)
    parts = base.split("_")
    N = int(parts[0][1:])  # strip leading 'N'
    d = int(parts[1][1:])  # strip leading 'd'
    r_part = [p for p in parts if p.startswith("r")][0]
    r = int(r_part[1:])
    return N, d, r


def compute_xeb_fidelity(counts_dict, ideal_probs_dict):
    """Compute linear XEB fidelity estimate and a simple bootstrap uncertainty.

    Args:
        counts_dict: {bitstring: count}
        ideal_probs_dict: {bitstring: p_ideal} over a subset of bitstrings.

    Returns:
        mean_f: estimated fidelity
        std_f: bootstrap std dev over samples
        nsamples: total shots used
    """
    # Intersect keys
    common_bits = list(set(counts_dict.keys()) & set(ideal_probs_dict.keys()))
    if len(common_bits) == 0:
        return np.nan, np.nan, 0

    counts = np.array([counts_dict[b] for b in common_bits], dtype=float)
    probs_raw = [ideal_probs_dict[b] for b in common_bits]
    probs = np.array([
        abs(complex(p)) ** 2 if isinstance(p, (complex, str)) else float(p)
        for p in probs_raw
    ], dtype=float)

    nsamples = counts.sum()
    # Normalize subset probabilities just in case
    probs = probs / probs.sum()

    # Expand into per-shot samples for bootstrap
    probs_per_shot = np.repeat(probs, counts.astype(int))

    # Linear XEB estimator: F = 2^n * mean(p_ideal) - 1, but n is unknown from local data.
    # Here we approximate using the subset dimension as an effective dimension D_eff.
    D_eff = len(probs)
    mean_p = probs_per_shot.mean()
    f_est = D_eff * mean_p - 1.0

    # Bootstrap
    if nsamples > 1:
        n_boot = int(min(200, nsamples))
        rng = np.random.default_rng(123)
        boot_vals = []
        for _ in range(n_boot):
            sample = rng.choice(probs_per_shot, size=len(probs_per_shot), replace=True)
            boot_vals.append(D_eff * sample.mean() - 1.0)
        std_f = float(np.std(boot_vals, ddof=1))
    else:
        std_f = np.nan

    return float(f_est), std_f, int(nsamples)


def analyze_N40_verification():
    records = []
    base_results = RESULTS_DIR / "N40_verification"
    base_amps = AMPLITUDES_DIR / "N40_verification"

    for depth_dir in sorted(base_results.glob("N40_d*_XEB")):
        d_str = depth_dir.name.split("_")[1]  # e.g. d10
        d = int(d_str[1:])
        amp_depth_dir = base_amps / depth_dir.name
        if not amp_depth_dir.exists():
            continue

        for counts_path in sorted(depth_dir.glob("*_counts.json")):
            N, d_parsed, r = parse_N_d_r_from_filename(counts_path.name)
            assert d_parsed == d
            amp_path = amp_depth_dir / counts_path.name.replace("_counts", "_amplitudes")
            if not amp_path.exists():
                continue

            counts_data = load_json(counts_path)
            amps_data = load_json(amp_path)

            # Flexible handling: amplitudes file may contain amplitudes or probabilities
            if "ideal_probs" in amps_data:
                ideal_probs = {k: float(v) for k, v in amps_data["ideal_probs"].items()}
            elif "probs" in amps_data:
                ideal_probs = {k: float(v) for k, v in amps_data["probs"].items()}
            elif "amplitudes" in amps_data:
                # convert amplitudes (complex, stored as strings) to probabilities
                ideal_probs = {}
                for k, v in amps_data["amplitudes"].items():
                    if isinstance(v, str):
                        # literal complex string like '(a+bj)'
                        amp = complex(v)
                    else:
                        amp = complex(v)
                    ideal_probs[k] = float(abs(amp) ** 2)
            else:
                # assume direct mapping
                ideal_probs = amps_data

            # counts_data could be a dict of bitstring->count or have a key
            if isinstance(counts_data, dict) and all(isinstance(v, int) for v in counts_data.values()):
                counts = counts_data
            elif "counts" in counts_data:
                counts = counts_data["counts"]
            else:
                counts = counts_data

            f_mean, f_std, nsamples = compute_xeb_fidelity(counts, ideal_probs)

            records.append(
                {
                    "N": N,
                    "d": d,
                    "r": r,
                    "f_xeb": f_mean,
                    "f_xeb_std": f_std,
                    "nsamples": nsamples,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "N40_verification_xeb.csv", index=False)
    return df


def analyze_N_scan_depth12():
    """Analyze N-scan data at fixed depth (e.g., d=12) using transport-style circuits.

    This uses full ideal bitstring information provided per instance for fidelity-like metrics
    (here we compute the fraction of shots matching the ideal bitstring as a simple proxy).
    """
    base_results = RESULTS_DIR / "N_scan_depth12"
    records = []

    for N_dir in sorted(base_results.glob("N*_d*_Transport_1QRB")):
        for counts_path in sorted(N_dir.glob("*_counts.json")):
            N, d, r = parse_N_d_r_from_filename(counts_path.name)
            ideal_path = counts_path.with_name(counts_path.name.replace("_counts", "_ideal_bitstring"))
            if not ideal_path.exists():
                continue
            counts_data = load_json(counts_path)
            ideal_data = load_json(ideal_path)

            if "counts" in counts_data:
                counts = counts_data["counts"]
            else:
                counts = counts_data

            # ideal_data may be a dict with key 'bitstring' or a raw bitstring/list
            if isinstance(ideal_data, dict) and "bitstring" in ideal_data:
                ideal_bs = ideal_data["bitstring"]
            else:
                ideal_bs = ideal_data

            # Convert ideal bitstring list to a hashable tuple or bitstring key
            if isinstance(ideal_bs, list):
                ideal_key = tuple(ideal_bs)
            else:
                ideal_key = ideal_bs

            total_shots = sum(counts.values())
            match_counts = counts.get(ideal_key, 0)
            frac_match = match_counts / total_shots if total_shots > 0 else np.nan

            records.append(
                {
                    "N": N,
                    "d": d,
                    "r": r,
                    "match_frac": frac_match,
                    "total_shots": total_shots,
                }
            )

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / "N_scan_depth12_match_frac.csv", index=False)
    return df


def plot_results(df_N40, df_Nscan):
    sns.set(style="whitegrid", context="talk")

    # Plot XEB fidelity vs depth for N=40
    plt.figure(figsize=(8, 5))
    df40 = df_N40[df_N40["N"] == 40]
    grouped = df40.groupby("d")["f_xeb"].agg(["mean", "std"]).reset_index()
    plt.errorbar(grouped["d"], grouped["mean"], yerr=grouped["std"], fmt="o-", capsize=3)
    plt.xlabel("Circuit depth d")
    plt.ylabel("XEB fidelity (arb. units)")
    plt.title("N=40 random circuit sampling: XEB vs depth")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "N40_xeb_vs_depth.png", dpi=300)
    plt.close()

    # Plot match fraction vs N for fixed depths (e.g., d=12, 32, 48, 64, 96)
    plt.figure(figsize=(8, 5))
    for d in sorted(df_Nscan["d"].unique()):
        sub = df_Nscan[df_Nscan["d"] == d]
        grouped = sub.groupby("N")["match_frac"].mean().reset_index()
        plt.plot(grouped["N"], grouped["match_frac"], marker="o", label=f"d={d}")
    plt.xlabel("Number of qubits N")
    plt.ylabel("Ideal-bitstring match fraction")
    plt.title("N-scan transport circuits: match fraction vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "Nscan_match_frac_vs_N.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df_N40 = analyze_N40_verification()
    df_Nscan = analyze_N_scan_depth12()
    plot_results(df_N40, df_Nscan)
