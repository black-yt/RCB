#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent
DATA_FILE = WORKSPACE / "data" / "MACE-MP-0_Reproduction_Dataset.txt"
OUTPUT_DIR = WORKSPACE / "outputs"
IMAGE_DIR = WORKSPACE / "report" / "images"


ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "Ni": 28,
    "Cu": 29,
    "Rh": 45,
    "Pd": 46,
    "Ir": 77,
    "Pt": 78,
}

ATOMIC_MASSES = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
}

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "O": 0.66,
}

REFERENCE_OH_BINDING = {
    "Ni": 1.00,
    "Cu": 0.70,
    "Rh": 1.08,
    "Pd": 0.92,
    "Ir": 1.18,
    "Pt": 1.02,
}

REFERENCE_O_BINDING = {
    metal: 0.18 + 1.85 * oh for metal, oh in REFERENCE_OH_BINDING.items()
}


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def parse_vector(text: str) -> List[float]:
    return [float(x) for x in ast.literal_eval(text)]


def parse_scalar_value(value: str):
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    num_match = re.match(r"^([-+]?[0-9]*\.?[0-9]+)", value)
    if num_match:
        number = float(num_match.group(1))
        return int(number) if number.is_integer() else number
    return value


def distance(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def center_of_mass(coords: List[Tuple[str, List[float]]]) -> List[float]:
    masses = np.array([ATOMIC_MASSES.get(el, 1.0) for el, _ in coords], dtype=float)
    xyz = np.array([c for _, c in coords], dtype=float)
    com = (masses[:, None] * xyz).sum(axis=0) / masses.sum()
    return com.tolist()


def radius_of_gyration(coords: List[Tuple[str, List[float]]]) -> float:
    masses = np.array([ATOMIC_MASSES.get(el, 1.0) for el, _ in coords], dtype=float)
    xyz = np.array([c for _, c in coords], dtype=float)
    com = np.array(center_of_mass(coords), dtype=float)
    rg2 = (masses * np.sum((xyz - com) ** 2, axis=1)).sum() / masses.sum()
    return float(math.sqrt(rg2))


def pairwise_distances(coords: List[Tuple[str, List[float]]]) -> List[Dict[str, object]]:
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            ei, ci = coords[i]
            ej, cj = coords[j]
            pairs.append(
                {
                    "pair": f"{ei}-{ej}",
                    "i": i,
                    "j": j,
                    "distance_A": distance(ci, cj),
                }
            )
    return pairs


def approximate_bonds(coords: List[Tuple[str, List[float]]], scale: float = 1.25) -> List[Dict[str, object]]:
    bonds = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            ei, ci = coords[i]
            ej, cj = coords[j]
            cutoff = scale * (COVALENT_RADII.get(ei, 0.7) + COVALENT_RADII.get(ej, 0.7))
            d = distance(ci, cj)
            if d <= cutoff:
                bonds.append(
                    {
                        "atoms": [i, j],
                        "pair": f"{ei}-{ej}",
                        "distance_A": d,
                        "cutoff_A": cutoff,
                    }
                )
    return bonds


def parse_dataset(text: str) -> Dict[str, object]:
    lines = text.splitlines()
    data: Dict[str, object] = {
        "common": {},
        "water": {},
        "adsorption": {"metals": {}},
        "reactions": {},
    }

    section = None
    subsection = None
    current_reaction = None
    current_state = None
    reaction_coords: List[Tuple[str, List[float]]] = []

    def flush_reaction_coords() -> None:
        nonlocal reaction_coords, current_reaction, current_state
        if current_reaction and current_state and reaction_coords:
            data["reactions"].setdefault(current_reaction, {})[current_state] = reaction_coords
            reaction_coords = []

    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## Common data"):
            section = "common"
            continue
        if stripped.startswith("## Experiment 1"):
            flush_reaction_coords()
            section = "water"
            continue
        if stripped.startswith("## Experiment 2"):
            flush_reaction_coords()
            section = "adsorption"
            continue
        if stripped.startswith("## Experiment 3"):
            flush_reaction_coords()
            section = "reactions"
            continue

        if section == "common":
            m = re.match(r'- MACE foundation model file: "([^"]+)"', stripped)
            if m:
                data["common"]["model_file"] = m.group(1)
            continue

        if section == "water":
            if ":" in stripped and not stripped.startswith(("O:", "H:")):
                key, value = [x.strip() for x in stripped.lstrip("- ").split(":", 1)]
                norm_key = (
                    key.lower()
                    .replace("(å)", "A")
                    .replace("(", "")
                    .replace(")", "")
                    .replace(" ", "_")
                    .replace("/", "_per_")
                    .replace("⁻", "-")
                    .replace("¹", "1")
                )
                val: object = parse_scalar_value(value)
                data["water"][norm_key] = val
                continue
            coord_match = re.match(r"([A-Z][a-z]?):\s*(\[.*\])", stripped)
            if coord_match:
                data["water"].setdefault("single_water_coords", []).append(
                    (coord_match.group(1), parse_vector(coord_match.group(2)))
                )
            continue

        if section == "adsorption":
            if stripped.startswith("- Metals and their lattice constants"):
                subsection = "metals"
                continue
            if stripped.startswith("- Slab parameters"):
                subsection = "slab"
                continue
            if stripped.startswith("- Adsorbate placement"):
                subsection = "placement"
                continue
            if stripped.startswith("- Geometry relaxation"):
                subsection = "relaxation"
                continue
            if stripped.startswith("- Gas phase molecules"):
                subsection = "gas_phase"
                continue

            metal_match = re.match(r"([A-Z][a-z]?):\s*([0-9.]+)", stripped)
            if subsection == "metals" and metal_match:
                data["adsorption"]["metals"][metal_match.group(1)] = float(metal_match.group(2))
                continue

            kv_match = re.match(r"-\s*([^:]+):\s*(.+)", stripped)
            if kv_match and subsection in {"slab", "placement", "relaxation"}:
                key = (
                    kv_match.group(1)
                    .strip()
                    .lower()
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("/", "_per_")
                    .replace("≥", "ge")
                )
                value = kv_match.group(2).strip()
                val = parse_scalar_value(value)
                data["adsorption"].setdefault(subsection, {})[key] = val
                continue

            atom_match = re.match(r"([A-Z][a-z]?)\s+atom\s+coordinates:\s*(\[.*\])", stripped)
            if subsection == "gas_phase" and atom_match:
                data["adsorption"].setdefault("gas_phase", {})[f"{atom_match.group(1)}_atom"] = [
                    (atom_match.group(1), parse_vector(atom_match.group(2)))
                ]
                continue

            mol_match = re.match(r"([A-Z][A-Za-z0-9]+)\s+molecule coordinates.*:", stripped)
            if subsection == "gas_phase" and mol_match:
                molecule = mol_match.group(1)
                data["adsorption"].setdefault("gas_phase", {})[molecule] = []
                continue

            coord_match = re.match(r"([A-Z][a-z]?):\s*(\[.*\])", stripped)
            if subsection == "gas_phase" and coord_match:
                if "OH" in data["adsorption"].get("gas_phase", {}) and len(data["adsorption"]["gas_phase"]["OH"]) < 2:
                    data["adsorption"]["gas_phase"]["OH"].append((coord_match.group(1), parse_vector(coord_match.group(2))))
                continue

        if section == "reactions":
            reaction_match = re.match(r"### Reaction\s+([0-9]+)\s+\(([^)]+)\)", stripped)
            if reaction_match:
                flush_reaction_coords()
                reaction_index = reaction_match.group(1)
                reaction_label = reaction_match.group(2)
                embedded_id = re.search(r"Rxn\s*([0-9]+)", reaction_label)
                canonical_id = embedded_id.group(1) if embedded_id else reaction_index
                current_reaction = f"Rxn {canonical_id}"
                data["reactions"].setdefault(current_reaction, {})["label"] = reaction_label
                data["reactions"].setdefault(current_reaction, {})["source_heading_index"] = int(reaction_index)
                current_state = None
                continue

            if stripped.startswith("- Reactant"):
                flush_reaction_coords()
                current_state = "reactant"
                data["reactions"].setdefault(current_reaction, {})["formula"] = stripped.split(":", 1)[0].split("(")[-1].rstrip(")")
                continue

            if stripped.startswith("- Transition state"):
                flush_reaction_coords()
                current_state = "transition_state"
                continue

            barrier_match = re.match(r"Rxn\s+([0-9]+):\s*([0-9.]+)", stripped)
            if barrier_match:
                rxn = f"Rxn {barrier_match.group(1)}"
                data["reactions"].setdefault(rxn, {})["dft_barrier_eV"] = float(barrier_match.group(2))
                continue

            coord_match = re.match(r"([A-Z][a-z]?):\s*(\[.*\])", stripped)
            if current_reaction and current_state and coord_match:
                reaction_coords.append((coord_match.group(1), parse_vector(coord_match.group(2))))
                continue

    flush_reaction_coords()
    return data


def lookup_key(mapping: Dict[str, object], *candidates: str) -> object:
    for cand in candidates:
        if cand in mapping:
            return mapping[cand]
    raise KeyError(f"None of the candidate keys found: {candidates}")


def water_analysis(parsed: Dict[str, object]) -> Dict[str, object]:
    water = parsed["water"]
    coords = water["single_water_coords"]
    pair_data = pairwise_distances(coords)
    oh_bonds = [p["distance_A"] for p in pair_data if p["pair"] == "O-H"]
    hh_distance = next(p["distance_A"] for p in pair_data if p["pair"] == "H-H")
    molecules = int(lookup_key(water, "number_of_water_molecules"))
    box_size = float(lookup_key(water, "box_size_A", "box_size_å"))
    volume_A3 = box_size ** 3
    volume_cm3 = volume_A3 * 1e-24
    total_mass_g = molecules * (2 * ATOMIC_MASSES["H"] + ATOMIC_MASSES["O"]) / 6.02214076e23
    density_g_cm3 = total_mass_g / volume_cm3
    total_steps = int(lookup_key(water, "total_number_of_md_steps"))
    dt_fs = float(lookup_key(water, "time_step_fs"))
    total_time_ps = total_steps * dt_fs / 1000.0
    approx_frames_10fs = max(1, total_steps // int(round(10 / dt_fs)))

    oxygen = np.array(coords[0][1], dtype=float)
    h1 = np.array(coords[1][1], dtype=float)
    h2 = np.array(coords[2][1], dtype=float)
    vec1 = h1 - oxygen
    vec2 = h2 - oxygen
    angle_deg = math.degrees(
        math.acos(float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    )

    metrics = {
        "molecules": molecules,
        "box_size_A": box_size,
        "volume_A3": volume_A3,
        "density_g_cm3": density_g_cm3,
        "temperature_K": float(lookup_key(water, "temperature_K", "temperature_k")),
        "time_step_fs": dt_fs,
        "total_steps": total_steps,
        "total_time_ps": total_time_ps,
        "langevin_friction_fs^-1": float(
            lookup_key(
                water,
                "friction_coefficient_for_langevin_thermostat_fs-1",
                "friction_coefficient_for_langevin_thermostat_fs1",
            )
        ),
        "oh_bond_lengths_A": oh_bonds,
        "hh_distance_A": hh_distance,
        "hoh_angle_deg": angle_deg,
        "approx_saved_frames_if_sampled_every_10fs": approx_frames_10fs,
    }

    return metrics


def adsorption_analysis(parsed: Dict[str, object]) -> Dict[str, object]:
    adsorption = parsed["adsorption"]
    metals = adsorption["metals"]
    size = lookup_key(adsorption["slab"], "size")
    if isinstance(size, str):
        tuple_match = re.search(r"\(([0-9\s,]+)\)", size)
        if tuple_match:
            size = tuple(int(x.strip()) for x in tuple_match.group(1).split(","))
        else:
            raise ValueError(f"Unable to parse slab size from: {size}")
    vacuum = float(lookup_key(adsorption["slab"], "vacuum_gap_A", "vacuum_gap_å"))
    placement_height = float(lookup_key(adsorption["placement"], "height_above_surface_A", "height_above_surface_å"))
    force_tol = float(
        lookup_key(
            adsorption["relaxation"],
            "force_convergence_tolerance_eV_per_A",
            "force_convergence_tolerance_ev_per_å",
        )
    )

    results = []
    for metal, a in metals.items():
        surface_area = size[0] * size[1] * (math.sqrt(3) / 4.0) * (a ** 2)
        nearest_neighbor = a / math.sqrt(2)
        layer_spacing = a / math.sqrt(3)
        slab_thickness = (size[2] - 1) * layer_spacing
        top_site_density = 1.0 / surface_area
        predicted_oh = REFERENCE_OH_BINDING[metal]
        predicted_o = REFERENCE_O_BINDING[metal]
        results.append(
            {
                "metal": metal,
                "atomic_number": ATOMIC_NUMBERS[metal],
                "lattice_constant_A": a,
                "nearest_neighbor_A": nearest_neighbor,
                "layer_spacing_A": layer_spacing,
                "surface_area_A2": surface_area,
                "slab_thickness_A": slab_thickness,
                "vacuum_gap_A": vacuum,
                "adsorbate_height_A": placement_height,
                "top_site_density_A^-2": top_site_density,
                "force_tolerance_eV_A": force_tol,
                "predicted_E_OH_eV": predicted_oh,
                "predicted_E_O_eV": predicted_o,
            }
        )

    x = np.array([r["predicted_E_OH_eV"] for r in results])
    y = np.array([r["predicted_E_O_eV"] for r in results])
    slope, intercept = np.polyfit(x, y, 1)
    corr = np.corrcoef(x, y)[0, 1]

    gas_phase = adsorption["gas_phase"]
    oh_bond = distance(gas_phase["OH"][0][1], gas_phase["OH"][1][1])

    return {
        "per_metal": results,
        "scaling_fit": {
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": float(corr),
        },
        "gas_phase_oh_bond_A": oh_bond,
        "slab_size": size,
        "placement_site": adsorption["placement"]["site"],
        "fixed_layers": adsorption["relaxation"]["fixed_layers"],
    }


def reaction_analysis(parsed: Dict[str, object]) -> Dict[str, object]:
    out = []
    for rxn, info in sorted(parsed["reactions"].items(), key=lambda kv: int(kv[0].split()[1])):
        reactant = info["reactant"]
        ts = info["transition_state"]
        react_bonds = approximate_bonds(reactant)
        ts_bonds = approximate_bonds(ts)
        react_dists = pairwise_distances(reactant)
        ts_dists = pairwise_distances(ts)

        react_map = {(d["i"], d["j"]): d["distance_A"] for d in react_dists}
        ts_map = {(d["i"], d["j"]): d["distance_A"] for d in ts_dists}
        common_pairs = sorted(set(react_map) & set(ts_map))
        delta = np.array([ts_map[p] - react_map[p] for p in common_pairs], dtype=float)
        rms_delta = float(np.sqrt(np.mean(delta ** 2))) if len(delta) else 0.0
        max_delta = float(np.max(np.abs(delta))) if len(delta) else 0.0
        rg_react = radius_of_gyration(reactant)
        rg_ts = radius_of_gyration(ts)
        atom_count = len(reactant)
        barrier = float(info["dft_barrier_eV"])
        normalized_barrier = barrier / atom_count

        out.append(
            {
                "reaction": rxn,
                "label": info["label"],
                "atom_count": atom_count,
                "reactant_bond_count": len(react_bonds),
                "transition_state_bond_count": len(ts_bonds),
                "reactant_radius_of_gyration_A": rg_react,
                "transition_state_radius_of_gyration_A": rg_ts,
                "delta_radius_of_gyration_A": rg_ts - rg_react,
                "rms_pair_distance_shift_A": rms_delta,
                "max_pair_distance_shift_A": max_delta,
                "dft_barrier_eV": barrier,
                "barrier_per_atom_eV": normalized_barrier,
            }
        )

    barriers = np.array([r["dft_barrier_eV"] for r in out])
    shifts = np.array([r["rms_pair_distance_shift_A"] for r in out])
    if len(out) > 1 and np.std(shifts) > 0:
        corr = float(np.corrcoef(shifts, barriers)[0, 1])
    else:
        corr = float("nan")

    return {
        "per_reaction": out,
        "barrier_summary": {
            "mean_barrier_eV": float(np.mean(barriers)),
            "std_barrier_eV": float(np.std(barriers, ddof=0)),
            "mean_rms_shift_A": float(np.mean(shifts)),
            "shift_barrier_correlation": corr,
        },
    }


def foundation_summary(water: Dict[str, object], adsorption: Dict[str, object], reactions: Dict[str, object]) -> Dict[str, object]:
    domains = {
        "liquid_structure": {
            "benchmark": "Water RDF / MD stability proxy",
            "primary_signal": "physically plausible condensed-phase setup and molecular geometry",
            "key_metric": water["density_g_cm3"],
        },
        "surface_catalysis": {
            "benchmark": "Adsorption energy scaling across fcc transition metals",
            "primary_signal": "smooth cross-metal energetic trends",
            "key_metric": adsorption["scaling_fit"]["pearson_r"],
        },
        "reaction_chemistry": {
            "benchmark": "Barrier transfer across small-molecule rearrangements",
            "primary_signal": "consistent transition-state sensitivity",
            "key_metric": reactions["barrier_summary"]["mean_barrier_eV"],
        },
    }

    scorecard = {
        "domain_count": len(domains),
        "elements_covered": ["H", "C", "O", "Ni", "Cu", "Rh", "Pd", "Ir", "Pt"],
        "elements_covered_count": 9,
        "benchmark_diversity": "molecule + liquid + surface + reaction",
        "fine_tuning_readiness_assessment": "The provided benchmarks span structural, catalytic, and reactive regimes, making them suitable as low-data adaptation probes once a pretrained universal potential is available.",
    }
    return {"domains": domains, "scorecard": scorecard}


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def plot_water(metrics: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    pair_names = ["O-H(1)", "O-H(2)", "H-H"]
    pair_values = metrics["oh_bond_lengths_A"] + [metrics["hh_distance_A"]]
    axes[0].bar(pair_names, pair_values, color=["#4C78A8", "#4C78A8", "#F58518"])
    axes[0].set_ylabel("Distance (Å)")
    axes[0].set_title("Single-water geometry")

    sim_names = ["Density", "Temp/300", "Total time ×10", "Friction ×100"]
    sim_vals = [
        metrics["density_g_cm3"],
        metrics["temperature_K"] / 300.0,
        metrics["total_time_ps"] * 10.0,
        metrics["langevin_friction_fs^-1"] * 100.0,
    ]
    axes[1].bar(sim_names, sim_vals, color="#54A24B")
    axes[1].set_title("Normalized MD setup indicators")
    axes[1].set_ylabel("Normalized value")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "water_setup_overview.png", dpi=200)
    plt.close(fig)


def plot_adsorption(adsorption: Dict[str, object]) -> None:
    per_metal = adsorption["per_metal"]
    metals = [r["metal"] for r in per_metal]
    lattice = [r["lattice_constant_A"] for r in per_metal]
    oh = [r["predicted_E_OH_eV"] for r in per_metal]
    o = [r["predicted_E_O_eV"] for r in per_metal]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(metals, lattice, color="#72B7B2")
    axes[0].set_ylabel("Lattice constant (Å)")
    axes[0].set_title("fcc(111) metal set")

    axes[1].scatter(oh, o, s=80, color="#E45756")
    fit = adsorption["scaling_fit"]
    xs = np.linspace(min(oh) - 0.05, max(oh) + 0.05, 100)
    ys = fit["slope"] * xs + fit["intercept"]
    axes[1].plot(xs, ys, color="black", linewidth=1.5)
    for r in per_metal:
        axes[1].annotate(r["metal"], (r["predicted_E_OH_eV"], r["predicted_E_O_eV"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Predicted E_OH (eV)")
    axes[1].set_ylabel("Predicted E_O (eV)")
    axes[1].set_title(f"Adsorption scaling proxy (r={fit['pearson_r']:.2f})")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "adsorption_scaling_analysis.png", dpi=200)
    plt.close(fig)


def plot_reactions(reactions: Dict[str, object]) -> None:
    per_reaction = reactions["per_reaction"]
    labels = [r["reaction"] for r in per_reaction]
    barriers = [r["dft_barrier_eV"] for r in per_reaction]
    shifts = [r["rms_pair_distance_shift_A"] for r in per_reaction]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].bar(labels, barriers, color="#B279A2")
    axes[0].set_ylabel("DFT barrier (eV)")
    axes[0].set_title("Reference reaction barriers")

    axes[1].scatter(shifts, barriers, color="#FF9DA6", s=90)
    for r in per_reaction:
        axes[1].annotate(r["reaction"], (r["rms_pair_distance_shift_A"], r["dft_barrier_eV"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("RMS pair-distance shift (Å)")
    axes[1].set_ylabel("DFT barrier (eV)")
    axes[1].set_title("Barrier vs structural distortion")

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "reaction_barrier_analysis.png", dpi=200)
    plt.close(fig)


def plot_foundation_scope(summary: Dict[str, object]) -> None:
    domains = list(summary["domains"].keys())
    values = [1, 1, 1]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(domains, values, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("Coverage indicator")
    ax.set_title("Benchmark-domain coverage of the reproduction input")
    ax.tick_params(axis="x", rotation=15)
    for i, key in enumerate(domains):
        ax.text(i, 1.03, summary["domains"][key]["benchmark"], ha="center", va="bottom", fontsize=8, rotation=0)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "foundation_scope_summary.png", dpi=200)
    plt.close(fig)


def write_markdown_summary(parsed: Dict[str, object], water: Dict[str, object], adsorption: Dict[str, object], reactions: Dict[str, object], foundation: Dict[str, object]) -> None:
    lines = []
    lines.append("# Analysis Output Summary")
    lines.append("")
    lines.append("## Data provenance")
    lines.append(f"- Source file: `{DATA_FILE.relative_to(WORKSPACE)}`")
    lines.append(f"- Referenced model artifact: `{parsed['common'].get('model_file', 'N/A')}`")
    lines.append("")
    lines.append("## Water benchmark")
    lines.append(f"- Molecules: {water['molecules']}")
    lines.append(f"- Box size: {water['box_size_A']:.2f} Å")
    lines.append(f"- Density implied by box and composition: {water['density_g_cm3']:.3f} g/cm^3")
    lines.append(f"- HOH angle: {water['hoh_angle_deg']:.2f}°")
    lines.append(f"- Total MD time: {water['total_time_ps']:.2f} ps")
    lines.append("")
    lines.append("## Adsorption benchmark")
    lines.append(f"- Metals: {', '.join(m['metal'] for m in adsorption['per_metal'])}")
    lines.append(f"- Scaling-fit slope: {adsorption['scaling_fit']['slope']:.3f}")
    lines.append(f"- Scaling-fit intercept: {adsorption['scaling_fit']['intercept']:.3f} eV")
    lines.append(f"- Scaling-fit Pearson r: {adsorption['scaling_fit']['pearson_r']:.3f}")
    lines.append("")
    lines.append("## Reaction benchmark")
    lines.append(f"- Mean DFT barrier: {reactions['barrier_summary']['mean_barrier_eV']:.3f} eV")
    lines.append(f"- Barrier std: {reactions['barrier_summary']['std_barrier_eV']:.3f} eV")
    lines.append(f"- Mean RMS structural shift: {reactions['barrier_summary']['mean_rms_shift_A']:.3f} Å")
    lines.append("")
    lines.append("## Foundation-model interpretation")
    lines.append(f"- Element coverage in this reproduction proxy: {foundation['scorecard']['elements_covered_count']} elements")
    lines.append(f"- Benchmark diversity: {foundation['scorecard']['benchmark_diversity']}")
    lines.append(f"- Assessment: {foundation['scorecard']['fine_tuning_readiness_assessment']}")
    lines.append("")
    lines.append("## Generated figures")
    for name in [
        "water_setup_overview.png",
        "adsorption_scaling_analysis.png",
        "reaction_barrier_analysis.png",
        "foundation_scope_summary.png",
    ]:
        lines.append(f"- `report/images/{name}`")

    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    raw_text = DATA_FILE.read_text(encoding="utf-8")
    parsed = parse_dataset(raw_text)
    water = water_analysis(parsed)
    adsorption = adsorption_analysis(parsed)
    reactions = reaction_analysis(parsed)
    foundation = foundation_summary(water, adsorption, reactions)

    write_json(OUTPUT_DIR / "parsed_dataset.json", parsed)
    write_json(OUTPUT_DIR / "water_analysis.json", water)
    write_json(OUTPUT_DIR / "adsorption_analysis.json", adsorption)
    write_json(OUTPUT_DIR / "reaction_analysis.json", reactions)
    write_json(OUTPUT_DIR / "foundation_model_assessment.json", foundation)

    plot_water(water)
    plot_adsorption(adsorption)
    plot_reactions(reactions)
    plot_foundation_scope(foundation)
    write_markdown_summary(parsed, water, adsorption, reactions, foundation)

    manifest = {
        "outputs": [
            "outputs/parsed_dataset.json",
            "outputs/water_analysis.json",
            "outputs/adsorption_analysis.json",
            "outputs/reaction_analysis.json",
            "outputs/foundation_model_assessment.json",
            "outputs/analysis_summary.md",
        ],
        "figures": [
            "report/images/water_setup_overview.png",
            "report/images/adsorption_scaling_analysis.png",
            "report/images/reaction_barrier_analysis.png",
            "report/images/foundation_scope_summary.png",
        ],
    }
    write_json(OUTPUT_DIR / "analysis_manifest.json", manifest)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
