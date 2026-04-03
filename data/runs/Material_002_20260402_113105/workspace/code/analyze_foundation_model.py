#!/usr/bin/env python3
"""Analyze the provided MACE-MP-0 reproduction note and related papers.

This script does not fabricate a trained universal potential. Instead, it
produces a reproducible benchmark characterization and a literature-grounded
foundation-model blueprint using only the inputs available in this workspace.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "MACE-MP-0_Reproduction_Dataset.txt"
RELATED_WORK = ROOT / "related_work"
OUTPUTS = ROOT / "outputs"
PAPER_TEXT_DIR = OUTPUTS / "paper_text"
IMAGES = ROOT / "report" / "images"
MPLCONFIG = ROOT / "outputs" / "mplconfig"

AVOGADRO = 6.02214076e23
ATOMIC_MASS = {
    "H": 1.008,
    "C": 12.011,
    "O": 15.999,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ir": 192.217,
    "Pt": 195.084,
}


def ensure_directories() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    PAPER_TEXT_DIR.mkdir(exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    MPLCONFIG.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)


def run_pdftotext() -> Dict[str, str]:
    if shutil.which("pdftotext") is None:
        raise RuntimeError("pdftotext is required but not available in PATH.")

    texts: Dict[str, str] = {}
    for pdf_path in sorted(RELATED_WORK.glob("*.pdf")):
        txt_path = PAPER_TEXT_DIR / f"{pdf_path.stem}.txt"
        subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            check=True,
            cwd=ROOT,
        )
        texts[pdf_path.stem] = txt_path.read_text(encoding="utf-8", errors="ignore")
    return texts


def parse_float_triplet(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",")]


def parse_dataset() -> Dict[str, object]:
    text = DATASET_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    water = {
        "num_molecules": int(re.search(r"Number of water molecules: (\d+)", text).group(1)),
        "box_size_angstrom": float(re.search(r"Box size \(Å\): ([0-9.]+)", text).group(1)),
        "temperature_K": float(re.search(r"Temperature \(K\): ([0-9.]+)", text).group(1)),
        "timestep_fs": float(re.search(r"Time step \(fs\): ([0-9.]+)", text).group(1)),
        "steps": int(re.search(r"Total number of MD steps: (\d+)", text).group(1)),
        "friction_fs_inv": float(
            re.search(r"Friction coefficient for Langevin thermostat \(fs⁻¹\): ([0-9.]+)", text).group(1)
        ),
        "single_molecule_coords": {},
    }

    water_coord_pattern = re.compile(r"^\s+(O|H): \[([^\]]+)\]$")
    in_water_block = False
    water_h_index = 1
    adsorption_metals: Dict[str, float] = {}
    adsorption = {}
    reactions: Dict[str, dict] = {}
    reaction_barriers: Dict[str, float] = {}
    current_reaction = None
    current_state = None

    coord_pattern = re.compile(r"^\s+([A-Z][a-z]?): \[([^\]]+)\]$")

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- Coordinates of a single water molecule"):
            in_water_block = True
            continue
        if in_water_block and stripped.startswith("##"):
            in_water_block = False
        if in_water_block:
            match = water_coord_pattern.match(line)
            if match:
                element = match.group(1)
                label = element
                if element == "H":
                    label = f"H{water_h_index}"
                    water_h_index += 1
                water["single_molecule_coords"][label] = parse_float_triplet(match.group(2))
            continue

        metal_match = re.match(r"^\s+([A-Z][a-z]?): ([0-9.]+)$", line)
        if metal_match and current_reaction is None and stripped.endswith(tuple(str(v) for v in range(10))) is False:
            pass

        if stripped == "- Metals and their lattice constants (Å) for fcc(111) surface:":
            adsorption["metals"] = adsorption_metals
            continue
        if stripped.startswith(("Ni:", "Cu:", "Rh:", "Pd:", "Ir:", "Pt:")):
            metal, value = [part.strip() for part in stripped.split(":")]
            adsorption_metals[metal] = float(value)
            continue

        if stripped == "- Slab parameters:":
            adsorption["slab"] = {}
            continue
        if stripped.startswith("- Miller indices:"):
            adsorption["slab"]["miller_indices"] = re.search(r"\(([^)]+)\)", stripped).group(1)
            continue
        if stripped.startswith("- Size:"):
            adsorption["slab"]["size"] = [int(v.strip()) for v in re.search(r"\(([^)]+)\)", stripped).group(1).split(",")]
            continue
        if stripped.startswith("- Vacuum gap"):
            adsorption["slab"]["vacuum_gap_angstrom"] = float(re.search(r": ([0-9.]+)", stripped).group(1))
            continue
        if stripped == "- Adsorbate placement:":
            adsorption["placement"] = {}
            continue
        if stripped.startswith("- Site:"):
            adsorption["placement"]["site"] = stripped.split(": ", 1)[1]
            continue
        if stripped.startswith("- Height above surface"):
            adsorption["placement"]["height_angstrom"] = float(re.search(r": ([0-9.]+)", stripped).group(1))
            continue
        if stripped == "- Geometry relaxation:":
            adsorption["relaxation"] = {}
            continue
        if stripped.startswith("- Fixed layers:"):
            adsorption["relaxation"]["fixed_layers"] = stripped.split(": ", 1)[1]
            continue
        if stripped.startswith("- Force convergence tolerance"):
            adsorption["relaxation"]["force_tolerance_eV_per_A"] = float(re.search(r": ([0-9.]+)", stripped).group(1))
            continue
        if stripped == "- Gas phase molecules (isolated in a 10 Å box):":
            adsorption["gas_phase"] = {"box_angstrom": 10.0}
            continue
        if stripped.startswith("O atom coordinates:"):
            adsorption["gas_phase"]["O"] = {"O1": parse_float_triplet(re.search(r"\[([^\]]+)\]", stripped).group(1))}
            continue
        if stripped.startswith("OH molecule coordinates"):
            adsorption["gas_phase"]["OH"] = {}
            continue
        if stripped.startswith("### Reaction"):
            reaction_id = re.search(r"Reaction (\d+) \(Rxn (\d+)", stripped)
            current_reaction = {
                "reaction_number": int(reaction_id.group(1)),
                "crbh20_id": int(reaction_id.group(2)),
                "title": stripped.split("–", 1)[1].rstrip(")").strip(),
                "reactant": [],
                "transition_state": [],
            }
            reactions[f"rxn_{reaction_id.group(2)}"] = current_reaction
            current_state = None
            continue
        if stripped.startswith("- Reactant"):
            current_state = "reactant"
            continue
        if stripped.startswith("- Transition state"):
            current_state = "transition_state"
            continue
        if current_reaction is not None and current_state is not None:
            coord_match = coord_pattern.match(line)
            if coord_match:
                current_reaction[current_state].append(
                    {
                        "element": coord_match.group(1),
                        "coords": parse_float_triplet(coord_match.group(2)),
                    }
                )
                continue

        barrier_match = re.match(r"^\s*Rxn (\d+): ([0-9.]+)$", line)
        if barrier_match:
            reaction_barriers[f"rxn_{barrier_match.group(1)}"] = float(barrier_match.group(2))

    for key, value in reaction_barriers.items():
        reactions[key]["dft_barrier_eV"] = value

    return {"water": water, "adsorption": adsorption, "reactions": reactions}


def composition_from_atoms(atoms: List[dict]) -> Dict[str, int]:
    counter = Counter(atom["element"] for atom in atoms)
    return dict(sorted(counter.items()))


def formula_from_composition(composition: Dict[str, int]) -> str:
    return "".join(f"{el}{'' if count == 1 else count}" for el, count in composition.items())


def distance(a: List[float], b: List[float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def angle_degrees(a: List[float], b: List[float], c: List[float]) -> float:
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_theta = float(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def compute_benchmark_metrics(dataset: Dict[str, object]) -> Dict[str, object]:
    water = dataset["water"]
    coords = water["single_molecule_coords"]
    water_metrics = {
        "num_atoms": water["num_molecules"] * 3,
        "trajectory_time_ps": water["steps"] * water["timestep_fs"] / 1000.0,
        "thermostat_relaxation_time_fs": 1.0 / water["friction_fs_inv"],
        "single_molecule_oh_bonds_angstrom": [
            distance(coords["O"], coords["H1"]),
            distance(coords["O"], coords["H2"]),
        ],
        "single_molecule_hoh_angle_deg": angle_degrees(coords["H1"], coords["O"], coords["H2"]),
    }
    water_mass_g = water["num_molecules"] * (2 * ATOMIC_MASS["H"] + ATOMIC_MASS["O"]) / AVOGADRO
    water_volume_cm3 = (water["box_size_angstrom"] ** 3) * 1e-24
    water_metrics["initial_number_density_molecules_per_A3"] = water["num_molecules"] / (water["box_size_angstrom"] ** 3)
    water_metrics["initial_mass_density_g_cm3"] = water_mass_g / water_volume_cm3

    ads_rows = []
    size_x, size_y, layers = dataset["adsorption"]["slab"]["size"]
    for metal, a0 in dataset["adsorption"]["metals"].items():
        area = math.sqrt(3.0) * (a0 ** 2)
        d111 = a0 / math.sqrt(3.0)
        ads_rows.append(
            {
                "metal": metal,
                "lattice_constant_angstrom": a0,
                "nearest_neighbor_distance_angstrom": a0 / math.sqrt(2.0),
                "surface_area_angstrom2": area,
                "interlayer_spacing_angstrom": d111,
                "approx_slab_thickness_angstrom": (layers - 1) * d111,
                "surface_atoms": size_x * size_y * layers,
                "adsorbate_site_density_per_A2": 1.0 / area,
                "coverage_monolayer_equivalent": 1.0 / (size_x * size_y),
                "O_ads_system_atoms": size_x * size_y * layers + 1,
                "OH_ads_system_atoms": size_x * size_y * layers + 2,
            }
        )
    adsorption_df = pd.DataFrame(ads_rows)

    reaction_rows = []
    reaction_instances = []
    for rxn_key, reaction in dataset["reactions"].items():
        reactant = reaction["reactant"]
        ts = reaction["transition_state"]
        max_delta = None
        max_pair = None
        for i in range(len(reactant)):
            for j in range(i + 1, len(reactant)):
                dr = distance(reactant[i]["coords"], reactant[j]["coords"])
                dt = distance(ts[i]["coords"], ts[j]["coords"])
                delta = dt - dr
                if max_delta is None or abs(delta) > abs(max_delta):
                    max_delta = delta
                    max_pair = (
                        f"{reactant[i]['element']}{i+1}",
                        f"{reactant[j]['element']}{j+1}",
                    )
        displacements = [
            distance(reactant[i]["coords"], ts[i]["coords"]) for i in range(len(reactant))
        ]
        composition = composition_from_atoms(reactant)
        reaction_rows.append(
            {
                "reaction_key": rxn_key,
                "crbh20_id": reaction["crbh20_id"],
                "title": reaction["title"],
                "formula": formula_from_composition(composition),
                "num_atoms": len(reactant),
                "elements": ";".join(composition.keys()),
                "max_atom_displacement_angstrom": max(displacements),
                "mean_atom_displacement_angstrom": float(np.mean(displacements)),
                "max_pair_distance_change_angstrom": max_delta,
                "most_changed_pair": f"{max_pair[0]}-{max_pair[1]}",
                "dft_barrier_eV": reaction["dft_barrier_eV"],
            }
        )
        reaction_instances.append(
            {
                "benchmark_instance": f"CRBH20 {reaction['crbh20_id']}",
                "domain": "reaction",
                "num_atoms": len(reactant),
                "elements": list(composition.keys()),
            }
        )
    reaction_df = pd.DataFrame(reaction_rows)

    benchmark_instances = [
        {
            "benchmark_instance": "Water-32",
            "domain": "liquid",
            "num_atoms": water_metrics["num_atoms"],
            "elements": ["H", "O"],
        }
    ]
    for _, row in adsorption_df.iterrows():
        benchmark_instances.append(
            {
                "benchmark_instance": f"{row['metal']}-O*",
                "domain": "surface",
                "num_atoms": int(row["O_ads_system_atoms"]),
                "elements": [row["metal"], "O"],
            }
        )
        benchmark_instances.append(
            {
                "benchmark_instance": f"{row['metal']}-OH*",
                "domain": "surface",
                "num_atoms": int(row["OH_ads_system_atoms"]),
                "elements": [row["metal"], "O", "H"],
            }
        )
    benchmark_instances.extend(reaction_instances)

    return {
        "water_metrics": water_metrics,
        "adsorption_metrics": adsorption_df,
        "reaction_metrics": reaction_df,
        "benchmark_instances": pd.DataFrame(benchmark_instances),
    }


def parse_literature_metrics(paper_texts: Dict[str, str]) -> Dict[str, object]:
    mace_text = paper_texts["paper_000"]
    chgnet_text = paper_texts["paper_001"]
    transfer_text = paper_texts["paper_003"]

    def grab(pattern: str, text: str, cast=float):
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
        if match is None:
            raise ValueError(f"Pattern not found: {pattern}")
        return cast(match.group(1).replace(",", ""))

    def contains(fragment: str, text: str) -> bool:
        return fragment.lower() in text.lower()

    literature = {
        "mace_architecture": {
            "four_body_messages": contains("using four-body messages", mace_text),
            "message_passing_iterations": 2 if contains("just two", mace_text) else None,
            "improved_learning_curve_steepness": contains("improved steepness of the learning curves", mace_text),
        },
        "mptrj_dataset": {
            "structures": grab(r"([0-9,]+) atom configurations", chgnet_text, int),
            "materials": grab(r"based on the ([0-9,]+) compounds", chgnet_text, int),
            "elements_in_mp_database": grab(r"~146,000 inorganic materials composed of (\d+) elements", chgnet_text, int),
            "force_labels": grab(r"magmoms,\s*([0-9,]+)\s*forces", chgnet_text, int),
            "stress_labels": grab(r"forces and\s*([0-9,]+)\s*stresses", chgnet_text, int),
            "coverage_summary": "all chemistries except noble gases and actinoids",
            "elements_over_100k_occurrences": grab(r"over 100,000 occurrences for (\d+) different elements", chgnet_text, int),
            "elements_over_10k_magmom_labels": grab(r"10,000 instances with magnetic information for (\d+) different", chgnet_text, int),
            "chgnet_parameters": grab(r"CHGNet with ([0-9,]+) trainable parameters", chgnet_text, int),
        },
        "fine_tuning_examples": {
            "li_fe_p_o_pre_finetune_meV_per_atom": grab(
                r"test error from (\d+)\s+meV per atom to 15\s+meV\s+per atom",
                chgnet_text,
                int,
            ),
            "li_fe_p_o_post_finetune_meV_per_atom": 15,
        },
        "cross_functional_transfer": {
            "mp_r2scan_structures": grab(r"material IDs with ([0-9,]+)\s+structures", transfer_text, int),
            "mp_r2scan_materials": grab(r"obtain ([0-9,]+)\s+material IDs with [0-9,]+\s+structures", transfer_text, int),
            "energy_mae_meV_per_atom": grab(r"energy MAE of (\d+) meV/atom", transfer_text, int),
            "force_mae_meV_per_A": grab(r"force\s+MAE of (\d+) meV/[ÅÅ]", transfer_text, int),
            "scratch_energy_slope": grab(r"Scratch curve exhibits a log-log slope of (-?[0-9.]+)", transfer_text, float),
            "transfer_energy_slope": grab(r"Transfer curve has a log-log slope of (-?[0-9.]+) with an R2 of 0.964", transfer_text, float),
            "scratch_force_slope": grab(r"For force\s+MAE.*?Scratch curve shows a log-log slope of (-?[0-9.]+)", transfer_text, float),
            "transfer_force_slope": grab(r"For force\s+MAE.*?Transfer curve has a log-log slope of (-?[0-9.]+)", transfer_text, float),
            "energy_saturation_training_points": grab(r"after ([0-9,]+) training points for energy", transfer_text, int),
            "force_saturation_training_points": grab(r"and ([0-9,]+) training points for\s+force", transfer_text, int),
            "correlation_before_atomref": grab(r"improves from ([0-9.]+)\s+between the unmodiﬁed", transfer_text, float),
            "correlation_after_atomref": grab(r"to ([0-9.]+)\s+between the r2SCAN energies", transfer_text, float),
            "ten_x_efficiency_claim": contains("more than 10-fold data efﬁciency", transfer_text),
            "one_k_beats_ten_k_claim": contains("1K high-ﬁdelity data points can outperform training from scratch on a high-ﬁdelity dataset with more than 10K data points", transfer_text),
        },
    }
    return literature


def save_machine_readable_outputs(dataset: Dict[str, object], metrics: Dict[str, object], literature: Dict[str, object]) -> None:
    benchmark_summary = {
        "dataset": dataset,
        "water_metrics": metrics["water_metrics"],
        "literature_summary": literature,
    }
    (OUTPUTS / "benchmark_summary.json").write_text(
        json.dumps(benchmark_summary, indent=2),
        encoding="utf-8",
    )
    metrics["adsorption_metrics"].to_csv(OUTPUTS / "adsorption_metrics.csv", index=False)
    metrics["reaction_metrics"].to_csv(OUTPUTS / "reaction_metrics.csv", index=False)
    (OUTPUTS / "literature_summary.json").write_text(
        json.dumps(literature, indent=2),
        encoding="utf-8",
    )

    proposed_model = {
        "name": "MACE-MP-X Blueprint",
        "objective": "Universal atomistic potential pretraining on MPtrj followed by low-data chemistry adaptation.",
        "architecture": {
            "base_encoder": "MACE",
            "many_body_messages": "4-body",
            "message_passing_iterations": 2,
            "recommended_extensions": [
                "energy/force/stress multitask heads",
                "elemental reference energy refit during cross-functional fine-tuning",
                "optional charge-aware auxiliary targets when labels exist",
                "curriculum spanning solids, surfaces, liquids, and reactive structures",
            ],
        },
        "training_hypothesis": {
            "pretraining_dataset": "MPtrj",
            "high_fidelity_adaptation": "r2SCAN or task-specific DFT",
            "fine_tuning_rule": "re-fit atomic reference energies before updating GNN weights",
            "expected_benefit": "greater than 10x data efficiency versus scratch training, based on related-work evidence",
        },
        "benchmark_targets": [
            "liquid water structure",
            "transition-metal adsorption scaling relations",
            "CRBH20 reaction barriers",
        ],
    }
    (OUTPUTS / "proposed_model_spec.json").write_text(
        json.dumps(proposed_model, indent=2),
        encoding="utf-8",
    )


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "font.family": "DejaVu Serif",
        }
    )


def plot_benchmark_instances(metrics: Dict[str, object]) -> None:
    df = metrics["benchmark_instances"].copy()
    df = df.sort_values(["domain", "num_atoms", "benchmark_instance"])
    palette = {"liquid": "#2a9d8f", "surface": "#e76f51", "reaction": "#264653"}

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=df, x="benchmark_instance", y="num_atoms", hue="domain", dodge=False, palette=palette, ax=ax)
    ax.set_title("Benchmark Instances Span Liquid, Surface, and Reactive Regimes")
    ax.set_xlabel("")
    ax.set_ylabel("Atoms in benchmark structure")
    ax.tick_params(axis="x", rotation=65)
    ax.legend(title="Domain", frameon=True)
    fig.tight_layout()
    fig.savefig(IMAGES / "benchmark_instances.png", bbox_inches="tight")
    plt.close(fig)


def plot_chemistry_matrix(metrics: Dict[str, object]) -> None:
    df = metrics["benchmark_instances"].copy()
    elements = sorted({el for entry in df["elements"] for el in entry})
    matrix = pd.DataFrame(0, index=df["benchmark_instance"], columns=elements)
    for _, row in df.iterrows():
        for el in row["elements"]:
            matrix.loc[row["benchmark_instance"], el] = 1

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, cmap=sns.color_palette(["#f1faee", "#1d3557"], as_cmap=True), cbar=False, linewidths=0.5, ax=ax)
    ax.set_title("Elemental Coverage of the Provided Reproduction Benchmarks")
    ax.set_xlabel("Element")
    ax.set_ylabel("Benchmark instance")
    fig.tight_layout()
    fig.savefig(IMAGES / "chemistry_matrix.png", bbox_inches="tight")
    plt.close(fig)


def plot_geometric_scales(metrics: Dict[str, object]) -> None:
    adsorption_df = metrics["adsorption_metrics"]
    reaction_df = metrics["reaction_metrics"]
    water_density = metrics["water_metrics"]["initial_mass_density_g_cm3"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(["Initial box"], [water_density], color="#457b9d")
    axes[0].axhline(1.0, color="#e63946", linestyle="--", linewidth=2)
    axes[0].set_title("Water Initialization Density")
    axes[0].set_ylabel("g cm$^{-3}$")
    axes[0].text(0, water_density + 0.03, f"{water_density:.2f}", ha="center", fontsize=11)
    axes[0].text(0.02, 1.02, "Ambient water", color="#e63946", transform=axes[0].get_yaxis_transform())

    sns.barplot(
        data=adsorption_df,
        x="metal",
        y="surface_area_angstrom2",
        color="#f4a261",
        ax=axes[1],
    )
    axes[1].set_title("fcc(111) 2x2 Surface Area by Metal")
    axes[1].set_xlabel("Metal")
    axes[1].set_ylabel(r"Area ($\AA^2$)")

    sns.scatterplot(
        data=reaction_df,
        x="max_pair_distance_change_angstrom",
        y="dft_barrier_eV",
        s=120,
        color="#2a9d8f",
        ax=axes[2],
    )
    for _, row in reaction_df.iterrows():
        axes[2].text(
            row["max_pair_distance_change_angstrom"] + 0.01,
            row["dft_barrier_eV"] + 0.005,
            f"Rxn {int(row['crbh20_id'])}",
            fontsize=10,
        )
    axes[2].set_title("Reactive Distortion vs. Reference Barrier")
    axes[2].set_xlabel(r"Max pair-distance change ($\AA$)")
    axes[2].set_ylabel("DFT barrier (eV)")

    fig.tight_layout()
    fig.savefig(IMAGES / "geometric_scales.png", bbox_inches="tight")
    plt.close(fig)


def plot_transferability(literature: Dict[str, object]) -> None:
    transfer = literature["cross_functional_transfer"]
    mptrj = literature["mptrj_dataset"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dataset_sizes = pd.DataFrame(
        {
            "dataset": ["MPtrj", "MP-r2SCAN"],
            "structures": [mptrj["structures"], transfer["mp_r2scan_structures"]],
        }
    )
    sns.barplot(
        data=dataset_sizes,
        x="dataset",
        y="structures",
        hue="dataset",
        palette=["#1d3557", "#a8dadc"],
        dodge=False,
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Pretraining vs. High-Fidelity Adaptation Scale")
    axes[0].set_ylabel("Structures")
    axes[0].set_xlabel("")
    for _, row in dataset_sizes.iterrows():
        axes[0].text(row.name, row["structures"] * 1.02, f"{row['structures']:,}", ha="center", fontsize=11)

    efficiency_df = pd.DataFrame(
        {
            "metric": ["Energy MAE", "Force MAE", "Correlation before", "Correlation after"],
            "value": [
                transfer["energy_mae_meV_per_atom"],
                transfer["force_mae_meV_per_A"],
                transfer["correlation_before_atomref"],
                transfer["correlation_after_atomref"],
            ],
        }
    )
    sns.barplot(
        data=efficiency_df,
        x="metric",
        y="value",
        hue="metric",
        palette=["#e76f51", "#f4a261", "#6d597a", "#2a9d8f"],
        dodge=False,
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Reported Transfer-Learning Anchors")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Reported value")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].text(
        0.02,
        0.98,
        "1K transferred points > 10K scratch points\nBest reported: 15 meV/atom, 36 meV/A",
        transform=axes[1].transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
    )

    fig.tight_layout()
    fig.savefig(IMAGES / "transferability_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_blueprint(literature: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    boxes = [
        (0.5, 3.4, 3.0, 1.6, "#d8f3dc", "1. Pretraining Corpus\nMPtrj\n1.58M structures\n145,923 materials"),
        (4.0, 3.4, 3.0, 1.6, "#bee1e6", "2. Base Model\nMACE\n4-body messages\n2 interaction steps"),
        (7.5, 3.4, 3.0, 1.6, "#ffd6a5", "3. Adaptation Rule\nRe-fit AtomRef\nthen fine-tune on\nhigh-fidelity/task data"),
        (11.0, 3.4, 2.5, 1.6, "#ffcad4", "4. Downstream Use\nLiquids\nSurfaces\nReactions"),
        (4.0, 0.8, 6.5, 1.4, "#f1faee", "Validation principle: benchmark transferability, stability, and low-data efficiency.\nRelated work reports >10x data efficiency and 15 meV/atom, 36 meV/A after transfer."),
    ]

    for x, y, w, h, color, label in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.08",
            linewidth=1.5,
            edgecolor="#3a3a3a",
            facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=13)

    arrows = [
        ((3.5, 4.2), (4.0, 4.2)),
        ((7.0, 4.2), (7.5, 4.2)),
        ((10.5, 4.2), (11.0, 4.2)),
        ((7.2, 3.4), (7.2, 2.2)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color="#3a3a3a"))

    ax.set_title("Proposed MACE-MP-X Foundation-Model Blueprint", fontsize=18, pad=12)
    fig.tight_layout()
    fig.savefig(IMAGES / "foundation_model_blueprint.png", bbox_inches="tight")
    plt.close(fig)


def write_report(dataset: Dict[str, object], metrics: Dict[str, object], literature: Dict[str, object]) -> None:
    water = metrics["water_metrics"]
    adsorption_df = metrics["adsorption_metrics"]
    reaction_df = metrics["reaction_metrics"]
    transfer = literature["cross_functional_transfer"]
    mptrj = literature["mptrj_dataset"]

    surface_area_span = (adsorption_df["surface_area_angstrom2"].min(), adsorption_df["surface_area_angstrom2"].max())
    barrier_span = (reaction_df["dft_barrier_eV"].min(), reaction_df["dft_barrier_eV"].max())

    report = f"""# Benchmark-Driven Blueprint for a Universal MACE Foundation Potential

## Abstract
This study analyzes the provided MACE-MP-0 reproduction note and related literature to build a reproducible, benchmark-driven research artifact for universal atomistic potentials. The workspace does **not** contain the raw MPtrj corpus, force labels, or a pretrained MACE-MP checkpoint, so direct training of a new foundation model is not possible from the available inputs alone. Instead, I constructed a transparent analysis pipeline that (i) parses the reproduction benchmarks, (ii) extracts quantitative constraints from the related papers, (iii) measures the geometric and chemical scope of the supplied benchmark structures, and (iv) synthesizes a concrete foundation-model blueprint for MACE-based pretraining and low-data fine-tuning. The resulting analysis supports three conclusions: MACE remains a strong architectural backbone because its four-body message construction reduces the required interaction depth to two passes; the benchmark suite probes substantially different regimes spanning condensed liquids, metal surfaces, and reactive molecular transition states; and cross-functional transfer evidence from related work strongly supports fine-tuning via re-fitted elemental reference energies before downstream adaptation.

## 1. Problem Setting and Available Evidence
The stated scientific goal is a universal atomistic foundation model that generalizes across liquids, solids, catalysis, and reactions while reaching ab initio accuracy after minimal task-specific fine-tuning. However, the only directly available benchmark input is a compact reproduction note for three MACE-MP-0 tests: liquid water structure, adsorption energy scaling on fcc(111) transition-metal surfaces, and CRBH20 reaction barriers. The related-work folder contributes the methodological context required to interpret that note: the original MACE architecture paper, the CHGNet MPtrj paper, a tensor-network extension for O(3)-equivariant models, and a 2025 study on cross-functional transfer in foundation potentials.

The quantitative constraints extracted from the literature are central. The CHGNet paper reports that MPtrj contains 1,580,395 structures across 145,923 materials, with broad chemistry coverage and tens of millions of force/stress labels. The MACE paper shows that four-body messages permit accurate modeling with only two message-passing iterations. The transfer-learning study reports that moving from GGA/GGA+U pretraining to r2SCAN becomes much more stable when atomic reference energies are re-fitted, improving the reported energy correlation from {transfer["correlation_before_atomref"]:.4f} to {transfer["correlation_after_atomref"]:.4f}, and that transfer with 1k high-fidelity structures can outperform scratch training on more than 10k structures.

## 2. Methods
### 2.1 Reproducible analysis pipeline
I implemented a single script, [`code/analyze_foundation_model.py`]({ROOT / "code" / "analyze_foundation_model.py"}), that converts the PDF references to text, parses the benchmark note, computes simple but informative geometry-based benchmark metrics, extracts transferable quantitative findings from the papers, and generates machine-readable outputs and figures.

### 2.2 Direct benchmark characterization
The analysis computes three categories of metrics directly from the supplied benchmark specification.

For water, the script derives atom count, box density, simulation length, thermostat timescale, and the internal geometry of the supplied H2O monomer. For adsorption, it computes fcc(111) nearest-neighbor distances, 2x2 cell areas, approximate interlayer spacings, and atom counts for O* and OH* slabs across Ni, Cu, Rh, Pd, Ir, and Pt. For CRBH20, it aligns the reactant and transition-state coordinate lists by atom index and measures atomic displacements plus the largest pair-distance change associated with each barrier reference.

### 2.3 Literature-grounded model synthesis
The report does not claim to have trained a new foundation model. Instead, it proposes a training blueprint constrained by the extracted evidence: MACE as the equivariant backbone, MPtrj as the broad-coverage pretraining source, and AtomRef re-fitting as the first step during high-fidelity or task-specific fine-tuning.

## 3. Results
### 3.1 The benchmark suite is genuinely multi-regime
![Benchmark instances](images/benchmark_instances.png)

The provided reproduction suite is small but diverse. It spans a 96-atom liquid-water cell, twelve adsorption structures over six late transition metals, and three molecular reaction benchmarks with 5 to 9 atoms. This matters because a convincing universal potential must bridge differences in bonding, periodicity, and target observables rather than simply average over one domain.

![Elemental coverage](images/chemistry_matrix.png)

The explicit element coverage of the reproduction benchmark is narrow compared with MPtrj, but it is strategically chosen. H, C, and O cover molecular and liquid chemistry, while Ni, Cu, Rh, Pd, Ir, and Pt probe metallic and catalytic surface environments. The benchmark therefore tests cross-domain transfer rather than broad compositional coverage by itself.

### 3.2 Physical scales differ substantially across the three tests
![Geometric scales](images/geometric_scales.png)

The water test uses 32 molecules in a 12 Å cubic box, corresponding to an initial mass density of {water["initial_mass_density_g_cm3"]:.3f} g cm$^{{-3}}$, below ambient liquid water. This is a useful reminder that the supplied benchmark note specifies an MD setup, not an equilibrated reference trajectory; a realistic potential must therefore relax toward the correct structure rather than merely score a fixed geometry.

For adsorption, the fcc(111) 2x2 surface area varies from {surface_area_span[0]:.2f} to {surface_area_span[1]:.2f} Å$^2$ across the six metals, while the monolayer-equivalent coverage remains fixed at 0.25 ML because one adsorbate occupies a 2x2 cell. This creates a clean scaling-relations setting where chemistry changes while the protocol remains constant.

For CRBH20, the three supplied reference barriers occupy a narrow range of {barrier_span[0]:.2f}-{barrier_span[1]:.2f} eV, but the associated geometric distortions differ. That combination is important: a transferable model should not collapse barrier prediction onto one simple structural heuristic.

### 3.3 Related work strongly favors transfer-aware fine-tuning
![Transferability summary](images/transferability_summary.png)

The literature evidence is more decisive than the benchmark note itself. MPtrj-scale pretraining provides coverage, but the 2025 transfer study shows that high-fidelity adaptation is not a trivial continuation of low-fidelity pretraining. Re-fitting atomic reference energies is the key stabilizing step: the reported correlation between source and target energy labels rises from {transfer["correlation_before_atomref"]:.4f} to {transfer["correlation_after_atomref"]:.4f}, and the best transfer result reaches {transfer["energy_mae_meV_per_atom"]} meV/atom and {transfer["force_mae_meV_per_A"]} meV/Å on MP-r2SCAN.

The practical implication is direct. A universal MACE foundation model should be viewed as a **pretraining prior**, not a final production potential. Downstream adaptation should preserve equivariant geometric features while rapidly correcting energy referencing and only then refining interaction weights on small task-specific datasets.

### 3.4 Proposed foundation-model blueprint
![Foundation-model blueprint](images/foundation_model_blueprint.png)

The proposed blueprint, saved as [`outputs/proposed_model_spec.json`]({ROOT / "outputs" / "proposed_model_spec.json"}), is:

1. Pretrain a MACE backbone on MPtrj-scale data for energy, forces, and stresses.
2. Preserve the many-body MACE design with four-body messages and two interaction steps for efficiency and expressivity.
3. During transfer to higher-fidelity or task-specific data, re-fit elemental reference energies before updating the GNN weights.
4. Validate on domain-shift benchmarks that separately probe liquids, surfaces, and reactions.

This blueprint is scientifically plausible because it matches the strongest available evidence in the provided literature rather than assuming that larger scale alone guarantees universal transferability.

## 4. Discussion
The direct analysis supports the benchmark logic behind MACE-MP-0: one compact suite can probe whether a single pretrained potential remains stable in liquid MD, preserves adsorption trends across metal surfaces, and captures reaction barriers outside crystalline training distributions. The related literature adds the missing systems-level interpretation. MPtrj offers enough structural variety to support broad pretraining, and MACE offers a computationally attractive many-body equivariant backbone. But cross-functional transfer remains fragile unless the energy reference problem is handled explicitly.

This leads to the central research hypothesis of the report: **the most reliable path to a universal atomistic foundation model is not a single monolithic fit, but a two-stage procedure of broad MACE pretraining plus reference-aware low-data fine-tuning**. That hypothesis is consistent with the reported >10x data-efficiency gain in the transfer study and with the chemistry-specific fine-tuning improvement from 23 to 15 meV/atom cited in the CHGNet paper.

## 5. Limitations
This workspace does not include the raw MPtrj trajectories, labels, or a MACE-MP checkpoint, and network access is disallowed. Consequently, I could not train, fine-tune, or directly benchmark a new atomistic model. The adsorption and water benchmarks also lack direct target energies or trajectories in the provided text file, so quantitative reproduction of RDFs, adsorption energies, or MD stability is outside what can be demonstrated here without fabricating data.

Accordingly, the artifact delivered here should be interpreted as a **reproducible benchmark analysis and model-design study**, not as a claim of newly achieved ab initio benchmark accuracy.

## 6. Deliverables
The workspace now contains:

1. Analysis code: [`code/analyze_foundation_model.py`]({ROOT / "code" / "analyze_foundation_model.py"})
2. Structured outputs: [`outputs/benchmark_summary.json`]({ROOT / "outputs" / "benchmark_summary.json"}), [`outputs/adsorption_metrics.csv`]({ROOT / "outputs" / "adsorption_metrics.csv"}), [`outputs/reaction_metrics.csv`]({ROOT / "outputs" / "reaction_metrics.csv"}), and [`outputs/literature_summary.json`]({ROOT / "outputs" / "literature_summary.json"})
3. Figures in `report/images/`
4. Proposed model specification: [`outputs/proposed_model_spec.json`]({ROOT / "outputs" / "proposed_model_spec.json"})

## References
[1] Kovács, D. P., Batatia, I., Simm, G. N. C., Ortner, C., and Csányi, G. *MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields*. NeurIPS 2022.

[2] Deng, B. et al. *CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling*. Nature Machine Intelligence 5, 1031-1041 (2023).

[3] Li, Z. et al. *Unifying O(3) equivariant neural networks design with tensor-network formalism*. Machine Learning: Science and Technology 5, 025044 (2024).

[4] Huang, X. et al. *Cross-functional transferability in foundation machine learning interatomic potentials*. npj Computational Materials 11, 313 (2025).
"""

    (ROOT / "report" / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_directories()
    setup_style()
    paper_texts = run_pdftotext()
    dataset = parse_dataset()
    metrics = compute_benchmark_metrics(dataset)
    literature = parse_literature_metrics(paper_texts)
    save_machine_readable_outputs(dataset, metrics, literature)
    plot_benchmark_instances(metrics)
    plot_chemistry_matrix(metrics)
    plot_geometric_scales(metrics)
    plot_transferability(literature)
    plot_blueprint(literature)
    write_report(dataset, metrics, literature)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
