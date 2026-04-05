#!/usr/bin/env python3
"""Reproducible analysis of the 1BRS barnase-barstar HADDOCK3 input structure
and validation against SKEMPI 2.0 mutational affinity data.

Outputs:
- outputs/interface_residues.csv
- outputs/interface_atom_contacts.csv
- outputs/chain_summary.csv
- outputs/skempi_1brs_single_mutants.csv
- outputs/skempi_global_summary.csv
- outputs/analysis_summary.json
- report/images/*.png
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser, ShrakeRupley
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
IMG = ROOT / "report" / "images"
PDB_PATH = DATA / "1brs_AD.pdb"
SKEMPI_PATH = DATA / "skempi_v2.csv"

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 160
R_GAS_KCAL = 1.98720425864083e-3  # kcal mol^-1 K^-1
CONTACT_CUTOFF = 5.0
INTERFACE_CUTOFF = 8.0

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}
CHARGE = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIS": 0.1,
    "ALA": 0, "ASN": 0, "CYS": 0, "GLN": 0, "GLY": 0, "ILE": 0, "LEU": 0,
    "MET": 0, "PHE": 0, "PRO": 0, "SER": 0, "THR": 0, "TRP": 0, "TYR": 0, "VAL": 0,
}
HYDROPHOBIC = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "PRO", "TYR"}


def ensure_dirs() -> None:
    OUT.mkdir(exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)


def parse_structure():
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("1brs_AD", str(PDB_PATH))
    return structure[0]


def is_standard_residue(residue) -> bool:
    return residue.id[0] == " " and residue.resname in AA3_TO_1


def residue_key(chain_id: str, residue) -> str:
    return f"{chain_id}:{AA3_TO_1.get(residue.resname, 'X')}{residue.id[1]}"


def get_residue_atoms(residue):
    return [a for a in residue.get_atoms() if a.element != "H"]


def calc_min_distance(res1, res2) -> float:
    atoms1 = get_residue_atoms(res1)
    atoms2 = get_residue_atoms(res2)
    mind = float("inf")
    for a in atoms1:
        c1 = a.coord
        for b in atoms2:
            d = np.linalg.norm(c1 - b.coord)
            if d < mind:
                mind = d
    return mind


def center_distance(res1, res2) -> float:
    ca1 = res1["CA"].coord if "CA" in res1 else np.mean([a.coord for a in get_residue_atoms(res1)], axis=0)
    ca2 = res2["CA"].coord if "CA" in res2 else np.mean([a.coord for a in get_residue_atoms(res2)], axis=0)
    return float(np.linalg.norm(ca1 - ca2))


def compute_sasa(model):
    sr = ShrakeRupley()
    sr.compute(model, level="R")


def classify_contact(res1, res2) -> str:
    c1 = CHARGE.get(res1.resname, 0)
    c2 = CHARGE.get(res2.resname, 0)
    if c1 * c2 < 0:
        return "electrostatic"
    if res1.resname in HYDROPHOBIC and res2.resname in HYDROPHOBIC:
        return "hydrophobic"
    return "mixed"


def extract_chain_data(model):
    chains = {}
    for chain in model:
        residues = [r for r in chain if is_standard_residue(r)]
        chains[chain.id] = residues
    return chains


def analyze_interface(model):
    compute_sasa(model)
    chains = extract_chain_data(model)
    chain_a = chains["A"]
    chain_d = chains["D"]

    contacts = []
    residue_stats = defaultdict(lambda: {
        "n_contacts": 0,
        "closest_partner": None,
        "min_atom_distance": float("inf"),
        "min_ca_distance": float("inf"),
        "contact_types": [],
    })

    for res_a in chain_a:
        for res_d in chain_d:
            min_dist = calc_min_distance(res_a, res_d)
            ca_dist = center_distance(res_a, res_d)
            if min_dist <= INTERFACE_CUTOFF:
                ctype = classify_contact(res_a, res_d)
                contacts.append({
                    "chain_1": "A",
                    "residue_1": residue_key("A", res_a),
                    "resname_1": res_a.resname,
                    "resseq_1": res_a.id[1],
                    "chain_2": "D",
                    "residue_2": residue_key("D", res_d),
                    "resname_2": res_d.resname,
                    "resseq_2": res_d.id[1],
                    "min_atom_distance": min_dist,
                    "ca_distance": ca_dist,
                    "contact_class": ctype,
                    "is_direct_contact": min_dist <= CONTACT_CUTOFF,
                })
                for chain_id, res, partner_key, partner_dist in [
                    ("A", res_a, residue_key("D", res_d), min_dist),
                    ("D", res_d, residue_key("A", res_a), min_dist),
                ]:
                    key = residue_key(chain_id, res)
                    st = residue_stats[key]
                    if min_dist <= CONTACT_CUTOFF:
                        st["n_contacts"] += 1
                    if partner_dist < st["min_atom_distance"]:
                        st["min_atom_distance"] = partner_dist
                        st["closest_partner"] = partner_key
                    st["min_ca_distance"] = min(st["min_ca_distance"], ca_dist)
                    st["contact_types"].append(ctype)

    interface_rows = []
    for chain_id, residues in chains.items():
        for res in residues:
            key = residue_key(chain_id, res)
            st = residue_stats[key]
            sasa_complex = getattr(res, "sasa", np.nan)
            # approximate buriedness via interface-contact membership due to absence of unbound states
            interface_rows.append({
                "chain": chain_id,
                "residue": key,
                "resname": res.resname,
                "aa": AA3_TO_1[res.resname],
                "resseq": res.id[1],
                "complex_sasa": float(sasa_complex),
                "n_direct_contacts": st["n_contacts"],
                "min_atom_distance_to_partner": st["min_atom_distance"] if st["min_atom_distance"] < float("inf") else np.nan,
                "min_ca_distance_to_partner": st["min_ca_distance"] if st["min_ca_distance"] < float("inf") else np.nan,
                "closest_partner": st["closest_partner"],
                "dominant_contact_type": Counter(st["contact_types"]).most_common(1)[0][0] if st["contact_types"] else "none",
                "is_interface_8A": key in residue_stats,
                "is_direct_contact_5A": st["n_contacts"] > 0,
            })
    contacts_df = pd.DataFrame(contacts).sort_values(["min_atom_distance", "resseq_1", "resseq_2"])
    interface_df = pd.DataFrame(interface_rows).sort_values(["chain", "resseq"])
    return interface_df, contacts_df


def parse_mutation_token(token: str):
    m = re.match(r"([A-Z])([A-Z])(\d+)([A-Z])", token.strip())
    if not m:
        return None
    wt, chain, pos, mut = m.groups()
    return {"wt": wt, "chain": chain, "position_raw": int(pos), "mut": mut}


def load_skempi(interface_df: pd.DataFrame):
    df = pd.read_csv(SKEMPI_PATH, sep=";")
    df = df.rename(columns={"#Pdb": "pdb_id"})
    df["Affinity_mut_parsed"] = pd.to_numeric(df["Affinity_mut_parsed"], errors="coerce")
    df["Affinity_wt_parsed"] = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    df["Temperature_num"] = pd.to_numeric(df["Temperature"].astype(str).str.extract(r"([0-9]+\.?[0-9]*)")[0], errors="coerce")
    df["Temperature_num"] = df["Temperature_num"].fillna(298.0)
    df = df[(df["Affinity_mut_parsed"] > 0) & (df["Affinity_wt_parsed"] > 0)].copy()
    df["ddG_kcal_mol"] = R_GAS_KCAL * df["Temperature_num"] * np.log(df["Affinity_mut_parsed"] / df["Affinity_wt_parsed"])
    df["n_mutations"] = df["Mutation(s)_cleaned"].astype(str).str.count(",") + 1
    df["is_single_mutation"] = df["n_mutations"] == 1

    onebrs = df[df["pdb_id"].astype(str).str.upper() == "1BRS_A_D"].copy()
    onebrs_single = onebrs[onebrs["is_single_mutation"]].copy()
    parsed = onebrs_single["Mutation(s)_cleaned"].apply(parse_mutation_token)
    onebrs_single = onebrs_single[parsed.notna()].copy()
    parsed_df = pd.DataFrame(parsed[parsed.notna()].tolist(), index=parsed[parsed.notna()].index)
    onebrs_single = pd.concat([onebrs_single, parsed_df], axis=1)
    onebrs_single["position"] = onebrs_single["position_raw"].astype(int)
    def skempi_residue_id(row):
        struct_pos = int(row['position'])
        if row['chain'] == 'A':
            candidates = [struct_pos, struct_pos - 2, struct_pos + 2]
            for cand in candidates:
                struct_res = interface_df[(interface_df['chain'] == 'A') & (interface_df['resseq'] == cand)]
                if not struct_res.empty and struct_res.iloc[0]['aa'] == row['wt']:
                    struct_pos = cand
                    break
        return f"{row['chain']}:{row['wt']}{struct_pos}"

    onebrs_single["residue"] = onebrs_single.apply(skempi_residue_id, axis=1)

    merged = onebrs_single.merge(interface_df, on="residue", how="left", suffixes=("", "_struct"))
    merged["mutation_to_alanine"] = merged["mut"] == "A"
    merged["abs_ddG"] = merged["ddG_kcal_mol"].abs()

    global_summary = df.groupby(["is_single_mutation", "Hold_out_type"]).agg(
        n_entries=("pdb_id", "size"),
        median_ddG=("ddG_kcal_mol", "median"),
        mean_ddG=("ddG_kcal_mol", "mean"),
    ).reset_index()
    return df, onebrs_single, merged, global_summary


def summarize_chains(model, interface_df):
    rows = []
    for chain in model:
        residues = [r for r in chain if is_standard_residue(r)]
        atoms = [a for r in residues for a in get_residue_atoms(r)]
        interface_count = int(interface_df[interface_df["chain"] == chain.id]["is_interface_8A"].sum()) if chain.id in {"A", "D"} else 0
        rows.append({
            "chain": chain.id,
            "n_residues": len(residues),
            "n_heavy_atoms": len(atoms),
            "interface_residues_8A": interface_count,
        })
    return pd.DataFrame(rows)


def save_tables(interface_df, contacts_df, skempi_1brs, global_summary, chain_summary):
    interface_df.to_csv(OUT / "interface_residues.csv", index=False)
    contacts_df.to_csv(OUT / "interface_atom_contacts.csv", index=False)
    skempi_1brs.to_csv(OUT / "skempi_1brs_single_mutants.csv", index=False)
    global_summary.to_csv(OUT / "skempi_global_summary.csv", index=False)
    chain_summary.to_csv(OUT / "chain_summary.csv", index=False)


def plot_chain_interface(interface_df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    for ax, chain in zip(axes, ["A", "D"]):
        dfc = interface_df[interface_df["chain"] == chain].copy()
        colors = dfc["dominant_contact_type"].map({"electrostatic": "#d62728", "hydrophobic": "#1f77b4", "mixed": "#2ca02c", "none": "#bdbdbd"})
        ax.bar(dfc["resseq"], dfc["n_direct_contacts"], color=colors)
        ax.set_title(f"Chain {chain} interface contact profile")
        ax.set_xlabel("Residue number")
        ax.set_ylabel("Direct contacts (≤5 Å)")
    fig.tight_layout()
    fig.savefig(IMG / "interface_contact_profile.png", bbox_inches="tight")
    plt.close(fig)


def plot_contact_heatmap(contacts_df):
    direct = contacts_df[contacts_df["is_direct_contact"]].copy()
    pivot = direct.pivot_table(index="residue_1", columns="residue_2", values="min_atom_distance", aggfunc="min")
    pivot = pivot.sort_index(key=lambda x: [int(v.split(":")[1][1:]) for v in x])
    pivot = pivot[pivot.columns.sort_values(key=lambda x: [int(v.split(":")[1][1:]) for v in x])]
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pivot, cmap="viridis_r", ax=ax, cbar_kws={"label": "Minimum heavy-atom distance (Å)"})
    ax.set_title("Barnase–barstar direct contact map from the HADDOCK3 input structure")
    ax.set_xlabel("Barstar residues (chain D)")
    ax.set_ylabel("Barnase residues (chain A)")
    fig.tight_layout()
    fig.savefig(IMG / "contact_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_skempi_overview(global_df, skempi_1brs):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.histplot(global_df, x="ddG_kcal_mol", bins=60, ax=axes[0], color="#4c72b0")
    axes[0].set_title("Global SKEMPI ΔΔG distribution")
    axes[0].set_xlabel("ΔΔG (kcal/mol)")

    sns.histplot(skempi_1brs, x="ddG_kcal_mol", bins=15, ax=axes[1], color="#dd8452")
    axes[1].set_title("1BRS single-mutation ΔΔG distribution")
    axes[1].set_xlabel("ΔΔG (kcal/mol)")
    fig.tight_layout()
    fig.savefig(IMG / "skempi_ddg_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_structure_validation(merged):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=merged,
        x="min_atom_distance_to_partner",
        y="ddG_kcal_mol",
        hue="chain",
        style="mutation_to_alanine",
        s=120,
        ax=axes[0],
    )
    axes[0].set_title("Experimental effect vs structural proximity")
    axes[0].set_xlabel("Minimum distance to partner chain (Å)")
    axes[0].set_ylabel("ΔΔG (kcal/mol)")

    merged_sorted = merged.sort_values("ddG_kcal_mol", ascending=False)
    sns.barplot(data=merged_sorted, x="residue", y="ddG_kcal_mol", hue="chain", dodge=False, ax=axes[1])
    axes[1].tick_params(axis="x", rotation=90)
    axes[1].set_title("1BRS mutational hotspots from SKEMPI")
    axes[1].set_xlabel("Mutated residue")
    axes[1].set_ylabel("ΔΔG (kcal/mol)")
    fig.tight_layout()
    fig.savefig(IMG / "structure_mutation_validation.png", bbox_inches="tight")
    plt.close(fig)


def plot_ddg_vs_contacts(merged):
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.regplot(data=merged, x="n_direct_contacts", y="ddG_kcal_mol", scatter=False, ax=ax, color="black")
    sns.scatterplot(data=merged, x="n_direct_contacts", y="ddG_kcal_mol", hue="dominant_contact_type", style="chain", s=140, ax=ax)
    ax.set_title("Mutational destabilization tracks local interface density")
    ax.set_xlabel("Number of direct contacts in the structure")
    ax.set_ylabel("ΔΔG (kcal/mol)")
    fig.tight_layout()
    fig.savefig(IMG / "ddg_vs_contacts.png", bbox_inches="tight")
    plt.close(fig)


def compute_statistics(merged, interface_df, contacts_df, global_df):
    stats = {
        "n_interface_residues_total": int(interface_df["is_interface_8A"].sum()),
        "n_direct_contact_pairs": int(contacts_df["is_direct_contact"].sum()),
        "n_1brs_single_mutants": int(len(merged)),
        "global_skempi_entries": int(len(global_df)),
    }
    valid_dist = merged[["ddG_kcal_mol", "min_atom_distance_to_partner"]].dropna()
    valid_contacts = merged[["ddG_kcal_mol", "n_direct_contacts"]].dropna()
    if len(valid_dist) >= 3:
        stats["pearson_ddg_vs_distance"] = float(pearsonr(valid_dist["ddG_kcal_mol"], valid_dist["min_atom_distance_to_partner"]).statistic)
        stats["spearman_ddg_vs_distance"] = float(spearmanr(valid_dist["ddG_kcal_mol"], valid_dist["min_atom_distance_to_partner"]).statistic)
    if len(valid_contacts) >= 3:
        stats["pearson_ddg_vs_contacts"] = float(pearsonr(valid_contacts["ddG_kcal_mol"], valid_contacts["n_direct_contacts"]).statistic)
        stats["spearman_ddg_vs_contacts"] = float(spearmanr(valid_contacts["ddG_kcal_mol"], valid_contacts["n_direct_contacts"]).statistic)
    hotspot = merged.sort_values("ddG_kcal_mol", ascending=False).head(10)[[
        "residue", "ddG_kcal_mol", "n_direct_contacts", "min_atom_distance_to_partner", "closest_partner"
    ]]
    stats["top_hotspots"] = hotspot.to_dict(orient="records")
    return stats


def main():
    ensure_dirs()
    model = parse_structure()
    interface_df, contacts_df = analyze_interface(model)
    global_df, skempi_1brs_raw, merged, global_summary = load_skempi(interface_df)
    chain_summary = summarize_chains(model, interface_df)
    save_tables(interface_df, contacts_df, merged, global_summary, chain_summary)

    plot_chain_interface(interface_df)
    plot_contact_heatmap(contacts_df)
    plot_skempi_overview(global_df, merged)
    plot_structure_validation(merged)
    plot_ddg_vs_contacts(merged)

    stats = compute_statistics(merged, interface_df, contacts_df, global_df)
    with open(OUT / "analysis_summary.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
