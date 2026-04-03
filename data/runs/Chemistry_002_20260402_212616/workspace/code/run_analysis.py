#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
OUTPUTS_DIR = WORKSPACE / "outputs"
REPORT_IMG_DIR = WORKSPACE / "report" / "images"
PDB_PATH = DATA_DIR / "1brs_AD.pdb"
SKEMPI_PATH = DATA_DIR / "skempi_v2.csv"

CONTACT_CUTOFF = 5.0
INTERFACE_CUTOFF = 8.0
R_GAS_KCAL = 1.98720425864083e-3  # kcal mol^-1 K^-1
DEFAULT_TEMP_K = 298.0

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

CHAIN_TO_PROTEIN = {"A": "Barnase", "D": "Barstar"}
PROTEIN_TO_CHAIN = {v.lower(): k for k, v in CHAIN_TO_PROTEIN.items()}


@dataclass(frozen=True)
class Atom:
    atom_name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    coord: np.ndarray
    element: str


@dataclass(frozen=True)
class Residue:
    chain: str
    resseq: int
    icode: str
    resname: str
    atoms: Tuple[Atom, ...]

    @property
    def residue_id(self) -> str:
        return f"{self.chain}:{self.resseq}{self.icode.strip() or ''}"

    @property
    def resname1(self) -> str:
        return AA3_TO_1.get(self.resname, "X")

    @property
    def ca_coord(self) -> np.ndarray | None:
        for atom in self.atoms:
            if atom.atom_name == "CA":
                return atom.coord
        return None


@dataclass(frozen=True)
class Mutation:
    wt: str
    chain: str
    position: int
    mutant: str

    @property
    def mutation_code(self) -> str:
        return f"{self.wt}{self.chain}{self.position}{self.mutant}"


def ensure_dirs() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def parse_pdb(path: Path) -> List[Residue]:
    residues: Dict[Tuple[str, int, str, str], List[Atom]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue
            resname = line[17:20].strip()
            chain = line[21].strip()
            resseq = int(line[22:26])
            icode = line[26].strip()
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            element = line[76:78].strip() or atom_name[0]
            atom = Atom(
                atom_name=atom_name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                icode=icode,
                coord=np.array([x, y, z], dtype=float),
                element=element,
            )
            residues[(chain, resseq, icode, resname)].append(atom)

    ordered = []
    for (chain, resseq, icode, resname), atoms in sorted(
        residues.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
    ):
        ordered.append(Residue(chain=chain, resseq=resseq, icode=icode, resname=resname, atoms=tuple(atoms)))
    return ordered


def residue_min_distance(res1: Residue, res2: Residue) -> float:
    coords1 = np.array([a.coord for a in res1.atoms])
    coords2 = np.array([a.coord for a in res2.atoms])
    d = coords1[:, None, :] - coords2[None, :, :]
    return float(np.sqrt(np.sum(d * d, axis=2)).min())


def build_interface_tables(residues: List[Residue]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    residues_by_chain = defaultdict(list)
    for residue in residues:
        if residue.chain in CHAIN_TO_PROTEIN:
            residues_by_chain[residue.chain].append(residue)

    chain_a = residues_by_chain["A"]
    chain_d = residues_by_chain["D"]

    pair_rows = []
    residue_stats = defaultdict(lambda: {
        "contact_count_5A": 0,
        "contact_count_8A": 0,
        "closest_partner_distance": math.inf,
        "closest_partner_residue": None,
    })

    for res_a in chain_a:
        for res_d in chain_d:
            min_dist = residue_min_distance(res_a, res_d)
            pair_rows.append({
                "chain_1": res_a.chain,
                "resseq_1": res_a.resseq,
                "resname_1": res_a.resname,
                "residue_1": res_a.residue_id,
                "protein_1": CHAIN_TO_PROTEIN[res_a.chain],
                "chain_2": res_d.chain,
                "resseq_2": res_d.resseq,
                "resname_2": res_d.resname,
                "residue_2": res_d.residue_id,
                "protein_2": CHAIN_TO_PROTEIN[res_d.chain],
                "min_atom_distance": min_dist,
                "is_contact_5A": min_dist <= CONTACT_CUTOFF,
                "is_interface_8A": min_dist <= INTERFACE_CUTOFF,
            })

            key_a = (res_a.chain, res_a.resseq)
            key_d = (res_d.chain, res_d.resseq)
            residue_stats[key_a]["closest_partner_distance"] = min(
                residue_stats[key_a]["closest_partner_distance"], min_dist
            )
            residue_stats[key_d]["closest_partner_distance"] = min(
                residue_stats[key_d]["closest_partner_distance"], min_dist
            )
            if min_dist < residue_stats[key_a]["closest_partner_distance"] + 1e-12:
                residue_stats[key_a]["closest_partner_residue"] = res_d.residue_id
            if min_dist < residue_stats[key_d]["closest_partner_distance"] + 1e-12:
                residue_stats[key_d]["closest_partner_residue"] = res_a.residue_id
            if min_dist <= CONTACT_CUTOFF:
                residue_stats[key_a]["contact_count_5A"] += 1
                residue_stats[key_d]["contact_count_5A"] += 1
            if min_dist <= INTERFACE_CUTOFF:
                residue_stats[key_a]["contact_count_8A"] += 1
                residue_stats[key_d]["contact_count_8A"] += 1

    pair_df = pd.DataFrame(pair_rows).sort_values("min_atom_distance", ascending=True)

    residue_rows = []
    for residue in residues:
        if residue.chain not in CHAIN_TO_PROTEIN:
            continue
        stats = residue_stats[(residue.chain, residue.resseq)]
        residue_rows.append({
            "chain": residue.chain,
            "protein": CHAIN_TO_PROTEIN[residue.chain],
            "resseq": residue.resseq,
            "resname_3": residue.resname,
            "resname_1": residue.resname1,
            "residue_id": residue.residue_id,
            "contact_count_5A": stats["contact_count_5A"],
            "contact_count_8A": stats["contact_count_8A"],
            "closest_partner_distance": stats["closest_partner_distance"],
            "closest_partner_residue": stats["closest_partner_residue"],
            "is_interface_5A": stats["contact_count_5A"] > 0,
            "is_interface_8A": stats["contact_count_8A"] > 0,
        })
    residue_df = pd.DataFrame(residue_rows).sort_values(["chain", "resseq"])
    return pair_df, residue_df


MUTATION_RE = re.compile(r"^([A-Z])([A-Z])(\d+)([A-Z])$")


def parse_mutation_code(text: str) -> Mutation | None:
    text = str(text).strip()
    m = MUTATION_RE.match(text)
    if not m:
        return None
    wt, chain, position, mutant = m.groups()
    return Mutation(wt=wt, chain=chain, position=int(position), mutant=mutant)


def parse_temperature(value) -> float:
    if pd.isna(value):
        return DEFAULT_TEMP_K
    text = str(value)
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
    return float(m.group(1)) if m else DEFAULT_TEMP_K


def compute_ddg(kd_mut: float, kd_wt: float, temperature: float) -> float:
    if kd_mut <= 0 or kd_wt <= 0:
        return float("nan")
    return R_GAS_KCAL * temperature * math.log(kd_mut / kd_wt)


def load_skempi() -> pd.DataFrame:
    df = pd.read_csv(SKEMPI_PATH, sep=";")
    df["Temperature_K"] = df["Temperature"].apply(parse_temperature)
    df["ddG_kcal_mol"] = [
        compute_ddg(km, kw, t)
        for km, kw, t in zip(df["Affinity_mut_parsed"], df["Affinity_wt_parsed"], df["Temperature_K"])
    ]
    df["is_single_mutation"] = ~df["Mutation(s)_cleaned"].astype(str).str.contains(",", na=False
    )
    df["n_mutations"] = df["Mutation(s)_cleaned"].astype(str).apply(
        lambda s: len([x for x in s.split(",") if x.strip()])
    )
    df["complex_code"] = df["#Pdb"].astype(str).str.upper()
    df["is_1brs"] = df["complex_code"].eq("1BRS_A_D")
    return df


def annotate_barnase_barstar_mutations(
    skempi_1brs: pd.DataFrame, residue_df: pd.DataFrame
) -> pd.DataFrame:
    residue_lookup = {
        (row.chain, int(row.resseq)): row
        for row in residue_df.itertuples(index=False)
    }
    annotated_rows = []

    for row in skempi_1brs.to_dict(orient="records"):
        muts = [x.strip() for x in str(row["Mutation(s)_cleaned"]).split(",") if x.strip()]
        parsed = [parse_mutation_code(x) for x in muts]
        proteins = []
        chains = []
        positions = []
        wt_match = True
        interface_any_5 = False
        interface_any_8 = False
        min_distance = math.inf
        total_contacts_5 = 0
        total_contacts_8 = 0
        valid = True

        for mut in parsed:
            if mut is None:
                valid = False
                continue
            info = residue_lookup.get((mut.chain, mut.position))
            chains.append(mut.chain)
            positions.append(mut.position)
            proteins.append(CHAIN_TO_PROTEIN.get(mut.chain, "Unknown"))
            if info is None:
                wt_match = False
                valid = False
                continue
            wt_match = wt_match and (info.resname_1 == mut.wt)
            interface_any_5 = interface_any_5 or bool(info.is_interface_5A)
            interface_any_8 = interface_any_8 or bool(info.is_interface_8A)
            min_distance = min(min_distance, float(info.closest_partner_distance))
            total_contacts_5 += int(info.contact_count_5A)
            total_contacts_8 += int(info.contact_count_8A)

        if not parsed:
            valid = False

        annotated_rows.append({
            "#Pdb": row["#Pdb"],
            "Mutation(s)_cleaned": row["Mutation(s)_cleaned"],
            "Protein 1": row["Protein 1"],
            "Protein 2": row["Protein 2"],
            "Temperature_K": row["Temperature_K"],
            "ddG_kcal_mol": row["ddG_kcal_mol"],
            "n_mutations": row["n_mutations"],
            "is_single_mutation": row["is_single_mutation"],
            "mutation_chains": ",".join(chains),
            "mutation_positions": ",".join(map(str, positions)),
            "mutation_proteins": ",".join(proteins),
            "all_mutations_parsed": valid,
            "wildtype_matches_structure": wt_match,
            "interface_any_5A": interface_any_5,
            "interface_any_8A": interface_any_8,
            "min_partner_distance": min_distance if min_distance < math.inf else np.nan,
            "sum_contact_count_5A": total_contacts_5,
            "sum_contact_count_8A": total_contacts_8,
        })

    annotated = pd.DataFrame(annotated_rows)
    merged = pd.merge(
        skempi_1brs.reset_index(drop=True),
        annotated,
        on=[
            "#Pdb",
            "Mutation(s)_cleaned",
            "Protein 1",
            "Protein 2",
            "Temperature_K",
            "ddG_kcal_mol",
            "n_mutations",
            "is_single_mutation",
        ],
        how="left",
    )
    return merged


def save_tables(pair_df: pd.DataFrame, residue_df: pd.DataFrame, skempi_df: pd.DataFrame, brs_df: pd.DataFrame) -> None:
    pair_df.to_csv(OUTPUTS_DIR / "interface_residue_pairs.csv", index=False)
    residue_df.to_csv(OUTPUTS_DIR / "interface_residue_summary.csv", index=False)
    skempi_df.to_csv(OUTPUTS_DIR / "skempi_with_ddg.csv", index=False)
    brs_df.to_csv(OUTPUTS_DIR / "barnase_barstar_skempi_annotated.csv", index=False)

    overview = {
        "structure": {
            "pdb_file": str(PDB_PATH.relative_to(WORKSPACE)),
            "chains": residue_df.groupby("chain").size().to_dict(),
            "interface_residues_5A": residue_df.groupby("chain")["is_interface_5A"].sum().astype(int).to_dict(),
            "interface_residues_8A": residue_df.groupby("chain")["is_interface_8A"].sum().astype(int).to_dict(),
            "contact_pairs_5A": int(pair_df["is_contact_5A"].sum()),
            "contact_pairs_8A": int(pair_df["is_interface_8A"].sum()),
        },
        "skempi": {
            "total_rows": int(len(skempi_df)),
            "single_mutation_rows": int(skempi_df["is_single_mutation"].sum()),
            "complexes": int(skempi_df["#Pdb"].nunique()),
            "barnase_barstar_rows": int(len(brs_df)),
            "barnase_barstar_single_mutation_rows": int(brs_df["is_single_mutation"].sum()),
        },
    }
    (OUTPUTS_DIR / "analysis_overview.json").write_text(json.dumps(overview, indent=2))


def plot_skempi_ddg_distribution(skempi_df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    vals = skempi_df["ddG_kcal_mol"].replace([np.inf, -np.inf], np.nan).dropna()
    plt.hist(vals, bins=40, color="#4C72B0", edgecolor="white")
    plt.xlabel(r"$\Delta\Delta G$ (kcal/mol)")
    plt.ylabel("Count")
    plt.title("SKEMPI 2.0 mutation effect distribution")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "skempi_ddg_distribution.png", dpi=300)
    plt.close()


def plot_interface_profile(residue_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    colors = {"A": "#1f77b4", "D": "#d62728"}
    for ax, chain in zip(axes, ["A", "D"]):
        sub = residue_df[residue_df["chain"] == chain].copy()
        ax.bar(sub["resseq"], sub["contact_count_5A"], color=colors[chain], alpha=0.85)
        ax.set_ylabel("5 Å contacts")
        ax.set_title(f"{CHAIN_TO_PROTEIN[chain]} interface profile (chain {chain})")
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("Residue number")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "interface_contact_profile.png", dpi=300)
    plt.close()


def plot_distance_heatmap(pair_df: pd.DataFrame) -> None:
    pivot = pair_df.pivot(index="resseq_1", columns="resseq_2", values="min_atom_distance")
    plt.figure(figsize=(9, 7))
    arr = pivot.values
    plt.imshow(arr, aspect="auto", origin="lower", cmap="viridis_r", vmin=np.nanmin(arr), vmax=np.nanpercentile(arr, 95))
    plt.colorbar(label="Minimum heavy-atom distance (Å)")
    plt.xlabel("Barstar residue number (chain D)")
    plt.ylabel("Barnase residue number (chain A)")
    plt.title("Barnase–barstar inter-chain residue distance map")
    xticks = np.linspace(0, len(pivot.columns) - 1, 7, dtype=int)
    yticks = np.linspace(0, len(pivot.index) - 1, 7, dtype=int)
    plt.xticks(xticks, [pivot.columns[i] for i in xticks])
    plt.yticks(yticks, [pivot.index[i] for i in yticks])
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "interface_distance_heatmap.png", dpi=300)
    plt.close()


def plot_brs_validation(brs_df: pd.DataFrame) -> None:
    single = brs_df[brs_df["is_single_mutation"]].copy()
    single = single.mask(np.isinf(single.select_dtypes(include=[np.number])), np.nan)
    single = single.dropna(subset=["ddG_kcal_mol", "min_partner_distance"])
    if single.empty:
        return

    plt.figure(figsize=(7, 5.5))
    colors = np.where(single["interface_any_5A"], "#C44E52", "#55A868")
    plt.scatter(single["min_partner_distance"], single["ddG_kcal_mol"], c=colors, alpha=0.9, edgecolor="black", linewidth=0.3)
    plt.xlabel("Closest partner distance in native complex (Å)")
    plt.ylabel(r"Experimental $\Delta\Delta G$ (kcal/mol)")
    plt.title("Mutation impact tracks native interface proximity in 1BRS")
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="Interface residue (≤5 Å)", markerfacecolor="#C44E52", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Non-interface residue", markerfacecolor="#55A868", markeredgecolor="black", markersize=8),
    ]
    plt.legend(handles=legend_items, frameon=False)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "barnase_barstar_ddg_vs_distance.png", dpi=300)
    plt.close()

    chain_order = ["A", "D"]
    chain_vals = []
    labels = []
    for chain in chain_order:
        subset = single[single["mutation_chains"] == chain]
        if not subset.empty:
            chain_vals.append(subset["ddG_kcal_mol"].values)
            labels.append(f"{CHAIN_TO_PROTEIN[chain]} ({chain})")
    if chain_vals:
        plt.figure(figsize=(6.5, 5))
        plt.boxplot(chain_vals, tick_labels=labels, patch_artist=True,
                    boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                    medianprops=dict(color="black"))
        plt.ylabel(r"Experimental $\Delta\Delta G$ (kcal/mol)")
        plt.title("Single-mutation effects by mutated partner in 1BRS")
        plt.tight_layout()
        plt.savefig(REPORT_IMG_DIR / "barnase_barstar_ddg_by_chain.png", dpi=300)
        plt.close()


def save_statistics(brs_df: pd.DataFrame, residue_df: pd.DataFrame) -> None:
    single = brs_df[brs_df["is_single_mutation"]].copy()
    stats = {}
    if not single.empty:
        stats["single_mutation_count"] = int(len(single))
        stats["single_mutation_median_ddg"] = float(single["ddG_kcal_mol"].median())
        stats["single_mutation_max_ddg"] = float(single["ddG_kcal_mol"].max())
        stats["single_mutation_min_ddg"] = float(single["ddG_kcal_mol"].min())
        if single["min_partner_distance"].notna().sum() >= 3:
            corr = single[["ddG_kcal_mol", "min_partner_distance"]].corr(method="spearman").iloc[0, 1]
            stats["spearman_ddg_vs_min_distance"] = float(corr)
        grouped = single.groupby("interface_any_5A")["ddG_kcal_mol"].agg(["count", "median", "mean"])
        stats["ddg_by_interface_5A"] = {
            str(k): {kk: float(vv) for kk, vv in vals.items()}
            for k, vals in grouped.to_dict(orient="index").items()
        }

    top_interface = residue_df.sort_values(["contact_count_5A", "closest_partner_distance"], ascending=[False, True]).head(15)
    top_interface.to_csv(OUTPUTS_DIR / "top_interface_residues.csv", index=False)
    (OUTPUTS_DIR / "barnase_barstar_statistics.json").write_text(json.dumps(stats, indent=2))


def main() -> None:
    ensure_dirs()
    residues = parse_pdb(PDB_PATH)
    pair_df, residue_df = build_interface_tables(residues)
    skempi_df = load_skempi()
    brs_df = skempi_df[skempi_df["is_1brs"]].copy()
    brs_df = annotate_barnase_barstar_mutations(brs_df, residue_df)

    save_tables(pair_df, residue_df, skempi_df, brs_df)
    save_statistics(brs_df, residue_df)
    plot_skempi_ddg_distribution(skempi_df)
    plot_interface_profile(residue_df)
    plot_distance_heatmap(pair_df)
    plot_brs_validation(brs_df)

    print("Analysis complete.")
    print(f"Wrote outputs to: {OUTPUTS_DIR}")
    print(f"Wrote figures to: {REPORT_IMG_DIR}")


if __name__ == "__main__":
    main()
