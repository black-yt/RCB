from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"
MPL_DIR = ROOT / "outputs" / "mplconfig"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="talk")

R_KCAL = 1.98720425864083e-3
AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
KD_SCALE = {
    "I": 4.5,
    "V": 4.2,
    "L": 3.8,
    "F": 2.8,
    "C": 2.5,
    "M": 1.9,
    "A": 1.8,
    "G": -0.4,
    "T": -0.7,
    "S": -0.8,
    "W": -0.9,
    "Y": -1.3,
    "P": -1.6,
    "H": -3.2,
    "E": -3.5,
    "Q": -3.5,
    "D": -3.5,
    "N": -3.5,
    "K": -3.9,
    "R": -4.5,
}
AA_VOLUME = {
    "G": 60.1,
    "A": 88.6,
    "S": 89.0,
    "C": 108.5,
    "D": 111.1,
    "P": 112.7,
    "N": 114.1,
    "T": 116.1,
    "E": 138.4,
    "V": 140.0,
    "Q": 143.8,
    "H": 153.2,
    "M": 162.9,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "R": 173.4,
    "F": 189.9,
    "Y": 193.6,
    "W": 227.8,
}
CHARGE = {
    "D": -1,
    "E": -1,
    "K": 1,
    "R": 1,
    "H": 0.1,
}
AROMATIC = {"F", "W", "Y", "H"}


@dataclass(frozen=True)
class Atom:
    chain: str
    resseq: int
    resname: str
    atom_name: str
    coord: np.ndarray


def parse_temperature(value: object) -> float:
    if pd.isna(value):
        return 298.0
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value))
    return float(match.group(1)) if match else 298.0


def parse_mutation_token(token: str) -> Dict[str, object]:
    token = token.strip()
    match = re.fullmatch(r"([A-Z])([A-Z])(\d+)([A-Z])", token)
    if not match:
        raise ValueError(f"Unsupported mutation token: {token}")
    wt, chain, residue, mutant = match.groups()
    return {
        "chain": chain,
        "wt": wt,
        "resseq": int(residue),
        "mut": mutant,
        "token": token,
    }


def parse_pdb(pdb_path: Path) -> Tuple[pd.DataFrame, Dict[Tuple[str, int], List[Atom]]]:
    atoms: List[Atom] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom = Atom(
                chain=line[21].strip(),
                resseq=int(line[22:26]),
                resname=line[17:20].strip(),
                atom_name=line[12:16].strip(),
                coord=np.array(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                ),
            )
            atoms.append(atom)

    atom_map: Dict[Tuple[str, int], List[Atom]] = {}
    residue_rows = []
    for atom in atoms:
        key = (atom.chain, atom.resseq)
        atom_map.setdefault(key, []).append(atom)

    for (chain, resseq), residue_atoms in sorted(atom_map.items()):
        resname = residue_atoms[0].resname
        ca = next((a.coord for a in residue_atoms if a.atom_name == "CA"), residue_atoms[0].coord)
        residue_rows.append(
            {
                "chain": chain,
                "resseq": resseq,
                "resname3": resname,
                "resname1": AA3_TO_1.get(resname, "X"),
                "n_atoms": len(residue_atoms),
                "ca_x": ca[0],
                "ca_y": ca[1],
                "ca_z": ca[2],
            }
        )

    residue_df = pd.DataFrame(residue_rows)
    return residue_df, atom_map


def build_interface_tables(
    residue_df: pd.DataFrame,
    atom_map: Dict[Tuple[str, int], List[Atom]],
    chain_a: str = "A",
    chain_b: str = "D",
    distance_cutoff: float = 5.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    residues_a = residue_df[residue_df["chain"] == chain_a].copy()
    residues_b = residue_df[residue_df["chain"] == chain_b].copy()
    pair_rows = []
    residue_stats: Dict[Tuple[str, int], Dict[str, float]] = {}

    for _, row_a in residues_a.iterrows():
        atoms_a = atom_map[(row_a.chain, row_a.resseq)]
        coords_a = np.vstack([a.coord for a in atoms_a])
        for _, row_b in residues_b.iterrows():
            atoms_b = atom_map[(row_b.chain, row_b.resseq)]
            coords_b = np.vstack([a.coord for a in atoms_b])
            distances = np.linalg.norm(coords_a[:, None, :] - coords_b[None, :, :], axis=2)
            min_distance = float(distances.min())
            atom_contacts = int((distances <= distance_cutoff).sum())
            pair_rows.append(
                {
                    "chain_a": row_a.chain,
                    "resseq_a": int(row_a.resseq),
                    "resname_a": row_a.resname1,
                    "chain_b": row_b.chain,
                    "resseq_b": int(row_b.resseq),
                    "resname_b": row_b.resname1,
                    "min_distance": min_distance,
                    "atom_contacts_5A": atom_contacts,
                    "is_contact": int(min_distance <= distance_cutoff),
                }
            )
            if min_distance <= distance_cutoff:
                for residue_key, partner_key in [
                    ((row_a.chain, int(row_a.resseq)), (row_b.chain, int(row_b.resseq))),
                    ((row_b.chain, int(row_b.resseq)), (row_a.chain, int(row_a.resseq))),
                ]:
                    stats = residue_stats.setdefault(
                        residue_key,
                        {
                            "contact_partner_count": 0,
                            "atom_contact_count": 0,
                            "min_interchain_distance": math.inf,
                        },
                    )
                    stats["contact_partner_count"] += 1
                    stats["atom_contact_count"] += atom_contacts
                    stats["min_interchain_distance"] = min(
                        stats["min_interchain_distance"],
                        min_distance,
                    )

    pair_df = pd.DataFrame(pair_rows)
    residue_interface = residue_df.copy()
    residue_interface["contact_partner_count"] = 0
    residue_interface["atom_contact_count"] = 0
    residue_interface["min_interchain_distance"] = np.nan
    residue_interface["is_interface"] = 0

    for idx, row in residue_interface.iterrows():
        key = (row["chain"], int(row["resseq"]))
        if key in residue_stats:
            residue_interface.at[idx, "contact_partner_count"] = int(
                residue_stats[key]["contact_partner_count"]
            )
            residue_interface.at[idx, "atom_contact_count"] = int(
                residue_stats[key]["atom_contact_count"]
            )
            residue_interface.at[idx, "min_interchain_distance"] = float(
                residue_stats[key]["min_interchain_distance"]
            )
            residue_interface.at[idx, "is_interface"] = 1

    return residue_interface, pair_df


def load_skempi(path: Path) -> pd.DataFrame:
    with path.open() as handle:
        header = handle.readline().strip().lstrip("#").split(";")
    df = pd.read_csv(path, sep=";", names=header, skiprows=1)
    df["Temperature_K"] = df["Temperature"].apply(parse_temperature)
    df["Affinity_mut_parsed"] = pd.to_numeric(df["Affinity_mut_parsed"], errors="coerce")
    df["Affinity_wt_parsed"] = pd.to_numeric(df["Affinity_wt_parsed"], errors="coerce")
    df["ddG_kcal_mol"] = (
        R_KCAL
        * df["Temperature_K"]
        * np.log(df["Affinity_mut_parsed"] / df["Affinity_wt_parsed"])
    )
    df["n_mutations"] = df["Mutation(s)_cleaned"].astype(str).str.count(",") + 1
    return df


def explode_mutations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cleaned_tokens = [token.strip() for token in str(row["Mutation(s)_cleaned"]).split(",")]
        pdb_tokens = [token.strip() for token in str(row["Mutation(s)_PDB"]).split(",")]
        if len(cleaned_tokens) != len(pdb_tokens):
            raise ValueError(
                f"Mismatch between cleaned and PDB mutation token counts for record {row['record_id']}"
            )
        for cleaned_token, pdb_token in zip(cleaned_tokens, pdb_tokens):
            parsed = parse_mutation_token(pdb_token)
            rows.append(
                {
                    "record_id": row["record_id"],
                    "Pdb": row["Pdb"],
                    "mutation_token": cleaned_token,
                    "mutation_token_pdb": pdb_token,
                    "chain": parsed["chain"],
                    "resseq": parsed["resseq"],
                    "wt": parsed["wt"],
                    "mut": parsed["mut"],
                }
            )
    return pd.DataFrame(rows)


def mutation_change_features(wt: str, mut: str) -> Dict[str, float]:
    wt_charge = CHARGE.get(wt, 0.0)
    mut_charge = CHARGE.get(mut, 0.0)
    return {
        "to_alanine": int(mut == "A"),
        "is_charge_reversal": int(wt_charge * mut_charge < 0),
        "is_charge_loss": int((wt_charge != 0.0) and (mut_charge == 0.0)),
        "is_charge_gain": int((wt_charge == 0.0) and (mut_charge != 0.0)),
        "is_aromatic_loss": int((wt in AROMATIC) and (mut not in AROMATIC)),
        "is_aromatic_gain": int((wt not in AROMATIC) and (mut in AROMATIC)),
        "hydropathy_change_abs": abs(KD_SCALE[mut] - KD_SCALE[wt]),
        "volume_change_abs": abs(AA_VOLUME[mut] - AA_VOLUME[wt]),
    }


def build_mutation_feature_table(
    skempi_1brs: pd.DataFrame,
    residue_interface: pd.DataFrame,
) -> pd.DataFrame:
    work = skempi_1brs.copy().reset_index(drop=True)
    work["record_id"] = np.arange(len(work))
    exploded = explode_mutations(work)
    residue_lookup = residue_interface.set_index(["chain", "resseq"])

    expanded_rows = []
    for _, row in exploded.iterrows():
        features = mutation_change_features(str(row["wt"]), str(row["mut"]))
        residue_row = residue_lookup.loc[(row["chain"], int(row["resseq"]))]
        expanded_rows.append(
            {
                **row.to_dict(),
                **features,
                "resname1": residue_row["resname1"],
                "resname3": residue_row["resname3"],
                "is_interface": int(residue_row["is_interface"]),
                "contact_partner_count": float(residue_row["contact_partner_count"]),
                "atom_contact_count": float(residue_row["atom_contact_count"]),
                "min_interchain_distance": float(residue_row["min_interchain_distance"])
                if not pd.isna(residue_row["min_interchain_distance"])
                else np.nan,
            }
        )

    expanded_df = pd.DataFrame(expanded_rows)
    agg = (
        expanded_df.groupby("record_id")
        .agg(
            parsed_n_mutations=("mutation_token", "count"),
            sum_contact_partner_count=("contact_partner_count", "sum"),
            sum_atom_contact_count=("atom_contact_count", "sum"),
            mean_min_interchain_distance=("min_interchain_distance", "mean"),
            n_interface_mutations=("is_interface", "sum"),
            n_to_alanine=("to_alanine", "sum"),
            n_charge_reversal=("is_charge_reversal", "sum"),
            n_charge_loss=("is_charge_loss", "sum"),
            n_charge_gain=("is_charge_gain", "sum"),
            n_aromatic_loss=("is_aromatic_loss", "sum"),
            n_aromatic_gain=("is_aromatic_gain", "sum"),
            sum_hydropathy_change_abs=("hydropathy_change_abs", "sum"),
            sum_volume_change_abs=("volume_change_abs", "sum"),
        )
        .reset_index()
    )
    merged = work.merge(agg, on="record_id", how="left")
    return merged, expanded_df


def loocv_predictions(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    target_col: str,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    X = df[list(feature_cols)].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    preds = np.zeros_like(y)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])

    pearson = float(np.corrcoef(y, preds)[0, 1])
    spearman = float(pd.Series(y).corr(pd.Series(preds), method="spearman"))
    metrics = {
        "mae_kcal_mol": float(mean_absolute_error(y, preds)),
        "r2": float(r2_score(y, preds)),
        "pearson_r": pearson,
        "spearman_rho": spearman,
    }

    final_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    final_model.fit(X, y)
    coefs = final_model.named_steps["ridge"].coef_
    return preds, metrics, coefs


def make_overview_figure(skempi: pd.DataFrame, skempi_1brs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(
        skempi["ddG_kcal_mol"].dropna(),
        bins=50,
        color="#9aa5b1",
        ax=axes[0],
        stat="density",
        label="All SKEMPI v2",
    )
    sns.kdeplot(
        skempi_1brs["ddG_kcal_mol"].dropna(),
        color="#c0392b",
        linewidth=2.5,
        ax=axes[0],
        label="1BRS subset",
    )
    axes[0].set_xlabel(r"$\Delta\Delta G$ (kcal/mol)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Binding free-energy changes in SKEMPI")
    axes[0].legend(frameon=False)

    location_order = (
        skempi_1brs["iMutation_Location(s)"]
        .fillna("UNK")
        .str.split(",")
        .explode()
        .value_counts()
        .index
    )
    location_counts = (
        skempi_1brs["iMutation_Location(s)"]
        .fillna("UNK")
        .str.split(",")
        .explode()
        .value_counts()
        .reindex(location_order)
        .reset_index()
    )
    location_counts.columns = ["location", "count"]
    sns.barplot(
        data=location_counts,
        x="location",
        y="count",
        color="#4c78a8",
        ax=axes[1],
    )
    axes[1].set_xlabel("Mutation location class")
    axes[1].set_ylabel("Count in 1BRS subset")
    axes[1].set_title("Barnase-barstar mutation coverage")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_1_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_contact_map_figure(pair_df: pd.DataFrame, expanded_mutations: pd.DataFrame) -> None:
    a_mut = sorted(expanded_mutations.loc[expanded_mutations["chain"] == "A", "resseq"].unique())
    d_mut = sorted(expanded_mutations.loc[expanded_mutations["chain"] == "D", "resseq"].unique())
    subset = pair_df[
        pair_df["resseq_a"].isin(a_mut) & pair_df["resseq_b"].isin(d_mut)
    ].copy()
    heatmap = subset.pivot(index="resseq_a", columns="resseq_b", values="min_distance").sort_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap,
        cmap="mako_r",
        linewidths=0.5,
        cbar_kws={"label": "Minimum heavy-atom distance (A)"},
        ax=ax,
    )
    ax.set_xlabel("Barstar chain D residue")
    ax.set_ylabel("Barnase chain A residue")
    ax.set_title("Inter-chain distances for mutated residues in 1BRS")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_2_contact_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_hotspot_figure(single_site: pd.DataFrame, residue_interface: pd.DataFrame) -> None:
    residue_summary = (
        single_site.groupby(["chain", "resseq"])
        .agg(
            mean_ddg=("ddG_kcal_mol", "mean"),
            max_ddg=("ddG_kcal_mol", "max"),
            n_measurements=("ddG_kcal_mol", "count"),
        )
        .reset_index()
    )
    residue_summary = residue_summary.merge(
        residue_interface[
            [
                "chain",
                "resseq",
                "resname1",
                "contact_partner_count",
                "atom_contact_count",
                "min_interchain_distance",
            ]
        ],
        on=["chain", "resseq"],
        how="left",
    )
    residue_summary["label"] = (
        residue_summary["resname1"] + residue_summary["chain"] + residue_summary["resseq"].astype(str)
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        residue_summary["atom_contact_count"],
        residue_summary["mean_ddg"],
        s=60 + 30 * residue_summary["n_measurements"],
        c=residue_summary["min_interchain_distance"],
        cmap="viridis_r",
        edgecolor="black",
        alpha=0.9,
    )
    for _, row in residue_summary.sort_values("mean_ddg", ascending=False).head(8).iterrows():
        ax.text(
            row["atom_contact_count"] + 0.6,
            row["mean_ddg"] + 0.03,
            row["label"],
            fontsize=10,
        )
    ax.set_xlabel("Inter-chain atom contacts within 5 A")
    ax.set_ylabel(r"Mean single-mutation $\Delta\Delta G$ (kcal/mol)")
    ax.set_title("Structural centrality tracks mutational hotspots")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Closest inter-chain distance (A)")
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_3_hotspots.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_model_figure(
    modeled_df: pd.DataFrame,
    predictions: np.ndarray,
    metrics: Dict[str, float],
    feature_cols: List[str],
    coefs: np.ndarray,
) -> None:
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": coefs}).sort_values("coefficient")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].scatter(
        modeled_df["ddG_kcal_mol"],
        predictions,
        c=modeled_df["n_mutations"],
        cmap="rocket",
        s=70,
        edgecolor="black",
        alpha=0.85,
    )
    lims = [
        min(modeled_df["ddG_kcal_mol"].min(), predictions.min()) - 0.2,
        max(modeled_df["ddG_kcal_mol"].max(), predictions.max()) + 0.2,
    ]
    axes[0].plot(lims, lims, linestyle="--", color="black", linewidth=1)
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel(r"Observed $\Delta\Delta G$ (kcal/mol)")
    axes[0].set_ylabel(r"LOOCV predicted $\Delta\Delta G$ (kcal/mol)")
    axes[0].set_title(
        "Cross-validated additive model\n"
        f"MAE={metrics['mae_kcal_mol']:.2f}, Pearson r={metrics['pearson_r']:.2f}"
    )

    sns.barplot(
        data=coef_df,
        x="coefficient",
        y="feature",
        color="#72b7b2",
        ax=axes[1],
    )
    axes[1].set_xlabel("Ridge coefficient")
    axes[1].set_ylabel("")
    axes[1].set_title("Full-data standardized model coefficients")

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "figure_4_model_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    residue_df, atom_map = parse_pdb(DATA_DIR / "1brs_AD.pdb")
    residue_interface, pair_df = build_interface_tables(residue_df, atom_map)
    skempi = load_skempi(DATA_DIR / "skempi_v2.csv")
    skempi_1brs = skempi[skempi["Pdb"] == "1BRS_A_D"].copy().reset_index(drop=True)
    mutation_features, expanded_mutations = build_mutation_feature_table(skempi_1brs, residue_interface)
    expanded_with_response = expanded_mutations.merge(
        mutation_features[["record_id", "ddG_kcal_mol", "n_mutations"]],
        on="record_id",
        how="left",
    )

    single_site = mutation_features[mutation_features["n_mutations"] == 1].copy()
    single_site_expanded = expanded_with_response[expanded_with_response["n_mutations"] == 1].copy()
    feature_cols = [
        "n_mutations",
        "sum_contact_partner_count",
        "sum_atom_contact_count",
        "mean_min_interchain_distance",
        "n_interface_mutations",
        "n_to_alanine",
        "n_charge_reversal",
        "n_charge_loss",
        "n_aromatic_loss",
        "sum_hydropathy_change_abs",
        "sum_volume_change_abs",
    ]
    modeled_df = mutation_features.dropna(subset=feature_cols + ["ddG_kcal_mol"]).copy()
    predictions, metrics, coefs = loocv_predictions(modeled_df, feature_cols, "ddG_kcal_mol")
    modeled_df["predicted_ddG_kcal_mol"] = predictions

    residue_mut_summary = (
        single_site_expanded.groupby(["chain", "resseq"])
        .agg(
            mean_ddg_kcal_mol=("ddG_kcal_mol", "mean"),
            max_ddg_kcal_mol=("ddG_kcal_mol", "max"),
            n_single_mutants=("ddG_kcal_mol", "count"),
        )
        .reset_index()
        .merge(residue_interface, on=["chain", "resseq"], how="left")
        .sort_values("mean_ddg_kcal_mol", ascending=False)
    )

    interface_summary = {
        "n_residues_total": int(len(residue_interface)),
        "n_interface_residues": int(residue_interface["is_interface"].sum()),
        "chain_A_interface_residues": int(
            residue_interface.query("chain == 'A'")["is_interface"].sum()
        ),
        "chain_D_interface_residues": int(
            residue_interface.query("chain == 'D'")["is_interface"].sum()
        ),
        "n_1brs_skempi_measurements": int(len(skempi_1brs)),
        "n_single_mutants": int((skempi_1brs["n_mutations"] == 1).sum()),
        "n_double_mutants": int((skempi_1brs["n_mutations"] == 2).sum()),
        "single_mutation_spearman_contact_vs_ddg": float(
            single_site["sum_atom_contact_count"].corr(single_site["ddG_kcal_mol"], method="spearman")
        ),
    }

    residue_interface.to_csv(OUTPUT_DIR / "1brs_residue_interface.csv", index=False)
    pair_df.to_csv(OUTPUT_DIR / "1brs_pairwise_contacts.csv", index=False)
    mutation_features.to_csv(OUTPUT_DIR / "1brs_mutation_features.csv", index=False)
    expanded_mutations.to_csv(OUTPUT_DIR / "1brs_expanded_mutations.csv", index=False)
    modeled_df.to_csv(OUTPUT_DIR / "1brs_model_predictions.csv", index=False)
    residue_mut_summary.to_csv(OUTPUT_DIR / "1brs_residue_hotspots.csv", index=False)
    with (OUTPUT_DIR / "summary_metrics.json").open("w") as handle:
        json.dump(
            {
                "interface_summary": interface_summary,
                "model_metrics": metrics,
                "feature_columns": feature_cols,
                "feature_coefficients": {feature: float(coef) for feature, coef in zip(feature_cols, coefs)},
            },
            handle,
            indent=2,
        )

    make_overview_figure(skempi, skempi_1brs)
    make_contact_map_figure(pair_df, expanded_mutations)
    make_hotspot_figure(single_site_expanded, residue_interface)
    make_model_figure(modeled_df, predictions, metrics, feature_cols, coefs)


if __name__ == "__main__":
    main()
