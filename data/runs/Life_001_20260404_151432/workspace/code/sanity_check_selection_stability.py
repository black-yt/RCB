import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"


def stable_hash(items):
    payload = "|".join(sorted(map(str, items)))
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def main():
    selected = pd.read_csv(DATA / "selected-vaccine-elements.budget-10.minsum.adaptive.csv")
    cells = pd.read_csv(DATA / "cell-populations.csv")
    sim = pd.read_csv(DATA / "sim-specific-response-likelihoods.csv")

    rows = []
    for rep, grp in selected.groupby("repetition"):
        selected_set = sorted(set(grp["peptide"]))
        cell_rep = cells[cells["repetition"] == rep]
        sim_rep = sim[sim["vaccine"].str.endswith(f"rep-{rep}")]
        rows.append({
            "repetition": int(rep),
            "selected_hash": stable_hash(selected_set),
            "selected_tuple": ";".join(selected_set),
            "n_selected": len(selected_set),
            "cell_mutation_hash": stable_hash(sorted(cell_rep["mutation"].astype(str).unique())),
            "n_unique_cell_mutations": int(cell_rep["mutation"].nunique()),
            "mean_num_presented_peptides": float(sim_rep["num_presented_peptides"].mean()),
            "mean_p_response": float(sim_rep["p_response"].mean()),
        })

    out = pd.DataFrame(rows).sort_values("repetition")
    out.to_csv(OUTPUTS / "selection_stability_sanity_check.csv", index=False)

    note = {
        "all_selected_hashes_identical": bool(out["selected_hash"].nunique() == 1),
        "all_cell_mutation_hashes_identical": bool(out["cell_mutation_hash"].nunique() == 1),
        "selected_hash": out["selected_hash"].iloc[0],
        "interpretation": "Perfect IoU is consistent with repetition-invariant selected sets. Cell-population mutation support is also invariant across repetitions at the set level, suggesting the optimizer repeatedly selects the same 10 mutation elements rather than exhibiting stochastic instability."
    }
    with open(OUTPUTS / "selection_stability_note.json", "w", encoding="utf-8") as f:
        json.dump(note, f, indent=2)


if __name__ == "__main__":
    main()
