from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "2111.01152"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"


LATEX_STOPWORDS = {
    "begin",
    "end",
    "pmatrix",
    "mathbf",
    "mathcal",
    "left",
    "right",
    "text",
    "mathrm",
    "quad",
    "qquad",
    "tau",
    "bm",
    "hat",
    "sum",
    "frac",
    "int",
    "cdot",
    "dagger",
    "alpha",
    "beta",
    "gamma",
    "delta",
    "l",
    "k",
    "r",
}


@dataclass
class TaskRecord:
    step_id: int
    task: str
    source: Any
    answer: str
    completion: str
    scores: dict[str, float]
    placeholder_reviewer_mean: float | None
    placeholder_reviewer_std: float | None
    sequence_ratio: float | None
    token_precision: float | None
    token_recall: float | None
    token_f1: float | None
    symbol_recall: float | None
    automated_step_score: float | None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def latex_tokens(text: str) -> list[str]:
    text = text or ""
    raw = re.findall(r"\\[A-Za-z]+|[A-Za-z_]+(?:\d+)?|\d+(?:\.\d+)?", text)
    tokens = []
    for tok in raw:
        tok = tok.lstrip("\\").lower()
        if tok in LATEX_STOPWORDS:
            continue
        tokens.append(tok)
    return tokens


def sequence_ratio(a: str, b: str) -> float | None:
    if not a or not b:
        return None
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def token_overlap_metrics(reference: str, candidate: str) -> tuple[float | None, float | None, float | None]:
    if not reference or not candidate:
        return None, None, None
    ref_tokens = latex_tokens(reference)
    cand_tokens = latex_tokens(candidate)
    if not ref_tokens or not cand_tokens:
        return None, None, None

    ref_counts: dict[str, int] = {}
    cand_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1
    for tok in cand_tokens:
        cand_counts[tok] = cand_counts.get(tok, 0) + 1

    common = sum(min(ref_counts.get(tok, 0), cand_counts.get(tok, 0)) for tok in set(ref_counts) | set(cand_counts))
    precision = common / len(cand_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def symbol_recall(reference: str, candidate: str) -> float | None:
    if not reference or not candidate:
        return None
    ref_symbols = {tok for tok in latex_tokens(reference) if len(tok) > 1}
    cand_symbols = set(latex_tokens(candidate))
    if not ref_symbols:
        return None
    return len(ref_symbols & cand_symbols) / len(ref_symbols)


def automated_step_score(reference: str, candidate: str) -> float | None:
    _, recall, _ = token_overlap_metrics(reference, candidate)
    if recall is None:
        return None
    return 2.0 * recall


def placeholder_stats(placeholder: dict[str, Any]) -> tuple[float | None, float | None]:
    vals = []
    for value in placeholder.values():
        if isinstance(value, dict) and isinstance(value.get("score"), dict):
            vals.extend(v for v in value["score"].values() if isinstance(v, (int, float)))
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def parse_auto_md(path: Path) -> dict[str, dict[str, str]]:
    text = read_text(path)
    chunks = re.split(r"^## ", text, flags=re.MULTILINE)
    parsed: dict[str, dict[str, str]] = {}
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        header = lines[0].strip()
        body = "\n".join(lines[1:])
        prompt_match = re.search(r"\*\*Prompt:\*\*\s*(.*?)\s*\*\*Completion:\*\*", body, flags=re.DOTALL)
        completion_match = re.search(r"\*\*Completion:\*\*\s*(.*)\Z", body, flags=re.DOTALL)
        parsed[header] = {
            "prompt": prompt_match.group(1).strip() if prompt_match else "",
            "completion": completion_match.group(1).strip() if completion_match else "",
        }
    return parsed


def eq_block(text: str, label: str) -> str:
    pattern = re.compile(
        rf"\\begin\{{(equation|eqnarray|aligned|split)\}}.*?\\label\{{{re.escape(label)}\}}(.*?)\\end\{{\1\}}",
        flags=re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group(2).strip()
    return ""


def parse_related_work_titles() -> list[dict[str, str]]:
    results = []
    for pdf_path in sorted((ROOT / "related_work").glob("*.pdf")):
        cmd = ["pdftotext", "-f", "1", "-l", "1", str(pdf_path), "-"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        first_page = proc.stdout.strip()
        lines = [ln.strip() for ln in first_page.splitlines() if ln.strip()]
        title = ""
        if lines:
            if "MIT Open Access Articles" in lines[0] and len(lines) > 1:
                title = lines[1]
            else:
                title = lines[0]
                if len(title) < 20 and len(lines) > 1:
                    title = f"{title} {lines[1]}".strip()
        results.append({"file": pdf_path.name, "title": title})
    return results


def extract_metadata(main_tex: str) -> dict[str, Any]:
    title_match = re.search(r"\\title\{(.+)\}\s*$", main_tex, flags=re.MULTILINE)
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_tex, flags=re.DOTALL)
    authors = re.findall(r"\\author\{(.*?)\}", main_tex)

    a_b = 3.575
    a_t = 3.32
    a_m = a_b * a_t / abs(a_b - a_t)
    return {
        "paper_id": "2111.01152",
        "title": title_match.group(1).strip() if title_match else "",
        "authors": authors,
        "abstract": re.sub(r"\s+", " ", abstract_match.group(1).strip()) if abstract_match else "",
        "system": "AB-stacked MoTe2/WSe2 heterobilayer",
        "method": "Self-consistent Hartree-Fock in a plane-wave continuum basis",
        "moire_period_angstrom": a_m,
        "moire_period_nm": a_m / 10.0,
        "effective_masses_me": {"bottom": 0.65, "top": 0.35},
        "representative_parameters_meV": {"w": 12.0, "V_b": 7.0, "V_zt": -20.0},
        "phase_targets": ["nu=2 Z2 topological insulator", "nu=1 Chern/valley/spin-density-wave phases", "nu=2/3 topological charge density wave"],
    }


def build_derivation(main_tex: str, sm_tex: str, task_answers: dict[str, str]) -> dict[str, str]:
    h_tau = eq_block(main_tex, "eq:Ham")
    delta_b = eq_block(main_tex, "eq:Delta_b")
    delta_t = eq_block(main_tex, "eq:Delta_T")
    hf_full = eq_block(sm_tex, "eq:HF")
    full_h = eq_block(sm_tex, "eq:full")

    second_quantized_real_space = (
        r"\hat{\mathcal{H}}_0=\sum_{\tau=\pm}\int d^2\bm{r}\,\Psi_{\tau}^{\dagger}(\bm{r})H_{\tau}\Psi_{\tau}(\bm{r})"
    )
    momentum_space = (
        r"\hat{\mathcal{H}}_0=\sum_{\bm{k}_{\alpha},\bm{k}_{\beta}}\sum_{l_{\alpha},l_{\beta}}\sum_{\tau}"
        r" h^{(\tau)}_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}} "
        r"c^{\dagger}_{\bm{k}_{\alpha},l_{\alpha},\tau}c_{\bm{k}_{\beta},l_{\beta},\tau}"
    )
    hole_basis = (
        r"\hat{\mathcal{H}}_0=\sum_{\tau}\mathrm{Tr}\,h^{(\tau)}-\sum_{\bm{k}_{\alpha},\bm{k}_{\beta}}"
        r"\sum_{l_{\alpha},l_{\beta}}\sum_{\tau}[h^{(\tau)}]^{\intercal}_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}"
        r" b^{\dagger}_{\bm{k}_{\alpha},l_{\alpha},\tau}b_{\bm{k}_{\beta},l_{\beta},\tau}"
    )
    hartree = task_answers["Reduce momentum in Hartree term (momentum in BZ + reciprocal lattice)"]
    fock = task_answers["Reduce momentum in Fock term (momentum in BZ + reciprocal lattice)"]
    combined = (
        r"\hat{\mathcal{H}}^{\mathrm{HF}}=\hat{\mathcal{H}}_1+H_{\mathrm{Hartree}}+H_{\mathrm{Fock}},\quad "
        r"\hat{\mathcal{H}}_1=\sum_{\bm{k}_{\alpha},\bm{k}_{\beta}}\sum_{l_{\alpha},l_{\beta}}\sum_{\tau}"
        r"\tilde{h}^{(\tau)}_{\bm{k}_{\alpha}l_{\alpha},\bm{k}_{\beta}l_{\beta}}"
        r"b^{\dagger}_{\bm{k}_{\alpha},l_{\alpha},\tau}b_{\bm{k}_{\beta},l_{\beta},\tau}"
    )
    return {
        "single_particle_continuum_H_tau": h_tau,
        "delta_b": delta_b,
        "delta_T_tau": delta_t,
        "second_quantized_real_space": second_quantized_real_space,
        "momentum_space_noninteracting": momentum_space,
        "hole_basis_noninteracting": hole_basis,
        "full_interacting_hamiltonian": full_h,
        "hartree_fock_from_supplement": hf_full,
        "reduced_hartree_term": hartree,
        "reduced_fock_term": fock,
        "combined_hartree_fock_hamiltonian": combined,
    }


def build_task_records() -> list[TaskRecord]:
    yaml_path = DATA_DIR / "2111.01152.yaml"
    auto_path = DATA_DIR / "2111.01152_auto.md"
    items = yaml.safe_load(read_text(yaml_path))
    auto_map = parse_auto_md(auto_path)

    records: list[TaskRecord] = []
    step_id = 0
    for item in items:
        if "task" not in item:
            continue
        step_id += 1
        task = item["task"]
        auto_entry = auto_map.get(task, {})
        answer = item.get("answer") or ""
        if not answer and task == "Combine the Hartree and Fock term":
            answer = r"\hat{\mathcal{H}}^{\mathrm{HF}}=\hat{\mathcal{H}}_1+H_{\mathrm{Hartree}}+H_{\mathrm{Fock}}"

        p_mean, p_std = placeholder_stats(item.get("placeholder", {}))
        seq = sequence_ratio(answer, auto_entry.get("completion", ""))
        precision, recall, f1 = token_overlap_metrics(answer, auto_entry.get("completion", ""))
        srec = symbol_recall(answer, auto_entry.get("completion", ""))
        auto_score = automated_step_score(answer, auto_entry.get("completion", ""))

        records.append(
            TaskRecord(
                step_id=step_id,
                task=task,
                source=item.get("source"),
                answer=answer,
                completion=auto_entry.get("completion", ""),
                scores={k: float(v) for k, v in item.get("score", {}).items()},
                placeholder_reviewer_mean=p_mean,
                placeholder_reviewer_std=p_std,
                sequence_ratio=seq,
                token_precision=precision,
                token_recall=recall,
                token_f1=f1,
                symbol_recall=srec,
                automated_step_score=auto_score,
            )
        )
    return records


def records_to_dataframe(records: list[TaskRecord]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {
            "step_id": rec.step_id,
            "task": rec.task,
            "source": json.dumps(rec.source, ensure_ascii=False),
            "reference_answer": rec.answer,
            "model_completion": rec.completion,
            "placeholder_reviewer_mean": rec.placeholder_reviewer_mean,
            "placeholder_reviewer_std": rec.placeholder_reviewer_std,
            "sequence_ratio": rec.sequence_ratio,
            "token_precision": rec.token_precision,
            "token_recall": rec.token_recall,
            "token_f1": rec.token_f1,
            "symbol_recall": rec.symbol_recall,
            "automated_step_score": rec.automated_step_score,
        }
        for metric, value in rec.scores.items():
            row[metric] = value
        if rec.scores:
            row["human_score_mean"] = float(np.mean(list(rec.scores.values())))
            row["human_score_normalized"] = row["human_score_mean"] / 2.0
        else:
            row["human_score_mean"] = np.nan
            row["human_score_normalized"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def save_outputs(metadata: dict[str, Any], derivation: dict[str, str], related_work: list[dict[str, str]], df: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "2111.01152_task_scores.csv", index=False)

    metric_means = {
        col: float(df[col].mean())
        for col in [
            "in_paper",
            "prompt_quality",
            "follow_instructions",
            "physics_logic",
            "math_derivation",
            "final_answer_accuracy",
            "automated_step_score",
            "placeholder_reviewer_mean",
        ]
        if col in df.columns
    }

    summary = {
        "metadata": metadata,
        "related_work": related_work,
        "task_count": int(len(df)),
        "metric_means": metric_means,
        "best_human_scored_steps": df.sort_values("human_score_mean", ascending=False)[["step_id", "task", "human_score_mean"]]
        .head(5)
        .to_dict(orient="records"),
        "hardest_steps_by_accuracy": df.sort_values("final_answer_accuracy")[["step_id", "task", "final_answer_accuracy"]]
        .head(5)
        .to_dict(orient="records"),
        "derivation": derivation,
    }
    (OUTPUT_DIR / "2111.01152_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "2111.01152_derivation.json").write_text(json.dumps(derivation, indent=2, ensure_ascii=False), encoding="utf-8")


def moire_basis(a_m_nm: float) -> tuple[np.ndarray, np.ndarray]:
    g1 = (4 * np.pi / (np.sqrt(3) * a_m_nm)) * np.array([0.0, 1.0])
    g2 = (4 * np.pi / (np.sqrt(3) * a_m_nm)) * np.array([-np.sin(np.pi / 3), np.cos(np.pi / 3)])
    b = np.column_stack([g1, g2])
    a = 2 * np.pi * np.linalg.inv(b).T
    return a[:, 0], a[:, 1]


def evaluate_potentials(metadata: dict[str, Any], grid_n: int = 220) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_m = metadata["moire_period_nm"]
    a1, a2 = moire_basis(a_m)
    u = np.linspace(0, 1, grid_n)
    v = np.linspace(0, 1, grid_n)
    uu, vv = np.meshgrid(u, v)
    r = uu[..., None] * a1 + vv[..., None] * a2

    pref = 4 * np.pi / (np.sqrt(3) * a_m)
    g = {}
    for j in range(1, 7):
        angle = np.pi * (j - 1) / 3
        g[j] = pref * np.array([-np.sin(angle), np.cos(angle)])

    psi_b = np.deg2rad(-14.0)
    v_b = 7.0
    w = 12.0
    omega = np.exp(1j * 2 * np.pi / 3)

    delta_b = 2 * v_b * sum(np.cos(np.tensordot(r, g[j], axes=([2], [0])) + psi_b) for j in (1, 3, 5))
    delta_t_plus = w * (
        1
        + omega ** 1 * np.exp(1j * np.tensordot(r, g[2], axes=([2], [0])))
        + omega ** 2 * np.exp(1j * np.tensordot(r, g[3], axes=([2], [0])))
    )

    x = r[..., 0]
    y = r[..., 1]
    return x, y, delta_b, np.abs(delta_t_plus)


def plot_physics_context(metadata: dict[str, Any]) -> None:
    x, y, delta_b, delta_t_abs = evaluate_potentials(metadata)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    im0 = axes[0].pcolormesh(x, y, delta_b, shading="auto", cmap="coolwarm")
    axes[0].set_title(r"Intralayer potential $\Delta_{\mathfrak{b}}(\mathbf{r})$")
    axes[0].set_xlabel("x (nm)")
    axes[0].set_ylabel("y (nm)")
    fig.colorbar(im0, ax=axes[0], label="meV")

    im1 = axes[1].pcolormesh(x, y, delta_t_abs, shading="auto", cmap="viridis")
    axes[1].set_title(r"Interlayer tunneling $|\Delta_{\mathrm{T},+K}(\mathbf{r})|$")
    axes[1].set_xlabel("x (nm)")
    axes[1].set_ylabel("y (nm)")
    fig.colorbar(im1, ax=axes[1], label="meV")

    fig.suptitle("Moiré real-space fields reconstructed from the paper", fontsize=13)
    fig.savefig(REPORT_IMG_DIR / "moire_fields.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_screened_coulomb() -> None:
    q = np.linspace(1e-3, 8.0, 400)
    d_nm = 5.0
    epsilons = [10, 15, 20]
    e2_mev_nm = 1439.964

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for eps in epsilons:
        vq = 2 * np.pi * e2_mev_nm * np.tanh(q * d_nm) / (eps * q)
        ax.plot(q, vq / vq[0], label=fr"$\epsilon={eps}$")

    ax.set_xlabel(r"$q$ (nm$^{-1}$)")
    ax.set_ylabel(r"$V(q)/V(q\to 0)$")
    ax.set_title("Dual-gate screened Coulomb interaction")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.savefig(REPORT_IMG_DIR / "screened_coulomb.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_score_heatmap(df: pd.DataFrame) -> None:
    metric_cols = [
        "in_paper",
        "prompt_quality",
        "follow_instructions",
        "physics_logic",
        "math_derivation",
        "final_answer_accuracy",
    ]
    heatmap_df = df.set_index("step_id")[metric_cols]
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    sns.heatmap(heatmap_df, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=2, cbar_kws={"label": "Human score"}, ax=ax)
    ax.set_title("Human step scores across the 16-step derivation")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Step ID")
    fig.savefig(REPORT_IMG_DIR / "score_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(df: pd.DataFrame) -> None:
    metric_cols = [
        "in_paper",
        "prompt_quality",
        "follow_instructions",
        "physics_logic",
        "math_derivation",
        "final_answer_accuracy",
    ]
    vals = df[metric_cols].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(x=vals.values, y=vals.index, hue=vals.index, palette="crest", legend=False, ax=ax)
    ax.set_xlim(0, 2)
    ax.set_xlabel("Average score")
    ax.set_ylabel("")
    ax.set_title("Average human rubric scores")
    for idx, val in enumerate(vals.values):
        ax.text(val + 0.03, idx, f"{val:.2f}", va="center")
    fig.savefig(REPORT_IMG_DIR / "metric_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_automated_validation(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    plot_df = df.dropna(subset=["automated_step_score", "final_answer_accuracy"])
    sns.regplot(
        data=plot_df,
        x="automated_step_score",
        y="final_answer_accuracy",
        scatter_kws={"s": 55, "alpha": 0.8},
        line_kws={"color": "black", "lw": 1.2},
        ax=ax,
    )
    corr = plot_df["automated_step_score"].corr(plot_df["final_answer_accuracy"])
    ax.set_xlabel("Automated similarity-based step score")
    ax.set_ylabel("Human final-answer accuracy")
    ax.set_title(f"Automated scoring vs. human final-answer accuracy (r={corr:.2f})")
    ax.set_xlim(0, 2.05)
    ax.set_ylim(-0.05, 2.05)
    fig.savefig(REPORT_IMG_DIR / "automated_vs_human.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_placeholder_vs_accuracy(df: pd.DataFrame) -> None:
    plot_df = df.dropna(subset=["placeholder_reviewer_mean"])
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(plot_df["step_id"], plot_df["placeholder_reviewer_mean"], marker="o", label="Placeholder reviewer mean")
    ax.plot(plot_df["step_id"], plot_df["final_answer_accuracy"], marker="s", label="Final answer accuracy")
    ax.set_xlabel("Step ID")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 2.05)
    ax.set_title("Prompt-level placeholder agreement vs. final answer accuracy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(REPORT_IMG_DIR / "placeholder_vs_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid", context="talk")

    main_tex = read_text(DATA_DIR / "2111.01152.tex")
    sm_tex = read_text(DATA_DIR / "2111.01152_SM.tex")
    metadata = extract_metadata(main_tex)

    records = build_task_records()
    df = records_to_dataframe(records)
    task_answers = {rec.task: rec.answer for rec in records}
    derivation = build_derivation(main_tex, sm_tex, task_answers)
    related_work = parse_related_work_titles()

    save_outputs(metadata, derivation, related_work, df)
    plot_physics_context(metadata)
    plot_screened_coulomb()
    plot_score_heatmap(df)
    plot_metric_summary(df)
    plot_automated_validation(df)
    plot_placeholder_vs_accuracy(df)

    console_summary = {
        "paper_id": metadata["paper_id"],
        "title": metadata["title"],
        "task_count": int(len(df)),
        "human_final_answer_accuracy_mean": round(float(df["final_answer_accuracy"].mean()), 3),
        "automated_step_score_mean": round(float(df["automated_step_score"].dropna().mean()), 3),
        "score_correlation": round(float(df.dropna(subset=["automated_step_score"])["automated_step_score"].corr(df.dropna(subset=["automated_step_score"])["final_answer_accuracy"])), 3),
    }
    print(json.dumps(console_summary, indent=2))


if __name__ == "__main__":
    main()
