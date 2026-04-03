#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run this analysis script.") from exc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "2111.01152"
RELATED_DIR = ROOT / "related_work"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"

YAML_PATH = DATA_DIR / "2111.01152.yaml"
MAIN_TEX = DATA_DIR / "2111.01152.tex"
SM_TEX = DATA_DIR / "2111.01152_SM.tex"
INSTRUCTIONS_PATH = ROOT / "INSTRUCTIONS.md"
UNDERSTANDING_PATH = OUTPUT_DIR / "task_understanding.txt"

SCORE_KEYS = [
    "in_paper",
    "prompt_quality",
    "follow_instructions",
    "physics_logic",
    "math_derivation",
    "final_answer_accuracy",
]
REVIEWERS = ["Haining", "Will", "Yasaman"]


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_yaml(path: Path):
    return yaml.safe_load(read_text(path))


def tex_lines(path: Path) -> list[str]:
    return read_text(path).splitlines()


def extract_line_range(lines: list[str], start: int, end: int) -> str:
    start_i = max(start - 1, 0)
    end_i = min(end, len(lines))
    selected = lines[start_i:end_i]
    return "\n".join(selected).strip()


def parse_source_snippets(task: dict, main_lines: list[str], sm_lines: list[str]) -> list[dict]:
    snippets = []
    source = task.get("source", {}) or {}
    for source_name, ranges in source.items():
        if not isinstance(ranges, list):
            continue
        if source_name.endswith("_SM.tex"):
            source_lines = sm_lines
        elif source_name.endswith(".tex"):
            source_lines = main_lines
        else:
            source_lines = []
        for item in ranges:
            if not isinstance(item, list) or len(item) != 2:
                continue
            start, end = int(item[0]), int(item[1])
            snippets.append(
                {
                    "source_file": source_name,
                    "line_start": start,
                    "line_end": end,
                    "text": extract_line_range(source_lines, start, end),
                }
            )
    return snippets


def classify_task(task_name: str) -> str:
    lowered = task_name.lower()
    if "kinetic" in lowered:
        return "kinetic"
    if "potential" in lowered:
        return "potential"
    if "second-quantized" in lowered or "second quantized" in lowered:
        return "second_quantization"
    if "momentum space" in lowered or "fourier" in lowered:
        return "momentum_conversion"
    if "particle-hole" in lowered or "particle hole" in lowered:
        return "particle_hole"
    if "interaction" in lowered:
        return "interaction"
    if "wick" in lowered or "hartree-fock" in lowered or "hartree and fock" in lowered:
        return "hartree_fock"
    if "extract quadratic" in lowered or "reduce momentum" in lowered or "swap the index" in lowered:
        return "simplification"
    return "other"


def task_records(tasks: list[dict], main_lines: list[str], sm_lines: list[str]) -> tuple[list[dict], list[dict], dict]:
    records = []
    placeholder_rows = []
    source_snippets = {}

    for idx, task in enumerate(tasks, start=1):
        task_name = task.get("task", f"task_{idx}")
        task_score = task.get("score", {}) or {}
        total_score = sum(float(task_score.get(k, 0) or 0) for k in SCORE_KEYS)
        max_total = 2 * len(SCORE_KEYS)
        category = classify_task(task_name)
        snippets = parse_source_snippets(task, main_lines, sm_lines)
        source_snippets[task_name] = snippets

        records.append(
            {
                "index": idx,
                "task": task_name,
                "category": category,
                "branch": task.get("branch", ""),
                "answer": task.get("answer", ""),
                "answer_length": len(str(task.get("answer", ""))),
                "source_count": len(snippets),
                **{k: float(task_score.get(k, 0) or 0) for k in SCORE_KEYS},
                "total_score": total_score,
                "max_total": max_total,
                "score_fraction": (total_score / max_total) if max_total else 0.0,
            }
        )

        placeholder = task.get("placeholder", {}) or {}
        for field_name, field_info in placeholder.items():
            if not isinstance(field_info, dict):
                continue
            llm_val = field_info.get("LLM")
            human_val = field_info.get("human")
            score_info = field_info.get("score", {}) or {}
            reviewer_values = {r: score_info.get(r) for r in REVIEWERS if score_info.get(r) not in (None, "(?)")}
            numeric_vals = [float(v) for v in reviewer_values.values() if isinstance(v, (int, float))]
            placeholder_rows.append(
                {
                    "task": task_name,
                    "field": field_name,
                    "llm": "" if llm_val is None else str(llm_val),
                    "human": "" if human_val is None else str(human_val),
                    "same_nonempty": int(bool(llm_val) and bool(human_val) and str(llm_val).strip() == str(human_val).strip()),
                    "llm_missing": int(llm_val in (None, "")),
                    "human_missing": int(human_val in (None, "")),
                    "reviewer_mean": mean(numeric_vals) if numeric_vals else None,
                    "reviewer_count": len(numeric_vals),
                    **{f"review_{r}": score_info.get(r) for r in REVIEWERS},
                }
            )

    return records, placeholder_rows, source_snippets


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_scores(records: list[dict]) -> dict:
    total_tasks = len(records)
    avg_total = mean(r["total_score"] for r in records) if records else 0.0
    avg_fraction = mean(r["score_fraction"] for r in records) if records else 0.0
    by_category = defaultdict(list)
    for r in records:
        by_category[r["category"]].append(r)
    category_summary = {
        cat: {
            "task_count": len(items),
            "avg_total_score": mean(x["total_score"] for x in items),
            "avg_score_fraction": mean(x["score_fraction"] for x in items),
        }
        for cat, items in by_category.items()
    }
    return {
        "task_count": total_tasks,
        "average_total_score": avg_total,
        "average_score_fraction": avg_fraction,
        "category_summary": category_summary,
        "best_task": max(records, key=lambda r: r["score_fraction"]) if records else None,
        "lowest_task": min(records, key=lambda r: r["score_fraction"]) if records else None,
    }


def build_paper_context(main_text: str) -> dict:
    title_match = re.search(r"\\title\{([^}]*)\}", main_text, flags=re.S)
    title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main_text, flags=re.S)
    abstract = re.sub(r"\s+", " ", abstract_match.group(1)).strip() if abstract_match else ""

    model_match = re.search(r"\\begin\{equation\}\\label\{eq:Ham\}(.*?)\\end\{equation\}", main_text, flags=re.S)
    model_equation = model_match.group(1).strip() if model_match else ""

    interaction_sentence = ""
    for line in main_text.splitlines():
        if "V(q)=2\\pi e^2 \\tanh(q d)/(\\epsilon q)" in line:
            interaction_sentence = line.strip()
            break

    return {
        "title": title,
        "abstract": abstract,
        "core_hamiltonian_equation": model_equation,
        "interaction_sentence": interaction_sentence,
    }


def write_text_summary(path: Path, context: dict, score_summary: dict, placeholder_rows: list[dict], related_inventory: list[dict]) -> None:
    mismatches = [r for r in placeholder_rows if not r["same_nonempty"] and not r["llm_missing"]]
    lines = [
        f"Title: {context.get('title', '')}",
        "",
        "Abstract summary:",
        context.get("abstract", ""),
        "",
        "Core continuum Hamiltonian equation:",
        context.get("core_hamiltonian_equation", ""),
        "",
        "Interaction model cue:",
        context.get("interaction_sentence", ""),
        "",
        f"Tasks analyzed: {score_summary['task_count']}",
        f"Average total task score: {score_summary['average_total_score']:.2f} / 12.00",
        f"Average normalized score: {100*score_summary['average_score_fraction']:.1f}%",
        f"Best task: {score_summary['best_task']['task'] if score_summary['best_task'] else ''}",
        f"Lowest-scoring task: {score_summary['lowest_task']['task'] if score_summary['lowest_task'] else ''}",
        "",
        "Top placeholder-mismatch count (non-identical filled LLM vs human fields):",
        str(len(mismatches)),
        "",
        "Related work inventory:",
    ]
    for item in related_inventory:
        lines.append(f"- {item['file_name']} ({item['size_kb']:.1f} KB)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def related_work_inventory() -> list[dict]:
    rows = []
    for path in sorted(RELATED_DIR.glob("*.pdf")):
        rows.append(
            {
                "file_name": path.name,
                "size_bytes": path.stat().st_size,
                "size_kb": path.stat().st_size / 1024.0,
            }
        )
    return rows


def make_figures(records: list[dict], placeholder_rows: list[dict]) -> list[str]:
    created = []
    if not records:
        return created

    # Figure 1: task total scores
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [f"{r['index']}. {r['task'][:36]}" for r in records]
    totals = [r["total_score"] for r in records]
    ax.bar(range(len(records)), totals, color="#4c78a8")
    ax.set_xticks(range(len(records)))
    ax.set_xticklabels(names, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Total score")
    ax.set_title("Task-level Hartree-Fock benchmark scores")
    ax.set_ylim(0, max(12, max(totals) + 1))
    fig.tight_layout()
    out = IMAGE_DIR / "task_total_scores.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    created.append(str(out.relative_to(ROOT)))

    # Figure 2: average score by evaluation dimension
    fig, ax = plt.subplots(figsize=(8, 5))
    dim_means = [mean(r[k] for r in records) for k in SCORE_KEYS]
    ax.bar(SCORE_KEYS, dim_means, color="#f58518")
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("Average score")
    ax.set_title("Average score by evaluation dimension")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out = IMAGE_DIR / "score_dimension_averages.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    created.append(str(out.relative_to(ROOT)))

    # Figure 3: reviewer placeholder averages
    reviewer_means = []
    labels = []
    for reviewer in REVIEWERS:
        vals = []
        key = f"review_{reviewer}"
        for row in placeholder_rows:
            val = row.get(key)
            if isinstance(val, (int, float)):
                vals.append(float(val))
        labels.append(reviewer)
        reviewer_means.append(mean(vals) if vals else 0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, reviewer_means, color=["#54a24b", "#e45756", "#72b7b2"])
    ax.set_ylim(0, 2.1)
    ax.set_ylabel("Average placeholder score")
    ax.set_title("Reviewer placeholder-score averages")
    fig.tight_layout()
    out = IMAGE_DIR / "reviewer_placeholder_averages.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    created.append(str(out.relative_to(ROOT)))

    return created


def build_analysis_summary(
    context: dict,
    records: list[dict],
    placeholder_rows: list[dict],
    source_snippets: dict,
    related_inventory: list[dict],
    figure_paths: list[str],
) -> dict:
    score_summary = summarize_scores(records)
    mismatch_counter = Counter(r["field"] for r in placeholder_rows if not r["same_nonempty"] and not r["llm_missing"])
    category_counter = Counter(r["category"] for r in records)
    return {
        "paper_context": context,
        "score_summary": score_summary,
        "task_categories": dict(category_counter),
        "placeholder_mismatch_top_fields": mismatch_counter.most_common(15),
        "output_figures": figure_paths,
        "related_work_inventory": related_inventory,
        "source_snippet_task_count": len(source_snippets),
    }


def main() -> None:
    ensure_dirs()

    data = load_yaml(YAML_PATH)
    if not isinstance(data, list):
        raise SystemExit("Expected YAML root to be a list of task records.")

    main_lines = tex_lines(MAIN_TEX)
    sm_lines = tex_lines(SM_TEX) if SM_TEX.exists() else []
    main_text = "\n".join(main_lines)

    records, placeholder_rows, source_snippets = task_records(data, main_lines, sm_lines)
    context = build_paper_context(main_text)
    related_inventory = related_work_inventory()
    figure_paths = make_figures(records, placeholder_rows)
    score_summary = summarize_scores(records)

    write_csv(OUTPUT_DIR / "task_score_table.csv", records)
    write_csv(OUTPUT_DIR / "placeholder_mismatch_table.csv", placeholder_rows)
    (OUTPUT_DIR / "source_snippets.json").write_text(json.dumps(source_snippets, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_DIR / "related_work_inventory.json").write_text(json.dumps(related_inventory, indent=2), encoding="utf-8")

    summary = build_analysis_summary(context, records, placeholder_rows, source_snippets, related_inventory, figure_paths)
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_text_summary(OUTPUT_DIR / "paper_context_summary.txt", context, score_summary, placeholder_rows, related_inventory)

    provenance = {
        "inputs": {
            "instructions": str(INSTRUCTIONS_PATH.relative_to(ROOT)),
            "task_understanding": str(UNDERSTANDING_PATH.relative_to(ROOT)),
            "yaml": str(YAML_PATH.relative_to(ROOT)),
            "main_tex": str(MAIN_TEX.relative_to(ROOT)),
            "supplement_tex": str(SM_TEX.relative_to(ROOT)),
            "related_work_dir": str(RELATED_DIR.relative_to(ROOT)),
        },
        "outputs": {
            "analysis_summary": "outputs/analysis_summary.json",
            "task_scores": "outputs/task_score_table.csv",
            "placeholder_mismatches": "outputs/placeholder_mismatch_table.csv",
            "source_snippets": "outputs/source_snippets.json",
            "paper_context_summary": "outputs/paper_context_summary.txt",
            "related_work_inventory": "outputs/related_work_inventory.json",
            "figures": figure_paths,
        },
    }
    (OUTPUT_DIR / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print("Analysis completed.")
    print(json.dumps({"tasks": len(records), "figures": figure_paths}, indent=2))


if __name__ == "__main__":
    main()
