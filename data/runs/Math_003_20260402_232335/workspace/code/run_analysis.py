#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "report" / "images"

IMO_PATH = DATA_DIR / "imo_ag_30.txt"
DEFS_PATH = DATA_DIR / "defs.txt"
RULES_PATH = DATA_DIR / "rules.txt"


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def clean_tokens(text: str) -> list[str]:
    return [tok for tok in re.split(r"\s+", text.strip()) if tok]


def parse_problem_blocks(path: Path) -> list[dict]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected alternating name/spec lines in {path}, got odd line count {len(lines)}")

    problems = []
    for i in range(0, len(lines), 2):
        pid = lines[i]
        spec = lines[i + 1]
        if " ? " not in spec:
            raise ValueError(f"Problem {pid} missing '?' separator")
        premises_text, goal_text = spec.split(" ? ", 1)
        premise_clauses = [cl.strip() for cl in premises_text.split("; ") if cl.strip()]
        constructions = []
        primitive_counter = Counter()
        primitive_arguments = []
        assignment_lhs_sizes = []
        clause_type_counter = Counter()
        coordinates_present = 0

        for clause in premise_clauses:
            lhs, rhs = clause.split(" = ", 1)
            lhs_tokens = clean_tokens(lhs)
            assignment_lhs_sizes.append(len(lhs_tokens))
            rhs_parts = [part.strip() for part in rhs.split(", ") if part.strip()]
            clause_primitives = []
            for part in rhs_parts:
                part_tokens = clean_tokens(part)
                if not part_tokens:
                    continue
                primitive = part_tokens[0]
                args = part_tokens[1:]
                primitive_counter[primitive] += 1
                primitive_arguments.append({
                    "primitive": primitive,
                    "arity": len(args),
                    "args": args,
                })
                clause_primitives.append(primitive)
                if any("@" in tok for tok in lhs_tokens + args):
                    coordinates_present = 1
            clause_type_counter["multi_constraint_clause" if len(rhs_parts) > 1 else "single_constraint_clause"] += 1
            constructions.append({
                "lhs": lhs_tokens,
                "rhs_parts": rhs_parts,
                "primitives": clause_primitives,
            })

        goal_tokens = clean_tokens(goal_text)
        goal_predicate = goal_tokens[0] if goal_tokens else "UNKNOWN"

        problems.append({
            "id": pid,
            "spec": spec,
            "premises_text": premises_text,
            "goal_text": goal_text,
            "num_premise_clauses": len(premise_clauses),
            "num_constraint_atoms": sum(len(c["rhs_parts"]) for c in constructions),
            "primitive_counter": dict(primitive_counter),
            "num_unique_primitives": len(primitive_counter),
            "primitive_arguments": primitive_arguments,
            "assignment_lhs_sizes": assignment_lhs_sizes,
            "clause_type_counter": dict(clause_type_counter),
            "coordinates_present": coordinates_present,
            "goal_predicate": goal_predicate,
            "goal_arity": max(0, len(goal_tokens) - 1),
            "goal_tokens": goal_tokens,
            "constructions": constructions,
        })
    return problems


def parse_defs(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = []
    current = []
    for line in lines:
        if line.strip():
            current.append(line.rstrip())
        else:
            if current:
                blocks.append(current)
                current = []
    if current:
        blocks.append(current)

    defs = []
    for block in blocks:
        header = block[0].strip()
        header_tokens = clean_tokens(header)
        if not header_tokens:
            continue
        name = header_tokens[0]
        params = header_tokens[1:]
        defs.append({
            "name": name,
            "params": params,
            "num_params": len(params),
            "raw_block": block,
            "num_lines": len(block),
        })
    return defs


def parse_rules(path: Path) -> list[dict]:
    rules = []
    for idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        if "=>" not in line:
            continue
        lhs, rhs = [part.strip() for part in line.split("=>", 1)]
        antecedents = [item.strip() for item in lhs.split(", ") if item.strip()]
        consequent_tokens = clean_tokens(rhs)
        consequent_pred = consequent_tokens[0] if consequent_tokens else "UNKNOWN"
        antecedent_preds = [clean_tokens(item)[0] for item in antecedents if clean_tokens(item)]
        rules.append({
            "index": idx,
            "text": line,
            "num_antecedents": len(antecedents),
            "antecedent_predicates": antecedent_preds,
            "consequent_predicate": consequent_pred,
        })
    return rules


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_bar(counter: Counter, title: str, xlabel: str, path: Path, top_n: int | None = None, color: str = "#4C78A8") -> None:
    items = counter.most_common(top_n)
    if not items:
        return
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    plt.figure(figsize=(10, max(4, 0.38 * len(labels) + 1.5)))
    plt.barh(labels[::-1], values[::-1], color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_hist(values: list[int], bins: int, title: str, xlabel: str, path: Path, color: str = "#F58518") -> None:
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, color=color, edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_scatter(x: list[int], y: list[int], labels: list[str], title: str, xlabel: str, ylabel: str, path: Path) -> None:
    if not x or not y:
        return
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color="#54A24B", alpha=0.85)
    for xi, yi, label in zip(x, y, labels):
        plt.annotate(label.replace("translated_", ""), (xi, yi), fontsize=7, alpha=0.75, xytext=(4, 4), textcoords="offset points")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def make_summary(problems: list[dict], defs: list[dict], rules: list[dict]) -> dict:
    primitive_counter = Counter()
    goal_counter = Counter()
    lhs_size_counter = Counter()
    primitive_arity_counter = Counter()
    problem_year_counter = Counter()
    clause_shape_counter = Counter()
    primitive_goal_overlap = Counter()

    premise_clause_counts = []
    constraint_atom_counts = []
    unique_primitive_counts = []
    goal_arities = []

    for problem in problems:
        primitive_counter.update(problem["primitive_counter"])
        goal_counter[problem["goal_predicate"]] += 1
        lhs_size_counter.update(problem["assignment_lhs_sizes"])
        clause_shape_counter.update(problem["clause_type_counter"])
        premise_clause_counts.append(problem["num_premise_clauses"])
        constraint_atom_counts.append(problem["num_constraint_atoms"])
        unique_primitive_counts.append(problem["num_unique_primitives"])
        goal_arities.append(problem["goal_arity"])
        m = re.search(r"imo_(\d{4})_", problem["id"])
        if m:
            problem_year_counter[m.group(1)] += 1
        for pa in problem["primitive_arguments"]:
            primitive_arity_counter[(pa["primitive"], pa["arity"])] += 1
        if problem["goal_predicate"] in problem["primitive_counter"]:
            primitive_goal_overlap[problem["goal_predicate"]] += 1

    def_counter = Counter(d["name"] for d in defs)
    def_param_hist = Counter(d["num_params"] for d in defs)

    antecedent_pred_counter = Counter()
    consequent_pred_counter = Counter()
    antecedent_count_hist = Counter()
    for rule in rules:
        antecedent_pred_counter.update(rule["antecedent_predicates"])
        consequent_pred_counter[rule["consequent_predicate"]] += 1
        antecedent_count_hist[rule["num_antecedents"]] += 1

    summary = {
        "num_problems": len(problems),
        "num_definitions": len(defs),
        "num_rules": len(rules),
        "goal_predicate_counts": dict(goal_counter),
        "construction_primitive_counts": dict(primitive_counter),
        "assignment_lhs_size_counts": dict(lhs_size_counter),
        "definition_param_histogram": dict(def_param_hist),
        "rule_antecedent_count_histogram": dict(antecedent_count_hist),
        "rule_antecedent_predicate_counts": dict(antecedent_pred_counter),
        "rule_consequent_predicate_counts": dict(consequent_pred_counter),
        "problem_year_counts": dict(problem_year_counter),
        "clause_shape_counts": dict(clause_shape_counter),
        "primitive_goal_overlap_counts": dict(primitive_goal_overlap),
        "dataset_statistics": {
            "premise_clauses": {
                "min": min(premise_clause_counts),
                "max": max(premise_clause_counts),
                "mean": mean(premise_clause_counts),
                "median": median(premise_clause_counts),
            },
            "constraint_atoms": {
                "min": min(constraint_atom_counts),
                "max": max(constraint_atom_counts),
                "mean": mean(constraint_atom_counts),
                "median": median(constraint_atom_counts),
            },
            "unique_primitives": {
                "min": min(unique_primitive_counts),
                "max": max(unique_primitive_counts),
                "mean": mean(unique_primitive_counts),
                "median": median(unique_primitive_counts),
            },
            "goal_arity": {
                "min": min(goal_arities),
                "max": max(goal_arities),
                "mean": mean(goal_arities),
                "median": median(goal_arities),
            },
        },
        "top_primitive_arity_pairs": [
            {"primitive": primitive, "arity": arity, "count": count}
            for (primitive, arity), count in primitive_arity_counter.most_common(20)
        ],
        "definition_names": sorted(def_counter),
    }
    return summary


def build_problem_table(problems: list[dict]) -> list[dict]:
    rows = []
    for problem in problems:
        rows.append({
            "id": problem["id"],
            "num_premise_clauses": problem["num_premise_clauses"],
            "num_constraint_atoms": problem["num_constraint_atoms"],
            "num_unique_primitives": problem["num_unique_primitives"],
            "goal_predicate": problem["goal_predicate"],
            "goal_arity": problem["goal_arity"],
            "coordinates_present": problem["coordinates_present"],
            "most_common_primitives": ";".join(f"{k}:{v}" for k, v in Counter(problem["primitive_counter"]).most_common(5)),
        })
    return rows


def build_def_table(defs: list[dict]) -> list[dict]:
    return [
        {
            "name": d["name"],
            "num_params": d["num_params"],
            "num_lines": d["num_lines"],
        }
        for d in defs
    ]


def build_rule_table(rules: list[dict]) -> list[dict]:
    return [
        {
            "index": r["index"],
            "num_antecedents": r["num_antecedents"],
            "consequent_predicate": r["consequent_predicate"],
            "antecedent_predicates": ";".join(r["antecedent_predicates"]),
            "text": r["text"],
        }
        for r in rules
    ]


def write_markdown_summary(summary: dict, problems: list[dict], defs: list[dict], rules: list[dict]) -> None:
    top_goals = Counter(summary["goal_predicate_counts"]).most_common()
    top_primitives = Counter(summary["construction_primitive_counts"]).most_common(15)
    top_rule_consequents = Counter(summary["rule_consequent_predicate_counts"]).most_common(12)

    hardest = sorted(problems, key=lambda p: (p["num_constraint_atoms"], p["num_premise_clauses"]), reverse=True)[:5]
    easiest = sorted(problems, key=lambda p: (p["num_constraint_atoms"], p["num_premise_clauses"]))[:5]

    md = []
    md.append("# Geometry Benchmark Analysis Summary")
    md.append("")
    md.append(f"- Problems analyzed: **{summary['num_problems']}**")
    md.append(f"- Construction/definition templates: **{summary['num_definitions']}**")
    md.append(f"- Inference rules parsed: **{summary['num_rules']}**")
    md.append("")
    md.append("## Dataset complexity")
    stats = summary["dataset_statistics"]
    for key, label in [
        ("premise_clauses", "Premise clauses per problem"),
        ("constraint_atoms", "Constraint atoms per problem"),
        ("unique_primitives", "Unique primitives per problem"),
        ("goal_arity", "Goal arity"),
    ]:
        s = stats[key]
        md.append(f"- {label}: min={s['min']}, median={s['median']}, mean={s['mean']:.2f}, max={s['max']}")
    md.append("")
    md.append("## Goal predicates")
    for k, v in top_goals:
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Most frequent construction primitives")
    for k, v in top_primitives:
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Most frequent rule consequents")
    for k, v in top_rule_consequents:
        md.append(f"- {k}: {v}")
    md.append("")
    md.append("## Most structurally complex benchmark items")
    for p in hardest:
        md.append(f"- {p['id']}: {p['num_premise_clauses']} clauses, {p['num_constraint_atoms']} atoms, goal={p['goal_predicate']}")
    md.append("")
    md.append("## Least structurally complex benchmark items")
    for p in easiest:
        md.append(f"- {p['id']}: {p['num_premise_clauses']} clauses, {p['num_constraint_atoms']} atoms, goal={p['goal_predicate']}")
    md.append("")
    md.append("## Interpretation")
    md.append("The benchmark combines a compact set of high-level geometric construction primitives with a rule base that rewrites between incidence, angle, ratio, congruence, cyclicity, and parallel/perpendicular relations. This structure is suitable for a neuro-symbolic theorem prover that first predicts useful intermediate predicates or construction expansions, then validates them through symbolic rule chaining. The analysis artifacts in outputs/ and report/images/ are intended to support that later modeling/reporting stage.")
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def generate_figures(problems: list[dict], defs: list[dict], rules: list[dict], summary: dict) -> None:
    plot_bar(
        Counter(summary["goal_predicate_counts"]),
        title="Goal predicate distribution across IMO geometry benchmark",
        xlabel="Number of problems",
        path=FIG_DIR / "goal_predicate_distribution.png",
        color="#4C78A8",
    )
    plot_bar(
        Counter(summary["construction_primitive_counts"]),
        title="Most frequent construction primitives in problem statements",
        xlabel="Number of occurrences",
        path=FIG_DIR / "construction_primitive_frequency.png",
        top_n=20,
        color="#72B7B2",
    )
    plot_hist(
        [p["num_constraint_atoms"] for p in problems],
        bins=min(10, len(problems)),
        title="Distribution of structural problem size",
        xlabel="Constraint atoms per problem",
        path=FIG_DIR / "constraint_atom_histogram.png",
    )
    plot_scatter(
        [p["num_premise_clauses"] for p in problems],
        [p["num_constraint_atoms"] for p in problems],
        [p["id"] for p in problems],
        title="Problem complexity: clauses vs. constraint atoms",
        xlabel="Premise clauses",
        ylabel="Constraint atoms",
        path=FIG_DIR / "problem_complexity_scatter.png",
    )
    plot_bar(
        Counter(summary["rule_consequent_predicate_counts"]),
        title="Rule-base consequent predicate distribution",
        xlabel="Number of rules",
        path=FIG_DIR / "rule_consequent_distribution.png",
        top_n=20,
        color="#E45756",
    )

    # Extra figure: definitions by parameter count.
    param_hist = Counter(d["num_params"] for d in defs)
    items = sorted(param_hist.items())
    plt.figure(figsize=(8, 5))
    plt.bar([str(k) for k, _ in items], [v for _, v in items], color="#B279A2")
    plt.title("Definition template arity distribution")
    plt.xlabel("Number of parameters")
    plt.ylabel("Count of definition templates")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "definition_arity_distribution.png", dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    problems = parse_problem_blocks(IMO_PATH)
    defs = parse_defs(DEFS_PATH)
    rules = parse_rules(RULES_PATH)

    summary = make_summary(problems, defs, rules)

    write_json(OUTPUT_DIR / "parsed_problems.json", problems)
    write_json(OUTPUT_DIR / "definitions_catalog.json", defs)
    write_json(OUTPUT_DIR / "rules_catalog.json", rules)
    write_json(OUTPUT_DIR / "analysis_summary.json", summary)

    write_csv(
        OUTPUT_DIR / "problem_level_statistics.csv",
        build_problem_table(problems),
        fieldnames=[
            "id",
            "num_premise_clauses",
            "num_constraint_atoms",
            "num_unique_primitives",
            "goal_predicate",
            "goal_arity",
            "coordinates_present",
            "most_common_primitives",
        ],
    )
    write_csv(
        OUTPUT_DIR / "definitions_catalog.csv",
        build_def_table(defs),
        fieldnames=["name", "num_params", "num_lines"],
    )
    write_csv(
        OUTPUT_DIR / "rules_catalog.csv",
        build_rule_table(rules),
        fieldnames=["index", "num_antecedents", "consequent_predicate", "antecedent_predicates", "text"],
    )

    write_markdown_summary(summary, problems, defs, rules)
    generate_figures(problems, defs, rules, summary)

    print("Analysis complete.")
    print(f"Problems: {len(problems)} | Definitions: {len(defs)} | Rules: {len(rules)}")
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"Figures written to: {FIG_DIR}")


if __name__ == "__main__":
    main()
