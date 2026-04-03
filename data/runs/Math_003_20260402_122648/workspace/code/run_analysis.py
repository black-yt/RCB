from __future__ import annotations

import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Atom:
    pred: str
    args: Tuple[str, ...]

    def to_dict(self) -> dict:
        return {"pred": self.pred, "args": list(self.args)}

    def __str__(self) -> str:
        return f"{self.pred} {' '.join(self.args)}"


@dataclass
class ProofStep:
    conclusion: Atom
    source: str
    premises: List[Atom]
    detail: str

    def to_dict(self) -> dict:
        return {
            "conclusion": self.conclusion.to_dict(),
            "source": self.source,
            "premises": [atom.to_dict() for atom in self.premises],
            "detail": self.detail,
        }


@dataclass
class Definition:
    name: str
    params: List[str]
    facts: List[Atom]


@dataclass
class Rule:
    index: int
    premises: List[Atom]
    conclusion: Atom
    text: str


@dataclass
class Problem:
    name: str
    constructions: List[Tuple[str, List[str], List[str]]]
    goal: Atom


def parse_atom_text(text: str) -> List[Atom]:
    atoms: List[Atom] = []
    for segment in text.split(";"):
        segment = segment.strip()
        if not segment:
            continue
        if ":" in segment:
            _, segment = segment.split(":", 1)
        for piece in segment.split(","):
            piece = piece.strip()
            if not piece or piece == "=":
                continue
            parts = piece.split()
            if not parts:
                continue
            pred, args = parts[0], tuple(parts[1:])
            atoms.append(Atom(pred, args))
    return atoms


def canonical_segment(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def canonicalize(atom: Atom) -> Atom:
    p, a = atom.pred, atom.args
    if p in {"coll", "cyclic", "ncoll"}:
        return Atom(p, tuple(sorted(a)))
    if p in {"diff"} and len(a) == 2:
        return Atom(p, canonical_segment(a[0], a[1]))
    if p in {"cong"} and len(a) == 4:
        s1 = canonical_segment(a[0], a[1])
        s2 = canonical_segment(a[2], a[3])
        return Atom(p, tuple(min((s1, s2)) + max((s1, s2))))
    if p in {"para", "perp", "npara", "nperp"} and len(a) == 4:
        s1 = canonical_segment(a[0], a[1])
        s2 = canonical_segment(a[2], a[3])
        return Atom(p, tuple(min((s1, s2)) + max((s1, s2))))
    if p in {"eqratio"} and len(a) == 8:
        g1 = tuple(sorted((canonical_segment(a[0], a[1]), canonical_segment(a[2], a[3]))))
        g2 = tuple(sorted((canonical_segment(a[4], a[5]), canonical_segment(a[6], a[7]))))
        groups = sorted((g1, g2))
        return Atom(p, tuple(groups[0][0] + groups[0][1] + groups[1][0] + groups[1][1]))
    if p in {"eqangle", "eqangle6", "eqangle2", "eqangle3"} and len(a) in {4, 8}:
        if len(a) == 4:
            g1 = canonical_segment(a[0], a[1])
            g2 = canonical_segment(a[2], a[3])
            return Atom(p, tuple(g1 + g2))
        g1 = tuple(sorted((canonical_segment(a[0], a[1]), canonical_segment(a[2], a[3]))))
        g2 = tuple(sorted((canonical_segment(a[4], a[5]), canonical_segment(a[6], a[7]))))
        groups = sorted((g1, g2))
        return Atom(p, tuple(groups[0][0] + groups[0][1] + groups[1][0] + groups[1][1]))
    if p == "midp" and len(a) == 3:
        return Atom(p, (a[0],) + canonical_segment(a[1], a[2]))
    return atom


def substitute(atom: Atom, mapping: Dict[str, str]) -> Atom:
    return canonicalize(Atom(atom.pred, tuple(mapping.get(token, token) for token in atom.args)))


def load_definitions(path: Path) -> Dict[str, Definition]:
    text = path.read_text().strip()
    blocks = [block.strip() for block in text.split("\n\n") if block.strip()]
    definitions: Dict[str, Definition] = {}
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        header = lines[0].split()
        name, params = header[0], header[1:]
        atoms: List[Atom] = []
        if len(lines) >= 3:
            condition_line = lines[2]
            if "=" in condition_line:
                _, rhs = condition_line.split("=", 1)
                atoms.extend(parse_atom_text(rhs))
        if len(lines) >= 4:
            atoms.extend(parse_atom_text(lines[3]))
        definitions[name] = Definition(name=name, params=params, facts=[canonicalize(atom) for atom in atoms])
    return definitions


def load_rules(path: Path) -> List[Rule]:
    rules: List[Rule] = []
    for idx, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        body_text, conclusion_text = [part.strip() for part in line.split("=>")]
        premises = [canonicalize(atom) for atom in parse_atom_text(body_text)]
        conclusion = canonicalize(parse_atom_text(conclusion_text)[0])
        rules.append(Rule(index=idx, premises=premises, conclusion=conclusion, text=line))
    return rules


def load_problems(path: Path) -> List[Problem]:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    problems: List[Problem] = []
    for idx in range(0, len(lines), 2):
        name = lines[idx]
        construction_text, goal_text = [part.strip() for part in lines[idx + 1].split("?")]
        constructions: List[Tuple[str, List[str], List[str]]] = []
        for chunk in construction_text.split(";"):
            chunk = chunk.strip()
            if not chunk or "=" not in chunk:
                continue
            lhs, rhs = chunk.split("=", 1)
            lhs_vars = lhs.strip().split()
            for call in rhs.split(","):
                call = call.strip()
                if not call:
                    continue
                parts = call.split()
                constructions.append((parts[0], lhs_vars, parts[1:]))
        goal = canonicalize(parse_atom_text(goal_text)[0])
        problems.append(Problem(name=name, constructions=constructions, goal=goal))
    return problems


def instantiate_problem(problem: Problem, definitions: Dict[str, Definition]) -> Dict[Atom, ProofStep]:
    facts: Dict[Atom, ProofStep] = {}
    for op, lhs_vars, rhs_args in problem.constructions:
        definition = definitions.get(op)
        if not definition:
            continue
        if len(rhs_args) == len(definition.params):
            args = rhs_args
        elif len(lhs_vars) + len(rhs_args) == len(definition.params):
            args = lhs_vars + rhs_args
        else:
            continue
        mapping = dict(zip(definition.params, args))
        for template in definition.facts:
            ground = substitute(template, mapping)
            if ground not in facts:
                facts[ground] = ProofStep(
                    conclusion=ground,
                    source=f"definition:{op}",
                    premises=[],
                    detail=f"Expanded from constructor `{op}`.",
                )
    return facts


def unify(pattern: Atom, ground: Atom, seed: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    if pattern.pred != ground.pred or len(pattern.args) != len(ground.args):
        return None
    mapping = dict(seed or {})
    for left, right in zip(pattern.args, ground.args):
        if left in mapping:
            if mapping[left] != right:
                return None
        else:
            mapping[left] = right
    return mapping


def apply_mapping(atom: Atom, mapping: Dict[str, str]) -> Atom:
    return canonicalize(Atom(atom.pred, tuple(mapping[token] for token in atom.args)))


def forward_chain(initial_facts: Dict[Atom, ProofStep], rules: List[Rule], max_new_facts: int = 500) -> Dict[Atom, ProofStep]:
    facts = dict(initial_facts)
    fact_list = list(facts)
    pred_index: Dict[str, List[Atom]] = defaultdict(list)
    for atom in fact_list:
        pred_index[atom.pred].append(atom)

    new_count = 0
    changed = True
    while changed and new_count < max_new_facts:
        changed = False
        for rule in rules:
            candidates = [pred_index[premise.pred] for premise in rule.premises]
            if any(not c for c in candidates):
                continue
            states = [({}, [])]
            for premise, matches in zip(rule.premises, candidates):
                next_states = []
                for mapping, used in states:
                    for atom in matches:
                        unified = unify(premise, atom, mapping)
                        if unified is not None:
                            next_states.append((unified, used + [atom]))
                states = next_states
                if not states:
                    break
            for mapping, used in states:
                try:
                    conclusion = apply_mapping(rule.conclusion, mapping)
                except KeyError:
                    continue
                if conclusion in facts:
                    continue
                facts[conclusion] = ProofStep(
                    conclusion=conclusion,
                    source=f"rule:{rule.index}",
                    premises=used,
                    detail=rule.text,
                )
                pred_index[conclusion.pred].append(conclusion)
                new_count += 1
                changed = True
                if new_count >= max_new_facts:
                    break
            if new_count >= max_new_facts:
                break
    return facts


def build_rule_index(rules: List[Rule]) -> Dict[str, List[Rule]]:
    by_pred: Dict[str, List[Rule]] = defaultdict(list)
    for rule in rules:
        by_pred[rule.conclusion.pred].append(rule)
    return by_pred


def rule_priority(rule: Rule, facts: Dict[Atom, ProofStep]) -> Tuple[int, int, int]:
    ready = sum(1 for premise in rule.premises if premise in facts)
    return (-ready, len(rule.premises), rule.index)


def prove_goal(
    goal: Atom,
    facts: Dict[Atom, ProofStep],
    rules_by_pred: Dict[str, List[Rule]],
    cache: Dict[Atom, Optional[ProofStep]],
    depth: int,
    seen: Optional[set] = None,
) -> Optional[ProofStep]:
    goal = canonicalize(goal)
    if goal in facts:
        return facts[goal]
    if depth <= 0:
        return None
    if goal in cache:
        return cache[goal]
    seen = set(seen or set())
    if goal in seen:
        return None
    seen.add(goal)

    rules = sorted(rules_by_pred.get(goal.pred, []), key=lambda rule: rule_priority(rule, facts))
    for rule in rules:
        mapping = unify(rule.conclusion, goal)
        if mapping is None:
            continue
        premises: List[Atom] = []
        premise_steps: List[ProofStep] = []
        ok = True
        for premise in rule.premises:
            try:
                grounded_premise = apply_mapping(premise, mapping)
            except KeyError:
                ok = False
                break
            step = prove_goal(grounded_premise, facts, rules_by_pred, cache, depth - 1, seen)
            if step is None:
                ok = False
                break
            premise_steps.append(step)
            premises.append(step.conclusion)
        if not ok:
            continue
        step = ProofStep(
            conclusion=goal,
            source=f"rule:{rule.index}",
            premises=premises,
            detail=rule.text,
        )
        facts[goal] = step
        cache[goal] = step
        return step

    cache[goal] = None
    return None


def proof_depth(goal: Atom, facts: Dict[Atom, ProofStep], memo: Optional[Dict[Atom, int]] = None) -> int:
    memo = memo or {}
    goal = canonicalize(goal)
    if goal in memo:
        return memo[goal]
    step = facts[goal]
    if not step.premises:
        memo[goal] = 1
        return 1
    value = 1 + max(proof_depth(premise, facts, memo) for premise in step.premises)
    memo[goal] = value
    return value


def linearize_proof(goal: Atom, facts: Dict[Atom, ProofStep]) -> List[ProofStep]:
    ordered: List[ProofStep] = []
    seen: set = set()

    def visit(atom: Atom) -> None:
        atom = canonicalize(atom)
        if atom in seen:
            return
        seen.add(atom)
        step = facts[atom]
        for premise in step.premises:
            visit(premise)
        ordered.append(step)

    visit(goal)
    return ordered


def collect_problem_stats(problem: Problem, initial_facts: Dict[Atom, ProofStep]) -> dict:
    constructor_counts = Counter(op for op, _, _ in problem.constructions)
    premise_predicates = Counter(atom.pred for atom in initial_facts)
    return {
        "name": problem.name,
        "goal_predicate": problem.goal.pred,
        "num_constructions": len(problem.constructions),
        "num_initial_facts": len(initial_facts),
        "constructor_histogram": dict(constructor_counts),
        "initial_predicate_histogram": dict(premise_predicates),
    }


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "proofs").mkdir(exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: dict | list) -> None:
    path.write_text(json.dumps(data, indent=2))


def save_proof_artifacts(problem: Problem, facts: Dict[Atom, ProofStep]) -> None:
    steps = linearize_proof(problem.goal, facts)
    json_path = OUTPUT_DIR / "proofs" / f"{problem.name}.json"
    txt_path = OUTPUT_DIR / "proofs" / f"{problem.name}.txt"
    write_json(json_path, [step.to_dict() for step in steps])

    lines = [f"Problem: {problem.name}", f"Goal: {problem.goal}", ""]
    for idx, step in enumerate(steps, start=1):
        premise_str = ", ".join(str(premise) for premise in step.premises) if step.premises else "none"
        lines.append(f"{idx}. {step.conclusion}")
        lines.append(f"   source: {step.source}")
        lines.append(f"   premises: {premise_str}")
        lines.append(f"   detail: {step.detail}")
    txt_path.write_text("\n".join(lines))


def plot_goal_distribution(problem_stats: List[dict]) -> None:
    counts = Counter(stat["goal_predicate"] for stat in problem_stats)
    labels = sorted(counts)
    values = [counts[label] for label in labels]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values, color="#4c78a8")
    plt.title("Benchmark Goal Predicate Distribution")
    plt.ylabel("Problems")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "goal_distribution.png", dpi=200)
    plt.close()


def plot_initial_fact_counts(problem_stats: List[dict]) -> None:
    names = [stat["name"].replace("translated_", "") for stat in problem_stats]
    values = [stat["num_initial_facts"] for stat in problem_stats]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(names)), values, color="#72b7b2")
    plt.title("Initial Symbolic Fact Count by Problem")
    plt.ylabel("Facts")
    plt.xlabel("Problem")
    plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "initial_fact_counts.png", dpi=200)
    plt.close()


def plot_solver_comparison(results: List[dict]) -> None:
    labels = ["definitions", "forward", "hybrid"]
    solved = [sum(1 for row in results if row[f"{label}_solved"]) for label in labels]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, solved, color=["#bab0ac", "#f58518", "#54a24b"])
    plt.title("Solved Problems by Solver Variant")
    plt.ylabel("Solved")
    plt.ylim(0, max(solved) + 2)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "solver_comparison.png", dpi=200)
    plt.close()


def plot_best_rule_gaps(results: List[dict]) -> None:
    ordered = sorted(results, key=lambda row: (row["best_rule_gap"], row["name"]))
    names = [row["name"].replace("translated_", "") for row in ordered]
    gaps = [row["best_rule_gap"] for row in ordered]
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(names)), gaps, color="#e45756")
    plt.title("Best Goal-Rule Premise Gap by Problem")
    plt.ylabel("Missing grounded premises")
    plt.xlabel("Problem")
    plt.xticks(range(len(names)), names, rotation=75, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / "best_rule_gaps.png", dpi=200)
    plt.close()


def analyze_goal_gap(goal: Atom, facts: Dict[Atom, ProofStep], rules: List[Rule]) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    best: Optional[Tuple[int, int, str]] = None
    for rule in rules:
        if rule.conclusion.pred != goal.pred:
            continue
        mapping = unify(rule.conclusion, goal)
        if mapping is None:
            continue
        missing = 0
        for premise in rule.premises:
            if any(token not in mapping for token in premise.args):
                missing += 1
                continue
            grounded = apply_mapping(premise, mapping)
            if grounded not in facts:
                missing += 1
        candidate = (missing, rule.index, rule.text)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None, None, None
    return best


def main() -> None:
    ensure_dirs()
    definitions = load_definitions(DATA_DIR / "defs.txt")
    rules = load_rules(DATA_DIR / "rules.txt")
    problems = load_problems(DATA_DIR / "imo_ag_30.txt")
    rules_by_pred = build_rule_index(rules)

    problem_stats: List[dict] = []
    results: List[dict] = []
    solved_names: List[str] = []
    constructor_hist = Counter()
    initial_pred_hist = Counter()

    for problem in problems:
        start = time.perf_counter()
        initial_facts = instantiate_problem(problem, definitions)
        problem_stats.append(collect_problem_stats(problem, initial_facts))
        constructor_hist.update(op for op, _, _ in problem.constructions)
        initial_pred_hist.update(atom.pred for atom in initial_facts)

        definitions_solved = problem.goal in initial_facts

        forward_facts = forward_chain(initial_facts, rules, max_new_facts=400)
        forward_solved = problem.goal in forward_facts

        hybrid_facts = dict(forward_facts)
        cache: Dict[Atom, Optional[ProofStep]] = {}
        hybrid_step = prove_goal(problem.goal, hybrid_facts, rules_by_pred, cache, depth=6)
        hybrid_solved = hybrid_step is not None

        row = {
            "name": problem.name,
            "goal": str(problem.goal),
            "goal_predicate": problem.goal.pred,
            "num_constructions": len(problem.constructions),
            "num_initial_facts": len(initial_facts),
            "num_forward_facts": len(forward_facts),
            "num_derived_facts": len(forward_facts) - len(initial_facts),
            "definitions_solved": definitions_solved,
            "forward_solved": forward_solved,
            "hybrid_solved": hybrid_solved,
            "runtime_sec": round(time.perf_counter() - start, 4),
            "hybrid_proof_depth": proof_depth(problem.goal, hybrid_facts) if hybrid_solved else None,
            "hybrid_proof_steps": len(linearize_proof(problem.goal, hybrid_facts)) if hybrid_solved else None,
        }
        best_gap, best_rule_idx, best_rule_text = analyze_goal_gap(problem.goal, forward_facts, rules)
        row["best_rule_gap"] = best_gap
        row["best_rule_index"] = best_rule_idx
        row["best_rule_text"] = best_rule_text
        results.append(row)
        if hybrid_solved:
            solved_names.append(problem.name)
            save_proof_artifacts(problem, hybrid_facts)

    with (OUTPUT_DIR / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    write_json(OUTPUT_DIR / "problem_stats.json", problem_stats)
    write_json(
        OUTPUT_DIR / "aggregate_stats.json",
        {
            "num_problems": len(problems),
            "num_definitions": len(definitions),
            "num_rules": len(rules),
            "constructor_histogram": dict(constructor_hist),
            "initial_predicate_histogram": dict(initial_pred_hist),
            "solved_problems": solved_names,
            "solver_totals": {
                "definitions": sum(1 for row in results if row["definitions_solved"]),
                "forward": sum(1 for row in results if row["forward_solved"]),
                "hybrid": sum(1 for row in results if row["hybrid_solved"]),
            },
            "mean_runtime_sec": round(sum(row["runtime_sec"] for row in results) / len(results), 4),
        },
    )

    plot_goal_distribution(problem_stats)
    plot_initial_fact_counts(problem_stats)
    plot_solver_comparison(results)
    plot_best_rule_gaps(results)


if __name__ == "__main__":
    main()
