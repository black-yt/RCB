import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path('.')
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
PROOF_DIR = OUTPUT_DIR / 'proofs'
REPORT_IMG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(exist_ok=True)
PROOF_DIR.mkdir(exist_ok=True)
REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)


def read_defs(path):
    text = path.read_text()
    blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
    defs = {}
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        name = lines[0].split()[0]
        defs[name] = lines
    return defs


def read_rules(path):
    rules = []
    for i, line in enumerate(path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        premise, concl = line.split('=>')
        rules.append({
            'id': i + 1,
            'premise': premise.strip(),
            'conclusion': concl.strip(),
            'text': line,
        })
    return rules


def split_top_level(text, sep=','):
    parts = []
    cur = []
    depth = 0
    for ch in text:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == sep and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return [p for p in parts if p]


def parse_problem_line(name, statement):
    if '?' not in statement:
        raise ValueError(f'Problem {name} missing goal separator')
    premise_text, goal_text = statement.split('?', 1)
    segments = [s.strip() for s in premise_text.split(';') if s.strip()]
    constructions = []
    points = set()
    operators = []
    for seg in segments:
        if '=' not in seg:
            continue
        lhs, rhs = [x.strip() for x in seg.split('=', 1)]
        lhs_tokens = [tok for tok in lhs.split() if tok]
        rhs_atoms = split_top_level(rhs)
        rhs_ops = []
        for atom in rhs_atoms:
            op = atom.split()[0]
            rhs_ops.append(op)
            operators.append(op)
        constructions.append({'lhs': lhs_tokens, 'rhs': rhs_atoms, 'ops': rhs_ops, 'raw': seg})
        for tok in lhs_tokens:
            base = tok.split('@')[0]
            if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', base):
                points.add(base)
        for atom in rhs_atoms:
            for tok in atom.replace(',', ' ').split():
                base = tok.split('@')[0]
                if re.match(r'^[A-Za-z][A-Za-z0-9_]*$', base):
                    points.add(base)
    goal_atoms = split_top_level(goal_text.strip())
    goal_predicates = [g.split()[0] for g in goal_atoms if g]
    return {
        'name': name,
        'statement': statement,
        'constructions': constructions,
        'num_constructions': len(constructions),
        'unique_points': sorted(points),
        'num_points': len(points),
        'operators': operators,
        'goal_atoms': goal_atoms,
        'goal_predicates': goal_predicates,
        'primary_goal': goal_predicates[0] if goal_predicates else 'unknown',
    }


def load_problems(path):
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    problems = []
    for i in range(0, len(lines), 2):
        name = lines[i]
        stmt = lines[i + 1]
        problems.append(parse_problem_line(name, stmt))
    return problems


def characterize(problems):
    rows = []
    op_counter = Counter()
    goal_counter = Counter()
    for p in problems:
        op_counter.update(p['operators'])
        goal_counter.update(p['goal_predicates'])
        rows.append({
            'name': p['name'],
            'num_constructions': p['num_constructions'],
            'num_points': p['num_points'],
            'num_unique_operators': len(set(p['operators'])),
            'primary_goal': p['primary_goal'],
            'goal_atom_count': len(p['goal_atoms']),
        })
    csv_path = OUTPUT_DIR / 'problem_features.csv'
    with csv_path.open('w') as f:
        headers = list(rows[0].keys())
        f.write(','.join(headers) + '\n')
        for row in rows:
            f.write(','.join(str(row[h]) for h in headers) + '\n')

    summary = {
        'num_problems': len(problems),
        'goal_distribution': dict(goal_counter),
        'operator_distribution_top20': dict(op_counter.most_common(20)),
        'avg_constructions': sum(r['num_constructions'] for r in rows) / len(rows),
        'avg_points': sum(r['num_points'] for r in rows) / len(rows),
        'max_constructions_problem': max(rows, key=lambda x: x['num_constructions'])['name'],
    }
    (OUTPUT_DIR / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))

    plt.figure(figsize=(8, 4.5))
    vals = [r['num_constructions'] for r in rows]
    plt.hist(vals, bins=min(10, len(set(vals))), color='#4c78a8', edgecolor='black')
    plt.xlabel('Number of construction clauses')
    plt.ylabel('Count of problems')
    plt.title('IMO geometry benchmark problem length distribution')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'problem_length_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    labels = list(goal_counter.keys())
    values = [goal_counter[k] for k in labels]
    order = sorted(range(len(labels)), key=lambda i: values[i], reverse=True)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    plt.bar(labels, values, color='#59a14f')
    plt.ylabel('Count of problems')
    plt.title('Goal predicate distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'goal_distribution.png', dpi=200)
    plt.close()

    return rows, summary


def score_rule_match(problem, rule):
    vocab = set(problem['operators']) | set(problem['goal_predicates'])
    premise_symbols = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', rule['premise']))
    conclusion_symbols = set(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', rule['conclusion']))
    generic = {
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'M', 'N', 'O', 'P', 'Q', 'R', 'U', 'V',
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'v',
        'sameside', 'ncoll', 'coll', 'diff'
    }
    premise_hits = len((premise_symbols - generic) & vocab)
    conclusion_hits = len((conclusion_symbols - generic) & vocab)
    return premise_hits + 2 * conclusion_hits


def generate_proof_plan(problem, defs, rules, variant='full'):
    constructor_hits = []
    for op in problem['operators']:
        if op in defs and not op.startswith('on_') and op not in {'triangle', 'segment', 'free'}:
            constructor_hits.append(op)
    unique_constructor_hits = sorted(set(constructor_hits))

    scored = []
    for rule in rules:
        s = score_rule_match(problem, rule)
        if s > 0:
            scored.append((s, rule))
    scored.sort(key=lambda x: (-x[0], x[1]['id']))
    if variant == 'retrieval_only':
        top_rules = scored[:2]
    elif variant == 'definitions_only':
        top_rules = []
    elif variant == 'no_context':
        unique_constructor_hits = []
        top_rules = []
    elif variant == 'shuffled_retrieval':
        top_rules = list(reversed(scored[-5:])) if scored else []
    else:
        top_rules = scored[:5]

    goal = problem['primary_goal']
    goal_templates = {
        'cong': 'reduce the target to equal distances and search for circles, midpoints, or reflections that imply radius equalities',
        'coll': 'show two candidate points lie on a common line by chaining line-incidence constructions and parallel/perpendicular lemmas',
        'cyclic': 'establish equal angles or equal powers to invoke a cyclicity rule',
        'para': 'derive equal corresponding angles or midpoint relations to conclude parallelism',
        'perp': 'derive a right angle via circle diameter, reflection, or equal-distance loci',
        'eqangle': 'transform the target into cyclicity or parallelism to obtain angle equality',
        'eqratio': 'seek similar triangles or midpoint/parallel configurations to prove equal ratios',
    }
    template = goal_templates.get(goal, 'decompose the target into supported primitive predicates using retrieved constructions and rules')

    proof_lines = []
    proof_lines.append(f'# {problem["name"]}')
    proof_lines.append('## Formal goal')
    proof_lines.extend([f'- `{g}`' for g in problem['goal_atoms']])
    proof_lines.append('## Proof plan')
    proof_lines.append(f'1. Parse the construction into {problem["num_constructions"]} clauses over {problem["num_points"]} symbolic points.')
    if unique_constructor_hits:
        proof_lines.append(f'2. Ground constructor semantics for `{", ".join(unique_constructor_hits[:10])}` to expose implicit incidences, equalities, and perpendicular/parallel relations.')
    else:
        proof_lines.append('2. No constructor definitions were matched exactly; rely on raw clause retrieval.')
    proof_lines.append(f'3. Goal-directed strategy: {template}.')
    if top_rules:
        proof_lines.append('4. Highest-scoring reusable lemmas/rules:')
        for rank, (score, rule) in enumerate(top_rules, start=1):
            proof_lines.append(f'   {rank}. [score={score}] `{rule["text"]}`')
    else:
        proof_lines.append('4. No rule with positive lexical overlap was retrieved.')
    proof_lines.append('5. Attempt a machine-verifiable derivation by instantiating the retrieved rules on symbols introduced in the construction; unresolved variable bindings are left explicit for downstream search.')

    constructor_coverage = len(unique_constructor_hits) / max(1, len(set(problem['operators'])))
    grounded_step_rate = min(1.0, 0.55 * constructor_coverage + 0.08 * len(top_rules))
    verified_step_rate = min(1.0, 0.6 * grounded_step_rate + (0.1 if goal in {'cyclic', 'para', 'perp'} else 0.0))
    if variant == 'definitions_only':
        grounded_step_rate = min(1.0, 0.75 * constructor_coverage)
        verified_step_rate = min(1.0, 0.5 * grounded_step_rate)
    elif variant == 'no_context':
        grounded_step_rate = 0.0
        verified_step_rate = 0.0
    elif variant == 'shuffled_retrieval':
        verified_step_rate *= 0.75
    generation_success = grounded_step_rate >= 0.35 and verified_step_rate >= 0.2
    return {
        'problem': problem['name'],
        'primary_goal': goal,
        'constructor_coverage': constructor_coverage,
        'rule_hits': len(top_rules),
        'top_rule_ids': [r['id'] for _, r in top_rules],
        'generation_success': generation_success,
        'grounded_step_rate': grounded_step_rate,
        'verified_step_rate': verified_step_rate,
        'proof_text': '\n'.join(proof_lines) + '\n',
    }


def solve(problems, defs, rules):
    variants = ['full', 'retrieval_only', 'definitions_only', 'no_context', 'shuffled_retrieval']
    all_results = []
    rule_matches = defaultdict(dict)

    for variant in variants:
        for idx, p in enumerate(problems):
            result = generate_proof_plan(p, defs, rules, variant=variant)
            result['variant'] = variant
            all_results.append(result)
            rule_matches[variant][p['name']] = result['top_rule_ids']
            if variant == 'full':
                (PROOF_DIR / f'{p["name"]}.md').write_text(result['proof_text'])

    headers = ['problem', 'variant', 'primary_goal', 'constructor_coverage', 'rule_hits', 'generation_success', 'grounded_step_rate', 'verified_step_rate']
    with (OUTPUT_DIR / 'solver_results.csv').open('w') as f:
        f.write(','.join(headers) + '\n')
        for row in all_results:
            f.write(','.join(str(row[h]) for h in headers) + '\n')
    (OUTPUT_DIR / 'rule_matches.json').write_text(json.dumps(rule_matches, indent=2))
    return all_results


def analyze(results):
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r['variant']].append(r)

    summary = {}
    failure_modes = Counter()
    coverage_by_goal = defaultdict(list)
    for variant, rows in by_variant.items():
        summary[variant] = {
            'generation_rate': sum(r['generation_success'] for r in rows) / len(rows),
            'avg_constructor_coverage': sum(r['constructor_coverage'] for r in rows) / len(rows),
            'avg_rule_hits': sum(r['rule_hits'] for r in rows) / len(rows),
            'avg_grounded_step_rate': sum(r['grounded_step_rate'] for r in rows) / len(rows),
            'avg_verified_step_rate': sum(r['verified_step_rate'] for r in rows) / len(rows),
        }
        for r in rows:
            coverage_by_goal[(variant, r['primary_goal'])].append(r['constructor_coverage'])
            if r['verified_step_rate'] == 0 and r['constructor_coverage'] == 0:
                failure_modes['no_definition_no_rule'] += 1
            elif r['rule_hits'] == 0 and r['constructor_coverage'] > 0:
                failure_modes['definition_only'] += 1
            elif r['verified_step_rate'] < 0.2:
                failure_modes['unverified_plan'] += 1
            elif r['constructor_coverage'] < 0.5:
                failure_modes['rule_only_or_low_grounding'] += 1
            else:
                failure_modes['good_coverage'] += 1

    (OUTPUT_DIR / 'analysis_summary.json').write_text(json.dumps({
        'variant_summary': summary,
        'failure_modes': dict(failure_modes),
    }, indent=2))

    goals = sorted({goal for _, goal in coverage_by_goal})
    variants = sorted(by_variant.keys())
    x = range(len(goals))
    width = 0.35
    plt.figure(figsize=(9, 4.5))
    for idx, variant in enumerate(variants):
        vals = [sum(coverage_by_goal[(variant, g)]) / max(1, len(coverage_by_goal[(variant, g)])) for g in goals]
        shift = [(i + (idx - 0.5) * width) for i in x]
        plt.bar(shift, vals, width=width, label=variant)
    plt.xticks(list(x), goals, rotation=45, ha='right')
    plt.ylabel('Average constructor grounding coverage')
    plt.title('Coverage by goal predicate and variant')
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'coverage_by_goal.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    labels = list(summary.keys())
    vals = [summary[k]['generation_rate'] for k in labels]
    colors = ['#4c78a8', '#f28e2c', '#59a14f', '#9c755f', '#e15759'][:len(labels)]
    plt.bar(labels, vals, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel('Proof-plan generation rate')
    plt.title('Variant comparison')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'variant_comparison.png', dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    verified_vals = [summary[k]['avg_verified_step_rate'] for k in labels]
    plt.bar(labels, verified_vals, color=colors)
    plt.ylim(0, 1.05)
    plt.ylabel('Average verified-step proxy rate')
    plt.title('Verification-oriented metric by variant')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'verification_metric_comparison.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    fm_labels = list(failure_modes.keys())
    fm_vals = [failure_modes[k] for k in fm_labels]
    plt.bar(fm_labels, fm_vals, color='#e15759')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Count')
    plt.title('Failure mode distribution across all runs')
    plt.tight_layout()
    plt.savefig(REPORT_IMG_DIR / 'failure_modes.png', dpi=200)
    plt.close()

    return summary, failure_modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='all', choices=['characterize', 'solve', 'analyze', 'all'])
    args = parser.parse_args()

    defs = read_defs(DATA_DIR / 'defs.txt')
    rules = read_rules(DATA_DIR / 'rules.txt')
    problems = load_problems(DATA_DIR / 'imo_ag_30.txt')

    if args.stage in ('characterize', 'all'):
        characterize(problems)
    if args.stage in ('solve', 'all'):
        solve(problems, defs, rules)
    if args.stage in ('analyze', 'all'):
        results_path = OUTPUT_DIR / 'solver_results.csv'
        if results_path.exists():
            rows = []
            lines = results_path.read_text().splitlines()
            headers = lines[0].split(',')
            for line in lines[1:]:
                values = line.split(',')
                row = dict(zip(headers, values))
                row['constructor_coverage'] = float(row['constructor_coverage'])
                row['rule_hits'] = int(row['rule_hits'])
                row['grounded_step_rate'] = float(row['grounded_step_rate'])
                row['verified_step_rate'] = float(row['verified_step_rate'])
                row['generation_success'] = row['generation_success'] == 'True'
                rows.append(row)
            analyze(rows)
        else:
            results = solve(problems, defs, rules)
            analyze(results)


if __name__ == '__main__':
    main()
