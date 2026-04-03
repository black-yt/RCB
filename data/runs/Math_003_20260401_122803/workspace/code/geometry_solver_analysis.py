import re
import json
from pathlib import Path

import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_PATH = Path('data/imo_ag_30.txt')
OUTPUT_DIR = Path('outputs')
FIG_DIR = Path('report/images')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

PROBLEM_RE = re.compile(r"^(translated_imo_\\d{4}_p\\d+[a-z]?)$")


def parse_benchmark(path: Path):
    problems = []
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('translated_'):
            if current is not None:
                problems.append(current)
            current = {"name": line, "spec": None}
        else:
            if current is None:
                continue
            current["spec"] = line
    if current is not None:
        problems.append(current)
    return problems


def analyse_spec(spec: str):
    if not spec or '?' not in spec:
        return {
            'num_objects': 0,
            'num_predicates': 0,
            'constructors': [],
            'conclusion': 'unknown',
        }
    lhs, rhs = spec.split('?', 1)
    premises, conclusion = lhs.strip(), rhs.strip()
    # objects: tokens before first '=' of the first statement
    first_stmt = premises.split(';')[0]
    if '=' in first_stmt:
        obj_tokens = first_stmt.split('=')[0].strip()
    else:
        obj_tokens = first_stmt.strip()
    objs = [t for t in obj_tokens.split() if t.isalpha()]

    # predicates separated by ';'
    predicates = []
    for p in premises.split(';'):
        p = p.strip()
        if not p:
            continue
        if '=' in p or '@' in p:
            predicates.append(p)

    constructors = []
    for p in predicates:
        if '=' in p:
            _, right = p.split('=', 1)
            rhs_terms = [t.strip() for t in right.split(',')]
            for term in rhs_terms:
                fun = term.split()[0]
                constructors.append(fun)
        elif '@' in p:
            # coordinate annotated objects; no constructor symbol
            continue

    concl_fun = conclusion.split()[0] if conclusion else 'unknown'
    return {
        'num_objects': len(set(objs)),
        'num_predicates': len(predicates),
        'constructors': constructors,
        'conclusion': concl_fun,
    }


def build_dataframe(problems):
    rows = []
    for prob in problems:
        info = analyse_spec(prob['spec'])
        rows.append({
            'name': prob['name'],
            'num_objects': info['num_objects'],
            'num_predicates': info['num_predicates'],
            'constructors': info['constructors'],
            'conclusion': info['conclusion'],
        })
    df = pd.DataFrame(rows)
    # expand constructors into counts
    all_cons = sorted({c for cs in df['constructors'].tolist() for c in cs}) if len(df) else []
    for c in all_cons:
        df[f'cons_{c}'] = df['constructors'].apply(lambda cs, c=c: cs.count(c))
    return df


def plot_overview(df: pd.DataFrame):
    if df.empty:
        return
    sns.set(style='whitegrid')

    plt.figure(figsize=(6, 4))
    sns.histplot(df['num_objects'], bins=range(0, int(df['num_objects'].max()) + 2), discrete=True)
    plt.xlabel('Number of geometric objects')
    plt.ylabel('Count of problems')
    plt.title('Distribution of object counts in IMO-AG-30')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'objects_hist.png')
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df['num_predicates'], bins=range(0, int(df['num_predicates'].max()) + 2), discrete=True)
    plt.xlabel('Number of constructor predicates')
    plt.ylabel('Count of problems')
    plt.title('Distribution of predicate counts in IMO-AG-30')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'predicates_hist.png')
    plt.close()

    cons_cols = [c for c in df.columns if c.startswith('cons_')]
    if cons_cols:
        cons_freq = df[cons_cols].sum().sort_values(ascending=False)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=cons_freq.index.str.replace('cons_', ''), y=cons_freq.values)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Total count across problems')
        plt.title('Constructor usage in IMO-AG-30')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'constructors_bar.png')
        plt.close()

    plt.figure(figsize=(6, 4))
    order = df['conclusion'].value_counts().index
    sns.countplot(y=df['conclusion'], order=order)
    plt.xlabel('Number of problems')
    plt.ylabel('Target relation')
    plt.title('Distribution of proof goals')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'conclusions_bar.png')
    plt.close()


def simple_symbolic_reasoner_example():
    """Very small synthetic geometry proof using SymPy.

    Segment AB with A(x1,y1), B(-x1,-y1) has midpoint at origin. Prove |AM| = |MB|.
    """
    x1, y1 = sp.symbols('x1 y1', real=True)
    A = sp.Matrix([x1, y1])
    B = sp.Matrix([-x1, -y1])
    M = sp.Matrix([0, 0])
    AM2 = (A - M).dot(A - M)
    MB2 = (B - M).dot(B - M)
    proof = sp.simplify(AM2 - MB2)
    return proof


def main():
    problems = parse_benchmark(DATA_PATH)
    with (OUTPUT_DIR / 'parsed_problems.json').open('w') as f:
        json.dump(problems, f, indent=2)

    df = build_dataframe(problems)
    df.to_csv(OUTPUT_DIR / 'benchmark_stats.csv', index=False)

    plot_overview(df)

    proof_example = simple_symbolic_reasoner_example()
    with (OUTPUT_DIR / 'symbolic_example.txt').open('w') as f:
        f.write(f'simplified difference AM^2-MB^2 = {proof_example}\n')


if __name__ == '__main__':
    main()
