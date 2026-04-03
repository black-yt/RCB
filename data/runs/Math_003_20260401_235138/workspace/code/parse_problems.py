"""
Parse and analyze IMO geometry problems from the AlphaGeometry benchmark format.
"""

import re
import json
from collections import defaultdict

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260401_235138"

def parse_problems(filepath):
    """Parse the IMO geometry benchmark file."""
    problems = []
    with open(filepath) as f:
        lines = [l.rstrip('\n') for l in f.readlines()]

    i = 0
    while i < len(lines):
        name_line = lines[i].strip()
        if not name_line:
            i += 1
            continue
        if i + 1 < len(lines):
            problem_line = lines[i+1].strip()
            if problem_line:
                problems.append({'name': name_line, 'statement': problem_line})
                i += 2
            else:
                i += 1
        else:
            i += 1

    return problems


def parse_statement(stmt):
    """Parse a geometry problem statement into components."""
    # Split on '?' to get premises and goal
    if '?' not in stmt:
        return None

    parts = stmt.split('?')
    premises_str = parts[0].strip()
    goal_str = parts[1].strip()

    # Parse constructions (semicolon-separated)
    constructions = []
    for constr in premises_str.split(';'):
        constr = constr.strip()
        if constr:
            constructions.append(constr)

    # Parse individual point definitions
    points = {}
    for constr in constructions:
        # Match patterns like: "a b c = triangle a b c" or "h = orthocenter h a b c"
        m = re.match(r'^([\w\s@.,-]+?)\s*=\s*(.+)$', constr)
        if m:
            lhs = m.group(1).strip()
            rhs = m.group(2).strip()
            # Extract point names from LHS
            point_names = []
            for token in lhs.split():
                if re.match(r'^[a-zA-Z]\w*', token) and not '@' in token:
                    base = token.split('@')[0]
                    point_names.append(base)
                elif '@' in token:
                    base = token.split('@')[0]
                    point_names.append(base)
            for pname in point_names:
                if pname not in points:
                    points[pname] = {'defined_by': rhs.split(',')[0].split()[0] if rhs else 'unknown'}

    # Extract construction primitives used
    primitives_used = set()
    for constr in constructions:
        # Extract the construction type from RHS
        if '=' in constr:
            rhs = constr.split('=', 1)[1].strip()
            for clause in rhs.split(','):
                clause = clause.strip()
                tokens = clause.split()
                if tokens:
                    primitives_used.add(tokens[0])

    # Parse goal
    goal_parts = goal_str.split()
    goal_type = goal_parts[0] if goal_parts else 'unknown'

    return {
        'constructions': constructions,
        'num_constructions': len(constructions),
        'num_points': len(points),
        'points': points,
        'primitives_used': list(primitives_used),
        'goal': goal_str,
        'goal_type': goal_type,
    }


def extract_year(name):
    """Extract IMO year from problem name."""
    m = re.search(r'(\d{4})', name)
    return int(m.group(1)) if m else None


def extract_problem_number(name):
    """Extract problem number from name."""
    m = re.search(r'_p(\d+)', name)
    return int(m.group(1)) if m else None


def classify_goal(goal_type):
    """Classify goal into categories."""
    mapping = {
        'cong': 'Congruence',
        'para': 'Parallelism',
        'perp': 'Perpendicularity',
        'coll': 'Collinearity',
        'cyclic': 'Concyclicity',
        'eqangle': 'Equal Angles',
        'eqratio': 'Equal Ratios',
        'simtri': 'Similar Triangles',
        'contri': 'Congruent Triangles',
    }
    return mapping.get(goal_type, 'Other')


def count_primitives(parsed):
    """Count occurrences of each geometric primitive."""
    primitives = defaultdict(int)
    for constr in parsed.get('constructions', []):
        if '=' in constr:
            rhs = constr.split('=', 1)[1].strip()
            for clause in rhs.split(','):
                tokens = clause.strip().split()
                if tokens:
                    primitives[tokens[0]] += 1
    return dict(primitives)


def analyze_problems(problems):
    """Analyze all problems and extract statistics."""
    results = []
    for prob in problems:
        parsed = parse_statement(prob['statement'])
        if not parsed:
            continue

        year = extract_year(prob['name'])
        pnum = extract_problem_number(prob['name'])
        goal_category = classify_goal(parsed['goal_type'])
        prims = count_primitives(parsed)

        results.append({
            'name': prob['name'],
            'year': year,
            'problem_number': pnum,
            'num_constructions': parsed['num_constructions'],
            'num_points': parsed['num_points'],
            'goal_type': parsed['goal_type'],
            'goal_category': goal_category,
            'goal': parsed['goal'],
            'primitives': prims,
            'statement': prob['statement'],
        })
    return results


if __name__ == '__main__':
    filepath = f"{WORKSPACE}/data/imo_ag_30.txt"
    problems = parse_problems(filepath)
    print(f"Total problems parsed: {len(problems)}")

    analysis = analyze_problems(problems)

    # Save analysis
    with open(f"{WORKSPACE}/outputs/problem_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    # Print summary
    print("\n=== Problem Summary ===")
    for p in analysis:
        print(f"  {p['name']}: {p['num_constructions']} constructions, "
              f"{p['num_points']} points, goal={p['goal_type']}")

    print(f"\n=== Goal Type Distribution ===")
    goal_counts = defaultdict(int)
    for p in analysis:
        goal_counts[p['goal_category']] += 1
    for cat, cnt in sorted(goal_counts.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {cnt}")

    print(f"\n=== Years covered ===")
    years = sorted(set(p['year'] for p in analysis if p['year']))
    print(f"  {years}")

    print(f"\n=== Complexity stats ===")
    constructions = [p['num_constructions'] for p in analysis]
    print(f"  Avg constructions: {sum(constructions)/len(constructions):.1f}")
    print(f"  Min: {min(constructions)}, Max: {max(constructions)}")
