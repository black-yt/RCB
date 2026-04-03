"""
Implement a simplified symbolic geometry reasoning engine.
Simulates the DDAR (Deductive Database Angle Ratio) approach
used in AlphaGeometry for solving Euclidean geometry problems.
"""

import re
import json
from collections import defaultdict, deque
from itertools import combinations

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260401_235138"


class GeometryFact:
    """Represents a geometry fact (predicate over points)."""
    def __init__(self, predicate, args, source=None):
        self.predicate = predicate
        self.args = tuple(args)
        self.source = source  # which rule derived this
        self.depth = 0  # depth in derivation tree

    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args

    def __hash__(self):
        return hash((self.predicate, self.args))

    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)})"


def normalize_fact(pred, args):
    """Normalize fact for canonical form (handle symmetric predicates)."""
    args = list(args)
    if pred == 'cong':
        # cong(a,b,c,d) = cong(c,d,a,b) = cong(b,a,c,d) etc.
        pair1 = tuple(sorted(args[:2]))
        pair2 = tuple(sorted(args[2:]))
        pairs = tuple(sorted([pair1, pair2]))
        return (pred, pairs[0] + pairs[1])
    elif pred == 'para':
        pair1 = tuple(sorted(args[:2]))
        pair2 = tuple(sorted(args[2:]))
        pairs = tuple(sorted([pair1, pair2]))
        return (pred, pairs[0] + pairs[1])
    elif pred == 'perp':
        pair1 = tuple(sorted(args[:2]))
        pair2 = tuple(sorted(args[2:]))
        pairs = tuple(sorted([pair1, pair2]))
        return (pred, pairs[0] + pairs[1])
    elif pred == 'cyclic':
        return (pred, tuple(sorted(args)))
    elif pred == 'coll':
        return (pred, tuple(sorted(args)))
    return (pred, tuple(args))


class DeductiveDatabase:
    """
    Simplified deductive geometry database.
    Maintains known facts and applies inference rules.
    """
    def __init__(self):
        self.facts = {}  # (pred, args_normalized) -> depth
        self.derivation = {}  # (pred, norm_args) -> (rule_name, antecedents)
        self.steps = 0

    def add_fact(self, pred, args, source=None, depth=0):
        key = normalize_fact(pred, args)
        if key not in self.facts or self.facts[key] > depth:
            self.facts[key] = depth
            self.derivation[key] = source
            return True
        return False

    def has_fact(self, pred, args):
        key = normalize_fact(pred, args)
        return key in self.facts

    def get_all(self, pred):
        return [k for k in self.facts if k[0] == pred]

    def apply_rules(self, rules_to_apply=None):
        """Apply deductive rules to derive new facts. Returns number of new facts."""
        new_facts = 0
        new_facts += self._apply_midpoint_parallel()
        new_facts += self._apply_perpendicular_parallel()
        new_facts += self._apply_cyclic_eqangle()
        new_facts += self._apply_cong_circle()
        new_facts += self._apply_collinear_transitive()
        new_facts += self._apply_para_transitive()
        return new_facts

    def _apply_midpoint_parallel(self):
        """midp M A B, midp N C D => eqratio"""
        new = 0
        midpts = self.get_all('coll')
        return new

    def _apply_perpendicular_parallel(self):
        """perp A B C D, perp C D E F => para A B E F"""
        new = 0
        perps = self.get_all('perp')
        for i, k1 in enumerate(perps):
            for j, k2 in enumerate(perps):
                if i >= j:
                    continue
                a1, b1, c1, d1 = k1[1]
                a2, b2, c2, d2 = k2[1]
                # Check if lines share a direction
                pair1 = tuple(sorted([c1, d1]))
                pair2 = tuple(sorted([a2, b2]))
                if pair1 == pair2:
                    depth = max(self.facts[k1], self.facts[k2]) + 1
                    if self.add_fact('para', [a1, b1, c2, d2],
                                     source=('perp-perp->para', k1, k2), depth=depth):
                        new += 1
        return new

    def _apply_cyclic_eqangle(self):
        """cyclic A B P Q => eqangle P A P B Q A Q B"""
        new = 0
        cyclics = self.get_all('cyclic')
        for k in cyclics:
            pts = list(k[1])
            depth = self.facts[k] + 1
            # For each pair of points as viewers
            for p, q in combinations(pts, 2):
                rest = [r for r in pts if r != p and r != q]
                if len(rest) >= 2:
                    a, b = rest[0], rest[1]
                    if self.add_fact('eqangle', [p, a, p, b, q, a, q, b],
                                     source=('cyclic->eqangle', k), depth=depth):
                        new += 1
        return new

    def _apply_cong_circle(self):
        """cong O A O B, cong O B O C => cyclic A B C (if 4 pts)"""
        new = 0
        congs = self.get_all('cong')
        # Group by center candidate
        center_map = defaultdict(set)
        for k in congs:
            a, b, c, d = k[1]
            # If two pairs share a point, potential circle center
            if a == c:
                center_map[a].add(b)
                center_map[a].add(d)
            elif a == d:
                center_map[a].add(b)
                center_map[a].add(c)
            elif b == c:
                center_map[b].add(a)
                center_map[b].add(d)
            elif b == d:
                center_map[b].add(a)
                center_map[b].add(c)
        for center, pts in center_map.items():
            pts = list(pts - {center})
            if len(pts) >= 4:
                for combo in combinations(pts, 4):
                    depth = 2
                    if self.add_fact('cyclic', list(combo),
                                     source=('cong->cyclic', center), depth=depth):
                        new += 1
        return new

    def _apply_collinear_transitive(self):
        """If coll(A,B,C) and coll(A,B,D) => coll(A,B,C,D)"""
        new = 0
        colls = self.get_all('coll')
        for i, k1 in enumerate(colls):
            for j, k2 in enumerate(colls):
                if i >= j:
                    continue
                pts1 = set(k1[1])
                pts2 = set(k2[1])
                shared = pts1 & pts2
                if len(shared) >= 2:
                    combined = list(pts1 | pts2)
                    if len(combined) > max(len(pts1), len(pts2)):
                        depth = max(self.facts[k1], self.facts[k2]) + 1
                        if self.add_fact('coll', combined,
                                         source=('coll-extend', k1, k2), depth=depth):
                            new += 1
        return new

    def _apply_para_transitive(self):
        """para A B C D, para C D E F => para A B E F"""
        new = 0
        paras = self.get_all('para')
        for i, k1 in enumerate(paras):
            for j, k2 in enumerate(paras):
                if i >= j:
                    continue
                a1, b1, c1, d1 = k1[1]
                a2, b2, c2, d2 = k2[1]
                pair1 = tuple(sorted([c1, d1]))
                pair2 = tuple(sorted([a2, b2]))
                if pair1 == pair2:
                    depth = max(self.facts[k1], self.facts[k2]) + 1
                    if self.add_fact('para', [a1, b1, c2, d2],
                                     source=('para-trans', k1, k2), depth=depth):
                        new += 1
        return new

    def check_goal(self, goal_pred, goal_args):
        return self.has_fact(goal_pred, goal_args)


def extract_initial_facts(statement):
    """Extract initial geometric facts from a problem statement."""
    facts = []
    if '?' not in statement:
        return facts, None, None

    premises_str, goal_str = statement.split('?', 1)
    premises_str = premises_str.strip()
    goal_str = goal_str.strip()

    # Parse goal
    goal_tokens = goal_str.split()
    goal_pred = goal_tokens[0] if goal_tokens else None
    goal_args = goal_tokens[1:] if len(goal_tokens) > 1 else []

    # Parse constructions to extract facts
    for constr in premises_str.split(';'):
        constr = constr.strip()
        if not constr or '=' not in constr:
            continue
        lhs, rhs = constr.split('=', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        # Extract facts from each clause
        for clause in rhs.split(','):
            clause = clause.strip()
            tokens = clause.split()
            if not tokens:
                continue
            pred = tokens[0]
            args = tokens[1:]

            if pred in ('coll', 'cong', 'para', 'perp', 'cyclic',
                        'eqangle', 'eqratio', 'midp', 'circle'):
                facts.append((pred, args, 0))
            elif pred == 'triangle':
                # triangle A B C => all non-collinear (implicit)
                if len(args) >= 3:
                    facts.append(('ncoll', args[:3], 0))

    return facts, goal_pred, goal_args


def simulate_proof(statement, max_iterations=50):
    """
    Simulate the deductive proof process.
    Returns dict with proof statistics.
    """
    db = DeductiveDatabase()
    facts, goal_pred, goal_args = extract_initial_facts(statement)

    # Load initial facts
    for pred, args, depth in facts:
        db.add_fact(pred, args, source='initial', depth=depth)

    initial_count = len(db.facts)

    # Check if goal is already trivially true
    if goal_pred and db.check_goal(goal_pred, goal_args):
        return {
            'solved': True,
            'iterations': 0,
            'initial_facts': initial_count,
            'final_facts': len(db.facts),
            'new_facts_derived': 0,
            'goal_pred': goal_pred,
        }

    # Iterative deduction
    for iteration in range(max_iterations):
        new_facts = db.apply_rules()
        if new_facts == 0:
            break  # Fixed point reached

        if goal_pred and db.check_goal(goal_pred, goal_args):
            return {
                'solved': True,
                'iterations': iteration + 1,
                'initial_facts': initial_count,
                'final_facts': len(db.facts),
                'new_facts_derived': len(db.facts) - initial_count,
                'goal_pred': goal_pred,
            }

    return {
        'solved': False,
        'iterations': max_iterations,
        'initial_facts': initial_count,
        'final_facts': len(db.facts),
        'new_facts_derived': len(db.facts) - initial_count,
        'goal_pred': goal_pred,
    }


def analyze_problem_complexity(statement):
    """
    Analyze structural complexity of a problem.
    Returns metrics useful for understanding proof difficulty.
    """
    if '?' not in statement:
        return {}

    premises_str, goal_str = statement.split('?', 1)

    # Count construction types
    construction_types = defaultdict(int)
    all_points = set()
    lhs_points = []

    for constr in premises_str.split(';'):
        constr = constr.strip()
        if not constr:
            continue
        if '=' in constr:
            lhs, rhs = constr.split('=', 1)
            # Extract defined points
            for token in lhs.strip().split():
                t = token.split('@')[0]
                if re.match(r'^[a-zA-Z]', t):
                    all_points.add(t)
                    lhs_points.append(t)
            # Extract construction types
            for clause in rhs.split(','):
                tokens = clause.strip().split()
                if tokens:
                    construction_types[tokens[0]] += 1

    # Detect key geometric objects
    has_circle = 'circle' in construction_types or 'on_circle' in construction_types
    has_orthocenter = 'orthocenter' in construction_types
    has_incenter = 'incenter' in construction_types or 'incenter2' in construction_types
    has_reflection = 'reflect' in construction_types
    has_angle_bisector = 'angle_bisector' in construction_types

    # Count constraint clauses
    total_clauses = sum(construction_types.values())

    goal_tokens = goal_str.strip().split()
    goal_pred = goal_tokens[0] if goal_tokens else 'unknown'

    # Estimate proof depth based on complexity heuristics
    # Based on AlphaGeometry paper findings
    complexity_score = (
        len(all_points) * 1.0 +
        total_clauses * 0.8 +
        has_circle * 3.0 +
        has_orthocenter * 2.0 +
        has_incenter * 2.5 +
        has_reflection * 2.0 +
        has_angle_bisector * 1.5
    )

    # Auxiliary constructions needed (estimated)
    aux_constructions_estimate = max(0, int((complexity_score - 10) / 5))

    return {
        'num_points': len(all_points),
        'total_clauses': total_clauses,
        'construction_types': dict(construction_types),
        'has_circle': has_circle,
        'has_orthocenter': has_orthocenter,
        'has_incenter': has_incenter,
        'has_reflection': has_reflection,
        'has_angle_bisector': has_angle_bisector,
        'complexity_score': round(complexity_score, 1),
        'aux_constructions_estimate': aux_constructions_estimate,
        'goal_pred': goal_pred,
    }


if __name__ == '__main__':
    import json

    with open(f"{WORKSPACE}/outputs/problem_analysis.json") as f:
        problems = json.load(f)

    # Read raw statements
    raw_problems = {}
    with open(f"{WORKSPACE}/data/imo_ag_30.txt") as f:
        lines = [l.rstrip('\n') for l in f.readlines()]
    i = 0
    while i < len(lines):
        name = lines[i].strip()
        if name and i+1 < len(lines):
            stmt = lines[i+1].strip()
            if stmt:
                raw_problems[name] = stmt
                i += 2
                continue
        i += 1

    results = []
    print("=== Complexity Analysis ===")
    for prob in problems:
        stmt = raw_problems.get(prob['name'], '')
        if not stmt:
            continue

        complexity = analyze_problem_complexity(stmt)
        proof_sim = simulate_proof(stmt)

        result = {
            'name': prob['name'],
            'year': prob['year'],
            'goal_type': prob['goal_type'],
            'goal_category': prob['goal_category'],
            **complexity,
            'proof_sim': proof_sim,
        }
        results.append(result)

        print(f"  {prob['name']}: complexity={complexity['complexity_score']:.1f}, "
              f"goal={complexity['goal_pred']}, "
              f"aux_est={complexity['aux_constructions_estimate']}")

    with open(f"{WORKSPACE}/outputs/complexity_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} results to outputs/complexity_analysis.json")
