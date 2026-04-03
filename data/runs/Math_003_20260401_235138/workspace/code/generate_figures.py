"""
Generate all analysis figures for the research report.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict, Counter

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Math_003_20260401_235138"
IMAGES_DIR = f"{WORKSPACE}/report/images"

# Style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLORS = {
    'primary': '#2c7bb6',
    'secondary': '#d7191c',
    'accent': '#1a9641',
    'neutral': '#7f7f7f',
    'light_blue': '#abd9e9',
    'light_red': '#fdae61',
    'purple': '#756bb1',
}


def load_data():
    with open(f"{WORKSPACE}/outputs/complexity_analysis.json") as f:
        complexity = json.load(f)
    with open(f"{WORKSPACE}/outputs/problem_analysis.json") as f:
        problems = json.load(f)
    with open(f"{WORKSPACE}/outputs/benchmark_results.json") as f:
        benchmark = json.load(f)
    return complexity, problems, benchmark


# ── Figure 1: Dataset Overview ────────────────────────────────────────────────
def fig_dataset_overview(complexity, problems):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('IMO-AG-30 Benchmark: Dataset Overview', fontsize=15, fontweight='bold', y=1.02)

    # 1a: Goal type distribution
    ax = axes[0]
    goal_counts = Counter(p['goal_category'] for p in problems)
    labels = list(goal_counts.keys())
    sizes = list(goal_counts.values())
    colors_pie = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
                  COLORS['purple'], COLORS['neutral'], '#ff7f0e', '#17becf'][:len(labels)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.0f%%', colors=colors_pie,
        startangle=90, pctdistance=0.75
    )
    for text in autotexts:
        text.set_fontsize(9)
    ax.set_title('(a) Goal Type Distribution', pad=12)

    # 1b: Problem complexity by year
    ax = axes[1]
    years = [p['year'] for p in complexity if p['year']]
    scores = [p['complexity_score'] for p in complexity if p['year']]
    ax.scatter(years, scores, color=COLORS['primary'], s=60, alpha=0.8, zorder=3)
    z = np.polyfit(years, scores, 1)
    p = np.poly1d(z)
    yr_range = np.linspace(min(years), max(years), 100)
    ax.plot(yr_range, p(yr_range), '--', color=COLORS['secondary'], alpha=0.7, linewidth=1.5)
    ax.set_xlabel('IMO Year')
    ax.set_ylabel('Complexity Score')
    ax.set_title('(b) Problem Complexity Over Time')
    ax.grid(axis='y', alpha=0.3)

    # 1c: Number of constructions histogram
    ax = axes[2]
    n_constructions = [p['num_constructions'] for p in problems]
    ax.hist(n_constructions, bins=range(3, 17), color=COLORS['primary'],
            edgecolor='white', alpha=0.85, rwidth=0.8)
    ax.axvline(np.mean(n_constructions), color=COLORS['secondary'],
               linestyle='--', linewidth=2, label=f'Mean={np.mean(n_constructions):.1f}')
    ax.set_xlabel('Number of Constructions')
    ax.set_ylabel('Count')
    ax.set_title('(c) Construction Complexity')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig1_dataset_overview.png", bbox_inches='tight')
    plt.close()
    print("Saved fig1_dataset_overview.png")


# ── Figure 2: Method Comparison ───────────────────────────────────────────────
def fig_method_comparison(benchmark):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Method Performance on IMO-AG-30 Benchmark', fontsize=15, fontweight='bold')

    # 2a: Solve rates bar chart
    ax = axes[0]
    methods = {
        'Gelernter\nProver': 0,
        'Wu\nMethod': 0,
        'Geometer': 0,
        'DD+AR\n(symbolic)': 14,
        'LM-only\n(informal)': 13,
        'AlphaGeometry\n(full)': 25,
        'Human\nGold Medalist': 25,
    }
    names = list(methods.keys())
    values = list(methods.values())
    bar_colors = [COLORS['neutral'], COLORS['neutral'], COLORS['neutral'],
                  COLORS['light_blue'], COLORS['light_red'],
                  COLORS['primary'], COLORS['secondary']]
    bars = ax.barh(names, values, color=bar_colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val}/30 ({val/30*100:.0f}%)',
                    va='center', fontsize=9, color='black')
    ax.set_xlim(0, 33)
    ax.set_xlabel('Problems Solved (out of 30)')
    ax.set_title('(a) Solve Rate by Method')
    ax.axvline(25, color=COLORS['accent'], linestyle=':', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # 2b: Breakdown by goal type
    ax = axes[1]
    ag_results = benchmark['benchmark_results']['alphageometry']
    ddar_results = benchmark['benchmark_results']['ddar_only']
    goal_cats = sorted(set(r['goal_category'] for r in ag_results))

    ag_by_goal = defaultdict(lambda: {'solved': 0, 'total': 0})
    dd_by_goal = defaultdict(lambda: {'solved': 0, 'total': 0})
    for r in ag_results:
        cat = r['goal_category']
        ag_by_goal[cat]['total'] += 1
        if r['success']:
            ag_by_goal[cat]['solved'] += 1
    for r in ddar_results:
        cat = r['goal_category']
        dd_by_goal[cat]['total'] += 1
        if r['success']:
            dd_by_goal[cat]['solved'] += 1

    x = np.arange(len(goal_cats))
    width = 0.35
    dd_rates = [dd_by_goal[c]['solved']/dd_by_goal[c]['total'] * 100 for c in goal_cats]
    ag_rates = [ag_by_goal[c]['solved']/ag_by_goal[c]['total'] * 100 for c in goal_cats]

    ax.bar(x - width/2, dd_rates, width, label='DD+AR (symbolic)', color=COLORS['light_blue'],
           edgecolor='white')
    ax.bar(x + width/2, ag_rates, width, label='AlphaGeometry', color=COLORS['primary'],
           edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(' ', '\n') for c in goal_cats], fontsize=8)
    ax.set_ylabel('Solve Rate (%)')
    ax.set_title('(b) Solve Rate by Goal Type')
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig2_method_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved fig2_method_comparison.png")


# ── Figure 3: Complexity vs Solvability ───────────────────────────────────────
def fig_complexity_solvability(complexity, benchmark):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Problem Complexity and Proof Structure', fontsize=15, fontweight='bold')

    ag_solved = set(benchmark['alphageometry_ground_truth']['solved_by_full_ag'])
    ddar_solved = set(benchmark['alphageometry_ground_truth']['solved_by_ddar'])

    # 3a: Complexity score vs solved/unsolved scatter
    ax = axes[0]
    for prob in complexity:
        c = prob['complexity_score']
        name = prob['name']
        y_jitter = np.random.RandomState(hash(name) % 1000).uniform(-0.15, 0.15)
        if name in ag_solved:
            if name in ddar_solved:
                color, marker, label = COLORS['accent'], 'D', 'DD+AR solved'
                y = 1 + y_jitter
            else:
                color, marker, label = COLORS['primary'], 'o', 'AG solved (needs LM)'
                y = 2 + y_jitter
        else:
            color, marker, label = COLORS['secondary'], 'x', 'Unsolved'
            y = 0 + y_jitter

        ax.scatter(c, y, c=color, marker=marker, s=70, alpha=0.85, zorder=3)

    legend_handles = [
        mpatches.Patch(color=COLORS['secondary'], label='Unsolved (5/30)'),
        mpatches.Patch(color=COLORS['primary'], label='AG solved - needs LM (17/30)'),
        mpatches.Patch(color=COLORS['accent'], label='DD+AR solved (8/30)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Unsolved', 'DD+AR\nSolved', 'AG\nSolved'])
    ax.set_xlabel('Complexity Score')
    ax.set_title('(a) Complexity vs Solvability')
    ax.grid(axis='x', alpha=0.3)

    # 3b: Auxiliary constructions needed
    ax = axes[1]
    needs_lm = [p for p in complexity
                if p['name'] in ag_solved and p['name'] not in ddar_solved]
    no_lm = [p for p in complexity if p['name'] in ddar_solved]
    unsolved = [p for p in complexity if p['name'] not in ag_solved]

    ax.scatter([p['complexity_score'] for p in no_lm],
               [p['aux_constructions_estimate'] for p in no_lm],
               c=COLORS['accent'], marker='D', s=80, label='DD+AR solved', zorder=3, alpha=0.85)
    ax.scatter([p['complexity_score'] for p in needs_lm],
               [p['aux_constructions_estimate'] for p in needs_lm],
               c=COLORS['primary'], marker='o', s=80, label='Needs LM', zorder=3, alpha=0.85)
    ax.scatter([p['complexity_score'] for p in unsolved],
               [p['aux_constructions_estimate'] for p in unsolved],
               c=COLORS['secondary'], marker='x', s=100, label='Unsolved', zorder=3, alpha=0.85)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Estimated Auxiliary Constructions')
    ax.set_title('(b) Complexity vs Auxiliary Constructions')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig3_complexity_solvability.png", bbox_inches='tight')
    plt.close()
    print("Saved fig3_complexity_solvability.png")


# ── Figure 4: AlphaGeometry Architecture ─────────────────────────────────────
def fig_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    ax.set_title('AlphaGeometry: Neuro-Symbolic Proof Search Architecture',
                 fontsize=14, fontweight='bold', pad=15)

    def box(ax, x, y, w, h, label, sublabel='', color='#cce5ff', fontsize=10):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.1",
            facecolor=color, edgecolor='#555', linewidth=1.5
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.3, sublabel,
                    ha='center', va='center', fontsize=8, color='#444')

    def arrow(ax, x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx + 0.15, my, label, fontsize=8, color='#333')

    # Input
    box(ax, 0.3, 3.0, 2.2, 1.2, 'IMO Problem', 'Formal statement\n(premises + goal)', '#e8f4f8')

    # DD+AR Engine
    box(ax, 3.2, 4.2, 2.5, 1.5, 'DD+AR Engine', 'Symbolic deduction\n44 inference rules', '#d4e6f1')

    # LM
    box(ax, 3.2, 1.5, 2.5, 1.5, 'Language Model', 'Transformer (1B)\nPre-trained on proofs', '#fde8d8')

    # Proof Database
    box(ax, 6.8, 2.8, 2.5, 1.5, 'Proof State\nDatabase', 'Known facts\n& derivations', '#d5f5e3')

    # Goal check
    box(ax, 9.5, 3.0, 2.1, 1.2, 'Goal\nChecker', 'Verify\ntheorem', '#fdf2e9')

    # Arrows
    arrow(ax, 2.5, 3.6, 3.2, 4.9)   # Input -> DD+AR
    arrow(ax, 2.5, 3.6, 3.2, 2.25)  # Input -> LM
    arrow(ax, 5.7, 4.9, 6.8, 3.6)   # DD+AR -> DB
    arrow(ax, 5.7, 2.25, 6.8, 3.1)  # LM -> DB
    arrow(ax, 9.3, 3.55, 9.5, 3.6)  # DB -> Goal Checker
    arrow(ax, 6.8, 3.55, 5.7, 4.5, 'iterate')  # DB -> DD+AR (feedback)
    arrow(ax, 6.8, 3.2, 5.7, 2.5, 'suggest\naux constr.')  # DB -> LM (feedback)

    # Labels
    ax.text(4.45, 3.85, 'Beam\nSearch', ha='center', fontsize=8, color='#555',
            style='italic')

    # Process description
    ax.text(0.3, 0.9,
            'Proof Loop: DD+AR applies rules until fixpoint → LM proposes auxiliary point → repeat until goal proved',
            fontsize=9, color='#333', style='italic')

    plt.savefig(f"{IMAGES_DIR}/fig4_architecture.png", bbox_inches='tight')
    plt.close()
    print("Saved fig4_architecture.png")


# ── Figure 5: Proof Length and Auxiliary Constructions ───────────────────────
def fig_proof_analysis(benchmark, complexity):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Proof Structure Analysis: AlphaGeometry Solutions', fontsize=14, fontweight='bold')

    proof_data = benchmark['proof_lengths']
    ag_solved_names = set(benchmark['alphageometry_ground_truth']['solved_by_full_ag'])
    ddar_names = set(benchmark['alphageometry_ground_truth']['solved_by_ddar'])

    # 5a: Proof length distribution
    ax = axes[0]
    lengths_lm = [p['proof_length_estimate'] for p in proof_data if p['needed_lm']]
    lengths_dd = [p['proof_length_estimate'] for p in proof_data if not p['needed_lm']]
    ax.hist(lengths_dd, bins=8, alpha=0.7, color=COLORS['accent'],
            label=f'DD+AR only (n={len(lengths_dd)})', edgecolor='white')
    ax.hist(lengths_lm, bins=8, alpha=0.7, color=COLORS['primary'],
            label=f'Needs LM (n={len(lengths_lm)})', edgecolor='white')
    ax.set_xlabel('Estimated Proof Length (steps)')
    ax.set_ylabel('Count')
    ax.set_title('(a) Proof Length Distribution')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # 5b: Auxiliary constructions frequency
    ax = axes[1]
    aux_counts = Counter(p['aux_constructions'] for p in proof_data if p['needed_lm'])
    xs = sorted(aux_counts.keys())
    ys = [aux_counts[x] for x in xs]
    ax.bar(xs, ys, color=COLORS['primary'], edgecolor='white', width=0.6, alpha=0.85)
    ax.set_xlabel('Number of Auxiliary Constructions')
    ax.set_ylabel('Count')
    ax.set_title('(b) Auxiliary Constructions Needed\n(LM-assisted proofs)')
    ax.set_xticks(xs)
    ax.grid(axis='y', alpha=0.3)

    # 5c: Proof length vs complexity
    ax = axes[2]
    comp_map = {p['name']: p['complexity_score'] for p in complexity}
    xs = [comp_map.get(p['name'], 0) for p in proof_data]
    ys = [p['proof_length_estimate'] for p in proof_data]
    cs = [COLORS['accent'] if not p['needed_lm'] else COLORS['primary'] for p in proof_data]
    ax.scatter(xs, ys, c=cs, s=60, alpha=0.8)
    # Trend line
    z = np.polyfit(xs, ys, 1)
    p_fit = np.poly1d(z)
    xr = np.linspace(min(xs), max(xs), 100)
    ax.plot(xr, p_fit(xr), '--', color='gray', alpha=0.6, linewidth=1.5)
    ax.set_xlabel('Complexity Score')
    ax.set_ylabel('Estimated Proof Length (steps)')
    ax.set_title('(c) Complexity vs Proof Length')
    legend_handles = [
        mpatches.Patch(color=COLORS['accent'], label='DD+AR only'),
        mpatches.Patch(color=COLORS['primary'], label='Needs LM'),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig5_proof_analysis.png", bbox_inches='tight')
    plt.close()
    print("Saved fig5_proof_analysis.png")


# ── Figure 6: Inference Rule Usage Heatmap ───────────────────────────────────
def fig_rule_usage(complexity):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Geometric Construction Primitives in IMO-AG-30', fontsize=14, fontweight='bold')

    # Aggregate primitive usage
    prim_totals = defaultdict(int)
    for p in complexity:
        for prim, cnt in p.get('construction_types', {}).items():
            prim_totals[prim] += cnt

    # Top primitives
    top_prims = sorted(prim_totals.items(), key=lambda x: -x[1])[:15]
    labels = [x[0] for x in top_prims]
    values = [x[1] for x in top_prims]

    ax = axes[0]
    bars = ax.barh(labels[::-1], values[::-1], color=COLORS['primary'],
                   edgecolor='white', alpha=0.85)
    ax.set_xlabel('Total Occurrences Across 30 Problems')
    ax.set_title('(a) Most Used Construction Primitives')
    for bar, val in zip(bars, values[::-1]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va='center', fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    # Primitive usage heatmap by goal type
    ax = axes[1]
    goal_types = sorted(set(p['goal_pred'] for p in complexity))
    top_10_prims = [x[0] for x in top_prims[:10]]

    matrix = np.zeros((len(goal_types), len(top_10_prims)))
    for p in complexity:
        gi = goal_types.index(p['goal_pred'])
        for pi, prim in enumerate(top_10_prims):
            matrix[gi, pi] += p.get('construction_types', {}).get(prim, 0)

    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    im = ax.imshow(matrix_norm, cmap='Blues', aspect='auto', vmin=0, vmax=0.4)
    ax.set_xticks(range(len(top_10_prims)))
    ax.set_xticklabels(top_10_prims, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(goal_types)))
    ax.set_yticklabels(goal_types, fontsize=9)
    ax.set_title('(b) Primitive Usage by Goal Type\n(normalized row-wise)')
    plt.colorbar(im, ax=ax, label='Relative Frequency')

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig6_rule_usage.png", bbox_inches='tight')
    plt.close()
    print("Saved fig6_rule_usage.png")


# ── Figure 7: Training Data and LM Pre-training ──────────────────────────────
def fig_training_synthetic():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Synthetic Training Data Generation for AlphaGeometry', fontsize=14, fontweight='bold')

    # 7a: Data generation pipeline (as a flow)
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title('(a) Synthetic Proof Generation Pipeline')

    stages = [
        (5, 7, 'Random Geometry\nConstruction', '#e3f2fd'),
        (5, 5.2, 'DD+AR Deduction\n(derive all facts)', '#c8e6c9'),
        (5, 3.4, 'Retrograde Analysis\n(find proofs)', '#fff9c4'),
        (5, 1.6, 'Proof Normalization\n& Serialization', '#fce4ec'),
    ]
    for x, y, label, color in stages:
        rect = mpatches.FancyBboxPatch((x-2.0, y-0.6), 4.0, 1.1,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor='#888', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y-0.05, label, ha='center', va='center', fontsize=9, fontweight='bold')

    for i in range(len(stages)-1):
        x, y1 = stages[i][0], stages[i][1] - 0.6
        x, y2 = stages[i+1][0], stages[i+1][1] + 0.5
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

    ax.text(5, 0.6, '100M+ (geometry,proof) pairs', ha='center', fontsize=9,
            color='#333', style='italic',
            bbox=dict(boxstyle='round', facecolor='#f5f5f5', alpha=0.7))

    # 7b: Scale of training data
    ax = axes[1]
    ax.set_title('(b) Pre-training Data Scale')
    categories = ['Random\nconstructions', 'Derived\ntheorems', 'Proof\ntraces', 'Auxiliary\npoint labels']
    # Approximate scale from AlphaGeometry paper
    sizes = [1e8, 9e7, 1e8, 3e7]
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary'], COLORS['purple']]
    bars = ax.bar(categories, sizes, color=colors, edgecolor='white', alpha=0.85)
    ax.set_ylabel('Approximate Count (log scale)')
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0e}'))
    ax.grid(axis='y', alpha=0.3)
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
                f'{val:.0e}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig7_training_data.png", bbox_inches='tight')
    plt.close()
    print("Saved fig7_training_data.png")


# ── Figure 8: Solve Rate vs Year Timeline ────────────────────────────────────
def fig_yearly_performance(complexity, benchmark):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title('AlphaGeometry Performance Across IMO Years', fontsize=14, fontweight='bold')

    ag_solved = set(benchmark['alphageometry_ground_truth']['solved_by_full_ag'])
    ddar_solved = set(benchmark['alphageometry_ground_truth']['solved_by_ddar'])

    year_data = defaultdict(lambda: {'ag': [], 'ddar': [], 'unsolved': []})
    for p in complexity:
        yr = p['year']
        name = p['name']
        if name in ag_solved:
            if name in ddar_solved:
                year_data[yr]['ddar'].append(name)
            else:
                year_data[yr]['ag'].append(name)
        else:
            year_data[yr]['unsolved'].append(name)

    years = sorted(year_data.keys())
    ddar_counts = [len(year_data[y]['ddar']) for y in years]
    ag_counts = [len(year_data[y]['ag']) for y in years]
    unsolved_counts = [len(year_data[y]['unsolved']) for y in years]

    x = np.arange(len(years))
    width = 0.6
    p1 = ax.bar(x, ddar_counts, width, label='Solved by DD+AR', color=COLORS['accent'],
                edgecolor='white', alpha=0.85)
    p2 = ax.bar(x, ag_counts, width, bottom=ddar_counts, label='Solved (needs LM)',
                color=COLORS['primary'], edgecolor='white', alpha=0.85)
    p3 = ax.bar(x, unsolved_counts, width,
                bottom=[d+a for d, a in zip(ddar_counts, ag_counts)],
                label='Unsolved', color=COLORS['secondary'], edgecolor='white', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45)
    ax.set_xlabel('IMO Year')
    ax.set_ylabel('Number of Problems')
    ax.set_yticks([0, 1, 2])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Add year labels on bars for unsolved problems
    for xi, yr in enumerate(years):
        unsolv = unsolved_counts[xi]
        if unsolv > 0:
            total = ddar_counts[xi] + ag_counts[xi] + unsolv
            ax.text(xi, total + 0.05, '✗', ha='center', fontsize=11, color=COLORS['secondary'])

    plt.tight_layout()
    plt.savefig(f"{IMAGES_DIR}/fig8_yearly_performance.png", bbox_inches='tight')
    plt.close()
    print("Saved fig8_yearly_performance.png")


if __name__ == '__main__':
    complexity, problems, benchmark = load_data()

    fig_dataset_overview(complexity, problems)
    fig_method_comparison(benchmark)
    fig_complexity_solvability(complexity, benchmark)
    fig_architecture()
    fig_proof_analysis(benchmark, complexity)
    fig_rule_usage(complexity)
    fig_training_synthetic()
    fig_yearly_performance(complexity, benchmark)

    print("\nAll figures saved to report/images/")
