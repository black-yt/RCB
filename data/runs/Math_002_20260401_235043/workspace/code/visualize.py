"""
Comprehensive visualization for the MAPF Hybrid MARL-LNS research.
Generates all figures for the research report.
"""

import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patches as mpatches
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_map

WORKSPACE = os.path.join(os.path.dirname(__file__), '..')
OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs')
IMG_DIR = os.path.join(WORKSPACE, 'report', 'images')
os.makedirs(IMG_DIR, exist_ok=True)

# Color scheme
COLORS = {
    'PP': '#E74C3C',
    'LNS-PP': '#E67E22',
    'MARL-Inspired': '#3498DB',
    'Hybrid-MARL-LNS': '#27AE60'
}
MARKERS = {'PP': 'o', 'LNS-PP': 's', 'MARL-Inspired': '^', 'Hybrid-MARL-LNS': 'D'}
LINESTYLES = {'PP': '--', 'LNS-PP': '-.', 'MARL-Inspired': ':', 'Hybrid-MARL-LNS': '-'}
ALGORITHMS = ['PP', 'LNS-PP', 'MARL-Inspired', 'Hybrid-MARL-LNS']


def load_results():
    path = os.path.join(OUTPUT_DIR, 'experiment_results.json')
    with open(path) as f:
        return json.load(f)


def fig1_map_gallery():
    """Figure 1: Gallery of map types used in experiments."""
    DATA_DIR = os.path.join(WORKSPACE, 'data')

    map_configs = [
        ('Random Small\n(10×10, 17.5% obstacles)', 'maps_60_10_10_0.175/eval_map_1.npy'),
        ('Random Medium\n(25×25, 17.5% obstacles)', 'random_medium/maps_312_25_25_0.175/eval_map_1.npy'),
        ('Random Large\n(50×50, 17.5% obstacles)', 'random_large/maps_1250_50_50_0.175/eval_map_1.npy'),
        ('Empty\n(25×25, no obstacles)', 'empty/empty_maps_453_25_25/eval_map_empty_1.npy'),
        ('Maze\n(25×25, complex corridors)', 'maze/maze_maps_125_25_25/eval_map_maze_1.npy'),
        ('Room\n(25×25, chambers)', 'room/room_maps_250_25_25/eval_map_room_1.npy'),
        ('Warehouse\n(25×25, shelf layout)', 'warehouse/warehouse_maps_266_25_25/eval_map_warehouse_1.npy'),
    ]

    fig, axes = plt.subplots(1, 7, figsize=(21, 3.5))
    fig.suptitle('MAPF Benchmark Map Categories', fontsize=14, fontweight='bold', y=1.02)

    for ax, (title, rel_path) in zip(axes, map_configs):
        grid = load_map(os.path.join(DATA_DIR, rel_path))
        # Color: white=free, dark=obstacle
        img = np.where(grid == -1, 0, 1)
        ax.imshow(img, cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=9, pad=4)
        ax.axis('off')

        # Add obstacle density
        obs_density = (grid == -1).sum() / grid.size
        ax.text(0.5, -0.05, f'{obs_density:.0%} obstacles',
                transform=ax.transAxes, ha='center', fontsize=8, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig1_map_gallery.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_map_gallery.png")


def fig2_overall_success_rate(results):
    """Figure 2: Overall success rates by algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of overall success rate
    ax = axes[0]
    algo_sr = {}
    algo_conflicts = {}
    for algo in ALGORITHMS:
        algo_res = [r for r in results if r['algorithm'] == algo]
        algo_sr[algo] = sum(1 for r in algo_res if r['success']) / len(algo_res) * 100
        algo_conflicts[algo] = np.mean([r['conflicts'] for r in algo_res])

    bars = ax.bar(range(len(ALGORITHMS)), [algo_sr[a] for a in ALGORITHMS],
                   color=[COLORS[a] for a in ALGORITHMS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(ALGORITHMS)))
    ax.set_xticklabels(['PP', 'LNS-PP', 'MARL\nInspired', 'Hybrid\nMARL-LNS'], fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Overall Success Rate Across All Benchmarks', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    for bar, algo in zip(bars, ALGORITHMS):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add improvement annotation
    hybrid_sr = algo_sr['Hybrid-MARL-LNS']
    pp_sr = algo_sr['PP']
    ax.annotate(f'+{hybrid_sr - pp_sr:.1f}%\nvs PP',
                xy=(3, hybrid_sr), xytext=(2.2, hybrid_sr + 10),
                fontsize=9, color='#27AE60',
                arrowprops=dict(arrowstyle='->', color='#27AE60'))

    # Right: Average conflicts remaining
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(ALGORITHMS)), [algo_conflicts[a] for a in ALGORITHMS],
                     color=[COLORS[a] for a in ALGORITHMS], alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(len(ALGORITHMS)))
    ax2.set_xticklabels(['PP', 'LNS-PP', 'MARL\nInspired', 'Hybrid\nMARL-LNS'], fontsize=11)
    ax2.set_ylabel('Average Remaining Conflicts', fontsize=12)
    ax2.set_title('Average Conflicts in Final Solution', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    for bar, algo in zip(bars2, ALGORITHMS):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                 f'{h:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig2_overall_success.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_overall_success.png")


def fig3_success_by_category(results):
    """Figure 3: Success rate by map category and algorithm."""
    categories = ['random_small', 'random_medium', 'random_large', 'empty', 'maze', 'room', 'warehouse']
    cat_labels = ['Random\nSmall', 'Random\nMedium', 'Random\nLarge', 'Empty', 'Maze', 'Room', 'Warehouse']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(categories))
    width = 0.2

    for i, algo in enumerate(ALGORITHMS):
        sr_vals = []
        for cat in categories:
            cat_res = [r for r in results if r['category'] == cat and r['algorithm'] == algo]
            if cat_res:
                sr = sum(1 for r in cat_res if r['success']) / len(cat_res) * 100
            else:
                sr = 0
            sr_vals.append(sr)

        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, sr_vals, width, label=algo,
                       color=COLORS[algo], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Success Rate by Map Category and Algorithm', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig3_success_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_success_by_category.png")


def fig4_success_vs_agents(results):
    """Figure 4: Success rate vs number of agents (scalability analysis)."""
    categories_to_show = {
        'random_small': ('Random Small (10×10)', [3, 5, 8, 10]),
        'random_medium': ('Random Medium (25×25)', [5, 10, 15, 20]),
        'empty': ('Empty (25×25)', [10, 20, 30]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (cat, (title, agent_counts)) in zip(axes, categories_to_show.items()):
        for algo in ALGORITHMS:
            sr_vals = []
            for n in agent_counts:
                cat_res = [r for r in results
                           if r['category'] == cat
                           and r['algorithm'] == algo
                           and r['n_agents'] == n]
                if cat_res:
                    sr = sum(1 for r in cat_res if r['success']) / len(cat_res) * 100
                else:
                    sr = 0
                sr_vals.append(sr)

            ax.plot(agent_counts, sr_vals, marker=MARKERS[algo],
                    color=COLORS[algo], linestyle=LINESTYLES[algo],
                    linewidth=2, markersize=8, label=algo)

        ax.set_xlabel('Number of Agents', fontsize=11)
        ax.set_ylabel('Success Rate (%)' if ax == axes[0] else '', fontsize=11)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(-5, 110)
        ax.set_xticks(agent_counts)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    plt.suptitle('Scalability Analysis: Success Rate vs. Agent Count', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig4_scalability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_scalability.png")


def fig5_computation_time(results):
    """Figure 5: Computation time analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Box plot of computation times
    ax = axes[0]
    time_data = []
    for algo in ALGORITHMS:
        times = [r['time'] for r in results if r['algorithm'] == algo]
        time_data.append(times)

    bp = ax.boxplot(time_data, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2})

    for patch, algo in zip(bp['boxes'], ALGORITHMS):
        patch.set_facecolor(COLORS[algo])
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(['PP', 'LNS-PP', 'MARL\nInspired', 'Hybrid\nMARL-LNS'], fontsize=11)
    ax.set_ylabel('Computation Time (s)', fontsize=12)
    ax.set_title('Computation Time Distribution', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.set_yscale('log')

    # Right: Time vs success rate scatter
    ax2 = axes[1]
    for algo in ALGORITHMS:
        algo_res = [r for r in results if r['algorithm'] == algo]
        avg_time = np.mean([r['time'] for r in algo_res])
        sr = sum(1 for r in algo_res if r['success']) / len(algo_res) * 100
        ax2.scatter(avg_time, sr, s=200, color=COLORS[algo], marker=MARKERS[algo],
                    label=algo, zorder=5, edgecolors='black', linewidths=1)

    ax2.set_xlabel('Average Computation Time (s)', fontsize=12)
    ax2.set_ylabel('Overall Success Rate (%)', fontsize=12)
    ax2.set_title('Quality-Efficiency Trade-off', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig5_computation_time.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_computation_time.png")


def fig6_conflict_reduction(results):
    """Figure 6: Analysis of conflict reduction - initial vs final conflicts."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Initial vs final conflicts for LNS-PP and Hybrid
    ax = axes[0]

    for algo in ['LNS-PP', 'MARL-Inspired', 'Hybrid-MARL-LNS']:
        algo_res = [r for r in results if r['algorithm'] == algo and 'initial_conflicts' in r]
        if algo_res:
            init_vals = [r['initial_conflicts'] for r in algo_res]
            final_vals = [r['conflicts'] for r in algo_res]
            reduction_pct = [(i - f) / max(i, 1) * 100 for i, f in zip(init_vals, final_vals) if i > 0]
            if reduction_pct:
                ax.hist(reduction_pct, bins=20, alpha=0.6, color=COLORS[algo],
                        label=f'{algo} (median={np.median(reduction_pct):.0f}%)',
                        density=True)

    ax.set_xlabel('Conflict Reduction (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Conflict Reduction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)

    # Right: Conflicts by category for all algorithms
    ax2 = axes[1]
    categories = ['random_small', 'random_medium', 'random_large', 'empty', 'maze', 'room', 'warehouse']
    cat_labels = ['R-Small', 'R-Med', 'R-Large', 'Empty', 'Maze', 'Room', 'Whouse']

    x = np.arange(len(categories))
    width = 0.2

    for i, algo in enumerate(ALGORITHMS):
        conflict_vals = []
        for cat in categories:
            cat_res = [r for r in results if r['category'] == cat and r['algorithm'] == algo]
            if cat_res:
                avg_c = np.mean([r['conflicts'] for r in cat_res])
            else:
                avg_c = 0
            conflict_vals.append(avg_c)

        offset = (i - 1.5) * width
        ax2.bar(x + offset, conflict_vals, width, label=algo,
                color=COLORS[algo], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cat_labels, fontsize=10)
    ax2.set_ylabel('Avg. Remaining Conflicts', fontsize=12)
    ax2.set_title('Average Remaining Conflicts by Category', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig6_conflict_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig6_conflict_analysis.png")


def fig7_path_quality(results):
    """Figure 7: Solution quality (path cost and makespan)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Average solution cost by algorithm (only successful instances)
    ax = axes[0]
    cost_data = {}
    makespan_data = {}
    for algo in ALGORITHMS:
        successful = [r for r in results if r['algorithm'] == algo
                      and r['success'] and r['cost'] > 0]
        if successful:
            cost_data[algo] = np.mean([r['cost'] for r in successful])
            makespan_data[algo] = np.mean([r['makespan'] for r in successful])
        else:
            cost_data[algo] = 0
            makespan_data[algo] = 0

    x = np.arange(len(ALGORITHMS))
    width = 0.35

    bars1 = ax.bar(x - width/2, [cost_data[a] for a in ALGORITHMS], width,
                    label='Sum of Path Costs', color=[COLORS[a] for a in ALGORITHMS],
                    alpha=0.7, edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, [makespan_data[a] for a in ALGORITHMS], width,
                    label='Makespan', color=[COLORS[a] for a in ALGORITHMS],
                    alpha=1.0, edgecolor='black', linewidth=0.5, hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(['PP', 'LNS-PP', 'MARL\nInspired', 'Hybrid\nMARL-LNS'], fontsize=11)
    ax.set_ylabel('Timesteps', fontsize=12)
    ax.set_title('Solution Quality (Successful Instances Only)', fontsize=12, fontweight='bold')
    legend_elements = [mpatches.Patch(facecolor='gray', alpha=0.7, label='Sum of Path Costs'),
                        mpatches.Patch(facecolor='gray', alpha=1.0, hatch='//', label='Makespan')]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Right: Success rate improvement over PP baseline
    ax2 = axes[1]
    pp_sr_by_cat = {}
    categories = ['random_small', 'random_medium', 'random_large', 'empty', 'maze', 'room', 'warehouse']
    cat_labels = ['R-Small', 'R-Med', 'R-Large', 'Empty', 'Maze', 'Room', 'Whouse']

    for cat in categories:
        pp_res = [r for r in results if r['category'] == cat and r['algorithm'] == 'PP']
        pp_sr_by_cat[cat] = sum(1 for r in pp_res if r['success']) / len(pp_res) * 100 if pp_res else 0

    for algo in ['LNS-PP', 'MARL-Inspired', 'Hybrid-MARL-LNS']:
        improvements = []
        for cat in categories:
            cat_res = [r for r in results if r['category'] == cat and r['algorithm'] == algo]
            algo_sr = sum(1 for r in cat_res if r['success']) / len(cat_res) * 100 if cat_res else 0
            improvements.append(algo_sr - pp_sr_by_cat[cat])
        x_pos = np.arange(len(categories))
        ax2.bar(x_pos + (ALGORITHMS.index(algo) - 2) * 0.2, improvements, 0.2,
                label=algo, color=COLORS[algo], alpha=0.85, edgecolor='black', linewidth=0.5)

    ax2.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
    ax2.set_xticks(np.arange(len(categories)))
    ax2.set_xticklabels(cat_labels, fontsize=10)
    ax2.set_ylabel('Success Rate Improvement over PP (%)', fontsize=12)
    ax2.set_title('Improvement over PP Baseline', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig7_solution_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig7_solution_quality.png")


def fig8_algorithm_diagram():
    """Figure 8: Algorithm flowchart/diagram."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')

    ax.set_title('Hybrid MARL-LNS Algorithm Architecture', fontsize=14, fontweight='bold', pad=20)

    def draw_box(x, y, w, h, text, color, fontsize=10, style='round,pad=0.3'):
        ax.add_patch(Rectangle((x - w/2, y - h/2), w, h,
                                linewidth=1.5, edgecolor='black',
                                facecolor=color, alpha=0.85,
                                joinstyle='round'))
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight='bold', wrap=True)

    def draw_arrow(x1, y1, x2, y2, label=''):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.1, my, label, ha='left', va='center', fontsize=9, color='#666')

    # Input
    draw_box(6, 6.3, 4, 0.8, 'MAPF Instance\n(Map + Agents + Goals)', '#ECF0F1', fontsize=9)
    draw_arrow(6, 5.9, 6, 5.5)

    # Phase 1 header
    ax.add_patch(Rectangle((1.0, 3.4), 4.8, 2.2,
                            linewidth=2, edgecolor='#3498DB',
                            facecolor='#EBF5FB', alpha=0.4, linestyle='--'))
    ax.text(3.4, 5.45, 'Phase 1: MARL-Inspired Cooperative Planning',
            ha='center', va='center', fontsize=9, color='#1A5276', fontstyle='italic')

    draw_box(3.4, 4.9, 4.2, 0.7, 'Independent A* Path Planning\n(No cooperation, baseline paths)', '#AED6F1', fontsize=8.5)
    draw_arrow(3.4, 4.55, 3.4, 4.15)
    draw_box(3.4, 3.85, 4.2, 0.7, 'Congestion Map Construction\n(Traffic density per cell)', '#AED6F1', fontsize=8.5)
    draw_arrow(3.4, 3.5, 3.4, 3.1)
    draw_box(3.4, 2.85, 4.2, 0.7, 'Conflict-Driven Cooperative Replanning\n(Congestion-aware A* rounds)', '#5DADE2', fontsize=8.5)

    # Phase 2 header
    ax.add_patch(Rectangle((6.2, 3.4), 4.8, 2.2,
                            linewidth=2, edgecolor='#E67E22',
                            facecolor='#FEF9E7', alpha=0.4, linestyle='--'))
    ax.text(8.6, 5.45, 'Phase 2: LNS with Prioritized Planning',
            ha='center', va='center', fontsize=9, color='#7D6608', fontstyle='italic')

    draw_box(8.6, 4.9, 4.2, 0.7, 'Conflict Detection\n(Vertex & edge conflicts)', '#FAD7A0', fontsize=8.5)
    draw_arrow(8.6, 4.55, 8.6, 4.15)
    draw_box(8.6, 3.85, 4.2, 0.7, 'Neighborhood Selection\n(Conflict-based agent grouping)', '#FAD7A0', fontsize=8.5)
    draw_arrow(8.6, 3.5, 8.6, 3.1)
    draw_box(8.6, 2.85, 4.2, 0.7, 'Prioritized Replanning\n(Space-Time A* per neighborhood)', '#F0B27A', fontsize=8.5)

    # Arrows between phases
    draw_arrow(5.5, 4.9, 6.5, 4.9, '')
    ax.text(6.0, 5.05, 'MARL\ninitializes', ha='center', fontsize=8, color='#666')

    # Combined output
    draw_arrow(3.4, 2.5, 5.0, 1.9)
    draw_arrow(8.6, 2.5, 7.0, 1.9)
    draw_box(6, 1.6, 4.5, 0.7, 'Best Solution: Collision-Free Paths', '#ABEBC6', fontsize=10)
    draw_arrow(6, 1.25, 6, 0.85)
    draw_box(6, 0.6, 4.5, 0.5, 'Output: Valid MAPF Solution', '#27AE60', fontsize=9)

    # Phase labels
    ax.text(1.1, 5.55, '30% time\nbudget', ha='center', fontsize=8, color='#2980B9',
            style='italic')
    ax.text(11.0, 5.55, '70% time\nbudget', ha='center', fontsize=8, color='#E67E22',
            style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig8_algorithm_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig8_algorithm_diagram.png")


def fig9_heatmap_analysis(results):
    """Figure 9: Heatmap of success rates across categories and agent counts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, algo in zip(axes, ALGORITHMS):
        # Build matrix: categories x agent_counts
        all_agent_counts = sorted(set(r['n_agents'] for r in results))
        categories = ['random_small', 'random_medium', 'random_large', 'empty', 'maze', 'room', 'warehouse']
        cat_labels = ['R-Small', 'R-Med', 'R-Large', 'Empty', 'Maze', 'Room', 'Whouse']

        matrix = np.full((len(categories), len(all_agent_counts)), np.nan)

        for i, cat in enumerate(categories):
            for j, n_agents in enumerate(all_agent_counts):
                cat_res = [r for r in results
                           if r['category'] == cat
                           and r['algorithm'] == algo
                           and r['n_agents'] == n_agents]
                if cat_res:
                    sr = sum(1 for r in cat_res if r['success']) / len(cat_res) * 100
                    matrix[i, j] = sr

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label='Success Rate (%)')

        ax.set_xticks(range(len(all_agent_counts)))
        ax.set_xticklabels(all_agent_counts, fontsize=9)
        ax.set_yticks(range(len(categories)))
        ax.set_yticklabels(cat_labels, fontsize=9)
        ax.set_xlabel('Number of Agents', fontsize=10)
        ax.set_title(f'{algo}', fontsize=11, fontweight='bold', color=COLORS[algo])

        # Add text annotations
        for i in range(len(categories)):
            for j in range(len(all_agent_counts)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f'{matrix[i, j]:.0f}',
                            ha='center', va='center', fontsize=7,
                            color='black' if 20 < matrix[i, j] < 80 else 'white')

    plt.suptitle('Success Rate Heatmap: Category × Agent Count', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig9_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig9_heatmap.png")


def fig10_sample_solution():
    """Figure 10: Visualization of a sample MAPF solution."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_instance
    from hybrid_marl_lns import run_all_algorithms

    DATA_DIR = os.path.join(WORKSPACE, 'data')
    # Use a medium-sized instance
    instance = load_instance(
        os.path.join(DATA_DIR, 'random_medium/maps_312_25_25_0.175/eval_map_69.npy'),
        n_agents=10, seed=42)

    results = run_all_algorithms(instance, time_limit=20.0, seed=42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Sample MAPF Solution Visualization (10 agents, 25×25 map)',
                  fontsize=13, fontweight='bold')

    colors_agents = plt.cm.tab10(np.linspace(0, 1, instance.n_agents))

    for ax, (algo, title) in zip(axes.flatten(),
                                   [('PP', 'Prioritized Planning'),
                                    ('LNS-PP', 'LNS-Prioritized Planning'),
                                    ('MARL-Inspired', 'MARL-Inspired Cooperative'),
                                    ('Hybrid-MARL-LNS', 'Hybrid MARL-LNS (Proposed)')]):
        algo_result = results[algo]
        paths = algo_result['paths']
        grid = instance.grid

        # Draw map
        img = np.where(grid == -1, 0, 1)
        ax.imshow(img, cmap='Greys_r', vmin=0, vmax=1, interpolation='nearest', alpha=0.7)

        # Draw paths
        for i, path in enumerate(paths):
            if path is None:
                continue
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, '-', color=colors_agents[i], alpha=0.6, linewidth=1.5)

            # Start (circle) and goal (star)
            ax.scatter([xs[0]], [ys[0]], s=80, c=[colors_agents[i]], marker='o',
                       zorder=5, edgecolors='black', linewidths=0.5)
            ax.scatter([instance.goals[i][1]], [instance.goals[i][0]], s=100,
                       c=[colors_agents[i]], marker='*',
                       zorder=5, edgecolors='black', linewidths=0.5)

        status = 'SUCCESS' if algo_result['success'] else f'FAILED ({algo_result["conflicts"]} conflicts)'
        color = '#27AE60' if algo_result['success'] else '#E74C3C'
        ax.set_title(f'{title}\n{status}', fontsize=10, fontweight='bold', color=color)
        ax.set_xticks([])
        ax.set_yticks([])

        # Legend
        legend_elements = [mpatches.Patch(facecolor='gray', alpha=0.3, label='Obstacles'),
                            plt.Line2D([0], [0], marker='o', color='gray', linestyle='None',
                                        markersize=6, label='Start'),
                            plt.Line2D([0], [0], marker='*', color='gray', linestyle='None',
                                        markersize=8, label='Goal')]
        ax.legend(handles=legend_elements, fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig10_sample_solution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig10_sample_solution.png")


def fig11_phase_analysis(results):
    """Figure 11: Analysis of Phase 1 vs Phase 2 contribution in Hybrid."""
    hybrid_res = [r for r in results if r['algorithm'] == 'Hybrid-MARL-LNS'
                  and 'phase1_conflicts' in r]

    if not hybrid_res:
        print("No phase data available, skipping fig11")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Phase 1 conflict reduction
    ax = axes[0]
    initial = [r.get('phase1_initial_conflicts', r['phase1_conflicts']) for r in hybrid_res]
    after_phase1 = [r['phase1_conflicts'] for r in hybrid_res]
    final = [r['conflicts'] for r in hybrid_res]

    n = len(hybrid_res)
    x = np.arange(3)
    means = [np.mean(initial), np.mean(after_phase1), np.mean(final)]
    stds = [np.std(initial), np.std(after_phase1), np.std(final)]

    ax.bar(x, means, yerr=stds, color=['#E74C3C', '#3498DB', '#27AE60'],
            alpha=0.8, edgecolor='black', linewidth=0.5, capsize=5, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Initial\n(PP-like)', 'After Phase 1\n(MARL-Inspired)', 'After Phase 2\n(LNS-PP)'],
                        fontsize=10)
    ax.set_ylabel('Average Conflicts', fontsize=12)
    ax.set_title('Conflict Reduction Through Algorithm Phases', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    # Add reduction percentages
    if means[0] > 0:
        r1 = (means[0] - means[1]) / means[0] * 100
        r2 = (means[1] - means[2]) / max(means[1], 0.01) * 100
        ax.annotate(f'-{r1:.0f}%', xy=(0.5, (means[0] + means[1]) / 2),
                    xytext=(0.55, (means[0] + means[1]) / 2),
                    fontsize=11, color='#2C3E50', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2C3E50'))
        ax.annotate(f'-{r2:.0f}%', xy=(1.5, (means[1] + means[2]) / 2),
                    xytext=(1.55, (means[1] + means[2]) / 2),
                    fontsize=11, color='#2C3E50', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2C3E50'))

    # Right: Cases where each phase was decisive
    ax2 = axes[1]
    phase1_decisive = sum(1 for r in hybrid_res if r['phase1_conflicts'] == 0 and r['conflicts'] == 0)
    phase2_decisive = sum(1 for r in hybrid_res if r['phase1_conflicts'] > 0 and r['conflicts'] == 0)
    both_failed = sum(1 for r in hybrid_res if r['conflicts'] > 0)
    pp_baseline_success = sum(1 for r in results if r['algorithm'] == 'PP' and r['success'])
    total_pp = sum(1 for r in results if r['algorithm'] == 'PP')

    labels = ['Phase 1\nalone solved', 'Phase 2\nfixed remaining', 'Both phases\nfailed',
               'PP baseline\nfailed']
    values = [phase1_decisive, phase2_decisive, both_failed,
               total_pp - pp_baseline_success]
    colors_pie = ['#27AE60', '#3498DB', '#E74C3C', '#E67E22']

    bars = ax2.bar(range(4), values, color=colors_pie, alpha=0.85, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel('Number of Instances', fontsize=12)
    ax2.set_title('Solution Contribution by Algorithm Component', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, alpha=0.4)
    ax2.set_axisbelow(True)

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.5, str(int(h)),
                  ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, 'fig11_phase_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig11_phase_analysis.png")


def generate_summary_stats(results):
    """Generate and save summary statistics."""
    summary = {}
    for algo in ALGORITHMS:
        algo_res = [r for r in results if r['algorithm'] == algo]
        successful = [r for r in algo_res if r['success']]
        summary[algo] = {
            'total_instances': len(algo_res),
            'successful': len(successful),
            'success_rate': len(successful) / len(algo_res) if algo_res else 0,
            'avg_conflicts': np.mean([r['conflicts'] for r in algo_res]),
            'avg_time': np.mean([r['time'] for r in algo_res]),
            'avg_cost_success': np.mean([r['cost'] for r in successful if r['cost'] > 0]) if successful else 0,
        }

    with open(os.path.join(OUTPUT_DIR, 'summary_stats.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Saved summary_stats.json")
    return summary


if __name__ == '__main__':
    print("Generating visualizations...")
    results = load_results()

    print(f"Loaded {len(results)} result records")

    fig1_map_gallery()
    fig2_overall_success_rate(results)
    fig3_success_by_category(results)
    fig4_success_vs_agents(results)
    fig5_computation_time(results)
    fig6_conflict_reduction(results)
    fig7_path_quality(results)
    fig8_algorithm_diagram()
    fig9_heatmap_analysis(results)
    fig10_sample_solution()
    fig11_phase_analysis(results)
    stats = generate_summary_stats(results)

    print("\nAll figures saved to:", IMG_DIR)
    print("\nSummary Statistics:")
    for algo, s in stats.items():
        print(f"  {algo}: SR={s['success_rate']:.1%}, AvgConflicts={s['avg_conflicts']:.2f}, AvgTime={s['avg_time']:.2f}s")
