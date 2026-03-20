# Research Task

## Task Description
Input: A MAPF (Multi-Agent Path Finding) instance I, which consists of a discrete 2D grid map with static obstacles and a set of agents, each with a distinct start position and a designated goal position.

Output: A solution path set P, consisting of collision-free paths for all agents that navigate them from their start positions to their goal positions without vertex or swapping collisions.

Scientific Goal (Task): To solve the Multi-Agent Path Finding problem by developing a hybrid algorithm that integrates Multi-Agent Reinforcement Learning into the Large Neighborhood Search framework. The specific objective is to balance solution quality (reducing collisions via MARL in early stages) and computational efficiency (using Prioritized Planning in later stages) to achieve higher success rates in complex environments compared to existing methods.

## Available Data Files
- **maps_60_10_10_0.175** [structure_data] (`data/maps_60_10_10_0.175`): The task set consists of static 2D grid maps )and multi-agent task configurations defined by preset parameters such as agent count, start positions, and target positions.
- **empty** [structure_data] (`data/empty`): Dataset of empty 2D grid maps (likely 25x25) with no static obstacles, containing multi-agent task configurations to evaluate navigation in high-density open spaces without structural blockages.
- **maze** [structure_data] (`data/maze`): Dataset of maze-structured 2D grid maps (25x25) characterized by complex corridors and dead-ends, containing task configurations designed to test pathfinding algorithms in highly constrained environments.
- **random_large** [structure_data] (`data/random_large`): Dataset of large-scale (50x50) 2D grid maps with randomly generated obstacles (17.5% density), containing task configurations serving as a benchmark for algorithm scalability in unstructured environments.
- **random_medium** [structure_data] (`data/random_medium`): Dataset of medium-sized (25x25) 2D grid maps with randomly distributed obstacles (17.5% density), containing task configurations that balance map complexity and computational load for standard evaluation.
- **random_small** [structure_data] (`data/random_small`): Dataset of small-scale (10x10) 2D grid maps with random obstacles (17.5% density), containing task configurations primarily used for rapid testing and analyzing algorithm behavior in tight spaces.
- **room** [structure_data] (`data/room`): Dataset of room-structured 2D grid maps (25x25) simulating indoor environments with connected chambers and narrow doorways, containing task configurations for analyzing bottleneck traversal.
- **warehouse** [structure_data] (`data/warehouse`): Dataset of warehouse-style 2D grid maps (25x25) with organized shelf layouts, containing task configurations specifically designed to simulate automated logistics and retrieval scenarios.

## Workspace Layout
- `data/` — Input datasets (read-only, do not modify)
- `related_work/` — Reference papers and materials
- `code/` — Write your analysis code here
- `outputs/` — Save intermediate results
- `report/` — Write your final research report here
- `report/images/` — Save all report figures here

## Deliverables
1. Write analysis code in `code/` that processes the data
2. Save intermediate outputs to `outputs/`
3. Write a comprehensive research report as `report/report.md`
   - Include methodology, results, and discussion
   - Use proper academic writing style
   - **You MUST include figures in your report.** Generate plots, charts, and visualizations that support your analysis
   - Save all report figures to `report/images/` and reference them in the report using relative paths: `images/figure_name.png`
   - Include at least: data overview plots, main result figures, and comparison/validation plots

## Guidelines
- Install any needed Python packages via pip
- Use matplotlib/seaborn for visualization
- Ensure all code is reproducible
- Document your approach clearly in the report
