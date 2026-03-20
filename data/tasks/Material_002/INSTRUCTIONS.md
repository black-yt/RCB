# Research Task

## Task Description
Input:
The MPtrj dataset from the Materials Project (~1.5 million inorganic crystal structures and relaxation trajectories) and the MACE graph neural network architecture.

Output:
A general-purpose foundation model for atomistic potentials that can be directly applied to diverse chemical systems (liquids, solids, catalysis, reactions, etc.), with the ability to achieve ab initio accuracy after fine-tuning on minimal task-specific data.

Scientific Goal:
To develop and validate a universal foundation model for atomistic simulations that covers the periodic table, stably simulates diverse material systems, and achieves quantitative accuracy with minimal fine-tuning.

## Available Data Files
- **MACE-MP-0_Reproduction Dataset.txt** [structure_data] (`data/MACE-MP-0_Reproduction_Dataset.txt`): This dataset contains all parameters and structural information needed to reproduce the performance of the MACE-MP-0 foundation model on three key tests: liquid water structure, adsorption energy scaling relations on transition metal surfaces, and CRBH20 reaction barriers.

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
