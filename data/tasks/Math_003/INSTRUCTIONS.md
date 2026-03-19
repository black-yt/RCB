# Research Task

## Task Description
Input: Formal statements of olympiad-level geometry problems (e.g., IMO diagrams and premises).
Output: Machine-verifiable, human-readable proofs for Euclidean geometry theorems.
Scientific Goal: To develop an AI system that autonomously solves complex geometry problems without human demonstrations, advancing neuro-symbolic reasoning in mathematics.

## Available Data Files
- **imo_ag_30.txt** (`data/imo_ag_30.txt`): A curated benchmark of 30 geometry problems from the International Mathematical Olympiad (since 2000), used for final evaluation.

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
