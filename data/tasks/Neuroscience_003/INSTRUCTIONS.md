# Research Task

## Task Description
The input is single-cell readouts (such as scRNA-seq or protein imaging), and the output is a selected subset of dynamically expressed molecular features that best preserves continuous cellular trajectories. This setup supports analyses of neural lineage progression, glial activation, and neurodegeneration-related state transitions while reducing confounding variation.

## Available Data Files
- **adata_RPE.h5ad** [feature_data] (`data/adata_RPE.h5ad`): A preprocessed single-cell dataset (protein iterative indirect immunofluorescence imaging) showcasing cellular state transitions in a retina-related context that is compatible with neuroscience-adjacent analysis.

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
