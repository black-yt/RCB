# Research Task

## Task Description
Develop an AI-guided inverse-design framework for recyclable vitrimeric polymers by combining molecular dynamics simulations, Gaussian-process calibration, and a graph variational autoencoder, with the goal of generating new vitrimer chemistries that achieve desired glass transition temperatures (Tg) and validating selected candidates experimentally.

## Available Data Files
- **tg_calibration** [feature_data] (`data/tg_calibration.csv`): Contains molecular SMILES, experimental Tg values, and MD simulated Tg values used to train and evaluate the Gaussian process calibration model.
- **tg_vitrimer_MD** [feature_data] (`data/tg_vitrimer_MD.csv`): Contains molecular structures of vitrimer systems and their MD simulated Tg values used as input for the Gaussian process calibration to generate calibrated Tg predictions.

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
