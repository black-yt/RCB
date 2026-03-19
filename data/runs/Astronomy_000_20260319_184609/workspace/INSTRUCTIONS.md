# Research Task

## Task Description
To constrain the properties of ultralight bosons (ULBs) by developing and applying a novel Bayesian statistical framework. This framework translates the physics of black hole superradiance into a probabilistic model that ingests the full posterior distributions (not just point estimates) of black hole mass and spin measurements. The goal is to derive statistically rigorous upper limits on ULB masses and self-interaction coupling strengths, thereby using astrophysical data to probe fundamental particle physics.

## Available Data Files
- **IRAS_09149-6206_samples.dat** (`data/IRAS_09149-6206_samples.dat`): Contains the posterior distribution samples for the mass and dimensionless spin parameter of the supermassive black hole IRAS 09149-6206, providing the fundamental observational input for the constraint analysis in the supermassive black hole regime.
- **M33_X-7_samples.dat** (`data/M33_X-7_samples.dat`): Contains the posterior distribution samples for the mass and dimensionless spin parameter of the stellar-mass black hole in the X-ray binary M33 X-7, serving as the primary dataset for demonstrating and applying the Bayesian constraint framework.

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
