# Research Task

## Task Description
The scientific goal of this paper is to achieve a ~1% precision measurement of the Hubble constant \(H_0\) by constructing a “Local Distance Network” that combines multiple distance indicators through a covariance‑weighted approach, providing a robust consensus result to address the Hubble tension. Inputs include geometric anchors (Milky Way parallaxes, LMC/SMC detached eclipsing binaries, NGC4258 masers), primary distance indicators (Cepheids, TRGB, Miras, JAGB), secondary indicators (SNe Ia, SBF, SNe II, FP, TF), and Hubble‑flow measurements. Outputs are a consensus value of \(H_0\) (baseline \(H_0 = 73.50 \pm 0.81 \ \mathrm{km\,s^{-1}\,Mpc^{-1}}\)), results from various analysis variants, and comparisons with early‑universe (CMB) constraints, along with publicly released software and data products.

## Available Data Files
- **H0DN_MinimalDataset.txt** [feature_data] (`data/H0DN_MinimalDataset.txt`): A minimal dataset to reproduce the Hubble constant measurement using the generalized least squares framework of the Distance Network, including geometric anchors, primary distance indicator measurements, secondary calibrations, and Hubble flow observations.

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
