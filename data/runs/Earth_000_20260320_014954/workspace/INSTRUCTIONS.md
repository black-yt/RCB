# Research Task

## Task Description
Input: 233 sets of mass change estimates for 19 global glacial regions, derived from four observational methods (glaciological measurements, DEM differencing, altimetry, gravimetry) and hybrid methods.
Output: 2000–2023 regional and global glacial mass change time series (annual resolution), including uncertainties, expressed as specific mass change (m w.e.) and total mass change (Gt).
Scientific Objective: Reconcile diverse observational methods to deliver a consistent and high-confidence assessment of global glacial mass change, and establish an observational benchmark for IPCC reports and climate model calibration.

## Available Data Files
- **glambie** [sequence_data] (`data/glambie`): This dataset is the result of collecting, homogenizing, combining, and analyzing regional estimates derived from four primary observation methods (in situ glaciological measurements, digital elevation model differencing, satellite altimetry, and gravimetry) as well as integrated methods. It incorporates 233 regional estimates of glacier mass change contributed by 35 research teams and approximately 450 data contributors.

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
