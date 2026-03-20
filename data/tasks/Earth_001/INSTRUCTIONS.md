# Research Task

## Task Description
Input: NOAA weather-modification records released by the target paper, covering reported cloud-seeding projects in the United States from 2000 to 2025. Output: reproducible tables and figure-level evidence for spatial concentration, annual activity dynamics, purpose composition, and agent-apparatus deployment patterns. Scientific objective: test whether the paper's central empirical conclusions can be independently recovered from the published structured dataset using transparent, script-based analysis.

## Available Data Files
- **cloud_seeding_us_2000_2025** [sequence_data] (`data/dataset1_cloud_seeding_records`): Official project-level cloud-seeding records released with the target paper, covering reported U.S. activities from 2000 to 2025. This is the only dataset used in this submission and it supports all reproduced tables, figures, and summary conclusions. The dataset contains 12 structured fields per record: filename, project name, year, season, state, operator affiliation, seeding agent, deployment apparatus, stated purpose, target area, control area, start date, and end date. The data enables comprehensive analysis of temporal trends, geographic distributions, operational characteristics, and methodological patterns in U.S. weather modification activities over a 25-year period.

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
