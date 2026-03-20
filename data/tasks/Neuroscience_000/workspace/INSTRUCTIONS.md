# Research Task

## Task Description
Input: pose-derived frame-level feature tables and aligned behavior labels (Attack, Sniffing) from the official SimBA sample project. Output: trained supervised classifiers, quantitative evaluation reports, precision-recall diagnostics, confusion matrices, and feature-importance tables. Scientific objective: verify, on open data and executable code, whether the SimBA-style workflow can reproducibly transform tracked behavior features into transparent and auditable behavior classification evidence.

## Available Data Files
- **Together_1_features_extracted.csv** [feature_data] (`data/Together_1_features_extracted.csv`): Frame-level engineered features derived from tracked animal pose signals in the official SimBA sample project; used as model input matrix X for both behavior labels.
- **Together_1_targets_inserted.csv** [sequence_data] (`data/Together_1_targets_inserted.csv`): Frame-aligned target annotations for Attack and Sniffing from the same sample project; used as supervised targets y during train/test evaluation.
- **Together_1_machine_results_reference.csv** [feature_data] (`data/Together_1_machine_results_reference.csv`): Reference output table provided with the sample project, retained as auxiliary material for contextual comparison with reproduced classifier outputs.

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
