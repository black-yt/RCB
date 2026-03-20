# Research Task

## Task Description
The input to HADDOCK3 consists of atomic coordinates of biomolecules (proteins, glycans, etc.) in PDB format, along with optional experimental restraints (e.g., ambiguous interaction restraints) and user-defined workflows. The output is an ensemble of modeled three-dimensional structures of biomolecular complexes, ranked and clustered according to various scoring functions. The scientific goal is to provide a versatile, modular platform for integrative modeling that leverages experimental data to predict accurate structures of biomolecular complexes, complementing machine learning approaches.

## Available Data Files
- **1brs_AD.pdb** [structure_data] (`data/1brs_AD.pdb`): Processed structure of barnase-barstar complex (chains A and D) without water, used as input for HADDOCK3 analysis.
- **skempi_v2.csv** [feature_data] (`data/skempi_v2.csv`): SKEMPI 2.0 database containing experimental binding affinity changes upon mutation, used for validation.

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
