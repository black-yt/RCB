# Research Task

## Task Description
Input is protein sequence features converted to monomer compositions; Output is hydrogel adhesive strength. To de novo design synthetic hydrogels that achieve robust underwater adhesion (>1 MPa) by statistically replicating the sequence features of natural adhesive proteins.

## Available Data Files
- **Initial_Training_Data_180** [feature_data] (`data/Original Data_ML_20220829.xlsx`): Batch 1, The initial experimental dataset containing monomer compositions and adhesive strengths for the 180 bio-inspired hydrogels used to train the base models.
- **Initial_Training_Data_180** [feature_data] (`data/Original Data_ML_20221031.xlsx`): Batch 2, The initial experimental dataset containing monomer compositions and adhesive strengths for the 180 bio-inspired hydrogels used to train the base models.
- **Initial_Training_Data_180** [feature_data] (`data/Original Data_ML_20221129.xlsx`): Batch 3, The initial experimental dataset containing monomer compositions and adhesive strengths for the 180 bio-inspired hydrogels used to train the base models.
- **Initial_Training_Data** [feature_data] (`data/184_verified_Original Data_ML_20230926.xlsx`): The cleaned and verified dataset containing the initial 184 hydrogel formulations. This file is the primary input for the 'rfr_gp.py' script to train the initial machine learning models.
- **Final_Optimization_Dataset** [feature_data] (`data/ML_ei&pred (1&2&3rounds)_20240408.xlsx`): The comprehensive dataset aggregating experimental results from all optimization rounds (1, 2, and 3). It serves as the input for evaluation notebooks to analyze the overall optimization trajectory and validate performance.
- **Final_Optimization_Dataset** [feature_data] (`data/ML_ei&pred_20240213.xlsx`): The comprehensive dataset aggregating experimental results from another batch of final optimization dataset. It serves as the input for evaluation notebooks to analyze the overall optimization trajectory and validate performance.

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
