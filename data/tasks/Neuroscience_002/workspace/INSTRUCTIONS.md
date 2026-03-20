# Research Task

## Task Description
Input: An over-segmented electron microscopy (EM) image volume of a fly brain and a pair of adjacent neuron segments (a query segment and a candidate segment) located near a potential truncation point
Output: A binary prediction (0 or 1) indicating whether the two given segments belong to the same neuron and should be merged.
Scientific Goal: To automate the proofreading process in large-scale connectomics by accurately predicting connectivity between over-segmented neuron fragments, thereby reducing the massive manual workload required to reconstruct complete neurons from petascale EM data.

## Available Data Files
- **test_simulated.csv** [structure_data] (`data/test_simulated.csv`): Contains approximately 3600 samples (30% of total). Identical structure to the training set: 20 features, label, and degradation type. Used for evaluating model performance on unseen data.
- **train_simulated.csv** [structure_data] (`data/train_simulated.csv`): Contains approximately 8400 samples (70% of total). Each sample has 20 feature columns (0‑19) representing morphology, intensity, and embedding modalities, a binary label (1 for same neuron, 0 otherwise), and a degradation type (Misalignment, Missing Sections, Mixed, or Average). The data is stratified by degradation to ensure balanced representation.

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
