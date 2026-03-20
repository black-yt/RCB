# Research Task

## Task Description
(Definition of input, output, and scientific goal)Text to copy:Input: Experimental macroscopic data (voltage, temperature, and capacity curves under discharge conditions) and a multi-parameter search space defined by Latin Hypercube Sampling (LHS).Output: A set of identified high-fidelity internal parameters (such as particle radius, reaction rates, and thermal coefficients) for the electrochemical-aging-thermal (ECAT) coupled model.Scientific Goal: To develop a rapid and accurate parameter identification framework (MMGA) that uses an Artificial Neural Network (ANN) meta-model to replace computationally expensive physical simulations, thereby solving the trade-off between model complexity and calculation efficiency for Lithium-ion battery digital twins.

## Available Data Files
- **NASA PCoE Dataset Repository** [structure_data] (`data/NASA PCoE Dataset Repository`): Experimental aging data of 18650 Li-ion batteries provided by the NASA Prognostics Center of Excellence (PCoE). It includes voltage, current, and temperature profiles recorded during constant current (CC) discharge cycles at room temperature, used here for experimental validation of the identification algorithm.
- **CS2_36** [sequence_data] (`data/CS2_36`): Cycle life test data for a Commercial NCM (Nickel Cobalt Manganese) 18650 cell provided by the University of Maryland CALCE Battery Research Group. The dataset features standard 1C constant current discharge curves, used as the primary reference for parameter identification.
- **Oxford Battery Degradation Dataset** [feature_data] (`data/Oxford Battery Degradation Dataset`): Long-term battery degradation data provided by the Oxford Battery Intelligence Lab. It contains dynamic urban driving profiles (highly transient current loads) obtained from 740mAh pouch cells, utilized to validate the model's generalization ability under dynamic conditions.

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
