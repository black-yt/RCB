# Research Task

## Task Description
Input multi-step analytic calculation tasks of the Hartree-Fock method from 15 quantum many-body physics research papers; output correctly derived Hartree-Fock Hamiltonians, calculation step scores, and automated results of paper information extraction and step scoring; the scientific goal is to verify whether large language models (LLMs) can accurately perform research-level theoretical physics calculations via structured prompt templates and mitigate key bottlenecks in the research process.

## Available Data Files
- **2111.01152** [feature_data] (`data/2111.01152`): The target scientific paper defining the AB-stacked MoTe2/WSe2 moiré system and its Hamiltonian parameters.

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
