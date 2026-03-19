# Research Task

## Task Description
Input:  1. Particle types and sizes: Atoms of varying sizes (e.g., alkali metals Na, K, Rb, Cs; transition metals Ag, Cu, Ni, etc.) or colloidal particles.  2. Path rules: Shell sequence paths defined in the hexagonal lattice (e.g., $(0,0) ightarrow (0,1) ightarrow (1,1) ightarrow (1,2)\dots$).  3. Interaction parameters: Interatomic potentials (e.g., Lennard-Jones, Gupta) or first-principles calculation parameters.  Output:  1. Predicted stable multi-shell icosahedral structures (e.g., $\mathrm{Na_{13}@K_{32}}$, $\mathrm{Ni_{147}@Ag_{192}}$).  2. Optimal size mismatch values between adjacent shells.  3. Shell sequences and paths formed via self-assembly in growth simulations.  Scientific Objective: To establish a universal theoretical framework for the rational design of multi-component nanoclusters and nanoparticles with specific symmetry (chiral or achiral) and compositional sequences, and to predict their stability and growth behavior, ultimately enabling targeted material fabrication for applications in catalysis, optics, and related fields.

## Available Data Files
- **Multi-component Icosahedral Reproduction Data.txt** (`data/Multi-component Icosahedral Reproduction Data.txt`): This dataset contains complete parameters and result data for reproducing all simulation experiments from the paper "General theory for packing icosahedral shells into multi-component aggregates", enabling full reproduction of theoretical calculations, experimental verification, and dynamic growth simulations in any computational environment.

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
