# Research Task

## Task Description
input: Atomic configurations (atomic positions, element types, and optionally periodic boundary conditions and total charge of the system).
output: Predicted total potential energy, atomic forces, and interpretable latent charges that can be used to derive physical quantities such as dipole moments, quadrupole moments, and Born effective charges.
Scientific Objective: To develop a machine-learning interatomic potential that accurately and efficiently incorporates long-range electrostatic interactions without explicitly learning atomic charges or performing charge equilibration, thereby improving predictions for systems where electrostatics are critical (e.g., electrochemical interfaces, charged molecules, ionic liquids).

## Available Data Files
- **random_charges.xyz** [structure_data] (`data/random_charges.xyz`): This dataset contains 128-atom configurations with fixed point charges (+1e and -1e) randomly placed in a box. The interactions are modeled by Coulomb potential and a repulsive Lennard‑Jones term. It is used to benchmark whether the Latent Ewald Summation (LES) method can recover the exact atomic charges solely from energy and force data, as shown in Fig. 1 of the paper.
- **charged_dimer.xyz** [structure_data] (`data/charged_dimer.xyz`): This dataset consists of configurations of two charged molecular dimers (each with total charges +1e and -1e) at various separation distances, with small internal distortions. It is employed to evaluate the ability of long‑range models to capture binding energy curves when molecules are beyond the short‑range cutoff, as illustrated in Fig. 3 of the paper.
- **ag3_chargestates.xyz** [structure_data] (`data/ag3_chargestates.xyz`): This dataset includes Ag₃ trimers in two different charge states (+1 and -1) with varying bond lengths and random distortions. It is used to demonstrate that a short‑range model with global charge embedding (or separate training) is necessary to distinguish potential energy surfaces of different charge states, as shown in Fig. 5e and Table 1 of the paper.

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
