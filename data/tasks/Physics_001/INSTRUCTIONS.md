# Research Task

## Task Description
Input
A magic-angle twisted bilayer graphene (MATBG) device with gate-tunable carrier density, subjected to DC bias current and microwave probe signals at cryogenic temperatures (~20 mK).

Output
The device's DC resistance, microwave resonance frequency, and their dependence on temperature, gate voltage, and current. The core extracted physical quantity is the superfluid stiffness and its temperature and current dependence.

Scientific Goal
To directly measure the superfluid stiffness of MATBG, test whether it significantly exceeds predictions of conventional Fermi liquid theory, investigate its power-law temperature dependence to reveal the nature of unconventional pairing (anisotropic gap), and verify the crucial role of quantum geometric effects in flat-band superconductivity.

## Available Data Files
- **MATBG Superfluid Stiffness Core Dataset.txt** [feature_data] (`data/MATBG Superfluid Stiffness Core Dataset.txt`): This dataset fully contains all simulated data required to reproduce the three core experiments of the target study, covering carrier density dependence, temperature dependence, and current dependence. It can be directly used to independently verify key conclusions such as quantum geometry-dominated enhancement of superfluid stiffness, power-law behavior of anisotropic gaps, and quadratic current relationships.

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
