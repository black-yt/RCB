# Research Task

## Task Description
Develop a unified deep learning framework that takes protein sequences, nucleic acid sequences, and small molecule structures as input, and outputs accurate 3D structures of biomolecular complexes using a diffusion-based architecture to predict interactions across diverse biological molecules.

## Available Data Files
- **2l3r_protein.pdb** [protein_structure] (`data/sample/2l3r/2l3r_protein.pdb`): This file contains the experimental structure of the FKBP12 protein (PDB ID: 2L3R) determined by NMR. It includes only the CA atoms of all 107 residues in PDB format. This file serves as the ground truth protein structure for evaluating AlphaFold 3 predictions. The CA coordinates are directly comparable to the model's protein backbone output and are used in RMSD calculations and structural overlay visualizations.
- **2l3r_ligand.sdf** [ligand_structure] (`data/sample/2l3r/2l3r_ligand.sdf`): This file provides the experimental 3D conformation of the FK506 ligand, stored in the standard Structure-Data File (SDF) format. It contains the full atomic coordinates, bond connectivity, and chemical properties of the molecule. This is the primary reference for evaluating ligand pose prediction accuracy; the ligand RMSD is computed by aligning the predicted ligand coordinates to this reference using symmetry-aware Hungarian matching.

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
