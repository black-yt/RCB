# Research Task

## Task Description
The input is the three-dimensional structure of a protein complex (represented in PDB/mmCIF format or Foldseek database format), and the output is the structural alignment result between the complexes, including the correspondence between chains, superimposition vectors, and the TM score used to quantify similarity. Its scientific goal is to achieve efficient search and similarity detection in large-scale protein complex structure databases (such as those containing millions of structures) through ultra-fast and sensitive alignment algorithms.

## Available Data Files
- **7xg4.pdb** [structure_data] (`data/7xg4.pdb`): Query complex structure from PDB ID 7xg4 (Pseudomonas aeruginosa type IV‑A CRISPR–Cas system). Used in the paper as a known structural hit against an environmental Sulfitobacter sp. JL08 complex.
- **6n40.pdb** [structure_data] (`data/6n40.pdb`): Target complex structure from PDB ID 6n40. Used for pairwise structural alignment with 7xg4 to test Foldseek‑Multimer’s alignment capability.

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
