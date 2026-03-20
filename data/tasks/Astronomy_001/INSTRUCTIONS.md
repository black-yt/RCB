# Research Task

## Task Description
The scientific goal of this paper is to investigate whether an early dark energy (EDE) model can alleviate the acoustic tension between measurements from the cosmic microwave background (CMB) and baryon acoustic oscillations (BAO).  
Inputs: BAO data from DESI DR2, CMB data from Planck and ACT (including temperature and polarization power spectra and lensing), and Union3 supernova data (in some analyses).  
Outputs: Constraints on cosmological parameters (Ωm, H₀, σ₈, etc.) from model fitting, comparison of goodness-of-fit (Δχ²) for ΛCDM, EDE, and w₀wₐ models, and posterior distributions of EDE parameters (f_EDE, log₁₀a_c). The results show that EDE can partially relieve the tension but leads to different parameter shifts compared to late-time dark energy models.

## Available Data Files
- **DESI_EDE_Repro_Data.txt** [structure_data] (`data/DESI_EDE_Repro_Data.txt`): This dataset contains the best-fit parameters with 1σ errors for ΛCDM, EDE, and w₀wₐ models from Tables II/III of the paper, along with manually extracted DESI BAO and Union3 SNe data points from Figure 6, used to reproduce the paper's key parameter constraints and distance comparison results.

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
