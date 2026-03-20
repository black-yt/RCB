# Research Task

## Task Description
The task of this paper is to develop a composite risk index combining tropical cyclone regime shifts and sea level rise, and apply it globally to evaluate where and to what extent mangroves and their ecosystem services are at risk by the end of the century, in order to inform climate-adaptive conservation and management strategies.

## Available Data Files
- **gmw_v4_ref_smpls_qad_v12.gpkg** [vector_data] (`data/mangroves/gmw_v4_ref_smpls_qad_v12.gpkg`): Global mangrove extent polygons from Global Mangrove Watch (Bunting et al., 2018), used to derive centroid points and calculate mangrove area. Sampled to 10% for efficiency.
- **total_ssp245_medium_confidence_rates.nc** [netcdf] (`data/slr/total_ssp245_medium_confidence_rates.nc`): Regional relative sea level rise rates for SSP2-4.5 (medium confidence) from IPCC AR6 (Garner et al., 2021), used to extract median rates 2020–2100.
- **total_ssp370_medium_confidence_rates.nc** [netcdf] (`data/slr/total_ssp370_medium_confidence_rates.nc`): Regional relative sea level rise rates for SSP3-7.0 (medium confidence) from IPCC AR6 (Garner et al., 2021), used to extract median rates 2020–2100.
- **total_ssp585_medium_confidence_rates.nc** [netcdf] (`data/slr/total_ssp585_medium_confidence_rates.nc`): Regional relative sea level rise rates for SSP5-8.5 (medium confidence) from IPCC AR6 (Garner et al., 2021), used to extract median rates 2020–2100.
- **tracks_mit_mpi-esm1-2-hr_historical_reduced.nc** [netcdf] (`data/tc/tracks_mit_mpi-esm1-2-hr_historical_reduced.nc`): Historical tropical cyclone tracks from the MIT model (Emanuel et al., 2006) downscaled from CMIP6 MPI-ESM1-2-HR, covering 1850–2014. Used to calculate baseline cyclone frequencies after filtering and downsampling.

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
