# Research Task

## Task Description
Input: Global atmospheric reanalysis data (ERA5) at 0.25° resolution, including 5 upper-air variables (geopotential, temperature, u-wind, v-wind, relative humidity) at 13 pressure levels and 5 surface variables (2m temperature, 10m u-wind, 10m v-wind, mean sea level pressure, total precipitation), from two consecutive 6-hour time steps. Output: 15-day global weather forecasts at 6-hour temporal resolution. Scientific Goal: Develop a cascade machine learning forecasting system using three specialized U-Transformer models to mitigate forecast error accumulation and extend skillful weather prediction to 15 days, achieving performance comparable to the ECMWF ensemble mean.

## Available Data Files
- **20231012-06_input_netcdf.nc** [sequence_data] (`data/20231012-06_input_netcdf.nc`): Pre-processed input data (20231012-06_input_netcdf.nc) with shape (2, 70, 721, 1440) representing two consecutive 6-hour atmospheric states at 0.25° resolution with 70 variables/channels
- **006.nc** [sequence_data] (`data/006.nc`): FuXi output forecasts (006.nc) at 6-hour intervals up to 15 days.

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
