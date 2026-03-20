# Research Task

## Task Description
Build a transparent geospatial levelized-cost model to estimate the delivered cost of African green hydrogen to Europe (via ammonia shipping and reconversion) by 2030 under multiple financing and policy scenarios, identify least-cost/competitive locations, and quantify how de-risking and the interest-rate environment change cost competitiveness relative to producing green hydrogen in Europe.

## Available Data Files
- **hex_final_NA_min.csv** [feature_data] (`data/hex_final_NA_min.csv`): Simulated dataset for African hydrogen production sites, including latitude, longitude, PV potential, wind potential, and distances to road, grid, ocean, and water infrastructure. Used as input for LCOH calculations.
- **ne_10m_admin_0_countries.shp** [vector_data] (`data/africa_map/ne_10m_admin_0_countries.shp`): Main shapefile containing country boundary geometries for Africa and the world at 1:10m scale. Used as basemap for spatial visualizations.
- **ne_10m_admin_0_countries.shx** [vector_data] (`data/africa_map/ne_10m_admin_0_countries.shx`): Shape index file required for reading the shapefile geometry data efficiently.
- **ne_10m_admin_0_countries.dbf** [vector_data] (`data/africa_map/ne_10m_admin_0_countries.dbf`): Attribute database file containing tabular information (country names, codes, etc.) associated with each geometry.
- **ne_10m_admin_0_countries.prj** [vector_data] (`data/africa_map/ne_10m_admin_0_countries.prj`): Projection file that defines the coordinate system and map projection of the shapefile data.
- **ne_10m_admin_0_countries.cpg** [vector_data] (`data/africa_map/ne_10m_admin_0_countries.cpg`): Code page file specifying the character encoding used in the .dbf attribute file (typically UTF-8 or Latin-1).

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
