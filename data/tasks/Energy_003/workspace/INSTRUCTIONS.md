# Research Task

## Task Description
Input: Raw data sourced from sensor measurements (electricity, heat, cooling loads, PV generation of 147 buildings) of the Arizona State University Campus Metabolism Project and meteorological observations (temperature, humidity, wind speed, pressure, precipitation) from the U.S. National Weather Service.

Output: A multi-source, hierarchical time-series dataset (HEEW) comprising 11,987,328 records with 13 hourly variables (electricity, heat, cooling loads, PV generation, GHG emissions, and 7 weather attributes) from 2014 to 2022, along with data cleaning algorithms.

Scientific Goal: To provide a publicly available, comprehensive, and hierarchical benchmark dataset for energy system management, machine learning, and data-driven optimization (e.g., load forecasting, anomaly detection, clustering, imputation), addressing the gaps in existing multi-energy datasets regarding thermal loads, PV generation, emissions, and long-term coverage.

## Available Data Files
- **HEEW_Mini-Dataset** [sequence_data] (`data/HEEW_Mini-Dataset`): This compact and small dataset version of the core features of the target literature HEEW contains hourly electricity, heat, cooling load, photovoltaic power generation, greenhouse gas emissions, and seven meteorological data for the entire year of 2014. The data is organized in a hierarchical structure, covering 10 independent buildings (BN001–BN010), one aggregated community (CN01), and the entire area (Total), and is used to replicate the core experiments in the paper such as data cleaning, correlation analysis, and consistency verification of hierarchical aggregation, providing example data support for multi-energy system research and the development of machine learning algorithms.

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
