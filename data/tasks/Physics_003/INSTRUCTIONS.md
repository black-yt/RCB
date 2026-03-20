# Research Task

## Task Description
Input: Monolayer epitaxial graphene samples and mid-infrared pump excitation parameters (wavelength: 5 μm, intensity, polarization angle). Output: Direct, energy- and momentum-resolved observation of Floquet-Bloch states (replica bands of the Dirac cone) via time-resolved and angle-resolved photoemission spectroscopy (tr-ARPES). Scientific Goal: To experimentally confirm the existence of Floquet-Bloch states in a paradigmatic 2D material and elucidate the underlying scattering mechanism involving photon-dressed Volkov final states.

## Available Data Files
- **raw_trARPES_data.h5** [structure_data] (`data/raw_trARPES_data.h5`): Raw, unprocessed 4D data arrays (energy, momentum kx/ky, time delay) from the tr-ARPES experiment.
- **processed_band_data.json** [feature_data] (`data/processed_band_data.json`): Processed data containing the extracted positions and intensities of the main Dirac cone and replica bands.
- **polarization_dependence_data.csv** [sequence_data] (`data/polarization_dependence_data.csv`): Tabular data containing the measured intensity of the replica band for each pump polarization angle (θp).

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
