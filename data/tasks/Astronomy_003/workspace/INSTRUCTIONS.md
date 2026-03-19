# Research Task

## Task Description
Input: Initial parameters of binary black hole systems, including mass ratio, spin vectors, orbital eccentricity, etc.
Output: Gravitational waveforms (strain and Weyl scalar) produced by numerical relativity simulations, black hole horizon properties (mass, spin, trajectories), and detailed metadata.
Scientific goal: To construct a high-accuracy, high-coverage catalog of binary black hole simulations for gravitational-wave data analysis, waveform model calibration, and fundamental physics research.

## Available Data Files
- **fig6_data.csv** (`data/fig6_data.csv`): This dataset contains synthetic waveform differences representing the mismatch between the two highest numerical resolutions used in the SXS binary black hole simulations, after minimal time and phase alignment. The file has a single column with 1500 entries, each corresponding to one simulation in the catalog. The values are drawn from a log‑normal distribution with a median of approximately 4×10⁻⁴, matching the typical resolution error reported in the SXS collaboration’s third catalog paper. The distribution spans roughly 10⁻⁶ to 0.5, with a long tail toward larger differences. In the paper, such data are used to assess the overall numerical uncertainty of the waveform catalog and to demonstrate that the majority of simulations achieve high accuracy.
- **fig7_data.csv** (`data/fig7_data.csv`): This file provides synthetic waveform differences decomposed by spherical harmonic mode ℓ, covering ℓ=2 through ℓ=8. It consists of 1500 rows (simulations) and 7 columns, where each column corresponds to a specific ℓ value and contains the minimal‑alignment waveform difference for that mode alone. The data are generated such that the median difference increases with ℓ (from about 3×10⁻⁴ at ℓ=2 to a few times 10⁻³ at ℓ=8), and the scatter also grows slightly for higher ℓ. In the original SXS study, such modal error distributions are critical for understanding how waveform accuracy varies across different multipoles and for guiding the truncation of mode contributions in gravitational‑wave models.
- **fig8_data.csv** (`data/fig8_data.csv`): This dataset contrasts waveform differences arising from two extrapolation‑order comparisons: N=2 vs N=3 and N=2 vs N=4. It contains 1200 rows and two columns; the first column stores the differences between extrapolation orders 2 and 3, the second column stores differences between orders 2 and 4. The synthetic values are drawn from log‑normal distributions with medians of 2×10⁻⁵ (for N2 vs N3) and 5×10⁻⁵ (for N2 vs N4), reflecting the trend that higher‑order extrapolation pairs yield larger discrepancies. In the SXS catalog paper, such comparisons are used to evaluate the convergence of the extrapolation procedure that extracts waveforms from finite‑radius simulation data to infinite null infinity, an essential step for producing reliable templates for gravitational‑wave astronomy.

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
