# Research Task

## Task Description
Input:
Multimodal materials data: atomic structures, chemical compositions, crystal graphs, microscopy images, spectral data (XRD, FTIR, etc.), text from scientific literature, material property databases, synthesis conditions, and experimental parameters.  

Output:
Predicted material properties (mechanical, electronic, catalytic, etc.), generated novel material structures/microstructures, optimized synthesis/processing parameters, and classification/segmentation results for material characterization.  

Scientific Objective:
To accelerate the discovery, development, and optimization of advanced materials by integrating multimodal data through AI/ML models, enabling data-driven inverse design and reducing reliance on traditional trial-and-error approaches.

## Available Data Files
- **M-AI-Synth__Materials_AI_Dataset_.txt** (`data/M-AI-Synth__Materials_AI_Dataset_.txt`): This dataset is designed for rapid validation of three core AI application workflows in materials science—property prediction, structure generation, and experimental optimization—supporting code prototyping and fundamental algorithm testing.

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
