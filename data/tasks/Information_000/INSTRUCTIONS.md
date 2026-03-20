# Research Task

## Task Description
Build a unified autoregressive framework that decouples visual encoding to perform both multimodal understanding (e.g., visual question answering) and visual generation (e.g., text-to-image generation) within a single Transformer architecture.

## Available Data Files
- **equation.png** [sequence_data] (`data/equation.png`): An image containing a mathematical equation used to evaluate the model's optical character recognition (OCR) and formula-to-LaTeX conversion capabilities.
- **doge.jpg** [sequence_data] (`data/doge.png`): A specific meme image ("Swole Doge vs. Cheems") used in Figure 5 of the paper. It contains embedded text ("Decoupling Visual Encoding" vs. "Single Visual Encoder") and visual metaphors to evaluate the model's high-level semantic understanding of humor.

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
