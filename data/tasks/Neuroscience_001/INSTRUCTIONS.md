# Research Task

## Task Description
Input
The connectome of the motion pathway in the Drosophila optic lobe (including the number of synaptic connections, polarity, and spatial distribution of 64 cell types). Unknown single-neuron kinetic parameters (time constants, resting potentials) and unit synaptic strengths, which need to be determined through optimization. Output
A deep mechanistic network (DMN) whose structure strictly follows the connectome, with parameters learned through task optimization, capable of simulating the voltage activities of 45,669 neurons in response to visual stimuli. Detailed predictions of the neural activity of each neuron, as well as quantitative analysis of motion detection mechanisms. Scientific Goal
To demonstrate that the activity of each neuron in a neural circuit can be accurately predicted solely based on connectome measurements (structure) and task knowledge (functional goals), thereby establishing a bridge from structure to function. Overall Task
Construct a connectome-constrained and task-optimized deep mechanistic network that can perform optical flow estimation tasks, and reveal the computational role of each neuron in the Drosophila visual system in motion detection through model predictions.

## Available Data Files
- **flow** [sequence_data] (`data/flow`): the complete ensemble of 50 pre-trained deep mechanistic network (DMN) models, along with all necessary configuration files, synapse count matrices, and cell‑type annotations. These models are constrained by the fly connectome and optimized for optic flow estimation, allowing users to directly simulate neural responses, reproduce key analyses from the paper, and generate experimentally testable hypotheses without retraining.

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
