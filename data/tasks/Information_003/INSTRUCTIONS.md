# Research Task

## Task Description
Input: Network traffic flow data, including both benign and malicious traffic samples with temporal and topological features.Output: Intrusion detection results, including binary classification (benign vs. attack) and multi-class classification (specific attack types), particularly for known, unknown, and few-shot attack scenarios.Scientific Objective: To address the inconsistent performance and poor detection capability of existing Network Intrusion Detection Systems (NIDS) across different attack types—especially for unknown and few-shot attacks—by proposing a disentangled dynamic intrusion detection framework (DIDS-MFL). The framework aims to disentangle entangled feature distributions in traffic data through statistical and representational disentanglement, incorporate dynamic graph diffusion for spatiotemporal aggregation, and enhance few-shot learning via multi-scale representation fusion, thereby improving detection accuracy, consistency, and generalization in real-world network environments.

## Available Data Files
- **NF-UNSW-NB15-v2_3d.pt** [feature_data] (`data/NF-UNSW-NB15-v2_3d.pt`): NF-UNSW-NB15 is a NetFlow‑based feature dataset where each row represents a single network flow described by 8 to 53 statistical features (e.g., timestamp, duration, bytes, packet rates, inter‑arrival times), extracted from packet headers and stored in CSV format for binary/multi‑class intrusion detection.

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
