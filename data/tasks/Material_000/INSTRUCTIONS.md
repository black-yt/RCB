# Research Task

## Task Description
The scientific objective of this work is to develop an AI-powered search engine that accelerates the discovery of new altermagnetic materials with targeted physical properties. The input consists of crystal structure data (represented as graphs) from databases such as the Materials Project, including a large unlabeled set for pre-training and a small labeled set of known altermagnets (148 positive samples) for fine-tuning. The output is a list of candidate materials predicted to be altermagnets with high probability, along with their electronic structure properties confirmed by first-principles calculations (e.g., 50 newly discovered altermagnets with classifications such as metal/insulator and d/g/i-wave anisotropy).

## Available Data Files
- **pretrain_data.pt** [structure_data] (`data/pretrain_data.pt`): This dataset contains a large number of unlabeled crystal structure graphs (5,000 samples) used for self-supervised pre-training. The model learns intrinsic representations of crystal structures without requiring magnetic labels, capturing general features that aid downstream tasks.
- **finetune_data.pt** [structure_data] (`data/finetune_data.pt`): This labeled dataset (2,000 samples) consists of crystal graphs with binary labels indicating altermagnetic (positive) or non-altermagnetic (negative) materials. It simulates the scarcity of known altermagnets by having only 5% positive samples (100 positives, 1900 negatives). It is used to fine-tune the pre-trained encoder into a classifier.
- **candidate_data.pt** [structure_data] (`data/candidate_data.pt`): This unlabeled dataset (1,000 samples) represents candidate materials whose magnetic properties are unknown. The trained classifier predicts the probability of each being an altermagnet. Hidden true labels are stored internally for evaluation, allowing measurement of discovery accuracy. Approximately 50 true positives are embedded based on generation rules.

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
