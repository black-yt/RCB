# Research Task

## Task Description
The task of this study is to design and evaluate a novel graph neural network architecture, termed Kolmogorov–Arnold Graph Neural Networks (KA-GNNs), for molecular property prediction by representing molecules as graphs with atom-level and bond-level features (including both covalent and non-covalent interactions) as input, and producing predictions of molecular properties such as toxicity, bioactivity, and physiological effects as output; the overarching scientific objective is to enhance predictive accuracy, computational efficiency, and interpretability by replacing conventional MLP-based transformations in graph neural networks with Fourier-based Kolmogorov–Arnold network modules that provide stronger expressive power and theoretical approximation guarantees.

## Available Data Files
- **bace.csv** [structure_data] (`data/bace.csv`): The BACE dataset contains small-molecule compounds represented by SMILES strings along with binary labels indicating whether each molecule inhibits human β-secretase 1 (BACE-1). The dataset is used for molecular property prediction tasks in drug discovery, where molecular structures are converted into graph representations (atoms as nodes and bonds as edges) for classification modeling.
- **bbbp.csv** [structure_data] (`data/bbbp.csv`): The BBBP (Blood–Brain Barrier Penetration) dataset contains small-molecule compounds represented by SMILES strings and binary labels indicating whether a compound can penetrate the blood–brain barrier. The dataset is used for molecular property classification tasks in pharmacology and drug design.
- **clintox.csv** [structure_data] (`data/clintox.csv`): The ClinTox dataset consists of small-molecule compounds represented by SMILES strings and labeled according to their clinical trial toxicity outcomes and FDA approval status. It is a multi-task binary classification dataset designed to evaluate a model’s ability to predict both drug toxicity and regulatory approval likelihood. Molecules are converted into graph representations for molecular property prediction tasks.
- **hiv.csv** [structure_data] (`data/hiv.csv`): The HIV dataset contains small-molecule compounds represented by SMILES strings and labeled according to their ability to inhibit HIV replication. It is a binary classification dataset commonly used in molecular property prediction benchmarks. Each molecule is transformed into a graph structure for deep learning-based prediction of antiviral activity.
- **muv.csv** [structure_data] (`data/muv.csv`): The MUV (Maximum Unbiased Validation) dataset is a large-scale molecular benchmark dataset consisting of small-molecule compounds represented by SMILES strings and labeled across multiple virtual screening tasks. It is designed to provide challenging and unbiased evaluation settings for molecular property prediction models. The dataset includes multiple binary classification tasks and exhibits high class imbalance, making it particularly difficult for predictive modeling.

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
