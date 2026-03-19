# Research Task

## Task Description
Input: Raw nanopore electrical signal (FAST5/POD5), basecalled reads, reference genome/transcriptome sequences, and k-mer pore models.
Output: Signal-to-reference alignments in BAM format, nucleotide modification calls (e.g., m6A sites), performance benchmarks, and trained pore models.
Scientific Objective: To develop Uncalled4, a fast and accurate toolkit for nanopore signal alignment, enabling more sensitive and comprehensive detection of DNA and RNA modifications, while overcoming limitations of existing tools in speed, file format, and compatibility with new sequencing chemistries.

## Available Data Files
- **dna_r9.4.1_400bps_6mer_uncalled4.csv** (`data/dna_r9.4.1_400bps_6mer_uncalled4.csv`): Contains 6-mer sequences and associated current statistics (mean, std, dwell time) for the DNA r9.4.1 pore model, used to generate substitution profiles and analyze base-position effects.
- **dna_r10.4.1_400bps_9mer_uncalled4.csv** (`data/dna_r10.4.1_400bps_9mer_uncalled4.csv`): Provides 9-mer pore model data for DNA r10.4.1 chemistry, including mean current, standard deviation, and dwell time, enabling comparison of substitution patterns between different pore versions.
- **rna_r9.4.1_70bps_5mer_uncalled4.csv** (`data/rna_r9.4.1_70bps_5mer_uncalled4.csv`): RNA001 pore model with 5-mer current parameters, used to study the relationship between nucleotide composition and ionic current in direct RNA sequencing.
- **rna004_130bps_9mer_uncalled4.csv** (`data/rna004_130bps_9mer_uncalled4.csv`): RNA004 pore model (9-mer) containing current mean, standard deviation, and dwell time, facilitating comparison of RNA pore chemistries and their impact on signal characteristics.
- **performance_summary.csv** (`data/performance_summary.csv`): Summary table of alignment time and file size for Uncalled4, f5c, Nanopolish, and Tombo across four sequencing chemistries, used to reproduce Table 1 performance benchmarks.
- **m6a_predictions_uncalled4.csv** (`data/m6a_predictions_uncalled4.csv`): m6Anet prediction probabilities for each candidate site based on Uncalled4 alignments, used to compute precision-recall curves and compare modification detection sensitivity.
- **m6a_predictions_nanopolish.csv** (`data/m6a_predictions_nanopolish.csv`): m6Anet prediction probabilities for the same sites using Nanopolish alignments, serving as a baseline for evaluating relative performance.
- **m6a_labels.csv** (`data/m6a_labels.csv`): Ground truth binary labels (0/1) for each site, derived from GLORI or m6A-Atlas, used as the reference for precision-recall and ROC analysis.

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
