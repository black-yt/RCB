# Metric Definitions

## Primary definitions
- **Per-cell immune response probability**: the `p_response` value provided in `sim-specific-response-likelihoods.csv`, summarized by repetition and overall.
- **Tumor-cell coverage ratio (mutation-level)**: for each cell and repetition, define the set of unique presented mutations from `cell-populations.csv`. A cell is covered if at least one selected vaccine mutation for the same repetition is present in that set. Coverage ratio is the fraction of covered cells.
- **Coverage sensitivity analyses**:
  - mutation-hit thresholds: covered if the cell contains at least 1, 2, or 3 selected mutations;
  - response-probability thresholds: fraction of cells with `p_response` >= 0.5, 0.9, and 0.95.
- **IoU of optimal vaccine compositions**: Jaccard index between repetition-specific sets of selected vaccine mutations from `selected-vaccine-elements.budget-10.minsum.adaptive.csv`.
- **Optimization runtime**: descriptive scaling analysis using `optimization_runtime_data.csv`; repetition-level selection runtime in `selected-vaccine-elements...csv` is summarized separately.

## Notes
- The selection file column named `peptide` contains mutation-like identifiers (e.g., `mut11`); figures and tables refer to these as selected mutation elements.
- `final-response-likelihoods.csv` is treated as an aggregate consistency check, while repetition-level inference uses `sim-specific-response-likelihoods.csv`.
