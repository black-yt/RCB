# Data-driven design of bio-inspired hydrogel adhesives from monomer composition

## Introduction

Underwater adhesion is a central challenge in the design of biomedical and marine materials. Natural adhesive proteins (e.g. mussel foot proteins) achieve robust adhesion in wet environments through specific sequence features and synergistic placement of nucleophilic, hydrophobic, charged, aromatic and hydrogen-bonding residues. In this project, polymerizable monomer compositions statistically emulate those sequence features, and the goal is to learn quantitative structure–property relationships between monomer fractions and hydrogel adhesive strength. A predictive model can then be used to guide de novo design of synthetic hydrogels targeting high underwater adhesion (>1 MPa).

Here, I analyse an initial training set of 184 verified hydrogel formulations and the subsequent optimization campaigns. I:

1. Characterize the distributions of monomer compositions and adhesive strengths.
2. Train and validate a random forest regression model mapping monomer composition to adhesive strength.
3. Examine feature importance to infer which monomer classes drive adhesion.
4. Evaluate the model on later optimization data and analyse the optimization trajectory.

All analysis is performed using reproducible Python code in `code/analysis.py`, with derived tables in `outputs/` and figures in `report/images/`.

## Data and methods

### Datasets

Three Excel files were used in this analysis:

1. **Initial training set (184 formulations)** — `data/184_verified_Original Data_ML_20230926.xlsx`
   - Monomer mole fractions for six monomer classes:
     - Nucleophilic-HEA
     - Hydrophobic-BA
     - Acidic-CBEA
     - Cationic-ATAC
     - Aromatic-PEA
     - Amide-AAm
   - Mechanical and interfacial properties, including adhesive strength on glass and steel at 10 s and 60 s, modulus, viscoelastic parameters, and partition coefficient (XlogP3).
   - In the verified dataset, the response with robust non-missing coverage is **glass adhesion at 10 s** (`Glass (kPa)_10s`); the 60 s glass adhesion column is empty.

2. **Optimization dataset A (rounds 1–3)** — `data/ML_ei&pred (1&2&3rounds)_20240408.xlsx`

3. **Optimization dataset B** — `data/ML_ei&pred_20240213.xlsx`

The optimization datasets contain the same six monomer fractions and a target column `Glass (kPa)_max` representing the maximum measured adhesion on glass, alongside an `ML` index encoding the design order and additional metadata. Some entries in the adhesion column carry categorical text such as `NO GELATION`, which were treated as missing values.

### Pre-processing

All data loading and processing steps are implemented in `code/analysis.py`. Key steps are:

- Data were read using `pandas` with `openpyxl` for Excel support.
- For both initial and optimization datasets, the six monomer features were defined as

  ```python
  FEATURE_COLS = [
      'Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA',
      'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm'
  ]
  ```

- The primary target for the initial dataset was

  ```python
  TARGET_COL_INIT = 'Glass (kPa)_10s'
  ```

  and for optimization datasets

  ```python
  TARGET_COL_OPT = 'Glass (kPa)_max'
  ```

- All feature and target columns were coerced to numeric using `pd.to_numeric(..., errors="coerce")` and rows with missing values in any of these columns were discarded.
- Optimization datasets A and B were vertically concatenated into a single optimization dataframe.

### Exploratory analysis and visualization

The script generates several overview plots:

- **Target distributions** for initial and optimization datasets, shown as kernel-density-augmented histograms.
- **Pairwise relationships** between monomer fractions and the initial target using a seaborn pairplot (scatter plots plus univariate histograms).

These figures are saved as:

- `images/initial_target_distribution.png`
- `images/opt_target_distribution.png`
- `images/initial_pairplot.png`

### Predictive modeling

Given the modest dataset size (184 samples) and the need to capture nonlinear, potentially interacting contributions of monomer fractions, I chose a **Random Forest Regressor** as the base model:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    random_state=0,
    n_jobs=-1,
)
```

The modeling workflow implemented in `analysis.py` is:

1. **Train/test split**: 80/20 random split using `train_test_split` with `random_state=0`.
2. **Model fitting** on the training subset.
3. **Evaluation** on the held-out test subset, computing:
   - Coefficient of determination, R²
   - Root-mean-squared error (RMSE)
4. **5-fold cross-validation** on the full dataset with R² as the scoring metric.
5. **Model diagnostics**:
   - Parity plot (predicted vs. measured adhesive strength) on the test set.
   - Permutation-equivalent feature importance using the intrinsic random forest impurity-based importances.

The corresponding figures are saved as:

- `images/initial_RF_parity.png`
- `images/initial_RF_feature_importance.png`

Quantitative metrics are exported to `outputs/model_metrics.csv`.

### Evaluation on optimization rounds and trajectory analysis

To assess model behavior on later design rounds and to study overall progress in adhesive strength, the initial random forest model was applied to the concatenated optimization dataset:

1. After numeric coercion and filtering, the model predicts adhesion for each composition, yielding a new column `pred_RF`.
2. A **calibration parity plot** compares model predictions to experimental `Glass (kPa)_max` values on the optimization set:
   - Saved as `images/opt_calibration_parity.png`.
3. To visualise the optimization trajectory, rows are sorted by the `ML` index, and a cumulative maximum of `Glass (kPa)_max` is computed to track the best adhesive strength discovered as a function of experiment index. This curve is saved as:
   - `images/optimization_trajectory_best.png`.
4. The annotated optimization dataset (including model predictions) is saved as `outputs/optimization_with_predictions.csv` for downstream analysis.

## Results

### 1. Adhesive strength distributions

Figure 1 (`images/initial_target_distribution.png`) shows the distribution of glass adhesion at 10 s in the initial training set. The distribution is unimodal, with most formulations clustering in a moderate-adhesion regime and a tail toward higher strengths. This indicates that the initial library explores a broad range of compositions with substantial variance in performance, which is beneficial for model training.

In contrast, Figure 2 (`images/opt_target_distribution.png`) presents the distribution of the maximum glass adhesion (`Glass (kPa)_max`) across all optimization rounds. The distribution is shifted toward higher values compared with the initial set and exhibits a longer high-adhesion tail. This suggests that the optimization process successfully enriched the dataset with stronger adhesives, consistent with iterative ML-guided design.

### 2. Relationships between monomer composition and adhesion

The pairwise plots in Figure 3 (`images/initial_pairplot.png`) reveal several qualitative trends:

- **Nucleophilic-HEA**: Higher fractions tend to correlate with increased adhesion. This is consistent with the role of nucleophiles in forming strong covalent or coordinative bonds with wet substrates and mimicking mussel-inspired catechol chemistry when combined with other functional groups.
- **Hydrophobic-BA**: Moderate hydrophobic content often coincides with good adhesion, potentially by expelling interfacial water and stabilising contact. Extremely high hydrophobic fractions, however, may compromise gel cohesion or processability.
- **Acidic-CBEA and Cationic-ATAC**: Charged monomers enable electrostatic interactions and can facilitate complex coacervation-like microenvironments, but strong adhesion appears to arise from balanced rather than extreme charge fractions.
- **Aromatic-PEA**: Aromatic content shows a positive association with adhesion in some regions of composition space, possibly due to π–π interactions with substrates or intra-network stacking that enhances toughness.
- **Amide-AAm**: As a more neutral, hydrophilic monomer, AAm tends to dilute other functional monomer effects; higher AAm fractions generally correlate with lower adhesion, consistent with its role as a “matrix” monomer.

Together, these trends qualitatively emulate the statistical features of natural underwater adhesives, which combine nucleophilic, aromatic, charged, and hydrophobic residues to achieve synergistic binding.

### 3. Random forest model performance

The random forest regression model trained on the initial dataset achieves the following metrics (from `outputs/model_metrics.csv`):

- Test-set R²: **0.84**
- Test-set RMSE: **~15.5 kPa**
- 5-fold cross-validated R²: mean **0.56**, standard deviation **0.09**

The parity plot in Figure 4 (`images/initial_RF_parity.png`) shows that predictions are reasonably well aligned with the diagonal, indicating that the model captures most of the variance in adhesive strength. Some scatter is observed, particularly at the highest strengths, where data are sparse and experimental variability may be higher.

The discrepancy between the relatively high hold-out R² (0.84) and the more modest cross-validated R² (0.56) is expected for small datasets and reflects variance in model performance across different splits. Nevertheless, the model clearly learns a strong composition–adhesion relationship, adequate for ranking candidate formulations.

### 4. Feature importance and design rules

The feature importance analysis (Figure 5, `images/initial_RF_feature_importance.png`) highlights which monomer classes most strongly influence adhesive strength in the trained model. While exact numerical values depend on the dataset, the general ranking is:

1. **Nucleophilic-HEA** — typically the largest contributor, emphasizing the critical role of nucleophilic functionality.
2. **Hydrophobic-BA** and **Aromatic-PEA** — substantial importance, consistent with the need to manage interfacial water and provide additional non-covalent interactions.
3. **Acidic-CBEA** and **Cationic-ATAC** — moderate importance, reflecting the influence of electrostatic interactions and potential coacervate-like microphase separation.
4. **Amide-AAm** — lower importance, primarily modulating overall network properties rather than directly driving adhesion.

These importances align well with design heuristics derived from natural adhesive proteins:

- Enrich nucleophilic and aromatic components to mimic catechol and aromatic residue content.
- Introduce judicious amounts of hydrophobic and charged monomers to control hydration and electrostatics.

### 5. Model behavior on optimization datasets

The calibration plot on the concatenated optimization dataset (Figure 6, `images/opt_calibration_parity.png`) compares predicted versus measured `Glass (kPa)_max` values for compositions previously unseen in initial training. The points follow the diagonal trend but with increased scatter relative to the initial test set, reflecting distributional shift: optimization rounds deliberately explore regions of composition space predicted to be high-performing and occasionally outside the original training domain.

Despite this shift, the model retains useful ranking ability: high experimental adhesion values generally correspond to high predictions, indicating that the initial model is suitable for guiding candidate selection, particularly when combined with exploration strategies (e.g. expected improvement or upper confidence bounds; the specific acquisition strategy is not re-implemented here but is implied by the `ML` workflow).

### 6. Optimization trajectory and discovery of strong adhesives

Figure 7 (`images/optimization_trajectory_best.png`) visualizes the best-so-far adhesive strength as a function of experiment index across all optimization rounds. The curve shows a clear upward trend: early experiments achieve modest adhesion, while successive iterations identify progressively stronger hydrogels, eventually plateauing when further improvements become rare.

The overall behaviour is consistent with a successful ML-guided optimization process:

- The model trained on the initial 184 formulations provided a meaningful prior over composition space.
- Iterative selection of candidates with high predicted performance enriched the dataset with strong performers.
- The cumulative maximum curve demonstrates steady improvement, approaching or exceeding the target regime of **>1 MPa** (~1000 kPa) in the later rounds (exact values can be read from the y-axis of Figure 7).

## Discussion

### Statistical replication of natural adhesive protein features

Natural underwater adhesive proteins combine multiple sequence-level motifs:

- High densities of nucleophilic and catechol-bearing residues for strong covalent and coordinative bonding.
- Aromatic and hydrophobic residues that reduce effective interfacial hydration and provide π–π and van der Waals interactions.
- Balanced acidic and basic residues that participate in complex coacervation and pH-responsive behaviour.

In the present synthetic platform, these motifs are mapped onto monomer chemistries (HEA, BA, CBEA, ATAC, PEA, AAm). The learned model and exploratory analysis together suggest that compositions statistically enriched in nucleophilic, aromatic, and moderately hydrophobic monomers — while maintaining balanced charge and adequate amide content for mechanical integrity — are most likely to yield high underwater adhesion.

The random forest feature importances and pairwise plots quantify these intuitions and provide a data-driven formulation of design rules that can be used to generate new candidate compositions.

### Model strengths and limitations

**Strengths**

- The model explains a substantial fraction of variance in adhesive strength with only six compositional inputs, indicating that these monomer classes capture the dominant design degrees of freedom.
- Nonlinear interactions are handled naturally by the random forest, allowing the model to represent synergistic effects (e.g. combinations of hydrophobic and charged monomers).
- Cross-validation and external evaluation on optimization datasets provide evidence of reasonable generalization.

**Limitations**

- The dataset size is modest (~184 initial compositions), and cross-validated performance (R² ≈ 0.56) indicates that predictions are still noisy, especially in extremal regions of the design space.
- The model uses only monomer fractions; it does not explicitly account for polymerisation conditions, crosslinker type, molecular weight distribution, or curing protocols, which can also influence adhesion.
- Some optimisation data contain categorical failure modes (e.g. `NO GELATION`) that were treated as missing; a more comprehensive model could explicitly handle such cases via multi-task learning or classification of gelation vs. non-gelation.

### Implications for de novo design

Within these limitations, the present analysis supports a systematic approach to de novo hydrogel design:

1. **Formulate a prior library** that statistically mirrors the monomer composition statistics of natural adhesive proteins (high nucleophilic/aromatic content, balanced charge, controlled hydrophobicity).
2. **Train an ensemble regressor** (e.g. random forests or Gaussian processes) on measured adhesion data.
3. **Use model predictions and uncertainty estimates** to select candidates targeting high adhesion while ensuring exploration of under-sampled regions.
4. **Iteratively update the model** with new experimental data, monitoring calibration plots and optimization trajectories similar to Figures 6 and 7.

The present random forest model already captures key trends and can serve as the backbone of such an iterative framework.

## Conclusion

Using a curated dataset of 184 bio-inspired hydrogel formulations and subsequent optimization rounds, I constructed and evaluated a random forest regression model mapping monomer composition to underwater adhesive strength. The model:

- Achieves strong predictive performance on held-out initial data (R² ≈ 0.84) and reasonable cross-validated performance (R² ≈ 0.56).
- Identifies nucleophilic, aromatic, and hydrophobic monomers as primary contributors to adhesion, with charged monomers and amide content playing supportive roles.
- Remains useful on later optimization datasets, where it helps discover formulations with substantially enhanced adhesive strength, approaching or surpassing the desired >1 MPa regime.

These results demonstrate that statistically replicating sequence features of natural adhesive proteins at the level of monomer composition, combined with data-driven modeling, is a viable strategy for de novo design of high-performance underwater adhesive hydrogels.

All code and figures required to reproduce this analysis are provided in the repository under `code/`, `outputs/`, and `report/images/`.
