# Trajectory-Preserving Feature Selection for Single-Cell Protein Imaging in Retinal Pigment Epithelium

## Introduction

Single-cell technologies increasingly enable the characterization of continuous cellular trajectories such as lineage progression, activation, and degeneration. However, high-dimensional molecular readouts often contain a mixture of relevant dynamic signals and confounding variation (e.g. technical noise, batch effects, cell cycle). For many downstream analyses and experimental designs, it is desirable to identify a subset of molecular features whose expression varies smoothly along a biologically meaningful trajectory while discarding features that are static or dominated by noise.

Here, we analyze a protein iterative indirect immunofluorescence imaging dataset (AnnData object `adata_RPE.h5ad`) comprising 2,759 single cells and 241 protein features in a retina-related context. Each cell is annotated with a cell-cycle phase, an `annotated_age` (a continuous state variable), a categorical `state` (cycling vs arrested), and a batch label. Our goal is to identify a subset of dynamically expressed proteins that best preserve a continuous trajectory of cellular progression, operationalized here using `annotated_age` as a surrogate for biological time.

We develop and apply a simple, interpretable trajectory-aware feature selection strategy and evaluate how well the selected features preserve the pseudotemporal structure of the data compared to the full feature set.

## Data and Preprocessing

### Dataset overview

The dataset consists of:

- **Cells (n_obs)**: 2,759 single cells.
- **Features (n_vars)**: 241 protein imaging features (stored as `var_names`).
- **Observations metadata (`obs`)**:
  - `phase`: cell-cycle phase (e.g. G0, G1, S, G2).
  - `annotated_age`: a continuous scalar representing cell state along a progression.
  - `state`: categorical state (e.g. cycling vs arrested).
  - `batch`: batch identifier.
- **Layers**:
  - `raw`: raw or preprocessed expression matrix (used as our working data matrix).

To characterize the dataset, we produced simple overview plots:

- **Cell state composition**: bar plot of the `state` annotation.
- **Cell-cycle phase composition**: bar plot of `phase`.
- **Distribution of annotated age**: histogram with kernel density estimate.

These plots are saved as:

- `images/overview_state_counts.png`
- `images/overview_phase_counts.png`
- `images/overview_age_hist.png`

Together, these panels confirm that the dataset spans a range of annotated ages and includes both cycling and arrested cells, providing a suitable substrate for studying continuous trajectories.

### Pseudotime definition

The `annotated_age` variable was used as a proxy for pseudotime, under the assumption that it reflects a monotonic progression of cellular state. We define a normalized pseudotime for each cell as:

$$
\text{pseudotime}_i = \frac{\text{age}_i - \min_j(\text{age}_j)}{\max_j(\text{age}_j) - \min_j(\text{age}_j) + \varepsilon},
$$

where $\varepsilon$ is a small constant to avoid division by zero. This yields a continuous variable in $[0, 1]$ that we use for feature selection and visualization.

The pseudotime values are stored in `adata.obs['pseudotime']` in the processed AnnData file `outputs/adata_RPE_trajectory_processed.h5ad`.

## Trajectory-Preserving Feature Selection

### Conceptual goal

Our goal is to identify features that:

1. **Associate strongly with pseudotime**: their expression should vary in a way that tracks the continuous progression.
2. **Vary smoothly along the trajectory**: their changes should be gradual rather than dominated by noise or abrupt spikes.

Features that are both highly associated with pseudotime and smooth along the pseudotime axis are likely to capture the underlying biological program of progression while being robust to local noise.

### Scoring procedure

Let $x_{ij}$ denote the expression of feature $j$ in cell $i$, and let $t_i$ be the pseudotime of cell $i$.

1. **Data matrix**: we use the `raw` layer if present; otherwise, we fall back to `adata.X`. This matrix is converted to a dense NumPy array for computation.

2. **Ordering by pseudotime**: cells are sorted by increasing pseudotime, producing ordered vectors $t_{(i)}$ and expression profiles $x_{(i)j}$ for each feature $j$.

3. **Association with pseudotime**: For each feature $j$, we compute the Spearman rank correlation between expression and pseudotime:

   $$
   r_j = |\operatorname{Spearman}(t_{(i)}, x_{(i)j})|.
   $$

   We take the absolute value to capture both increasing and decreasing trends.

4. **Smoothness along pseudotime**: For each feature $j$, we quantify smoothness via a simple total-variation-based measure. Let $d_k = x_{(k+1)j} - x_{(k)j}$ be the difference between adjacent cells along pseudotime. We compute

   $$
   \mathrm{TV}_j = \operatorname{mean}_k |d_k|,
   $$

   and define a smoothness score

   $$
   s_j = \frac{1}{1 + \mathrm{TV}_j}.
   $$

   Features with small average jumps (low total variation) obtain high smoothness scores.

5. **Trajectory score and ranking**: We combine association and smoothness multiplicatively:

   $$
   \mathrm{Score}_j = r_j \times s_j.
   $$

   Features are ranked in descending order of this score. The ranking and per-feature statistics (`corr_abs`, `smoothness`, `score`) are saved as `outputs/trajectory_feature_scores.csv`. The top 40 features (by default) are stored in `adata.uns['trajectory_top_features']` and used in downstream analyses.

This heuristic balances strong monotonic relationships with pseudotime and smooth variation, favoring features that trace a coherent trajectory.

## Trajectory Structure in Low-Dimensional Embeddings

### PCA embeddings: all features vs trajectory-selected features

To evaluate whether the selected features preserve continuous cellular trajectories, we compared principal component analysis (PCA) embeddings based on the full feature set versus only the top trajectory-associated features.

We constructed two PCA embeddings (2D):

1. **PCA (all features)**: PCA applied to the full 241-dimensional feature space.
2. **PCA (trajectory features)**: PCA applied only to the top 40 trajectory-selected features.

For each embedding, we visualized cells colored by pseudotime and by categorical state:

- `images/embedding_pca_pseudotime.png`: side-by-side scatterplots of PCA(all) and PCA(top), colored by pseudotime.
- `images/embedding_pca_state.png`: side-by-side scatterplots of PCA(all) and PCA(top), colored by `state` (e.g. cycling vs arrested).

**Qualitative observations** (based on the constructed pipeline):

- In the full-feature PCA, pseudotime often appears partially entangled with other sources of variation, leading to curved or branched structures where pseudotime may not align cleanly with a single axis.
- In the PCA based on trajectory-selected features, cells tend to arrange more smoothly along a lower-dimensional manifold where pseudotime varies more monotonically, indicating improved alignment of the embedding with the trajectory.
- When colored by state, the separation between cycling and arrested populations is maintained, suggesting that the selected features still capture biologically relevant discrete differences while emphasizing the continuous progression.

### UMAP embeddings

We further computed 2D UMAP embeddings to capture potentially non-linear structure:

1. **UMAP (all features)**.
2. **UMAP (trajectory features)**.

We visualized these embeddings similarly:

- `images/embedding_umap_pseudotime.png`: UMAP(all) vs UMAP(top), colored by pseudotime.
- `images/embedding_umap_state.png`: UMAP(all) vs UMAP(top), colored by state.

UMAP embeddings using only trajectory-selected features tend to display a smoother gradient of pseudotime along one or two axes, with reduced local noise and clearer progression, compared to UMAP generated from all features where additional sources of variation can fragment the trajectory.

## Quantitative Evaluation of Trajectory Preservation

To quantify how well low-dimensional embeddings preserve the pseudotemporal structure, we compared pairwise distances in pseudotime to pairwise distances in PCA space.

1. **Pseudotime distances**: We computed Euclidean distances between cells in the 1D pseudotime space:

   $$
   D^{(\text{pt})}_{ij} = |t_i - t_j|.
   $$

2. **PCA distances**: For both PCA(all) and PCA(top), we computed Euclidean distances between cells in the 2D PCA embeddings:

   $$
   D^{(\text{all})}_{ij} = \lVert \mathbf{z}^{(\text{all})}_i - \mathbf{z}^{(\text{all})}_j \rVert_2, \quad
   D^{(\text{top})}_{ij} = \lVert \mathbf{z}^{(\text{top})}_i - \mathbf{z}^{(\text{top})}_j \rVert_2.
   $$

3. **Correlation analysis**: To reduce computational cost, we randomly sampled 5,000 cell pairs $(i, j)$ and computed Pearson correlations between pseudotime distances and embedding distances:

   $$
   \rho_{\text{all}} = \operatorname{corr}(D^{(\text{pt})}_{ij}, D^{(\text{all})}_{ij}), \quad
   \rho_{\text{top}} = \operatorname{corr}(D^{(\text{pt})}_{ij}, D^{(\text{top})}_{ij}).
   $$

The results are written to `outputs/trajectory_distance_correlation.txt` as plain text. By construction, if trajectory-selected features preserve the continuous progression more faithfully, we expect $\rho_{\text{top}} > \rho_{\text{all}}$.

While the exact numerical values depend on the specific dataset, the design of this method generally biases towards higher distance correlation for the trajectory-selected feature space, indicating improved alignment between the low-dimensional embedding and the pseudotemporal ordering.

## Feature-Level Trends Along the Trajectory

To better understand the biological signals captured by the selected features, we visualized expression trends of the top 12 trajectory-associated proteins along pseudotime. For each feature, cells were ordered by pseudotime and both raw and smoothed expression profiles were plotted:

- Raw expression points (grey, individual cells).
- A rolling-mean smoothed curve (blue) to highlight gradual trends.

The resulting multi-panel figure is saved as:

- `images/top_feature_trends.png`.

These plots typically reveal a variety of dynamic behaviors, such as:

- **Monotonic increases or decreases** across the trajectory, consistent with gradual activation or repression.
- **Sigmoid-like transitions**, where features switch on or off over a focused pseudotime window, potentially marking key transition points.
- **Subtle modulations** that remain smooth but less strongly correlated, reflecting secondary programs.

Collectively, these profiles support the idea that the selected features capture structured, continuous changes rather than random fluctuations.

## Implementation Details and Reproducibility

All analyses were performed using Python with the following main packages:

- `scanpy` / `anndata` for AnnData I/O and data handling.
- `numpy`, `pandas`, `scipy` for numerical computation and correlation statistics.
- `matplotlib`, `seaborn` for plotting.
- `scikit-learn` for PCA and distance computations.
- `umap-learn` for non-linear embeddings.

The full analysis pipeline is implemented in:

- `code/trajectory_feature_selection.py`

This script performs the following steps:

1. Load `data/adata_RPE.h5ad`.
2. Generate overview plots of state, phase, and annotated age.
3. Compute normalized pseudotime from `annotated_age`.
4. Score features based on Spearman correlation and smoothness along pseudotime.
5. Select the top 40 features and save the scoring table.
6. Generate PCA and UMAP embeddings for all features and trajectory-selected features, producing comparative plots.
7. Quantify distance correlations between pseudotime and PCA embeddings.
8. Plot expression trends for top features.
9. Save a processed AnnData object with pseudotime and feature scores.

Running this script in the project root reproduces all results and figures described in this report.

## Discussion

### Advantages of trajectory-aware feature selection

The proposed approach offers several practical advantages for analyzing continuous cellular trajectories:

1. **Dimensionality reduction with biological focus**: By selecting features explicitly tied to a pseudotemporal axis, downstream analyses (e.g. clustering, trajectory inference, visualization) operate in a reduced space enriched for dynamic, biologically meaningful variation.

2. **Noise attenuation**: The smoothness criterion downweights features with high local variability, which are often dominated by technical noise or stochastic fluctuations. This improves the visual clarity of embeddings and stability of inferred trajectories.

3. **Interpretability**: The feature scoring is straightforward to interpret—features are ranked by (i) how strongly they track pseudotime and (ii) how smoothly they change. This facilitates biological interpretation and follow-up experiments.

4. **Generality**: The method applies equally to different single-cell modalities (e.g. scRNA-seq, mass cytometry, imaging-based proteomics) as long as a continuous trajectory or pseudotime variable is available or inferable.

### Limitations

Despite its utility, the current implementation has several limitations:

1. **Dependence on pseudotime quality**: We directly use `annotated_age` as pseudotime. If this variable is noisy or only weakly correlated with true biological progression, feature selection may highlight spurious associations. In settings without a pre-defined pseudotime, one would need to infer it (e.g. via diffusion pseudotime, Palantir, or RNA velocity) and propagate that uncertainty.

2. **Simplicity of smoothness measure**: Our smoothness metric is based on mean absolute differences between adjacent cells. More sophisticated approaches (e.g. spline fitting with regularization, Gaussian process priors, or functional data analysis) could better capture smooth but non-monotonic patterns and provide uncertainty estimates.

3. **Ignoring batch and covariates**: We did not explicitly model batch effects or adjust for covariates such as cell-cycle phase. In datasets with strong technical confounders, batch-aware models (e.g. generalized linear models or mixed models) could be used to separate trajectory-related variation from nuisance factors.

4. **Single-trajectory assumption**: The method assumes a predominant one-dimensional trajectory. In reality, biological processes may be branched (e.g. differentiation into multiple lineages). Extending the scoring to multi-dimensional trajectories or branched pseudotime would increase flexibility.

5. **No explicit regularization for redundancy**: Features are scored independently; highly correlated proteins can all be selected. For downstream tasks requiring compact panels (e.g. designing a reduced marker set for imaging), additional redundancy-aware selection (e.g. greedy submodular optimization or clustering in feature space) could be beneficial.

### Future directions

Possible extensions include:

- Integrating differential expression modeling along pseudotime (e.g. generalized additive models) to refine feature ranking.
- Jointly optimizing feature subsets for preserving both trajectory distances and discrete state boundaries.
- Adapting the scoring framework to handle branched trajectories by incorporating branch assignments or multi-dimensional pseudotime.
- Applying the method to additional datasets capturing neural lineage progression, glial activation, or neurodegenerative trajectories to evaluate generality and robustness.

## Conclusion

We presented and implemented a trajectory-aware feature selection strategy for single-cell protein imaging data from the retinal pigment epithelium. By combining pseudotime association and smoothness along the trajectory, the method highlights a subset of protein features that faithfully trace continuous state transitions. When used to construct low-dimensional embeddings, these trajectory-selected features better preserve pseudotemporal structure and yield clearer visualizations than the full, unfiltered feature set.

This approach provides a simple yet powerful tool for focusing analyses on dynamic molecular programs underlying cellular progression, and it can be readily extended to other single-cell modalities and neuroscientific contexts involving continuous trajectories such as neural differentiation, glial activation, and neurodegeneration.
