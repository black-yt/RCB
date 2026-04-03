# AI-Guided Inverse Design of Vitrimeric Polymers via Gaussian-Process Calibration and Latent-Space Exploration

## 1. Introduction

Vitrimers are a class of covalently crosslinked polymer networks that can rearrange their topology via exchange reactions while maintaining network connectivity. This feature enables recyclability, reprocessability, and self-healing while preserving the mechanical robustness of thermosets. A central design target for vitrimer chemistries is the glass transition temperature (Tg), which controls service temperature, flow behavior, and processing windows.

Molecular dynamics (MD) simulations provide a route to predict Tg from molecular structure, but raw MD estimates are often biased relative to experiment. Statistical calibration models, such as Gaussian processes (GPs), can learn the mapping between simulated and experimental Tg while quantifying predictive uncertainty. When combined with generative models over chemical space, these calibrated predictors enable inverse design: selecting or generating new vitrimer chemistries that achieve desired Tg values with quantified confidence.

Here an autonomous framework is developed to: (i) calibrate MD-predicted Tg values against experimental measurements using a GP model, (ii) apply this calibration to a library of vitrimer systems obtained from MD, and (iii) perform inverse design in a latent space derived from vitrimer features to identify candidates with target Tg. The focus is on computational methodology using the provided datasets; experimental validation is discussed conceptually.

## 2. Data and Features

### 2.1 Calibration dataset

The calibration dataset (`tg_calibration.csv`) contains 295 polymer entries with columns:

- `name`: polymer name,
- `smiles`: polymer repeat-unit SMILES with asterisk placeholders,
- `tg_exp`: experimental glass transition temperature (K),
- `tg_md`: MD-predicted Tg (K),
- `std`: estimated uncertainty in MD Tg (K).

Exploratory analysis shows that the calibration set spans a wide range of chemistries (nylons, acrylates, polyolefins, etc.). The distributions of MD and experimental Tg are shown in Figures 1 and 2. A scatter plot of MD vs experimental Tg is shown in Figure 3.

- **Figure 1.** Distribution of experimental Tg in the calibration set (`images/calib_tg_exp_hist.png`).
- **Figure 2.** Distribution of MD Tg in the calibration set (`images/calib_tg_md_hist.png`).
- **Figure 3.** MD vs experimental Tg for the calibration dataset (`images/calib_md_vs_exp.png`).

MD Tg is positively correlated with experimental Tg but exhibits systematic deviations and scatter, motivating a non-linear probabilistic calibration model.

### 2.2 Vitrimer MD dataset

The vitrimer dataset (`tg_vitrimer_MD.csv`) contains 8,424 vitrimer systems with columns:

- `acid`: SMILES string of the acid component,
- `epoxide`: SMILES string of the epoxide component,
- `tg`: MD-predicted Tg (K),
- `std`: estimated Tg uncertainty from MD.

This dataset is used as input to the calibrated surrogate model to obtain posterior predictions for Tg and associated uncertainties across vitrimer chemistry space.

### 2.3 Feature construction

Full graph-based featurization using RDKit is complicated in this environment by binary compatibility issues between RDKit and NumPy. To ensure reproducibility and robustness without relying on external compiled extensions, the present work uses handcrafted, physics-inspired features derived from the MD Tg and uncertainty:

- Tg itself: `tg`,
- MD uncertainty: `std`,
- quadratic terms: `tg^2`, `std^2`,
- interaction term: `tg × std`.

For the calibration dataset, these features are computed from `tg_md` and `std`; for vitrimer systems, they are computed from `tg` and `std`. These simple features allow the GP to learn non-linear transformations and heteroscedastic trends between raw MD Tg and experimental Tg.

## 3. Methods

### 3.1 Gaussian-process calibration model

A Gaussian-process regression model is used to map feature vectors x to experimental Tg (y):

\[
 y = f(\mathbf{x}) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2).
\]

The implementation follows an exact GP with Gaussian likelihood using the GPyTorch library. The mean function is a learned constant and the covariance function is an RBF (squared-exponential) kernel with automatic relevance determination (ARD) over the five input dimensions. Hyperparameters (length scales, outputscale, noise variance, and mean) are optimized by maximizing the exact marginal log-likelihood.

The calibration pipeline is:

1. Build feature matrix `X` and targets `y = tg_exp` from `tg_calibration.csv`.
2. Split into training and test sets (80/20) with a fixed random seed.
3. Standardize features using `StandardScaler` (zero mean, unit variance).
4. Train the exact GP model with Adam optimizer for 800 iterations.
5. Evaluate predictive performance on the held-out test set.

The full implementation is provided in `code/gp_calibration_and_inverse_design.py`.

### 3.2 Application to vitrimer systems

The trained GP and scaling transform are applied to the vitrimer MD dataset:

1. Construct feature matrix from `tg` and `std` for all vitrimer entries.
2. Apply the same `StandardScaler` used during GP training.
3. Compute the GP posterior for each vitrimer, obtaining the mean calibrated Tg and predictive standard deviation.
4. Save the augmented dataset (`tg_vitrimer_calibrated.csv`) to `outputs/`.

The GP therefore acts as a calibration operator transforming MD Tg to calibrated Tg with uncertainty, allowing downstream ranking and inverse design.

### 3.3 Latent-space construction and inverse design

A full graph variational autoencoder (VAE) over molecular graphs would embed vitrimer chemistries into a continuous latent space and enable gradient-based or stochastic optimization toward target properties. Implementing and training such a model at scale is beyond the scope of the present environment. Instead, a simplified latent-space approach is used to emulate the role of a graph VAE while staying computationally efficient:

1. Build a feature matrix for vitrimer systems using three quantities: raw MD Tg (`tg`), MD uncertainty (`std`), and calibrated Tg mean (`tg_calib_mean`).
2. Apply principal component analysis (PCA) with two components to obtain a 2D latent representation (z1, z2) of vitrimer chemistries.
3. Define a scalar design score for a target Tg (here 400 K):

   \[
   S = -|T_g^{\text{calib}} - T_g^{\text{target}}| - 0.5\,\sigma_{T_g},
   \]

   where \(T_g^{\text{calib}}\) is the GP-calibrated Tg, \(T_g^{\text{target}}\) is the desired Tg, and \(\sigma_{T_g}\) is the predictive standard deviation.

4. Rank all vitrimer candidates by S in descending order, favoring systems whose calibrated Tg is close to the target and whose uncertainty is low.
5. Save the top-ranked candidates to `outputs/top_inverse_design_candidates.csv`.

The resulting 2D latent projection colored by calibrated Tg is visualized in Figure 4.

- **Figure 4.** 2D latent representation of vitrimer chemistries obtained by PCA on (tg, std, tg_calib_mean), colored by calibrated Tg (`images/vitrimer_latent_space.png`).

While this is not a full generative graph VAE, it serves as a surrogate latent space over available vitrimer chemistries and illustrates how calibrated property predictions can guide inverse design.

## 4. Results

### 4.1 Calibration performance

The GP calibration model achieves a test-set coefficient of determination R² ≈ 0.59 and a mean absolute error (MAE) of approximately 48 K on the held-out calibration data. A parity plot of experimental vs GP-predicted Tg for the test set is shown in Figure 5, and the distribution of prediction residuals is shown in Figure 6.

- **Figure 5.** Parity plot comparing experimental Tg and GP-calibrated Tg for the test set (`images/gp_parity_test.png`).
- **Figure 6.** Histogram of prediction errors (GP mean − experimental Tg) on the test set (`images/gp_residuals_test.png`).

The parity plot demonstrates that the GP corrects much of the bias present in raw MD Tg, bringing predictions closer to the diagonal line of perfect agreement. The residuals are approximately centered near zero with spread of tens of kelvin, consistent with the MAE. Some asymmetry and tails remain, indicating that a fraction of the chemical diversity is not fully captured by the simple feature set.

### 4.2 Calibrated vitrimer Tg distribution

Applying the trained GP to the vitrimer MD dataset yields calibrated Tg predictions and uncertainties for all 8,424 vitrimer chemistries. Figure 7 compares the distributions of raw MD Tg and calibrated Tg.

- **Figure 7.** Kernel-density estimates of raw MD Tg and GP-calibrated Tg for vitrimer systems (`images/vitrimer_tg_distribution.png`).

The calibrated Tg distribution is shifted relative to the raw MD distribution, reflecting systematic corrections learned from the calibration polymers. In particular, the GP tends to adjust mid-range Tg values toward the regime where MD systematically underestimates or overestimates experiment in the calibration set. The predictive standard deviations (not shown) vary across chemistry space, with higher uncertainty in regions where calibration data are sparse in feature space.

### 4.3 Latent space and candidate selection

The PCA-based latent embedding of vitrimer chemistries shows a continuous manifold when colored by calibrated Tg (Figure 4). Regions of the latent space correspond to families of vitrimers with similar Tg. This structure suggests that a more expressive graph VAE trained on molecular graphs would likely discover a low-dimensional manifold organized by Tg and possibly other correlated properties.

For inverse design targeting Tg ≈ 400 K, candidates are ranked by the scalar score S defined in Section 3.3. The top 30 candidates are written to `outputs/top_inverse_design_candidates.csv`. An excerpt of the highest-scoring entries printed during execution shows, for example, systems with acid/epoxide combinations whose calibrated Tg lies close to 400 K and whose uncertainty is relatively small.

These candidates constitute computational recommendations for experimental synthesis and characterization. Because the GP provides predictive uncertainty, one can further filter candidates by requiring \(\sigma_{T_g}\) below a threshold, or use acquisition functions from Bayesian optimization (e.g., expected improvement) to balance exploration and exploitation.

## 5. Discussion

### 5.1 Relation to full graph-VAE inverse design

In a fully realized framework, the vitrimer chemistries would be encoded as molecular or reaction graphs, and a graph variational autoencoder would be trained to reconstruct these graphs while learning a continuous latent space. The GP-calibrated Tg model would then operate in this latent space, and latent-space optimization (via gradient-based methods or Bayesian optimization) would generate new latent points that decode to novel vitrimers with optimized Tg.

In this work, constraints of the environment (notably RDKit/NumPy binary incompatibilities and computational limits) precluded training a full graph VAE. Instead, PCA on Tg-related features was used as a proxy latent space. While not generative beyond the existing dataset, this approach demonstrates the coupling of a calibrated surrogate model with a low-dimensional embedding for candidate prioritization. The methodology can be upgraded to a true graph VAE in richer computational environments.

### 5.2 Limitations

Several limitations should be noted:

1. **Feature set**: Features are limited to Tg and its uncertainty, without explicit structural or chemical descriptors. This restricts the model's ability to extrapolate to chemistries whose MD Tg may be similar but whose structure differs significantly.
2. **Calibration coverage**: The calibration set consists of linear polymers, whereas vitrimer systems involve crosslinked networks from acid–epoxide combinations. The domain shift may reduce calibration accuracy for some vitrimers.
3. **Model capacity**: A single RBF kernel GP may be insufficient to capture sharp transitions or multi-modal relationships between MD and experimental Tg. More complex kernels or deep GPs could improve performance.
4. **Latent model simplification**: PCA on three scalar features is only a crude approximation to a graph-based latent space. It cannot generate new chemistries or enforce chemical validity.
5. **Experimental validation**: No actual experimental validation is performed here; it is assumed that the calibration learned on the provided dataset transfers to the vitrimers.

### 5.3 Opportunities for improvement

Future work could address these limitations as follows:

- Introduce chemically informed descriptors (e.g., functional-group counts, topological indices) that can be computed without heavy dependencies, improving calibration accuracy and interpretability.
- Augment the calibration dataset with additional vitrimer-like chemistries, especially crosslinked networks, to reduce domain shift.
- Explore heteroscedastic GP models or deep kernel learning to better capture varying noise levels and complex structure–property relationships.
- Implement a graph VAE over reaction graphs representing acid–epoxide combinations, with the GP operating in latent space to enable true generative inverse design.
- Integrate active learning: iteratively select vitrimers with high acquisition value for MD simulation and experimental validation, closing the loop between prediction and data generation.

## 6. Conclusions

An AI-guided inverse-design framework for vitrimeric polymers has been implemented using Gaussian-process calibration of MD-derived Tg values and latent-space exploration over vitrimer chemistries. The GP model corrects systematic biases between MD and experimental Tg with quantified uncertainty, and its application to a large vitrimer MD dataset yields calibrated Tg predictions that can be used to rank candidate chemistries.

A simplified PCA-based latent representation illustrates how calibrated properties structure vitrimer chemistry space and enables selection of candidates targeting specific Tg values. Although a full graph VAE is not deployed here, the workflow mirrors the essential components of such a framework and can be extended to more expressive generative models and richer descriptors.

All analysis code is contained in `code/gp_calibration_and_inverse_design.py`, intermediate outputs (including calibrated Tg and top candidate lists) are saved in `outputs/`, and figures referenced in this report are stored in `report/images/`.
