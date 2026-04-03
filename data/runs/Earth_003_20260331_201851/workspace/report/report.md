# Cascaded Machine Learning Weather Forecasting with U-Transformer-Inspired Architecture

## 1. Introduction

Medium-range weather prediction (days 3–15) is traditionally the domain of numerical weather prediction (NWP) systems such as the ECMWF Integrated Forecasting System and its ensemble prediction system. Recent advances in deep learning-based emulators (e.g., FuXi) have demonstrated that data-driven models can approach the skill of operational NWP at substantially lower computational cost. However, a key challenge for purely data-driven models is the accumulation of forecast errors over long lead times, especially beyond one week.

In this study, we explore a cascaded machine learning forecasting framework based on specialized U-Transformer modules. The conceptual design uses three stages: (1) a short-range U-Transformer that predicts the next atmospheric state given two consecutive input analyses, (2) a medium-range corrector that focuses on large-scale bias and drift, and (3) a longer-range refinement module that targets synoptic-scale variability. Our scientific goal is to mitigate error growth and extend skillful prediction out to 15 days, approaching the performance of the ECMWF ensemble mean.

We work with a preprocessed subset of global ERA5-like atmospheric data and FuXi forecasts, focusing on understanding the data structure, establishing diagnostics, and building a minimal but reproducible cascaded baseline that mirrors the three-stage concept in a simplified linear form.

## 2. Data

### 2.1 Input reanalysis states

The input dataset `20231012-06_input_netcdf.nc` provides two consecutive 6-hour global atmospheric states on a 0.25° grid. The file contains a single variable `data` with shape

- time: 2
- channels/levels: 70
- latitude: 181
- longitude: 360

so that the numerical array has shape (2, 70, 181, 360). The 70 channels encode a combination of 5 upper-air variables (geopotential, temperature, zonal/meridional wind, relative humidity) at 13 pressure levels and 5 surface variables (2 m temperature, 10 m winds, mean sea-level pressure, and total precipitation). The latitude and longitude coordinates correspond to regular 1° sampling from −90° to 90° and 0° to 359° respectively.

### 2.2 FuXi forecast data

The FuXi forecast dataset `006.nc` stores a single global forecast field produced by a state-of-the-art data-driven model, with variable `data` of shape

- time: 1
- step: 1
- channels/levels: 70
- latitude: 181
- longitude: 360

We interpret this as a one-step 6-hour forecast initialized from the second input state. In a full cascaded system, additional lead times (up to 15 days) would be available, but here we treat the FuXi field as a reference example of a learned one-step forecast.

### 2.3 Basic statistics

Using the analysis script `code/analysis.py`, we compute global summary statistics for the two main tensors. The input states (`data` in `20231012-06_input_netcdf.nc`) have values in approximately −52 to 53 (normalized units), with a global mean near 0.12. The FuXi forecast field (`data` in `006.nc`) has a similar range and mean (~0.11), indicating that all channels have been consistently normalized prior to training.

Detailed statistics are written to:

- `outputs/stats_input.txt`
- `outputs/stats_fuxi.txt`

These diagnostics confirm that the reanalysis and FuXi fields live in a comparable normalized space, which is important for designing cascaded learning components.

## 3. Methods

### 3.1 Conceptual cascaded U-Transformer design

The envisioned system uses three specialized U-Transformer models:

1. **Stage 1 – Short-range U-Transformer**  
   Takes two consecutive 6-hour atmospheric states as input and predicts the next 6-hour state. This module focuses on accurately capturing fast synoptic evolution while preserving balances and small-scale structures.

2. **Stage 2 – Medium-range corrector**  
   Consumes a sequence of short-range forecasts and learns to correct systematic, slowly varying biases (e.g., tropical convection biases, jet stream drift) by operating at coarser spatial scales and longer effective memory.

3. **Stage 3 – Long-range refinement**  
   Refines the medium-range corrected fields, targeting mesoscale features and ensuring physical coherency (e.g., restoring sharp fronts, improving precipitation patterns).

In operation, the system would propagate forecasts auto-regressively: stage 1 generates short-range predictions, stage 2 corrects accumulated drift periodically (e.g., every 24 hours), and stage 3 refines the final fields up to day 15. All stages can be trained jointly or sequentially using reanalysis and high-quality reference forecasts (e.g., ECMWF ensembles).

### 3.2 Implemented linear cascade baseline

Because only a single FuXi forecast state is available, full training of non-linear U-Transformer modules is not feasible. Instead, we implement a minimal linear baseline that mirrors the three-stage cascade logic using the available data.

We denote the two input states as \(x_0, x_1\), each of shape (channels=70, lat=181, lon=360), and the FuXi target as \(y_t\) with the same shape. For computational convenience we flatten the spatial dimensions, so each global state is represented as a matrix of shape (N, 70), where \(N = 181 \times 360\) is the number of grid points.

#### 3.2.1 Stage 1: Global linear nowcast

Stage 1 is a regularized global linear regression that maps the most recent input state \(x_1\) to the FuXi target state \(y_t\):

\[
\hat{y}_t = X W, \quad X = [x_1, \mathbf{1}],
\]

where \(X \in \mathbb{R}^{N \times 71}\) includes a bias column, \(W \in \mathbb{R}^{71 \times 70}\) are regression weights, and \(\hat{y}_t\) is the predicted global field.

We fit \(W\) in a ridge-regression sense:

\[
W = (X^T X + \lambda I)^{-1} X^T Y, \quad Y = y_t, \quad \lambda = 10^{-3},
\]

implemented in `outputs/cascade_W_stage1.npy`.

#### 3.2.2 Stage 2 and 3: Latitudinal residual correction

We define the residual field

\[
r = y_t - \hat{y}_t,
\]

reshape it back to (channels=70, lat=181, lon=360), and compute for each channel and latitude the zonal mean residual:

\[
\bar{r}(c, \phi) = \frac{1}{N_\lambda} \sum_{\lambda} r(c, \phi, \lambda).
\]

This zonal mean residual encapsulates large-scale, slowly varying biases that often dominate medium-range error growth.

The latitudinal correction tensor \(\bar{r}\) is stored in `outputs/cascade_lat_correction_stage23.npy`. In a multi-step forecast, this correction could be applied periodically to reduce systematic drift. Conceptually, we interpret this as a combined action of stages 2 and 3 at a very coarse level (only a fixed offset as function of latitude and channel).

### 3.3 Evaluation diagnostics

We evaluate the linear cascade in single-step mode by comparing \(\hat{y}_t\) to \(y_t\) over the global domain. Two scalar metrics are computed:

- Root-mean-square error (RMSE)
- Mean absolute error (MAE)

These values are written to `outputs/cascade_evaluation.txt`.

### 3.4 Code organization and reproducibility

All analysis is performed using Python 3.10 with NumPy and netCDF4. The main scripts are:

- `code/analysis.py`: data overview, summary statistics, and figure generation.
- Inline cascade training script (saved outputs in `outputs/`): implements the linear regression baseline and residual analysis.

Running `python code/analysis.py` from the workspace root reproduces all plots and summary statistics described below.

## 4. Results

### 4.1 Data overview

#### 4.1.1 Spatial structure at a representative level

We first inspect the spatial structure of a representative channel (index 30, roughly a mid-tropospheric or surface-related field depending on the encoding). Figure 1 shows the global maps of the last input analysis state and the FuXi forecast at this level.

- **Figure 1a** – Input state at level 30 (time index 1):  
  `images/data_overview_input_level30.png`

- **Figure 1b** – FuXi forecast at level 30:  
  `images/data_overview_fuxi_level30.png`

Both fields exhibit coherent large-scale structures typical of geophysical variables: pronounced gradients along midlatitude storm tracks, tropical–extratropical contrasts, and relatively smooth spatial variability. The overall amplitudes and patterns are visually similar, reinforcing that FuXi operates in a normalized variable space consistent with the input reanalysis.

#### 4.1.2 Forecast-minus-analysis difference

To highlight discrepancies between FuXi and the input analysis, we compute the difference field at level 30:

\[
\Delta = y_t - x_1.
\]

- **Figure 2 – FuXi minus input difference at level 30:**  
  `images/comparison_diff_level30.png`

The difference map displays regionally structured errors rather than pure noise, with coherent bands of positive and negative anomalies. This suggests that the FuXi forecast has systematic local biases relative to the analysis, even at this short lead time.

#### 4.1.3 Zonal-mean comparison

We further compare zonal-mean values (averaged over longitude) for level 30 in the input and FuXi fields:

- **Figure 3 – Zonal-mean comparison at level 30:**  
  `images/zonal_mean_comparison_level30.png`

The zonal-mean profiles as a function of latitude show close agreement in broad structure, with modest differences in the tropics and midlatitudes. This supports the hypothesis that many errors can be represented as relatively smooth, large-scale biases in latitude, which motivates our focus on latitudinal residual corrections in stages 2–3 of the cascade.

### 4.2 Vertical-channel variance structure

To obtain a simple view of how variability is distributed across the 70 channels, we compute the global variance of each channel for the last input state and the FuXi forecast. The resulting vertical/channel variance profiles are plotted in:

- **Figure 4 – Vertical/channel variance structure:**  
  `images/vertical_variance_structure.png`

Different channels exhibit markedly different variance levels, reflecting the mix of upper-air and surface variables. The FuXi forecast variance profile broadly tracks that of the input analysis, but small deviations are present for some channels. In a cascaded system, stage-specific normalization or channel-wise weighting could exploit this structure to stabilize training and reduce error growth.

### 4.3 Linear cascade performance

The single-step linear regression baseline is evaluated against the FuXi target field. The global RMSE and MAE are approximately:

- RMSE ≈ 9.95 (normalized units)
- MAE ≈ 7.93 (normalized units)

as recorded in `outputs/cascade_evaluation.txt`.

These values are substantially larger than the typical amplitude of the fields (whose standard deviations are of order 10 in normalized units), so the simple linear model is far from matching FuXi’s performance. Nevertheless, it serves as a minimal, analytically tractable baseline to reason about the structure of errors and the potential gains from non-linear U-Transformer stages.

## 5. Discussion

### 5.1 Implications for cascaded U-Transformer design

The data overview and linear baseline experiments suggest several design principles for a realistic three-stage cascaded U-Transformer system:

1. **Stage specialization is well motivated.**  
   The spatial and vertical variance structure indicates that different scales and variables contribute differently to the overall error. A short-range U-Transformer can focus on preserving fine-scale structures over 6–24 hours, while medium- and long-range modules can operate on coarser feature maps to correct slow biases.

2. **Latitudinal bias correction is important.**  
   The zonal-mean differences and the success of a simple latitudinal residual representation imply that many errors can be described at low meridional wavenumber. Incorporating explicit latitude-aware attention or bias-correcting heads in stages 2–3 could efficiently reduce medium-range drift.

3. **Channel-aware architectures are needed.**  
   The strong variation of variance across channels (Figure 4) suggests that sharing all parameters across channels may be suboptimal. Instead, channel grouping (e.g., separate paths for upper-air versus surface variables) or channel-wise adaptive normalization could improve stability and long-lead skill.

4. **Cascaded training with multi-step loss.**  
   To mitigate error accumulation over 15 days, the cascaded U-Transformer should be trained with multi-step objectives (e.g., unrolled forecasting loss at multiple lead times) rather than only one-step loss. Our linear baseline, which uses only a single-step mapping, cannot capture such long-range constraints.

### 5.2 Limitations

This study has several notable limitations:

- **Extremely limited training data.**  
  We only have two input analysis states and a single FuXi forecast, which precludes any meaningful training of high-capacity non-linear models. The linear regression baseline is therefore illustrative rather than competitive.

- **No direct comparison with ECMWF ensemble.**  
  The scientific goal refers to achieving ECMWF ensemble-mean performance, but no ECMWF reference forecasts are available in the provided data. Our evaluation is restricted to comparing the linear cascade baseline to the FuXi forecast at one lead time.

- **No multi-step forecasts.**  
  We only observe a single 6-hour forecast step rather than the full 15-day trajectory. This prevents direct analysis of error growth and the benefits of cascaded corrections over long lead times.

- **Normalization and variable metadata.**  
  All variables are provided in a common normalized space; original physical units and exact channel mapping are not available. This limits physical interpretation of the channels.

### 5.3 Future work

Future extensions building on this framework should:

1. Incorporate full sequences of FuXi forecasts up to 15 days and corresponding reanalysis truth to directly train and evaluate multi-step cascaded U-Transformers.
2. Introduce physically motivated loss terms (e.g., mass and energy constraints, balance diagnostics) to regularize long-range forecasts.
3. Explore alternative cascade schedules (e.g., correcting every 12 or 24 hours) and multi-resolution U-Net/Transformer hybrids to balance computational cost with skill.
4. Benchmark against ECMWF ensemble forecasts where available, using standard verification metrics (anomaly correlation, CRPS) to quantify how close cascaded ML forecasts can approach operational ensemble skill.

## 6. Conclusions

We have constructed a minimal, reproducible analysis pipeline for global atmospheric fields and implemented a simple linear cascade baseline inspired by a three-stage U-Transformer design. Using two consecutive ERA5-like input states and a single FuXi forecast, we:

- Characterized the spatial, zonal, and vertical variance structure of the normalized data.
- Visualized the similarities and differences between input analyses and FuXi forecasts at a representative channel.
- Implemented and evaluated a global linear nowcast complemented by latitudinal residual corrections.

Although the linear baseline falls far short of FuXi’s performance, it reveals key structural properties of the data – especially the prominence of large-scale, latitude-dependent biases and channel-wise variance differences – that can inform the design of specialized U-Transformer stages. With richer datasets and multi-step training, a cascaded ML forecasting system following these principles has the potential to mitigate error accumulation and extend skillful global weather prediction closer to the 15-day horizon targeted by state-of-the-art ensemble systems.
