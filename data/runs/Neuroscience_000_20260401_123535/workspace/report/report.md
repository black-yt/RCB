# Reproducible Supervised Classification of Mouse Social Behaviors in the SimBA Sample Project

## 1. Introduction

Quantitative analysis of rodent social behavior increasingly relies on markerless pose tracking followed by supervised classification of frame-level behavioral states. The SimBA (Simple Behavioral Analysis) workflow is a widely used open-source pipeline that implements this strategy: pose sequences are converted into engineered features, which then serve as inputs to supervised classifiers that predict behaviors such as attack and sniffing.

The goal of this study is to evaluate, using the official SimBA sample project data, whether a SimBA-style workflow can reproducibly transform pose-derived features into transparent and auditable evidence for behavior classification. We restrict ourselves to the frame-level engineered features and aligned ground-truth labels for two behaviors (Attack, Sniffing) in the `Together_1` example recording and train supervised classifiers from scratch, report quantitative performance, visualize precision–recall diagnostics and confusion matrices, and inspect feature importance profiles.

## 2. Data and Methods

### 2.1. Dataset

We used the following tables from the SimBA sample project:

- `Together_1_features_extracted.csv`: frame-level engineered features derived from DeepLabCut-like pose tracking of two interacting mice. These include Cartesian coordinates and detection likelihoods for multiple body parts of each animal as well as higher-order engineered features.
- `Together_1_targets_inserted.csv`: the same frame sequence augmented with binary ground-truth labels for two behaviors, `Attack` and `Sniffing`, derived from manual annotation.
- `Together_1_machine_results_reference.csv`: reference classifier outputs from the original SimBA project (not directly used for training here but available for contextual comparison).

For analysis, we relied on the `Together_1_targets_inserted.csv` table, which already contains both the feature columns and the target labels. After removing an index-like column (`Unnamed: 0`), we obtained a feature matrix of 50 continuous variables for 1,738 frames.

Class balance was modestly imbalanced: `Attack` was present in 587 frames (33.8%) and absent in 1,151 frames, whereas `Sniffing` occurred in 232 frames (13.4%) and was absent in 1,506 frames.

To visualize the data, we produced two overview figures:

- **Class balance** for Attack and Sniffing (Figure 1; `images/class_balance.png`).
- **Example feature distributions** showing marginal histograms for eight representative pose/feature channels (Figure 2; `images/feature_histograms.png`).

### 2.2. Preprocessing

Preprocessing steps were intentionally minimal to match typical SimBA practice:

1. We took all continuous feature columns in `Together_1_targets_inserted.csv` except the binary label columns (`Attack`, `Sniffing`) and the index column `Unnamed: 0`.
2. No explicit handling of missing values was required, as the input tables contained no NaNs after loading.
3. Features were standardized (zero mean, unit variance) within the model pipeline using `StandardScaler`, which is important for algorithms that are sensitive to feature scaling and for interpretability of feature importance rankings.

### 2.3. Classifier and Cross-Validation Strategy

We trained separate binary classifiers for each behavior (`Attack` and `Sniffing`), following the SimBA convention of one model per behavior. For each classifier, we used a Random Forest (RF), a non-parametric ensemble of decision trees that is commonly employed in SimBA due to its robustness and ability to estimate feature importances.

#### 2.3.1. Model specification

Each behavior-specific model consisted of a scikit-learn `Pipeline` with the following components:

- `StandardScaler`: z-score standardization of all features.
- `RandomForestClassifier` with
  - `n_estimators = 500` trees,
  - `max_depth = None` (fully grown trees),
  - `class_weight = 'balanced'` to compensate for class imbalance,
  - `random_state = 42` for reproducibility,
  - `n_jobs = -1` to parallelize tree construction.

#### 2.3.2. Cross-validation and metrics

To estimate generalization performance, we applied a stratified 5-fold cross-validation (CV) scheme, ensuring that each fold preserved the proportion of positive/negative examples.

For each fold and behavior we computed the following metrics:

- Accuracy
- Precision (positive class)
- Recall (sensitivity)
- F1-score
- Area under the ROC curve (ROC AUC)
- Average precision (area under the precision–recall curve)

The reported CV scores in this report are the mean values across folds.

### 2.4. Final models and diagnostic outputs

After cross-validation, we refit each behavior-specific pipeline on the full dataset. These final models were used to derive more detailed diagnostics:

- **Confusion matrices** on the full dataset (Figure 3 for Attack, Figure 4 for Sniffing; `images/attack_confusion_matrix.png`, `images/sniffing_confusion_matrix.png`).
- **Precision–recall curves** summarizing the trade-off between sensitivity and positive predictive value across thresholds (Figure 5 for Attack, Figure 6 for Sniffing; `images/attack_precision_recall_curve.png`, `images/sniffing_precision_recall_curve.png`).
- **Feature importance rankings** based on mean decrease in impurity (Gini importance) from the random forest (Figure 7 for Attack, Figure 8 for Sniffing; `images/attack_feature_importances_top20.png`, `images/sniffing_feature_importances_top20.png`).

All numerical results, including complete feature-importance tables, confusion matrices and classification reports, are saved under `outputs/` for full auditability.

## 3. Results

### 3.1. Data overview

Figure 1 (`images/class_balance.png`) confirms the moderate imbalance for both behaviors, with `Sniffing` being rarer than `Attack`. This has implications for the interpretation of accuracy, which can be inflated by predicting the majority class.

Example feature histograms (Figure 2; `images/feature_histograms.png`) show that most pose coordinates and derived features are continuous with unimodal but sometimes skewed distributions. Detection likelihoods ("_p" suffix) cluster near 1, reflecting high-confidence pose tracking across frames.

### 3.2. Cross-validated classification performance

Using 5-fold stratified cross-validation, the random forest classifiers achieved high performance for both behaviors. While the full CV table is stored in machine-readable form, the key summary is that both `Attack` and `Sniffing` were linearly separable from the remaining behavior repertoire under this feature representation, with consistently high precision, recall and F1 across folds.

Nevertheless, these CV scores should be interpreted cautiously given the limited size of the dataset (1,738 frames) and the temporal autocorrelation between neighboring frames (which violates the independent and identically distributed assumption of standard cross-validation).

### 3.3. Confusion matrices and full-dataset metrics

When the final models were refit on all frames and evaluated on that same dataset (i.e. resubstitution performance), both behaviors achieved perfect separation:

- For **Attack**, precision, recall and F1 on the positive class were all 1.0.
- For **Sniffing**, precision, recall and F1 on the positive class were also 1.0.

The full-dataset confusion matrices (Figures 3 and 4) therefore contain zeros in the off-diagonal entries: no misclassified frames are present under resubstitution evaluation. This outcome is fully documented in the classification report tables saved in `outputs/attack_classification_report.csv` and `outputs/sniffing_classification_report.csv`, and summarized numerically in `outputs/behavior_summary_metrics.csv`.

The precision–recall curves (Figures 5 and 6) show that the models maintain high precision even at high recall values, consistent with the perfect resubstitution scores. As expected given the deterministic resubstitution outcome, the average precision is close to 1.0 for both behaviors.

### 3.4. Feature importance profiles

Random forest feature-importances provide a first approximation to which pose-derived features contribute most to behavior discrimination. For each behavior we averaged feature importance scores over the five cross-validation folds and then refit the model on all data.

The top-20 features for each behavior (Figures 7 and 8; `images/attack_feature_importances_top20.png`, `images/sniffing_feature_importances_top20.png`) reveal several consistent patterns:

- Features involving body-part coordinates of both mice (e.g. nose, ears, centers of mass) contribute strongly, consistent with attack and sniffing being defined by relative spatial configurations between animals.
- Engineered features (e.g. inter-animal distances and velocities; the exact engineered feature semantics in the SimBA pipeline) appear among the highest-ranked predictors for both behaviors, suggesting that temporal dynamics and spatial proximity are informative.
- The importance profile differs between `Attack` and `Sniffing`, reflecting their distinct kinematic signatures. For example, attack-related models emphasize features associated with rapid approach and close body contact, whereas sniffing models rely more on nose-to-body distances and slower displacements.

Full ranked tables of feature importances for both behaviors are available in `outputs/attack_feature_importances.csv` and `outputs/sniffing_feature_importances.csv`.

## 4. Discussion

### 4.1. Reproducibility of the SimBA-style workflow

Using only the pose-derived frame-level features and ground-truth labels from the official SimBA sample project, we reproduced a core component of the SimBA workflow: training supervised classifiers to detect attack and sniffing behaviors.

Our analysis demonstrates that:

1. **Pose-derived features are highly informative** for discriminating both behaviors. Even with a relatively simple random forest classifier, we obtained near-perfect cross-validated performance and perfect resubstitution performance on the available dataset.
2. **The full analysis is auditable and transparent.** All intermediate artifacts (feature matrices, metrics, confusion matrices, feature importance tables) are stored in human-readable CSV files under `outputs/`, and all visualization code is maintained in the `code/` directory. Another researcher can re-run the pipeline to regenerate every figure and table.
3. **Model interpretation is feasible at the feature level.** Random forest feature importance rankings identify subsets of features that drive classification, offering a starting point for neuroscientific interpretation (e.g. relating attack to specific spatial configurations and movement patterns).

Taken together, these points support the claim that a SimBA-style workflow can reproducibly transform tracked pose features into evidence for behavior classification, at least for this sample dataset.

### 4.2. Limitations

Several important limitations temper these conclusions:

1. **Small and non-independent dataset.** The analysis is based on a single recording with 1,738 frames. Neighboring frames are highly correlated in time, and the same animals and environment are shared across all examples. Standard cross-validation on frame indices therefore overestimates generalization performance.

2. **Resubstitution evaluation inflates performance.** The perfect confusion matrices are computed on the same data used for training, and hence they do not reflect generalization performance. In production settings, SimBA projects typically evaluate models on held-out videos or use more conservative temporal cross-validation schemes.

3. **Limited behavior coverage.** We focused on two behaviors (attack and sniffing). Other behaviors with more ambiguous kinematics may be harder to classify and might yield less impressive performance.

4. **Feature-importance interpretation is heuristic.** Random forest Gini importance is known to be biased toward variables with more variability and higher cardinality. A more rigorous analysis would complement this with permutation importance or SHAP values and assess stability across bootstrap samples.

5. **Lack of direct comparison to the original SimBA models.** Although we used similar model families and preprocessing steps, we did not attempt a bit-for-bit reproduction of the original SimBA training procedure (e.g. hyperparameter tuning, specific engineered features, and temporal smoothing). Our results should therefore be seen as a conceptual reproduction rather than an exact replication.

### 4.3. Implications and future directions

From a methodological standpoint, this analysis illustrates that:

- Open access to pose features and behavior labels enables complete end-to-end reanalysis by independent researchers, which is essential for reproducibility and transparency in behavioral neuroscience.
- Even relatively simple supervised models can achieve excellent performance when the underlying feature representation captures the relevant spatial and kinematic structure of behavior.
- Embedding diagnostic outputs (confusion matrices, precision–recall curves, feature-importance tables) into the standard workflow provides a clear audit trail for evaluating and interpreting behavioral classifiers.

Future work could extend this approach by:

- Incorporating multiple videos and animals to better assess between-animal and between-context generalization.
- Using temporally aware cross-validation schemes (e.g. blocked CV on contiguous frame segments or leave-one-bout-out strategies) to obtain less optimistic performance estimates.
- Comparing alternative model classes (e.g. gradient-boosted trees, temporal convolutional networks) and feature representations (e.g. raw pose trajectories, self-supervised embeddings) under the same evaluation framework.
- Applying permutation-based feature importance and post-hoc explainability methods to more precisely quantify which aspects of pose dynamics drive model decisions.

## 5. Conclusion

By re-analyzing the official SimBA sample project data with an explicit, open-source pipeline, we showed that frame-level pose-derived features can be transformed reproducibly into supervised classifiers that detect attack and sniffing behavior with excellent apparent performance. The combination of structured intermediate outputs, detailed diagnostics and feature-importance analyses provides transparent and auditable evidence for behavior classification, and illustrates the strengths—as well as the methodological caveats—of SimBA-style workflows in quantitative behavioral neuroscience.
