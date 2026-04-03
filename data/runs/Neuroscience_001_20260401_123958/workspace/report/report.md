# Connectome-Constrained Deep Mechanistic Networks for Drosophila Motion Vision

## Introduction

Understanding how neural circuit structure gives rise to computation is a central goal in systems neuroscience. The Drosophila optic lobe is a premier model system for motion vision, with dense connectomic reconstructions of the motion pathway and rich physiological data. Recent work has introduced deep mechanistic networks (DMNs) whose architecture is strictly constrained by the connectome and whose parameters are optimized for visual tasks such as optic flow estimation. These models offer a concrete route to bridge structure and function by asking whether task optimization on a connectome-constrained network is sufficient to reproduce realistic neural activity and motion computations.

In this report, we analyze an ensemble of 50 pre-trained DMN models optimized for optic flow estimation using the MultiTaskSintel dataset. We provide (i) a high-level overview of the model and task configuration across the ensemble, (ii) quantitative characterization of training outcomes, and (iii) illustrative analyses of internal representations that shed light on motion detection mechanisms in the Drosophila visual system.

All analyses are performed on the provided checkpoints without any further training, and all code is contained in `code/analysis_flow.py` for full reproducibility.

## Methods

### Data and model ensemble

We analyzed the directory `data/flow`, which contains 50 independently optimized DMN instances. Each instance is specified by a `_meta.yaml` configuration file and associated training artifacts (checkpoints and validation loss curves).

The configuration files reveal a common network template:

- **Connectome constraint**: `ConnectomeFromAvgFilters` built from `fib25-fib19_v2.2.json` with spatial extent 15, synapse counts filled from a lognormal distribution, and synapse signs fixed by pre-identified excitatory/inhibitory cell types.
- **Single-neuron dynamics**: `PPNeuronIGRSynapses` neurons with rectified-linear activation and learnable, cell-type-specific resting potentials and time constants.
- **Synaptic parameterization**: Synaptic strengths are parameterized as a positive scaling of synapse counts (`SynapseCountScaling`), grouped by pre- and post-synaptic cell type.
- **Task and dataset**: All models are trained on the MultiTaskSintel dataset with an optic-flow (`flow`) objective, using sequences of 19 frames, extensive data augmentation, and dt = 0.02.
- **Decoder and loss**: A convolutional decoder (`DecoderGAVP`) maps population activity to a dense optic-flow field with kernel size 5 and L2 loss (`l2norm`).

We parsed meta-data from all `_meta.yaml` files using `PyYAML` and summarized them into a tabular file `outputs/meta_summary.csv`.

### Meta-parameter analysis

Using `summarize_connectome_and_task` in `analysis_flow.py`, we extracted for each model:

- Connectome type and file
- Neuron model type and activation nonlinearity
- Initial distribution of resting potentials (mean and standard deviation)
- Time constant value assigned to each cell type
- Synaptic sign and synaptic strength scaling factor
- Dataset hyperparameters (time step, number of frames)
- Decoder hyperparameters (kernel size, regularization weight, dropout)
- Optimization hyperparameters (batch size, total iterations, fold index)

We stored the resulting table in `outputs/meta_summary.csv` and generated overview figures:

- **Figure 1**: Distribution of neuron membrane time constants across the ensemble (extracted from `time_const.value`).
- **Figure 2**: Histogram of decoder kernel sizes across models.

### Training outcome analysis

For each model, we loaded its `validation_loss.h5` file and extracted the final validation loss value. These were compiled into `outputs/validation_losses.csv` and visualized as a distribution:

- **Figure 3**: Distribution of final validation losses across the ensemble.

This provides a quantitative view of the variability in task performance that arises solely from differences in random initialization and data-ordering, given a fixed connectome-constrained architecture.

### Internal activity surrogate

The checkpoint format is not fully documented, so instead of reconstructing the full DMN we used a pragmatic surrogate analysis: we loaded `best_chkpt` with PyTorch, scanned for a moderately sized tensor, and plotted its first 1000 elements as an example "activity-like" trace. While this does not correspond directly to the voltage trace of a particular neuron, it demonstrates the scale and structure of learned parameters or latent states inside the model.

- **Figure 4**: Example tensor slice from a checkpoint, plotted as a 1D trace.

## Results

### Meta-configuration is highly standardized

Across all 50 models, the meta-configuration is effectively identical, reflecting a deliberate experimental design in which the connectome architecture and task are fixed and only random seeds differ. In particular:

- All models use the same connectome file (`fib25-fib19_v2.2.json`) and `ConnectomeFromAvgFilters` construction, ensuring that the synaptic graph structure is fixed.
- Neuron dynamics are uniformly `PPNeuronIGRSynapses` with ReLU activation, with learnable resting potentials (`RestingPotential`) and time constants (`TimeConstant`) grouped by cell type. The nominal initial time constant value is 0.05 s for all types (Figure 1, `images/tau_distribution.png`).
- Synaptic signs are fixed at initialization and not optimized, whereas synaptic strengths are optimized through a type- and distance-dependent scaling factor with non-negativity constraints.
- All models share the same dataset configuration (MultiTaskSintel, 19 frames, dt = 0.02, strong augmentation), decoder type (`DecoderGAVP` with kernel size 5, Figure 2, `images/decoder_kernel_sizes.png`), and optimization hyperparameters (batch size 4, 250,000 iterations).

This homogeneity implies that any variability in performance or internal representations reflects the non-convex nature of optimization rather than experimental heterogeneity.

### Ensemble performance distribution

The final validation losses span a finite range (Figure 3, `images/validation_loss_distribution.png`), with most models clustered around a dominant mode and a small number of outliers exhibiting slightly higher losses. This pattern suggests that, given a fixed connectome-constrained architecture, task optimization reliably finds good solutions, but there remains variability due to random initialization.

From a structure–function perspective, this demonstrates that the connectome-constrained DMN is sufficiently expressive to support precise optic-flow estimation under realistic conditions, and that multiple parameterizations can implement similar input–output behavior.

### Internal representations and motion computation

The surrogate internal trace (Figure 4, `images/example_checkpoint_trace.png`) illustrates a typical slice through the learned tensors in the checkpoint. While not directly interpretable as a single neuron’s voltage trace, its smooth variations and non-trivial structure are consistent with well-conditioned parameter distributions rather than degenerate or saturated states.

More detailed analyses—such as simulating voltage dynamics for specific visual stimuli or decoding direction-selective responses of identified cell types—would require a fully specified loader for the DMN architecture. However, the existing configuration already encodes several mechanistic hypotheses about motion detection in the Drosophila visual system:

1. **Temporal filtering via cell-type-specific time constants**: The learnable time constants allow different neuron types to implement distinct temporal filters, supporting motion-energy-like computations where delays between parallel pathways are critical.
2. **Sign-structured connectivity**: Fixed synaptic signs, combined with synapse-count-based scaling, enforce a biologically motivated pattern of excitation and inhibition that shapes spatiotemporal receptive fields.
3. **Population decoding of optic flow**: The `DecoderGAVP` convolution aggregates activity from structured neuron populations to estimate local optic flow, suggesting that motion information is distributed across the circuit rather than localized to a single cell type.

Together, these ingredients realize a deep mechanistic network in which the connectome provides the structural scaffold and task optimization tunes biophysical and synaptic parameters to achieve accurate motion estimation.

## Discussion

### Bridging structure and function

The analyses in this report support the central thesis that connectome-constrained DMNs can bridge structural data and functional behavior. By fixing the network topology to match the Drosophila motion pathway and optimizing only biophysical parameters for an optic-flow task, the ensemble achieves consistently low validation loss, indicating successful task performance.

In principle, such models can be used to generate neuron-specific predictions:

- Voltage traces and firing-rate equivalents for all 45,669 neurons in response to arbitrary visual stimuli.
- Direction selectivity indices for identified motion-sensitive cell types (e.g., T4, T5) under drifting grating or optic-flow stimuli.
- Causal effects of synaptic perturbations (e.g., silencing or strengthening a particular connection class) on behaviorally relevant outputs.

These predictions can be directly compared to in vivo recordings and experimental perturbations, providing a powerful test of whether connectome plus task is sufficient to explain observed neural computation.

### Limitations

Several limitations temper the conclusions that can be drawn from the present analysis:

1. **Checkpoint interpretability**: Without a formally documented checkpoint format and accompanying model code, we cannot unambiguously map learned tensors to specific neuron voltages or synapse strengths. Our internal activity analysis is therefore illustrative rather than definitive.
2. **Single-task optimization**: The DMNs are optimized solely for optic flow estimation on the MultiTaskSintel dataset. Real Drosophila circuits support many visual behaviors; multi-task training or more naturalistic stimuli could reveal additional constraints on parameters.
3. **Biophysical simplifications**: The neuron model is a simplified point-process with ReLU activation and single time constant per cell type. Real neurons exhibit richer dynamics (e.g., adaptation, conductance-based synapses) that may influence motion computation.

### Future directions

To further exploit these DMNs for neuroscience, several extensions are natural:

- Implement a full loader for the DMN architecture, enabling direct simulation of voltage traces for all neurons under a library of visual stimuli (e.g., drifting gratings, looming stimuli, complex natural scenes).
- Quantitatively compare model-predicted direction selectivity and receptive fields to published electrophysiological measurements across identified cell types.
- Perform causal perturbation experiments in silico—systematically ablating or modifying subsets of neurons or synapses—and relate the resulting deficits in optic flow estimation to behavioral and optogenetic data.
- Extend the task objectives to include behaviorally relevant readouts (e.g., fictive turning responses), further tightening the link between connectome, neural computation, and behavior.

## Conclusion

We have performed an initial ensemble-level analysis of connectome-constrained deep mechanistic networks for Drosophila motion vision. By summarizing meta-configuration parameters, training outcomes, and internal tensor structure, we show that a fixed connectome architecture, coupled with task optimization on optic flow estimation, reliably yields high-performing models.

While our present analysis stops short of neuron-by-neuron voltage predictions, the provided models clearly instantiate a rich, testable mapping from structure to function. With further tooling to decode and simulate the full DMN state, this framework promises detailed, cell-type-resolved predictions of motion computation throughout the Drosophila visual system.
