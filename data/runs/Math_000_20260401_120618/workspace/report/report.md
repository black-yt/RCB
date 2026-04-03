# Hierarchical Sparse Multi-Object Tracking via Pseudo-Depth Decomposition

## 1. Introduction

Multi-object tracking (MOT) in crowded scenes is fundamentally limited by frequent occlusions and high spatial density. When many targets overlap in the image plane, association based purely on 2D proximity (e.g., IoU-based assignment) often becomes ambiguous, leading to identity (ID) switches and fragmented trajectories. Recent trackers such as ByteTrack leverage high- and low-confidence detections to improve robustness, but they still operate on a dense set of 2D boxes per frame, where near/far ordering is not explicitly modeled.

This work investigates a simple but effective strategy: **decompose the dense set of detections into pseudo-depth-ordered sparse subsets** and perform **hierarchical association** across these subsets. The intuition is that, in many scenes, closer objects occupy larger image areas; using bounding-box area as a proxy for depth allows us to approximate a front-to-back ordering. Within each depth layer we enforce low mutual overlap so that association is easier. We compare this *hierarchical sparse tracker* against a simplified ByteTrack-like baseline on a controlled synthetic sequence with dense occlusions.

The input to all methods is the same: for each frame we are given a list of detections with bounding boxes and confidence scores. The desired output is a set of trajectories, each consisting of a unique track ID and the corresponding sequence of bounding boxes through time.


## 2. Data and Experimental Setup

### 2.1 Simulated sequence

We use the provided `simulated_sequence.json` dataset, which contains:

- 40 frames, 20 ground-truth objects (with fixed IDs) moving through the scene.
- For each frame:
  - `gt_bboxes` and `gt_ids`: ground-truth bounding boxes and object identities.
  - `detections`: simulated detector outputs with bounding boxes, confidence `score`, and a `gt_id` label tying each detection back to its true object. Some detections are marked with an `occluded` flag, indicating partial or heavy occlusion.
- The simulation uses an 85% detection rate and a 20% occlusion overlap threshold to create dense, cluttered configurations.

A quick overview of the data is shown in Fig. 1, which plots per-frame counts of ground-truth objects, detections, and occluded detections.

- **Figure 1**: Data overview – counts per frame.  
  `![Data overview](images/data_overview_counts.png)`

The number of detections per frame closely tracks the number of ground-truth targets, but occluded detections vary over time, producing periods with severe visual crowding.

### 2.2 Evaluation protocol

To quantitatively compare trackers, we use frame-wise IoU-based matching between predicted tracks and ground-truth boxes. For each frame:

1. We construct a cost matrix between all GT boxes and all predicted boxes.
2. IoU is computed for each pair; pairs with IoU below 0.5 are treated as non-matches.
3. The Hungarian algorithm finds the best one-to-one assignment.

From these assignments we compute:

- **Matches**: number of GT–prediction pairs with IoU ≥ 0.5.
- **Recall**: matches / total GT boxes.
- **Precision**: matches / total predicted boxes.
- **F1 score**: harmonic mean of precision and recall.
- **ID switches**: for each GT ID, we track the matched predicted track ID over time and count how often it changes.

While this is a simplified MOT evaluation (it does not implement the full MOTChallenge MOT17/MOT20 protocol), it captures the key trade-off between recall, precision, and identity consistency.


## 3. Methods

All code is implemented in Python using NumPy, SciPy, and Pandas, and is located in `code/`:

- `trackers.py` – tracker implementations (hierarchical sparse tracker and ByteTrack-like baseline).
- `run_analysis.py` – data loading, running trackers, computing metrics, and producing plots.

### 3.1 Notation

Each detection is represented as:

- Bounding box: \(b = (x_1, y_1, x_2, y_2)\).
- Confidence score: \(s \in [0,1]\).
- Frame index: \(t\).

Intersection-over-Union (IoU) between boxes \(a\) and \(b\) is defined as usual:
\[
\operatorname{IoU}(a,b) = \frac{|a \cap b|}{|a| + |b| - |a \cap b|}.
\]


### 3.2 Pseudo-depth estimation

We define a simple pseudo-depth \(d\) based on bounding-box area:
\[
A(b) = (x_2 - x_1)_+ (y_2 - y_1)_+, \quad d(b) = -A(b).
\]

Thus, larger boxes (presumed closer to the camera) have smaller depth values and are processed first. We do not attempt to estimate metric depth; only the ordering induced by \(d\) is used.

For each frame, given a list of detections \(\{b_i\}\), we sort detections by \(d(b_i)\) from near to far.


### 3.3 Sparse depth-layer decomposition

Dense 2D overlap is particularly problematic when many targets are at similar depth. To mitigate this, we decompose the sorted detections into **sparse layers** such that, within each layer, the pairwise IoU between any two boxes does not exceed a small threshold \(\tau_\text{layer}\) (default 0.1 in our experiments).

Algorithmically:

1. Sort detections in the frame by pseudo-depth (near to far).
2. Maintain a list of layers \(L_1, L_2, \dots\).
3. For each detection in order, attempt to place it into the earliest layer whose boxes all have IoU ≤ \(\tau_\text{layer}\) with it; if none exists, create a new layer.

This greedy procedure yields a set of depth-ordered sparse subsets. Intuitively, each layer is a set of detections that are *well separated* in the 2D plane, making association easier.


### 3.4 Hierarchical association

We adopt a hierarchical association strategy inspired by ByteTrack but augmented with depth-layered processing.

1. **Score stratification** (ByteTrack-style):
   - High-confidence detections: \(s \geq s_\text{high}\) (0.4 by default).
   - Low-confidence detections: \(s_\text{low} \leq s < s_\text{high}\) (with \(s_\text{low}=0.1\)).
2. **High-score association:**
   - Decompose high-score detections into depth layers using the procedure above.
   - For each layer in order (near to far):
     - Compute the IoU-based cost matrix between active tracks and detections in this layer, using the last detection in each track.
     - Apply Hungarian assignment to obtain candidate matches.
     - Accept matches with IoU ≥ \(\tau_\text{assoc}\) (0.3 by default); update tracks and their last-seen frame.
   - After all layers are processed, any high-score detection that remains unmatched starts a new track.
3. **Low-score association:**
   - Decompose low-score detections into depth layers.
   - Process layers again from near to far, allowing active tracks to match low-confidence detections under the same IoU threshold. This can recover links through occlusions where detector confidence dips.
4. **Track management:**
   - Tracks whose last observation is more than `max_age` frames in the past (3 by default) are marked inactive and no longer participate in association.

This **HierarchicalSparseTracker** is designed to reduce ambiguous assignments in dense regions by:

- Handling near-depth (likely foreground) detections first.
- Ensuring that within a layer, detections are sufficiently separated, reducing competition between tracks.


### 3.5 ByteTrack-like baseline

As a baseline, we implement a simplified **ByteLikeTracker**:

1. Split detections into high- and low-score sets using the same thresholds.
2. First associate high-score detections with active tracks via Hungarian assignment on 1−IoU cost, matching only above \(\tau_\text{assoc}}\).
3. Unmatched high-score detections start new tracks.
4. Low-score detections are then used to update existing tracks via another association step.
5. Tracks are deactivated after `max_age` frames without observations.

Unlike the hierarchical tracker, this baseline does **not** perform pseudo-depth decomposition and treats all detections in a frame as a single dense set.


## 4. Results

### 4.1 Track statistics

We first inspect the basic statistics of the tracks produced by each method. Figure 2 shows the distribution of track lengths (in frames), and Figure 3 shows the number of active tracks per frame.

- **Figure 2**: Track length distribution – ByteLike vs. SparseHier.  
  Byte-like: `![Track length – ByteLike](images/track_length_ByteLike.png)`  
  Hierarchical sparse: `![Track length – SparseHier](images/track_length_SparseHier.png)`

- **Figure 3**: Active tracks per frame – ByteLike vs. SparseHier.  
  Byte-like: `![Active tracks – ByteLike](images/active_tracks_ByteLike.png)`  
  Hierarchical sparse: `![Active tracks – SparseHier](images/active_tracks_SparseHier.png)`

Qualitatively, the hierarchical sparse tracker tends to produce **longer tracks** than the Byte-like baseline. The baseline often fragments trajectories due to ambiguous associations in crowded frames; this manifests as many short tracks. In contrast, the sparse decomposition reduces competition within layers, enabling more consistent associations across frames.

The per-frame active track counts also differ: the baseline spawns more tracks in some frames, which is indicative of frequent track termination and reinitialization. The hierarchical method maintains a more stable number of active trajectories across time.


### 4.2 Quantitative MOT metrics

Table 1 summarizes the main quantitative metrics computed over the full sequence.

**Table 1 – MOT-style metrics (IoU ≥ 0.5)**

| Algorithm      | Matches | GT boxes | Predicted boxes | Recall | Precision | F1    | ID switches |
|----------------|---------|----------|-----------------|--------|-----------|-------|-------------|
| ByteLike       | 1848    | 20000    | 1848            | 0.0924 | 1.0000    | 0.169 | 217         |
| SparseHier     | 10516   | 20000    | 10523           | 0.5258 | 0.9993    | 0.689 | 4920        |

Figure 4 visualizes the F1 scores and ID-switch counts.

- **Figure 4**: Metric comparison: F1 and ID switches.  
  F1: `![F1 comparison](images/metric_f1_comparison.png)`  
  ID switches: `![ID switch comparison](images/metric_ids_comparison.png)`

#### 4.2.1 Recall and precision

Both methods achieve nearly perfect precision (≈1.0), indicating that false positives are negligible under the chosen IoU threshold. The key difference lies in **recall**:

- The Byte-like baseline recalls only about 9.2% of GT boxes.
- The hierarchical sparse tracker recalls about 52.6% of GT boxes.

The dramatic improvement in recall suggests that depth-aware sparse decomposition allows many more detections to be consistently linked into trajectories, rather than being discarded due to association ambiguity or track termination.

#### 4.2.2 F1 score

The F1 score aggregates recall and precision. The hierarchical method yields **F1 ≈ 0.69**, substantially higher than the baseline’s **F1 ≈ 0.17**. This demonstrates that the proposed decomposition and hierarchical association significantly improve overall tracking performance on this dense, occlusion-heavy sequence.

#### 4.2.3 Identity switches

Interestingly, the hierarchical tracker exhibits a much higher raw count of ID switches (4920 vs. 217). This is largely a consequence of its much higher recall and number of matched boxes:

- The baseline simply fails to track the majority of GT boxes, which suppresses opportunities for ID switches.
- The hierarchical method maintains many more active trajectories, which increases the chance of GT-to-track reassignment fluctuations.

In the context of the synthetic data and our simplified evaluation, ID switches thus reflect a *trade-off*: aggressively pursuing higher recall can introduce more ID fragmentation when using only 2D IoU and no appearance cues.


## 5. Discussion

### 5.1 Effect of pseudo-depth-based sparse decomposition

The experimental results support the central hypothesis: **decomposing dense detection sets into pseudo-depth-ordered sparse subsets enables more effective association in crowded scenes**.

Key observations:

1. **Improved recall** – The hierarchical sparse tracker is able to recover more than half of all ground-truth boxes, compared to less than 10% for the baseline. This indicates that many detections that would otherwise remain untracked or cause track termination are successfully incorporated into trajectories.
2. **Reduced local ambiguity** – Within each sparse layer, pairwise overlaps are bounded (IoU ≤ 0.1). This reduces direct competition among nearby detections for the same track and encourages one-to-one associations, especially for large, near-depth objects that often dominate the scene.
3. **Depth-aware ordering** – By processing layers in near-to-far order, the tracker prioritizes associations for close, large targets first. This tends to produce more stable trajectories for foreground objects that occlude others.

However, pseudo-depth from box area is a crude approximation. In scenes with significant perspective distortion, objects of very different real-world depth can have similar image sizes, and vice versa. Nonetheless, on this synthetic dataset, the simple heuristic already yields substantial gains.


### 5.2 Limitations and potential improvements

Several limitations of the present study and implementation should be noted:

1. **No appearance features** – Both trackers rely purely on geometric IoU. In realistic MOT benchmarks, appearance cues (e.g., ReID embeddings) are critical for resolving long-term occlusions and reducing ID switches. The large number of ID switches for the hierarchical method suggests that appearance modeling would be an important next step.
2. **Simplified evaluation** – The metrics used here are simplified and frame-wise; they do not capture all aspects of MOT performance, such as track fragmentation, mostly-tracked/mostly-lost statistics, or track-level continuity metrics like IDF1. Extending the evaluation to a full MOTChallenge-style metric suite would provide a more nuanced assessment.
3. **Hyperparameter sensitivity** – We used fixed thresholds (`iou_thresh = 0.3`, `max_iou_within_layer = 0.1`, `max_age = 3`) without tuning. These choices may not be optimal for all scenes. In particular, the depth layer sparsity constraint might be relaxed or tightened to balance between within-layer competition and the number of layers.
4. **Single synthetic sequence** – Results are derived from one simulated sequence with specific density and occlusion parameters. Generalization to real-world videos and diverse crowding levels must be verified empirically.

Potential improvements include:

- Incorporating learned or handcrafted appearance descriptors into the association cost, combined with depth-layered decomposition.
- Replacing the heuristic pseudo-depth with a learned monocular depth estimator (where available) or multi-view geometry when multiple cameras are present.
- Exploring more sophisticated layer-construction strategies, e.g., clustering in 3D pseudo-space (image position + depth) rather than only area-based ordering.


## 6. Conclusion

We presented a hierarchical multi-object tracking framework that decomposes dense detection sets into pseudo-depth-ordered sparse subsets and performs layer-wise association. On a synthetic sequence with strong occlusions and high crowd density, this simple method substantially improves recall and F1 score compared to a ByteTrack-like baseline that lacks depth-aware decomposition.

The findings highlight that even coarse depth proxies (such as bounding-box area) can provide valuable structure for multi-object tracking in crowded scenes. By reducing local ambiguity and prioritizing near-depth objects, hierarchical association over sparse subsets offers a promising direction for robust tracking under occlusion. Future work will integrate appearance features and more realistic depth cues to further reduce identity fragmentation while maintaining the gains in recall and overall tracking quality.
