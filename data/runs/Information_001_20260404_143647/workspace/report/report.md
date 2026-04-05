# Training-Free Task-Guided Cropping for Fine-Grained Perception: A Small-Scale Reproduction Study

## 1. Summary and objective
Fixed-resolution vision encoders compress the entire image into a small input grid, which can suppress small objects, local text, and subtle boundaries. The provided related work describes a task-guided visual search/cropping paradigm that preserves global context while zooming into informative local regions. Because the workspace only provides three demo images and no MLLM checkpoints or detector weights, this study targets the most reproducible part of the claim: **whether a training-free crop-selection strategy can recover local detail that is lost when the full image is resized to a fixed encoder resolution**.

The study evaluates a simple, reproducible pipeline:
1. Resize each full image to a CLIP-like encoder resolution of 336×336 as the baseline.
2. Estimate task-relevant regions with a training-free fine-detail proxy based on gradient energy and local contrast.
3. Select the top non-overlapping crop, resize that crop to the same 336×336 encoder resolution, and compare how much local detail survives.
4. Report both qualitative and quantitative evidence.

## 2. Relation to prior work
Among the supplied PDFs, `related_work/paper_000.pdf` is the closest match to the target idea. It describes a guided visual search mechanism that identifies regions of interest, crops them, and reintegrates those local observations into the final reasoning process. The full paper includes learned components and is not a pure training-free system, but its central insight is directly testable here: **global context and local zoom should be combined because fixed-resolution processing alone loses fine-grained information**.

This reproduction therefore focuses on a narrower scientific question:

> Can a lightweight, training-free region proposal strategy select crops that contain more recoverable local detail than the full-image baseline under the same encoder input size?

## 3. Data audit
The input directory `data/demo_imgs/` contains three images:
- `demo1.png`: 1024 × 768
- `demo2.png`: 2250 × 1500
- `method_case.png`: 2500 × 1681

The small sample size makes this a qualitative-plus-analytic study rather than a statistical benchmark. No labels or question-answer pairs are provided, so the evaluation must use visibility/detail proxies rather than answer accuracy.

### Data overview
![Data overview](images/data_overview.png)

## 4. Experimental design
### 4.1 Baseline
The baseline simulates a fixed-resolution vision encoder by resizing the full image to 336 × 336. This matches the core failure mode described in the task: all scene content competes for a fixed number of visual tokens.

### 4.2 Training-free crop proposal
A crop proposal heatmap is computed from the original image using only image statistics:
- grayscale conversion
- gradient magnitude as an edge/fine-structure signal
- local contrast (5×5 neighborhood standard deviation)
- elementwise combination of gradient and contrast to emphasize likely fine-detail regions, including text-like structure

A sliding-window search is then run over scales {0.2, 0.3, 0.4, 0.5} of the original image dimensions with a step size of 10% of image width/height. Each candidate window is scored by:

\[
\text{score(window)} = \text{mean}(H) + 0.25\,P_{95}(H) + 0.1\log(\text{area})
\]

where \(H\) is the heatmap inside the window. The logarithmic area term mildly favors windows that retain some context instead of collapsing to tiny patches. The top non-overlapping crop is used for the main quantitative comparison.

### 4.3 Metrics
Because no ground-truth labels are available, the analysis uses detail-preservation proxies computed after resizing to encoder resolution:
- **Mean gradient**: average edge strength in the resized view
- **Text saliency p95**: 95th percentile of the heatmap in the resized view; intended to capture strong local detail/text-like peaks
- **Relative detail gain**: crop metric divided by the global metric
- **Crop area fraction**: selected crop area divided by original image area

These are not end-task accuracy metrics. They are image-based surrogates for the recoverability of fine detail after resizing.

## 5. Reproducible implementation
All analysis code is in `code/run_analysis.py`.

Main outputs:
- `outputs/metrics.csv`
- `outputs/summary.csv`
- `outputs/scale_curve.csv`
- `outputs/roi_windows.json`
- `outputs/analysis_notes.md`

To reproduce:

```bash
python code/run_analysis.py
```

The script is deterministic and uses a fixed random seed (`SEED = 0`), although the current pipeline is effectively deterministic even without randomness.

## 6. Results

### 6.1 Task-guided crop proposals
The proposed windows generally fall on dense local-detail regions. The overlays below show the ranked candidate crops for each demo image.

#### Demo 1
![Demo1 overlay](images/demo1_overlay.png)
![Demo1 crops](images/demo1_crops.png)

#### Demo 2
![Demo2 overlay](images/demo2_overlay.png)
![Demo2 crops](images/demo2_crops.png)

#### Method case
![Method-case overlay](images/method_case_overlay.png)
![Method-case crops](images/method_case_crops.png)

The corresponding heatmaps indicate that the scoring function attends to high-frequency, high-contrast regions rather than to uniformly textured background.

![Demo1 heatmap](images/demo1_heatmap.png)
![Demo2 heatmap](images/demo2_heatmap.png)
![Method-case heatmap](images/method_case_heatmap.png)

### 6.2 Quantitative comparison: global resize vs crop resize
The central comparison keeps the encoder resolution fixed at 336 × 336 for both views.

![Main comparison](images/main_comparison.png)

From `outputs/summary.csv`, the selected crop occupies approximately one quarter of the original image area in every case (area fraction ≈ 0.25), but still improves local detail preservation:

| Image | Crop area fraction | Gradient gain | Text-saliency gain |
|---|---:|---:|---:|
| demo1 | 0.2500 | 1.026 | 1.577 |
| demo2 | 0.2500 | 1.073 | 0.916 |
| method_case | 0.2499 | 1.271 | 1.022 |
| **Mean** | **0.2500** | **1.123** | **1.172** |

Interpretation:
- **Mean gradient** improves in all three images, with an average relative gain of **1.123×**. This indicates that local edges survive resizing better when the model is allowed to zoom.
- **Text-saliency p95** improves on two of three images, with a mean gain of **1.172×**. The one failure case (`demo2`) suggests that the heuristic can choose a region rich in edges but not necessarily the most text-like or semantically relevant region.

### 6.3 Validation across encoder resolutions
To test whether the trend is specific to 336 × 336, the analysis compares full-image versus selected-crop detail across multiple encoder resolutions.

![Scale validation](images/scale_validation.png)

The crop view maintains an advantage across the evaluated resolutions in the mean-gradient proxy. This supports the main mechanism: **if local content is resampled after zooming rather than after global compression, more detail is preserved**.

### 6.4 Area-efficiency trade-off
The next plot compares crop area fraction with detail gain.

![Area vs gain](images/area_vs_gain.png)

Even with a moderate crop size (about 25% of the full image), substantial local-detail gains are observed, especially on `method_case`. This suggests that the improvement does not require extremely small crops; useful gains can appear with context-preserving windows.

### 6.5 Consolidated improvement figure
![Detail gain](images/detail_gain.png)

The gains are not uniform across metrics or images, which is expected for a generic, training-free heuristic. However, the dominant pattern is positive, especially for gradient-based detail preservation.

## 7. Discussion
### 7.1 What the study supports
This small-scale reproduction supports the core intuition behind task-guided cropping:
- A fixed-resolution global view loses local structure.
- A crop resized to the same encoder input recovers finer boundaries and local contrast.
- A simple training-free heuristic can identify promising ROI candidates without retraining a vision encoder.

These findings are consistent with the premise that global-only encoding is suboptimal for small or localized content.

### 7.2 What this study does **not** prove
This study does **not** establish improved MLLM answer accuracy, because:
- no question-answer annotations are provided;
- no multimodal model is available in the workspace;
- the evaluation uses image-detail proxies rather than end-task metrics.

Therefore, the results should be interpreted as a mechanistic validation of the information-loss argument, not as a full reproduction of the original paper's benchmark results.

### 7.3 Failure modes and limitations
1. **Very small sample size**: only three images are available, so no statistical generalization is possible.
2. **Proxy metrics**: gradient and contrast capture visibility, not semantic usefulness.
3. **Heuristic ROI selection**: the method can favor textured but semantically irrelevant regions.
4. **No language conditioning**: the original idea is task-guided; this reproduction is only weakly task-guided because it has no explicit question text.
5. **No uncertainty estimates**: with only three images, confidence intervals would be uninformative.

## 8. Conclusion
Within the constraints of the workspace, the reproduction demonstrates the most testable part of the proposed framework: **task-guided local cropping can preserve more fine-grained visual detail than a fixed-resolution global pass, even when implemented with a simple training-free ROI heuristic**. The average improvement is **1.123×** in mean gradient and **1.172×** in text-saliency peak, while using crops that cover only about **25%** of the full image area.

The next step toward a fuller reproduction would be to add explicit question-conditioned crop selection and measure downstream VQA or OCR accuracy using a multimodal model. Under the current constraints, the present evidence is best viewed as a controlled demonstration of the information-loss mechanism and the benefit of zoom-in views.

## 9. Files produced
- Code: `code/run_analysis.py`
- Outputs: `outputs/metrics.csv`, `outputs/summary.csv`, `outputs/scale_curve.csv`, `outputs/roi_windows.json`, `outputs/analysis_notes.md`
- Report figures: all files under `report/images/`
