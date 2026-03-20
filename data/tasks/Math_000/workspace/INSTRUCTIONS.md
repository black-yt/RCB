# Research Task

## Task Description
input:
Consecutive image frames and object detections per frame (including bounding box coordinates and confidence scores). outpt:
Complete trajectories for each target, i.e., identity labels (IDs) and corresponding bounding box sequences across video frames. scientific targets:
To effectively handle occlusions and improve multi-object tracking performance in crowded scenes by decomposing dense target sets into sparse subsets via pseudo-depth estimation and performing hierarchical association. 

## Available Data Files
- **simulated_sequence.json** [structured_data] (`data/simulated_sequence.json`): This dataset contains a simulated multi-object video sequence generated with controlled parameters (40 frames, 20 objects, 85% detection rate, 20% occlusion overlap threshold) to evaluate tracking performance under dense occlusion scenarios. It includes ground truth trajectories and detection boxes with confidence scores and occlusion labels, enabling reproducible comparison of SparseTrack and ByteTrack as presented in the paper.

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
