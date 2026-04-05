# Analysis Notes

- Fixed encoder resolution: 336x336
- ROI scoring combines gradient energy and local contrast as a training-free proxy for fine detail / text density.
- Selected the top non-overlapping crop per image for quantitative comparison.

## Summary Table

image,width,height,selected_scale,crop_area_fraction,detail_gain_gradient,detail_gain_text_p95
demo1,1024,768,0.5,0.25,1.0257755325981543,1.577211704731549
demo2,2250,1500,0.5,0.25,1.0730911430849988,0.9159704426443829
method_case,2500,1681,0.5,0.2498512790005949,1.2706017670898395,1.021512228773575
