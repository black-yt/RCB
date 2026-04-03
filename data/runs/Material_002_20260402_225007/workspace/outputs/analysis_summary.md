# Analysis Output Summary

## Data provenance
- Source file: `data/MACE-MP-0_Reproduction_Dataset.txt`
- Referenced model artifact: `MACE-MP-0b3-medium.model`

## Water benchmark
- Molecules: 32
- Box size: 12.00 Å
- Density implied by box and composition: 0.554 g/cm^3
- HOH angle: 104.00°
- Total MD time: 1.00 ps

## Adsorption benchmark
- Metals: Ni, Cu, Rh, Pd, Ir, Pt
- Scaling-fit slope: 1.850
- Scaling-fit intercept: 0.180 eV
- Scaling-fit Pearson r: 1.000

## Reaction benchmark
- Mean DFT barrier: 1.743 eV
- Barrier std: 0.021 eV
- Mean RMS structural shift: 0.204 Å

## Foundation-model interpretation
- Element coverage in this reproduction proxy: 9 elements
- Benchmark diversity: molecule + liquid + surface + reaction
- Assessment: The provided benchmarks span structural, catalytic, and reactive regimes, making them suitable as low-data adaptation probes once a pretrained universal potential is available.

## Generated figures
- `report/images/water_setup_overview.png`
- `report/images/adsorption_scaling_analysis.png`
- `report/images/reaction_barrier_analysis.png`
- `report/images/foundation_scope_summary.png`
