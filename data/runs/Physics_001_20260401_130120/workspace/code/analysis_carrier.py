import numpy as np
import matplotlib.pyplot as plt
import pathlib

base = pathlib.Path('.')
data_path = base / 'data' / 'MATBG Superfluid Stiffness Core Dataset.txt'
raw = pathlib.Path(data_path).read_text(encoding='utf-8')

sec = raw.split('File 1: superfluid_stiffness_measurement.py')[1]

import re

# helper to extract numpy-like arrays from between brackets on a named line

def extract_array(label):
    pat = label + r'.*?\n(\[.*?\])'
    m = re.search(pat, sec, flags=re.S)
    if not m:
        raise SystemExit(f'missing {label}')
    block = m.group(1)
    inside = block.strip()[1:-1]
    inside = re.sub(r"\s+", " ", inside)
    nums = [float(x) for x in inside.split()]
    return np.array(nums)

n_eff = extract_array('Carrier Density Data')
D_conv = extract_array('Conventional Superfluid Stiffness')
D_geom = extract_array('Quantum Geometric Superfluid Stiffness')
D_exp_h = extract_array('Experimental Superfluid Stiffness Hole-doped')
D_exp_e = extract_array('Experimental Superfluid Stiffness Electron-doped')

out_dir = base / 'outputs'
img_dir = base / 'report' / 'images'
out_dir.mkdir(exist_ok=True, parents=True)
img_dir.mkdir(exist_ok=True, parents=True)

# overview plot
plt.figure(figsize=(6,4))
plt.plot(n_eff*1e-14, D_conv/1e9, label='Conventional')
plt.plot(n_eff*1e-14, D_geom/1e9, label='Geometric')
plt.scatter(n_eff*1e-14, D_exp_h/1e11, s=10, label='Exp holes')
plt.scatter(n_eff*1e-14, D_exp_e/1e11, s=10, label='Exp electrons')
plt.xlabel(r'$n_\mathrm{eff}$ ($10^{14}$ m$^{-2}$)')
plt.ylabel(r'$D_s$ (arb units)')
plt.legend()
plt.tight_layout()
plt.savefig(img_dir / 'carrier_dependence.png', dpi=200)
plt.close()

# simple enhancement factor
enh_exp_h = D_exp_h / D_conv
enh_exp_e = D_exp_e / D_conv

np.savez(out_dir / 'carrier_dependence_analysis.npz', n_eff=n_eff, D_conv=D_conv, D_geom=D_geom,
         D_exp_h=D_exp_h, D_exp_e=D_exp_e, enh_exp_h=enh_exp_h, enh_exp_e=enh_exp_e)

print('ok')
