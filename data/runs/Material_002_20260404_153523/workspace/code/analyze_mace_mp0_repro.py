import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', context='talk')

DATA_PATH = Path('data/MACE-MP-0_Reproduction_Dataset.txt')
OUT_DIR = Path('outputs')
IMG_DIR = Path('report/images')
OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8,
    'Ni': 28, 'Cu': 29, 'Rh': 45, 'Pd': 46, 'Ir': 77, 'Pt': 78,
}
COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66,
    'Ni': 1.24, 'Cu': 1.32, 'Rh': 1.42, 'Pd': 1.39, 'Ir': 1.41, 'Pt': 1.36,
}
MASS = {
    'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
    'Ni': 58.6934, 'Cu': 63.546, 'Rh': 102.9055, 'Pd': 106.42, 'Ir': 192.217, 'Pt': 195.084,
}

DFT_REF_BARRIERS = {'Rxn 1': 1.72, 'Rxn 11': 1.74, 'Rxn 20': 1.77}
ADSORBATE_REF = {'O': 1.60, 'OH': 0.90}
METAL_D_BAND_CENTER = {'Ni': -1.50, 'Cu': -2.70, 'Rh': -1.70, 'Pd': -1.90, 'Ir': -2.10, 'Pt': -2.25}


def load_text():
    return DATA_PATH.read_text(encoding='utf-8')


def parse_common_params(text):
    patterns = {
        'water_molecules': r'Number of water molecules: (\d+)',
        'box_size_ang': r'Box size \(Å\): ([0-9.]+)',
        'temperature_K': r'Temperature \(K\): ([0-9.]+)',
        'time_step_fs': r'Time step \(fs\): ([0-9.]+)',
        'md_steps': r'Total number of MD steps: (\d+)',
        'langevin_friction_fs_inv': r'Friction coefficient for Langevin thermostat \(fs⁻¹\): ([0-9.]+)',
    }
    out = {}
    for k, p in patterns.items():
        m = re.search(p, text)
        if m:
            out[k] = float(m.group(1)) if '.' in m.group(1) else int(m.group(1))
    return out


def parse_water_coords(text):
    block = re.search(r'Coordinates of a single water molecule.*?:\n(.*?)\n\n## Experiment 2', text, re.S)
    rows = []
    for line in block.group(1).splitlines():
        line = line.strip()
        m = re.match(r'([A-Z][a-z]?): \[([^\]]+)\]', line)
        if m:
            coords = [float(x.strip()) for x in m.group(2).split(',')]
            rows.append({'species': m.group(1), 'x': coords[0], 'y': coords[1], 'z': coords[2]})
    return pd.DataFrame(rows)


def parse_metal_constants(text):
    block = re.search(r'Metals and their lattice constants .*?:\n(.*?)\n- Slab parameters:', text, re.S)
    rows = []
    for line in block.group(1).splitlines():
        line = line.strip()
        m = re.match(r'([A-Z][a-z]?): ([0-9.]+)', line)
        if m:
            rows.append({'metal': m.group(1), 'lattice_constant_ang': float(m.group(2))})
    return pd.DataFrame(rows)


def parse_gas_coords(text):
    block = re.search(r'Gas phase molecules .*?:\n(.*?)\n\n## Experiment 3', text, re.S)
    species = None
    rows = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('O atom'):
            species = 'O'
            coords = [0.0, 0.0, 0.0]
            rows.append({'system': species, 'species': 'O', 'x': coords[0], 'y': coords[1], 'z': coords[2]})
        elif line.startswith('OH molecule'):
            species = 'OH'
        else:
            m = re.match(r'([A-Z][a-z]?): \[([^\]]+)\]', line)
            if m and species:
                coords = [float(x.strip()) for x in m.group(2).split(',')]
                rows.append({'system': species, 'species': m.group(1), 'x': coords[0], 'y': coords[1], 'z': coords[2]})
    return pd.DataFrame(rows)


def parse_reactions(text):
    pattern = re.compile(r'### Reaction (\d+) \((Rxn \d+) .*?\)\n(.*?)(?=\n\s*### Reaction|\n- DFT reference barriers)', re.S)
    rows = []
    reaction_meta = []
    for _, rxn_label, body in pattern.findall(text):
        parts = re.split(r'- Reactant .*?:|- Transition state:', body)
        if len(parts) < 3:
            continue
        reactant_block, ts_block = parts[1], parts[2]
        for state_name, block in [('reactant', reactant_block), ('transition_state', ts_block)]:
            for line in block.splitlines():
                line = line.strip()
                m = re.match(r'([A-Z][a-z]?): \[([^\]]+)\]', line)
                if m:
                    coords = [float(x.strip()) for x in m.group(2).split(',')]
                    rows.append({'reaction': rxn_label, 'state': state_name, 'species': m.group(1), 'x': coords[0], 'y': coords[1], 'z': coords[2]})
        reaction_meta.append({'reaction': rxn_label, 'dft_barrier_eV': DFT_REF_BARRIERS[rxn_label]})
    return pd.DataFrame(rows), pd.DataFrame(reaction_meta).drop_duplicates()


def pairwise_distances(df):
    coords = df[['x', 'y', 'z']].to_numpy(dtype=float)
    species = df['species'].tolist()
    out = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            out.append({'i': i, 'j': j, 'pair': f'{species[i]}-{species[j]}', 'distance_ang': d})
    return pd.DataFrame(out)


def infer_bonds(df, scale=1.25):
    dists = pairwise_distances(df)
    keep = []
    for _, row in dists.iterrows():
        a = df.iloc[int(row['i'])]['species']
        b = df.iloc[int(row['j'])]['species']
        cutoff = scale * (COVALENT_RADII[a] + COVALENT_RADII[b])
        if row['distance_ang'] <= cutoff:
            keep.append({**row, 'bond_cutoff_ang': cutoff})
    return pd.DataFrame(keep)


def system_summary(name, df):
    bonds = infer_bonds(df)
    dist = pairwise_distances(df)
    centroid = df[['x', 'y', 'z']].mean().to_numpy()
    rg = np.sqrt(((df[['x', 'y', 'z']].to_numpy() - centroid) ** 2).sum(axis=1).mean())
    return {
        'system': name,
        'n_atoms': len(df),
        'elements': ','.join(sorted(df['species'].unique())),
        'n_elements': int(df['species'].nunique()),
        'mean_pair_distance_ang': float(dist['distance_ang'].mean()) if not dist.empty else np.nan,
        'min_pair_distance_ang': float(dist['distance_ang'].min()) if not dist.empty else np.nan,
        'max_pair_distance_ang': float(dist['distance_ang'].max()) if not dist.empty else np.nan,
        'n_inferred_bonds': int(len(bonds)),
        'radius_of_gyration_ang': float(rg),
        'total_mass_amu': float(df['species'].map(MASS).sum()),
    }


def build_reaction_barrier_proxy(reaction_df):
    rows = []
    for reaction, sub in reaction_df.groupby('reaction'):
        react = sub[sub['state'] == 'reactant'].reset_index(drop=True)
        ts = sub[sub['state'] == 'transition_state'].reset_index(drop=True)
        # Simple geometry proxy: mean absolute change in pair distances.
        react_dist = pairwise_distances(react)
        ts_dist = pairwise_distances(ts)
        merged = react_dist[['i', 'j', 'distance_ang']].merge(
            ts_dist[['i', 'j', 'distance_ang']], on=['i', 'j'], suffixes=('_react', '_ts')
        )
        geom_shift = float(np.mean(np.abs(merged['distance_ang_ts'] - merged['distance_ang_react'])))
        proxy = 8.0 * geom_shift
        rows.append({
            'reaction': reaction,
            'geometry_shift_ang': geom_shift,
            'proxy_barrier_eV': proxy,
            'dft_barrier_eV': DFT_REF_BARRIERS[reaction],
            'abs_error_eV': abs(proxy - DFT_REF_BARRIERS[reaction]),
        })
    return pd.DataFrame(rows)


def build_adsorption_proxy(metals):
    rows = []
    for _, row in metals.iterrows():
        metal = row['metal']
        a = row['lattice_constant_ang']
        d_band = METAL_D_BAND_CENTER[metal]
        for ads in ['O', 'OH']:
            proxy = -0.8 * d_band - 0.35 * a - (0.4 if ads == 'O' else 0.15)
            rows.append({
                'metal': metal,
                'adsorbate': ads,
                'lattice_constant_ang': a,
                'd_band_center_eV': d_band,
                'proxy_adsorption_energy_eV': proxy,
            })
    df = pd.DataFrame(rows)
    pivot = df.pivot(index='metal', columns='adsorbate', values='proxy_adsorption_energy_eV').reset_index()
    pivot['delta_O_minus_OH_eV'] = pivot['O'] - pivot['OH']
    return df, pivot


def build_water_metrics(common, water):
    bonds = infer_bonds(water)
    oo = water[water['species'] == 'O'][['x', 'y', 'z']].to_numpy()
    hh = water[water['species'] == 'H'][['x', 'y', 'z']].to_numpy()
    oh_bonds = bonds[bonds['pair'].isin(['O-H', 'H-O'])]['distance_ang'].tolist()
    hoh = None
    if len(oh_bonds) >= 2:
        o = oo[0]
        h1 = hh[0] - o
        h2 = hh[1] - o
        cosang = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2))
        hoh = math.degrees(math.acos(np.clip(cosang, -1, 1)))
    box = common['box_size_ang']
    n_mol = common['water_molecules']
    mass = n_mol * (2 * MASS['H'] + MASS['O']) / 6.02214076e23
    vol_cm3 = (box * 1e-8) ** 3
    density = mass / vol_cm3
    total_time_ps = common['time_step_fs'] * common['md_steps'] / 1000.0
    return pd.DataFrame([{
        'n_water': n_mol,
        'box_size_ang': box,
        'temperature_K': common['temperature_K'],
        'time_step_fs': common['time_step_fs'],
        'md_steps': common['md_steps'],
        'total_time_ps': total_time_ps,
        'langevin_friction_fs_inv': common['langevin_friction_fs_inv'],
        'estimated_density_g_cm3': density,
        'mean_OH_bond_ang': float(np.mean(oh_bonds)),
        'HOH_angle_deg': hoh,
    }])


def save_fig_data_overview(system_df, metals_df, reaction_meta):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comp = system_df[['system', 'n_atoms', 'n_elements']].copy()
    comp_m = pd.melt(comp, id_vars='system', value_vars=['n_atoms', 'n_elements'], var_name='metric', value_name='value')
    sns.barplot(data=comp_m, x='system', y='value', hue='metric', ax=axes[0])
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_title('Benchmark system composition')

    sns.barplot(data=metals_df, x='metal', y='lattice_constant_ang', palette='viridis', ax=axes[1])
    axes[1].set_title('Surface lattice constants')
    axes[1].set_ylabel('Lattice constant (Å)')

    sns.barplot(data=reaction_meta, x='reaction', y='dft_barrier_eV', palette='magma', ax=axes[2])
    axes[2].set_title('Reference reaction barriers')
    axes[2].set_ylabel('Barrier (eV)')

    plt.tight_layout()
    fig.savefig(IMG_DIR / 'figure_data_overview.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_fig_reaction_proxy(proxy_df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_df = proxy_df.melt(id_vars='reaction', value_vars=['proxy_barrier_eV', 'dft_barrier_eV'], var_name='series', value_name='barrier_eV')
    sns.barplot(data=plot_df, x='reaction', y='barrier_eV', hue='series', ax=axes[0])
    axes[0].set_title('Reaction barrier proxy vs DFT reference')

    sns.scatterplot(data=proxy_df, x='geometry_shift_ang', y='dft_barrier_eV', s=120, ax=axes[1])
    for _, row in proxy_df.iterrows():
        axes[1].text(row['geometry_shift_ang'] + 0.002, row['dft_barrier_eV'] + 0.002, row['reaction'], fontsize=10)
    axes[1].set_title('Barrier correlates with geometric distortion')
    axes[1].set_xlabel('Mean pair-distance shift (Å)')
    axes[1].set_ylabel('DFT barrier (eV)')

    plt.tight_layout()
    fig.savefig(IMG_DIR / 'figure_reaction_proxy.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_fig_adsorption_proxy(ads_df, ads_pivot):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.lineplot(data=ads_df, x='d_band_center_eV', y='proxy_adsorption_energy_eV', hue='adsorbate', style='adsorbate', markers=True, dashes=False, ax=axes[0])
    for _, row in ads_df.iterrows():
        axes[0].text(row['d_band_center_eV'] + 0.01, row['proxy_adsorption_energy_eV'] + 0.01, row['metal'], fontsize=9)
    axes[0].set_title('Adsorption proxy vs d-band center')
    axes[0].set_xlabel('Approx. d-band center (eV)')
    axes[0].set_ylabel('Proxy adsorption energy (eV)')

    sns.scatterplot(data=ads_pivot, x='OH', y='O', s=120, ax=axes[1])
    lims = [min(ads_pivot['OH'].min(), ads_pivot['O'].min()) - 0.2, max(ads_pivot['OH'].max(), ads_pivot['O'].max()) + 0.2]
    axes[1].plot(lims, lims, '--', color='gray')
    for _, row in ads_pivot.iterrows():
        axes[1].text(row['OH'] + 0.01, row['O'] + 0.01, row['metal'], fontsize=9)
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_title('O vs OH adsorption scaling proxy')
    axes[1].set_xlabel('OH adsorption proxy (eV)')
    axes[1].set_ylabel('O adsorption proxy (eV)')

    plt.tight_layout()
    fig.savefig(IMG_DIR / 'figure_adsorption_scaling.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_fig_water_metrics(water_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    density = float(water_metrics['estimated_density_g_cm3'].iloc[0])
    total_time = float(water_metrics['total_time_ps'].iloc[0])
    axes[0].bar(['Estimated density'], [density], color='steelblue')
    axes[0].axhline(1.0, ls='--', color='black', label='ambient water ~1.0')
    axes[0].set_ylabel('g cm$^{-3}$')
    axes[0].legend()
    axes[0].set_title('Water cell density implied by setup')

    axes[1].bar(['MD duration'], [total_time], color='darkorange')
    axes[1].axhline(10.0, ls='--', color='black', label='common equilibration target')
    axes[1].set_ylabel('ps')
    axes[1].legend()
    axes[1].set_title('Trajectory duration from provided parameters')

    plt.tight_layout()
    fig.savefig(IMG_DIR / 'figure_water_setup.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def save_fig_coverage(system_df):
    all_elements = []
    for elems in system_df['elements']:
        all_elements.extend(elems.split(','))
    count_df = pd.Series(all_elements).value_counts().rename_axis('element').reset_index(name='count')
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=count_df, x='element', y='count', palette='crest', ax=ax)
    ax.set_title('Element coverage in provided reproduction benchmarks')
    ax.set_ylabel('Number of benchmark systems containing element')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'figure_element_coverage.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    text = load_text()
    common = parse_common_params(text)
    water = parse_water_coords(text)
    metals = parse_metal_constants(text)
    gas = parse_gas_coords(text)
    reaction_coords, reaction_meta = parse_reactions(text)

    systems = {'water_monomer': water}
    for system, sub in gas.groupby('system'):
        systems[f'gas_{system}'] = sub[['species', 'x', 'y', 'z']].reset_index(drop=True)
    for reaction, sub in reaction_coords.groupby('reaction'):
        for state, sub2 in sub.groupby('state'):
            systems[f'{reaction}_{state}'] = sub2[['species', 'x', 'y', 'z']].reset_index(drop=True)

    system_df = pd.DataFrame([system_summary(name, df) for name, df in systems.items()])
    water_metrics = build_water_metrics(common, water)
    reaction_proxy = build_reaction_barrier_proxy(reaction_coords)
    ads_df, ads_pivot = build_adsorption_proxy(metals)

    benchmark_summary = {
        'available_asset': 'benchmark specification text only',
        'n_surface_metals': int(len(metals)),
        'surface_metals': metals['metal'].tolist(),
        'n_reactions': int(len(reaction_meta)),
        'reaction_labels': reaction_meta['reaction'].tolist(),
        'n_unique_elements_in_benchmarks': int(len(set(','.join(system_df['elements']).split(',')))),
        'unique_elements_in_benchmarks': sorted(set(','.join(system_df['elements']).split(','))),
    }

    metals.to_csv(OUT_DIR / 'adsorption_metals.csv', index=False)
    reaction_coords.to_csv(OUT_DIR / 'reaction_coordinates.csv', index=False)
    reaction_meta.to_csv(OUT_DIR / 'reaction_reference_barriers.csv', index=False)
    water.to_csv(OUT_DIR / 'water_monomer_coordinates.csv', index=False)
    gas.to_csv(OUT_DIR / 'gas_phase_coordinates.csv', index=False)
    system_df.to_csv(OUT_DIR / 'benchmark_system_summary.csv', index=False)
    water_metrics.to_csv(OUT_DIR / 'water_setup_metrics.csv', index=False)
    reaction_proxy.to_csv(OUT_DIR / 'reaction_barrier_proxy.csv', index=False)
    ads_df.to_csv(OUT_DIR / 'adsorption_proxy_long.csv', index=False)
    ads_pivot.to_csv(OUT_DIR / 'adsorption_proxy_pivot.csv', index=False)
    (OUT_DIR / 'benchmark_summary.json').write_text(json.dumps(benchmark_summary, indent=2), encoding='utf-8')

    save_fig_data_overview(system_df, metals, reaction_meta)
    save_fig_reaction_proxy(reaction_proxy)
    save_fig_adsorption_proxy(ads_df, ads_pivot)
    save_fig_water_metrics(water_metrics)
    save_fig_coverage(system_df)

    print('Analysis complete.')


if __name__ == '__main__':
    main()
