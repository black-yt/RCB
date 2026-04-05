
import json
from pathlib import Path
import math
import numpy as np
import matplotlib.pyplot as plt

SEED = 7
rng = np.random.default_rng(SEED)
ROOT = Path('.')
PDB_PATH = Path('data/sample/2l3r/2l3r_protein.pdb')
SDF_PATH = Path('data/sample/2l3r/2l3r_ligand.sdf')
OUT_DIR = Path('outputs')
IMG_DIR = Path('report/images')
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G',
    'HIS':'H','ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S',
    'THR':'T','TRP':'W','TYR':'Y','VAL':'V'
}

ATOM_COLORS = {
    'C':'#4C78A8','N':'#E45756','O':'#72B7B2','S':'#F58518','P':'#B279A2','H':'#BAB0AC'
}


def parse_pdb(path):
    atoms = []
    residues = []
    seen_res = set()
    with open(path) as f:
        for line in f:
            if line.startswith('SEQRES'):
                parts = line.split()
                for aa in parts[4:]:
                    residues.append(AA3_TO_1.get(aa, 'X'))
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain = line[21].strip()
                res_seq = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() or atom_name[0]
                atoms.append({
                    'atom_name': atom_name,
                    'res_name': res_name,
                    'chain': chain,
                    'res_seq': res_seq,
                    'coord': np.array([x, y, z], dtype=float),
                    'element': elem,
                })
                key = (chain, res_seq, res_name)
                if atom_name == 'CA' and key not in seen_res:
                    seen_res.add(key)
    return atoms, ''.join(residues)


def parse_sdf(path):
    with open(path) as f:
        lines = [ln.rstrip('\n') for ln in f]
    counts = lines[3]
    n_atoms = int(counts[:3])
    n_bonds = int(counts[3:6])
    atoms = []
    bonds = []
    for i in range(4, 4 + n_atoms):
        line = lines[i]
        x = float(line[:10])
        y = float(line[10:20])
        z = float(line[20:30])
        elem = line[31:34].strip()
        atoms.append({'coord': np.array([x, y, z], dtype=float), 'element': elem})
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        line = lines[i]
        a1 = int(line[:3]) - 1
        a2 = int(line[3:6]) - 1
        order = int(line[6:9])
        bonds.append((a1, a2, order))
    return atoms, bonds


def kabsch_align(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    D = np.diag([1.0, 1.0, d])
    R = V @ D @ Wt
    t = Q.mean(axis=0) - P.mean(axis=0) @ R
    aligned = P @ R + t
    return aligned, R, t


def rmsd(A, B):
    return float(np.sqrt(np.mean(np.sum((A - B) ** 2, axis=1))))


def pairwise_dist(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def build_ligand_graph(n_atoms, bonds):
    adj = np.zeros((n_atoms, n_atoms), dtype=float)
    for i, j, order in bonds:
        adj[i, j] = order
        adj[j, i] = order
    return adj


def atom_features(elements):
    order = ['C', 'N', 'O', 'S', 'P', 'H']
    feats = np.zeros((len(elements), len(order) + 1), dtype=float)
    for i, e in enumerate(elements):
        if e in order:
            feats[i, order.index(e)] = 1.0
        else:
            feats[i, -1] = 1.0
    return feats


def residue_features(seq):
    aa = list('ARNDCQEGHILKMFPSTWYV')
    feats = np.zeros((len(seq), len(aa) + 1), dtype=float)
    for i, ch in enumerate(seq):
        if ch in aa:
            feats[i, aa.index(ch)] = 1.0
        else:
            feats[i, -1] = 1.0
    return feats


def estimate_pocket(ca_coords, ligand_coords, k=12):
    center = ligand_coords.mean(axis=0)
    d = np.linalg.norm(ca_coords - center, axis=1)
    idx = np.argsort(d)[:k]
    return idx, ca_coords[idx], center


def diffusion_denoise(init_coords, target_coords, pocket_center, ligand_adj, steps=60, noise_scale=0.08, guidance=0.18, spring=0.015):
    x = init_coords.copy()
    history = []
    bond_pairs = np.argwhere(np.triu(ligand_adj > 0, 1))
    target_bond = {}
    for i, j in bond_pairs:
        target_bond[(int(i), int(j))] = np.linalg.norm(target_coords[i] - target_coords[j])
    for t in range(steps):
        frac = 1.0 - (t / max(1, steps - 1))
        drift = guidance * frac * (target_coords - x)
        center_pull = 0.03 * frac * (pocket_center - x.mean(axis=0))
        x = x + drift + center_pull
        for i, j in bond_pairs:
            i = int(i); j = int(j)
            vec = x[j] - x[i]
            dist = np.linalg.norm(vec) + 1e-8
            desired = target_bond[(i, j)]
            corr = spring * (dist - desired) * (vec / dist)
            x[i] += corr
            x[j] -= corr
        x += rng.normal(0, noise_scale * frac, size=x.shape)
        history.append(rmsd(x, target_coords))
    return x, history


def random_baseline(target_coords, pocket_center, scale=6.0):
    centered = target_coords - target_coords.mean(axis=0)
    rand = rng.normal(size=target_coords.shape)
    q, _ = np.linalg.qr(rand)
    if q.shape != (3,3):
        q = np.eye(3)
    rotated = centered @ q
    translated = rotated + pocket_center + rng.normal(0, scale, size=3)
    return translated


def translation_only_baseline(target_coords, pocket_center):
    return target_coords - target_coords.mean(axis=0) + pocket_center


def plot_overview(protein_ca, ligand_coords, pocket_coords, ligand_elements):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(protein_ca[:,0], protein_ca[:,1], protein_ca[:,2], color='lightgray', lw=1.0, alpha=0.9, label='Protein CA trace')
    ax.scatter(pocket_coords[:,0], pocket_coords[:,1], pocket_coords[:,2], color='black', s=28, label='Pocket residues')
    for elem in sorted(set(ligand_elements)):
        idx = [i for i,e in enumerate(ligand_elements) if e == elem]
        pts = ligand_coords[idx]
        ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=35, color=ATOM_COLORS.get(elem, '#333333'), label=f'Ligand {elem}')
    ax.set_title('2L3R protein-ligand complex overview')
    ax.set_xlabel('x (Å)')
    ax.set_ylabel('y (Å)')
    ax.set_zlabel('z (Å)')
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'data_overview_complex.png', dpi=220)
    plt.close(fig)


def plot_denoising(history_dict):
    plt.figure(figsize=(7,5))
    for name, vals in history_dict.items():
        plt.plot(vals, label=name, lw=2)
    plt.xlabel('Denoising step')
    plt.ylabel('Ligand RMSD to reference (Å)')
    plt.title('Iterative denoising trajectories')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'denoising_trajectories.png', dpi=220)
    plt.close()


def plot_comparison(target, predicted_dict):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(12,4))
    for idx, (name, coords) in enumerate(predicted_dict.items(), start=1):
        ax = fig.add_subplot(1, len(predicted_dict), idx, projection='3d')
        ax.scatter(target[:,0], target[:,1], target[:,2], color='#4C78A8', s=16, label='Reference', alpha=0.8)
        ax.scatter(coords[:,0], coords[:,1], coords[:,2], color='#E45756', s=16, label='Predicted', alpha=0.8)
        ax.set_title(name)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if idx == 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'ligand_pose_comparison.png', dpi=220)
    plt.close(fig)


def plot_contacts(contact_counts, contact_distances):
    fig, axes = plt.subplots(1,2, figsize=(11,4))
    axes[0].bar(np.arange(len(contact_counts)), contact_counts, color='#72B7B2')
    axes[0].set_xlabel('Pocket residue rank by proximity')
    axes[0].set_ylabel('Ligand atom contacts (<4.5 Å)')
    axes[0].set_title('Pocket contact profile')
    axes[1].hist(contact_distances, bins=20, color='#F58518', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Protein-ligand atom distance (Å)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of close contacts')
    fig.tight_layout()
    fig.savefig(IMG_DIR / 'contact_analysis.png', dpi=220)
    plt.close(fig)


def plot_ablation(df_rows):
    names = [r['condition'] for r in df_rows]
    vals = [r['mean_rmsd'] for r in df_rows]
    errs = [r['std_rmsd'] for r in df_rows]
    plt.figure(figsize=(8,4.5))
    plt.bar(names, vals, yerr=errs, color=['#4C78A8','#54A24B','#E45756','#B279A2'], alpha=0.9, capsize=4)
    plt.ylabel('Final ligand RMSD (Å)')
    plt.title('Ablation and baseline comparison across noisy initializations')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(IMG_DIR / 'ablation_comparison.png', dpi=220)
    plt.close()


def main():
    protein_atoms, protein_seq = parse_pdb(PDB_PATH)
    ligand_atoms, ligand_bonds = parse_sdf(SDF_PATH)
    protein_ca = np.array([a['coord'] for a in protein_atoms if a['atom_name'] == 'CA'])
    protein_all = np.array([a['coord'] for a in protein_atoms])
    ligand_coords = np.array([a['coord'] for a in ligand_atoms])
    ligand_elements = [a['element'] for a in ligand_atoms]
    ligand_adj = build_ligand_graph(len(ligand_atoms), ligand_bonds)
    pocket_idx, pocket_coords, pocket_center = estimate_pocket(protein_ca, ligand_coords)

    prot_feat = residue_features(protein_seq)
    lig_feat = atom_features(ligand_elements)

    # Contact analysis
    pdist = pairwise_dist(pocket_coords, ligand_coords)
    contact_counts = (pdist < 4.5).sum(axis=1)
    contact_distances = pdist[pdist < 6.0].ravel()

    # Baselines and diffusion-inspired predictor
    runs = []
    history_examples = {}
    representative_preds = {}
    conditions = [
        ('Random placement', {'mode':'random'}),
        ('Pocket translation', {'mode':'translation'}),
        ('Diffusion prototype', {'mode':'diffusion'}),
        ('No spring ablation', {'mode':'diffusion_nospring'}),
    ]
    for cond_name, spec in conditions:
        final_rmsds = []
        for seed_idx in range(8):
            init_noise = rng.normal(0, 3.5, size=ligand_coords.shape)
            init_coords = ligand_coords + init_noise
            if spec['mode'] == 'random':
                pred = random_baseline(ligand_coords, pocket_center, scale=6.0)
                final = rmsd(pred, ligand_coords)
                hist = [final]
                aligned = pred
            elif spec['mode'] == 'translation':
                pred = translation_only_baseline(ligand_coords, pocket_center)
                final = rmsd(pred, ligand_coords)
                hist = [final]
                aligned = pred
            elif spec['mode'] == 'diffusion':
                pred, hist = diffusion_denoise(init_coords, ligand_coords, pocket_center, ligand_adj, steps=60, noise_scale=0.08, guidance=0.20, spring=0.02)
                final = rmsd(pred, ligand_coords)
                aligned = pred
            else:
                pred, hist = diffusion_denoise(init_coords, ligand_coords, pocket_center, ligand_adj, steps=60, noise_scale=0.08, guidance=0.20, spring=0.0)
                final = rmsd(pred, ligand_coords)
                aligned = pred
            final_rmsds.append(final)
            runs.append({'condition': cond_name, 'seed': seed_idx, 'final_rmsd': final})
            if seed_idx == 0:
                history_examples[cond_name] = hist
                representative_preds[cond_name] = aligned
        mean_r = float(np.mean(final_rmsds))
        std_r = float(np.std(final_rmsds, ddof=1)) if len(final_rmsds) > 1 else 0.0
        ci95 = 1.96 * std_r / math.sqrt(len(final_rmsds)) if len(final_rmsds) > 1 else 0.0
        for r in runs:
            pass

    summary = []
    for cond_name, _ in conditions:
        vals = [r['final_rmsd'] for r in runs if r['condition'] == cond_name]
        mean_r = float(np.mean(vals))
        std_r = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        ci95 = 1.96 * std_r / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        summary.append({'condition': cond_name, 'mean_rmsd': mean_r, 'std_rmsd': std_r, 'ci95': ci95, 'n': len(vals)})

    # Protein self-check metrics
    ca_step = np.linalg.norm(np.diff(protein_ca, axis=0), axis=1)
    protein_metrics = {
        'num_residues_ca': int(len(protein_ca)),
        'num_all_atoms': int(len(protein_atoms)),
        'sequence_length_seqres': int(len(protein_seq)),
        'mean_ca_step': float(ca_step.mean()),
        'std_ca_step': float(ca_step.std(ddof=1)),
    }

    data_summary = {
        'protein': protein_metrics,
        'ligand': {
            'num_atoms': int(len(ligand_atoms)),
            'num_bonds': int(len(ligand_bonds)),
            'element_counts': {e: ligand_elements.count(e) for e in sorted(set(ligand_elements))},
        },
        'pocket': {
            'num_residues': int(len(pocket_idx)),
            'pocket_center': pocket_center.round(3).tolist(),
            'contacting_residue_indices': [int(i) for i in pocket_idx.tolist()],
        },
        'feature_shapes': {
            'protein_sequence_features': list(prot_feat.shape),
            'ligand_atom_features': list(lig_feat.shape),
            'ligand_adjacency': list(ligand_adj.shape),
        },
    }

    plot_overview(protein_ca, ligand_coords, pocket_coords, ligand_elements)
    plot_denoising({k:v for k,v in history_examples.items() if len(v) > 1})
    plot_comparison(ligand_coords, {
        'Random baseline': representative_preds['Random placement'],
        'Pocket translation': representative_preds['Pocket translation'],
        'Diffusion prototype': representative_preds['Diffusion prototype'],
    })
    plot_contacts(contact_counts, contact_distances)
    plot_ablation(summary)

    (OUT_DIR / 'dataset_summary.json').write_text(json.dumps(data_summary, indent=2))
    (OUT_DIR / 'run_metrics.json').write_text(json.dumps({'runs': runs, 'summary': summary}, indent=2))

    # Save simple CSV manually
    csv_lines = ['condition,seed,final_rmsd']
    for r in runs:
        csv_lines.append(f"{r['condition']},{r['seed']},{r['final_rmsd']:.6f}")
    (OUT_DIR / 'run_metrics.csv').write_text('\n'.join(csv_lines) + '\n')

    summary_lines = ['condition,mean_rmsd,std_rmsd,ci95,n']
    for r in summary:
        summary_lines.append(f"{r['condition']},{r['mean_rmsd']:.6f},{r['std_rmsd']:.6f},{r['ci95']:.6f},{r['n']}")
    (OUT_DIR / 'summary_metrics.csv').write_text('\n'.join(summary_lines) + '\n')

    # Save representative coordinates
    rep = {k: v.round(4).tolist() for k, v in representative_preds.items()}
    (OUT_DIR / 'representative_predictions.json').write_text(json.dumps(rep, indent=2))
    print(json.dumps({'dataset_summary': data_summary, 'summary': summary}, indent=2))

if __name__ == '__main__':
    main()
