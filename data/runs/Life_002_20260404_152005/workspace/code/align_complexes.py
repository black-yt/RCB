#!/usr/bin/env python3
import json
import math
import os
from dataclasses import dataclass, asdict
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import pairwise2
from Bio.PDB import PDBParser


THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F',
    'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'SEC': 'U',
    'PYL': 'O', 'ASX': 'B', 'GLX': 'Z', 'UNK': 'X'
}


@dataclass
class ChainRecord:
    chain_id: str
    sequence: str
    residue_ids: list
    ca_coords: np.ndarray


def ensure_dirs():
    for path in ['outputs', 'report/images']:
        os.makedirs(path, exist_ok=True)


def parse_structure(path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('struct', path)
    model = next(structure.get_models())
    chains = []
    for chain in model:
        residues = [res for res in chain if res.id[0] == ' ']
        sequence = []
        residue_ids = []
        coords = []
        for res in residues:
            if 'CA' not in res:
                continue
            residue_ids.append((int(res.id[1]), res.id[2].strip()))
            sequence.append(THREE_TO_ONE.get(res.resname.strip(), 'X'))
            coords.append(res['CA'].coord.astype(float))
        if coords:
            chains.append(ChainRecord(chain.id, ''.join(sequence), residue_ids, np.vstack(coords)))
    return chains


def kabsch(P, Q):
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    H = Q_centered.T @ P_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_P - centroid_Q @ R
    Q_aligned = Q @ R + t
    rmsd = np.sqrt(np.mean(np.sum((P - Q_aligned) ** 2, axis=1)))
    return R, t, Q_aligned, rmsd


def tm_d0(L):
    if L <= 15:
        return 0.5
    return 1.24 * ((L - 15) ** (1.0 / 3.0)) - 1.8


def tm_score(P, Q_aligned, L_norm):
    d0 = max(tm_d0(L_norm), 0.5)
    dists = np.sqrt(np.sum((P - Q_aligned) ** 2, axis=1))
    return float(np.sum(1.0 / (1.0 + (dists / d0) ** 2)) / L_norm), dists


def align_chains(query_chain, target_chain):
    aln = pairwise2.align.globalms(query_chain.sequence, target_chain.sequence, 2, -1, -5, -0.5, one_alignment_only=True)
    if not aln:
        return None
    aln = aln[0]
    q_seq, t_seq = aln.seqA, aln.seqB
    qi = ti = 0
    q_idx = []
    t_idx = []
    for qa, ta in zip(q_seq, t_seq):
        q_has = qa != '-'
        t_has = ta != '-'
        if q_has and t_has:
            q_idx.append(qi)
            t_idx.append(ti)
        if q_has:
            qi += 1
        if t_has:
            ti += 1
    if len(q_idx) < 20:
        return None
    P = query_chain.ca_coords[q_idx]
    Q = target_chain.ca_coords[t_idx]
    R, t, Q_aligned, rmsd = kabsch(P, Q)
    Lq = len(query_chain.sequence)
    Lt = len(target_chain.sequence)
    aligned_len = len(q_idx)
    tm_q, dists = tm_score(P, Q_aligned, Lq)
    tm_t, _ = tm_score(P, Q_aligned, Lt)
    seq_ident = float(np.mean([query_chain.sequence[i] == target_chain.sequence[j] for i, j in zip(q_idx, t_idx)]))
    return {
        'query_chain': query_chain.chain_id,
        'target_chain': target_chain.chain_id,
        'query_length': Lq,
        'target_length': Lt,
        'aligned_length': aligned_len,
        'coverage_query': aligned_len / Lq,
        'coverage_target': aligned_len / Lt,
        'sequence_identity': seq_ident,
        'rmsd': float(rmsd),
        'tm_score_query_norm': tm_q,
        'tm_score_target_norm': tm_t,
        'rotation_matrix': R.tolist(),
        'translation_vector': t.tolist(),
        'mean_distance': float(np.mean(dists)),
        'median_distance': float(np.median(dists)),
        'query_indices': q_idx,
        'target_indices': t_idx,
        'distances': dists.tolist(),
        'alignment_strings': {'query': q_seq, 'target': t_seq},
    }


def summarize_complex(query_chains, target_chains, pair_results):
    query_total = sum(len(c.sequence) for c in query_chains)
    target_total = sum(len(c.sequence) for c in target_chains)
    best = max(pair_results, key=lambda x: x['tm_score_query_norm'])
    complex_tm = best['tm_score_query_norm'] * (best['aligned_length'] / query_total)
    complex_target_tm = best['tm_score_target_norm'] * (best['aligned_length'] / target_total)
    return {
        'query_total_residues': query_total,
        'target_total_residues': target_total,
        'matched_chains': [{'query_chain': best['query_chain'], 'target_chain': best['target_chain']}],
        'unmatched_query_chains': [c.chain_id for c in query_chains if c.chain_id != best['query_chain']],
        'unmatched_target_chains': [c.chain_id for c in target_chains if c.chain_id != best['target_chain']],
        'global_rotation_matrix': best['rotation_matrix'],
        'global_translation_vector': best['translation_vector'],
        'best_pair_tm_query_norm': best['tm_score_query_norm'],
        'best_pair_tm_target_norm': best['tm_score_target_norm'],
        'coverage_penalized_complex_tm_query_norm': complex_tm,
        'coverage_penalized_complex_tm_target_norm': complex_target_tm,
        'best_pair_rmsd': best['rmsd'],
        'best_pair_aligned_length': best['aligned_length'],
        'best_pair_query_chain': best['query_chain'],
        'best_pair_target_chain': best['target_chain'],
    }


def plot_chain_lengths(query_chains, target_chains):
    rows = []
    for c in query_chains:
        rows.append({'structure': '7xg4', 'chain': c.chain_id, 'length': len(c.sequence)})
    for c in target_chains:
        rows.append({'structure': '6n40', 'chain': c.chain_id, 'length': len(c.sequence)})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x='chain', y='length', hue='structure', dodge=False)
    plt.title('Protein chain lengths in the two input structures')
    plt.tight_layout()
    plt.savefig('report/images/chain_lengths.png', dpi=200)
    plt.close()
    df.to_csv('outputs/chain_lengths.csv', index=False)


def plot_alignment_scores(pair_df):
    heat = pair_df.pivot(index='query_chain', columns='target_chain', values='tm_score_query_norm')
    plt.figure(figsize=(4, 5))
    sns.heatmap(heat, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'TM-score (query-normalized)'})
    plt.title('Best chain-to-chain similarity scores')
    plt.tight_layout()
    plt.savefig('report/images/tm_heatmap.png', dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=pair_df, x='aligned_length', y='rmsd', hue='query_chain', s=80)
    plt.title('Alignment extent versus RMSD')
    plt.tight_layout()
    plt.savefig('report/images/alignment_scatter.png', dpi=200)
    plt.close()


def plot_best_distance_profile(best_result):
    d = np.array(best_result['distances'])
    plt.figure(figsize=(7, 3.5))
    plt.plot(np.arange(1, len(d) + 1), d, linewidth=1.5)
    plt.xlabel('Aligned residue pair index')
    plt.ylabel('Cα distance after superposition (Å)')
    plt.title(f"Distance profile for best match {best_result['query_chain']} vs {best_result['target_chain']}")
    plt.tight_layout()
    plt.savefig('report/images/best_distance_profile.png', dpi=200)
    plt.close()


def plot_superposed_coordinates(query_chain, target_chain, best_result):
    q_idx = best_result['query_indices']
    t_idx = best_result['target_indices']
    P = query_chain.ca_coords[q_idx]
    Q = target_chain.ca_coords[t_idx]
    R = np.array(best_result['rotation_matrix'])
    t = np.array(best_result['translation_vector'])
    Q_aligned = Q @ R + t
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(P[:, 0], P[:, 1], P[:, 2], label=f"7xg4:{query_chain.chain_id}", color='tab:blue')
    ax.plot(Q_aligned[:, 0], Q_aligned[:, 1], Q_aligned[:, 2], label=f"6n40:{target_chain.chain_id} aligned", color='tab:orange')
    ax.set_title('Best chain pair after Kabsch superposition')
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('report/images/superposition_3d.png', dpi=200)
    plt.close()


def main():
    ensure_dirs()
    query_chains = parse_structure('data/7xg4.pdb')
    target_chains = parse_structure('data/6n40.pdb')

    pair_results = []
    for qc, tc in product(query_chains, target_chains):
        result = align_chains(qc, tc)
        if result is not None:
            pair_results.append(result)

    if not pair_results:
        raise RuntimeError('No valid chain alignments produced.')

    pair_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (list, dict))} for r in pair_results])
    pair_df = pair_df.sort_values(['tm_score_query_norm', 'aligned_length'], ascending=[False, False])
    pair_df.to_csv('outputs/chain_pair_alignment_metrics.csv', index=False)

    best = pair_results[int(pair_df.index[0])]
    complex_summary = summarize_complex(query_chains, target_chains, pair_results)

    structure_summary = {
        '7xg4': [asdict(c) | {'length': len(c.sequence)} for c in query_chains],
        '6n40': [asdict(c) | {'length': len(c.sequence)} for c in target_chains],
    }
    for key in structure_summary:
        for rec in structure_summary[key]:
            rec['ca_coords'] = None
    with open('outputs/structure_summary.json', 'w') as f:
        json.dump(structure_summary, f, indent=2)
    with open('outputs/best_alignment.json', 'w') as f:
        json.dump(best, f, indent=2)
    with open('outputs/complex_alignment_summary.json', 'w') as f:
        json.dump(complex_summary, f, indent=2)

    plot_chain_lengths(query_chains, target_chains)
    plot_alignment_scores(pair_df)
    plot_best_distance_profile(best)
    query_chain = next(c for c in query_chains if c.chain_id == best['query_chain'])
    target_chain = next(c for c in target_chains if c.chain_id == best['target_chain'])
    plot_superposed_coordinates(query_chain, target_chain, best)

    print('Best query chain:', best['query_chain'])
    print('Best target chain:', best['target_chain'])
    print('Aligned length:', best['aligned_length'])
    print('TM-score (query norm):', f"{best['tm_score_query_norm']:.4f}")
    print('Coverage-penalized complex TM-score:', f"{complex_summary['coverage_penalized_complex_tm_query_norm']:.4f}")


if __name__ == '__main__':
    main()
