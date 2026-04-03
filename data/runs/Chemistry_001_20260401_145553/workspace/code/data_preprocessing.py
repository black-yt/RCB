"""
Data preprocessing module for 2L3R protein-ligand complex.
Parses PDB and SDF files, extracts structural features, and computes basic metrics.
"""

import numpy as np
import os
from collections import defaultdict

DATA_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260401_145553/data/sample/2l3r"
OUTPUT_DIR = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Chemistry_001_20260401_145553/outputs"


# Amino acid 3-letter to 1-letter code mapping
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
}

# Amino acid properties
AA_PROPERTIES = {
    'A': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': False, 'mw': 89.1},
    'R': {'hydrophobic': False, 'polar': True,  'charged': True,  'aromatic': False, 'mw': 174.2},
    'N': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': False, 'mw': 132.1},
    'D': {'hydrophobic': False, 'polar': True,  'charged': True,  'aromatic': False, 'mw': 133.1},
    'C': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': False, 'mw': 121.2},
    'Q': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': False, 'mw': 146.2},
    'E': {'hydrophobic': False, 'polar': True,  'charged': True,  'aromatic': False, 'mw': 147.1},
    'G': {'hydrophobic': False, 'polar': False, 'charged': False, 'aromatic': False, 'mw': 75.0},
    'H': {'hydrophobic': False, 'polar': True,  'charged': True,  'aromatic': True,  'mw': 155.2},
    'I': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': False, 'mw': 131.2},
    'L': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': False, 'mw': 131.2},
    'K': {'hydrophobic': False, 'polar': True,  'charged': True,  'aromatic': False, 'mw': 146.2},
    'M': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': False, 'mw': 149.2},
    'F': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': True,  'mw': 165.2},
    'P': {'hydrophobic': False, 'polar': False, 'charged': False, 'aromatic': False, 'mw': 115.1},
    'S': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': False, 'mw': 105.1},
    'T': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': False, 'mw': 119.1},
    'W': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': True,  'mw': 204.2},
    'Y': {'hydrophobic': False, 'polar': True,  'charged': False, 'aromatic': True,  'mw': 181.2},
    'V': {'hydrophobic': True,  'polar': False, 'charged': False, 'aromatic': False, 'mw': 117.1},
}


def parse_pdb(filepath):
    """Parse PDB file and extract structural information."""
    atoms = []
    ca_atoms = []
    residues = {}
    seqres_sequence = []

    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('SEQRES'):
                parts = line.split()
                for aa in parts[4:]:
                    if aa in AA_3TO1:
                        seqres_sequence.append(AA_3TO1[aa])

            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    record_type = line[:6].strip()
                    atom_num = int(line[6:11])
                    atom_name = line[12:16].strip()
                    alt_loc = line[16].strip()
                    res_name = line[17:20].strip()
                    chain_id = line[21].strip()
                    res_num = int(line[22:26])
                    icode = line[26].strip()
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = float(line[54:60]) if len(line) > 54 else 1.0
                    b_factor = float(line[60:66]) if len(line) > 60 else 0.0
                    element = line[76:78].strip() if len(line) > 76 else ''

                    atom = {
                        'record_type': record_type,
                        'atom_num': atom_num,
                        'atom_name': atom_name,
                        'res_name': res_name,
                        'chain_id': chain_id,
                        'res_num': res_num,
                        'x': x, 'y': y, 'z': z,
                        'occupancy': occupancy,
                        'b_factor': b_factor,
                        'element': element,
                    }
                    atoms.append(atom)

                    if atom_name == 'CA':
                        ca_atoms.append(atom)

                    res_key = (chain_id, res_num)
                    if res_key not in residues:
                        residues[res_key] = {'res_name': res_name, 'atoms': [], 'res_num': res_num}
                    residues[res_key]['atoms'].append(atom)

                except (ValueError, IndexError):
                    continue

    # Get sequence from CA atoms
    ca_sequence = ''.join([AA_3TO1.get(a['res_name'], 'X') for a in ca_atoms])

    return {
        'atoms': atoms,
        'ca_atoms': ca_atoms,
        'residues': residues,
        'seqres_sequence': ''.join(seqres_sequence),
        'ca_sequence': ca_sequence,
        'n_atoms': len(atoms),
        'n_residues': len(residues),
        'n_ca': len(ca_atoms),
    }


def get_ca_coords(protein_data):
    """Extract CA atom coordinates as numpy array."""
    ca_atoms = protein_data['ca_atoms']
    coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    return coords


def compute_distance_matrix(coords):
    """Compute pairwise distance matrix."""
    n = len(coords)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
    return dist_matrix


def compute_contact_map(dist_matrix, threshold=8.0):
    """Compute contact map from distance matrix."""
    return (dist_matrix < threshold).astype(float)


def compute_radius_of_gyration(coords):
    """Compute radius of gyration."""
    center = np.mean(coords, axis=0)
    dists = np.sqrt(np.sum((coords - center)**2, axis=1))
    rg = np.sqrt(np.mean(dists**2))
    return rg


def compute_end_to_end_distance(coords):
    """Compute end-to-end distance (first to last CA)."""
    return np.linalg.norm(coords[-1] - coords[0])


def parse_sdf(filepath):
    """Parse SDF file and extract molecular information."""
    atoms = []
    bonds = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # SDF file format:
    # Line 1: molecule name
    # Line 2: header
    # Line 3: comment
    # Line 4: counts line
    mol_name = lines[0].strip()

    # Find the counts line (line 4, index 3)
    counts_line = lines[3]
    n_atoms = int(counts_line[0:3])
    n_bonds = int(counts_line[3:6])

    # Parse atoms
    for i in range(4, 4 + n_atoms):
        line = lines[i]
        try:
            x = float(line[0:10])
            y = float(line[10:20])
            z = float(line[20:30])
            element = line[31:34].strip()

            atoms.append({
                'x': x, 'y': y, 'z': z,
                'element': element,
                'idx': i - 4
            })
        except (ValueError, IndexError):
            continue

    # Parse bonds
    for i in range(4 + n_atoms, 4 + n_atoms + n_bonds):
        line = lines[i]
        try:
            atom1 = int(line[0:3]) - 1  # Convert to 0-indexed
            atom2 = int(line[3:6]) - 1
            bond_type = int(line[6:9])
            bonds.append({'atom1': atom1, 'atom2': atom2, 'bond_type': bond_type})
        except (ValueError, IndexError):
            continue

    # Parse properties from M  CHG and M  RAD lines
    formal_charges = {}
    for line in lines:
        if line.startswith('M  CHG'):
            parts = line.split()
            n_entries = int(parts[2])
            for j in range(n_entries):
                atom_idx = int(parts[3 + 2*j]) - 1
                charge = int(parts[4 + 2*j])
                formal_charges[atom_idx] = charge

    # Count elements
    element_counts = defaultdict(int)
    for atom in atoms:
        element_counts[atom['element']] += 1

    # Compute molecular properties
    coords = np.array([[a['x'], a['y'], a['z']] for a in atoms])

    return {
        'mol_name': mol_name,
        'atoms': atoms,
        'bonds': bonds,
        'n_atoms': len(atoms),
        'n_bonds': len(bonds),
        'element_counts': dict(element_counts),
        'formal_charges': formal_charges,
        'coords': coords,
    }


def compute_molecular_center(coords):
    """Compute molecular center."""
    return np.mean(coords, axis=0)


def compute_binding_pocket_residues(protein_ca_coords, ligand_coords, threshold=10.0):
    """Find protein residues within threshold distance of any ligand atom."""
    pocket_residues = []
    ligand_center = np.mean(ligand_coords, axis=0)

    for i, ca_coord in enumerate(protein_ca_coords):
        # Distance from CA to ligand center
        dist_to_center = np.linalg.norm(ca_coord - ligand_center)

        # Also check minimum distance to any ligand atom
        dists = np.sqrt(np.sum((ligand_coords - ca_coord)**2, axis=1))
        min_dist = np.min(dists)

        if min_dist < threshold:
            pocket_residues.append({'idx': i, 'min_dist': min_dist, 'center_dist': dist_to_center})

    return pocket_residues


def compute_secondary_structure_by_phi_psi(protein_data):
    """
    Estimate secondary structure from backbone dihedral angles (simplified).
    Uses CA-based geometry as approximation.
    """
    ca_atoms = protein_data['ca_atoms']
    coords = np.array([[a['x'], a['y'], a['z']] for a in ca_atoms])
    n = len(coords)

    ss_assignments = []

    for i in range(n):
        if i < 2 or i > n - 3:
            ss_assignments.append('C')  # Coil at termini
            continue

        # Compute vectors
        v1 = coords[i] - coords[i-1]
        v2 = coords[i+1] - coords[i]
        v3 = coords[i+2] - coords[i+1]

        # Use local geometry to estimate secondary structure
        # Alpha helix: consecutive CA distances ~3.8A, i to i+4 ~6.0A
        d_i_i2 = np.linalg.norm(coords[i+2] - coords[i]) if i+2 < n else 0
        d_i_i3 = np.linalg.norm(coords[i+3] - coords[i]) if i+3 < n else 0

        # Angle between successive bond vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1, 1)
        angle = np.degrees(np.arccos(cos_angle))

        # Approximate helix: i to i+4 ~ 6A, angle ~ 50-80 degrees
        if d_i_i3 < 7.0 and 40 < angle < 90:
            ss_assignments.append('H')
        elif d_i_i2 > 8.0 and angle < 60:
            ss_assignments.append('E')  # Extended / beta
        else:
            ss_assignments.append('C')  # Coil

    return ss_assignments


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Parse protein
    print("Parsing protein PDB file...")
    protein = parse_pdb(os.path.join(DATA_DIR, '2l3r_protein.pdb'))
    print(f"  Atoms: {protein['n_atoms']}")
    print(f"  Residues: {protein['n_residues']}")
    print(f"  CA atoms: {protein['n_ca']}")
    print(f"  SEQRES sequence length: {len(protein['seqres_sequence'])}")
    print(f"  CA sequence (first 20): {protein['ca_sequence'][:20]}")

    # Parse ligand
    print("\nParsing ligand SDF file...")
    ligand = parse_sdf(os.path.join(DATA_DIR, '2l3r_ligand.sdf'))
    print(f"  Atoms: {ligand['n_atoms']}")
    print(f"  Bonds: {ligand['n_bonds']}")
    print(f"  Elements: {ligand['element_counts']}")

    # Compute protein structural metrics
    ca_coords = get_ca_coords(protein)
    dist_matrix = compute_distance_matrix(ca_coords)
    contact_map = compute_contact_map(dist_matrix)
    rg = compute_radius_of_gyration(ca_coords)
    ete = compute_end_to_end_distance(ca_coords)

    print(f"\nProtein structural metrics:")
    print(f"  Radius of gyration: {rg:.2f} A")
    print(f"  End-to-end distance: {ete:.2f} A")
    print(f"  Contacts (8A threshold): {int(contact_map.sum()//2)}")

    # Ligand metrics
    lig_center = compute_molecular_center(ligand['coords'])
    lig_rg = compute_radius_of_gyration(ligand['coords'])

    print(f"\nLigand structural metrics:")
    print(f"  Center: {lig_center}")
    print(f"  Radius of gyration: {lig_rg:.2f} A")

    # Binding pocket
    pocket = compute_binding_pocket_residues(ca_coords, ligand['coords'])
    print(f"\nBinding pocket residues (within 10A of ligand): {len(pocket)}")

    # Secondary structure
    ss = compute_secondary_structure_by_phi_psi(protein)
    ss_counts = defaultdict(int)
    for s in ss:
        ss_counts[s] += 1
    print(f"\nSecondary structure distribution: {dict(ss_counts)}")

    # Save outputs
    np.save(os.path.join(OUTPUT_DIR, 'ca_coords.npy'), ca_coords)
    np.save(os.path.join(OUTPUT_DIR, 'dist_matrix.npy'), dist_matrix)
    np.save(os.path.join(OUTPUT_DIR, 'contact_map.npy'), contact_map)
    np.save(os.path.join(OUTPUT_DIR, 'ligand_coords.npy'), ligand['coords'])

    # Save summary dict
    import json
    summary = {
        'protein': {
            'n_atoms': protein['n_atoms'],
            'n_residues': protein['n_residues'],
            'n_ca': protein['n_ca'],
            'ca_sequence': protein['ca_sequence'],
            'seqres_sequence': protein['seqres_sequence'],
            'radius_of_gyration': float(rg),
            'end_to_end_distance': float(ete),
            'n_contacts_8A': int(contact_map.sum()//2),
            'secondary_structure': dict(ss_counts),
            'pocket_residues': len(pocket),
        },
        'ligand': {
            'n_atoms': ligand['n_atoms'],
            'n_bonds': ligand['n_bonds'],
            'element_counts': ligand['element_counts'],
            'radius_of_gyration': float(lig_rg),
            'center': lig_center.tolist(),
        }
    }

    with open(os.path.join(OUTPUT_DIR, 'structural_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {OUTPUT_DIR}")
