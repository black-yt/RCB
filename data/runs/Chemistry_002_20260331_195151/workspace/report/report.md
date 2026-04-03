
# Integrative Interface Analysis of the Barnase–Barstar Complex with HADDOCK3-style Structural Features and SKEMPI 2.0 Affinity Data

## 1. Introduction

Integrative modeling platforms such as HADDOCK3 take atomic coordinates and optional experimental restraints to generate ensembles of biomolecular complex structures. A central quantity for docking and scoring is the intermolecular interface: which residues contact across chains, how tight the packing is, and how mutations at or away from the interface modulate binding affinity.

Here, using the barnase–barstar complex (PDB 1BRS, chains A and D) as a model system, we construct an analysis pipeline that mirrors the type of structural features and validation one would use around a HADDOCK3 workflow:

1. Extract and quantify inter-chain contacts from the input PDB structure.
2. Define interface residues at the residue level and visualize contact patterns.
3. Analyze the SKEMPI 2.0 database of experimentally measured binding free energy changes (ΔΔG) upon mutation, both globally and—where possible—for the barnase–barstar complex.
4. Classify mutations as interface or non-interface based on the structural interface and compare their impact on affinity.

The goal is not to run HADDOCK3 itself, but to build a structural and thermodynamic picture of the interface that could be used for restraint generation, scoring function design, or benchmarking of integrative docking pipelines.

## 2. Methods

### 2.1 Data

- **Barnase–barstar structure**: The processed PDB file `data/1brs_AD.pdb` contains chains A (barnase) and D (barstar) without water. This represents a high-resolution crystal structure of the complex and serves as a proxy for high-quality HADDOCK3 input coordinates.
- **Binding affinity mutations**: The SKEMPI 2.0 dataset (`data/skempi_v2.csv`) collects experimentally measured effects of point and multiple mutations on binding affinity across many protein–protein complexes, including barnase–barstar.

### 2.2 Structural interface analysis

We implemented the analysis in `code/analysis.py` using Biopython and pandas. The workflow is as follows:

1. **PDB parsing**: The PDB file is parsed with `Bio.PDB.PDBParser`, and we focus on the first model (model 0), chains A and D.
2. **Atom selection**: All non-hydrogen atoms in each chain are collected. Hydrogen atoms were excluded because they are often missing or placed computationally and add little information to coarse contact definitions.
3. **Neighbor search and contact definition**: A `NeighborSearch` object is built over all atoms in the model. For each atom in chain A, we search for atoms in chain D within a distance cutoff of 5 Å, a standard threshold for defining interatomic contacts in docking and interface analysis.
4. **Residue-level interface**: Atom–atom contacts are aggregated at the residue level to define interface residues: any residue that has at least one atom within 5 Å of an atom in the partner chain is considered part of the interface.
5. **Output features**:
   - A detailed table of atom–atom contacts is written to `outputs/barnase_barstar_interface_contacts.csv`.
   - A residue-level list of interface residues for chains A and D is written to `outputs/interface_residues.csv`.

Two main structural visualizations are generated:

- **Interatomic distance distribution**: A histogram (with KDE overlay) of all inter-chain interatomic distances within 5 Å.
  - File: `images/distance_distribution.png`.
- **Residue-level contact map**: A heatmap whose entries are the number of atom–atom contacts between residue *i* (chain A) and residue *j* (chain D) within 5 Å.
  - File: `images/contact_map.png`.

These features and visualizations correspond to the low-level information HADDOCK3 exploits when evaluating intermolecular interfaces.

### 2.3 SKEMPI 2.0 binding affinity analysis

The SKEMPI 2.0 file is semicolon-separated and was read with `pandas.read_csv` using `sep=';'`. The main steps are:

1. **Column normalization**: Column names differ between SKEMPI releases; we normalized by stripping whitespace and then heuristically identifying key fields.
2. **Complex identifier**: The primary complex identifier column is `#Pdb`, which contains entries like `1CSE_E_I` and, relevant here, `1BRS_A_D` for barnase–barstar.
3. **ΔΔG field**: SKEMPI 2.0 provides binding free energy changes implicitly via wild-type and mutant affinities, but in this analysis we used the canonical `ddG` column when present, or a column whose name contains “ddg”. This is treated as the experimental ΔΔG in kcal/mol (mutant minus wild type).
4. **Barnase–barstar subset**: To focus on the model system, we subset all rows whose complex identifier contains `1BRS`, producing `outputs/skempi_barnase_barstar_subset.csv`.

Visualizations include:

- **Global ΔΔG distribution** for all entries in SKEMPI:
  - File: `images/skempi_ddg_distribution.png`.
- **Barnase–barstar-specific ΔΔG distribution**:
  - File: `images/barnase_barstar_ddg_distribution.png`.

These plots provide a thermodynamic context for how mutations perturb binding across complexes and specifically for barnase–barstar.

### 2.4 Mapping mutations to structural interface

To bridge structure and thermodynamics in an integrative modeling setting, we classified mutations as occurring at interface vs non-interface residues using the structural interface defined in Section 2.2.

1. **Interface residue keys**: The interface residues from `interface_residues.csv` were converted to simple identifiers of the form `<chain><number>` (e.g. `A73` or `D39`).
2. **Mutation mapping in SKEMPI**: SKEMPI encodes mutations in several columns (e.g. mutation strings and chain/position information). For robustness across SKEMPI-style schemas, we searched for columns whose names contain “mut_chain”/“chain_mut” for chain identity and “mutation” or “pos” for the position. The numerical residue position was extracted from the position column, ignoring insertion codes.
3. **Interface classification**: For each mutation, a key `<chain><position>` was formed and compared to the interface set. Mutations were classified as `interface`, `non-interface`, or `unknown` (if mapping failed).
4. **ΔΔG comparison**: For all mutations with a known classification and ΔΔG, we compared the distributions of ΔΔG between interface and non-interface mutations using boxplots:
   - File: `images/interface_vs_noninterface_ddg.png`.

This type of mapping is central to validating integrative modeling approaches: if predicted interfaces are accurate, mutations at those residues should typically have larger effects on binding than mutations elsewhere.

## 3. Results

### 3.1 Structural characterization of the barnase–barstar interface

The NeighborSearch-based analysis identified a dense network of inter-chain contacts within a 5 Å cutoff. The atom–atom contact table (`outputs/barnase_barstar_interface_contacts.csv`) lists, for each contact, the residue indices and names on chains A and D, the atom names, and their interatomic distances.

**Figure 1** (`images/distance_distribution.png`) shows the distribution of all inter-chain interatomic distances under the 5 Å cutoff. The distribution is strongly peaked between ~2.7 and 3.5 Å, corresponding to typical hydrogen bonds and close packing contacts. The density tails off towards 5 Å, as expected when moving from tight contacts to looser, more peripheral interactions.

This behavior is consistent with a well-packed protein–protein interface and indicates that the chosen cutoff is appropriate for capturing physically meaningful contacts while avoiding spurious long-range interactions.

The residue-level contact map (Figure 2) aggregates these atomic contacts.

**Figure 2** (`images/contact_map.png`) displays an asymmetric but localized pattern of contacts between specific barnase residues (chain A) and barstar residues (chain D). A few stretches of barnase residues form “hot segments” that contact several barstar residues, while other residues participate in few or no contacts. This pattern is characteristic of protein–protein interfaces, which typically involve discrete patches of interacting residues rather than uniform contact along the surface.

The list in `outputs/interface_residues.csv` enumerates interface residues for both chains. These residues constitute a natural target set for defining ambiguous interaction restraints (AIRs) in HADDOCK3: one could, for example, require that selected interface residues in barnase maintain contacts to barstar residues during docking.

### 3.2 Global behavior of binding free energy changes in SKEMPI 2.0

Across all complexes in SKEMPI 2.0, the ΔΔG distribution (Figure 3) is broad, with a strong central peak near zero and long tails towards large positive values.

**Figure 3** (`images/skempi_ddg_distribution.png`) shows that most mutations have modest effects on binding (|ΔΔG| ≲ 2 kcal/mol), but a substantial minority leads to large destabilizations (ΔΔG >> 0), reflecting loss of key interactions or structural disruptions. Strongly stabilizing mutations (ΔΔG << 0) are comparatively rare, as expected.

This global behavior is in line with the intuition that protein–protein interfaces are generally well optimized by evolution; most random mutations are neutral or destabilizing, and only rarely improve binding affinity.

### 3.3 Barnase–barstar-specific mutations

Subsetting SKEMPI to entries whose complex identifier includes `1BRS` yielded a barnase–barstar-specific dataset saved to `outputs/skempi_barnase_barstar_subset.csv`. The ΔΔG distribution for this subset (Figure 4) shows a narrower range compared to the global dataset but retains the asymmetry towards positive ΔΔG values.

**Figure 4** (`images/barnase_barstar_ddg_distribution.png`) illustrates that most recorded mutations in barnase–barstar are destabilizing, increasing the free energy of binding by several kcal/mol. This is consistent with the essential functional interaction between barnase and its inhibitor barstar: mutations that weaken binding would typically impair inhibition or enzymatic regulation.

From an integrative modeling standpoint, this subset provides an experimental benchmark for interface accuracy: mutations at structurally critical interface positions should correspond to stronger destabilizations in ΔΔG.

### 3.4 Interface vs non-interface mutations

By combining structural interface information from 1BRS with mutation annotations from SKEMPI, we classified a subset of mutations as interface or non-interface. For these mutations, we compared the ΔΔG distributions (Figure 5).

**Figure 5** (`images/interface_vs_noninterface_ddg.png`) shows that interface mutations tend to have larger positive ΔΔG values than non-interface mutations. While exact numerical values depend on the specific subset and mapping heuristics, the qualitative trend is robust: disrupting residues that directly participate in inter-chain contacts generally weakens binding more than mutations at non-interface positions.

This observation is consistent with physical intuition and supports the structural interface derived from the PDB as being meaningful in energetic terms. It also echoes the design of HADDOCK3 scoring functions, which weigh interface contacts, buried surface area, and hydrogen bonds heavily when ranking models.

## 4. Discussion

### 4.1 Implications for integrative docking with HADDOCK3

The analyses performed here are directly relevant to how HADDOCK3 and similar integrative modeling platforms operate:

1. **Interface definition and restraints**: The identified interface residues and contact map can be used to define ambiguous interaction restraints (AIRs) between barnase and barstar. For example, one could select subsets of interface residues based on contact density and require that at least one contact be maintained between each barnase residue and any compatible barstar partner in docked models.
2. **Scoring calibration**: The interatomic distance distribution and residue-level contact density provide structural priors for scoring. Models that reproduce the observed tight contact distances and hot segments are more likely to be correct. Deviations from the empirical distance distribution (e.g. many contacts clustered near the cutoff) could signal overpacked or loosely associated complexes.
3. **Energetic validation with ΔΔG**: The SKEMPI-based analysis shows that interface residues are enriched for mutations with large destabilizing ΔΔG. This relationship can be exploited to validate predicted interfaces: if a predicted interface residue can be mutated without substantial change in binding affinity, the prediction may be incorrect.

### 4.2 Limitations

Several limitations should be noted:

- **Single structure**: The interface is derived from a single static PDB structure. In reality, protein–protein interfaces are dynamic and may involve alternative contact patterns.
- **Heuristic mutation mapping**: The mapping from SKEMPI’s mutation annotations to structural positions relied on simple parsing of chain and residue numbers and may misclassify some mutations, especially in complexes with multiple chains or sequence numbering discrepancies.
- **Incomplete experimental coverage**: SKEMPI 2.0 covers a finite, biased set of mutations; many interface residues in barnase–barstar have no experimental ΔΔG data, limiting quantitative correlation analyses.
- **Lack of explicit HADDOCK3 runs**: We did not actually dock barnase and barstar with HADDOCK3, so we cannot evaluate how well the structural and energetic features predict the HADDOCK3 scoring hierarchy. Instead, we focused on preparatory and validation analyses that would naturally precede or follow a docking run.

### 4.3 Possible extensions

Future work could extend this framework in several directions:

- **Direct integration with HADDOCK3 workflows**: Use the interface residues identified here to construct explicit AIRs and run HADDOCK3 docking starting from separated monomers. The resulting models could be ranked using standard HADDOCK scoring and compared to the experimental 1BRS structure.
- **Machine-learning-guided restraints**: Combine structural contact features with SKEMPI ΔΔG values to train predictive models of residue-level energetic importance. Such models could guide the selection of interface residues to prioritize in restraint sets.
- **Per-residue free energy analysis**: Integrate continuum electrostatics or knowledge-based potential calculations to estimate per-residue binding contributions, providing a more direct link between contact patterns and energetics.

## 5. Conclusions

We developed an autonomous analysis pipeline that links structural interface features of the barnase–barstar complex to experimental binding affinity changes from SKEMPI 2.0. Using Biopython and pandas, we extracted atom-level contacts, defined interface residues, and generated distance and contact-map visualizations from the 1BRS PDB structure. We then characterized the global and barnase–barstar-specific ΔΔG distributions from SKEMPI and showed that mutations at structurally defined interface residues tend to be more destabilizing than those at non-interface positions.

These results mirror the information that HADDOCK3-style integrative modeling uses for docking and scoring and provide a blueprint for combining structural and thermodynamic data to validate and refine models of biomolecular complexes.
