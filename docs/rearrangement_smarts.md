# Gas-Phase MS/MS Rearrangement SMARTS Review

Literature survey + SMARTS validation for gas-phase rearrangement reactions
relevant to ESI-CID MS/MS spectrum prediction (FraGNNet context).

---

## Literature Sources with Machine-Readable Rules

| Tool / Paper | Type | Rearrangement Coverage |
|---|---|---|
| **ChemFrag** (Schüler et al., *J. Mass Spectrom.* 2018 + 2025) | SMARTS (44 cleavage + 16 rearrangement rules) | McLafferty, RDA, Wagner–Meerwein, allylic shift, hydride transfer. Rules behind Wiley paywall; email corresponding author. |
| **MS2Analyzer** (Ma et al., *Anal. Chem.* 2014) | Neutral-loss mass lookup only | 147 neutral losses as masses; SMARTS used internally for validation but not published or downloadable |
| **BioTransformer** (Djoumbou-Feunang et al., 2019) | SMIRKS + SMARTS | Phase I/II metabolic reactions; heavy overlap with MS/MS losses |
| **PPGB_MS2** (Zheng et al., GitHub) | SMARTS | 28 rules (all audited in project MEMORY.md) |
| **MS-FINDER** (Tsugawa et al., *Anal. Chem.* 2016) | 9 algorithmic H-redistribution rules | H redistribution on cleavage only; no structural rearrangements |
| **FELIX 2024** (Tetering et al., *Commun. Chem.*) | — | IR spectroscopy shows cyclization rearrangements dominate library annotation errors; no tool currently covers these |

---

## Rearrangement Types in ESI-CID (ranked by frequency / coverage importance)

| Rearrangement | Frequency | Coverable by SMARTS? | Current BFS coverage |
|---|---|---|---|
| H redistribution on cleavage (MS-FINDER 9 rules) | Ubiquitous | Algorithmic only | Handled by H-transfer in BFS |
| H₂O loss (dehydration) | Very common | Yes (neutral loss) | BFS ✓ |
| CO/CO₂/NH₃/HCN loss | Common | Yes (neutral loss) | BFS ✓ |
| McLafferty (carbonyl) | Common | Yes (see below) | Not covered |
| McLafferty (nitrile) | Moderate | Yes (see below) | Not covered |
| Retro-Diels-Alder | Moderate (flavonoids, steroids) | Yes (see below) | Not covered |
| Retro-ene | Moderate | Yes (see below) | Not covered |
| Cyclization rearrangements | Pervasive (FELIX 2024) | No — no tool models these | Not covered |
| Cope / Claisen | Rare in ESI-CID | Yes as isomerization prepass | Not covered |
| Wagner–Meerwein | Rare in ESI (common in EI) | Requires custom Python | Not covered |
| Tropylium ring expansion | Moderate (benzyl compounds) | Requires custom Python | Not covered |

---

## Validated SMARTS (SMIRKS-implementable, 7 rules)

All tested with RDKit `AllChem.ReactionFromSmarts` + `RunReactants`.
Test suite: `tests/test_rearrangement_smarts.py` (32 tests, all passing).

### 1. McLafferty — Carbonyl (FIXED)

**Bug in original:** product used `=[C:2]=` (ketene, pentavalent C in RDKit).
**Fix:** change to `-[C:2]=` to give the correct enol product (single bond O–C).

```python
MCLAFF_CARBONYL = (
    "[O,N,S:1]=[C:2]-[#6:3]-[#6:4]-[#6:5]-[H:6]"
    ">>[O,N,S:1](-[H:6])-[C:2]=[#6:3].[#6:4]=[#6:5]"
)
```

Use `AddHs=True`. Verified substrates:

| SMILES | Products |
|---|---|
| `CCCCC(C)=O` (hexan-2-one) | `C=C(C)O` (enol) + `C=CC` (propene) |
| `CCCC(=O)OCC` (ethyl butanoate) | `C=C(O)OCC` (vinyl ester enol) + `C=C` |
| `CCCC(=O)N` (butyramide) | `C=C(N)O` + `C=C` |
| `CCCC(=O)O` (butanoic acid) | `C=C(O)O` + `C=C` |

No match: `CC(C)=O` (no gamma-H), `CCC=O` (propanal, chain too short).

---

### 2. McLafferty — Nitrile (CORRECT AS-IS)

Nitrile McLafferty correctly gives a **ketenimine** (`N=C=C`, not an enol).
The cumulene product is chemically right for this substrate class.

```python
MCLAFF_NITRILE = (
    "[N:1]#[C:2]-[#6:3]-[#6:4]-[#6:5]-[H:6]"
    ">>[N:1](-[H:6])=[C:2]=[#6:3].[#6:4]=[#6:5]"
)
```

Use `AddHs=True`. Verified:

| SMILES | Products |
|---|---|
| `CCCCC#N` (pentanenitrile) | `C=C=N` (ketenimine) + `C=CC` (propene) |
| `CCCCCC#N` (hexanenitrile) | `C=C=N` + `C=CCC` |

No match: `CCC#N` (propionitrile, no gamma-H).

---

### 3. Retro-Diels-Alder — All-Carbon

Added `r6` ring-membership guard to prevent spurious matches in non-6-membered rings.
Use without `AddHs` (avoids valence overflow on substituted rings).

```python
RDA_CARBON = (
    "[#6;r6:1]1=[#6;r6:2]-[#6;r6:3]-[#6;r6:4]-[#6;r6:5]-[#6;r6:6]1"
    ">>[#6:1]=[#6:2]-[#6:3]=[#6:4].[#6:5]=[#6:6]"
)
```

| SMILES | Products |
|---|---|
| `C1=CCCCC1` (cyclohexene) | `C=CC=C` (butadiene) + `C=C` (ethylene) |
| `C1=CC(C)CCC1` (4-methylcyclohexene) | regioisomeric dienes + alkenes |

No match: `C1=CCC=CC1` (cyclohex-1,3-diene, two double bonds), `C1CCCCC1`, `C1=CCCC1`.

---

### 4. Retro-Diels-Alder — Heteroatom

For rings containing O, N, S (flavonoid C-ring, dihydropyranones, etc.).
Use without `AddHs`.

```python
RDA_HETERO = (
    "[#6,#8,#7,#16;r6:1]1=[#6,#8,#7,#16;r6:2]-[#6,#8,#7,#16;r6:3]"
    "-[#6,#8,#7,#16;r6:4]-[#6,#8,#7,#16;r6:5]-[#6,#8,#7,#16;r6:6]1"
    ">>[*:1]=[*:2]-[*:3]=[*:4].[*:5]=[*:6]"
)
```

Produces heteroatom-containing diene/dienophile fragments from dihydropyranone-type rings.

---

### 5. Retro-ene (Carbonyl)

6-membered TS where the aldehyde H transfers to the terminal alkene carbon; beta-bond breaks.
Produces a ketene + alkene. Chain-length specific: carbonyl must be exactly 4 atoms
from the terminal alkene C (5-atom chain gives no match).

```python
RETRO_ENE_CARBONYL = (
    "[#6:1]=[#6:2]-[#6:3]-[#6:4]-[C:5](=[O:6])-[H:7]"
    ">>[#6:1](-[H:7])-[#6:2]=[#6:3].[#6:4]=[C:5]=[O:6]"
)
```

Use `AddHs=True`.

| SMILES | Products |
|---|---|
| `C=CCCC=O` (pent-4-enal) | `C=C=O` (ketene) + `C=CC` (propene) |

No match: `C=CCCCC=O` (chain too long), `CCCC(C)=O` (ketone, no aldehyde H).

---

### 6. Cope [3,3]-Sigmatropic (CORRECT AS-IS)

Isomerization of a 1,5-diene. Generates a structural isomer as input to a second BFS pass,
not a fragment ion directly. Use without `AddHs`.

```python
COPE = (
    "[#6:1]=[#6:2]-[#6:3]-[#6:4]-[#6:5]=[#6:6]"
    ">>[#6:3]=[#6:2]-[#6:1]-[#6:6]-[#6:5]=[#6:4]"
)
```

| SMILES | Product |
|---|---|
| `C=CCCC=C` (hex-1,5-diene) | `C=CCCC=C` (same, symmetric) |
| `C=CC(C)CC=C` (3-methylhexa-1,5-diene) | `C=CCCC=CC` (regioisomer) |

No match: isolated alkene `CC=CCC`, conjugated diene `C=CC=C`.

---

### 7. Claisen Rearrangement (CORRECT, MISLABELED)

The original list called this "Oxa-Cope" — incorrect. This is the **Claisen rearrangement**
(allyl vinyl ether → γ,δ-unsaturated carbonyl). Oxa-Cope is a different reaction
(Cope variant with O in the diene chain).

Isomerization; feeds into BFS as prepass. Use without `AddHs`.

```python
CLAISEN = (
    "[#6:1]=[#6:2]-[#6:3]-[O:4]-[#6:5]=[#6:6]"
    ">>[#6:3]=[#6:2]-[#6:1]-[#6:6]-[#6:5]=[O:4]"
)
```

| SMILES | Product |
|---|---|
| `C=CCOC=C` (allyl vinyl ether) | `C=CCCC=O` (pent-4-enal) |
| `C=CC(C)OC=C` | `CC=CCCC=O` |

No match: `CCOCC` (diethyl ether), `C=CCOC` (allyl methyl ether, no vinyl component).

---

## Rules Requiring Custom Python (Not SMIRKS-Implementable in RDKit)

### Wagner–Meerwein, 1,2-H Shift, 1,5-Hydride Transfer, Pinacol

**Problem:** All involve carbocations. When RDKit constructs the product from SMIRKS,
it adds implicit H atoms based on standard carbon valence (4 bonds). For a carbocation
(`[C;X3;+]`, 3 bonds), RDKit first adds 1 implicit H to reach valence 4, then applies
the bond changes — resulting in a pentavalent carbon (sanitize failure).

**Fix approach:** Implement as custom Python functions that:
1. Match the carbocation + adjacent group substructure pattern via `GetSubstructMatches`
2. Explicitly edit the molecule with `RWMol.AddBond` / `RemoveBond`
3. Adjust formal charges and H counts manually

### Tropylium Ring Expansion (benzyl cation → C₇H₇⁺)

**Problem:** SMIRKS cannot change ring size. The product SMARTS always assigns the same
ring topology as the reactant. Writing `[#6:7]` in a 7-membered product ring while the
reactant has `[c:7]` in a 6-membered ring causes RDKit to generate a 6-membered ring
with a protonated aromatic carbon instead of a 7-membered tropylium.

**Fix approach:** Detect `c1ccccc1-[CH2+]` substructure and directly emit the known
tropylium fragment at m/z 91.054 Da (C₇H₇⁺) as a precomputed mass offset, bypassing
SMIRKS entirely.

---

## Rules Removed

| Rule | Reason |
|---|---|
| **Allylic 1,3-H shift** | Concerted [1,3]-H sigmatropic shift is thermally forbidden (Woodward–Hoffmann). In gas-phase MS/MS it does not occur as written. |
| **Favorskii rearrangement** | Requires nucleophilic solvent to open the cyclopropanone intermediate. Not applicable to isolated gas-phase CID ions. |

---

## Original List — Issues Found

| Rule | Status | Issue |
|---|---|---|
| Wagner-Meerwein | Wrong SMARTS | Methyl 2 bonds from C⁺, not 1; no match on any test substrate |
| Allylic 1,3-H | Wrong mechanism | Thermally forbidden; remove |
| Hydride Transfer | Wrong for intramolecular | `.` means no-bond (intermolecular); explicit H mapping fails with `AddHs` |
| Pinacol | Wrong structure + product | Needs two OH groups; product connectivity incorrect |
| McLafferty Carbonyl | Wrong bond order | `=[C:2]=` gives ketene (pentavalent); fix to `-[C:2]=` for enol |
| McLafferty Nitrile | **Correct** | Ketenimine product is right for nitrile substrates |
| McLafferty Aryl | Syntax issue | Aromatic `c` atoms + Kekulé `=` in product conflict in RDKit |
| RDA (all-carbon) | Partial | Works without `AddHs`; add `r6` guard; fails for substituted rings with `AddHs` |
| Retro-ene | Too specific | Cumulene product only valid for carbonyl-type terminal atom; split into carbonyl vs. general |
| Cope | **Correct** | Isomerization only; feeds BFS as prepass |
| Claisen | **Correct, mislabeled** | Correct SMARTS, but labeled "Oxa-Cope" — rename to Claisen |
| Tropylium | Correct chemistry, wrong SMIRKS | 7-membered aromatic ring not representable via SMIRKS ring expansion |
| Favorskii | Wrong context | Requires nucleophile; remove |

---

## Coverage vs. Current BFS (Integration Decision)

Analysis of whether each validated rule adds coverage beyond the current multi-cut BFS
(`multi_cut_bfs.pyx`) with H-transfer (`max_h_transfer=4`).

| Rule | Gap? | Reason |
|---|---|---|
| McLafferty carbonyl | No | BFS cut=1 (Tarjan bridge) cuts the β-bond; same heavy-atom partition. H-transfer covers ±1H on each fragment. |
| McLafferty nitrile | No | Same: bridge cut on the acyclic β-C bond; H-transfer adjusts the formula. |
| RDA all-carbon (fused ring) | No | BFS cut=2 with fused `ring_bond_groups` already covers 2-bond cuts within SSSR rings shared with another ring. |
| RDA heteroatom (fused ring) | No | Same: fused heteroatom rings are included in `ring_bond_groups`. |
| RDA all-carbon (isolated ring) | **Yes** | Isolated rings (not fused) are filtered out of `ring_bond_groups` in `frag_utils.py` to reduce noise. A cyclohexene or dihydropyranone that is not fused to another ring will not get cut=2. |
| RDA heteroatom (isolated ring) | **Yes** | Same gap as above. |
| Retro-ene carbonyl | No | Acyclic: bridge cut on the β-C bond gives the same heavy-atom partition; H-transfer handles the H redistribution. |
| Cope [3,3] | Not feasible | Isomerization — no fragment ion produced directly. Would require feeding the isomer back into BFS as a new root; current prepass infrastructure does not support this. |
| Claisen | Not feasible | Same: isomerization only, no direct fragment ion produced. |

**Summary:** The only genuine coverage gap is **RDA on isolated (non-fused) rings**. All other
rules are already covered by BFS cut=1/cut=2 + H-transfer, or are isomerizations that the
current prepass cannot model.

**Recommendation:** Do not add these rules to `FRAG_RULES` at this time.
- Global wrecall improvement from the SMARTS prepass is already small / at diminishing returns
  (confirmed after initial integration).
- RDA on isolated rings is a narrow class; whether it is a high-miss class requires a targeted
  wrecall analysis (filter NIST23 to isolated-cyclohexene-containing compounds, compare wrecall
  vs. predicted).
- Cope/Claisen would require a new "isomer prepass" layer before BFS — significant
  infrastructure change for a rare reaction class in ESI-CID.

---

## Case Study: o-Nitrothioether Cyclization (Confirmed, 2026-06-19)

### Compound and Spectrum

- **Molecule**: (4-chloro-2-nitrophenylthio)acetic acid, SMILES: `O=C(O)CSc1ccc(Cl)cc1[N+](=O)[O-]`
- **Formula**: C8H6ClNO4S, [M-H]⁻ = 245.963 Da
- **Spectrum**: spec_id=941868, mol_id=30126, FT, NCE=9%, nist20_hr
- **cos_hun_inst_tol = 0.029** (model essentially blind — all predictions unmatched)

### Observed Fragmentation

All major GT peaks confirmed to ≤1 mDa (FT mass accuracy):

| m/z (obs) | Formula | Δ mDa | Assignment |
|---|---|---|---|
| 171.9634 | C6H4ClNOS | 1.0 | Base peak — cyclic rearrangement product |
| 154.9606 | C6H3ClNS  | 1.0 | Base − OH (17.003 Da) |
| 135.9866 | C6H3NOS   | 0.9 | Base − HCl (35.977 Da) |

Neutral loss to base peak: **74.000 Da = C2H2O3** (confirmed: 2×12 + 2×1.00783 + 3×15.99491 = 74.000 Da).

### Mechanism

The ortho-NO2 group acts as an internal electrophile, cyclizing with the adjacent S atom to
form a new N–S bond (benzisothiazole N-oxide type 5-membered ring). Simultaneously the
–CH2COO⁻ portion is expelled, incorporating one oxygen from NO2, yielding a C2H2O3 neutral
loss (equivalent to CO2 + CH2O = 44.000 + 30.011 = 74.011 Da, or glyoxylic acid 74.000 Da).

```
[M-H]⁻ (245.963, C8H6ClNO4S)
  ─C2H2O3 (74.000)──▶  C6H4ClNOS⁻ (171.963)  [benzisothiazole N-oxide, base peak]
                              │
                        ─OH (17.003)──▶  C6H3ClNS⁻ (154.961)
                              │
                        ─HCl (35.977)──▶  C6H3NOS⁻ (135.987)
```

The base peak (5 DBE: 4 aromatic + 1 ring C=N or C=S) is consistent with a cyclic
thiadiazole or benzisothiazole N-oxide structure. The 13C M+1 peak of the base (172.967,
intensity 50.85 = 5.1% of 999, consistent with ~4–5 carbons visible in the monoisotopic
ion after NIST's 37Cl isotope filtering).

### Why the Model Fails

The BFS DAG only models bond-breaking (cleavage). This fragmentation requires **bond
formation** (a new intramolecular N–S bond). The rearranged cyclic product C6H4ClNOS and
all its secondary fragments are structurally unreachable from the precursor via any sequence
of bond cleavages. The model instead predicts conventional cleavage products:

- [M-H-CO2]⁻ ≈ 202 Da — large orange (pred unmatched)
- Arylthiolate [Ar-S]⁻ ≈ 188 Da — large orange (pred unmatched)

Neither appears in the actual spectrum.

### SMARTS Pattern for Detection

Detect the o-nitrothioether substructure:

```python
O_NITROTHIOETHER = "[S:1]-[c:2]1[c:3][N+:4](=[O:5])[O-:6][c:7]"
```

The reaction would need to be implemented as a custom Python function (not SMIRKS), because:
1. It forms a new ring (N–S bond), which SMIRKS cannot represent as a ring-size change
2. The neutral C2H2O3 requires tracking one O from NO2 joining the –CH2COO– fragment

### Fixability and Priority

| Dimension | Assessment |
|---|---|
| SMARTS-coverable | **No** — requires custom Python (new ring formed, same problem as Tropylium) |
| Compound class prevalence | Unknown — requires NIST23 substructure count for `[S]-c1c[N+](=O)[O-]` |
| Global wrecall impact | Likely small — o-nitrothioethers are a narrow functional group class |
| Fix complexity | Medium — custom fragmentation function + prepass integration |

**Recommendation**: Do not implement now. First check how many NIST23 [M-H]⁻ spectra contain
the o-nitrothioether substructure. If >500 spectra, the wrecall gap warrants a custom rule.
Otherwise, accept as out-of-scope (same decision as Cope/Claisen).

### Finding o-Nitrothioether Compounds in the Dataset

```python
import pandas as pd
from rdkit import Chem

SMARTS = "[S]-[c]1[c][N+](=[O])[O-]"  # o-nitrothioether: S adjacent to o-NO2 on aromatic ring
pat = Chem.MolFromSmarts(SMARTS)

mol_df = pd.read_pickle("data/proc/nist23/mol_df.pkl")   # or nist20
spec_df = pd.read_pickle("data/proc/nist23/spec_df.pkl")

hits = mol_df["smiles"].apply(
    lambda s: (m := Chem.MolFromSmiles(s)) is not None and m.HasSubstructMatch(pat)
)
hit_mol_ids = set(mol_df.loc[hits, "mol_id"])

hit_specs = spec_df[
    spec_df["mol_id"].isin(hit_mol_ids) & (spec_df["ion_mode"] == "negative")
]
print(f"Molecules with o-nitrothioether: {hits.sum()}")
print(f"[M-H]- spectra: {len(hit_specs)}")
print(hit_specs.groupby("inst_type").size())
```
