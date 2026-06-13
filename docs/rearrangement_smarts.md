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
