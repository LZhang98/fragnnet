# Gas-Phase MS/MS Rearrangement SMARTS Review

> **Status:** REFERENCE &nbsp;·&nbsp; **Updated:** 2026-10-12
>
> Literature survey + SMARTS validation for gas-phase rearrangements.

Literature survey + SMARTS validation for gas-phase rearrangement reactions
relevant to ESI-CID MS/MS spectrum prediction (FraGNNet context).

---

## Literature Sources with Machine-Readable Rules

| Tool / Paper | Type | Rearrangement Coverage |
|---|---|---|
| **ChemFrag** (Schüler et al., *J. Mass Spectrom.* 2018 + 2025) | SMARTS-based rules (44 cleavage + 16 rearrangement rules) | [2025 article and supporting Tables S1/S2](https://doi.org/10.1002/jms.5129) describe rules, including steroid methyl migration. Complete machine-readable rule availability remains unverified; see the [current review](../findings/masster/masster_smarts_review_2026_09_13.md#other-rule-collections-to-compare). |
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
| RDA all-carbon (isolated ring) | **Yes** | Isolated rings (not fused) are filtered out of `ring_bond_groups` in `dag/builder.py` (was `frag_utils.py`) to reduce noise. A cyclohexene or dihydropyranone that is not fused to another ring will not get cut=2. |
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

---

## Case Study: Chloral Hemiacetal Amide — Cl Migration + Retro-Addition (Confirmed, 2026-06-20)

### Compound and Spectrum

- **Molecule**: N-(2,2,2-trichloro-1-(4-methoxyphenoxy)ethyl)butanamide, SMILES: `CCCC(=O)NC(Oc1ccc(OC)cc1)C(Cl)(Cl)Cl`
- **Formula**: C13H16Cl3NO3, [M+H]⁺ = 340.027 Da
- **Spectrum**: spec_id=182359, mol_id=9989, FT, NCE=10%, nist20_hr
- **cos_hun_inst_tol = 0.000** (total failure — all 30 model predictions unmatched)

### Observed Fragmentation

Only 2 GT peaks; both confirmed to ≤1 mDa; neither is reachable by BFS:

| m/z (obs) | Formula | Δ mDa | NL from [M+H]⁺ | Assignment |
|---|---|---|---|---|
| 121.0285 | C4H8NOCl | 0.9 | 218.998 Da | Base — Cl migration from CCl3 to acyl chain |
| 265.9543 | C9H7NO2Cl3 | 0.1 | 74.073 Da (C4H10O) | Retro-addition, loss of n-butanol equivalent |

### Mechanisms

**Fragment at 265.954 — retro-addition (NL = C4H10O = 74.073 Da):**

The compound is a chloral hemiacetal amide: `Bu-CO-NH-CH(O-Ar)-CCl3`. Under CID [M+H]⁺,
the N-H proton mobilises and the molecule undergoes a retro-addition that rebuilds the
butanoyl N-H into a C4H10O neutral (incorporating both H atoms from C–H and N–H into the
departing piece). The remaining fragment C9H7NO2Cl3⁺ (DBE=6 as cation: 4 aromatic + C=N
+ C=O) contains the Ar-O-CH-N portion fused into a cyclic iminium or oxazoline-type cation
with the intact CCl3 group:

```
[M+H]⁺ (340.027) ──C4H10O (74.073)──▶  C9H7NO2Cl3⁺ (265.954)  [Ar-O-C=N-CCl3 type]
```

**Fragment at 121.029 — 1,3-Cl migration (base peak):**

One Cl migrates from the CCl3 group (across the C–N bond to the acyl chain α-carbon),
while the C–N bond cleaves. Products: C4H8NOCl⁺ (chlorinated butanamide-type cation,
DBE=1) + neutral C9H9Cl2O2 (the Ar-O-CH=CCl2 portion, 2 Cl retained):

```
CCl3–CH(OAr)–NH–CO–nBu  →  [nBu–CO–NH–CH2Cl]⁺  +  neutral(ArO–CCl2)
      [M+H]⁺ = 340.027        m/z = 121.029                 = 218.998 Da
```

Both mechanisms require **new bond formation** (Cl–C migration creates a new C–Cl bond;
the retro-addition forms a C=N bond while breaking C–N): structurally outside the BFS DAG.

### Why the Model Fails Completely

The BFS generates 30 standard cleavage products (4-methoxyphenol [M+H]⁺ = 126.07,
butanamide [M+H]⁺ = 88.08, CCl3-containing direct cuts at various masses). None fall
within 10 ppm (≤1.2 mDa at 121, ≤2.7 mDa at 266) of either GT peak. cos_hun = 0.000.

### SMARTS Fixability

**Neutral loss rule (partial fix, implementable):**

The C4H10O (74.073 Da) neutral loss can be encoded as a mass-based rule conditioned on
the substructure pattern:

```python
CCL3_HEMIACETAL_AMIDE = (
    "[C:1](Cl)(Cl)Cl"      # CCl3 group
    "-[C:2](-[O:3][c])"    # alpha-C with aryloxy
    "-[N:4]"               # amide N
    "-[C:5](=[O:6])"       # amide carbonyl
    "-[#6]-[#6]-[#6]"      # at least 3-carbon chain (n-propyl or longer)
)
# If matched: add [M+H - 74.073]+ as candidate fragment
```

This does not require expressing the ring closure — it directly encodes the observed neutral
loss mass and the structural trigger. Implementable as a `NeutralLossRule` in the prepass
alongside the existing SMARTS rules.

**Cl-migration fragment (not implementable as SMARTS):**

The 1,3-Cl shift (CCl3 → acyl chain) creates a new C–Cl bond. Same limitation as
o-nitrothioether (new ring) and tropylium (ring expansion): cannot be expressed in SMIRKS.
Would require custom Python that:
1. Matches `[C](Cl)(Cl)Cl-C-N-C(=O)-[CCC]`
2. Redistributes one Cl to the alpha-C of the acyl group
3. Emits the Cl-migrated acyliminium cation formula directly

### Fixability and Priority

| Dimension | Assessment |
|---|---|
| Neutral loss (74 Da) via SMARTS | **Implementable** — `NeutralLossRule` conditioned on CCl3 hemiacetal amide pattern |
| Cl-migration fragment | **Not via SMARTS** — requires custom Python |
| Compound class prevalence | Unknown — `[C](Cl)(Cl)Cl-C-N` is an unusual substructure (chloral derivatives) |
| Global wrecall impact | Very likely small — chloral amides are rare in biological/pharmaceutical context |

**Recommendation**: Do not implement. Check NIST23 prevalence of `[CX4](Cl)(Cl)Cl-[CX4]-N`
first. If <200 spectra, accept as out-of-scope. The neutral loss rule is easy but useless
without the companion Cl-migration fragment rule, and the latter requires custom Python for
marginal gain.

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

---

## Case Study: 3-Hydroxythiophene Enol Dehydration — Model Ranking Failure (Confirmed, 2026-06-20)

### Compound and Spectrum

- **Molecule**: Methyl 4-hydroxy-5-(trifluoromethyl)thiophene-2-carboxylate, SMILES: `COC(=O)c1sc(C(F)(F)F)cc1O`
- **Formula**: C7H5F3O3S, [M+H]⁺ = 226.998 Da
- **Spectrum**: spec_id=474127, mol_id=16901, FT, NCE=35%, nist20_hr
- **cos_hun_inst_tol = 0.000** (all 31 predictions unmatched)

### Observed Fragmentation

| m/z (obs) | Formula | NL | Assignment |
|---|---|---|---|
| 208.9875 | C7H4F3O2S | 18.011 (H2O) | Base peak — enol-OH dehydration |
| 121.0281 | C7H5O2 | 105.969 | Minor — full desulfurization (mechanism unclear) |

### Why the Model Fails

This is **not a BFS coverage gap**. The fragment C7H4F3O2S at 208.9884 Da (0.9 mDa from GT) **is
in the BFS DAG** — generated by cutting the C3–OH bond with delta_h=0.  Confirmed via
`formula_peak_mzs` for mol_id=16901 in the precomputed DAG.

The model scores 102 formula candidates and outputs 31 predictions. Its actual top predictions:

| Rank | m/z | NL | Chemical interpretation |
|---|---|---|---|
| #1 | 194.972 | 32.026 (CH₃OH) | Methyl ester methanol loss |
| #3 | 206.992 | 20.006 (HF) | CF₃ → CF₂ + HF |
| #6 | 183.009 | 43.990 (CO₂) | CO₂ loss |

The model applies correct general rules (methyl ester → CH₃OH loss; CF₃ → HF loss) but fails
for this scaffold because the dominant pathway is **enol-OH dehydration**: the 3-OH is a vinyl/
enol hydroxyl stabilized by adjacent ring sulfur and C2 ester resonance. Dehydration is
energetically preferred over CH₃OH or HF loss for this specific 3-hydroxythiophene-2-ester core.

### Failure Mode: Model Intensity Ranking Failure

This is a priority inversion between competing pathways — the same failure category as
the glycoside charge-partitioning case. The fragment exists in the DAG; the model just
doesn't rank it above the competing ester and CF₃ fragments.

**Root cause**: no atom/node feature distinguishes enol-type OH (directly on 5-membered
heteroaromatic ring → dehydration dominant) from aliphatic OH or phenol OH. The model's
`n_hydroxyl` feature counts OH groups but encodes nothing about the OH chemical context.

### SMARTS Fixability

Not a SMARTS/prepass problem — the fragment is already in the DAG. **Fix requires a model
feature change**.

#### Why `n_hydroxyl` / `charge_retention` (the glycoside fix) does NOT help here

The `charge_retention` feature (`[n_hydroxyl, n_acidic, ion_mode_sign]`) solves charge
partitioning: "which fragment keeps the proton/deprotonation site?" For glycosides, the
sugar fragment has high `n_hydroxyl` → model learns it wins `[M-H]⁻`. That is a
charge-affinity problem.

For spec_id=474127 the problem is different — it is a **bond-breaking priority problem** between
three competing pathways, all in positive mode, all simple bond cuts:

| Competing pathway | Fragment `n_hydroxyl` |
|---|---|
| CH₃OH loss → 194.972 (model #1) | 0 — OH leaves with the methanol |
| HF loss → 206.992 (model #3) | 0 — no OH in the CF₂ product |
| **H₂O loss → 208.988 (GT base)** | **0** — OH leaves as the water |

All three product fragments have `n_hydroxyl = 0`. The feature is identical for all three
competing nodes, so it cannot change the ranking.

#### What would actually help

The signal needed is on the **type of the OH being lost**, not on the fragment that
remains. An `is_enol_oh` atom-level feature on the OH oxygen in the parent molecule
encodes: "this OH is directly bonded to an aromatic ring atom in a 5-membered heteroaromatic
(`[OX2H]-[c;r5]`)." When BFS cuts the C–OH bond, the model's node for the dehydrated
product would inherit a signal that the lost OH was enol-type → dehydration path priority.

Implementation: add `is_enol_oh` (0/1) as an atom feature in `extract_mol_info`, propagate
it to fragment nodes as "did this fragment lose an enol OH?" via a complement-side feature
(analogous to the existing `h_range` complement edge feature). Alternatively, a 3-class
`oh_type` {0: none, 1: aliphatic OH, 2: enol/aromatic OH} on each fragment's composition
would carry the same signal without requiring complement-side bookkeeping.

### Fixability and Priority

| Dimension | Assessment |
|---|---|
| BFS coverage | Already covered — fragment in DAG at 0.9 mDa |
| `n_hydroxyl` / `charge_retention` feature | Does **not** help — all competing products have n_hydroxyl=0 |
| `is_enol_oh` feature | Would help — distinguishes enol-OH dehydration from CH₃OH/HF loss |
| Scope of `is_enol_oh` | Generalizes to 3-hydroxypyridines, 3-hydroxyfurans, hydroxyl-quinolines |
| Training data | Likely few 3-hydroxythiophene examples in NIST; feature fix more tractable than data |
| Global wrecall impact | Unknown — check NIST23 prevalence of `[OX2H]-[c;r5]` substructure |

**Recommendation**: Low priority as a standalone fix. Add `is_enol_oh` / `oh_type` to the
feature audit list (`docs/reference/feature_set_audit.md`) for the next feature engineering pass. Do
not confuse with the `charge_retention` feature (glycoside fix) — they solve orthogonal
problems.

---

## Design Principle: GNN Learning vs. Explicit Feature Encoding

### Can the molecular GNN learn functional-group dominance rules automatically?

In principle yes — a 3-layer GINE with standard atom features can express any local subgraph
pattern (sulfonamide, nitrile, nitro, etc.) within Weisfeiler-Lehman expressivity. Given
enough training examples the model will learn "aromatic sulfonamide → SO₂ loss dominant."

**The real bottleneck is not GNN expressivity — it is architecture pooling:**

```
mol_embedder(molecule) ──mean pool──▶  one vector  (global)
                                              │
frag_embedder(DAG nodes + global)  ──▶  per-node scores
```

The molecular GNN output is pooled to a single vector before it reaches the fragment scorer.
Even if the mol GNN perfectly detects "this molecule has enol-OH," that signal is diluted
into the global embedding. The frag GNN must then learn TWO implicit steps:
1. Decode from the mol embedding that enol-OH is present.
2. Associate that signal with the specific fragment node that LOST the enol-OH.

Explicit per-fragment features (e.g. `n_enol_oh` = 0 on the dehydrated fragment, 1 on all
others) bypass both steps — they give a direct, per-node signal with no pooling dilution.

### Practical threshold: training-set size per functional group class

| Functional group | Est. NIST23 spectra | GNN learns? | Explicit feature needed? |
|---|---|---|---|
| Aromatic sulfonamide | ~8,000+ | Yes | No — GNN has sufficient signal |
| Aromatic nitrile | ~5,000+ | Yes | No |
| Aromatic nitro | ~3,000+ | Probably | Borderline |
| Ortho-OH + C=O (flavonoids) | ~1,500+ | Borderline | Borderline |
| Phosphate ester (HPO₃ vs H₃PO₄) | ~2,000+ | Borderline | Yes — mono vs di-ester distinction subtle |
| **Enol-OH on 5-membered heteroaromatic** | **<300** | **No** | **Yes** |

### Decision rule

- **>3,000 training spectra with that FG**: GNN learns it. Check if model already performs
  well on that compound class before adding a feature. If broken, the fix is more data or a
  deeper mol GNN, not an explicit feature.
- **300–3,000**: Borderline. Explicit feature gives ~10× better sample efficiency; add if
  the failure mode appears repeatedly in worst-case analysis.
- **<300**: GNN cannot reliably learn it. Explicit feature is necessary.

### Which features to actually add

Based on the above, add at most two new explicit scalar features beyond `n_hydroxyl` /
`n_acidic`:

1. **`n_enol_oh`** — `[OX2H]-[c;r5]`: enol OH on 5-membered aromatic ring.
   Needed because the compound class is rare (<300 spectra). Resolves 3-hydroxythiophene
   priority inversion (and related: 3-hydroxyfuran, 3-hydroxypyrrole).

2. **`n_phosphate_mono`** — `[OX2H]-[PX4](=O)(-[OX2])-[OX2]`: phosphate monoester OH.
   Needed because HPO₃ (79.966) vs H₃PO₄ (97.977) loss depends on whether it is a mono-
   or di-ester, a distinction the GNN may conflate from the pooled embedding.

**Do not add**: `n_sulfonamide`, `n_aryl_cn`, `n_aryl_no2`. The GNN learns these from
NIST23-scale data. Verify the model actually fails on these compound classes before
engineering features — premature feature engineering for learnable patterns adds noise.

**Important caveat — BFS coverage prerequisite**: The GNN can only learn to rank fragment
intensities for fragments that are ALREADY IN THE BFS DAG. If a dominant fragmentation
pathway requires new bond formation (e.g., aromatic sulfonamide SO₂ extrusion → new Ar−N
bond), the product fragment is completely absent from the DAG. In that case the GNN cannot
learn the pathway regardless of training set size, and a SMARTS prepass rule is required to
add the product to the DAG FIRST.

Revised decision rule:

1. Does the pathway require new bond formation? → Always needs a prepass rule, GNN cannot help.
2. Is the product already in the BFS DAG? → Check if model fails. If yes, is it data size or a missing feature?
   - Large training set (>3,000) + model still fails → likely a DAG feature problem; add explicit feature.
   - Small training set (<300) → explicit feature needed for sample efficiency.

---

## Case Study: Lignan Glucoside [M+Na]⁺ — Systematic Fragment Mass Calibration Artifact (2026-06-20)

### Compound and Spectrum

- **Molecule**: Podophyllotoxin-type lignan glucoside, SMILES: `COc1cc(C2c3cc4c(cc3C(OC3OC(CO)C(O)C(O)C3O)C3COC(=O)C23)OCO4)cc(OC)c1OC`
- **Formula**: C28H32O13, [M+Na]⁺ = 599.174 Da
- **Spectrum**: spec_id=533415, mol_id=18136, FT/CID, NCE=35%, nist20_hr
- **cos_hun_inst_tol = 0.000** (all predictions unmatched)

### Observed Fragmentation and Model Predictions

| GT m/z | GT int | Model m/z | Model int | Δ (mDa) | Assignment |
|---|---|---|---|---|---|
| 419.1285 | 999 | 419.1107 | 1000 | +17.8 | [aglycone+Na]⁺ (NL=glucose) |
| 437.1017 | 7.6 | 437.1207 | 98 | −19.0 | [aglycone+Na]⁺ (NL=dehydrated glucose) |
| 437.1403 | 7.6 | 437.1207 | 98 | +19.6 | — (model between the two GT peaks) |
| 375.1053 | 4.3 | 375.1203 | 51 | −15.0 | deeper aglycone fragment |
| 375.1342 | 4.6 | 375.1203 | 51 | +13.9 | — |

**FT tolerance at m/z 420**: 10 ppm × 420 = 4.2 mDa. All GT peaks are 15–20 mDa from the
nearest model prediction. cos_hun = 0.000.

### Failure Mode: MS2 Fragment Mass Calibration Artifact

The precursor is accurately measured: [M+Na]⁺ theory = 599.174, measured = 599.174 (Δ = 0.6 mDa).
But ALL MS2 fragment ions are systematically offset from theoretical Na-adduct fragment masses.

Most striking: the 437.1017 / 437.1403 pair is **±19 mDa symmetric around the model's
theoretical 437.121**. This symmetry rules out a chemistry explanation — no formula shift
produces a ±19 mDa symmetric pair. The model is correct; the GT spectrum has a calibration
issue in the MS2 scan.

The paired GT peaks at each nominal mass (419+, 437+, 375+) also suggest the spectrum may
contain co-eluting structural isomers or two fragmentation pathways, but neither interpretation
fixes the systematic offset from theory.

**Note**: precursor (MS1) calibration is fine; the offset is in the MS2 fragment scan only.
This is consistent with instruments that calibrate MS1 and MS2 scans independently.

### Fixability

None. The model generates correct theoretical [fragment+Na]⁺ masses. The GT spectrum
calibration cannot be corrected post-hoc from within the model. This is a NIST20 data
quality artifact. Flag class: `nist20_hr` CID spectra with FT tolerance failures despite
correct formula.

---

## Case Study: Triterpene Glucuronide [M+H]⁺ — Calibration Artifact + Ranking Failure (2026-06-20)

### Compound and Spectrum

- **Molecule**: Ursolic-type triterpene glucuronide ester, SMILES: `CC(CCC(=O)OC1OC(C(=O)O)C(O)C(O)C1O)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CCC12C`
- **Formula**: C30H48O10, [M+H]⁺ = 569.332 Da
- **Spectrum**: spec_id=296336, mol_id=5307, FT/CID, NCE=35%, nist20_hr
- **cos_hun_inst_tol = 0.000** (all predictions unmatched)

### Observed Fragmentation vs. Model Predictions

GT base peak is a cascade of H₂O losses from the intact precursor:

| GT m/z | GT int | Model m/z | Model int | Δ (mDa) | Assignment |
|---|---|---|---|---|---|
| 551.3406 | 999 | 551.3215 | 425 | +19.1 | [M+H−H₂O]⁺ |
| 533.3302 | 229 | 533.3109 | 125 | +19.3 | [M+H−2H₂O]⁺ |
| 515.3218 | 46 | 515.3003 | 78 | +21.5 | [M+H−3H₂O]⁺ |
| 411.1983 | 329 | — | — | — | Unassigned (NL=158.134 ≈ C₉H₁₈O₂?) |
| 409.182 | 28 | — | — | — | Minor unassigned |

Model top predictions (not in GT):

| Model m/z | int | NL | Assignment |
|---|---|---|---|
| 393.300 | 1000 | 176.032 (C₆H₈O₆) | [aglycone+H]⁺ via glucuronolactone loss |
| 357.279 | 831 | 212.053 (glucuronolactone+H₂O) | secondary |
| 375.289 | 643 | 194.043 (glucuronic acid) | [aglycone+H]⁺ direct |

### Two Overlapping Failure Modes

**1. Systematic +20 mDa calibration artifact (same as spec_id=533415)**

The H₂O-loss series (551, 533, 515) is correctly identified by the model in formula, but the
GT masses are consistently +19–22 mDa above theoretical. FT tolerance is 5.5 mDa at these
masses. This is the same nist20_hr MS2 calibration artifact as the lignan glucoside case
above — adduct type ([M+H]⁺ vs [M+Na]⁺) does not matter; the offset is instrument-level.

**2. Glucuronolactone loss over-ranked vs. H₂O loss cascade**

This molecule is a **glucuronic acid ESTER** (the glucuronic acid is attached via an ester
bond at the triterpenoid C-28 carboxylic acid). For ester glucuronides, the model predicts
glucuronolactone neutral loss (176.032 Da) as the dominant pathway, ranked #1 at intensity
1000. But for this large, lipophilic triterpenoid, the GT base peak is consecutive H₂O loss
from the intact [M+H]⁺ — the glucuronic ester bond is retained.

The model correctly knows about the glucuronic ester in the BFS DAG. The failure is that it
over-weighs glycosidic cleavage relative to OH-dehydration for large hydrophobic terpenoids
where multiple OH groups drive repeated dehydration.

Additionally, GT peak 411.198 (2nd highest, NL=158.134) has no clean formula assignment
consistent with the molecular formula — possibly a retro-ene or side-chain loss unique to the
triterpenoid scaffold, outside BFS coverage.

### Fixability

| Dimension | Assessment |
|---|---|
| Calibration artifact | Not fixable in model — data quality issue |
| H₂O loss ranking for large terpenoids | Potentially learnable from training data (if sufficient large terpenoid glucuronides); no obvious explicit feature fix |
| GT 411.198 peak | Possible retro-ene — outside BFS coverage, low priority |

The calibration artifact alone is sufficient to explain cos_hun = 0.000. Even if the ranking
were perfect, the +20 mDa offset would prevent any match within FT tolerance.

---

## Case Study: Complex Glycoconjugate — Na-Adducted Fragment Ions in [M+H]⁺ Spectrum (2026-06-20)

### Compound and Spectrum

- **Molecule**: NeuAc-Hex-HexNAc-Rha glycoconjugate with NHAc-aldehyde aglycone, SMILES: `CC(=O)NC(C=O)C(OC1OC(CO)C(O)C(OC2(C(=O)O)CC(O)C(NC(C)=O)C(C(O)C(O)CO)O2)C1O)C(OC1OC(C)C(O)C(O)C1O)C(O)CO`
- **Formula**: C31H52N2O23, [M+H]⁺ = 821.303 Da
- **Spectrum**: spec_id=350043, mol_id=2032, FT/CID, NCE=30%, nist20_hr
- **cos_hun_inst_tol = 0.000** (n_pred=121, all predictions unmatched)

### Model Predictions vs. GT

Model generates standard glycan **B-type oxonium ions** (correct chemistry):

| Model m/z | int | Assignment |
|---|---|---|
| 204.087 | 1000 | [HexNAc−H₂O+H]⁺ (B₁) |
| 186.076 | 876 | [HexNAc−2H₂O+H]⁺ |
| 168.066 | 401 | [HexNAc−3H₂O+H]⁺ |
| 292.103 | 125 | [NeuAc−H₂O+H]⁺ (B₁) |
| 274.092 | 218 | [NeuAc−2H₂O+H]⁺ |
| 350.145 | 187 | [Rha+HexNAc−H₂O+H]⁺ (B₂) |

GT peaks:

| GT m/z | int | Assignment |
|---|---|---|
| 226.067 | 20 | [HexNAc−H₂O+**Na**]⁺ → model 204.087 + (Na−H) = 226.069, Δ=**1.9 mDa** ✓ |
| 336.065 | 999 | Base peak — unassigned (no standard glycan fragment fits) |
| 354.077 | 182 | 336.065 + H₂O (dehydration pair) |
| 352.032 | 91 | 336.065 + ~16 Da (O-shifted variant?) |
| 370.043 | 27 | 336.065 + ~34 Da |
| 328.099 | 26 | Unassigned |
| 508.168 | 15 | Unassigned; NL = 313.136 Da |
| 132.987 | 87 | **Anomalous**: no C/H/N/O formula within 5 mDa. Likely contaminant. |

### Failure Mode: Na-Adducted Glycan Fragments

The GT 226.067 is the **Na-adducted HexNAc B₁ oxonium** ion, formed when residual Na⁺ in the
ion source adducts onto the [HexNAc−H₂O]⁺ cation:

```
[HexNAc−H₂O + H]⁺  (model: 204.087)
  → replace H with Na →
[HexNAc−H₂O + Na]⁺  (GT: 226.067, theory: 226.069, Δ=1.9 mDa)
```

This is a known phenomenon in glycan ESI-MS/MS: even from a [M+H]⁺ precursor, glycan
fragment ions can appear as Na⁺ adducts when Na contamination is present. FraGNNet does not
model inter-element fragment adducts — it only computes [frag+H]⁺ or [frag−H]⁻ products.

The GT base peak cluster at 336−370 Da remains unassigned. The NL of 485.238 Da does not
correspond to any combination of the constituent sugar residues (NeuAc=291.095, Hex=162.053,
HexNAc=203.079, Rha=146.057). The 132.987 peak has no valid C/H/N/O formula — likely an
instrument background or co-eluting impurity.

### Summary

| Failure source | Impact |
|---|---|
| Na-adducted fragment ions (model has no Na-fragment prediction) | Explains 226.067 miss |
| Unidentified base peak cluster 336−370 (possibly Y-type or cross-ring ions + Na) | Explains cos_hun=0.000 |
| Anomalous 132.987 peak (probable contaminant) | Inflates apparent GT complexity |

**Systematic gap**: FraGNNet cannot predict [frag+Na]⁺ ions from [M+H]⁺ precursor spectra.
This affects all heavily Na-contaminated glycan spectra in NIST20. Fix requires generating
both H and Na adducts for all fragment ions — a significant change to the fragment scoring
infrastructure. Low priority: Na-contamination is common in metabolomics data but NIST20 spectra
are typically low-Na; this compound is an outlier.

**Note**: spec_id=350044 (same compound mol_id=2032, NCE=35%) and spec_id=350046 (same compound,
NCE=35%, n_pred=246) share the same failure modes and are omitted as duplicates.

---

## Case Study: N-Nitroimidazolidine — N₂O Neutral Loss Rearrangement (Confirmed, 2026-06-20)

### Compound and Spectrum

- **Molecule**: 2-nitroimino-imidazolidine (N-nitroimidazolidine-2-imine), SMILES: `O=[N+]([O-])NC1=NCCN1`
- **Formula**: C3H6N4O2, [M−H]⁻ = 129.042 Da
- **Spectrum**: spec_id=550113, mol_id=32618, FT/CID, NCE=35%, nist20_hr
- **cos_hun_inst_tol = 0.000**

### Observed Fragmentation

| m/z (obs) | Δ mDa | NL from [M−H]⁻ | Assignment |
|---|---|---|---|
| 85.041 | 0.1 | **44.001 = N₂O** (44.001 Da) | Base peak — N₂O neutral loss |
| 86.044 | 1.0 | — | ¹³C isotope of 85.041 (+1.003 Da) |
| 85.044 | 2.7 | — | Near-coincident; likely satellite |
| 131.046 | — | (−2.004, heavier than precursor) | Impurity: [C4H8N2O3−H]⁻ = 131.046 |

N₂O mass = 2×14.003 + 15.995 = **44.001 Da** (exact). Confirmed to 0.1 mDa.

### Mechanism

The compound is a **N-nitroimidazolidine-2-imine**: the nitro group is attached directly to
an amidine N (N−NO₂ nitramine). Under CID [M−H]⁻, the N-N bond undergoes:

```
          O    O
          ‖    ‖
[M−H]⁻ = N − N − NH − C(=N) − ring
          ↓
    N₂O expelled  +  [ring-C=N]⁻ fragment

[C3H5N4O2]⁻  →  [C3H5N2O]⁻ (85.040)  +  N₂O (44.001)
```

The N₂O (nitrous oxide) is formed by recombination of the exocyclic N from −NH− and the N
from −NO₂, with one O from the nitro group. This requires a **bond order change** in the N−N
bond (from single → N=N=O triple in N₂O), which makes it a gas-phase rearrangement —
**not a simple BFS bond cleavage**.

### Why the Model Fails

Model's top prediction at m/z 98.036 (NL = 31.006 = HNO loss) and its nearest fragment to GT:
- m/z 85.028 (formula C2H3N3O⁻, NL = 44.014) → **13 mDa below GT** (outside 2 mDa floor)

The model generates BFS fragments via single/double bond cuts. It cannot produce the N₂O
neutral loss because N₂O formation requires both N atoms from different positions of the
molecule to combine. The formula difference: model predicts C2H3N3O (loses C1H3N1O1 = CH3NO)
vs GT C3H5N2O (loses N₂O).

### SMARTS Fixability

**N₂O loss is implementable as a conditional neutral loss rule:**

```python
N_NITROAMIDINE_PATTERN = (
    "[N+](=O)([O-])-[NH]-[C;!H0]=[N]"  # N-nitroamidine substructure
)
# OR broader nitramine:
N_NITRAMINE_PATTERN = "[N+](=O)([O-])-[NH]-"  # any N-nitroamine NH

# If matched in [M−H]⁻ mode: add [M−H − 44.001]⁻ as candidate fragment
# NL = N₂O = 44.00106 Da
```

This does NOT require expressing the bond order change — it directly encodes the known
neutral loss mass conditioned on the N-nitroamine substructure.

### Compound Class and Prevalence

N-nitroamines / nitraimines that show N₂O loss:
- **Energetic materials**: RDX (hexahydro-1,3,5-trinitro-1,3,5-triazine), HMX, CL-20
- **Pharmaceuticals**: nitroimidazoles with N-substituents (ronidazole, metronidazole derivatives)
- **Agrochemicals**: some nitramine fungicides

In NIST20/23, energetic material spectra are rare but pharmaceutical nitroimidazoles are
more common (~50–200 spectra estimated). Check NIST23 prevalence of `[N+](=O)([O-])-N`
before implementing.

### Fixability and Priority

| Dimension | Assessment |
|---|---|
| BFS coverage | **Gap** — N₂O loss requires bond order change, not representable |
| SMARTS-fixable as NL rule | **Yes** — `NeutralLossRule` conditioned on N-nitramine substructure |
| Compound prevalence | Moderate (energetic materials + some pharmaceuticals) |
| Implementation complexity | Low — same pattern as existing neutral loss rules |

**Note (spec_id=572128)**: Isoflavone-7-O-glucoside [M+Na]⁺ (mol_id=18632, dset_spec_id=1915489,
NCE=35%) shows identical failure — systematic +16.5 mDa GT fragment offset with paired GT peaks
at every nominal mass. Same contributor batch as spec_id=533415 (dset_spec_id=1869751). Brief
summary: same [M+Na]⁺ calibration artifact, model prediction at 383.073 ([aglycone+Na]⁺, NL=162.053)
vs GT at 383.090 (Δ=+17 mDa). Not documented separately.

**Recommendation**: Implement as a `NeutralLossRule` with SMARTS trigger `[N+](=O)([O-])-[NH]-`.
Check NIST23 prevalence first. Unlike the previous cases (Cl migration, o-nitrothioether),
this reaction IS fully encodeable as a conditional mass-based rule without custom Python.

---

## Case Study: Aromatic Sulfonamide — SO₂ Extrusion BFS Coverage Gap (Confirmed, 2026-06-20)

### Compound and Spectrum

- **Molecule**: N-(pyrimidin-2-yl)-4-nitrobenzenesulfonamide, SMILES: `O=[N+]([O-])c1ccc(S(=O)(=O)Nc2ncccn2)cc1`
- **Formula**: C10H8N4O4S, [M−H]⁻ = 279.019 Da
- **Spectrum**: spec_id=610784, mol_id=32771, FT/CID, NCE=35%, nist20_hr
- **cos_hun_inst_tol = 0.000** (n_pred=115, single GT peak, zero matched)

### Observed Fragmentation

GT has exactly **one peak**:

| m/z (obs) | Δ mDa | NL | Assignment |
|---|---|---|---|
| 215.057 | 0.1 | **63.962 = SO₂** | Base peak (999) — SO₂ extrusion |

NL = 63.962 = S + 2O = 31.972 + 2×15.995 = 63.962 Da (0.1 mDa from theoretical).

Fragment = [4-NO₂-C₆H₄-NH-Pyr − H]⁻ = [C10H7N4O2]⁻ = 215.057 (2 mDa from theory 215.055).

### Mechanism: SO₂ Extrusion with New Ar−N Bond Formation

Classic gas-phase reaction of aromatic sulfonamides in [M−H]⁻ mode:

```
Ar−SO₂−NH−Het  →  Ar−NH−Het  +  SO₂  (NL = 63.962 Da)
     [M−H]⁻              [frag]⁻         neutral
```

The product fragment [Ar−NH−Het]⁻ contains a NEW Ar−N bond that does not exist in the
parent molecule (original connectivity: Ar−S−N−Het). This requires **simultaneous S−Ar and
S−N bond cleavage with Ar−N bond formation** — a concerted 4-membered cyclic transition state.

Because a NEW bond forms, this is a **rearrangement** and is **completely outside BFS DAG
coverage**. The product [4-NO₂-phenyl-NH-pyrimidine]⁻ has zero representation in the BFS DAG.

### Why the Model Fails Completely

Model top prediction: **[M−H]⁻ at m/z 279.019 (NL=0.000, int=1000)** — the intact precursor
ion, not any fragment. This indicates the model finds no high-probability fragment in the DAG
and defaults to "no fragmentation" (via the out-of-scope prediction node).

The model then predicts [NO₂-phenyl]⁻ (122.025, NL=157), [M−H−NO]⁻ (249.021, NL=30), etc.
— none of these appear in the GT. The correct fragment (215.057) is absent from the DAG
entirely.

**Key distinction from spec_id=474127 (3-hydroxythiophene)**: there the fragment WAS in the
DAG but ranked incorrectly. Here the fragment is categorically absent from the DAG. No GNN
improvement changes this — the prepass gap must be filled first.

### Design Principle Correction

This case invalidates the earlier claim that "the GNN learns aromatic sulfonamide SO₂ loss
from NIST23-scale data." Even with 8,000+ sulfonamide training spectra, the GNN cannot predict
a fragment that has zero probability of appearing in its DAG. The correct statement:

> The GNN can improve RANKING of fragments that are already in the DAG, but cannot generate
> fragments whose existence requires new bond formation. For SO₂ extrusion from aromatic
> sulfonamides, a SMARTS prepass rule must add the rearranged product to the DAG first.

### SMARTS Fixability

**Fully implementable as a conditional neutral loss rule:**

```python
ARYL_SULFONAMIDE_SO2 = "[c]-[SX4](=O)(=O)-[NH]"
# If matched in [M±H]± mode: add [M ± H − 63.962]± as candidate fragment
# NL = SO₂ = 63.9619 Da (S + 2O exact mass)
```

### Compound Class and Priority

SO₂ extrusion is one of the most documented MS/MS losses in pharmaceutical mass spectrometry:
- **Sulfonamide antibiotics**: sulfadiazine, sulfamethoxazole, sulfadimethoxine, etc.
- **Diuretics**: furosemide, hydrochlorothiazide
- **Other pharmaceuticals**: celecoxib, saccharin derivatives

Estimated NIST23 prevalence: **1,000–5,000+ spectra** (very common class).

### Fixability and Priority

| Dimension | Assessment |
|---|---|
| BFS coverage | **Gap** — new Ar−N bond required, cannot be BFS generated |
| SMARTS-fixable as NL rule | **Yes** — simple `NeutralLossRule` |
| Compound prevalence | **Very high** — ~1,000–5,000+ NIST23 spectra |
| Implementation complexity | **Low** — same infrastructure as existing rules |
| Expected wrecall improvement | **Significant** — dominant base peak in virtually all aromatic sulfonamide [M−H]⁻ spectra |

**Recommendation**: **Implement immediately.** This is the highest-priority fixable gap found
in this analysis. The SMARTS is simple, the compound class is large, and the fix is a direct
`NeutralLossRule` with no custom Python required.

---

## Note: Thioxothiazole-Oxindole Hybrid — Calibration Artifact, Opposite Sign (spec_id=686582, 686598)

- **mol_id=31850**, C11H7N3O2S2, SMILES: `O=C1Nc2ccccc2C1=Nn1c(O)csc1=S`
- **686582**: [M+H]+, prec_mz=278.005; **686598**: [M-H]-, prec_mz=275.991; nist20_hr, NCE=35%
- **cos_hun_inst_tol = 0.000** for both; dominant NL = C₂H₂OS = 73.983 Da (thioxothiazole ring opening)

**Same calibration artifact, opposite sign**: model predicts 202.008 and 204.023; GT peaks are
201.997 and 204.012 — consistently ~11 mDa BELOW theoretical. Δ = −11 mDa (GT lower than model).

For reference: spec_id=533415/572128 showed GT ~+17 mDa ABOVE model. Here GT is −11 mDa below.
Both are calibration artifacts — NIST20 fragment mass calibration errors, different batches,
different sign.

Secondary issue in [M+H]+: model ranks NL=73.983 (C₂H₂OS, 204 Da fragment) above NL=101.989
(C₂H₂N₂OS, 176 Da fragment = GT base peak). Model prefers smaller neutral loss. Not a
rearrangement coverage gap — both neutrals are achievable by BFS; this is a ranking preference.

No new actionable finding.

---

## Confirmation: Second Aromatic Sulfonamide SO₂ Case (spec_id=856552)

- **Molecule**: N-(pyrimidin-2-yl)benzenesulfonamide (unsubstituted phenyl), SMILES: `O=S(=O)(Nc1cnccn1)c1ccccc1`
- **Formula**: C10H9N3O2S, [M−H]⁻ = 234.034 Da
- **Spectrum**: spec_id=856552, mol_id=32451, FT/CID, NCE=35%, nist20_hr

GT peaks:
| m/z | Δ mDa | Assignment |
|---|---|---|
| 170.072 | 0.4 | [M−H−SO₂]⁻ = base peak — SO₂ extrusion |
| 143.061 | 0.5 | [M−H−SO₂−HCN]⁻ = secondary ring-opening |
| 235.038 | — | ¹³C isotope of [M−H]⁻ |

Confirms SO₂ extrusion mechanism across two structural variants:
- spec_id=610784: 4-nitrophenyl variant (electron-withdrawing group)
- spec_id=856552: phenyl variant (no substituent)

Both show the exact same NL=63.962 Da (0.4 mDa) to [M−H−SO₂]⁻. The reaction is
substituent-independent. The SMARTS trigger `[c]-[SX4](=O)(=O)-[NH]` covers both variants.

---

## Summary: FT Worst-Case Analysis, NCE=30-40%, Complete Survey (2026-06-20)

All spectra in `output/inference_eval/pqx7ywc9/plots/by_ce/FT/worst/30-40/` examined.

### Pattern 1: NIST20 Fragment Mass Calibration Artifact (≥5 cases)

Systematic mass offsets in GT fragment peaks vs. theoretical; precursor is well-calibrated.
FT tolerance (10 ppm / 2 mDa floor) is too tight to tolerate the offsets → cos=0.

| spec_id | compound class | adduct | offset direction | Δ mDa |
|---|---|---|---|---|
| 533415 | lignan glucoside | [M+Na]⁺ | GT > theory | +17.8 |
| 572128 | isoflavone glucoside | [M+Na]⁺ | GT > theory | +17 |
| 686582/686598 | thioxothiazole-oxindole | [M+H]⁺/[M−H]⁻ | GT < theory | −11 |
| 890352 | fluorinated kinase inhibitor | [M+H]⁺ | triple GT peaks at 271 | ±19 |
| 892079 | NHS ester coumarin | [M+H]⁺ | paired GT peaks at 243 | +11 |

Mitigation: flagging these spectra in training data as low-quality; they cannot be fixed by
model improvements. Consider adding a NIST20 spec quality filter based on precursor vs.
fragment calibration consistency.

### Pattern 2: Aromatic Sulfonamide SO₂ Extrusion BFS Gap (2 confirmed cases)

| spec_id | compound | GT base → | assignment |
|---|---|---|---|
| 610784 | 4-NO₂-PhSO₂NHPyr | 215.057 (only peak) | [M−H−SO₂]⁻, NL=63.962 |
| 856552 | PhSO₂NHPyr | 170.072 (base), 143.061 (secondary) | [M−H−SO₂]⁻ → −HCN |

Root cause: SO₂ extrusion requires new Ar−N bond formation → outside BFS DAG.
Fix: `NeutralLossRule` with SMARTS `[c]-[SX4](=O)(=O)-[NH]`, NL=63.962 Da.
**Priority: HIGH.** Compound class very common in pharmaceutical spectra.

### Pattern 3: Unusual Ion Types (2 cases)

- **spec_id=350043** (glycoconjugate [M+H]⁺): GT shows [HexNAc−H₂O+Na]⁺ at 226.067 —
  Na-adducted oxonium ions from Na contamination. Model predicts [frag+H]⁺ only.
  
- **spec_id=550113** (N-nitroimidazolidine [M−H]⁻): GT base = [M−H−N₂O]⁻ = 85.041 (NL=44.001,
  0.1 mDa). N₂O extrusion via N−N bond order change. Fix: `NeutralLossRule` on N-nitramine.

### Pattern 4: Model Ranking Failures (≥3 cases)

- **spec_id=296336**: glucuronolactone loss (NL=176) over-ranked vs H₂O cascade.
- **spec_id=686582**: C₂H₂OS NL (small loss) ranked above C₂H₂N₂OS NL (large loss).
- **spec_id=957726**: carbazole-thiadiazole [M+Na]⁺ — model prefers large fragments (150-200 Da)
  while GT base is 81.033 (small fragment). Model consistently under-predicts high-NL fragments.

### Pattern 5: Large Glycoside / [M+Na]⁺ Generalization Failures

- **spec_id=866788** (quercetin triglycoside [M−H]⁻): GT base = 718, model at 596, Δ=122 Da
  — wrong sugar sequence predicted.
- **spec_id=725593, 825515, 884280** ([M+Na]⁺ adducts): model predicts large fragments, GT
  base is a small or mid-size fragment. [M+Na]⁺ fragmentation behavior differs from [M+H]⁺;
  model may be undertrained on this adduct type.

### Actionable Findings Priority

1. **Implement aromatic sulfonamide SO₂ NeutralLossRule** — highest priority, confirmed in 2
   spectra, common in pharmaceutical databases.
2. **Implement N-nitramine N₂O NeutralLossRule** — moderate priority, niche compound class.
3. **Flag NIST20 calibration artifact spectra** — filter spectra where GT fragment masses
   diverge systematically from precursor calibration.
4. **[M+Na]⁺ model training** — model may need more [M+Na]⁺ training data or a dedicated
   Na-adduct pathway representation.
