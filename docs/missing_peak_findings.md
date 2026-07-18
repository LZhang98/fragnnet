# DAG Missing-Peak Findings, Grouped by Root Cause

Analysis of `dag_missing_by_class_m+h.txt` (FT, [M+H]+, min_intensity=0.05,
min_spectra=20). Source script: `scripts/eval/find_missing_peaks.py --frag_dp
data/frag/nist23_ma_mi_d3_h4_isoFalse.tar`.

---

Each subclass below is filed under its **dominant** cause; mixed cases note their
secondary contributors inline. Seven causes, in rough order of impact:

1. **Cascade depth** — fragment lies beyond BFS depth = 3 (largest structural gap)
2. **Rearrangement** — bond formation BFS cannot model
3. **Halide-retaining direct cations** — heterolytic C–X cleavage; the `.9` m/z bins
4. **Model weakness** — fragment is in the DAG but predicted ≈0 (not a BFS gap)
5. **Doubly-charged fragment ions** — [frag+2H]²⁺ from large lipids
6. **Data artifacts & unsupported precursors** — missing-from-tar, contamination, permanent cations
7. **Insufficient data** — n too small to conclude

---

## Cause 1 — Cascade Depth (fragment lies beyond BFS depth = 3)

BFS enumerates fragments up to `max_depth=3` sequential bond cuts from the [M+H]⁺. Multi-ring scaffolds and high-collision-energy spectra reach the observed ions only after **4+ sequential cleavages** (usually with concomitant aromatization / dehydrogenation), so the fragment is never enumerated and cannot appear in the DAG. This is the single largest structural gap, and it is inherently a **high-CE** phenomenon — at NCE ≤ 50 (the range most used for library search) BFS typically covers the dominant peaks.

**Fix levers:** a targeted SMARTS prepass that injects a pivot intermediate as a shallow leaf (letting depth-3 BFS reach the rest of the cascade); or increasing `max_depth` (combinatorially expensive). Recurring signatures: the C₅–C₁₄ aromatic cation series (65/77/91/115/128/129/141/155/165 …), tropylium, and cyclopentadienyl.

### Key Case: Phenanthrenes and Derivatives (morphine alkaloids)

**Report:** n=330, DAG miss rate=78.3%

**What ClassyFire calls "phenanthrenes":** Mostly morphine-family alkaloids
(morphine, codeine, oxymorphone, naltrexone, etc.). ClassyFire assigns them to
this subclass because the morphinan skeleton contains a phenanthrene-like
tricyclic aromatic core.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spectra | mean intensity |
|---------|-----------|----------------|
| 155.1   | 73        | 0.163          |
| 181.1   | 67        | 0.183          |
| 153.1   | 63        | 0.251          |
| 184.1   | 61        | 0.140          |
| 169.1   | 60        | 0.142          |
| 165.1   | 58        | 0.249          |
| 141.1   | 57        | 0.208          |
| 129.1   | 55        | 0.202          |

#### Fragment identities

These are protonated PAH ions produced by retro-Diels-Alder of the bridged
morphinan ring system:

| obs m/z | neutral formula | identity |
|---------|----------------|----------|
| 115.054 | C₉H₆           | protonated indene / isoindene |
| 128.062 | C₁₀H₇          | protonated naphthalene-related |
| 152.062 | C₁₂H₇          | protonated acenaphthylene |
| 168.081 | C₁₂H₉N         | N-containing aromatic ring |
| 181.088 | C₁₃H₁₀N        | N-containing PAH fragment |

#### CE dependence (codeine, mol_id=28912)

DAG coverage collapses completely at high NCE:

| NCE  | sig. peaks | in-DAG | coverage |
|------|-----------|--------|----------|
| ≤45  | 1–4       | 100%   | fine     |
| 60   | 17        | 8      | 47%      |
| 75   | 34        | 12     | 35%      |
| 90   | 71        | 15     | 21%      |
| 110  | 91        | 9      | 10%      |
| 130  | 71        | 7      | 10%      |
| 160  | 42        | 3      | 7%       |

At low CE (≤45 NCE), BFS covers everything. The problem is purely high-energy.

#### Mechanism

The morphinan skeleton (fused 5-ring bridged system) undergoes cascade
fragmentation at high CE via sequential retro-Diels-Alder steps:

```
C₁₈H₂₁NO₄⁺  →  C₁₅H₁₅NO₂⁺  →  C₁₄H₁₄N⁺  →  C₁₃H₁₀N⁺  →  C₁₂H₈⁺  →  C₉H₇⁺
    316              241             212             181             152         115
```

Each arrow is one fragmentation event of an intermediate ion. BFS models
sequential bond cuts up to max_depth=3 from the parent ion. Steps 4–6 in the
cascade above are unreachable at depth=3 regardless of h_transfer.

Additionally, at each step the fragment **aromatizes** (sp³ CH₂ → sp² CH as the
ring contracts), shedding 2–4 H as part of the structural rearrangement rather
than as inter-fragment H-transfer. BFS models H-transfer at the moment of bond
cutting, not intra-fragment post-cleavage rearrangement.

#### Why h=7 doesn't fix it

Increasing `max_h_transfer` from 4 → 7:

- **Helps:** intermediate fragments like C₁₃H₁₁N⁺ (181.088) that are 1–2 cuts
  from the parent and need ±5–6 H adjustment at a single cut.
- **Doesn't help:** the deepest cascade fragments (115.054, 128.062) that require
  5–6 sequential fragmentation events from the parent — beyond depth=3 regardless
  of h.
- **Cost:** 15 H-variants per node instead of 9 (~67% more DAG nodes globally,
  across all 52k molecules, most of which don't need it).

The binding constraint for the worst-missing fragments is **depth**, not **h**.

#### Possible fixes

1. **SMARTS prepass for morphinan RDA:** Targeted rule cutting both bonds of the
   dihydrofuran bridge in one step, injecting the aromatic ring fragment directly
   as a depth-1 leaf. Would cover 181.088 and 168.081 at minimum.
2. **Increase max_depth to 4–5:** Very expensive; combinatorial explosion.
3. **Accept the gap:** The affected spectra are all high-NCE (≥60). At the NCE
   range most useful for library search (20–50), BFS covers everything. The
   model's performance on these molecules at low/medium CE is fine.

---

### Morphinans (direct_parent='morphinans') (n=341, DAG miss rate=70.3%)

**Molecules:** 34 FT [M+H]+ molecules (385 spectra). Includes morphine, codeine, levorphanol, and
opioid analogs. Both oxygenated (with dihydrofuran ring D: morphine C₁₇H₁₉NO₃) and
non-oxygenated (levorphanol C₁₇H₂₃NO, no ring D bridge) types.

**Distinct from the "Phenanthrenes" analysis above:** The phenanthrenes group (n=330, 78.3%) covers
oxycodone/oxymorphone-type alkaloids (C₁₈, 4–5 oxygens) whose cascade retains N (C₁₃H₁₀N⁺ at
181.088). The morphinans group (n=341, 70.3%) covers morphine/codeine/levorphanol (C₁₇, 2–3
oxygens) whose cascade expels N first, producing an entirely **N-free** fragment series.

#### Missing peaks (category A — absent from DAG)

| m/z bin | exact m/z | neutral formula | identity | n_spectra | mean intensity |
|---------|-----------|-----------------|----------|-----------|----------------|
| 155.1   | 155.086   | C₁₂H₁₀         | biphenyl-type | 166 | 0.196 |
| 157.1   | 157.065   | C₁₁H₈O         | methylbenzofuran-type | 126 | 0.422 |
| 141.1   | 141.070   | C₁₁H₈          | — | 122 | 0.169 |
| 128.1   | 128.062   | C₁₀H₆          | azulene/pentalene-type | 110 | 0.469 |
| 153.1   | 153.070   | C₁₂H₈          | acenaphthylene | 103 | 0.397 |
| 181.1   | 181.065   | C₁₃H₈O         | fluorenone | 103 | 0.374 |
| 165.1   | 165.070   | C₁₃H₈          | fluorene-type | 102 | 0.471 |
| 152.1   | 152.062   | C₁₂H₈          | acenaphthylene (−H) | 102 | 0.480 |
| 129.1   | 129.070   | C₁₀H₈          | naphthalene | 96  | 0.147 |
| 183.1   | 183.080   | C₁₃H₁₀O        | fluorenol-type | 92 | 0.213 |

m/z 155.1 (biphenyl C₁₂H₁₀) appears in **30 of 34 molecules**, both oxygenated and non-oxygenated
types. The fragment series is entirely N-free: **all 10 missing peaks have no nitrogen**.

#### Cascade mechanism

Measured fragment intensities for morphine (mol_id=21671, C₁₇H₁₉NO₃, [M+H]⁺=286.144) across NCE:

```
NCE:         35    60    75    90   110   130   160
286 (prec)  0.01  0.40  0.34  0.01  0.00  0.00  0.00
229 (DAG)   0.24  0.05  0.04  0.01  0.00  0.00  0.00   ← C14H12O3, ring E opened
201 (DAG)   0.40  0.03  0.03  0.01  0.01  0.00  0.00   ← C13H12O2, -CO
183 (MISS)  0.08  0.06  0.36  0.08  0.04  0.00  0.00   ← C13H10O, fluorenol, -H2O
181 (MISS)  0.01  0.04  0.55  0.32  0.44  0.13  0.03   ← C13H8O, fluorenone, -H2
165 (MISS)  0.02  0.04  0.52  0.35  0.79  0.41  0.23   ← C13H8, -H2O from fluorenol
153 (MISS)  0.00  0.02  0.43  0.40  0.85  0.29  0.11   ← C12H8, acenaphthylene, -CO
155 (MISS)  0.03  0.03  0.39  0.19  0.16  0.02  0.00   ← C12H10, biphenyl-type
128 (MISS)  0.00  0.00  0.04  0.09  0.69  0.68  0.61   ← C10H6, -C2H2 from C12H8
```

Full cascade:

```
C17H19NO3+H  [M+H]+ = 286.144
    │ cut ring E (2 N–C bonds, -C3H7N = 57 Da)
    ▼
C14H12O3+H = 229.086   [BFS depth ≤3, IN DAG ✓]
    │ -CO (28 Da)
    ▼
C13H12O2+H = 201.091   [BFS depth ≤3, IN DAG ✓]
    │ -H2O (18 Da)
    ▼
C13H10O+H  = 183.080   [depth 4 → MISSING]   fluorenol-type
    ├─ -H2  → C13H8O+H = 181.065  fluorenone
    │             │ -CO  → C12H8+H = 153.070  acenaphthylene
    │             │           │ -C2H2 → C10H6+H = 128.062
    │             │           └──────────────────────────────
    └─ -H2O → C13H8+H  = 165.070
                  │ -C  / ring contract
                  └── C11H8+H = 141.070
C13H10O (-H2O)  ← also gives C11H8O+H = 157.065 (side branch)
Multiple paths  → C12H10+H = 155.086  (biphenyl-type, from ring-open 157 or 165)
                → C10H8+H  = 129.070  (naphthalene, from 153 -C2H2+2H)
```

#### Why BFS stops at step 3

DAG confirmation for morphine (depth=3):
- C₁₄H₁₂O₃ (229.086): **IN DAG** (Δ=0.5 mDa)
- C₁₃H₁₂O₂ (201.091): **IN DAG** (Δ=12 mDa)
- C₁₃H₁₀O  (183.080): **ABSENT** (nearest DAG fragment 45 mDa off)
- All 10 missing peaks: **ABSENT**

To reach C₁₃H₁₀O directly from morphine in 3 cuts requires isolating exactly C₄H₉NO₂ as the
neutral — impossible given the ring topology (ring E requires ≥2 cuts, ring D oxygen bridge
requires 2 more cuts; 4 cuts minimum). The BFS finds 229 and 201 via alternative 3-cut paths,
but no 3-cut path from morphine reaches C₁₃H₁₀O.

#### Fix options

1. **SMARTS prepass — morphinan ring-E+D cleavage:** Detect the morphinan tricyclic+piperidine
   scaffold, inject C₁₃H₈O (fluorenone, 181.065) as a depth-1 leaf. From there, BFS at depth+3
   can reach 153.070 (−CO) and 128.062 (−C₂H₂); 165.070 and 141.070 via further aromatization.
   Covers 103+ spectra per fragment.
2. **Inject the full N-free series:** Rules for morphinan → [C₁₃H₈O, C₁₃H₈, C₁₂H₈, C₁₂H₁₀,
   C₁₁H₈O, C₁₁H₈, C₁₀H₆, C₁₀H₈] directly.
3. **Accept the gap:** These are high-CE fragments (NCE ≥ 75). At NCE ≤ 60, BFS covers the
   dominant peaks (229, 201). The morphinan gap is moderate-CE (75–90) rather than extreme-CE.

---

### Thienothiophenes (n=104, DAG miss rate=63.1%)

**Molecules:** 6 mols (all in tar). Fused S-heterocycle carboxylic acids and nitriles, including
one with a thioether bridge (`O=C(O)c1cc2c(s1)CSC2`) and one with 3 sulfur atoms
(`c1cc2sc3ccsc3c2s1`). All have MW < 250.

**Pattern: mixed category A and B, both large.**

Category A (absent from DAG): m/z 69.0 (49 spectra, 0.53), 51.0 (29 spectra, 0.21),
121.0 (27 spectra, 0.16), 109.0 (24 spectra, 0.42), 93.0/95.0/65.0 (~20 spectra each)

Category B (in DAG, ~0 predicted): m/z 71.0 (29), 95.0 (21), 131.0 (20), 111.0 (10), 140.0 (9)

m/z 51.0 = C₄H₃⁺ (51.023) and similar small ring fragments appear in aromatic compound
fragmentation at high CE. For thienothiophenes, reaching these small (4-carbon) fragments from
a 6-carbon fused bicyclic system requires multiple sequential bond cuts beyond depth=3. This
is the same cascade depth problem as morphinans, manifesting in the thiophene ring series
instead of the PAH series.

The category B (in DAG but underpredicted) peaks at 71/95/131/140 suggest the model has
partially learned the cascade but can't assign them proper intensity.

**Cascade depth problem. Similar to morphinans for fused heterocyclic systems.**

---

### Phenoxazines (n=112, DAG miss rate=62.8%)

**Molecules:** 8 mols (all in tar). Dibenzo[1,4]oxazine compounds including N-acetyl
hydroxyphenoxazine (mol_id=2952), chlorophenoxazine with aminoalkyl chain (mol_id=17835),
and several oxidized/nitrated variants.

**Category A top peaks:**

| m/z bin | n_spectra | mean int | note |
|---------|-----------|----------|------|
| 115.1   | 39        | 0.772    | Very high intensity |
| 170.1   | 42        | 0.369    | — |
| 146.1   | 39        | 0.326    | — |
| 117.1   | 36        | 0.420    | — |
| 91.1    | 36        | 0.238    | — |

m/z 115.054 = [C₉H₆ + H]⁺ (same indene/isoindene cation seen in phenanthrene RDA case).
DAG check: m/z 115.054 is absent from mol_id=2952 DAG, present in mol_id=17835 (within 20 mDa)
→ inconsistent coverage depending on substituents.

The missing series (91.1, 115.1, 117.1, 130.1, 146.1, 170.1) is consistent with cascade ring
opening of the phenoxazine tricyclic system at high CE, producing progressively smaller aromatic
ions. The phenoxazine bridge (O between two arene rings) cleaves first, then the ring system
further fragments through 3+ steps beyond what depth=3 BFS can reach.

**Cascade depth problem + partial DAG coverage depending on substitution pattern.**

---

### Protoberberine Alkaloids and Derivatives (n=105, DAG miss rate=62.1%)

**Molecules:** 7 mols; 5 in tar, **2 absent**:
- mol_id=28968: `COc1ccc2cc3[n+](cc2c1OC)CCc1cc2c(cc1-3)OCO2` — **berberine** (permanent [n+] quaternary ammonium cation, 3 spectra)
- mol_id=28900: `COc1ccc2c(c1OC)CN1CCc3cc4c(cc3C1C2)OCO4` — neutral tetrahydroisoquinoline (1 spectrum)

Berberine is a **permanently charged quaternary N⁺ salt** — its precursor ion is [M]⁺ not [M+H]⁺.
BFS is designed for neutral molecules that are protonated; it likely cannot handle the berberine
scaffold where N is already positively charged before fragmentation. This explains why BFS was
never run on it (absent from tar).

The 5 in-tar mols (tetrahydroberberines and epiberberine-type) are neutral molecules with
genuine fragmentation data. Their missing peaks (91.1, 131.1, 105.1, 103.1, 148.1) are small
aromatic ions (C₇H₇⁺, C₁₀H₇⁺, C₈H₉⁺) produced by deep cascade of the tetracyclic ring
system at high CE — depth cascade problem, not a structural BFS limitation.

**Mixed: artifact (berberine absent from tar — unsupported permanent cation) + cascade depth
gap for the neutral tetrahydroprotoberberines.**

---

### Cephalotaxus Alkaloids (n=46, DAG miss rate=60.5%)

**Molecules:** 3 mols; 2 in tar, 1 absent (mol_id=23186, homoharringtonine with a complex
diterpenoid ester side chain, 21 spectra → artifact for those).

**Mol_id=23185** (cephalotaxine, 14 spectra) and **mol_id=24265** (homodaphniphyllate-type,
13 spectra) are in the tar.

**Missing peaks match the morphinan/N-free aromatic cascade:**

| m/z bin | n_spectra | mean int | overlap with morphinans? |
|---------|-----------|----------|--------------------------|
| 172.1   | 18        | 0.140    | Close to morphinan 181.1 |
| 141.1   | 14        | 0.154    | **Yes** — same bin as morphinans |
| 157.1   | 10        | 0.085    | **Yes** — same bin as morphinans |
| 155.1   | 12        | 0.128    | **Yes** — same bin as morphinans |
| 115.1   | 10        | 0.689    | Common aromatic series |
| 170.1   | 12        | 0.249    | Close to morphinan 165.1 |

The cephalotaxane skeleton is an ABCDE pentacyclic ring system (benzazulene-like) that undergoes
cascade fragmentation analogous to morphinans, producing overlapping aromatic ion series
(141.1, 155.1, 157.1 match morphinan cascade exactly). The missing mol (homoharringtonine)
has an additional diterpenoid side chain that would produce even more cascade fragments.

**Cascade depth problem. Same mechanism as morphinans but with cephalotaxane core.**

---

### Pyranodioxins (n=45, DAG miss rate=58.0%)

**2 mols; 1 in tar.** Only mol_id=24496 (COC1OC2COC(c3ccccc3)OC2C(O)C1O — a benzylidene acetal
of a hexose sugar) contributes. The "pyranodioxin" ClassyFire label reflects the bicyclic 1,3-dioxane +
pyranose ring fusion of acetalized hexoses.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Assignment |
|---------|--------|-----------|------------|
| **69.0** | **36** | **0.615** | C₄H₅O⁺ furanyl cation (69.034) |
| **81.0** | **36** | **0.599** | C₅H₅O⁺ methylenefuranyl (81.034) |
| **109.0** | **36** | **0.279** | C₇H₉⁺ or C₆H₅O₂⁺ |
| **99.0** | **33** | **0.185** | C₅H₇O⁺ (99.044) |
| 71.0    | 18     | 0.104     | C₄H₇O⁺    |
| 97.0    | 18     | 0.107     | C₅H₅O⁺ dehydrated |

#### Root cause

**Cascade depth from pyranose ring fragmentation.** The category A fragments appear in 33–36 of 45
spectra at high intensity (0.60 and 0.28) — extremely consistent for a single compound. These are
**furanyl/furfuryl cations** (C₄H₅O⁺ at 69.034, C₅H₅O⁺ at 81.034) produced by pyranose ring opening
followed by multiple sequential dehydration steps. Pyranose→furanose ring contraction plus 2–3
dehydrations requires 4+ sequential steps from the molecular ion — beyond BFS depth=3. Category B
fragments (121.1, 127.0, 145.0 in DAG) are larger glycan-related fragments modestly underpredicted.

---

### Lactones (n=57, DAG miss rate=57.6%)

**12 mols; 7 in tar.** 5 mols not in tar (10242, 10338, 14113, 24387, 47370) contribute 8.2% of
FT [M+H]+ spectra — minor artifact contribution. Remaining mols include macrocyclic lactones
(C12–C14 rings), terpenoid lactones (sesquiterpene-like multi-ring systems), and acyclic hydroxy
lactones.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 191.1   | 33     | 0.644     | C₁₄H₁₁⁺ aromatic cation |
| 119.1   | 30     | 0.422     | C₉H₇⁺ indanyl cation |
| 209.1   | 27     | 0.492     | C₁₄H₁₃O⁺ or C₁₅H₁₃⁺ |
| 139.1   | 24     | 0.347     | C₁₀H₁₁⁺ |
| 95.0    | 21     | 0.204     | C₅H₅O⁺ furanyl cation |
| 69.0    | 15     | 0.096     | C₄H₅O⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **91.1** | **27** | **0.737** |
| 111.1   | 27     | 0.334     |
| 105.1   | 27     | 0.357     |
| 77.0    | 21     | 0.336     |
| 115.1   | 17     | 0.175     |
| 175.1   | 15     | 0.178     |

#### Root cause

**Mixed: cascade depth for terpenoid ring fragments + model weakness for aromatic ions.**

Category A fragments (191.1 = C₁₄H₁₁⁺, 119.1 = C₉H₇⁺, 209.1 = C₁₄H₁₃O⁺) are consistent with
cascade aromatization of the terpenoid/sesquiterpene lactone ring system — same mechanism as
androstane steroids and morphinans. The 191.1 and 209.1 series suggests 4+ bond cuts from the
multi-ring lactone precursor.

Category B: **m/z 91.1 = tropylium [C₇H₇]⁺ (27 spectra, 0.737 intensity) IS in DAG but model
predicts ≈0** — likely from benzyl-containing lactone mols where BFS generates the tropylium
fragment but model has not learned to assign high probability to it.

---

### Androstane Steroids (n=919, DAG miss rate=56.3%)

**112 mols; 86 in tar.** 26 mols not in tar contribute only 2.2% of FT [M+H]+ spectra — minor
artifact. Largest class analyzed to date (n=919 spectra). Includes testosterone, DHT,
androstenedione, and related C19 steroidal androgens.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Assignment |
|---------|--------|-----------|------------|
| **129.1** | **274** | **0.160** | [C₁₀H₉]⁺ = 129.070 — azulene/naphthalenyl cation |
| 67.1    | 259    | 0.245     | [C₅H₇]⁺ cyclopentadienyl |
| **128.1** | **241** | **0.289** | [C₁₀H₈]⁺ naphthalene-type |
| **115.1** | **202** | **0.303** | [C₉H₇]⁺ indanyl |
| 141.1   | 186    | 0.175     | [C₁₁H₉]⁺ or [C₁₀H₉O]⁺ |
| 117.1   | 185    | 0.166     | [C₉H₉]⁺ |
| 65.0    | 165    | 0.250     | [C₅H₅]⁺ cyclopentadienyl |
| 157.1   | 152    | 0.194     | [C₁₂H₉]⁺ |
| 77.0    | 147    | 0.341     | [C₆H₅]⁺ phenyl |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **105.0** | **184** | **0.387** |
| 95.1    | 103    | 0.106     |
| **159.1** | **101** | **0.255** |
| 161.1   | 97     | 0.091     |
| 107.0   | 93     | 0.152     |
| 145.1   | 89     | 0.118     |

#### Root cause

**Cascade depth — aromatic cascade of the androstane ring system.**

Confirmed: m/z 129.1 measured at exactly 129.0697–129.0698 in actual spectra = **[C₁₀H₉]⁺ =
129.070** (azulene or naphthalenyl cation). This is the classic indicator of steroidal ring
aromatization in high-CE CID: the A/B-ring system of androstane undergoes Djerassi-type cascade
fragmentation (retro-Diels-Alder + dehydrogenation), producing C₅–C₁₀ aromatic/partially-aromatic
cations (65.0 = C₅H₅⁺, 67.1 = C₅H₇⁺, 77.0 = C₆H₅⁺, 91.1, 115.1, 128.1, 129.1, 141.1). These
require 4+ sequential bond cuts + H-transfers from the [M+H]⁺ — beyond BFS depth=3.

Category B: m/z 105.0 (184 spectra, 0.387) = [C₇H₅O]⁺ from aromatized steroid ring — IS in DAG
but severely underpredicted. Model has not learned steroid ring aromatization fragment intensities.

This class has the largest sample size of any subclass analyzed (n=919) and represents a systemic
failure: cascade aromatization is a universal, high-intensity pathway for androstane-type steroids
that BFS depth=3 cannot reach.

---

### Dibenzoxazepines (n=126, DAG miss rate=54.5%)

**7 mols; all in tar.** All are clozapine-type antipsychotics: dibenz[b,f][1,4]oxazepine core +
N-methylpiperazine/piperidine + Cl substituent on one benzene ring. Includes clozapine (mol_id=21770),
methoxy variant (28815), N-oxide (30271), norclozapine (34833), hydroxy variant (51705).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity |
|---------|--------|-----------|
| 164.0   | 18     | 0.396     |
| **153.1** | **17** | **0.428** |
| 200.0   | 16     | 0.099     |
| 152.0   | 15     | 0.277     |
| **229.1** | **13** | **0.655** |
| 138.0   | 13     | 0.205     |
| **140.0** | **11** | **0.487** |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 180.0   | 20     | 0.271     |
| 125.0   | 14     | 0.304     |
| **126.0** | **13** | **0.598** |
| 70.1    | 12     | 0.183     |
| 127.0   | 11     | 0.214     |
| 257.0   | 10     | 0.177     |

#### Root cause

**Mixed: cascade depth + model weakness.**

m/z 229.1 (13 spectra, 0.655) absent from DAG: confirmed in mol_id=28815 (methoxy clozapine) in
11/24 spectra. For these [M+H]+ ≈ 362 molecules, 229.1 corresponds to a neutral loss of ~133 Da —
consistent with simultaneous ejection of N-methylpiperazine + a CO-equivalent from the tricyclic
oxazepine ring. Requires opening the oxazepine ring AND the piperazine linkage → cascade depth.

Category B: m/z 126.0 (13 spectra, 0.598) IS in DAG but model predicts ≈0 — likely the
[piperazine + CO]⁺ ring fragment or a N-methylpiperazinyl cation from the side chain; model weakness
for these psychopharmaceutical ring systems.

---

### Estrane Steroids (n=739, DAG miss rate=53.7%)

**57 mols; 49 in tar.** 8 mols not in tar contribute 0 FT [M+H]+ spectra — no artifact. C18 estrogens
(estradiol, ethinylestradiol, synthetic progestins) with an **aromatic phenol A-ring** — unlike
androstane steroids (keto A-ring).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Assignment |
|---------|--------|-----------|------------|
| 115.1   | 235    | 0.321     | [C₉H₇]⁺ = 115.054 |
| **128.1** | **205** | **0.303** | [C₁₀H₈]⁺ naphthalenyl |
| 141.1   | 173    | 0.133     | [C₁₁H₉]⁺ |
| 155.1   | 171    | 0.115     | [C₁₂H₁₁]⁺ |
| **129.1** | **170** | **0.195** | [C₁₀H₉]⁺ = 129.070 — azulene/naphthalenyl |
| 165.1   | 137    | 0.159     | [C₁₃H₉]⁺ |
| **91.1** | **130** | **0.620** | [C₇H₇]⁺ tropylium |
| 153.1   | 117    | 0.098     | [C₁₂H₉]⁺ |
| 181.1   | 110    | 0.134     | [C₁₄H₉]⁺ or [C₁₃H₉O]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **105.0** | **143** | **0.322** |
| 133.1   | 116    | 0.144     |
| 109.1   | 103    | 0.154     |
| 147.1   | 100    | 0.100     |
| 185.1   | 90     | 0.121     |
| 159.1   | 84     | 0.151     |

#### Root cause

**Cascade depth — same aromatic cascade as androstane steroids, extended by the phenol A-ring.**

The same C₅–C₁₃ aromatic ion series (115.1, 128.1, 129.1, 141.1) mirrors androstane steroids
exactly, arising from retro-Diels-Alder + dehydrogenation of the B/C/D ring system (4+ steps).
Distinct addition: **m/z 91.1 = tropylium [C₇H₇]⁺ absent in 130 spectra at 0.620 intensity**.
Estrogens have a phenol A-ring (C₆H₄OH-) that provides a direct route to tropylium via ring
contraction; this pathway does not appear as prominently in androstane steroids.

Category B: 105.0 (143 spectra, 0.322) IS in DAG (aromatized A-ring [C₇H₅O]⁺ from phenol) but
model predicts ≈0 — second largest model weakness after androstane steroids (184 spectra). Together
the two steroid classes represent the most widespread model failure in the dataset.

---

### Hydrolyzable Tannins (n=354, DAG miss rate=53.6%)

**68 mols; 41 in tar.** 3 spectra from missing-tar mols (negligible, <1% artifact). Gallotannins
and ellagitannins — polyester macromolecules with gallic acid or hexahydroxydiphenic acid (HHDP)
esterified to a glucose core. Large polyphenolic structures (MW 500–3000).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Assignment |
|---------|--------|-----------|------------|
| **65.0** | **57** | **0.574** | [C₅H₅]⁺ cyclopentadienyl |
| 81.1    | 52     | 0.123     | [C₅H₅O]⁺ or [C₆H₉]⁺ |
| 213.1   | 48     | 0.332     | galloyl cascade fragment |
| 67.1    | 44     | 0.153     | [C₅H₇]⁺ |
| 95.0    | 38     | 0.330     | [C₆H₇O]⁺ |
| 255.1   | 38     | 0.190     | ellagyl/galloyl fragment |
| 241.0   | 37     | 0.317     | ellagyl fragment |
| 183.0   | 36     | 0.212     | gallic acid-derived |
| 155.0   | 35     | 0.307     | galloyl fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **91.1** | **97** | **0.370** |
| **115.1** | **44** | **0.617** |
| 119.0   | 65     | 0.143     |
| 69.0    | 51     | 0.151     |
| 103.1   | 45     | 0.207     |
| **128.1** | **45** | **0.461** |
| 105.0   | 44     | 0.163     |

#### Root cause

**Cascade depth from ester hydrolysis cascade + model weakness for aromatic cations.**

BFS cuts ester bonds (depth 1) to release galloyl/HHDP fragments, then further fragments the
aromatic ring (depth 2–3) to give phenol/catechol cations. But reaching the small aromatic cations
(65.0 = C₅H₅⁺, 67.1, 81.1) requires 4+ steps from the tannin [M+H]⁺ — cascade depth.

Category B: **115.1 (44 spectra, 0.617!), 128.1 (45 spectra, 0.461), 91.1 (97 spectra, 0.370)** —
all in DAG (galloyl/phenyl ring fragments correctly generated by BFS) but model predicts ≈0.
The tannin dataset amplifies model weakness because all mols share the same polyphenolic chemistry.

---

### Strychnos Alkaloids (n=117, DAG miss rate=51.8%)

**33 mols; 4 in tar.** 29 mols not in tar; those 29 mols contribute **33/107 = 30.8%** of FT [M+H]+
spectra. However, after the BFS bug fix, those 33 spectra are **skipped** (dag_mzs is None →
continue) — they do NOT inflate the A/B miss rate. The 51.8% miss rate is computed from the 4
in-tar mols (strychnine, brucine, and 2 structural analogs).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity |
|---------|--------|-----------|
| 156.1   | 40     | 0.364     |
| 220.1   | 35     | 0.142     |
| 154.1   | 31     | 0.228     |
| 155.1   | 29     | 0.105     |
| **184.1** | **23** | **0.604** |
| 204.1   | 25     | 0.227     |
| 192.1   | 25     | 0.115     |
| 234.1   | 23     | 0.241     |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **180.1** | **18** | **0.550** |
| **222.1** | **18** | **0.354** |
| 206.1   | 18     | 0.185     |
| 194.1   | 16     | 0.178     |
| 91.1    | 32     | 0.195     |

#### Root cause

**Cascade depth — heptacyclic strychnane ring system.**

Strychnine and brucine have ABCDEFG heptacyclic ring systems (indole fused to the
azatricyclo[3.3.3.0]decane cage). The category A fragments (154–234 Da range across 23–40 spectra)
are mid-size ring-opened products requiring 4+ sequential bond cuts from the heptacyclic cage.
m/z 184.1 (23 spectra, **0.604**) is a particularly prominent absent fragment — reported in
strychnine MS literature as a characteristic B/C-ring fragment.

Category B: m/z 180.1 (18 spectra, 0.550) IS in DAG but severely underpredicted — model weakness
component. The 4 in-tar mols all share the same scaffold so both cascade depth and model failure
appear consistently.

---

### Oxepanes (n=353, DAG miss rate=50.4%)

**51 mols; 34 in tar.** 17 mols not in tar contribute ~5.4% of FT [M+H]+ spectra — minor artifact.
7-membered oxygen-containing rings: sesquiterpene oxepanes, monoterpene oxides, bicyclic ether
terpenoids. High spectral count (n=353 FT [M+H]+).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity |
|---------|--------|-----------|
| 105.1   | 98     | 0.314     |
| 117.1   | 94     | 0.176     |
| 65.0    | 91     | 0.294     |
| 119.1   | 89     | 0.206     |
| **115.1** | **86** | **0.419** |
| 93.1    | 82     | 0.227     |
| 77.0    | 79     | 0.374     |
| **91.1** | **74** | **0.577** |
| 79.1    | 67     | 0.252     |
| 105.0   | 65     | 0.231     |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 155.1   | 114    | 0.115     |
| 145.1   | 98     | 0.196     |
| 143.1   | 87     | 0.195     |
| 141.1   | 74     | 0.225     |
| 157.1   | 60     | 0.180     |
| 165.1   | 60     | 0.204     |
| 129.1   | 51     | 0.137     |

#### Root cause

**Cascade depth — same aromatic cation series as androstane and estrane steroids.**

The category A fragment list (65.0, 77.0, 79.1, 91.1, 93.1, 105.1, 115.1, 117.1, 119.1) is the
**complete C₅–C₉ aromatic/polyunsaturated cation series** seen across multiple steroid and terpenoid
classes. The oxepane ring opening produces allylic cations that then cascade into these small
aromatic ions via dehydration and aromatization — all requiring 4+ steps.

Category B is dominated by the **C₁₀–C₁₂ range** (141.1–165.1 in DAG at 0.11–0.20 intensity) —
model underpredicts the intermediate cascade products. The contrast between large B and small A
fragmentation is consistent with the model learning BFS-depth-2 fragments but not depth-3 ones that
then fragment further to the depth-4+ species in category A.

---

### Flavonoids (n=161, DAG miss rate=49.8%)

**27 mols; 23 in tar.** Unsubstituted and lightly-substituted flavones (flavone, methylflavones,
naphthoflavones, and ester-substituted flavones). 4 mols missing from tar: 4194 (pentaacetyl
quercetin), 4203 (triacetyl apigenin), 33168 (6-Br-4'-methylflavone), 50287 (2'-hydroxyflavone).
The sole Br-mol (33168) is missing from tar → its spectra skipped → no .9-bin Br direct-cation
peaks appear in category A.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| **165.1** | **52** | **0.171** | C₁₃H₉⁺ = 165.070 confirmed (mean 165.064 ± 0.019) |
| 178.1   | 22     | 0.169     | C₁₄H₁₀⁺ = 178.078 — PAH fragment from naphtho-fused flavones |
| 152.1   | 17     | 0.156     | C₁₂H₈⁺ = 152.063 — same aromatic cascade |
| 202.1   | 14     | 0.125     | C₁₆H₁₀⁺ = 202.078 — pyrene/fluoranthene cation |
| 167.1   | 12     | 0.113     | C₁₃H₁₁⁺ = 167.086 — pre-aromatization intermediate |
| 62.0    | 9      | 0.424     | Unassigned; exact m/z ≈ 62.011–62.019 |
| 151.0   | 10     | 0.218     | C₈H₇O₃⁺ = 151.040 — A-ring fragment (trihydroxy) |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **105.0** | **49** | **0.226** |
| 95.0    | 26     | 0.197     |
| 69.0    | 26     | 0.079     |
| **123.0** | **20** | **0.428** |
| 103.1   | 16     | 0.194     |
| **113.0** | **11** | **0.386** |

#### Root cause

**Mixed — cascade depth for 165.1/178.1/152.1 aromatic series + model weakness for 105.0/123.0.**

The 165.1 peak (C₁₃H₉⁺ = 165.070 confirmed by FT exact mass) appears in **150/249 spectra** (67%)
across all flavone types, not just the naphtho-fused ones. Mechanism: two sequential CO losses from
the chromone C-ring (BFS depth=2 → C₁₃H₁₁⁺ = 167.086) followed by ring closure + H₂ neutral loss
(dehydrogenation to give aromatic C₁₃H₉⁺). The H₂ neutral loss from an already-fragmented ion is
the same BFS gap as in steroid ring aromatization. The broader PAH series (152.1, 178.1, 202.1)
comes primarily from the naphtho/benzo-fused flavones (50265, 50276, 50281, 50285) which have a
pre-formed tricyclic system requiring fewer cascade steps.

Category B: **m/z 105.0 (49 spectra, 0.226) = [C₇H₅O]⁺ benzoyl cation** from B-ring RDA IS in DAG
but model predicts only 0.020. This is the canonical flavone B-ring fragment — the model's failure to
predict it at high intensity is a significant training gap. Similarly, 123.0 at 0.428 (in DAG, pred=
0.005) is the highest-intensity category B fragment seen in this subclass. **The B category
(model weakness) likely contributes more to the overall miss rate than cascade depth.**

---

### Furofurans (n=93, DAG miss rate=48.6%)

**17 mols; 12 in tar** (5 missing: mol 24208 contributes 5 and mol 25416 contributes 2 skipped spectra). Furofurans here are predominantly sesquiterpene- and diterpene-lactone natural products bearing a fused bicyclic furan core (MW 130–533), alongside a few small saturated bicyclic furofuranols.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 95.1 | 38 | 0.497 | [C₇H₁₁]⁺ 95.086 (9/10 mols) AND [C₆H₇O]⁺ 95.049 (9/10 mols) — two co-binned species |
| 105.1 | 35 | 0.298 | [C₈H₉]⁺ (105.070) — 9/10 mols |
| 91.1 | 32 | 0.515 | [C₇H₇]⁺ tropylium (91.054) — **10/10 mols** |
| 107.1 | 32 | 0.417 | [C₈H₁₁]⁺ 107.086 AND [C₇H₇O]⁺ 107.049 — two co-binned species |
| 81.1 | 32 | 0.152 | [C₆H₉]⁺ (81.070) — 8/10 mols |
| 79.1 | 29 | 0.375 | [C₆H₇]⁺ (79.055) — 8/10 mols |
| 93.1 | 23 | 0.156 | [C₇H₉]⁺ (93.070) — 9/10 mols |
| 67.1 | 22 | 0.487 | [C₅H₇]⁺ (67.054) — 9/10 mols |
| 65.0 | 20 | 0.359 | [C₅H₅]⁺ cyclopentadienyl (65.039) — 8/10 mols |
| 145.1 | 15 | 0.220 | [C₁₁H₁₃]⁺ 145.101 AND [C₁₀H₉O]⁺ 145.065 — 6–7/10 mols |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 169.1 | 22 | 0.162 |
| 141.1 | 20 | 0.185 |
| 125.1 | 19 | 0.133 |
| 155.1 | 18 | 0.129 |
| 129.1 | 18 | 0.195 |
| 165.1 | 18 | 0.175 |
| 167.1 | 17 | 0.157 |
| 143.1 | 16 | 0.172 |
| 103.1 | 16 | 0.100 |
| 128.1 | 15 | 0.440 |

#### Root cause

**Cascade depth + rearrangement, with secondary model weakness.**

Category A is dominated by a complete C₅–C₁₁ aliphatic/aromatic cation series absent from the BFS DAG — and it is nearly universal: [C₇H₇]⁺ tropylium (91.054) appears in **all 10 in-tar mols**, and 8–9/10 mols show each of [C₅H₅]⁺, [C₅H₇]⁺, [C₆H₇]⁺, [C₆H₉]⁺, [C₇H₉]⁺, [C₇H₁₁]⁺, [C₈H₉]⁺, [C₆H₇O]⁺ (confirmed by FT exact masses ±2 mDa). Two mechanisms co-operate: (1) **Cascade depth** — large sesqui/diterpene precursors (MW 309–533, e.g. mol 9915) require 5+ bond cuts to reach m/z 65–145; (2) **Rearrangement** — the cyclic hydrocarbon cations (tropylium via ring expansion, cyclopentadienyl via ring contraction, [C₇H₁₁]⁺ via ring closure from open chain) require bond-forming steps BFS cannot model, and appear even in the smallest mols (e.g. 9377, MW 196). Category B shows additional **model weakness**: [C₁₀H₈]⁺ at 128.1 (n=15, true=0.440) is in the DAG but the model predicts exactly 0.000 — the starkest model-weakness signal in this subclass.

---

### Quinolizines (n=38, DAG miss rate=48.5%)

**14 mols; 7 in tar** (7 missing, of which only mol 26936 — a permanent [n+] cation — contributes 3 skipped spectra). In-tar mols include Br-substituted (mol 77), N-oxide (mols 26903, 28813), aldehyde-bearing (mols 48295, 48296), and alkyl-substituted (mols 26880, 26881) variants of the quinolizidine/dihydroquinolizine ring system.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 117.1 | 24 | 0.234 | [C₈H₇N]⁺ 117.058 AND [C₉H₉]⁺ 117.070 — two co-binned species; **7/7 mols** |
| 115.1 | 14 | 0.486 | [C₉H₇]⁺ (115.054) — 7/7 mols |
| 103.1 | 13 | 0.237 | [C₈H₇]⁺ (103.054) — 7/7 mols |
| 116.1 | 9 | 0.115 | [C₉H₈]⁺ or [C₈H₆N]⁺ — 7/7 mols |
| 105.1 | 7 | 0.328 | [C₈H₉]⁺ (105.070) |
| 91.1 | 7 | 0.631 | [C₇H₇]⁺ tropylium (91.054) — also in Category B for other spectra |
| 89.0 | 7 | 0.203 | [C₇H₅]⁺ (89.039) |
| 130.1 | 6 | 0.104 | [C₁₀H₁₀]⁺ or [C₉H₈N]⁺ |
| 131.1 | 6 | 0.127 | [C₁₀H₁₁]⁺ |
| 102.0 | 6 | 0.141 | [C₈H₆]⁺ (102.047) |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 172.1 | 9 | 0.368 |
| 146.1 | 9 | 0.107 |
| 91.1 | 7 | 0.732 |
| 147.1 | 7 | 0.148 |
| 145.1 | 6 | 0.485 |
| 118.1 | 6 | 0.235 |
| 191.1 | 5 | 0.299 |

#### Root cause

**Cascade depth, with secondary model weakness.**

The quinolizine/dihydroquinolizine ring system uniformly produces a deep C₇–C₉ aromatic cascade series — [C₇H₅]⁺ (89.0), [C₇H₇]⁺ tropylium (91.1), [C₈H₇]⁺ (103.1), [C₉H₇]⁺ (115.1), [C₈H₇N]⁺ and [C₉H₉]⁺ (co-binned at 117.1) — absent from the DAG in all 7/7 in-tar mols (confirmed by FT exact masses ±2 mDa). Even the smallest precursor (mol 48295, [M+H]⁺=202) must lose ≥111 Da to reach m/z 91, requiring 3–4 sequential bond cuts plus ring-rearrangement (tropylium formation, C₉H₇⁺ via ring contraction). Notably, [C₇H₇]⁺ appears in BOTH category A (absent in 7 spectra from simpler mols) and category B (in DAG but predicted at 0.012 for 7 spectra from larger mols) — the same fragment is either absent or severely underpredicted depending on precursor size. Category B shows additional model weakness: 91.1 true intensity=0.732 (highest in either category), 145.1 true=0.485, and 172.1 true=0.368, all predicted at ≤0.015.

---

### Phenanthrolines (n=68, DAG miss rate=48.5%)

**68 spectra; ~20 mols missing from tar.** Fused tricyclic N-heterocycles (1,10-phenanthroline and substituted derivatives) used as metal chelators and in pharmaceuticals.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 170.1   | 25     | 0.125     | Aromatic N-heterocycle ring fragment [C₁₂H₈N]⁺ |
| 99.0    | 14     | 0.332     | Small ring fragment |
| 146.1   | 9      | 0.343     | [C₁₀H₈N]⁺? phenanthridine fragment |
| 115.1   | 10     | 0.277     | [C₉H₇]⁺ or [C₈H₅N]⁺ aromatic cascade |
| 155.1   | 8      | 0.228     | Aromatic amine ring fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 139.1   | 8      | 0.668          | 0.019          |
| 125.0   | 10     | 0.356          | 0.007          |
| 75.0    | 11     | 0.294          | 0.015          |
| 77.0    | 14     | 0.194          | 0.016          |
| 143.0   | 13     | 0.078          | 0.005          |

#### Root cause

**Cascade depth + model weakness.**

The category A aromatic series (115.1, 146.1, 170.1) requires sequential ring-opening of the fused tricyclic phenanthroline scaffold beyond depth=3. The 170.1 peak (25 spectra) is likely a direct ring-opening product requiring 4+ cuts from larger substituted precursors. Category B dominance (139.1 at 0.668 intensity, predicted 0.019) points to GNN model weakness on N-heterocycle fragments that appear across many scaffolds — the phenanthroline ring fragment embeddings are non-discriminative. High tar-miss rate (~20 mols) inflates category A counts.

---

### Benzofuranones (n=500, DAG miss rate=47.4%)

**500 spectra; large, diverse class.** Benzofuran-2(3H)-one and related lactone variants — abundant in natural products (coumarins, chromenes) and pharmaceutical compounds.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 115.1   | 57     | 0.390     | [C₉H₇]⁺ — aromatic cascade |
| 152.1   | 36     | 0.338     | Benzofuranyl fragment |
| 165.1   | 50     | 0.346     | [C₁₃H₉]⁺ or [C₁₁H₉O]⁺ |
| 91.1    | 35     | 0.313     | Tropylium [C₇H₇]⁺ |
| 226.1   | 38     | 0.475     | Large aromatic fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 152.1   | 51     | 0.571          | 0.001          |
| 239.1   | 24     | 0.469          | 0.003          |
| 269.1   | 25     | 0.373          | 0.002          |
| 105.0   | 57     | 0.259          | 0.025          |
| 165.1   | 24     | 0.258          | 0.004          |

#### Root cause

**Mixed: cascade depth + model weakness (dual-mode for 152.1 and 165.1).**

The 152.1 peak appears in both category A (36 spectra, absent) and category B (51 spectra, 0.571, pred 0.001) — it is at the coverage boundary where some mols generate it in BFS and others don't. Aromatic cation series (91.1, 115.1, 128.1) absent from DAG reflect the benzofuranone ring-opening cascade requiring >3 sequential cuts. The 105.0=[C₇H₅O]⁺ benzoyl cation (57 spectra in DAG, predicted 0.025) is a persistent model weakness across O-containing aromatic classes. With n=500, this class has high statistical power — the gaps are genuine.

---

### Pyranocoumarins (n=248, DAG miss rate=47.4%)

**248 spectra; 18 mols confirmed in tar.** Fused pyran-coumarin ring systems (angular and linear) — common in umbelliferae plant secondary metabolites (xanthyletin, decursin, etc.).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 66     | 0.485     | Tropylium — 66/248 spectra affected! |
| 128.1   | 37     | 0.533     | [C₁₀H₈]⁺ — high intensity naphthalene cation |
| 115.1   | 37     | 0.489     | [C₉H₇]⁺ aromatic cascade |
| 105.0   | 58     | 0.258     | Benzoyl/aromatic cation |
| 65.0    | 43     | 0.307     | [C₅H₅]⁺ cyclopentadienyl |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 165.1   | 34     | 0.384          | 0.000          |
| 128.1   | 30     | 0.350          | 0.000          |
| 257.1   | 32     | 0.269          | 0.025          |
| 215.1   | 32     | 0.131          | 0.016          |
| 107.0   | 31     | 0.131          | 0.014          |

#### Root cause

**Cascade depth dominant — pyranochromone aromatization cascade.**

The complete C₅–C₁₀ aromatic cation series (65.0, 77.0, 91.1, 105.0, 115.1, 128.1) is absent from 37–66 spectra each. The cascade pathway: pyran O-ring opens → chromene → chromone → benzopyrilium → phenyl → cyclopentadienyl, each step requiring bond cleavage + hydrogen transfer. With n=248, the absolute gap (91.1 absent in 66 spectra at 0.485 intensity) rivals the androstane steroid gap. The 128.1 peak appears in both cat A (37 spec, absent) and cat B (30 spec, 0.350, pred 0.000) — same dual-mode pattern as benzofuranones. A single SMARTS prepass injecting the dihydrocoumarin pivot would allow BFS to reach the aromatic ion series.

---

### Protopine Alkaloids (n=123, DAG miss rate=47.1%)

**123 spectra; 4 mols in tar.** Pentacyclic N-benzyltetrahydroisoquinoline alkaloids (protopine, allocryptopine, cryptopine) — common in Papaveraceae.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 57     | 0.417     | Tropylium — cascade from the benzylic C |
| 247.1   | 54     | 0.152     | Large ring fragment [M+H−164]⁺ type |
| 275.1   | 48     | 0.166     | Pentacyclic ring fragment |
| 131.1   | 41     | 0.206     | [C₉H₇N]⁺ or isoquinoline fragment |
| 130.1   | 37     | 0.313     | [C₉H₈N]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 189.1   | 36     | 0.485          | 0.030          |
| 165.1   | 69     | 0.183          | 0.008          |
| 178.1   | 32     | 0.261          | 0.023          |
| 179.1   | 50     | 0.137          | 0.008          |
| 205.1   | 39     | 0.128          | 0.006          |

#### Root cause

**Cascade depth — pentacyclic ring requires >3 sequential cuts.**

Protopine has a pentacyclic scaffold (ABCDE ring system): reaching the benzylic 91.1 (tropylium) requires opening 3 rings sequentially before the methylene bridge can rearrange. The large mid-size fragments at 247.1 and 275.1 (absent in 54/48 spectra) are likely B/C-ring retaining fragments that require loss of one or two complete rings. The 189.1 (0.485, pred 0.030) and 165.1 (0.183, pred 0.008) in category B are model weakness on isoquinoline-type fragments — same pattern as protoberberines. Only 4 mols in tar limits statistical power.

---

### Butyrophenones (n=582, DAG miss rate=46.8%)

**582 spectra; 15+ mols.** Aryl butyrate esters and butyrophenone drugs (haloperidol-type) — large pharmaceutical class with aryl-C(=O)-alkyl scaffold.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 65.0    | 79     | 0.670     | [C₅H₅]⁺ — very high intensity cascade fragment! |
| 117.1   | 73     | 0.385     | [C₉H₉]⁺ aromatic cascade |
| 91.1    | 107    | 0.174     | Tropylium [C₇H₇]⁺ |
| 131.1   | 58     | 0.338     | Aromatic fragment |
| 89.0    | 43     | 0.430     | [C₇H₅]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 105.0   | 53     | 0.148          | 0.025          |
| 176.1   | 28     | 0.237          | 0.026          |
| 156.1   | 38     | 0.072          | 0.002          |
| 146.1   | 33     | 0.154          | 0.008          |
| 72.1    | 39     | 0.083          | 0.024          |

#### Root cause

**Mixed: cascade depth (dominant) + rearrangement.**

Butyrophenones have a phenyl ring connected via a 4-carbon chain to a carbonyl, and haloperidol-type compounds have an additional piperidyl fluorobenzene. The 65.0 (0.670, 79 spectra) and 117.1 (0.385, 73 spectra) absent from DAG are C₅–C₉ aromatic cascade ions from the fluorophenyl/chlorophenyl ring after sequential loss of the chain and Cl/F. The 91.1 (tropylium, 107 spectra) is rearrangement-driven. Category B shows 105.0 and 176.1 in DAG but underpredicted — haloperidol-type fragments where the model fails on the CF₃/Cl-ring retaining larger fragments.

---

### Thienopyrroles (n=87, DAG miss rate=46.3%)

**87 spectra; ~8 mols in tar.** Fused thiophene-pyrrole bicyclic heterocycles — found in pharmaceuticals and natural alkaloids; some members have Br/Cl substituents.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 172.9   | 12     | 0.572     | .9 bin → Br-retaining aromatic direct cation |
| 51.0    | 5      | 0.702     | [C₄H₃]⁺ — highest intensity; small thiophene cascade |
| 93.0    | 7      | 0.555     | [C₄H₃S]⁺ thienyl cation |
| 94.0    | 11     | 0.408     | [C₄H₄S]⁺ or [C₅H₄N]⁺ |
| 122.9   | 8      | 0.169     | .9 bin → Br-containing fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 80.0    | 5      | 0.474          | 0.001          |
| 53.0    | 6      | 0.398          | 0.000          |
| 70.0    | 12     | 0.307          | 0.000          |
| 150.0   | 7      | 0.195          | 0.037          |
| 96.0    | 6      | 0.248          | 0.026          |

#### Root cause

**Cascade depth (S-heterocycle cleavage) + halide direct cations.**

The .9 bins (172.9, 122.9) are Br-retaining direct cations from brominated thienopyrrole mols — same mechanism as triazolopyrazines and organobromides. The cascade depth pattern (51.0 at 0.702, 93.0 at 0.555) mirrors thienothiophenes: sequential ring-opening of the fused S-heterocycle requires [C₄H₃]⁺ and [C₄H₃S]⁺ to be produced after 4+ cuts. Category B shows 80.0 (0.474) and 53.0 (0.398) — these thienyl ring fragments are in DAG but the model assigns 0.000, likely because thienopyrrole scaffolds are underrepresented in training.

---

### Triterpenoids (n=1017, DAG miss rate=46.0%)

**1017 spectra; 88/199 mols missing from tar (~113 spectra skipped).** Pentacyclic triterpenes (oleanolic acid, ursolic acid, betulinic acid, lupeol-type) — ubiquitous plant secondary metabolites with C30 pentacyclic skeleton.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 286    | 0.620     | Tropylium — 286/1017 spectra; LARGEST ABSOLUTE GAP |
| 145.1   | 283    | 0.194     | [C₁₁H₉]⁺ aromatic cascade |
| 147.1   | 266    | 0.239     | [C₁₁H₁₁]⁺ isoprenyl cation |
| 161.1   | 263    | 0.205     | [C₁₂H₁₃]⁺ terpenoid cation |
| 79.1    | 244    | 0.362     | [C₆H₇]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 119.1   | 201    | 0.410          | 0.003          |
| 105.1   | 192    | 0.416          | 0.004          |
| 159.1   | 150    | 0.239          | 0.002          |
| 187.1   | 135    | 0.227          | 0.006          |
| 131.1   | 132    | 0.231          | 0.001          |

#### Root cause

**Cascade depth dominant — complete C₅–C₁₄ terpenoid cation series; largest absolute miss in dataset.**

Triterpenoids have a pentacyclic C30 skeleton. Reaching the terpenoid aromatic cation series (91.1, 145.1, 147.1, 161.1 = C₅–C₁₂ cations formed by sequential methylenic cleavage + aromatization) requires 5+ sequential bond cuts from the intact C30 precursor. This is the same cascade depth mechanism as androstane/estrane steroids but for a larger ring system. Category B is equally severe: 119.1 (201 spectra, 0.410, pred 0.003) and 105.1 (192 spectra, 0.416, pred 0.004) are smaller terpenoid cations in DAG but model assigns near-zero probability — GNN repetitive-chain embedding failure (same as quinone lipids, saccharolipids). With 286 spectra missing tropylium, triterpenoids have the largest absolute category A gap in the dataset.

---

### Matrine Alkaloids (n=64, DAG miss rate=45.8%)

**64 spectra; ~8 mols in tar.** Tetracyclic quinolizidine alkaloids (matrine, oxymatrine, sophocarpine) — from Sophora plants; all-saturated or partially unsaturated bicyclic/tetracyclic N-ring systems.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 120.1   | 37     | 0.590     | High intensity! [C₈H₁₀N]⁺ quinolizidine ring fragment |
| 91.1    | 32     | 0.538     | Tropylium — rearrangement or cascade |
| 106.1   | 29     | 0.494     | [C₇H₈N]⁺ dihydropyridinium |
| 132.1   | 27     | 0.312     | [C₉H₁₀N]⁺ matrinyl fragment |
| 118.1   | 25     | 0.300     | [C₈H₈N]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 79.1    | 31     | 0.448          | 0.007          |
| 93.1    | 50     | 0.220          | 0.004          |
| 176.1   | 33     | 0.143          | 0.005          |
| 98.1    | 37     | 0.167          | 0.025          |
| 70.1    | 28     | 0.242          | 0.006          |

#### Root cause

**Cascade depth — tetracyclic quinolizidine ring requires >3 sequential cuts to reach small N-fragments.**

Matrine has a tetracyclic skeleton (quinolizidine fused twice: ABCD ring). To reach 120.1=[C₈H₁₀N]⁺ (the D ring dihydropyridinyl fragment) or 91.1 (tropylium), 4+ ring-opening events are needed sequentially. The 106.1=[C₇H₈N]⁺ dihydropyridinium (29 spectra, 0.494) is a classic quinolizidine ring fragmentation product absent because BFS can only reach 2–3 rings deep. Category B (79.1 at 0.448, 93.1 at 0.220) shows model weakness for alicyclic N-containing fragments that appear across many alkaloid classes — the same GNN embedding problem as morphinans. A SMARTS prepass for the matrine bicyclo[2.2.2]octane core would inject pivot intermediates and resolve the cascade depth.

---

### Organonitrogen Compounds (n=733, DAG miss rate=45.2%)

**733 spectra; broad catch-all class.** Generic N-containing organic molecules not assigned to more specific ClassyFire subclasses — structurally highly diverse.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 54     | 0.520     | Tropylium — rearrangement across diverse N-scaffolds |
| 77.0    | 43     | 0.504     | [C₆H₅]⁺ phenyl — cascade |
| 95.0    | 50     | 0.345     | [C₇H₇]⁺ or [C₆H₅N]⁺ |
| 65.0    | 44     | 0.239     | [C₅H₅]⁺ cyclopentadienyl |
| 51.0    | 26     | 0.354     | [C₄H₃]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 67.1    | 56     | 0.278          | 0.004          |
| 55.1    | 27     | 0.123          | 0.032          |
| 83.1    | 23     | 0.244          | 0.015          |
| 95.1    | 32     | 0.178          | 0.023          |
| 55.0    | 21     | 0.293          | 0.010          |

#### Root cause

**Cascade depth dominant (aromatic cation series) + model weakness on aliphatic fragments.**

The full C₄–C₇ aromatic cation series (51.0, 65.0, 77.0, 91.1, 95.0) is absent from DAG in 26–54 spectra each — the same cascade depth mechanism seen across multiple aromatic-containing classes. As a catch-all class, molecules range from simple anilines to complex polycyclic N-alkaloids, so cascade depth affects only the subset with ring systems. The 71.9 bin (27 spectra, 0.163) is a .9 fractional bin suggesting some Br/Cl-retaining compounds in the class. Category B aliphatic fragments (67.1, 55.1) reflect model weakness on small unsaturated chains.

---

### Oxanes (n=234, DAG miss rate=44.3%)

**234 spectra; ~12 mols in tar.** Six-membered oxygen-containing rings (tetrahydropyrans, oxacyclohexanes) — sugars, O-glycoside aglycones, and O-substituted ring compounds.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 22     | 0.676     | Tropylium — highest intensity; cascade from pendant aryl |
| 51.0    | 19     | 0.633     | [C₄H₃]⁺ — small aromatic cascade |
| 80.0    | 19     | 0.545     | [C₅H₄O]⁺ or [C₄H₄S]⁺ |
| 79.1    | 43     | 0.435     | [C₆H₇]⁺ cyclohexenyl or allylic |
| 65.0    | 46     | 0.188     | [C₅H₅]⁺ cyclopentadienyl |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 95.0    | 39     | 0.423          | 0.007          |
| 67.1    | 27     | 0.363          | 0.014          |
| 81.1    | 19     | 0.208          | 0.010          |
| 57.1    | 21     | 0.223          | 0.022          |
| 57.0    | 24     | 0.077          | 0.018          |

#### Root cause

**Cascade depth — same aromatic cation series as oxepanes; 6-membered O-ring.**

Oxanes follow the exact same fragmentation cascade as oxepanes (50.4% miss): the O-ring opens, the resulting allylic carbocation undergoes sequential dehydration + aromatization, producing the C₅–C₉ aromatic cation series (51.0, 65.0, 79.1, 91.1). Category A shows 65.0 (46 spectra), 79.1 (43 spectra), and 91.1 (22 spectra at 0.676) all absent — 3+ sequential cuts required from the 6-membered ring. Category B (95.0 at 0.423, 67.1 at 0.363) shows model weakness on intermediate ring-opening products correctly in DAG. A shared SMARTS prepass for 5/6/7-membered O-ring scaffolds would address all three subclasses (1,3-dioxanes, oxanes, oxepanes) simultaneously.

---

### Hexahydrobenzophenanthridine Alkaloids (n=94, DAG miss rate=43.4%)

**94 spectra; ~6 mols in tar.** Hexahydrobenzophenanthridine-type alkaloids (chelerythrine, sanguinarine reduced forms) — tetracyclic benzylisoquinoline derivatives from Papaveraceae.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 289.1   | 30     | 0.613     | Large ring-retaining fragment [M+H−58]⁺ type |
| 275.1   | 32     | 0.608     | Tetracyclic ring fragment |
| 247.1   | 27     | 0.409     | [M+H−86]⁺ type after multiple ring losses |
| 117.1   | 16     | 0.110     | [C₉H₉]⁺ aromatic cascade |
| 103.1   | 16     | 0.176     | [C₈H₇]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 188.1   | 19     | 0.518          | 0.020          |
| 261.1   | 24     | 0.441          | 0.000          |
| 189.1   | 18     | 0.363          | 0.019          |
| 217.1   | 18     | 0.304          | 0.001          |
| 190.1   | 37     | 0.156          | 0.017          |

#### Root cause

**Cascade depth — tetracyclic benzophenanthridine ring requires >3 sequential cuts for large fragments.**

The large absent fragments (275.1, 289.1 at 0.608–0.613 intensity) result from sequential losses of the C and D rings from the intact [M+H]⁺ — each ring loss requires a bond cut, so losing 2 rings takes 2 cuts, leaving only 1 cut for further fragmentation at depth=3. For fully substituted hexahydrobenzophenanthridines ([M+H]⁺ ~350 Da), reaching 247.1 from [M+H]⁺ requires 3+ cuts just to lose the outer rings. Category B (188.1 at 0.518, 261.1 at 0.441) shows model weakness on isoquinoline-type fragments — same pattern as protoberberines and protopine alkaloids in this Papaveraceae alkaloid cluster.

---

### Metallotetrapyrroles (n=227, DAG miss rate=43.4%)

**227 spectra; 17/27 mols in tar.** Despite the subclass name, this group in NIST23 is dominated by free-base porphyrins and bilins (linear tetrapyrroles) — the ClassyFire label includes metal-free tetrapyrrole macrocycles (porphyrin, chlorin, bilin scaffolds).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 167.1   | 123    | 0.314     | Largest cat A; pyrrole ring fragment |
| 230.1   | 111    | 0.158     | Dipyrrole-type fragment |
| 244.1   | 102    | 0.120     | Dipyrrole+CH₂CH₂ |
| 243.1   | 101    | 0.222     | Dipyrrole variant |
| 256.1   | 101    | 0.164     | Tripyrrole fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 115.1   | 78     | 0.475          | 0.001          |
| 192.1   | 87     | 0.244          | 0.009          |
| 317.1   | 87     | 0.226          | 0.004          |
| 305.1   | 112    | 0.179          | 0.006          |
| 179.1   | 78     | 0.264          | 0.010          |

#### Root cause

**Cascade depth (pyrrole macrocycle sequential cleavage) + model weakness.**

Free-base porphyrins ([M+H]⁺ ≈ 400–900 Da) fragment by sequential loss of peripheral substituents (propionic acid chains, vinyl groups) then ring cleavage to produce pyrrole and dipyrrole cations. The 167.1 peak (123 spectra, 0.314) is absent from DAG — a pyrrole ring fragment that requires 4+ cuts from the large macrocycle: lose 2–3 peripheral chains, then cleave 1–2 pyrrole units. The series 230.1, 243.1, 244.1, 256.1 (dipyrrole fragments, ~100 spectra each) reflect progressive loss of pyrrole units from the tetrapyrrole macrocycle. Category B shows 115.1 (78 spectra, 0.475, pred 0.001) — the model fails on this common aromatic fragment across the diverse porphyrin scaffold. With 10 mols missing from tar, some category A elevation is artifactual.

---

### Imidazo[4,5-c]pyridine Ribonucleosides and Ribonucleotides (n=93, DAG miss rate=67.6%)

**3 mols; 3 in tar; 39 FT [M+H]⁺ spectra.** All 2-amino-1H-imidazo[4,5-c]pyridine N-glycosides: 6-chloro riboside (41441, [M+H]⁺=301.07), riboside (41680, 267.11), deoxyriboside (41679, 251.11).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 136.1 | 20 | 0.234 | Exact 136.051 = [C₆H₆N₃O]⁺ — base+O rearrangement (ribose O incorporated during glycosidic cleavage) |
| 64.0 | 18 | 0.710 | Exact 64.018 = [C₄H₂N]⁺ — deep ring fragment, cascade depth |
| 98.0 | 6 | 0.204 | Exact 97.979 — Cl-retaining fragment from 6-chloro mol_id=41441 at high NCE (90–160%) |
| 52.0 | 3 | 0.122 | Deep ring fragment, cascade depth |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 133.1 | 5 | 0.297 | 0.021 |
| 106.0 | 3 | 0.153 | 0.019 |

#### Root cause

**Cascade depth (64.0) + rearrangement (136.1) + Cl direct cation (98.0).**

The highest-intensity missing peak, 64.018 Da = [C₄H₂N]⁺, is a small ring fragment from deep sequential ring-opening of the 9-atom bicyclic base — BFS depth=3 cannot reach it from the full riboside. The 136.1 peak at exact 136.051 = [C₆H₆N₃O]⁺ is the base ion with an oxygen incorporated from the ribose during glycosidic bond cleavage (rearrangement; normal [base+H]⁺ = 135.067); it appears in 20 spectra across both non-Cl mols. The 98.0 bin (97.979) is a Cl-retaining fragment from the 6-chloro derivative only, growing with NCE from 90–160% (Cl direct cation). Category B weakness on 133.1 ([base+H−HCl]⁺, pred=0.021) is minor.

---

### Organic Pyrophosphates (n=32, DAG miss rate=83.3%)

**6 mols; 3 in tar; 0 spectra from non-tar mols.** All spectra derive from a single molecule: tetraethyl pyrophosphate [(EtO)₂P(=O)−O−P(=O)(OEt)₂, mol_id=19750, [M+H]⁺=291.075].

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 178.9 | 10 | 0.656 | Exact m/z=178.9501 — same physical peak as 179.0 bin |
| 179.0 | 8 | 0.856 | Same peak at 178.9501, binned differently at boundary |
| 63.0 | 2 | 0.070 | [PO₂]⁺ at 62.964 — terminal phosphorus fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 207.0 | 4 | 0.362 | 0.025 |

#### Root cause

**Neutral-loss cascade depth — 4 successive ethylene losses stack past depth = 3.** (Not a rearrangement gap: each C₂H₄ loss *is* coverable by BFS.)

Tetraethyl pyrophosphate fragments by successive loss of C₂H₄ (28.031 Da): the β-hydrogen on each ethyl group migrates to the phosphate oxygen, expelling ethylene, giving the series 291.075 → 263.044 → 235.013 → 206.982 → 178.950. Each step is a single O−C bond cut with one H transferred back to the fragment — squarely inside BFS's `cut + max_h_transfer` model (the DAG is built at `d3_h4`). The proof is in the data: **m/z 207.0 = 206.982 = 3 ethylene losses is already in the DAG** (category B), so BFS covers the per-step ethylene loss and only *underpredicts* it (0.025 vs 0.362 → model weakness). The dominant missing peak, 178.950 (the base peak here, 0.86 intensity), is the **4th** successive loss — that is what exceeds depth = 3, and the terminal [PO₂]⁺ (62.964) needs full dephosphorylation deeper still. So the gap is depth (for the 4th loss) plus model weakness (for the 3rd), not an uncoverable rearrangement. Because the neutral loss has a fixed formula and a clean SMARTS pattern (ethyl phosphate ester), a `NeutralLossRule` prepass for −C₂H₄ would inject the deep members directly — a cheaper fix than raising `max_depth`.

---

## Cause 2 — Rearrangement (bond formation BFS cannot model)

BFS only **breaks** bonds; it cannot **form** them. Fragments that require a skeletal rearrangement cannot be generated at any depth or `max_h_transfer`:

- **Ring expansion** benzyl → tropylium [C₇H₇]⁺ (6→7-membered)
- **Ring contraction** phenyl → cyclopentadienyl [C₅H₅]⁺ (6→5)
- **Ring fusion** phenyl + cyclohexyl → naphthalene/azulene
- **Atom migration** methyl migration → silyl cation (TMS⁺); P migration → phosphacyclopentadienyl [C₅H₄P]⁺

Tropylium and cyclopentadienyl also drive many *cascade-depth* classes; here they appear even in small molecules where depth is not the limiter, isolating rearrangement as the dominant cause.

> **Not here: McLafferty / β-elimination neutral losses.** A single ethylene loss from an
> ethyl ester is one bond cut plus one H transferred to the leaving oxygen — inside BFS's
> `cut + max_h_transfer` model, not a skeletal rearrangement. These are covered per step
> (they become cascade-depth cases only when *many* successive losses stack past depth = 3).
> See Organic Pyrophosphates under Cause 1.

### Organic Metalloid Salts (n=100, DAG miss rate=76.1%)

**Molecules:** 7 mols with FT [M+H]+ spectra: cyclic siloxanes D3–D7
(`C[Si]1(C)O[Si]...`), trimethylsilylmethylamine, and methylarsenic acid. All 6
siloxane mols are in the tar; mol_id=30196 (methylarsenic acid, `C[As](=O)(O)O`)
is absent → 10 spectra skipped after the `continue on dag_mzs is None` fix.

**Real BFS gap — silyl cation formation:** Cyclic siloxanes produce a well-known
fragment series by sequential Si-O cleavage with methyl migration:

| obs m/z | identity | formula | in DAG? |
|---------|----------|---------|---------|
| 73.047  | TMS⁺ = (CH₃)₃Si⁺ | C₃H₉Si⁺ | **NO** — 37 mDa from DAG's 73.010 |
| 91.0    | rearranged siloxyl | Si₂CₓHᵧ | NO |
| 147.07  | [TMS-O-SiMe₂]⁺ | C₅H₁₃Si₂O⁺ | NO |
| 221.09  | [TMS-(O-SiMe₂)₂]⁺ | C₇H₁₉Si₃O₂⁺ | NO |
| 345.14  | [TMS-(O-SiMe₂)₄]⁺ | C₁₁H₃₁Si₅O₄⁺ | NO |

BFS does generate a methylsiloxyl ion [CH₃SiO+H]⁺ (C₂H₄OSi, 73.010), but this
differs by 37 mDa from the observed TMS⁺ (73.047). At the 20 mDa matching
tolerance, TMS⁺ is correctly classified as absent.

The root cause: TMS⁺ = (CH₃)₃Si⁺ requires **methyl migration** before Si-O
cleavage — an intra-molecular rearrangement BFS cannot represent. The larger
series (m/z 147, 221, 295, 345) is the same mechanism extended for D4–D7.

**Silyl cation rearrangement. Not RDA.**

---

### Lycorine-Type Amaryllidaceae Alkaloids (n=96, DAG miss rate=55.4%)

**4 mols; 1 in tar (50701=lycorine, 32 spectra).** Mols 3711/3764 (lycorine diacetates) and 50712 absent from tar with 0 FT [M+H]⁺ spectra. All are tetracyclic Amaryllidaceae alkaloids with a methylenedioxyphenyl ring and a fused pyrroline system.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 119.0 | 66 | 0.659 | Exact 119.049 = [C₈H₇O]⁺ — base peak; methylenedioxy aromatic rearrangement |
| 91.1 | 39 | 0.507 | Exact 91.054 = [C₇H₇]⁺ — tropylium, ring expansion rearrangement |
| 95.0 | 39 | 0.082 | Exact 95.049 = [C₆H₇O]⁺ — methylenedioxy fragment via rearrangement |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 65.0 | 14 | 0.423 | 0.012 |
| 147.0 | 11 | 0.321 | 0.028 |
| 177.1 | 12 | 0.230 | 0.024 |

#### Root cause

**Rearrangement dominant — methylenedioxy ring fragmentation + tropylium.**

The base peak at 119.049 = [C₈H₇O]⁺ is absent from the DAG in 66 of 96 spectra at 0.659 intensity. For lycorine's methylenedioxyphenyl ring, this fragment requires ring contraction or bond formation during fragmentation (BFS can only cut bonds, not form them). Tropylium at 91.054 (39 spectra, 0.507) is the canonical rearrangement product from ring expansion. Both are structurally impossible for BFS to generate. Category B shows significant model weakness on 65.0 = [C₅H₅]⁺ (cyclopentadienyl, 0.423 true vs 0.012 predicted) and 147.0 (0.321 vs 0.028) — fragments that ARE in the DAG but severely underpredicted, suggesting a secondary model weakness component.

---

### Gamma-Keto Acids and Derivatives (n=115, DAG miss rate=55.0%)

**16 mols; 14 in tar.** Aliphatic γ-keto acids (levulinic acid type), plus furanyl and thienyl variants with aromatic rings (mols 8483, 18285). Predominantly aliphatic chain structures with keto and ester groups.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1 | 10 | 0.744 | Exact 91.054 = [C₇H₇]⁺ tropylium — rearrangement from furanyl/thienyl mols |
| 95.0 | 10 | 0.438 | Exact 95.049 = [C₆H₇O]⁺ — methylenedioxy/furanyl cation rearrangement |
| 79.1 | 11 | 0.469 | [C₆H₇]⁺ / [C₅H₅N]⁺ mixed — ring cation from aromatic cascade |
| 93.1 | 9 | 0.383 | [C₇H₉]⁺ — aliphatic ring closure rearrangement |
| 67.1 | 10 | 0.201 | [C₅H₇]⁺ — aliphatic cation series |
| 81.1 | 9 | 0.244 | [C₅H₅O]⁺ / [C₆H₉]⁺ — aliphatic + furanyl cascade |
| 213.1 | 14 | 0.113 | Larger fragment, cascade depth |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 107.1 | 8 | 0.488 | 0.000 |
| 105.0 | 9 | 0.314 | 0.001 |
| 150.1 | 13 | 0.328 | 0.025 |
| 167.1 | 8 | 0.332 | 0.019 |

#### Root cause

**Rearrangement dominant (aromatic mols) + cascade depth (aliphatic mols) + model weakness.**

Tropylium at 91.054 (highest intensity at 0.744, 10 spectra) dominates the missing peaks — this arises from furanyl (mol 8483) and thienyl (mol 18285) ring-containing mols via ring expansion rearrangement (5→7 membered ring). The aliphatic cation series (67.1, 79.1, 81.1, 93.1, 95.0) requires multi-step ring closure and H-migration from the open-chain keto acid backbone — also rearrangement-based. Category B shows 107.1 (0.488 true, pred=0.000) and 105.0 (0.314, pred=0.001) severely underpredicted despite being in DAG — model weakness on benzoyl/phenyl-type fragments present in the aromatic-ring mols.

---

### Phenylphosphines and Derivatives (n=495, DAG miss rate=49.1%)

**57 mols; 37 in tar.** 20 mols missing from tar — ALL are phosphonium salts ([P+] in SMILES:
butyltriphenylphosphonium, methyltriphenylphosphonium, tetraphenylphosphonium, etc.) with
**zero FT [M+H]⁺ spectra** (recorded as [M]⁺ direct cations in ESI, not [M+H]⁺). No
artifact contribution. In-tar mols are neutral phosphines, phosphine oxides, ylides, and
iminophosphoranes — all with the Ph-P bond framework.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| **95.0** | **131** | **0.269** | C₅H₄P⁺ = 95.005 confirmed — phosphacyclopentadienyl cation |
| 115.1   | 84     | 0.241     | C₉H₇⁺ — aromatic cascade (indanyl/naphthalenyl) |
| 152.1   | 80     | 0.318     | C₁₂H₈⁺ or C₁₁H₉⁺ — also in category B |
| 105.0   | 62     | 0.194     | C₈H₉⁺/[C₇H₅O]⁺ — aromatic cascade |
| 133.0   | 52     | 0.119     | also in category B — structural heterogeneity |
| 128.1   | 52     | 0.158     | C₁₀H₈⁺ — aromatic cascade |
| 139.1   | 44     | 0.265     | also in category B |
| 91.1    | 43     | 0.290     | tropylium [C₇H₇]⁺ — aromatic cascade |
| 165.1   | 36     | 0.360     | C₁₃H₉⁺ — also in category B |
| 219.1   | 35     | 0.483     | C₁₂H₁₂O₂P⁺ = 219.058 — phosphine oxide fragment; also in B |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 152.1   | 65     | 0.310     |
| **165.1** | **57** | **0.530** |
| 184.0   | 47     | 0.099     |
| **219.1** | **38** | **0.377** |
| 133.0   | 36     | 0.271     |
| 78.0    | 35     | 0.125     |
| 183.0   | 34     | 0.277     |
| 139.1   | 31     | 0.157     |
| 185.1   | 30     | 0.227     |

#### Root cause

**Mixed — phosphorus rearrangement (C₅H₄P⁺) + cascade depth + model weakness for P-containing ions.**

**m/z 95.0 = C₅H₄P⁺ = 95.005 (confirmed)**: the phosphacyclopentadienyl cation — the
phosphorus analog of the cyclopentadienyl cation. Present in spectra of 8+ structurally
distinct mol types (phosphine oxides, neutral phosphines, ylides). Root cause: **rearrangement**
— the phenyl ring contracts and P migrates into a 5-membered ring before departure. BFS cannot
model bond FORMATION, only bond breaking.

**m/z 183.0 = C₁₂H₈P⁺ = 183.037 (confirmed)** = [Ph₂P]⁺ (diphenylphosphinyl cation) and
**m/z 184.0 = C₁₂H₉P⁺ = 184.044** ARE in DAG (direct Ph-P bond cut from triarylphosphines)
but the model predicts ≈0 intensity. Similarly **78.0 = C₆H₆⁺ = 78.047** (benzene cation) is
in DAG but underpredicted. Model weakness for all P-containing and phenyl-derived secondary
fragments.

**Structural heterogeneity in BFS coverage**: m/z 152.1, 165.1, 133.0, 139.1, 219.1 appear in
BOTH category A (absent from DAG in complex phosphines) and category B (in DAG but underpredicted
in simpler phosphines) — for simpler phosphines BFS reaches these fragments at depth ≤3;
for complex polyphenyl phosphines they require depth >3 cascade aromatization.

---

### Cyclohexylphenols (n=86, DAG miss rate=49.0%)

**17 mols; 14 in tar** (3 missing: mol 14796 contributes 9 skipped FT [M+H]⁺ spectra). Cyclohexylphenols are phenols bearing a cyclohexyl group; the phenyl + cyclohexyl ring system is prone to aromatization cascade upon CID.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1 | 24 | 0.516 | [C₇H₇]⁺ tropylium (91.0543) — 9/14 mols; 6→7-membered ring expansion |
| 103.1 | 20 | 0.310 | [C₈H₇]⁺ (103.054) — 7/14 mols; indenyl/phenylacetylene cation |
| 65.0 | 19 | 0.296 | [C₅H₅]⁺ cyclopentadienyl (65.039) — 8/14 mols; phenyl ring contraction |
| 128.1 | 14 | 0.613 | [C₁₀H₈]⁺ naphthalene/azulene cation (128.062) — 6/14 mols; ring fusion |
| 53.0 | 12 | 0.133 | [C₄H₅]⁺ (53.039) — deep aromatic cascade |
| 127.1 | 12 | 0.114 | [C₁₀H₇]⁺ — deep cascade |
| 102.0 | 12 | 0.251 | [C₈H₆]⁺ — deep cascade |
| 129.1 | 12 | 0.222 | [C₁₀H₉]⁺ (129.070) — deep cascade |
| 50.0 | 10 | 0.171 | [C₄H₂]⁺ — deep cascade |
| 139.1 | 9 | 0.071 | [C₁₁H₇]⁺ — deep cascade |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 105.0 | 34 | 0.290 |
| 105.1 | 15 | 0.156 |
| 79.1 | 11 | 0.375 |
| 117.1 | 11 | 0.215 |
| 131.0 | 10 | 0.235 |
| 95.0 | 10 | 0.300 |
| 81.1 | 10 | 0.131 |
| 116.1 | 10 | 0.063 |

#### Root cause

**Rearrangement + cascade depth.**

Category A is dominated by aromatic rearrangement products that require bond-making steps BFS cannot perform: tropylium [C₇H₇]⁺ (91.054, 9/14 mols) requires 6→7-membered ring expansion from benzyl carbocation; cyclopentadienyl [C₅H₅]⁺ (65.039, 8/14 mols) requires phenyl ring contraction (6→5); naphthalene/azulene cation [C₁₀H₈]⁺ (128.062, 6/14 mols) requires ring fusion of the phenyl and cyclohexyl rings via dehydrogenation. All three are confirmed by FT exact masses (±2 mDa). The pattern is nearly subclass-universal: 9 of 14 in-tar mols show 91.1, across both simple cyclohexylphenols (mol 51573, 31309) and functionalized variants (nitro, carboxylic acid, amine). Secondary: model weakness on 105.0 (n=34, true=0.290, pred=0.004) — the largest category B hit in this subclass.

---

### Acetophenones (n=20, DAG miss rate=48.0%)

**20 spectra; very small sample (n=20 → caution on conclusions).** Aryl methyl ketones — benzoylmethane-type compounds with substituted phenyl ring.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 91.1    | 5      | 0.446     | Tropylium [C₇H₇]⁺ — ring-expansion rearrangement |
| 115.1   | 3      | 0.177     | [C₉H₇]⁺ aromatic cascade |
| 93.1    | 3      | 0.059     | Allylic/cyclic cation |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 105.0   | 3      | 0.529          | 0.032          |
| 123.0   | 4      | 0.178          | 0.018          |
| 59.0    | 2      | 0.054          | 0.034          |

#### Root cause

**Rearrangement + model weakness; n=20 too small for firm conclusions.**

Tropylium 91.1 (5 spectra at 0.446) is a ring-expansion rearrangement of the phenyl ring to the 7-membered tropylium cation — BFS cannot model bond formation. The 105.0=[C₇H₅O]⁺ benzoyl cation (0.529 intensity in DAG, predicted 0.032) represents model weakness for a chemically sensible fragment. The entire miss rate is driven by ≤5 spectra per bin, so these numbers are not statistically robust.

---

## Cause 3 — Halide-Retaining Direct Cations (heterolytic C–X cleavage)

BFS models every fragment as **[neutral_formula + H]⁺**. Heterolytic C–X cleavage that leaves the charge on the **halogen-bearing** fragment (CH₂Br⁺, aryl–Br⁺, C–I⁺, organochloride cations) cannot be represented — BFS can only let the halide depart as an anion.

**Diagnostic fingerprint:** category-A m/z bins ending in **.9**, from the mass defects of ⁷⁹Br (+0.918), ³⁵Cl (+0.969) and ¹²⁷I (+0.904). The same mechanism recurs across benzyl halides, brominated azoles, iodo-azoles, and chloro/bromo sulfoxides.

### Benzyl Halides (n=97, DAG miss rate=76.0%)

**Molecules:** 8 cyanobenzyl halides and benzyloxy bromides. All 8 active FT
[M+H]+ mols are in the tar — **zero artifact contribution**. The 76% miss rate
is entirely genuine.

**Missing peaks (category A):**

| obs m/z bin | likely identity | why BFS misses it |
|-------------|-----------------|-------------------|
| 92.9  | CH₂⁷⁹Br⁺ = 92.934 | Direct halide-retaining cation |
| 151.1 | Cl/Br-containing or rearrangement fragment | Direct cation, not [neutral+H]⁺ |
| 152.1 | Related halide fragment | Direct cation, not [neutral+H]⁺ |
| 118.1 | Halide-loss rearrangement | — |

Root cause: BFS models all fragments as **[neutral_formula + H]⁺**. Benzyl
halide fragmentation proceeds by heterolytic C-X cleavage producing **direct
cations that carry the halogen**:

```
ArCH₂Br → ArCH₂⁺  +  Br⁻     (Br leaves as anion — BFS handles this)
ArCH₂Br → Ar       +  CH₂Br⁺  (Br stays in cation — BFS cannot model)
```

CH₂Br⁺ at m/z 92.934 (⁷⁹Br) is the clearest case: a direct carbocation retaining
Br, not a protonated neutral. It appears consistently across benzyl bromide spectra
and grows with CE. BFS has no mechanism to route the charge to the
halide-containing fragment while the aromatic ring departs as neutral.

**Heterolytic C-X cleavage producing halide-retaining cations. Not RDA.**

---

### Triazolopyrazines (n=38, DAG miss rate=63.3%)

**Molecules:** 5 mols (all in tar). Mix of unsubstituted triazolopyrazine, CF₃-substituted
(sitagliptin-like), and Br-substituted variants:
- mol_id=275: `Brc1cn2ncnc2c(Br)n1` — 3,5-dibromotriazolopyrazine (2 Br atoms)
- mol_id=5039: `CC(C)(C)OC(=O)N1CCn2c(Br)nnc2C1` — mono-Br Boc-protected
- mol_id=42172: sitagliptin intermediate (CF₃, no Br)
- mol_id=35852: CF₃-triazolopyrazine (no Br)
- mol_id=52783: parent ring (no substituents)

**Category A — all .9 decimal m/z bins:**

| obs m/z bin | actual m/z | assignment | why BFS misses |
|-------------|------------|------------|----------------|
| 90.9  | 90.918 | C⁷⁹Br⁺ (aromatic C–Br cation) | Direct Br-retaining cation |
| 142.9 | ~142.93 | C₃N₃⁷⁹Br⁺ or C₂HN₄⁷⁹Br⁺ | Direct Br-retaining cation |
| 160.9 | ~160.92 | C₄HN₃⁷⁹Br⁺ | Direct Br-retaining cation |
| 169.9 | ~169.94 | C₄H₂N₄⁷⁹Br⁺ or similar | Direct Br-retaining cation |
| 129.9 | ~129.93 | C₂N₃⁷⁹Br⁺ | Direct Br-retaining cation |

All category A m/z bins end in .9 — the mass defect of ⁷⁹Br (78.918) means any fragment
retaining one Br atom has a mass offset of ~0.92 from integer. Confirmed by observing m/z
90.918 growing with CE in mol_id=275 spectra.

Root cause: same as benzyl halides — **heterolytic aromatic C-Br cleavage produces direct
Br-retaining cations** that BFS cannot model as [neutral+H]⁺. The CF₃ mols show different
missing peaks (integer bins, CF₃-containing fragments) mixed into the combined subclass stats.

---

### Pyrazolopyrazoles (n=28, DAG miss rate=63.0%)

**Molecules:** 3 mols (all in tar).
- mol_id=30481: `Cc1[nH]nc2cn[nH]c12` — unsubstituted methylpyrazolopyrazole (no halogen)
- mol_id=30601: `Cc1c(C)n2c(CBr)c(C)c(=O)n2c1=O` — mono-CH₂Br
- mol_id=30612: `Cc1c(CBr)n2c(CBr)c(C)c(=O)n2c1=O` — di-CH₂Br (two bromomethyl groups)

The two CH₂Br-containing mols are the dominant source of category A gaps. The pattern is
identical to the benzyl halides case: CH₂Br⁺ and CH₂Br-containing ring cations are produced
by heterolytic C-Br cleavage but cannot be generated as [neutral+H]⁺ by BFS.

Category B: m/z 348.9 (n=6, intensity 0.687) with a .9 decimal confirms a large Br-containing
fragment retained in DAG but underpredicted by the model.

mol_id=30481 (no Br) contributes genuine missing peaks from smaller aromatic ring fragments
produced at high CE (depth cascade problem like thienothiophenes).

**Mixed: halide-retaining cations (for CH₂Br mols) + cascade depth (for the non-halide mol).**

---

### Phenyl Sulfoxides (n=162, DAG miss rate=59.0%)

**12 mols total; 11 in tar.** Missing: mol_id=52128 (diphenyl sulfoxide PhS(O)Ph, 1 spectrum).
All 11 in-tar molecules are **halogenated phenyl sulfoxides** (Br and/or Cl substituents on phenyl rings).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note        |
|---------|--------|-----------|-------------|
| 69.0    | 33     | 0.340     | no halogen  |
| 82.0    | 23     | 0.201     | no halogen  |
| 183.0   | 18     | 0.163     | no halogen  |
| 95.0    | 16     | 0.198     | no halogen  |
| **254.9** | **15** | **0.614** | **Cl/Br mass defect** |
| 192.0   | 12     | 0.686     | no halogen  |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | Note             |
|---------|--------|----------------|------------------|
| **186.9** | **13** | **0.660** | **Br mass defect** |
| 105.0   | 13     | 0.373     |                  |
| **370.9** | **12** | **0.668** | **Br mass defect** |
| 57.1    | 10     | 0.333     |                  |
| 96.0    | 9      | 0.673     |                  |
| **225.9** | 9      | 0.100     | **Br/Cl mass defect** |

#### Root cause

**Real gap (halide-retaining direct cations) + model weakness.**

The .9 m/z bins — both in A (254.9) and B (186.9, 370.9, 225.9) — reflect ³⁵Cl (offset +0.969) and
⁷⁹Br (offset +0.918) mass defects: **halide-retaining direct cations** from heterolytic C−X cleavage,
which BFS cannot generate (produces only [neutral+H]⁺). Confirmed: mol_id=52125
([O-][S+](c1cc(Cl)cc(Cl)c1O)c1cc(Cl)cc(Cl)c1O, a tetrachlorodiphenyl sulfoxide) contributes
m/z 254.9 peaks in 5/14 spectra.

Category A integer-bin peaks (69.0, 82.0, 95.0, 192.0) are a separate gap: aliphatic and ring fragments
likely from sulfoxide bridge cascade fragmentation requiring >3 sequential cuts.

---

### Imidazopyrazines (n=209, DAG miss rate=58.0%)

**13 mols; 12 in tar.** Missing: mol_id=47557 (a thienobenzazepine misclassified here). Nearly all
12 in-tar mols are **heavily halogenated**: Br+I (mol_id=274, 351), Cl+I (34969), Cl+Br (35000,
35078, 35079), di-Br+ester (19457, 33753), or mono-Br+amino (41243, 41397).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| **64.0**  | **68** | **0.453** | unexplained (not .9 bin) |
| 52.0    | 47     | 0.233     | unexplained |
| **141.9** | **45** | **0.599** | Br/I mass defect |
| 90.0    | 45     | 0.336     | ring fragment (no halogen) |
| **129.9** | **42** | **0.231** | Br mass defect: [C₄H₃⁷⁹Br]⁺ = 129.942 |
| 136.1   | 32     | 0.331     | ring fragment |
| 63.0    | 29     | 0.434     | small ring fragment |
| **189.9** | **24** | **0.460** | Br/I mass defect |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | Note |
|---------|--------|----------------|------|
| **289.9** | **21** | **0.790** | Br-retaining large fragment |
| **319.9** | **18** | **0.652** | Br-retaining large fragment |
| 65.0    | 38     | 0.185     |      |
| 91.0    | 28     | 0.310     |      |
| **142.9** | **20** | **0.111** | Br mass defect |

#### Root cause

**Mixed: halide-retaining direct cations (absent from DAG) + model weakness on large Br-containing
fragments (in DAG, underpredicted).**

- **Category A .9 bins** (141.9, 129.9, 189.9): Br and I mass-defect direct cations from ring cleavage.
  m/z 129.9 = [C₄H₃⁷⁹Br]⁺ = 129.942. m/z 141.9 ≈ [H₂N¹²⁷I]⁺ = 141.915 (iodamine ring-opening
  cation). Same mechanism as benzyl halides and triazolopyrazines, extended to Br+I combinations.
- **Category B .9 bins** (289.9 at 0.790, 319.9 at 0.652): large Br-retaining fragments ARE in DAG
  but severely underpredicted — model weakness on multi-halogen compounds. m/z 289.9 tentatively
  = [M+H−EtOH]⁺ from dibromo ester mol_id=19457 ([M+H]⁺ = 335.9, loss of EtOH = 46.042 Da).
- **m/z 64.0 and 52.0** (68 and 47 spectra, not .9 bins): unexplained. Same anomalous 64.0 peak
  appears in imidazo[4,5-c]pyridine ribonucleosides (67.6% miss rate), suggesting a shared imidazo-
  ring origin. Possibly SO₂⁺ from DMSO solvent contamination (63.962 → 64.0 bin).

---

### Alpha-Keto Acids and Derivatives (n=32, DAG miss rate=57.6%)

**5 mols; all in tar.** One mol has a Br substituent: mol_id=22647 (COC(=O)C(=O)CBr = methyl
bromopyruvate). The other 4 are unsubstituted alpha-keto acids (pyruvate, cyclohexyl pyruvate,
mercaptopyruvate, sulfopyruvate).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Assignment |
|---------|--------|-----------|------------|
| **92.9** | **28** | **0.753** | [CH₂⁷⁹Br]⁺ = 92.926 → 92.9 bin |
| 90.9    | 4      | 0.109     | [C⁷⁹Br]⁺ = 90.918 → 90.9 bin  |
| 91.9    | 2      | 0.057     | [CH⁷⁹Br]⁺ = 91.926 → 91.9 bin |

#### Root cause

**Real gap — Br-retaining direct cations.** Confirmed: m/z 92.9335 observed in mol_id=22647
spectra (actual [CH₂⁷⁹Br]⁺ = 92.9268, within ~7 mDa). All three category A bins (.9 offset) are
the C⁷⁹Br⁺ family — same mechanism as benzyl halides and triazolopyrazines.

The miss rate is driven entirely by the single brominated mol (22647) despite only 5 total mols.
Category B (148.9 at 0.356, 119.9 at 0.063) are larger Br-containing fragments in DAG but
underpredicted. Non-Br mols (pyruvate, etc.) contribute little to the miss rate.

---

### Aryl Iodides (n=315, DAG miss rate=53.2%)

**30 mols; 29 in tar.** Missing: mol_id=52605 (diphenylene iodonium salt, permanent I⁺ cation —
same unsupported charge model as berberine). The 29 in-tar mols are predominantly **iodo-azoles**:
iodoimidazoles (CC(C)Cn1cc(I)cn1 type), iodopyrazoles, iodotriazoles; plus a few iodoxybenzene
(PhI=O) and hypervalent iodine derivatives.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| **136.0** | **27** | **0.607** | heteroaryl ring cation (I-free) |
| **328.9** | **24** | **0.256** | I mass defect |
| 69.0    | 23     | 0.113     | furanyl cation |
| 105.0   | 20     | 0.371     | ring fragment |
| **94.0** | **20** | **0.431** | ring fragment |
| 55.0    | 20     | 0.240     | |
| **262.9** | **18** | **0.270** | I mass defect |
| **290.9** | **15** | **0.156** | I mass defect |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | Note |
|---------|--------|----------------|------|
| 81.0    | 29     | 0.139     |      |
| **230.9** | **14** | **0.457** | I mass defect — in DAG |
| **105.0** | **18** | **0.340** |      |
| 144.0   | 13     | 0.308     |      |
| 90.0    | 10     | 0.243     |      |

#### Root cause

**Real gap (I-retaining direct cations) + cascade depth for heteroaryl ring fragments.**

The .9 m/z bins in category A (328.9, 262.9, 290.9) reflect ¹²⁷I mass defect (+0.904) from
**iodine-retaining direct cations** after C-I heterolytic cleavage — same mechanism as benzyl
halides and triazolopyrazines, extended to iodine. Category B also has an I-retaining fragment
(230.9 at 0.457) IS in DAG but underpredicted.

m/z 136.0 (27 spectra, 0.607): a high-intensity heteroaryl ring cation (I-free) across many
different iodo-azole mols. Assignment uncertain given the structural diversity of 30 mols, but
consistent with the imidazolyl or pyrazolyl ring cation produced after I ejection + alkyl-chain
rearrangement from N-alkyl iodoimidazoles.

Missing mol_id=52605: diphenyleneiodonium [I⁺(Ph)₂], a permanent cation — BFS cannot model
preformed ionic species (same limitation as berberine).

---

### Organobromides (n=62, DAG miss rate=46.7%)

**62 spectra; all mols contain Br.** Aromatic or aliphatic organobromine compounds — pharmaceuticals, agrochemicals, and natural products with C-Br bonds.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 92.9    | 12     | 0.620     | .9 bin → [CH₂⁷⁹Br]⁺ or [C₂H₂⁷⁹Br]⁺ Br-retaining direct cation |
| 90.9    | 6      | 0.109     | .9 bin → [⁷⁹BrC₂H₂]⁺ direct cation |
| 53.0    | 10     | 0.112     | [C₄H₅]⁺ — aromatic cascade |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 124.0   | 10     | 0.640          | 0.013          |
| 106.9   | 11     | 0.283          | 0.013          |
| 53.0    | 7      | 0.139          | 0.020          |
| 135.0   | 2      | 0.290          | 0.016          |

#### Root cause

**Halide direct cations (.9 bins) + model weakness (124.0).**

The .9 fractional bins 92.9 and 90.9 are Br-retaining direct cations ([C₂H₂⁷⁹Br]⁺ and [⁷⁹Br+allyl]⁺) — BFS cannot generate these without modeling direct C-Br heterolytic cleavage. The 124.0 bin (0.640 intensity, pred 0.013) is a large model weakness: a Br-containing fragment correctly in DAG (BFS generates [C₂H₄Br + H]⁺ or similar) but the model assigns near-zero probability. The class is small (n=62) but the Br mass defect fingerprint makes the mechanism unambiguous.

---

## Cause 4 — Model Weakness (fragment in DAG, predicted ≈0)

The fragment **is** enumerated by BFS and present in the DAG, but the model assigns it near-zero intensity (**category B**). This is not a BFS limitation — it is a training / generalization failure, and the fix is model-side.

Two GNN-specific failure modes recur:

1. **Repetitive chains → indistinguishable embeddings.** Every internal unit of a polyisoprenyl / long aliphatic chain has an identical k-hop neighborhood, so a depth-k GNN cannot tell chain positions apart; supervision from different positions cancels and the prediction collapses to ≈0.
2. **Generic small cations average to ≈0.** Fragments like [C₇H₇]⁺ or [C₇H₅O]⁺ appear across thousands of molecular contexts, most of which do not produce them at high intensity, so the model learns an ≈-zero mean.

**Fix levers:** focused retraining on the affected classes; position encoding on chain atoms; isoprenyl-repeat count features. All are feature/training changes, not architecture changes.

### Organic Phosphoric Acids and Derivatives (n=58, DAG miss rate=75.0%)

**Molecules:** 58 spectra, diverse organophosphate structures.

#### Missing peaks

**Category A (absent from DAG):**

| m/z bin | n_spectra | mean intensity | likely identity |
|---------|-----------|----------------|-----------------|
| 69.1    | 28        | 0.730          | C₅H₉⁺ or C₄H₅O⁺ — aliphatic chain fragment |
| 83.0    | 18        | 0.433          | C₆H₁₁⁺ or C₅H₇O⁺ |
| 55.1    | 14        | 0.299          | C₄H₇⁺ — aliphatic |
| 53.0    | 10        | 0.082          | C₄H₅⁺ |

**Category B (in DAG, under-predicted):**

| m/z bin | n_spectra | mean true intensity | mean pred intensity |
|---------|-----------|---------------------|---------------------|
| 99.0    | 22        | 0.570               | 0.006               |

m/z 99.0 is the **[M+H−H₃PO₄]⁺** product ion class (loss of 98 Da = H₃PO₄), a well-known
diagnostic neutral loss for organophosphates. BFS generates this fragment but the model
badly under-predicts it.

**Finding:** Two separate problems:
1. **Model weakness (B):** Neutral loss of H₃PO₄ (98 Da) is in the DAG but the model has not
   learned to assign it high weight. This is likely due to underrepresentation of organophosphates
   in training.
2. **DAG gap (A):** m/z 69.1/83.0/55.1 are aliphatic chain fragments from the ester side-chain —
   likely the organic tail of the phosphate ester producing standard alkyl carbocations (e.g.,
   C₅H₉⁺ = 69.071) that BFS does not generate. Low sample size (n=58) limits confidence.

---

### Lignan Glycosides (n=115, DAG miss rate=75.8%)

**Molecules:** 19 FT [M+H]+ mols, 15 in tar. 4 mols absent from tar → 12
artifact spectra (contributes fractional m/z bins like 60.7, 53.6, 56.4 in
category A — likely low-level noise misclassified by the pre-fix bug).

**Main finding is category B, not category A:** for the 15 in-tar mols, the
phenylpropanoid aglycone fragment series is **in the DAG but the model predicts
~0 intensity**:

| m/z bin | n_spectra | mean true int | mean pred int |
|---------|-----------|---------------|---------------|
| 173.1   | 38        | 0.505         | 0.000         |
| 145.1   | 32        | 0.591         | 0.008         |
| 131.0   | 36        | 0.168         | 0.000         |
| 115.1   | 29        | 0.479         | 0.007         |
| 91.1    | 35        | 0.282         | 0.006         |

These are the standard lignin/phenylpropanoid aglycone series (m/z 91, 105, 115,
117, 131, 145, 173, 189) produced after the glycan moiety is lost at high CE.

Root cause: **model weakness, not a DAG gap**. The DAG contains these fragments,
but the model has not learned to assign them high probability. Most likely cause:
lignan glycosides are large (MW 400–700+) with structurally diverse sugar moieties;
training signal for the aglycone fragments is diluted and the "glycan loss →
aglycone fragmentation" cascade at high CE is not generalized.

**Model training/generalization issue. Not RDA.**

---

### Organic Phosphoramides (n=80, DAG miss rate=59.7%)

**5 mols; all in tar.** Includes organofluorophosphates (nerve agent analogs: CC(C)NP(=O)(F)NC(C)C,
CN(C)P(=O)(F)N(C)C) and a cyclic phosphoramide with Br/Cl substituents (mol_id=48686:
O=P1(NCCBr)OCCCN1CCCl).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity |
|---------|--------|-----------|
| 128.0   | 18     | 0.628     |
| 73.1    | 16     | 0.372     |
| 60.0    | 14     | 0.605     |
| 63.0    | 14     | 0.152     |
| 135.1   | 12     | 0.253     |
| 76.0    | 12     | 0.712     |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **135.1** | **36** | **0.866** |
| 92.0    | 16     | 0.470     |
| 110.0   | 12     | 0.108     |
| 99.0    | 6      | 0.428     |

#### Root cause

**Model weakness dominant, with genuine BFS gaps for small P-containing ions.**

The dominant signal is **category B: m/z 135.1 in 36 spectra at 0.866 intensity** — one of the strongest
model failures seen across all subclasses. Verified to be m/z 135.069 in actual spectra. Two routes
coincidentally hit the same bin:
- Fluorophosphoramides (mols 7133, 21145): **[M+H−HF]⁺** (loss of HF, 20.006 Da from the P−F bond)
- Tris-amide mol 21146 (CN(C)P(=O)(N(C)C)N(C)C): **[(NMe₂)₂P=O]⁺** from loss of dimethylenimine
  (CH₂=NMe, 43.042 Da); C₄H₁₂N₂OP⁺ = 135.069 ✓

Both generate m/z 135.07 which IS in DAG (correctly fragmented by BFS) but model predicts ≈0.
Model has not learned phosphoramide fragmentation chemistry.

Category A small ions (73.1, 60.0, 128.0): P-containing cascade fragments or P−F cleavage products;
BFS cannot model P−F heterolysis to give PF-retaining ionic fragments.

---

### Pyranopyridines (n=108, DAG miss rate=58.8%)

**9 mols; 8 in tar.** Missing: mol_id=9141 (Br-containing fused ring compound, 1 spectrum).
Complex fused heterocyclic structures (pyranothiazolopyridines, morpholine-containing variants).

#### Missing peaks (category A — absent from DAG)

Top 5: 189.0 (25 spectra, 0.224), 203.1 (22, 0.102), 118.1 (21, 0.153), 169.1 (21, 0.126),
155.1 (18, 0.408).

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **131.1** | **24** | **0.588** |
| 77.0    | 19     | 0.389     |
| 114.0   | 19     | 0.211     |
| 182.1   | 17     | 0.294     |
| **89.0**  | **16** | **0.411** |
| **216.1** | **14** | **0.595** |
| **105.0** | **14** | **0.467** |

#### Root cause

**Model weakness dominant.** Category B contains the highest-intensity fragments (131.1 at 0.588,
216.1 at 0.595, 89.0 at 0.411), all in DAG but model predicts ≈0. Category A fragments (189.0,
155.1) are genuine BFS gaps from complex heterocyclic ring-opening sequences.

---

### Sulfur Mustard Compounds (n=28, DAG miss rate=58.3%)

**2 mols; 1 in tar.** Only mol_id=48686 (O=S(=O)(CCBr)CCBr = bis(2-bromoethyl)sulfone) contributes.
Despite classification, this is a **sulfone** (not the classic bis(2-chloroethyl)sulfide mustard). 1 mol
missing from tar (artifact: ~3 spectra).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity |
|---------|--------|-----------|
| 278.0   | 16     | 0.092     |
| 169.1   | 4      | 0.069     |
| 126.9   | 4      | 0.092     |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | Assignment |
|---------|--------|----------------|------------|
| **278.9** | **10** | **0.693** | [M+H]⁺ (⁷⁹Br₂, 278.870 → 278.9 bin) |
| **154.9** | **10** | **0.226** | [M+H−CH₂CH₂Br]⁺ (Br-retaining) |

#### Root cause

**Model weakness (Br-containing fragments in DAG, underpredicted) + minor real gap.**

m/z 278.9 is the **[M+H]⁺ precursor itself** (⁷⁹Br₂: C₄H₈Br₂O₂S + H = 278.870 → 278.9 bin).
It IS in the DAG as the root node, but model predicts ≈0 intensity — model does not retain the
precursor ion at any CE. m/z 154.9 = [M+H−CH₂CH₂Br]⁺ (loss of 2-bromoethyl, 123.9 Da),
also Br-retaining, also in DAG but underpredicted.

Category A m/z 278.0 (16 spectra, low 0.092 intensity) is likely the ¹²C₄H₈⁷⁹Br⁸¹BrO₂S isotope
contribution or an unrelated neutral loss product.

---

### Auronols (n=21, DAG miss rate=56.4%)

**1 mol; in tar.** Only mol_id=47941 (O=C1c2c(O)cc(O)cc2OC1(O)Cc1ccc(O)cc1 =
3-hydroxy-5,7-dihydroxy-4'-hydroxybenzyl chromenone, a flavanone-3-ol/auronol). No halogen
substituents — the .9 bin peaks in category A (229.9, 222.9, 240.9) suggest additional molecules
classified here via ClassyFire alternative_parents that were not captured in the subklass query.

#### Key peaks

Category A: **215.1** (11 spectra, 0.514), 148.0 (3, 0.261), 106.1 (3, 0.307), 88.1 (3, 0.271),
  240.9 (3, 0.682), 222.9 (3, 0.412) — .9 bins suggest Br-containing mols in this group.

Category B: **149.0** (13 spectra, 0.529!), **115.1** (5, 0.508!), 105.0 (4, 0.378).

#### Root cause

**Model weakness dominant** (category B at high intensity) **+ possible halide direct cations** for
.9-bin category A entries from Br-containing compounds in the broader auronol group.

m/z 149.0 (13 spectra, 0.529) = C₈H₅O₃⁺ (149.025, auronol B-ring retaining ion or phthalic
anhydride-type acylium) — IS in DAG but model predicts ≈0. m/z 215.1 (11 spectra, 0.514) absent
from DAG: cascade fragmentation of the chromone ring.

---

### 1,3-Dioxanes (n=25, DAG miss rate=54.8%)

**10 mols; 6 in tar; 70 spectra; nce=nan=0%.** Cyclic acetal structures: neopentyl glycol acetals of cyclohexanones/benzaldehydes (mols 9131/9132/9367) plus complex bicyclic acetals (mol 528). Clean data.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 77.0 | 8 | 0.228 | Exact 77.038 = [C₆H₅]⁺ — phenyl cation (mol 9367 with phenyl); cascade depth |
| 95.0 | 8 | 0.246 | Exact 95.013 = [C₅H₃O₂]⁺ — dioxane ring fragment |
| 105.0 | 8 | 0.156 | Exact 105.044 = [C₈H₅]⁺/[C₇H₅O]⁺ — aromatic cascade |
| 115.1 | 5 | 0.094 | Aromatic cation cascade |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 83.1 | 9 | 0.632 | 0.009 |
| 69.1 | 7 | 0.264 | 0.024 |
| 129.1 | 4 | 0.170 | 0.000 |

#### Root cause

**Model weakness dominant + cascade depth (phenyl mols).**

83.1 = [C₅H₇O]⁺ at 83.049 (confirmed by exact mass) is the acetal ring-opening fragment — the cyclic acetal C-O bond cleaves to give a cyclic oxocarbenium ion. It appears in 9 of 70 spectra at 0.632 intensity and is in the DAG, but the model predicts only 0.009. This is the primary actionable gap. Category A peaks 77.0 and 105.0 arise from the phenyl group in mol 9367 (1-phenyl-1,3-dioxane) via aromatic cascade (phenyl cation series, depth >3). 4 mols missing from tar limit the class size.

---

### Medium-Chain Keto Acids and Derivatives (n=51, DAG miss rate=50.0%)

**18 mols; 15 in tar.** 3 missing mols have 0 FT [M+H]+ spectra — no artifact. Class includes
2-oxogluconic acid (mol_id=43884), levulinate variants, and acetoacetate-related structures.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 194.2   | 6      | 0.113     | satellite to 196.2 |
| **196.2** | **6** | **1.000** | **base peak** — spurious (see below) |
| 197.2   | 6      | 0.054     | satellite |
| 153.0   | 4      | 0.073     | |
| 154.0   | 4      | 0.140     | |
| 167.0   | 4      | 0.244     | |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| **99.0** | **12** | **0.689** |
| **142.1** | **8** | **0.334** |
| **141.1** | **5** | **0.419** |
| 274.1   | 5      | 0.309     |

#### Root cause

**Model weakness dominant + data quality issue.**

**m/z 196.2 anomaly**: confirmed in mol_id=43884 (2-oxogluconic acid, [M+H]⁺ = 195.050). The base
peak at 196.170 Da in the product spectrum exceeds the precursor m/z — it cannot be a true CID
fragment. Actual m/z measured as 196.1695 (consistent across 6 spectra) suggests a **co-isolated
compound** ([C₁₂H₂₂NO]⁺ = 196.174 is the nearest formula match) contaminating the MS2 isolation
window. A data quality artifact from the NIST23 acquisition.

**Category B: m/z 99.0 (12 spectra, 0.689)** = [C₅H₇O₂]⁺ = 99.044 (levulinyl/β-keto ester cation)
— IS in DAG but model predicts ≈0. m/z 142.1 and 141.1 are also in DAG but underpredicted.
Model has not learned the keto-acid fragment pattern (loss of CO₂ then further dehydration).

---

### Quinone and Hydroquinone Lipids (n=412, DAG miss rate=19.2%)

**21 mols in tar (15 missing, all with 0 FT [M+H]+ spectra — no artifact).** Tocopherols,
tocotrienols, plastoquinones, and ubiquinones: a benzoquinone or chromanol ring fused to a
long polyisoprenyl (C15–C50) side chain.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 77.0    | 31     | 0.322     | [C₆H₅]⁺ phenyl cation — aromatic cascade |
| 79.1    | 30     | 0.431     | [C₆H₇]⁺ |
| 103.1   | 29     | 0.340     | [C₈H₇]⁺ |
| 105.1   | 21     | 0.096     | [C₈H₉]⁺ |
| 89.0    | 19     | 0.138     | [C₇H₅]⁺ |
| 128.1   | 19     | 0.090     | [C₁₀H₈]⁺ |
| 183.0   | 18     | 0.137     | [C₁₂H₈P]⁺ or aromatic cascade |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 83.0    | 73     | 0.240          |
| 115.1   | 69     | 0.307          |
| 107.0   | 58     | 0.233          |
| 109.1   | 55     | 0.188          |
| 161.1   | 49     | 0.337          |
| 128.1   | 48     | 0.259          |
| 121.1   | 46     | 0.147          |
| 105.0   | 45     | 0.230          |

#### Root cause

**Model weakness dominant — GNN failure on repetitive isoprenyl chains.**

Category B has ~2× the spectra count of category A and higher intensities. The category B
series (83.0, 107.0, 109.1, 121.1, 128.1, 161.1) are isoprenoid side-chain fragments
([C₅H₇O]⁺, [C₇H₇O]⁺, [C₈H₉O]⁺) produced by BFS cuts of the polyisoprenyl chain — these
are correctly in the DAG but the model assigns ~0 probability. Category A (77.0, 79.1,
103.1) are small aromatic cation cascade fragments from the chromanol/quinone ring system
(minor cascade depth component, n=19–31).

Two GNN-specific mechanisms explain the model failure:

1. **Repetitive chain → indistinguishable node embeddings.** For a polyisoprenyl chain
   (ubiquinone-10 has a C50 side chain: 10× isoprene units), every internal isoprene unit
   has an identical k-hop neighborhood. A GNN with depth k cannot distinguish the 3rd from
   the 7th isoprenyl unit — they produce the same embedding. The model has no signal to
   assign different fragmentation probabilities to different chain positions; supervision
   from different positions cancels during training and the result is ≈0 for all.

2. **Generic aliphatic embeddings lack discriminative signal.** Isoprenyl fragments like
   C₅H₇O⁺ (83 Da) or C₆H₉⁺ (81 Da) appear as fragments in hundreds of molecular contexts
   (terpenes, fatty acids, steroids), most of which don't produce them at high intensity.
   The model learns an average ≈0 across contexts. Aromatic fragments (C₇H₇⁺, C₉H₇⁺)
   have highly distinctive ring structures that produce discriminative GNN embeddings —
   aliphatic chain fragments do not.

Fix: **position encoding on chain atoms** (encode distance from ring system into atom
features) so the GNN can distinguish chain position; and/or **isoprenyl-repeat fragment
features** that count the number of isoprene subunits in a fragment. Both are atom/fragment
feature additions, not architectural changes.

---

### Saccharolipids (n=466, DAG miss rate=17.3%)

**95 mols in tar (43 missing).** Terpenoid glycosides and flavonoid glycolipids — an
aglycone (terpenoid, flavonoid, or phenylpropanoid) esterified or O-glycosylated to a sugar
moiety (glucose, maltose, trehalose).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 109.0   | 39     | 0.193     | [C₇H₅O]⁺ or [C₈H₉]⁺ |
| 65.0    | 26     | 0.168     | [C₅H₅]⁺ cyclopentadienyl |
| 109.1   | 21     | 0.219     | [C₈H₉]⁺ |
| 53.0    | 18     | 0.242     | [C₄H₅]⁺ |
| 95.1    | 13     | 0.452     | [C₇H₁₁]⁺ or [C₆H₇O]⁺ |
| 285.0   | 13     | 0.149     | aglycone fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 91.1    | 87     | 0.310          |
| 97.0    | 63     | 0.173          |
| 93.1    | 59     | 0.322          |
| 69.0    | 57     | 0.121          |
| 107.1   | 55     | 0.271          |
| 121.1   | 55     | 0.268          |
| 79.1    | 48     | 0.406          |
| 135.1   | 46     | 0.276          |

#### Root cause

**Model weakness dominant — same GNN failure as quinone lipids, compounded by structural
diversity.**

Category B dominates heavily (87 vs 39 max spectra in A). The category B series (79.1,
91.1, 93.1, 97.0, 107.1, 121.1, 135.1) are isoprenoid/aliphatic chain cations from the
terpenoid aglycone, all in DAG but predicted ≈0. The same two GNN mechanisms apply as in
quinone/hydroquinone lipids: repetitive chain units produce indistinguishable node
embeddings, and generic aliphatic fragments accumulate ≈0 supervision across diverse
molecular contexts. Here the problem is compounded further: saccharolipids have both a
repetitive terpenoid/aliphatic aglycone AND a variable sugar moiety (glucose, maltose,
trehalose, etc.) — the sugar context changes across mols while the chain fragment is the
same, making it even harder for the model to generalize. Category A (65.0, 95.1) is a
minor cascade depth component.

---

### Furanoquinolines (n=58, DAG miss rate=48.1%)

**58 spectra; low tar coverage.** Fused furan-quinoline N-heterocycles — common in plant alkaloids (rutaecarpine-type). [M+H]⁺ typically 200–260 Da.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 104.0   | 8      | 0.731     | Highest intensity absent peak; [C₇H₄O]⁺ or furanyl cation |
| 77.0    | 9      | 0.417     | [C₆H₅]⁺ phenyl cation — cascade from quinoline ring |
| 156.0   | 10     | 0.254     | [C₁₀H₈NO]⁺ quinoline ring fragment |
| 128.1   | 5      | 0.541     | [C₁₀H₈]⁺ naphthalene/azanaphthalene |
| 51.0    | 5      | 0.352     | [C₄H₃]⁺ small aromatic |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 227.1   | 10     | 0.554          | 0.001          |
| 199.1   | 6      | 0.425          | 0.037          |
| 216.1   | 12     | 0.400          | 0.015          |
| 105.0   | 10     | 0.200          | 0.019          |
| 79.0    | 6      | 0.305          | 0.000          |

#### Root cause

**Mixed: cascade depth + model weakness (dominant).**

The cascade series (51.0, 77.0, 128.1) absent from DAG reflects the furan ring-opening pathway at >3 sequential cuts — same mechanism as thienothiophenes but with O instead of S. However, category B is more severe: 227.1 (0.554, pred 0.001) and 216.1 (0.400, pred 0.015) are mid-size quinoline-type fragments correctly generated by BFS but assigned near-zero probability. The extremely high absent intensity at 104.0 (0.731) suggests a direct ring-contraction or rearrangement product that BFS cannot generate.

---

### Organic Sulfuric Acids and Derivatives (n=36, DAG miss rate=47.8%)

**36 spectra; ~4 mols with halogens.** Sulfate esters (ArOSO₃H) and sulfonic acids — common in drug metabolites, detergents, and environmental samples.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 94.9    | 9      | 0.507     | .9 bin → [C₂H₂³⁵Cl₂]⁺ organochloride direct cation |
| 130.9   | 10     | 0.433     | .9 bin → [C₂H₂³⁵Cl₃]⁺ trichloroethyl cation |
| 60.0    | 4      | 0.645     | [C₂H₄S]⁺ or SO₂⁺ sulfonate fragment |
| 106.0   | 5      | 0.251     | Aromatic sulfonate fragment |
| 96.0    | 4      | 0.178     | [SO₃+H₂O]⁺ / aliphatic fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 141.0   | 6      | 0.802          | 0.022          |
| 80.0    | 12     | 0.628          | 0.011          |
| 151.0   | 5      | 0.607          | 0.020          |
| 61.0    | 6      | 0.169          | 0.017          |
| 98.0    | 4      | 0.310          | 0.025          |

#### Root cause

**Mixed: halide direct cations (.9 bins) + model weakness dominant (integer bins).**

Category A .9 bins (94.9, 130.9) are organochloride direct cations ([C₂H₂Cl₂]⁺, [C₂H₂Cl₃]⁺) from Cl₃-substituted sulfate ester mols — BFS cannot generate these halide-retaining fragments. However, the integer category B bins are the bigger problem: 80.0 (SO₃⁺ from sulfonate loss) and 141.0 (0.802 intensity!) are correctly in the DAG but predicted ≈0. The model fails to learn the SO₃ loss pathway despite its consistent appearance across sulfonate mols.

---

### Azacyclic Compounds (n=36, DAG miss rate=47.4%)

**36 spectra; catch-all N-heterocycle class dominated by piperazine/piperidine-type.** Saturated or partially saturated N-containing rings with diverse substituents.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 60.0    | 9      | 0.720     | Very high intensity! [C₂H₆NO]⁺ or [C₃H₆N]⁺ azacyclic fragment |
| 88.0    | 9      | 0.219     | [C₃H₆NO₂]⁺ or N-formyl fragment |
| 61.0    | 6      | 0.101     | Small N-containing fragment |
| 74.0    | 6      | 0.076     | [C₃H₈NO]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 202.1   | 9      | 0.387          | 0.024          |
| 146.1   | 9      | 0.399          | 0.013          |
| 118.1   | 12     | 0.295          | 0.006          |
| 72.1    | 2      | 0.354          | 0.043          |
| 85.1    | 6      | 0.127          | 0.002          |

#### Root cause

**DAG gap for small N-containing fragments + model weakness.**

The 60.0 peak at 0.720 intensity absent from DAG (9/36 spectra) is a small azacyclic fragment — likely [C₂H₆NO]⁺ or [C₃H₆N]⁺ from piperazine ring-opening requiring sequential N−C bond cleavages. BFS generates these in principle but may miss the specific protonation site on the N atom leading to the correct mass. Category B shows large mid-size fragments (202.1, 146.1) in DAG but severely underpredicted. The broad catch-all taxonomy means these 36 spectra span many different ring systems, so each mechanism has few examples.

---

### Flavans (n=441, DAG miss rate=47.0%)

**441 spectra; 15+ mols in tar.** Flavan (2-phenylchroman) and flavanone-type compounds — polyphenols from fruit and bark extracts, including catechin-related scaffolds.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 95.0    | 18     | 0.389     | Chromene ring-opening fragment |
| 65.0    | 37     | 0.347     | [C₅H₅]⁺ cyclopentadienyl |
| 153.0   | 36     | 0.149     | [C₁₀H₉O]⁺ flavanyl ring fragment |
| 151.0   | 32     | 0.180     | [C₁₀H₇O]⁺ |
| 119.0   | 31     | 0.129     | [C₉H₇]⁺ or [C₇H₇O]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 147.0   | 86     | 0.152          | 0.021          |
| 105.0   | 62     | 0.309          | 0.021          |
| 79.0    | 27     | 0.384          | 0.015          |
| 123.0   | 26     | 0.316          | 0.019          |
| 165.1   | 61     | 0.124          | 0.012          |

#### Root cause

**Model weakness dominant.**

Category B is larger than category A: 147.0=[C₉H₇O₂]⁺ (86 spectra, 0.152, pred 0.021) is a flavanyl ring-opening fragment correctly in the DAG but systematically underpredicted. The 105.0 benzoyl cation (62 spectra) and 123.0 (26 spectra, 0.316) follow the same pattern seen in benzofuranones, flavonoids, and phenoxy compounds — the model fails to learn the C-ring retro-Diels-Alder fragmentation intensity at low CE. Category A integer bins (65.0, 153.0) reflect residual cascade depth gaps at high CE, but with 441 spectra the category B dominance is conclusive.

---

### Lignan Lactones (n=80, DAG miss rate=45.5%)

**80 spectra; ~8 mols in tar.** Lactone-containing lignans (podophyllotoxin, deoxypodophyllotoxin) — aryl-substituted cyclohexane lactones with methylenedioxy groups.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 145.1   | 35     | 0.605     | [C₉H₉O₂]⁺ trimethoxybenzyl or arylmethyl cation |
| 173.1   | 29     | 0.495     | [C₁₁H₉O₂]⁺ methylenedioxy-phenyl fragment |
| 131.0   | 20     | 0.333     | [C₉H₇O]⁺ |
| 175.1   | 8      | 0.264     | Aromatic fragment |
| 77.0    | 7      | 0.223     | [C₆H₅]⁺ phenyl |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 115.1   | 19     | 0.640          | 0.000          |
| 219.1   | 11     | 0.640          | 0.011          |
| 129.1   | 14     | 0.454          | 0.000          |
| 91.1    | 13     | 0.453          | 0.000          |
| 128.1   | 16     | 0.372          | 0.000          |

#### Root cause

**Model weakness dominant — same failure mode as lignan glycosides.**

Category B completely dominates: 115.1 (0.640, pred 0.000), 219.1 (0.640, pred 0.011), 129.1 (0.454, pred 0.000), 91.1 (0.453, pred 0.000), and 128.1 (0.372, pred 0.000) are all correctly in the DAG but the model assigns zero probability. These are the aglycone aromatic fragments of the podophyllotoxin/deoxypodophyllotoxin scaffold — the cyclohexane lactone ring opens to produce arylmethyl and methylenedioxy-phenyl cations, and the DAG correctly generates these. The GNN fails because the dense aryl-cyclohexane-lactone embedding is non-discriminative across the many diverse scaffolds that produce the same small aromatic fragments. This is identical to the lignan glycoside failure; combining both subclasses would strengthen the signal for targeted retraining.

---

### Phenoxy Compounds (n=396, DAG miss rate=44.1%)

**396 spectra; 20+ mols in tar.** Aryl ether compounds (Ph-O-R) — herbicides (2,4-D type), drug metabolites, and natural O-aryl ethers.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 105.0   | 72     | 0.352     | Benzoyl/phenoxy fragment [C₇H₅O]⁺ — 72 spectra affected! |
| 109.0   | 15     | 0.603     | Highest intensity in cat A; [C₈H₉]⁺ or [C₆H₅F]⁺ |
| 50.0    | 52     | 0.183     | [C₄H₂]⁺ — small aromatic |
| 152.1   | 35     | 0.321     | Larger phenoxy fragment |
| 153.1   | 27     | 0.194     | [C₁₀H₉O]⁺ |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity | pred_intensity |
|---------|--------|----------------|----------------|
| 97.0    | 32     | 0.684          | 0.005          |
| 127.0   | 30     | 0.661          | 0.000          |
| 134.0   | 25     | 0.590          | 0.000          |
| 219.1   | 27     | 0.516          | 0.000          |
| 77.0    | 28     | 0.546          | 0.012          |

#### Root cause

**Model weakness dominant — high-intensity category B peaks across diverse phenoxy scaffolds.**

Category B shows four peaks at ≥0.516 intensity predicted at 0.000–0.005: 97.0 (0.684), 127.0 (0.661), 77.0 (0.546), and 219.1 (0.516). These are correctly generated by BFS (O-ether cleavage products and ring-retaining fragments) but the model cannot assign meaningful probability across the diverse set of phenoxy substituents. The 105.0=[C₇H₅O]⁺ benzoyl/phenoxy cation absent in 72 spectra is a genuine cascade depth gap — the ether oxygen prevents direct retro-fragmentation. With n=396 spectra and consistent high-intensity failures, phenoxy compounds are a strong target for focused model retraining.

---

## Cause 5 — Doubly-Charged Fragment Ions

BFS emits only singly-charged **[frag + H]⁺**. Large multi-basic lipids (gangliosides, sphingolipids, glycerolipids; MW 700–1800+) carry two protons and produce **[frag + 2H]²⁺**, which land at **.4–.7 fractional m/z bins**. Confirmed by FT exact masses. Structurally impossible for the current singly-charged output model to generate.

### Glycosphingolipids (n=32, DAG miss rate=14.3%)

**27 mols in tar (9 missing).** Gangliosides and cerebrosides — ceramide backbone with one
to several sugar units (glucose, galactose, sialic acid/NeuAc). MW ranges from ~700 (simple
cerebrosides) to ~1800+ (polysialogangliosides). Confirmed examples: GM3/GM1-type
gangliosides (mol_id=15728, [M+H]⁺=991.7), sulfatide (mol_id=15729, [M+H]⁺=864.6).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 864.5   | 6      | 0.162     | **doubly-charged fragment ion** |
| 625.4   | 5      | 0.398     | **doubly-charged fragment ion** |
| 626.4   | 4      | 0.089     | **doubly-charged fragment ion** |
| 785.7   | 4      | 0.274     | **doubly-charged fragment ion** |
| 865.6   | 3      | 0.253     | **doubly-charged fragment ion** |
| 311.2   | 3      | 0.140     | singly-charged aglycone fragment |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 265.3   | 13     | 0.114          |
| 283.3   | 9      | 0.230          |
| 252.3   | 9      | 0.123          |
| 766.7   | 6      | **0.747**      |
| 784.7   | 5      | **0.595**      |
| 97.0    | 7      | 0.318          |
| 91.0    | 7      | 0.125          |

#### Root cause

**Doubly-charged fragment ions (category A) + model weakness for large fragment ions (category B).**

Confirmed by direct spectrum inspection: mol_id=15728 ([M+H]⁺=991.7 Da, GM3-type
ganglioside) produces abundant fragment ions at 490.5, 502.5, 519.5–522.5, 538.5,
682.6–683.6, 700.6 — all with .5–.6 fractional parts indicating [frag+2H]²⁺. These are
singly-charged glycan-chain fragments (~980–1040 Da) that carry 2 protons from the
multi-basic sphingolipid backbone. BFS generates all fragments as singly-charged; it cannot
output [frag+2H]²⁺ species. Category B bins 766.7 and 784.7 (true intensity 0.75 and 0.60)
are large singly-charged ceramide-related fragments in DAG but severely underpredicted.

---

## Cause 6 — Data Artifacts & Unsupported Precursors

Not model or BFS limitations. Three sub-kinds:

- **Missing from the precomputed DAG tar** — the molecule was never enumerated; `find_missing_peaks.py` now skips these (`continue` on `dag_mzs is None`), so they no longer inflate the miss rate, but historical rows may still reflect them.
- **Contaminated / co-isolated / mislabeled NIST23 spectra** — lipid co-elution, out-of-window co-isolation, CE=nan junk.
- **Unsupported precursors** — permanent cations ([n+]/[P+]/[I+] salts: berberine, phosphonium salts, diphenyleneiodonium). Their precursor is [M]⁺, not [M+H]⁺; BFS's neutral-protonation model does not support them.

### Phosphosphingolipids (n=144, DAG miss rate=15.7%)

**11 mols in tar (35 missing).** Sphingomyelins and ceramide phosphates — a sphingoid base
backbone (C18 long-chain amino alcohol) with a phosphocholine or phosphoethanolamine head
group. Most in-tar mols contain a quaternary [N+](C)(C)C choline group (zwitterion in
neutral form, protonated head in [M+H]⁺).

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 77.0    | 14     | 0.203     | [C₆H₅]⁺ — deep aromatic cascade |
| 50.0    | 8      | 0.204     | small ring fragment |
| 79.1    | 6      | 0.066     | [C₆H₇]⁺ |
| 633.5   | 2      | 0.342     | **doubly-charged fragment ion** [M+2H]²⁺ type |
| 651.5   | 2      | 0.054     | **doubly-charged fragment ion** |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 74.1    | 14     | 0.072          |
| 91.1    | 12     | 0.173          |
| 81.0    | 11     | 0.152          |
| 60.1    | 11     | 0.114          |
| 60.0    | 9      | 0.506          |
| 247.2   | 6      | 0.125          |

#### Root cause

**Artifact (dominant) + minor model weakness + doubly-charged fragment ions.**

35/46 mols missing from tar accounts for the majority of the miss rate. For the 11 in-tar
mols, m/z 60.0 (9 spectra, 0.506) = [N(CH₃)₃+H]⁺ = trimethylammonium (60.058), the
choline head fragment — IS in DAG but model predicts ≈0.022. The .5-fractional category A
bins (633.5, 651.5) are doubly-charged fragment ions from the large sphingolipid backbone
(see glycosphingolipids for detailed confirmation). Low overall n (n=144) limits confidence.

---

### Other Glycerolipids (n=66, DAG miss rate=11.1%)

**3 mols in tar (9 missing).** Glycerophospholipids and ether lipids with long unsaturated
acyl chains (C18:1, C18:2). One mol (mol_id=14740) contains a quaternary [N+](C)(C)C.
Very small in-tar sample limits conclusions.

#### Missing peaks (category A — absent from DAG)

| m/z bin | n_spec | intensity | Note |
|---------|--------|-----------|------|
| 77.0    | 15     | 0.164     | [C₆H₅]⁺ phenyl cascade |
| 622.5   | 12     | 0.171     | **doubly-charged fragment ion** |
| 50.0    | 6      | 0.067     | small ring fragment |
| 529.5   | 3      | 0.089     | **doubly-charged fragment ion** |

#### In DAG but model underpredicts (category B)

| m/z bin | n_spec | true_intensity |
|---------|--------|----------------|
| 91.1    | 24     | 0.343          |
| 340.3   | 21     | 0.167          |
| 105.1   | 21     | 0.097          |
| 111.1   | 19     | 0.092          |
| 604.5   | 12     | 0.143          |

#### Root cause

**Artifact dominant (9/12 mols missing from tar) + doubly-charged fragment ions + model
weakness.**

The .5-fractional category A bins (622.5, 529.5) and category B bin (604.5) are
doubly-charged fragment ions from large glycerolipid precursors — same mechanism as
glycosphingolipids. The 91.1 (24 spectra, 0.343 = tropylium) and 105.1 are in DAG but
model predicts 0.000. With only 3 mols in tar the sample is too small for definitive
conclusions beyond these two patterns.

---

### Quinolizidinones (n=33, DAG miss rate=100%)

**Root cause:** Only 1 unique molecule (mol_id=1968, `C=CCC1NCC2CC1CN1C(=O)CCCC21`)
in this subclass, and it is **absent from the precomputed DAG tar archive**. When
`DagLoader.get()` returns `None`, the script (before the fix) classified all
peaks as absent from DAG.

Running BFS directly on the SMILES generates m/z 193.1335 (Δ=0.5 mDa from the
observed base peak 193.134). BFS coverage is fine — the molecule just fell
through precomputation. The `find_missing_peaks.py` script now skips spectra
whose mol_id is missing from the tar (`continue` on `dag_mzs is None`).

**Not an RDA issue.**

---

### Furan Lignans (n=21, DAG miss rate=82.6%)

**Root cause:** 3 of 6 spectra for mol_id=52196 (spec_ids 1788996–1788998,
CE=nan) are contaminated — no matching [M+H]⁺ at 355.118, base peak m/z 91.058
(C₇H₇⁺), and an aliphatic hydrocarbon ion series (57, 69, 71, 81, 95 Da). These
are lipid/wax co-elution or mislabeled spectra in NIST23. The genuine CE=10/15/20
spectra fragment cleanly and are fully BFS-covered.

**Not an RDA issue. Data quality problem in NIST23.**

---

## Cause 7 — Insufficient Data

Sample size too small (often n≈1 per missing peak) to distinguish a real gap from noise. No actionable finding.

### Pyrimidine Deoxyribonucleotides (n=24, DAG miss rate=75.0%)

n=24 spectra from very few molecules; all absent/under-predicted peaks have n_spectra=1.
**No actionable finding** — sample size too small to distinguish real DAG gap from noise.

---

### 1-Benzylquinolines (n=25, DAG miss rate=71.4%)

**Category A:** m/z 306.2, n=5 spectra, mean intensity 0.125. Likely a [M+H−benzyl]⁺ or
ring-retaining rearrangement product. Small n (5 of 25 spectra) indicates this is not a
systematic gap.

**Category B:** m/z 226.1 (n=1, intensity 0.059) and 213.1 (n=1, intensity 0.060) — in DAG
but not predicted. Effectively noise given n=1.

**No actionable finding** at this sample size.

---

### Aminoxides (n=51, DAG miss rate=57.7%)

**Brief.** Only one significant category A bin: m/z 62.1 (15 spectra, 0.091 intensity). Category B is
minimal (m/z 57.1 at 5 spectra, 0.069). No clear structural pattern from the dominant peaks alone.

**Root cause: minor model weakness.** Too few absent peaks and too low intensity to identify a structural
mechanism. Not a significant BFS gap.

---

## Summary (grouped by cause)

### Cascade depth

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Phenanthrenes & deriv. | 330 | 78.3% | High-CE RDA of oxycodone/naltrexone-type alkaloids; N-CONTAINING cascade (C₁₃H₁₀N⁺) beyond BFS depth=3 |
| Morphinans | 341 | 70.3% | High-CE cascade of morphine/codeine/levorphanol-type; N-FREE cascade (fluorenone C₁₃H₈O, acenaphthylene C₁₂H₈) beyond BFS depth=3; distinct pathway from Phenanthrenes group |
| Thienothiophenes | 104 | 63.1% | Small thiophene ring fragments (51.0=C₄H₃⁺, 69.0, 93.0) require >3 sequential bond cuts from fused bicyclic S-heterocycle |
| Phenoxazines | 112 | 62.8% | Ring-opening of phenoxazine O-bridge → aromatic ions (115.1=C₉H₇⁺, 170.1, 146.1) beyond depth=3; partial DAG coverage depending on substituents |
| Protoberberines | 105 | 62.1% | Neutral tetrahydro mols have cascade depth gap (91.1, 131.1, 105.1); berberine (permanent N⁺ cation) absent from tar |
| Cephalotaxus alkaloids | 46 | 60.5% | Same aromatic ion series as morphinans (141.1, 155.1, 157.1); 1 mol missing from tar (21 artifact spectra) |
| Pyranodioxins | 45 | 58.0% | Furanyl cations (69.0/81.0/109.0 at 0.60 intensity in 36/45 spectra) from pyranose ring-opening + sequential dehydration; requires >3 steps |
| Lactones | 57 | 57.6% | Terpenoid ring aromatization (191.1=C₁₄H₁₁⁺, 209.1); + model weakness (tropylium 91.1 at 0.737 in DAG); 5 mols not in tar (~8% artifact) |
| Androstane steroids | 919 | 56.3% | Djerassi cascade aromatization of C19 ring: C₅–C₁₀ aromatic ions (129.1=[C₁₀H₉]⁺ confirmed, 128.1, 115.1); 274 spectra missing 129.1 |
| Dibenzoxazepines | 126 | 54.5% | Ring-opening of tricyclic clozapine framework (229.1 at 0.655); + model weakness (126.0 at 0.598 in DAG); all 7 mols clozapine-type antipsychotics |
| Estrane steroids | 739 | 53.7% | Same aromatic series as androstane (128.1, 129.1, 115.1) + tropylium 91.1 at 0.620 (130 spectra) from phenol A-ring; model weakness 105.0 |
| Hydrolyzable tannins | 354 | 53.6% | Galloyl ester hydrolysis then ring aromatization (65.0=C₅H₅⁺ at 0.574); + model weakness on 115.1/128.1/91.1 (all in DAG) |
| Strychnos alkaloids | 117 | 51.8% | Heptacyclic strychnane ring (184.1 at 0.604 absent); 29/33 mols not in tar but skipped — does NOT inflate miss rate |
| Oxepanes | 353 | 50.4% | Complete C₅–C₉ aromatic cation series (91.1 at 0.577, 65.0, 77.0, 115.1) mirrors androstane; model weak on C₁₀–C₁₂ intermediates in DAG |
| Flavonoids | 161 | 49.8% | C₁₃H₉⁺ = 165.070 (150/249 spectra) via 2×CO loss + dehydrogenation; PAH series from naphtho-fused variants; + model weakness 105.0/123.0 in DAG |
| Furofurans | 93 | 48.6% | Complete C₅–C₁₁ cyclic cation series absent in 8–10/10 mols; tropylium universal (10/10, rearrangement); model weak on 128.1 (pred=0.000) |
| Quinolizines | 38 | 48.5% | C₇–C₉ aromatic cascade (117.1, 115.1) absent in 7/7 mols; tropylium in both cat A and cat B; model weak on 145.1 (true=0.485) |
| Phenanthrolines | 68 | 48.5% | Aromatic N-heterocycle cascade (115.1, 146.1, 170.1) beyond depth=3; + model weakness (139.1 at 0.668 in DAG); high tar-miss inflates cat A |
| Benzofuranones | 500 | 47.4% | Benzofuranone ring-opening aromatic cascade (91.1, 115.1, 165.1); 152.1/165.1 dual-mode; + model weakness 105.0 benzoyl |
| Pyranocoumarins | 248 | 47.4% | Pyranochromone aromatization cascade C₅–C₁₀ (91.1=66 spec at 0.485, 128.1=37 spec at 0.533); SMARTS prepass pivot would fix |
| Protopine alkaloids | 123 | 47.1% | Pentacyclic ring; tropylium (57 spec) + large fragments 247.1/275.1 absent; + model weakness on isoquinoline fragments (189.1); only 4 mols in tar |
| Butyrophenones | 582 | 46.8% | C₅–C₉ cascade (65.0=79 spec at 0.670, 117.1=73 spec) + tropylium 91.1 (107 spec, rearrangement); + model weakness 105.0/176.1 |
| Thienopyrroles | 87 | 46.3% | S-heterocycle cleavage (51.0 at 0.702, 93.0 at 0.555) mirrors thienothiophenes; + Br .9-bin direct cations (172.9, 122.9) |
| Triterpenoids | 1017 | 46.0% | LARGEST ABSOLUTE GAP — C₅–C₁₄ terpenoid cation series; 91.1 in 286 spectra at 0.620; cat B 119.1 GNN repetitive-chain failure; 88/199 mols missing from tar |
| Matrine alkaloids | 64 | 45.8% | Tetracyclic quinolizidine; 120.1 (0.590) and 106.1 (0.494) dihydropyridinium fragments absent; SMARTS prepass for matrine bicyclo core would fix |
| Organonitrogen compounds | 733 | 45.2% | C₄–C₇ aromatic cascade (51.0, 65.0, 77.0, 91.1) absent in 26–54 spectra; + model weakness on aliphatic fragments; catch-all class |
| Oxanes | 234 | 44.3% | Same O-ring cascade as oxepanes; 91.1 (0.676), 51.0 (0.633), 79.1 absent; shared SMARTS prepass for 5/6/7-membered O-rings would fix all three |
| Hexahydrobenzophenanthridines | 94 | 43.4% | Tetracyclic benzophenanthridine; 289.1 (0.613) and 275.1 (0.608) absent; + model weakness 188.1/261.1; Papaveraceae alkaloid cluster |
| Metallotetrapyrroles | 227 | 43.4% | 167.1 (123 spec) pyrrole macrocycle fragment absent; dipyrrole series 230.1–256.1 absent; + model weakness 115.1 (pred 0.001) |
| imidazo[4,5-c]pyridine ribonucleosides | 93 | 67.6% | Cascade depth ([C₄H₂N]⁺ at 64.0, 0.710) + rearrangement ([C₆H₆N₃O]⁺ at 136.1, O from ribose) + Cl direct cation (97.979, 6-chloro mol) |
| Organic pyrophosphates | 32 | 83.3% | 4 successive C₂H₄ losses (tetraethyl pyrophosphate, 291→263→235→207→179); each loss is cut+H-transfer and covered — 207 (3 losses) is in the DAG; only the 4th loss (base peak 179) exceeds depth=3. NOT a rearrangement gap; −C₂H₄ NeutralLossRule would fix. Single molecule drives the class |

### Rearrangement

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Organic metalloid salts | 100 | 76.1% | TMS⁺ methyl-migration (silyl cation, 37 mDa off DAG fragment); 10 artifact spectra from methylarsenic acid not in tar |
| Lycorine-type alkaloids | 96 | 55.4% | Methylenedioxy ring → [C₈H₇O]⁺ 119.049 (66 spec, 0.659, base peak) via ring contraction; tropylium 91.054 ring expansion; + model weakness 65.0 |
| Gamma-keto acids | 115 | 55.0% | Tropylium 91.054 (0.744) from furanyl/thienyl mols via ring expansion; cascade depth for aliphatic cation series; + model weakness 107.1/105.0 |
| Phenylphosphines & deriv. | 495 | 49.1% | C₅H₄P⁺ = 95.005 (phosphacyclopentadienyl, 131 spectra, P migration) absent; + cascade depth C₉–C₁₃ series; + model weakness [Ph₂P]⁺=183.0, 165.1, 219.1 in DAG |
| Cyclohexylphenols | 86 | 49.0% | Tropylium (ring expansion), cyclopentadienyl (ring contraction), naphthalene (ring fusion) all absent from BFS; + model weak on 105.0 |
| Acetophenones | 20 | 48.0% | Tropylium 91.1 (ring expansion) absent; 105.0 benzoyl (0.529) in DAG; n=20 too small for firm conclusions |

### Halide direct cations

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Benzyl halides | 97 | 76.0% | Halide-retaining cations (CH₂Br⁺ etc.) from heterolytic C–X cleavage; BFS only models [neutral+H]⁺ |
| Triazolopyrazines | 38 | 63.3% | Br-retaining aromatic cations (C⁷⁹Br⁺ at 90.918, etc.); all .9 m/z bins = Br mass defect; same mechanism as benzyl halides |
| Pyrazolopyrazoles | 28 | 63.0% | CH₂Br mols: halide-retaining cations (same as benzyl halides); non-halide mol: cascade depth |
| Phenyl sulfoxides | 162 | 59.0% | .9 bins (254.9/186.9/370.9): Br/Cl halide-retaining direct cations; integer bins (69.0, 192.0): cascade depth from sulfoxide bridge |
| Imidazopyrazines | 209 | 58.0% | Br/I direct cations absent from DAG (.9 bins: 141.9, 129.9, 189.9); large Br fragments in DAG severely underpredicted (289.9 at 0.790); m/z 64.0/52.0 unexplained |
| Alpha-keto acids | 32 | 57.6% | Br-retaining direct cations: [CH₂⁷⁹Br]⁺ = 92.926 dominates 28/32 spectra at 0.753; 1 brominated mol drives the entire miss rate |
| Aryl iodides | 315 | 53.2% | I direct cations (.9 bins 328.9/262.9/290.9) + cascade depth (136.0 at 0.607 from iodo-azole ring); 1 missing mol = permanent I⁺ cation (unsupported) |
| Organobromides | 62 | 46.7% | Br-retaining [C₂H₂⁷⁹Br]⁺ absent from DAG (.9 bins 92.9/90.9); + model weakness 124.0 (0.640) Br-fragment in DAG underpredicted |

### Model weakness

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Lignan glycosides | 115 | 75.8% | Aglycone fragments ARE in DAG; model predicts ~0; glycan-loss cascade not learned |
| Organic phosphoric acids | 58 | 75.0% | m/z 99 (H₃PO₄ loss) in DAG but under-predicted; m/z 69/83/55 aliphatic chain absent from DAG |
| Organic phosphoramides | 80 | 59.7% | m/z 135.1 (0.866 intensity) in 36 spectra, in DAG but predicted ≈0; HF loss and P–N cleavage both hit 135.07; real gap for small P-containing fragment ions |
| Pyranopyridines | 108 | 58.8% | 131.1/216.1/89.0 (0.41–0.60 intensity) all in DAG but predicted ≈0; some cascade gaps in category A |
| Sulfur mustard compounds | 28 | 58.3% | 278.9 (0.693) and 154.9 (0.226) Br-retaining fragments IN DAG but predicted ≈0; [M+H]⁺ itself at 278.9 not retained by model |
| Auronols | 21 | 56.4% | 149.0 (0.529) and 115.1 (0.508) in DAG, predicted ≈0; .9-bin cat A (240.9, 222.9) suggest Br mols via alternative_parents |
| 1,3-Dioxanes | 25 | 54.8% | 83.1=[C₅H₇O]⁺ (acetal ring-opening, 0.632) in DAG predicted ≈0; cascade depth for phenyl/furanyl cations (77.0, 95.0) |
| Medium-chain keto acids | 51 | 50.0% | 99.0=[C₅H₇O₂]⁺ (0.689) in DAG predicted ≈0; 196.2 base peak = data quality artifact (co-isolated compound in MS2 window) |
| Furanoquinolines | 58 | 48.1% | 227.1 (0.554) and 216.1 (0.400) in DAG predicted ≈0; 77.0/51.0/128.1 aromatic cascade absent; 104.0 (0.731) absent suggests rearrangement |
| Organic sulfuric acids | 36 | 47.8% | 80.0 (SO₃⁺, 0.628) and 141.0 (0.802) in DAG predicted ≈0; + halide direct cations (94.9/130.9 = [C₂H₂Cl₂/Cl₃]⁺) |
| Azacyclic compounds | 36 | 47.4% | 60.0 (0.720) absent = [C₂H₆NO]⁺ azacyclic fragment; 202.1/146.1 severely underpredicted in DAG; catch-all class limits conclusions |
| Flavans | 441 | 47.0% | 147.0=[C₉H₇O₂]⁺ (86 spec, 0.152) and 105.0 (0.309) in DAG predicted ≈0; retro-DA fragmentation intensity not learned; cascade residual at high CE |
| Lignan lactones | 80 | 45.5% | 115.1 (0.640), 219.1 (0.640), 129.1, 91.1, 128.1 ALL predicted 0.000; same aglycone GNN failure as lignan glycosides |
| Phenoxy compounds | 396 | 44.1% | 97.0 (0.684), 127.0 (0.661), 134.0 (0.590), 219.1 (0.516) all in DAG predicted 0.000; 105.0 absent in 72 spectra = cascade depth; strong retraining target |
| Quinone & hydroquinone lipids | 412 | 19.2% | GNN cannot distinguish positions in repetitive polyisoprenyl chain (identical k-hop neighborhoods); isoprenoid chain fragments in DAG predicted ≈0 |
| Saccharolipids | 466 | 17.3% | Same GNN repetitive-chain failure as quinone lipids, compounded by variable sugar moiety context; terpenoid/aliphatic chain fragments in DAG predicted ≈0 |

### Doubly-charged ions

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Glycosphingolipids | 32 | 14.3% | Large gangliosides ([M+H]⁺ ~992 Da) produce [frag+2H]²⁺ at .4–.7 fractional m/z bins (625.4, 785.7, 864.5); BFS generates only singly-charged; + model weakness for large singly-charged ceramide fragments |

### Data artifact / unsupported precursor

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Quinolizidinones | 33 | 100% | Mol missing from tar; BFS actually works |
| Furan lignans | 21 | 82.6% | Contaminated CE=nan spectra in NIST23 |
| Phosphosphingolipids | 144 | 15.7% | 35/46 mols missing from tar (dominant) + doubly-charged fragment ions (.5 bins 633.5, 651.5) + model weakness (60.0=[N(CH₃)₃+H]⁺ at 0.506) |
| Other glycerolipids | 66 | 11.1% | 9/12 mols missing from tar (dominant) + doubly-charged fragment ions (622.5, 529.5) + model weakness (91.1 tropylium pred=0.000) |

### Insufficient data

| Subclass | n | DAG miss | Notes |
|----------|---|----------|-------|
| Pyrimidine deoxyribonucleotides | 24 | 75.0% | n=1 per missing peak |
| 1-Benzylquinolines | 25 | 71.4% | m/z 306.2 in only 5 spectra |
| Aminoxides | 51 | 57.7% | Only m/z 62.1 missing (15 spectra, 0.091 intensity); insufficient for mechanistic conclusion |

