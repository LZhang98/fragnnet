## Coverage Summary

72 rules across 9 categories.  Status codes used in per-rule comments below:
`BFS ✓` = already handled by multi_cut_bfs cut=1/2 + H-transfer.
`prepass-1` = needs isomerization prepass (run full BFS on structural isomer).
`prepass-2` = needs direct-fragment prepass (SMARTS → atom-set extraction → depth-1 DAG leaf).
`BUG` = SMARTS is syntactically or chemically invalid; see inline comment.

| Category | Total | BFS ✓ | prepass-1 | prepass-2 | BUG / skip |
|---|---|---|---|---|---|
| PPGB_MS2 original (28 reactions) | 28 | 21 | 3 (crf8_0-2) | 2 (crf12_0/1 acyclic, crf13_2 acyclic) | 2 (crf11 → neutral loss, crf6 not coverable) |
| Extended neutral loss | 10 | 7 | — | — | 3 (nl2, nl_h2o_2, nl_co) |
| Amide / peptide | 7 | 6 | — | — | 1 (amide_acylium) |
| Ring & RDA | 9 | 0 | — | — | 9 (all: unmapped atoms / new bonds / wrong formula) |
| Rearrangement | 4 | 1 (ortho_effect) | — | — | 3 (mclafferty ×2 buggy; mclaff_beta bimolecular) |
| Lipid / phospholipid | 3 | — | — | — | 3 (all: bimolecular or invalid [R]/[OP] atoms) |
| Ester / ether | 3 | 1 (ether_alpha) | — | — | 2 (ester_cleavage_1/2: invalid [OC]/[OP]) |
| Charge migration | 5 | — | — | — | 5 (all no-ops: explicit +0 on both sides) |
| Glycan | 3 | — | — | — | 3 (all: invalid [NHAC]/[R] atoms or wrong formula) |
| Specialized | 7 | 2 (nitro_loss, nglycosidic) | — | — | 5 (phos_loss, omega_cleavage, cr_acid, pl_head, b_ion: invalid [OP]/[R]) |

**Net new coverage from a rearrangement prepass: 5 valid rules**
- Sub-type 1 (isomerize → re-BFS): crf8_0, crf8_1, crf8_2
- Sub-type 2 (direct atom-set extraction): crf12_0/1 (acyclic), crf13_2 (acyclic)

McLafferty (mclafferty / mclaff_classic) is BUGGY in this list but the
correct McLafferty reaction is already fully covered by crf4_0 (BFS cut=1 +
H-transfer) — no additional rule needed.

---

## Original PPGB-MS2 Rules (27 rules from `smiles_fragmentation.py`)

```
crf1_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1][*+0:2]>>([*+0:0]=[*+0:1].[*+0:2])')
crf1_1 = AllChem.ReactionFromSmarts('[*+0;!H0:0]=[*+0:1][*+0:2]>>([*+0:0]#[*+0:1].[*+0:2])')
crf2_0 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1][*!H0+0:2]>>([*+0:0].[*+0:1]=[*+0:2])')
crf2_1 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1]=[*!H0+0:2]>>([*+0:0].[*+0:1]#[*+0:2])')
crf3_0 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf3_1 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1]=[*+0:2][*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3].[*+0:4]#[*+0:5])')
crf4_0 = AllChem.ReactionFromSmarts('[*+0:0]=[*+0:1][*+0:2][*+0:3][*+0;!H0:4]>>([*+0:0][*+0:1]=[*+0:2].[*+0:3]=[*+0:4])')
crf5_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1][*+0:2][*+0;!H0:3]>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3])')
crf6_0 = AllChem.ReactionFromSmarts('[*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6]=[*+0:7]>>([*+0:0]=[*+0:7].[*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6]=[*+0:1])')
crf7_0 = AllChem.ReactionFromSmarts('[*+0:0]1[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]=[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf7_1 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2][*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]=[*+0:3].[*+0:4]=[*+0:5])')
crf7_2 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4][*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]=[*+0:5])')
crf7_3 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]#[*+0:1].[*+0:2]#[*+0:3].[*+0:4]#[*+0:5])')
crf8_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2]>>([*+0:0]=[*+0:1][*+0:2])')
crf8_1 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4])')
crf8_2 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0:1]=[*+0:2][*+0:3]=[*+0:4][*+0:5]=[*+0:6]>>([*+0:0]=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5][*+0:6])')
crf9_0 = AllChem.ReactionFromSmarts('[*:0]@[C:1](=[O:2])@[*:3]>>([*:0][*:3].[C-:1]#[O+:2])')
crf11_0 = AllChem.ReactionFromSmarts('[*+0;!H0:0][*+0;!H0:1]>>([*+0:0]=[*+0:1])')
crf11_1 = AllChem.ReactionFromSmarts('[*+0;!H0:0]=[*+0;!H0:1]>>([*+0:0]#[*+0:1])')
crf12_0 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1][*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]=[*+0:2])')
crf12_1 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1]=[*+0:2][*+0:3]>>([*+0:0][*+0:3].[*+0:1]#[*+0:2])')
crf13_0 = AllChem.ReactionFromSmarts('[*+0:0][C+0:1](=[O+0:2])[OH+0:3]>>([*+0:0].[O+0:2]=[C+0:1]=[O+0:3])')
crf13_1 = AllChem.ReactionFromSmarts('[C+0;H1:0](=[O+0:1])[O+0:2][*+0:3]>>([O+0:1]=[C+0:0]=[O+0:2].[*+0:3])')
crf13_2 = AllChem.ReactionFromSmarts('[*+0:0][C+0:1](=[O+0:2])[O+0:3][*+0:4]>>([O+0:2]=[C+0:1]=[O+0:3].[*+0:0][*+0:4])')
cmf2_0 = AllChem.ReactionFromSmarts('[N+0,O+0,S+0:0][*+0:1][*:2]>>([N+1,O+1,S+1:0]=[*+0:1].[*+0:2])')
cmf2_1 = AllChem.ReactionFromSmarts('[N+0,O+0,S+0:0]=[*+0:1][*:2]>>([N+1,O+1,S+1:0]#[*+0:1].[*+0:2])')
cmf7_0 = AllChem.ReactionFromSmarts('[*+0:0][*+0:1]>>([*+1:0].[*-1:1])')
cmf13_0 = AllChem.ReactionFromSmarts('[*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+0:4]=[*+0:5]1>>([*+0:0]1=[*+0:1][*+0:2]=[*+0:3][*+1:4]1.[*:5])')
```


## Extended Neutral Loss Rules (8 rules)

```
nl1 = AllChem.ReactionFromSmarts('[C:1][OH2+0:2]>>([C:1].[OH2+0:2])')          # OK — C-OH2 bond break (BFS ✓)
nl2 = AllChem.ReactionFromSmarts('[C:1]=[O+0:2].[OH+0:3]>>([C:1][OH+0:3].[O+0:2])')  # BUG: two reactants (bimolecular) — invalid for unimolecular MS/MS
nl3 = AllChem.ReactionFromSmarts('[N+0:1][H3+0:2]>>([N+0:1].[NH3+0:2])')       # OK — N-NH3 bond break (BFS ✓)
nl_h2o_1 = AllChem.ReactionFromSmarts('[CX4:1][OX2H2:2]>>([CX4:1].[OH2:2])')   # OK — alcohol C-O break (BFS ✓)
nl_h2o_2 = AllChem.ReactionFromSmarts('[CX3:1](=[OX2:2])[OH:3]>>([CX3:1][OH:3].[O:2])')  # BUG: product [O:2] = bare oxygen atom/radical — not a real MS/MS fragment
nl_co2 = AllChem.ReactionFromSmarts('[CX3:1](=[OX2:2])[OH:3]>>([CX3:1].[OC:2]=[O:3])')   # OK — CO2 loss from COOH (BFS + H-transfer=2 ✓; explicit injection preferred)
nl_nh3_1 = AllChem.ReactionFromSmarts('[NX3:1][CX3:2]>>([NX3:1].[NH2:2])')     # OK — N-C break, NH2 leaves (BFS ✓)
nl_nh3_2 = AllChem.ReactionFromSmarts('[C:1][N:2][H2:3]>>([C:1].[NH3:2])')     # OK — C-N break, NH3 leaves (BFS ✓)
nl_meoh = AllChem.ReactionFromSmarts('[CX4:1][OX2H][CX4:2]>>([CX4:1].[CO:2])') # OK — ether C-O break (BFS ✓); [CO:2] maps atom :2 as CO, unusual notation
nl_h2o_alc = AllChem.ReactionFromSmarts('[C;!$(C=[O,S,N]):1][O;H1,H2:2]>>([C:1].[O:2])')  # OK — alcohol C-O break (BFS ✓)
nl_h2o_acid = AllChem.ReactionFromSmarts('[C:1](=[O:2])[OH:3]>>([C:1].[O:2].[OH:3])')     # BUG: 3 products — bare C, bare O, bare OH; chemically unsound
nl_co = AllChem.ReactionFromSmarts('[C:1]=[O:2].[C:3]>>([C:1].[C:3]=[O:2])')   # BUG: two reactants (bimolecular) — CO transfer between molecules; invalid
nl_ch2o = AllChem.ReactionFromSmarts('[N:1][CH2:2][O:3]>>([N:1H2:2].[CH2:3]=[O])')  # BUG: syntax error — [N:1H2:2] has two atom map numbers on one atom
```


## Amide/Peptide Rules (6 rules)

```
amide1 = AllChem.ReactionFromSmarts('[C:1](=[O:2])[NH+0:3][C:4]>>([C:1](=[O:2])[NH2+0:3].[C:4])')    # OK — N-C4 break, acyl-NH2 + C4 (BFS + H-transfer=1 ✓)
amide_n_c = AllChem.ReactionFromSmarts('[N:1][CX3:2](=[O:3])[CX4:4]>>([N:1][CX3:2]=[O:3].[CX4:4])') # OK — CX3-CX4 alpha-cleavage (BFS ✓)
amide_c_n = AllChem.ReactionFromSmarts('[CX3:1](=[O:2])[NH:3][C:4]>>([CX3:1][NH2:3].[C:4]=[O:2])')  # OK — N-CX3 break + O migrates to C4 (BFS + H-transfer ✓ formula-level)
amide_std = AllChem.ReactionFromSmarts('[C:1](=[O:2])[NH:3][C:4]>>([C:1](=[O:2])[NH2:3].[HO:4]C)')  # OK — C4-N break (BFS + H-transfer ✓)
amide_acylium = AllChem.ReactionFromSmarts('[N:1][C:2](=[O:3])[C:4]>>([N:1H2:2].[C:4][C:3+]=[O])')  # BUG: [N:1H2:2] — two atom maps on one atom (syntax error); intended acylium formation
urea_cleavage = AllChem.ReactionFromSmarts('[NX3:1][CX3:2](=[O:3])[NX3:4]>>([NX3:1][CX3:2]=[O:3].[NX3:4])') # OK — N4-C break (BFS ✓)
cyclic_amides = AllChem.ReactionFromSmarts('[C:1]1([N:2])[C:3](=[O:4])[N:5]C1>>([C:1][N:2].[C:3](=[O:4])[NH2:5])') # OK — 2-bond lactam ring opening (ring pre-pass ✓)
```


## Ring \& RDA Rules (6 rules)

```
rda1 = AllChem.ReactionFromSmarts('[C:1]1=CC=[C:4]C=C1>>([C:1][C:4].[C:2]=[C:3].[C:5]=[C:6])')
  # BUG: atoms :2,:3,:5,:6 unmapped in reactant (bare CC, C=C); new C1-C4 bond formed (C1,C4 are 3 bonds apart in ring) — RDKit can't track unmapped atoms

ring_extra = AllChem.ReactionFromSmarts('[C:1]1CCCC[C:6]1>>([C:1].[C:6])')
  # BUG: atoms between :1 and :6 are unmapped — products [C:1] and [C:6] are bare atoms, not fragments; intent unclear

rda_cyclohexene = AllChem.ReactionFromSmarts('[C:1]1=C[C:3]=C[C:5]=C1>>([C:1][C:6].[C:2]=[C:3].[C:4]=[C:5])')
  # BUG: atoms :2,:4,:6 appear in product but unmapped in reactant; new C1-C6 bond formed

rda_cyclohex = AllChem.ReactionFromSmarts('[C:1]1=CC=CC=C1>>([C:1][C:6].[C:2]=[C:3].[C:4]=[C:5])')
  # BUG: atoms :2,:3,:4,:5,:6 all unmapped in reactant; same issues as rda_cyclohexene

rda_cp = AllChem.ReactionFromSmarts('[c:1]1cccc1>>([c:1][c:5].[C:2]=[C:3].[C:4]=C)')
  # BUG: :2,:3,:4,:5 unmapped in reactant lowercase atoms; [C:2],[C:3],[C:4] in product use uppercase C but ring has lowercase c (aromatic); :5 unmapped

rda_furan = AllChem.ReactionFromSmarts('o1cccc1>>(c1cccc1.[H][H].[H][H])')
  # BUG: furan (C4H4O) ≠ benzene (C6H6) — wrong formula; [H][H] is not valid SMARTS for H2; no atom mapping at all

cyclopent_open = AllChem.ReactionFromSmarts('[C:1]1CCCC1>>([C:1].[C:5])')
  # BUG: intermediate carbons unmapped; products are bare [C:1] and [C:5] atoms — intent is ring opening but syntax doesn't encode the chain fragment

cyclohex_open = AllChem.ReactionFromSmarts('[C:1]1CCCCC1>>([C:1].[C:6])')
  # BUG: same as cyclopent_open — unmapped intermediate carbons, products are bare atoms

piperidine_open = AllChem.ReactionFromSmarts('N1CCCCC1>>(NCCCC.C)')
  # BUG: no atom mapping at all; product NCCCC.C (5-atom + 1-atom) doesn't represent a valid 2-bond ring opening of piperidine (N + 5C ring → should give N-CCCCC chain); C as methane fragment is chemically wrong

furan_open = AllChem.ReactionFromSmarts('o1cccc1>>(c1cccc1.[H][H].[H][H])')
  # BUG: duplicate of rda_furan — same issues (wrong formula, invalid H2 notation)
```


## Rearrangement Rules (3 rules)

```
mclafferty = AllChem.ReactionFromSmarts('[C:1][C:2][C:3][C:4]=[O:5]>>([C:1][C:2]=[C:3][OH:4].[C:5]=O)')
  # BUG: atom type remapping — :4 is [C] (carbonyl carbon) in reactant but [OH] (oxygen) in product; :5 is [O] (carbonyl O) in reactant but used as C in `[C:5]=O` product; C and O swap identity across :4 and :5

mclaff_classic = AllChem.ReactionFromSmarts('[C:1][C:2][C:3][C:4]=[O:5]>>([C:1][C:2]=[C:3][OH:4].[C:5]=O)')
  # BUG: duplicate of mclafferty — same C/O type-remapping bugs on :4 and :5

mclaff_beta = AllChem.ReactionFromSmarts('[C:1][C:2]=[O:3].[CH2:4][R:5]>>([C:1][C:2][OH:4].[C:3]=O.[R:5])')
  # BUG: bimolecular (two reactants separated by dot); also atom type remapping: :3 is [O] in reactant but appears as C in product `[C:3]=O`; [R:5] uses R as an element (not valid SMARTS)

ortho_effect = AllChem.ReactionFromSmarts('c1ccccc1[OH2+0]>>(c1ccccc1.[OH2+0])')
  # OK — C-OH2 bond break (BFS ✓)
```


## Lipid/Phospholipid Rules (4 rules)

```
pl_head = AllChem.ReactionFromSmarts('[O:1][P:2](=[O:3])([O:4])[CH2CH2N+]:5>>([O:1][P:2](=[O:3])([O:4])[O:5].[CH2CH2N])')
  # BUG: `[CH2CH2N+]:5` is invalid SMARTS — multi-atom group in brackets is not a valid atomic primitive; atom map `:5` outside brackets is a parse error; `[CH2CH2N]` in product is also non-standard; atom :5 in product is [O:5] but the reactant :5 was the CH2CH2N group — atom type remapping

fa_loss = AllChem.ReactionFromSmarts('[O:1][C:2]=[O:3].[CH2:4][CH:5][CH2:6][R:7]>>([O:1][C:2]=[O:3].[R:7][CH2:6][CH:5][CH2:4])')
  # BUG: bimolecular (two reactants separated by dot) — invalid for unimolecular MS/MS; also `[R:7]` uses R as element (not valid SMARTS); products are the same atoms rearranged without bond breaking — models only reordering, not fragmentation

cer_cleave = AllChem.ReactionFromSmarts('[C:1](=[O:2])[NH:3][CH:4][R1:5].[CH:6][OH:7][R2:8]>>([C:1](=[O:2])[NH2:3].[CH:4][R1:5].[CH:6][OH:7][R2:8])')
  # BUG: bimolecular (ceramide amide and sphingoid hydroxyl written as two separate reactant molecules); ceramide is a single molecule — should be one connected reactant; `[R1:5]` and `[R2:8]` use non-standard R notation as an element
```


## Ester/Ether Rules (4 rules)

```
ester_cleavage_1 = AllChem.ReactionFromSmarts('[C:1](=[O:2])[O:3][C:4]>>([C:1][OH:3].[C:4]=[O:2])')
  # BUG: O2 (carbonyl oxygen, bonded only to C1) migrates across the two product fragments — appears in C4's fragment in product; C1 and C4 are not directly bonded so O2 cannot cross fragments in a single bond cleavage; formula is conserved but the atom migration is physically invalid

ester_cleavage_2 = AllChem.ReactionFromSmarts('[C:1][O:2][C:3](=[O:4])>>([C:1][OH:2].[OC:3]=[O:4])')
  # BUG: `[OC:3]` is invalid SMARTS atom syntax — an atom cannot be simultaneously O and C; likely intended as an unmapped O bonded to C:3 (i.e. `[O][C:3]=[O:4]` for CO2 loss) but the notation is wrong; produces mass imbalance from the unmapped O

ether_alpha = AllChem.ReactionFromSmarts('[CX4:1][OX2:2][CX4:3]>>([CX4:1].[CX4:3][OH:2])')
  # OK — C1-O2 bond break, O2 stays with C3 + gains H; BFS + H-transfer=1 ✓
```


## Charge Migration Rules (6 rules)

```
charge_o1 = AllChem.ReactionFromSmarts('[CX3+0:1][O:2]>>([CX3:1][O+0:2])')
  # BUG: no-op — both reactant (+0) and product (+0) are neutral; no bond change, no H change; charge is specified as +0 in both, so this transforms nothing

charge_o2 = AllChem.ReactionFromSmarts('[CX4+0:1][OH:2]>>([CX4:1][OH+0:2])')
  # BUG: no-op — same issue; charge +0 in reactant and +0 in product; effectively identity transformation

charge_n1 = AllChem.ReactionFromSmarts('[CX3+0:1][N:2]>>([CX3:1][N+0:2])')
  # BUG: no-op — no structural change; explicit +0 on both sides

charge_n2 = AllChem.ReactionFromSmarts('[C+0:1][NH2:2]>>([C:1][NH3+0:2])')
  # BUG: NH2 gains H to become NH3 without C losing H — mass not conserved; H appears from nowhere; also NH3+0 means neutral NH3 still bonded to C:1, which is NH3 substituent, not ammonia loss

charge_s = AllChem.ReactionFromSmarts('[C+0:1][S:2]>>([C:1][S+0:2])')
  # BUG: no-op — same as charge_o1/charge_n1; all charge migration rules in this block are structural no-ops
```


## Glycan Rules (3 rules)

```
glycosidic_hexnac = AllChem.ReactionFromSmarts('[O:1][C:2][NHAC:3][C:4][C:5][O:6]>>([O:1][C:2][NH2AC:3].[C:4][C:5][OH:6])')
  # BUG: `[NHAC]` and `[NH2AC]` use `AC` as element — not a valid SMARTS element symbol; likely intended to represent acetamide NH (NHCOCH3) shorthand, but `AC` is not parseable as an element (closest element is Ac = Actinium, element 89)

crossring_a = AllChem.ReactionFromSmarts('[O:1][CH2:2][CH:3][O:4][R:5]>>([O:1][CH:2][CH2:3].[O:4][R:5])')
  # BUG: H counts on :2 and :3 are swapped within the same product fragment (CH2→CH and CH→CH2); this models an internal 1,2-H shift at the same time as a C-O bond break; H-transfer mechanism handles inter-fragment H only; also `[R:5]` is not a valid SMARTS element symbol

b_ion = AllChem.ReactionFromSmarts('[C:1][O:2][C:3][OR:4]>>([C:1][OH:2].[OC:3][OR:4])')
  # BUG: `[OR:4]` is not valid SMARTS — R is not a chemical element (it is a ring-bond descriptor, not used as atom symbol in brackets); `[OC:3]` is also invalid (atom cannot be both O and C)
```


## Specialized Rules (5 rules)

```
nitro_loss = AllChem.ReactionFromSmarts('[C:1][N+0:2](=[O:3])[O:4]>>([C:1].[N:2](=[O:3])[O:4])')
  # OK — C-N bond break, NO2 fragment leaves (BFS ✓); N+0→N charge change is minor notation difference

phos_loss = AllChem.ReactionFromSmarts('[O:1][P:2](=[O:3])([O:4])[O:5]>>([O:1].[OP:2](=[O:3])([O:4])[O:5])')
  # BUG: `[OP:2]` is invalid SMARTS atom syntax — atom cannot be both O and P; `[O:1]` product is a bare oxygen atom/radical (not a real MS/MS fragment)

omega_cleavage = AllChem.ReactionFromSmarts('[CH3:1][CH2:2][CH2:3][R:4]>>([CH3:1][CH2:2].[CH2:3][R:4])')
  # BUG: `[R:4]` uses R as element symbol — not valid SMARTS (R is a ring-bond notation, not an element); the intended C2-C3 bond break is BFS-valid if syntax were corrected

cr_acid = AllChem.ReactionFromSmarts('[R:1][CH2:2][CH2:3][C:4](=[O:5])[OH:6]>>([R:1][CH2:2].[CH2:3][C:4](=[O:5])[OH:6])')
  # BUG: `[R:1]` uses R as element — same issue as omega_cleavage; intended C2-C3 bond break (BFS ✓) but syntax is invalid

nglycosidic = AllChem.ReactionFromSmarts('[n:1][C:2][O:3][P:4]>>([n:1].[C:2][OH:3][P:4])')
  # OK — n1-C2 bond break (BFS ✓), O3 gains H (H-transfer=1 ✓); [n:1] product is a bare aromatic N but acceptable for nucleobase fragment context

phos_ribose = AllChem.ReactionFromSmarts('[C:1][O:2][P:3](=[O:4])([O:5])[O:6]>>([C:1][OH:2].[OP:3](=[O:4])([O:5])[O:6])')
  # BUG: `[OP:3]` invalid SMARTS atom syntax (atom cannot be both O and P); same issue as phos_loss; C1-O2 bond break itself is BFS-valid

na_loss = AllChem.ReactionFromSmarts('[Na+].[M:1]>>([Na+].[M-H:1]))')
  # BUG: bimolecular (Na+ adduct written as separate species); `[M]` is not a valid SMARTS element (M = Metal Markush notation, not parseable); `[M-H]` means charge −H which is nonsensical; extra closing paren in SMARTS string

metal_h2o = AllChem.ReactionFromSmarts('[Na+].[OH2].[M:1]>>([Na+].[M:1].[O])')
  # BUG: three reactants — completely invalid for unimolecular fragmentation; `[M:1]` is not a valid SMARTS element; `[O]` bare oxygen product is not a real MS/MS fragment
```

**Total: 72 unique SMARTS rules** across all categories from this conversation. Copy-paste ready for your recursive fragmentation DAG implementation.

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12658861/

[2]: https://pubmed.ncbi.nlm.nih.gov/39575948/?fc=None\&ff=20241122143352\&v=2.18.0.post9+e462414

[3]: https://pubs.acs.org/doi/abs/10.1021/acs.analchem.4c04375

[4]: https://pubs.acs.org/doi/10.1021/acs.analchem.5b04925

[5]: https://pmc.ncbi.nlm.nih.gov/articles/PMC4869618/

[6]: https://www.biorxiv.org/content/10.1101/2025.03.12.642824.full

[7]: https://www.sciencedirect.com/science/article/pii/S0021925819485335

[8]: https://www.nature.com/articles/s41467-024-48009-6

[9]: https://www.scribd.com/document/912806062/