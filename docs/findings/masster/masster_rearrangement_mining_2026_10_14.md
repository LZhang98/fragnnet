# Rearrangement peaks in [M+H]+ spectra and mechanism rules that reach them

> **Status:** FINDING · **Updated:** 2026-10-15
> About 1% of Orbitrap MS/MS intensity sits on peaks no bond-cut fragment can make, concentrated in
> 1-2% of spectra. Thirteen mechanism rules applied with continuation reach 39% of that intensity on
> MSGYM+NIST20 [M+H]+ (null 7%) and replicate on NIST23 [M+H]+ (37% vs 7%) and [M-H]- (53% vs 12%).
> Four more rules pass for [M-H]- and [M+Na]+. Model-free; no model trained.

Plan built on this: [rewire rules in training](../../plans/masster/masster_rewire_rules_training_plan.md).

## Question

The user asked whether more SMARTS rules are worth writing, and what they would cover. Earlier gates
measured registered whole-precursor templates only ([SMARTS pathway gates](masster_smarts_pathway_gates_2026_09_30.md),
[outlier reach](masster_smarts_outlier_reach_2026_09_29.md)). This finding measures the ceiling directly
from the spectra, then gates mechanism rules that are followed by ordinary cuts.

## Method

**Mining** ([`debug_masster_rearrangement_mining.py`](../../../scripts/debug/smarts/debug_masster_rearrangement_mining.py)).
- **Scope:** Orbitrap (`ORB`, `ORB_F`) [M+H]+ spectra of the MassSpecGym benchmark and NIST20
  (`nist20_inchikey`), train and val only. 29,802 molecules (12,912 / 16,890), 282,637 spectra. Test was
  not inspected. QTOF is excluded because its tolerance admits too many formulas.
- **Peaks:** every peak at >= 5% of the base peak gets the sub-formulas of [M+H]+ within 10 ppm (2 mDa floor)
  and passing a ring-plus-double-bond check.
- **Classes:**
  - `connected`: some formula's heavy atoms form a connected set of the precursor within +-4 own H.
  - `rearrangement`: formulas exist, but none has a connected heavy-atom set.
  - `h_limit`: connected, but only beyond +-4 H.
  - `capped`: the connected-set search hit its cap (50,000 steps).
  - `isotope` and `no_formula`.
- **Annotation:** a rearrangement peak takes its lowest-error formula. The loss is [M+H]+ minus the ion.

**Rule gate** ([`debug_masster_rewire_rule_gate.py`](../../../scripts/debug/smarts/debug_masster_rewire_rule_gate.py)).
- **Rule:** a mapped SMARTS plus a graph edit: delete atoms, cut bonds, form bonds.
- **Real reach:** a peak counts as reached when a connected set of the REWIRED precursor graph has its
  formula within +-4 H. This equals applying the rule to every connected fragment that contains the motif
  and then cutting normally (continuation), so it is an unlimited-pool ceiling.
- **Null:** the same edit shape at random positions, with elements and bond orders wildcarded and terminal
  atoms kept terminal. It uses the same number of edits as real matches, three seeded draws.
- **Registered:** an existing MASSter template side already reaches the peak as a whole product.
- **Uncertainty:** molecule-bootstrap 95% CI of the real minus null per-molecule explained fraction.

Rules came from seven chemistry-class searches. Each class result was checked by one chemistry reviewer
and one statistics reviewer, who recomputed the numbers. A rule was kept only when neither reviewer
dropped it.

## Results

### How much intensity is affected

Mean share of spectrum intensity. Peaks below 5% of base are unclassified, so these are lower bounds.

| Class | MassSpecGym | NIST20 |
| --- | ---: | ---: |
| connected (ordinary fragments) | 83.3-86.6% | 87.1% |
| rearrangement | 1.42% | 1.07% |
| capped (undecided) | 2.6-5.5% | 1.1-1.2% |
| no formula | 0.73% | 1.27% |

**Formula fit.** All rearrangement peaks fit a sub-formula by construction: 83% fit exactly one, 16% fit
2-3, and 1.3% fit more than 3.

**How rearrangement intensity is spread over spectra.**

| | MassSpecGym | NIST20 |
| --- | ---: | ---: |
| Spectra with any rearrangement peak | 10.3% | 10.2% |
| Spectra with >= 10% of intensity on them | 4.32% | 3.26% |
| Spectra with >= 25% of intensity on them (the tail) | 1.89% | 1.11% |
| Mean cosine ceiling lost to these peaks (whole set) | 0.0080 | 0.0057 |

**The validation tail:** 669 spectra from 194 molecules.

**Losses are a long tail.**
- There are 11,175 distinct loss formulas, 8,219 of them seen in one molecule only.
- The top 20 losses hold 8.6% of rearrangement intensity, and the top 500 hold 35%.
- Grouping by exact loss formula hides mechanisms whose substituent varies. Grouping by the elements of
  the loss, the >= 25% tail is:

| Loss contains | Share of tail rearrangement intensity |
| --- | ---: |
| S and O | 43.5% |
| N and O | 18.5% |
| halogen | 13.2% |
| O only | 7.6% |
| P | 6.7% |
| S, no O | 4.7% |
| N only | 4.3% |
| C/H only | 1.6% |

### What the tail looks like (939 molecules, 3,711 spectra)

- **It belongs to the molecule but shows at high energy.**
  - In NIST20 the tail rate rises from 0.3% of spectra at NCE <= 20 to 1.5-1.7% above NCE 50.
  - Within a molecule, tail spectra have mean NCE 77 against 53 for its other spectra.
  - The per-molecule correlation of rearrangement share with NCE has a median of 0.60.
  - Only 45 molecules are tail in every spectrum, and 215 in exactly one.
- **The ions need continuation.** 72% of tail rearrangement intensity is in ions keeping 30-70% of the heavy
  atoms, and 18% in ions keeping more than 70%. Whole-precursor templates (no continuation) cannot reach
  most of it.
- **Rearrangement peaks dominate these spectra.** The base peak is a rearrangement peak in 57% of tail
  spectra, with 3.7 such peaks per spectrum.
- **Enrichment.** P(molecule has a tail spectrum) against a base rate of 3.2%:

| Molecule has | P(tail) |
| --- | ---: |
| sulfonyl | 14.7% |
| phosphorus | 15.1% |
| nitro | 8.1% |
| permanent charge | 6.9% |
| halogen | 4.4% |

- **It replicates.** NIST20 train and val both have 1.1% tail spectra; MassSpecGym 2.0% train, 1.1% val.

### Registered rules on MassSpecGym

The 10 registered rules eligible for [M+H]+ match 7.3% of MassSpecGym test molecules (170 / 2,319) but put
>= 10% unique intensity on only 16 of them (34 spectra). Script:
[`debug_masster_smarts_msgym_reach.py`](../../../scripts/debug/smarts/debug_masster_smarts_msgym_reach.py),
which applies the 2026-09-27 per-class gate without continuation.

### Mechanism rules that pass (final combined run, 2026-10-15)

Percentages are shares of ALL rearrangement intensity, on train+val, and the rules overlap. Output:
`output/rewire_rule_gate_2026_10_15/survivors/`. "Beyond reg." is tail intensity the registered templates
do not already reach as whole products.

| Rule | Edit | Real / null, all | Real / null, tail | Beyond reg., tail | CI (real - null, per molecule) | Molecules (val) | Precision |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `s_to_o_aromatic_C` | Ar-SO2-X: cut C-S, S=O; form Ar-O | 15.9 / 1.2 | 18.3 / 1.1 | 5.7 | 0.47-0.57 | 636 (145) | 0.40 |
| `sulfonyl_so2_extrusion` | C-SO2-X: delete SO2; form C-X | 10.7 / 4.4 | 13.8 / 4.8 | 7.5 | 0.11-0.32 | 419 (87) | 0.21 |
| `diaryl_ether_o_excision` | Ar-O-Ar: delete O; form Ar-Ar | 6.4 / 0.2 | 8.0 / 0.2 | 8.0 | 0.84-0.92 | 217 (49) | 0.46 |
| `acyclic_thioether_s_extrusion` | C-S-C (acyclic): delete S; form C-C | 3.3 / 0.3 | 3.4 / 0.2 | 3.4 | 0.46-0.64 | 226 (52) | 0.18 |
| `thiono_thiolo_aryl` | ArO-P=S: cut Ar-O, P=S; form Ar-S | 2.1 / 0.1 | 3.1 / 0.0 | 3.1 | 0.43-0.76 | 42 (7) | 0.60 |
| `nitro_nitrite_NO_loss` | C-NO2: delete N, O; form C-O | 2.1 / 0.7 | 1.5 / 0.5 | 1.5 | 0.15-0.24 | 220 (48) | 0.24 |
| `carbamate_co2_extrusion` | N-C(=O)-O-C: delete CO2; form N-C | 1.2 / 0.3 | 1.8 / 0.4 | 1.8 | 0.09-0.40 | 53 (7) | 0.10 |
| `hetaryl_thio_alkyl_s_extrusion` | C-S-c:n: delete S; form C-c | 1.4 / 0.0 | 1.7 / 0.0 | 1.7 | 0.45-0.75 | 98 (27) | 0.17 |
| `cyclic_thioether_s_extrusion` | ring C-S-C: delete S; form C-C | 1.0 / 0.1 | 1.5 / 0.1 | 1.5 | 0.30-0.79 | 55 (11) | 0.12 |
| `PS_O_transfer_6ring` | P-S-C-*-C=O: cut P-S; form P-O | 0.9 / 0.1 | 1.4 / 0.2 | 1.4 | 0.52-0.88 | 14 (2) | 0.70 |
| `diaryl_sulfide_s_extrusion` | Ar-S-Ar: delete S; form Ar-Ar | 0.9 / 0.0 | 1.1 / 0.0 | 1.1 | 0.67-0.91 | 41 (5) | 0.45 |
| `co2_extrusion_aryl_Oside` | X-C(=O)-O-Ar: delete CO2; form X-Ar | 0.5 / 0.0 | 0.3 / 0.1 | 0.3 | 0.36-0.79 | 37 (4) | 0.14 |
| `thiadiazole_cs2_extrusion` | alkyl-S-thiadiazole: delete CS2 unit; form C-N | 0.3 / 0.0 | 0.4 / 0.0 | 0.4 | 0.41-0.94 | 12 (4) | 0.38 |
| **Union** | | **39.0 / 6.8** | **47.4 / 7.0** | **33.2** | 0.54-0.63 | 1,568 (336) | |

The full SMARTS are in `output/rewire_rule_gate_2026_10_15/survivors.json`. Every rule, with status
(admitted, candidate or unreviewed), precursor types and gate numbers, is in
`output/rewire_rule_gate_2026_10_15/rule_catalog.json`.

**Validation only (union).**

| Validation subset | Real | Null | Spectra / molecules |
| --- | ---: | ---: | --- |
| All | 35.9% | 6.8% | 5,831 / 1,423 |
| >= 10% rearrangement intensity | 40.1% | 7.2% | 1,835 / 474 |
| Tail (>= 25%) | 41.1% | 6.7% | 668 / 194 |

**Cosine ceiling** (`sqrt(1 - s^2)`, where s is the L2 share of peaks no candidate reaches):

| Validation subset | Ceiling now | With union products reachable | All rearrangement reached |
| --- | ---: | ---: | ---: |
| Tail | 0.652 | 0.782 | 1.0 |
| >= 10% | 0.834 | 0.894 | |
| Whole split | 0.9945 | 0.9964 (+0.0019) | |

These ceilings include the null-level share and assume an unlimited pool and perfect intensities.

### Rejected or unconfirmed

- **Halogen:** no halogen-specific rule survives. The halogen tail is mostly ether O excision in halogenated
  aryl ethers.
- **Aryl-alkyl ether O excision:** no specificity. The same edit on a bridging atom other than O hits as often
  (6.6% against 6.9% of matched molecules).
- **Generic carboxyl, ester and aryl-C-side CO2 extrusion:** precision 3-6%, and most hits are hydrocarbon
  ions.
- **Sulfonyl variants:**
  - Sulfamide SO2 extrusion also matches sulfate esters, where it forms an O-O bond.
  - Sulfinyl S-to-O transfer does not replicate on val.
  - S-to-O transfer to the N side is below the null.
  - Smiles-type ipso tethers do not survive.
- **Nitro:** ortho O transfer, NO-then-CO loss and ortho cyclizations are at or below the null.
- **Others:** Beckmann, Lossen, N-oxide and nitrosamine rules have too few molecules.
- **Phosphorus:** triaryl P biaryl coupling, aryl P-oxide to aryloxy, the P-S 5-ring transfer and small P
  families are kept by one reviewer only, or have no val support.
- **N-only and skeletal rules** were gated but never reviewed. The strongest are aryl N-to-C acyl shift
  (6.1 / 0.5), diarylamine N excision (7.5 / 1.5) and amide N excision (4.5 / 0.8). Gate output:
  `output/rewire_rule_gate_2026_10_15/nitrogen_and_skeletal/r3_full.log`.

## NIST23 replication and other precursor types (2026-10-15)

**Mining.** `--dataset nist23` uses Orbitrap spectra from the `nist23_ma_mi_inchikey` train and val splits.
- **Held-out molecules dropped:** any molecule whose InChIKey block 1 is in a test split of NIST23 or NIST20
  `ma_mi`, `nist20_inchikey` or MassSpecGym.
- **Adducts:** Na, K, Cl, NH4 and formate are kept or lost as a group. Only the molecular part of an ion
  (`ion_mol`) is tested for connectivity.
- **Output:** `output/rearrangement_mining_nist23_2026_10_15/<slug>/`.

| Precursor | Molecules | Rearrangement % of intensity | Tail spectra (molecules) | Largest tail loss classes |
| --- | ---: | ---: | --- | --- |
| [M+H]+ | 26,966 | 1.21 | 1.25% (1,218) | S+O 38.5%, N+O 20.0%, halogen 15.4%, O-only 10.2% |
| [M-H]- | 12,188 | 1.16 | 1.55% (630) | S+O 49.6%, N+O 18.1%, P 11.7% |
| [M+NH4]+ | 2,164 | 1.00 | 1.19% (55) | S+O 48%, N+O 36% |
| [M+Na]+ | 6,706 | 0.82 | 0.82% (134) | N+O 36%, S+O 32%; 43% of rearrangement intensity keeps Na |
| [Cat]+ | 250 | 0.96 | 1.34% (9) | halogen 76% |
| [M+K]+ | 1,249 | 1.55 | 0.70% (13) | 32% of intensity has no formula (solvent/background) |
| [M+Cl]- | 808 | 0.25 | 0.19% (6) | none of note |
| [M+CHO2]- | 921 | 0.03 | 0 | none |

- **New molecules:** 16,713 of the NIST23 [M+H]+ molecules, and 752 of its 1,218 tail molecules, were not in
  the NIST20 run.
- **[M-H]- losses:** the most common is O2S (222 molecules), then NO (76) and N2 (34).
- **Not mined:** [M+2H]2+ and in-source fragment precursors.

**The 13 [M+H]+ rules on other data.** Gate runs with unchanged rules, output
`output/rewire_rule_gate_2026_10_15/transfer/`. Real / null, % of rearrangement intensity.

| Data | All | Tail | Molecules explained (val) |
| --- | ---: | ---: | ---: |
| NIST23 [M+H]+ (mostly new molecules) | 36.9 / 7.4 | 44.3 / 7.1 | 1,959 (474) |
| NIST23 [M-H]- | 53.1 / 12.3 | 56.9 / 12.6 | 675 (168) |
| NIST23 [M+NH4]+ | 37.2 / 8.4 | 40.0 / 7.9 | 79 (21) |
| NIST23 [M+Na]+ | 13.3 / 2.6 | 12.7 / 1.9 | 142 (33) |

- **Replication on [M+H]+:** every rule beats its null on NIST23, with the per-molecule CI lower bound > 0.
- **[M-H]-:** the rules transfer better than on [M+H]+, led by SO2 extrusion (31.3 / 6.3), S-to-O migration
  (21.9 / 5.5), nitro NO loss (9.2 / 2.3) and acyclic thioether S extrusion (6.9 / 0.3).
- **[M+Na]+:** the rules do not transfer. Sodiated ions rearrange through Na-bound carboxyl and carbonyl groups.

**New rules (2026-10-15).** Same discovery and two-reviewer process as above, output
`output/rewire_rule_gate_2026_10_15/adducts/`. Kept by both reviewers:

| Rule | Precursor | Edit | Real / null, all | Real / null, tail | Molecules (val) |
| --- | --- | --- | ---: | ---: | ---: |
| `azo_N2_extrusion` | [M-H]- | C-N=N-X: delete N2; form C-X (azo, triazene, diazosulfonate) | 0.98 / 0.02 | 0.84 / 0.00 | 35 (12) |
| `haloalkyl_sulfonyl_X_to_S` | [M-H]- | 1,3 F/Cl/Br shift from alpha C to sulfonyl S (FSO3-, FSO2- from PFAS sulfonates) | 1.81 / 0.09 | 1.79 / 0.14 | 12 (4) |
| `dinitrophenol_24_2NO` | [M-H]- | two nitro-nitrite NO losses on 2,4-dinitrophenolates | 0.70 / 0.04 | 0.76 / 0.02 | 9 (2) |
| `acid_acyl_transfer_5` | [M+Na]+ | carboxyl O attacks the N-acyl carbonyl (5-ring TS); Na kept ([b(n-1)+OH+Na]+ type) | 9.85 / 0.40 | 15.71 / 0.71 | 25 (5), 28 structures |

**Candidates, kept by one reviewer or statistically unproven:**
- `aryl_sulfoxide_S_to_O` ([M-H]-; 6 structures; S(IV) cases lumped together).
- `phosphate_pyrophosphate_condensation` ([M-H]-). Against a loose null it is 2.46 / 2.06, with 69% of the
  weight from 3 inositol-phosphate molecules.
- `aryl_O_CH2R_excision` ([M+NH4]+; no known mechanism, 57% of its hits are CxHy).
- Nitroaryl NO+SO2 composites (no null is possible on these small molecules).

**Rejected:**
- Tetrazole N2 extrusion and generic dinitroarene 2NO loss.
- Halogen 1,2 and 1,4 shifts, which are below the null.
- [M+Na]+ ester and carbonate CO2 extrusions.
- Ether O excisions on [M-H]-.

## Limits

- **Ceiling only.** Reach assumes the parent fragment is in the pool, which the training plan's gate G0b
  measures. It also assumes the model can rank the products, which is untested.
- **SO2 extrusion's null is not size-matched.** Real deletes three atoms, the null one non-terminal atom.
  Its margin over null (13.8 against 4.8 in the tail) is the least certain of the large rules.
- **Hydrocarbon ions.** C7H7+, C9H7+ and similar ions are reachable by many edits. 68% of the diaryl ether
  rule's hit weight is pure CxHy, which a position-matched null should re-test.
- **Concentration and duplicates.** A few molecules dominate some rules; for example `msgym:11093` holds
  18.6% of SO2-extrusion hit intensity. The same compound can appear in both MassSpecGym and NIST20, so
  molecule counts are not de-duplicated by structure.
- **Nulls before the fix.** Class runs before 2026-10-15 seeded nulls with Python `hash`, which is not
  reproducible. The final combined run uses CRC-32.
- **Scope.** Evidence is [M+H]+ Orbitrap only, so rules need precursor-type guards.

## Artifacts

- Mining: `output/rearrangement_mining_2026_10_14/` (`peaks.pkl`, `rearrangement.pkl`, `losses.csv`,
  `molecules.csv`, `families/`).
- Gates: `output/rewire_rule_gate_2026_10_15/`, one directory per class with the rules tried and reviewer
  re-runs, `survivors.json` and `survivors/`.
- Per-class and reviewer records: workflow run `wf_5d6e0b39-5bd` (session transcript, not in the repo).
