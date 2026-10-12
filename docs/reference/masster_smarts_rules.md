# MASSter SMARTS Rule Catalog

> **Status:** REFERENCE · **Updated:** 2026-10-01
> 15 rules are registered but off by default (`masster_smarts_templates: False`), and no
> recorded trained run uses them. Eight more were disabled on 2026-09-29, to re-check on
> Enveda-180. Three registered rules fail the 2026-09-27 gate and stay pending an ablation.
> Also holds the literature backlog.

This document defines which gas-phase rearrangement rules belong in MASSter,
records the rules currently registered, and keeps literature candidates separate from
validated code. It is intentionally narrower than the general
[rearrangement review](rearrangement_smarts.md).

Related records:

- [2026-09-27 CUDA audit](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md): pool
  reach (§3), the ring-rule and duplicate-template drop (§5) and the per-class gate (§7). The
  registry tests are in [`test_masster_smarts.py`](../../tests/test_masster_smarts.py).
- [2026-09-28 primary-literature follow-up](../notes/masster_excision_literature_2026_09_28.md):
  an uncovered N-phenylbenzamide O-transfer branch, and mass coverage versus product
  connectivity. Its candidates have not been gated.
- [SMARTS pathway plan](../plans/masster/masster_smarts_pathway_plan.md): planned use of these
  rules inside the MASSter pathway (rule provenance to the intensity head, gated rule steps on
  candidates, `[M-H]-`), including `t1i`, the first templates-on arm.
- [2026-09-13 set and literature audit](../findings/masster/masster_smarts_review_2026_09_13.md):
  earlier eligibility checks and positive-ion expansion priorities.
- Paper URLs are attached to every MASSter-only rule definition in
  [`masster_templates.py`](../../src/fragnnet/models/masster/masster_templates.py), disabled ones
  included, except `tertiary_aryl_sulfonamide_so2`.

## Scope and admission rule

MASSter does not need every chemically valid SMARTS rule. Its ordinary mechanism
already enumerates bond-cut masks and the connected precursor components produced by
those cuts. A SMARTS rule is useful only when its product reaches peaks that no connected
atom set can. A new atom partition whose formula a connected atom set already has adds no
m/z coverage, only another explanation of the same peaks.

Admit a rule when all of the following hold:

1. It creates a new heavy-atom assembly, normally by reconnecting retained atoms that
   are disconnected in the precursor after the neutral atoms are removed.
2. In the polarity it is admitted for, its product explains target intensity that no
   connected atom set of the same heavy-atom composition reaches within the model's
   hydrogen-shift range, beyond random disconnected atom sets of the same size. This is the
   [2026-09-27 per-class gate](#rules-added-by-the-per-class-gate-2026-09-27); a new atom
   set alone does not pass it.
3. The precursor substructure can be expressed narrowly enough to avoid a large
   number of unsupported candidates.
4. The ion mode, adduct, scaffold, and substituent restrictions supported by the
   evidence can be encoded or otherwise enforced.
5. The retained and neutral atom maps conserve elemental composition apart from the
   model's explicit hydrogen-transfer channels.

Do not add a candidate rule merely because a rearrangement occurs. Proton transfers,
hydride shifts, resonance changes, and topology changes with the same ordinary atom
partition belong in ion-channel scoring or learned priors unless they demonstrably
recover missing candidate formulas.

Eleven rules admitted before this gate fail it or never fire on NIST20. Eight of them are
disabled and three stay pending an ablation; see
[Existing rules under the gate](#existing-rules-under-the-gate).

## How MASSter represents a rule

Templates are off by default. `masster_smarts_templates` in
[`config/template.yml`](../../config/template.yml) (default `False`) forces each matched
template's cut mask into the candidate pool and adds its assembly under that mask. No
recorded trained run has turned it on.

The curated registry is
[`masster_templates.py`](../../src/fragnnet/models/masster/masster_templates.py).
It selects shared definitions from
[`smarts_prepass.py`](../../src/fragnnet/fragmentation/smarts_prepass.py) but owns a separate
allowlist.

- Rule index `0` is the ordinary connected-component mechanism.
- SMARTS rules take one-based indices from `MASSTER_RULE_VOCABULARY`, an append-only name
  list (since 2026-09-29): 1-7 the shared neutral-loss rules, 8-10 the compound rules, 11-22
  the MASSter-only neutral-loss rules.
  - **A new rule goes at the end of that list**, and a removed rule's name stays in it, so no
    index ever shifts. Import fails if a registered rule is missing from it.
  - Every rule-indexed parameter follows that order. A SMARTS-on model records the names'
    CRC-32 hashes in its state dict (`smarts_rule_name_hashes`). A checkpoint trained on a
    shorter prefix loads with zero rows for the appended rules and ignores their templates. A
    checkpoint whose list is not a prefix, or that has no list, is refused.
  - The 2026-09-27 change renumbered every rule (26, then 14, then 22); no SMARTS checkpoint
    existed then.
- A matched template stores the precursor bond cuts and an alternative partition of
  all precursor atoms.
- A side assembled from `k` disconnected precursor pieces forms `k - 1` bonds
  (`component_formed_bond_counts`), and the `valence` H bound takes two slots back for each.
- The cut mask keeps one probability; a SMARTS match adds another assembly mechanism
  under that mask rather than duplicating the mask probability.
- Multiple mechanisms that produce the same atom set remain latent alternatives.
- Each fragment's rules (either template side) are its provenance, read in three places.
  - The model's annotation output: `fragment_rule_fragment_idxs` / `fragment_rule_idxs`.
  - The export column `pred_node_rules`, opt-in by `--return_rule_provenance`.
  - Optionally the intensity token: `masster_rule_pathway_token`, arm `p1`, off by default.
- Literature-derived rules can restrict eligible precursor types and fragmentation
  modes; unsupported matches are removed before candidate construction. The four
  registered shared rules and `tertiary_aryl_sulfonamide_so2` have no guard and match every
  precursor type. The tables below give each rule's guard.
- Generic `crf12`/`crf13` patterns and pure bond-deletion rules are intentionally
  excluded.

## Implemented allowlist

The current registry contains **15 rules: 2 compound-fragment rules and 13
neutral-loss rules**. These names are the executable source of truth. Eight were added on
2026-09-27 by the per-class gate described in
[Rules added by the per-class gate](#rules-added-by-the-per-class-gate-2026-09-27). Eight
others were disabled on 2026-09-29; see
[Disabled rules](#disabled-rules-2026-09-29).

On 2026-09-27 MASSter dropped the eleven compound ring-cut rules and `aryl_sulfonate_so3`,
and it now skips any template whose sides are all connected in the precursor. Evidence,
from `c0`'s val pool:
- **Ring rules:** forcing their cuts covers no more target intensity than random fragments
  of the same size, and their merged byproducts cover less.
- **Duplicate templates:** a template with every side connected only repeats the ordinary
  partition of its own cut.

The shared `FRAG_RULES`/`NL_RULES` keep all of them for FraGNNet's prepass
([finding](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md)).

### Compound-fragment rules

These rules perform several coordinated cuts and merge selected outer components into
a new assembly.

| Rule | Restricted scaffold | Candidate added | Guard |
|---|---|---|---|
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | N-acyl arylsulfonamide | Phenoxide of the arylsulfonyl ring and the amide carbonyl O | `[M-H]-` |
| `diacylhydrazine_carboxylate_o_transfer` | N'-tert-butyl diacylhydrazine | `[R1CO2]-` with the opposite amide O | `[M-H]-` |

### Neutral-loss rules

Here, "neutral loss" means a mapped set of precursor atoms, not just a floating-point
mass. This distinction is essential when the retained atoms require a new assembly.

| Rule | Precursor class | Nominal transformation | Exact neutral mass (Da) | Guard |
|---|---|---|---:|---|
| `aryl_sulfonamide_so2` | Secondary (N-H) aryl sulfonamide | SO2 removal with new Ar-N assembly | 63.9619 | none |
| `aryl_sulfonate_so2` | Aryl sulfonate | SO2 removal with new Ar-O assembly | 63.9619 | none |
| `n_nitramine_n2o` | N-nitramine | N2O loss with retained-atom reassembly | 44.0010 | none |
| `sulfoxide_so_loss` | Dialkyl/diaryl sulfoxide, `S=O` or `[S+][O-]` | SO loss with new C-C assembly | 47.9669 | none |
| `n_phenylbenzamide_co_smiles` | N-Phenylbenzamide | CO loss with new N-aryl assembly | 27.9949 | `[M-H]-` |
| `short_haloalkyl_ammonium_c2_transfer` | 2-Haloethyl amine/ammonium | C2H4 loss with Cl/Br/I transfer to N | 28.0313 | `[M+H]+`, `[M]+` |
| `primary_aryl_sulfonamide_so2` | Primary arylsulfonamide | SO2 removal with new Ar-N assembly | 63.9619 | `[M-H]-` |
| `tertiary_aryl_sulfonamide_so2` | Tertiary arylsulfonamide | SO2 removal with new Ar-N assembly | 63.9619 | none |
| `primary_aryl_sulfonamide_o_transfer` | Primary arylsulfonamide | A sulfonyl O transfers to the aryl ring; S, the other O and N leave | Not set; H count from the channels | `[M+H]+` |
| `substituted_aryl_sulfonamide_o_transfer` | Secondary or tertiary arylsulfonamide | As above; the whole N side leaves as R2N-S=O (`nl_side_map_nums`) | Variable; H count from the channels | `[M+H]+` |
| `phenylalanine_carboxyl_o_transfer` | Phenylalanine | NH3 and CH2CO loss; carboxyl O stays with the benzyl residue | 59.0371 (NH3 + CH2CO) | `[M+H]+` |
| `hydroxamic_acid_nh_excision` | Hydroxamic acid (either tautomer) | NH excision joining the N-hydroxyl O to the carbonyl C | 15.0109 | `[M+H]+` |
| `pfas_sulfonate_f_transfer` | Perfluoroalkyl sulfonate | FSO3- with an alpha F transferred | Variable (product mapped) | `[M-H]-` |

New rules must match the standardized molecules the model sees, i.e.
`data_utils.mol_from_smiles` output. The standardizer writes sulfoxides as `[S+][O-]` and
hydroxamic acids as the hydroximic tautomer `ON=C(O)R`. The old `S=O`-only sulfoxide
pattern matched no NIST20 molecule for that reason.

### Disabled rules (2026-09-29)

User decision: these eight are commented out in
[`masster_templates.py`](../../src/fragnnet/models/masster/masster_templates.py), to re-check on
Enveda-180. Evidence is in
[Existing rules under the gate](#existing-rules-under-the-gate).
- Their names keep their `MASSTER_RULE_VOCABULARY` slots, so uncommenting a rule restores its
  old index. `MASSTER_NUM_RULES` is 23 since the 2026-09-30 addition below.
- The code, guard entries and tests are tagged `DISABLED 2026-09-29`. The three shared rules
  stay in `NL_RULES` for FraGNNet's prepass.
- **Enveda-180 check (2026-09-29): none finds support.** Five match no Enveda-180 molecule
  (the library has no ketones). Methyl benzoate and diarylvinyl match but score 0.00
  ([finding](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md)). The probe,
  `scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py`, scores these rules from copies of
  their definitions, so re-checking them on another dataset needs no uncommenting.

| Rule | Index | Transformation | Guard when enabled | Why disabled |
|---|---:|---|---|---|
| `diarylvinyl_c2h2_loss` | 5 | C2H2 loss with new aryl-aryl assembly | none | Formula twin; no gain on `c0`'s pool |
| `diaryl_ketone_co_loss` | 6 | CO loss with new aryl-aryl assembly | none | Formula twin; no gain on `c0`'s pool |
| `aminobenzothiazole_hcn_loss` | 7 | HCN loss with ring reassembly | none | Formula twin; no gain on `c0`'s pool |
| `pyridyl_carbamate_alkyl_transfer` | 8 | Me, Et, or nBu transfer to the pyridyl-O residue | `[M+H]+` | One NIST20 molecule, no spectra |
| `methyl_benzoate_co2_transfer` | 12 | CO2 loss with methyl transfer to the aryl residue | `[M+H]+` | Formula twin; no gain on `c0`'s pool |
| `anisoyl_fluoride_co_f_transfer` | 13 | CO loss with F transfer to the anisole ring | `[M]+`, EI only | No NIST20 molecule; no EI spectra |
| `short_haloalkyl_ammonium_c3_transfer` | 15 | C3H6 loss with Cl/Br/I transfer to N | `[M+H]+`, `[M]+` | Formula twin; no gain on `c0`'s pool |
| `n_fluoro_sulfonamide_f_transfer` | 16 | Diagnostic SO2F- product with neutral ArN loss | `[M-H]-` | No NIST20 molecule |

## Rules added by the per-class gate (2026-09-27)

Candidates came from the lists in this catalog, the 09-23 audit's variants and AIMe's
worked example. Each was first counted on standardized NIST20 molecules. Each supported one
was then scored on every NIST20 spectrum of its molecules, both polarities and both
instruments:
- the target intensity its disconnected product explains through MASSter's own channels;
- counting only peaks that no connected atom set of the same composition can reach;
- against random disconnected atom sets of the same size;
- averaged per molecule, with molecule-bootstrap CIs.

Each new rule is restricted to the polarity that passes. `tertiary_aryl_sulfonamide_so2`
passes both and is left unguarded. Full table:
[finding §7](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md#7-new-rules-from-the-documented-lists).

| Rule | Mode | NIST20 molecules | Beyond random, % of intensity (95% CI) | Strength |
|---|---|---:|---|---|
| `primary_aryl_sulfonamide_so2` | `[M-H]-` | 67 | +6.39 (3.71 to 9.59) | clear |
| `primary_aryl_sulfonamide_o_transfer` | `[M+H]+` | 175 | +2.61 (1.85 to 3.45) | clear |
| `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | `[M-H]-` | 10 | +5.69 (0.79 to 12.85) | clear |
| `diacylhydrazine_carboxylate_o_transfer` | `[M-H]-` | 3 | +49.6 (0.0 to 78.7); at least 1% in 59% of spectra | positive; CI reaches 0 |
| `pfas_sulfonate_f_transfer` | `[M-H]-` | 10 | +1.43 (0.00 to 3.38); at least 1% in 22% | positive; CI reaches 0 |
| `phenylalanine_carboxyl_o_transfer` | `[M+H]+` | 66 | +0.15 (0.00 to 0.36) | weak |
| `tertiary_aryl_sulfonamide_so2` | both (unguarded) | 174 / 57 | +0.11 (0.01 to 0.27) / +0.07 (0.01 to 0.18) | weak |
| `hydroxamic_acid_nh_excision` | `[M+H]+` | 47 | +0.04 (0.01 to 0.08) | weak |

Candidates not admitted:
- **No NIST20 support:** `nitrobenzyl_hydroxybenzoate_co2_coupling`,
  `prazole_sulfone_so2_extraction`, `pyrimidylsulfanyl_benzamide_dual_smiles`,
  `benzaldimine_hcn_benzyl_transfer`, `dibenzylaniline_benzene_loss_transfer`, and the
  perfluoroalkyl sulfinate variant.
- **Three or fewer NIST20 molecules:** `n_aryl_methacrylamide_co_claisen`,
  `proline_benzyl_ester_n_benzyl_transfer`, `dichlorvos_phosphate_halogen_transfer`,
  `pirimicarb_pyrimidinyl_carbamate_methyl_transfer`.
- **Inconclusive:** `n_methylanilinoethanol_o_smiles`, 2 molecules, where the reachability
  search hit its cap.
- **Not encoded, product map unresolved:** `sulfonylurea_herbicide_n_o_smiles`
  (16 molecules) and `benzyl_phenyl_sulfone_benzyl_transfer` (10).
- **Null:** SO2 loss from primary sulfonamides in `[M+H]+`.
- **ChemFrag's 32 rearrangement transforms, all closed (2026-09-28):** 26 change no
  heavy-atom bond. Thioether cyclization (#16) scores below random, and every Wagner–Meerwein
  fragment has a formula twin
  ([triage](../findings/masster/chemfrag_rearrangement_triage_2026_09_14.md#unique-coverage-check-2026-09-28)).

### Existing rules under the gate

The 14 rules registered before 2026-09-27 were scored the same way. Three pass, eight fail,
and three never fire on NIST20. Numbers are from
`output/smarts_reach_2026_09_27/rule_gate_log.txt`, except the last three rows' molecule
counts, which are a direct SMARTS count on 2026-09-29. The pool column is
[finding §3.1](../findings/masster/masster_smarts_cuda_audit_2026_09_27.md#31-coverage-the-templates-add-beyond-the-pool),
measured on `c0`'s `[M+H]+` pool only.

| Rule | Mode | NIST20 molecules | Beyond random, % (95% CI) | Gain on `c0`'s pool | Status |
|---|---|---:|---|---|---|
| `aryl_sulfonate_so2` | `[M+H]+` / `[M-H]-` | 86 / 121 | +2.06 (1.06 to 3.59) / +6.40 (4.59 to 8.50) | +1.8% | passes |
| `aryl_sulfonamide_so2` | `[M+H]+` / `[M-H]-` | 446 / 237 | +0.29 (0.19 to 0.42) / +3.75 (2.76 to 4.77) | +0.22% | passes |
| `n_nitramine_n2o` | `[M+H]+` / `[M-H]-` | 10 / 5 | +0.20 (0.00 to 0.48) / +6.98 (0.00 to 20.77) | +0.43% | passes; CIs reach 0 |
| `short_haloalkyl_ammonium_c2_transfer` | `[M+H]+` | 29 | −0.01 (−0.07 to 0.04) | +0.49% | fails; kept pending ablation |
| `sulfoxide_so_loss` | `[M+H]+` / `[M-H]-` | 90 / 35 | −0.05 (−0.22 to 0.05) / 0.00 | not measured | fails; kept pending ablation |
| `diaryl_ketone_co_loss` | `[M+H]+` / `[M-H]-` | 373 / 137 | 0.00 / 0.00 | none | fails; disabled 2026-09-29 |
| `diarylvinyl_c2h2_loss` | `[M+H]+` / `[M-H]-` | 61 / 47 | 0.00 / 0.00 | none | fails; disabled 2026-09-29 |
| `aminobenzothiazole_hcn_loss` | `[M+H]+` / `[M-H]-` | 35 / 4 | 0.00 / 0.00 | none | fails; disabled 2026-09-29 |
| `methyl_benzoate_co2_transfer` | `[M+H]+` | 313 | 0.00 | none | fails; disabled 2026-09-29 |
| `short_haloalkyl_ammonium_c3_transfer` | `[M+H]+` | 7 | 0.00 | none | fails; disabled 2026-09-29 |
| `n_phenylbenzamide_co_smiles` | `[M-H]-` | 199 | 0.00 | not measured | fails; kept pending ablation |
| `pyridyl_carbamate_alkyl_transfer` | `[M+H]+` | 1, no spectra | — | — | never fires on NIST20; disabled 2026-09-29 |
| `n_fluoro_sulfonamide_f_transfer` | `[M-H]-` | 0 | — | — | never fires on NIST20; disabled 2026-09-29 |
| `anisoyl_fluoride_co_f_transfer` | `[M]+`, EI | 0 | — | — | never fires on NIST20; disabled 2026-09-29 |

The CO, CO2, HCN, C2H2 and C3H6 losses score exactly zero because their product formula
always equals that of some connected atom set, e.g. dropping a different C and O. So they add
no m/z coverage, only a more plausible explanation. The sulfonyl and nitramine rules are
unique because their loss removes a bridging heteroatom that no other cut can remove while
leaving the rest connected.

User decisions (2026-09-29):
- **Disabled, to re-check on Enveda-180:** the five failing rules that add nothing on `c0`'s
  pool either, and the three that never fire
  ([Disabled rules](#disabled-rules-2026-09-29)).
- **Kept until an ablation decides them:** the other three failing rules. Pool reach is the
  only measured argument for them, and it holds for one. A formula twin can exist but be
  missing from the proposal's 256-fragment pool, which is how the C2 haloalkyl rule adds
  +0.49% on `c0`'s pool. Sulfoxide and N-phenylbenzamide were not measured on a pool.
- **Outliers (2026-09-29):** averaged over NIST20 the rules are negligible, but they reach
  spectra where much of the intensity sits at rule-only m/z. On `i1`'s val set those are 37
  `[M+H]+` spectra with cosine 0.25-0.56. They are not the floor: none of the bottom 500
  spectra carries rule-only intensity. `[M-H]-` is untested
  ([finding](../findings/masster/masster_smarts_outlier_reach_2026_09_29.md)).
- **Enveda-180 (2026-09-29):** sulfoxide passes there, +0.56% in `[M-H]-` (CI 0.23 to 1.01).
  N-phenylbenzamide still scores 0.00, and no Enveda-180 molecule matches the C2 haloalkyl rule
  ([finding](../findings/masster/masster_smarts_enveda_gate_2026_09_29.md)).

## Rule added from the `i1` worst-prediction read (2026-09-30)

`substituted_aryl_sulfonamide_o_transfer` extends the primary O-transfer to secondary and
tertiary arylsulfonamides (`[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3;!H2:5]`, `[M+H]+` only).
- **Why.** In `i1`'s val tail, 59% of the uncovered intensity of sulfonamide molecules is ions that
  lost S and part of the O. The implied losses are X–N–S–O units (HNOS, C3H7NOS, CH3BrNOS).
- **Product.** The leaving neutral is S, one O and the whole N side, which needs the new
  `NeutralLossRule.nl_side_map_nums`. A match is skipped when the N side is fused to the aryl ring
  (sultams).
- **Standardization.** N-sulfonyl amidines are written as the imine `S(=O)(=O)N=C`, so they do
  not match.
- **Gate.** Same scoring as the per-class gate, on val and on the 4,000-molecule train sample,
  molecules the primary rule does not match:

| Mode | Molecules | Beyond random, % of intensity (95% CI) | Strength |
|---|---:|---|---|
| `[M+H]+`, val | 115 | +1.81 (0.70 to 2.80) | clear |
| `[M+H]+`, train sample | 104 | +2.33 (1.64 to 3.20) | clear |

- **Where it matters.** On the val tail (inst-tol cosine < 0.2) it reaches 74 spectra of 26
  molecules, 15 of them at ≥ 10% rule-only intensity.
- **Not admitted.**
  - Alkyl sulfonamide O-transfer: 18 val molecules, +1.67 (0.00 to 5.02); 0.00 on train.
  - Aryl sulfone O-transfer: 25 molecules, +0.31 (0.03 to 0.69).
- **Evidence.** [Worst-predictions finding](../findings/masster/masster_i1_worst_predictions_2026_09_30.md);
  scripts in `output/i1_worst_predictions_2026_09_30/smarts_o_transfer/`.

## Literature additions implemented in 2026-08

These additions have a clear transferred heavy atom or group, a constrained substrate
class, and a product atom partition that ordinary cuts do not produce. They use the
ionization guards shown below. The short-chain halogen family requires separate C2 and C3
templates because the leaving atom maps differ.

The "Why it fits MASSter" column is the atom-partition rationale at admission, not measured
coverage. Under the 2026-09-27 gate none of these seven rules adds m/z coverage: four score
at or below random and three never fire on NIST20
([Existing rules under the gate](#existing-rules-under-the-gate)). Only
`n_phenylbenzamide_co_smiles` and `short_haloalkyl_ammonium_c2_transfer` are still registered;
the other five were disabled on 2026-09-29.

| Priority | Rule or family | Evidence-specific transformation | Eligibility | Why it fits MASSter | Source |
|---:|---|---|---|---|---|
| 1 | `n_phenylbenzamide_co_smiles` | Deprotonated N-phenylbenzamide loses CO and forms a new N-aryl bond | `[M-H]-` | CO is internal to the path between retained groups | [Rapid Communications in Mass Spectrometry, 2015](https://pubmed.ncbi.nlm.nih.gov/26377014/) |
| 2 | `methyl_benzoate_co2_transfer` | Protonated aryl methyl ester loses CO2 while methyl becomes attached to the aryl residue | `[M+H]+` | Retains groups from opposite sides of the ester carbonyl | [Journal of Mass Spectrometry, 2018](https://pubmed.ncbi.nlm.nih.gov/29441643/) |
| 3 | `anisoyl_fluoride_co_f_transfer` | Anisoyl fluoride radical cation loses CO while F migrates from the acyl carbon to the ring | `[M]+`, EI | Explicit heavy-atom migration creates a new retained assembly | [Organic Mass Spectrometry, 1991](https://doi.org/10.1002/oms.1210261122) |
| 4 | `short_haloalkyl_ammonium_c2_transfer` and `short_haloalkyl_ammonium_c3_transfer` | Amine/ammonium chains of two or three methylenes lose C2H4/C3H6 and retain Cl, Br, or I on N | `[M+H]+` or `[M]+` | Strong chain-length and halogen guards make the transfer rule narrow | [Journal of the American Society for Mass Spectrometry, 2015](https://doi.org/10.1007/s13361-015-1261-4) |
| 5 | `pyridyl_carbamate_alkyl_transfer` | Protonated pyridyl dialkylcarbamates transfer Me, Et, or nBu to the pyridyl residue | `[M+H]+` | The carbamate linker is discarded while an alkyl group is retained remotely | [Tetrahedron, 2015](https://www.sciencedirect.com/science/article/pii/S004040201401518X) |
| 6 | `n_fluoro_sulfonamide_f_transfer` | Deprotonated ArSO2NF gives SO2F- plus the aryl-nitrogen remainder | `[M-H]-` | F and the sulfonyl group become a new charged assembly | [Rapid Communications in Mass Spectrometry, 2024](https://doi.org/10.1002/rcm.9722) |

The anisoyl rule also requires `EI` fragmentation mode and is dropped when that metadata is
missing. NIST20 tandem spectra are HCD or CID and no MASSter config lists `EI`, so it never
fired, and it was disabled on 2026-09-29. The other new rules are restricted by precursor type even when fragmentation mode is
unavailable. The general shared BFS registries remain unchanged.

## Verified ESI-CID expansion candidates

The following candidates were checked against primary reports that explicitly used
electrospray ionization followed by collisional activation. None of these candidates
requires EI metadata. The 2026-09-27 gate screened most of them one at a time; the Status
column records the outcome, detailed in
[Rules added by the per-class gate](#rules-added-by-the-per-class-gate-2026-09-27). An
untested row still needs an exact mapped product, a coverage-delta check, and
positive/negative SMARTS tests.

Priority `A` means the paper identifies the transferred heavy atom or group and the
proposed product appears to combine precursor atoms that ordinary bond deletion would
separate. Priority `B` means the chemistry is supported but the scope, multistage path,
or product mapping needs more work. Priority `C` requires metadata or graph handling
beyond a narrow small-molecule SMARTS.

| Priority | Proposed family | Status | ESI eligibility | Rearranged candidate to encode | Remaining guard or mapping issue | Primary source |
|---|---|---|---|---|---|---|
| A | `n_methylanilinoethanol_o_smiles` | Inconclusive: 2 molecules, reachability search capped | `[M-H]-`, CID or HCD | Phenoxide retains the aryl group and terminal alcohol O while the intervening N-alkyl chain leaves | Require N-alkyl substitution; the N-H analogue follows a different pathway and N-ethyl gives much less product | [JASMS, 2022](https://doi.org/10.1021/jasms.2c00210) |
| A | `n_acyl_aryl_sulfonamide_carbonyl_o_transfer` | Implemented 2026-09-27 | `[M-H]-`, CID | Phenoxide contains the arylsulfonyl ring plus the oxygen transferred from the remote amide carbonyl | Keep N-acyl sulfonamides separate from primary sulfonamides and the existing SO2 rules | [JASMS, 2021](https://pubmed.ncbi.nlm.nih.gov/33586949/) |
| A | `diacylhydrazine_carboxylate_o_transfer` | Implemented 2026-09-27 | `[M-H]-`, CID | `[R1CO2]-` is assembled by transferring the opposite amide oxygen to the R1 carbonyl carbon | Evidence is for `R1C(O)-N(t-Bu)NH-C(O)R2`; encode the tert-butyl and side-specific O map | [PLOS ONE, 2013](https://doi.org/10.1371/journal.pone.0063097) |
| A | `nitrobenzyl_hydroxybenzoate_co2_coupling` | Not admitted: no NIST20 molecules | `[M-H]-`, CID | CO2 loss is followed by coupling of the retained 4-nitrobenzyl and hydroxyaryl groups | Require a phenolic anion and the para-nitrobenzyl ester; validate vanillate substitutions separately | [JASMS, 2016](https://doi.org/10.1007/s13361-016-1339-7) |
| A | `proline_benzyl_ester_n_benzyl_transfer` | Not admitted: three or fewer NIST20 molecules | `[M+H]+`, CID | Benzyl transfers from the ester oxygen to the proline N while H2O and CO are lost | Separate the N-benzyl product from the competing benzyl-cation and hydride-transfer channels | [Journal of Mass Spectrometry, 2013](https://doi.org/10.1002/jms.3162) |
| B | `pirimicarb_pyrimidinyl_carbamate_methyl_transfer` | Not admitted: three or fewer NIST20 molecules | Positive ESI, collisional activation | Proposed methyl transfer in pirimicarb accompanies the reported nominal 57 Da loss | Current pyridyl rule does not match this pyrimidine scaffold; extract Figure 6 atom mapping and prove additional coverage before implementation | [Molecules, 2018, section 2.3/Figure 6](https://pmc.ncbi.nlm.nih.gov/articles/PMC6222809/) |
| A | `phenylalanine_carboxyl_o_transfer` | Implemented 2026-09-27 (weak) | `[M+H]+`, CID/MS3 | After NH3 and CH2CO loss, carboxyl O is retained with the benzyl/aryl residue as a hydroxybenzyl cation | This is sequential and has competing benzyl-O versus ortho-aryl-O products with the same formula | [Frontiers in Chemistry, 2023](https://doi.org/10.3389/fchem.2022.1094329) |
| A | `primary_aryl_sulfonamide_o_transfer` | Implemented 2026-09-27 | `[M+H]+`, CID/MS3 | A sulfonyl O transfers to the aryl ring to form a substituted phenoxonium ion | Product abundance depends strongly on ring substituents; require a supported conjugating substituent guard | [International Journal of Mass Spectrometry, 2019](https://doi.org/10.1016/j.ijms.2018.09.033) |
| B | `prazole_sulfone_so2_extraction` | Not admitted: no NIST20 molecules | Positive ESI, CID | Prazole sulfones lose the SO2 linker and form an N-(2-benzimidazolyl)pyridine framework | Restrict to sulfones; sulfoxides rarely form the product, thioethers require an additional SH-radical loss, and pyridine N-oxides are negative controls | [Molecules, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10420917/) |
| B | `pyrimidylsulfanyl_benzamide_dual_smiles` | Not admitted: no NIST20 molecules | `[M-H]-`, CID | Two competing Smiles products retain either the dimethoxypyrimidyl side or the N-phenylamide side across the thiosalicyl linker | Requires two product templates and substituent-sensitive scoring | [JASMS, 2007](https://doi.org/10.1016/j.jasms.2007.07.012) |
| B | `sulfonylurea_herbicide_n_o_smiles` | Not encoded: product map unresolved (16 molecules) | `[M-H]-`, CID | Nitrogen- and oxygen-initiated Smiles channels produce alternative remotely assembled fragments | Evidence is strongest for rimsulfuron and nicosulfuron, not a generic sulfonylurea pattern | [Chinese Journal of Chemistry, 2012](https://doi.org/10.1002/cjoc.201200610) |
| B | `n_terminal_protected_amino_acid_o_migration` | Untested | Negative ESI, CID | A carboxyl oxygen participates through a five-membered transition state in protected Ser/Thr/Tyr derivatives | Exact retained atom maps must be extracted; activation depends on whether the C terminus is blocked | [Indian Journal of Biochemistry & Biophysics, 2006](https://pubmed.ncbi.nlm.nih.gov/17285801/) |
| B | `benzaldimine_hcn_benzyl_transfer` | Not admitted: no NIST20 molecules | `[M+H]+`, CID | Protonated N-benzylbenzaldimines lose HCN while benzyl transfer couples atoms from opposite sides of the imine | Start with the reported p-methoxybenzylidene series; ring electronics control transfer | [JASMS, 2014](https://doi.org/10.1007/s13361-014-0935-7) |
| B | `benzyl_phenyl_sulfone_benzyl_transfer` | Not encoded: product map unresolved (10 molecules) | `[M+H]+`, CID | Benzyl transfers across the sulfone before formation of characteristic product ions | The abstract establishes transfer across substituted analogues but the exact charged product map still requires full-scheme extraction | [Journal of Mass Spectrometry, 2021](https://pubmed.ncbi.nlm.nih.gov/33295108/) |
| B | `dibenzylaniline_benzene_loss_transfer` | Not admitted: no NIST20 molecules | `[M+H]+`, CID | A benzyl group transfers to a benzylic or aniline ring, followed by benzene loss | Two transfer destinations and several subsequent H shifts require separate atom-map validation | [Journal of Mass Spectrometry, 2017](https://pubmed.ncbi.nlm.nih.gov/28109035/) |
| B | `chromene_sulfonamide_sulfonyl_transfer` | Untested | `[M+H]+`, CID | Sulfonyl cation transfers from N to a phenyl carbon, then benzenesulfinic acid is lost with cyclization | Extremely narrow chromene-imine scaffold; electron-donating substituents favor the pathway | [Rapid Communications in Mass Spectrometry, 2016](https://pubmed.ncbi.nlm.nih.gov/26661975/) |
| B | `n_aryl_methacrylamide_co_claisen` | Not admitted: three or fewer NIST20 molecules | `[M+H]+`, CID | Amide Claisen rearrangement enables CO loss and a cyclized retained assembly | Electron-donating para substituents favor the reaction; product topology and atom map are multistep | [JASMS, 2012](https://pubmed.ncbi.nlm.nih.gov/23055072/) |
| B | `dichlorvos_phosphate_halogen_transfer` | Not admitted: three or fewer NIST20 molecules | `[M+H]+`, CID | P-O cleavage is followed by halogen-assisted transfer that forms protonated dimethyl halophosphate | Narrow organophosphate scaffold; Cl, Br, and F analogues were examined, but halogen and H maps must both be verified | [JASMS, 2017](https://doi.org/10.1007/s13361-017-1736-6) |
| B | `pfas_sulfonyl_f_transfer` | Sulfonate implemented 2026-09-27 as `pfas_sulfonate_f_transfer`; sulfinate: no NIST20 molecules | Negative ESI, CID/AIF | Perfluoroalkyl sulfonate or sulfinate loses CF2 and transfers F to give `FSO3-` or `FSO2-` | Keep sulfonates and sulfinates separate and restrict chain patterns to the studied PFAS series | [Rapid Communications in Mass Spectrometry, 2026](https://doi.org/10.1002/rcm.70035) |
| C | `protonated_glycan_fucose_transfer` | Untested | `[M+H]+` or `[M+NH4]+`, CID/in-source activation | A terminal fucose is retained on a remote glycan fragment, including inter-antenna transfer | Do not apply to `[M+Na]+` or `[M-H]-`; arbitrary glycan rings, multiple acceptor sites, and mobile-proton dependence make one SMARTS insufficient | [JASMS, 2002](https://pubmed.ncbi.nlm.nih.gov/11951970/), [Angewandte Chemie, 2018](https://pubmed.ncbi.nlm.nih.gov/29688603/) |
| C | `trimethyllysine_peptide_methyl_transfer` | Untested | Singly charged protonated peptide, CID | A methyl from trimethyllysine is retained on a remote C-terminal Arg fragment after backbone cleavage | Explicit precursor charge state and peptide-residue context are required; the transfer is absent for doubly charged ESI ions | [JASMS, 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2709603/) |
| C | `phosphopeptide_phosphate_relocation` | Untested | Positive or negative ESI, low-energy CID | A phosphate can relocate before backbone cleavage, yielding non-native phosphorylated sequence ions | Charge, basic residues, residue spacing, and activation method matter; HCD does not show the same slow-CID relocation | [International Journal of Mass Spectrometry, 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4375673/), [JASMS, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5479408/) |

## EI follow-up screen (2026-10-01)

Three EI candidate families were screened; **none is admitted**. Anisoyl fluoride and
narrow para-hydroxyazobenzene products add no nominal coverage beyond ordinary cuts and
natural isotope support. The N-H aromatic-acyl arylsulfonamide O-transfer candidate has
seven training molecules and no validation support. The production allowlist and guards
remain unchanged. See the [screen finding](../findings/masster/masster_ei_smarts_screen_2026_10_01.md)
for exact SMARTS scope, null controls, isotope collisions and limits. This closes the
coverage screen only; it does not establish an intensity-model null.

## Additional literature backlog

These transformations are plausible MASSter rules but need more spectra, narrower
guards, or a demonstrated coverage delta before implementation. "Provisional" means
the mechanism or scope is not yet strong enough to encode as a production rule.

| Proposed family | Transformation to investigate | Main restriction or concern | Source |
|---|---|---|---|
| `tosyl_imidamide_transfer` | Tosyl migrates to an anilinic N while phenylacetonitrile is lost | Limited imidamide series; substituent effects matter | [Journal of the American Society for Mass Spectrometry, 2015](https://pubs.acs.org/doi/10.1007/s13361-015-1156-4) |
| `catechol_phenethylamide_acylation` | Activated phenethylamides cyclize by internal acylation and then lose NH3 | Usually requires a highly activated, often dihydroxy aryl ring | [Journal of Mass Spectrometry, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5555735/) |
| `aryl_phosphorothioate_s_transfer` | Protonated phosphorothioates ultimately form ArS+ after sequential methanol and phosphonate losses | Multi-step and likely needs pathway-specific scoring | [Journal of Mass Spectrometry, 2019](https://pubmed.ncbi.nlm.nih.gov/30569428/) |
| `dimethyl_diester_ketene_methoxy_transfer` | Dimethyl diesters lose ketene with methoxy migration | Old EI evidence and broad ester matching risk | [Journal of the Chemical Society C, 1968](https://doi.org/10.1039/J39680000202) |
| `benzoyl_carboxymethyl_transacylation` | Benzoyl migrates to a carboxyl group before benzoic-acid and CO losses | Narrow heteroaromatic scaffold and sequential mechanism | [Rapid Communications in Mass Spectrometry, 2009](https://pubmed.ncbi.nlm.nih.gov/19504483/) |
| `n_acyl_glycinamide_residue_loss` | N-acyl glycinamide loses the complete glycine residue to form a nitrilium-type ion | Confirm exact retained map and distinguish homologues | Literature lead; primary source still needs rule-level extraction |
| `n_acyl_beta_alaninamide_residue_loss` | N-acyl beta-alaninamide loses the complete beta-alanine residue | Homologue-specific chain guard required | Literature lead; primary source still needs rule-level extraction |
| `aryl_azo_n2_loss` | Aryl azo systems lose N2 and couple retained carbon centers | Charge state and substitution strongly affect pathway | Literature lead; primary source still needs rule-level extraction |
| `diaryl_sulfone_so2_loss` | Diaryl sulfone loses SO2 and joins the two aryl residues | Evidence is mainly EI; avoid broad sulfone matches | Literature lead; primary source still needs rule-level extraction |
| `heteroaryl_alpha_diketone_co_loss` | A heteroaryl alpha-diketone loses internal CO and reconnects retained groups | Evidence centered on 2-pyridyl systems | Literature lead; primary source still needs rule-level extraction |
| `aromatic_benzyl_ether_ch2o_loss` | Benzyl aryl ethers lose CH2O with retained-group reassembly | Must distinguish ordinary benzyl cleavage | Literature lead; primary source still needs rule-level extraction |
| `aminoacyl_benzylamine_formamide_loss` | Formamide loss transfers/reconnects the benzyl-bearing group | Narrow substrate family; map needs validation | Literature lead; primary source still needs rule-level extraction |
| `n_sulfonyl_phenylacetamide_hnco_loss` | HNCO loss accompanies benzyl-anion transfer | Negative-ion and substituent constraints needed | Literature lead; primary source still needs rule-level extraction |
| `meta_donor_benzamide_no_exchange` | N/O exchange gives a phenolic product plus benzonitrile | Requires meta electron donor; easy to overmatch | Literature lead; primary source still needs rule-level extraction |
| `benzyl_phenylalaninate_h2o_co_transfer` | Sequential H2O and CO loss retains a transferred benzyl group | Multi-step; exact charged product map required | Literature lead; primary source still needs rule-level extraction |
| `carbamate_co2_reassembly` | Carbamate CO2 loss reconnects the groups on either side | Too generic until the supported connectivity is resolved | Provisional mechanism |
| `polyfluoroaryl_cf2_ring_contraction` | Polyfluoroaryl ions lose CF2 with ring contraction | EI-only and specialized | Literature lead; primary source still needs rule-level extraction |
| `insti_hnco_benzyl_migration` | Integrase-inhibitor scaffold loses HNCO with benzyl migration | Provisional, drug-class-specific mechanism | Provisional mechanism |
| `tms_acetate_ch2co_migration` | Trimethylsilylacetic derivatives lose CH2CO with dimethylsilyl migration | Derivatization-specific EI rule | [Rapid Communications in Mass Spectrometry, 2002](https://pubmed.ncbi.nlm.nih.gov/11921371/) |
| `tms_steroid_a_ring_migration` | Trimethylsilylated cholesterol loses the A ring with TMS migration | Extremely scaffold- and derivatization-specific | [Journal of Mass Spectrometry, 2014](https://pubmed.ncbi.nlm.nih.gov/24234867/) |

Backlog entries without a linked primary source are research notes, not implementation
specifications. Locate and inspect the primary paper before turning any of them into
SMARTS.

## Excluded or scoring-only transformations

| Transformation | Decision | Reason |
|---|---|---|
| McLafferty and other proton/hydride rearrangements | Scoring or hydrogen-transfer channel | The heavy-atom subset is generally already reachable by ordinary cuts |
| N-benzylaniline benzyl-cation transfer | Exclude as candidate rule | The retained benzyl atom set is produced by ordinary bond cleavage despite the product topology change ([study](https://pubmed.ncbi.nlm.nih.gov/22823406/)) |
| Curtius/Wolff-type N2 loss | Exclude unless a coverage counterexample is found | The retained heavy-atom subset is normally an ordinary connected remainder ([study](https://pubmed.ncbi.nlm.nih.gov/40512019/)) |
| Metal-assisted AZT/phosphoramidate methoxy shifts | Exclude from the core registry | Strong metal/adduct dependence and often no new candidate mass ([study](https://pubmed.ncbi.nlm.nih.gov/15712358/)) |
| Remote fluorobenzyl transfer between amide nitrogens | Scoring-only by default | Transfer alone changes connectivity but not the precursor atom mask or mass ([study](https://pubmed.ncbi.nlm.nih.gov/22215578/)) |
| Generic cyclization, resonance, or charge migration | Exclude as candidate rule | Same retained atom set; use the model's mechanism and ion-channel scores |
| Pure single- or multi-bond deletion | Exclude from SMARTS | Already represented by ordinary MASSter masks and connected components |
| Broad `crf12`/`crf13` patterns | Exclude | Match breadth creates unsupported candidates without a scaffold-specific mechanism |

## Validation gate for every new rule

A literature mechanism enters the registry only after checks 1-6 and 8. Check 7 decides
whether templates are enabled by default; no rule has passed it, because no ablation has
run.

1. **SMARTS specificity:** positive examples match; close negative controls do not.
2. **Map correctness:** retained and neutral map numbers partition the intended
   precursor atoms with no omissions or overlap.
3. **Formula conservation:** calculated neutral composition and exact mass agree with
   the proposed mechanism; hydrogen differences stay within the configured channel
   range.
4. **Coverage delta:** admission criterion 2, measured with the 2026-09-27 per-class gate.
5. **Candidate load:** corpus-wide match counts and candidates per molecule remain
   acceptably small.
6. **Spectral evidence:** the added candidate explains reproducible peaks in held-out
   spectra, not only the paper's demonstration molecule.
7. **Ablation:** validation recall or likelihood improves without degrading common
   scaffolds or overwhelming the mechanism softmax.
8. **Tests:** add focused positive, negative, deduplication, partition, rule-index, and
   model-forward tests.

For a proposed rule, record the supported ion mode and adduct explicitly. If the
current template interface cannot enforce those restrictions, extend the interface
only alongside the first rule that requires them; do not add speculative generic
configuration.

## Maintenance

When the allowlist changes, update this document and record the date. Append a new rule's
name to `MASSTER_RULE_VOCABULARY`, and never remove or reorder a name there; to drop a rule,
comment it out of the allowlist, as on 2026-09-29
([How MASSter represents a rule](#how-masster-represents-a-rule)). `MASSTER_NUM_RULES`
counts the vocabulary, not the registered rules. Keep broad chemistry surveys in
[`rearrangement_smarts.md`](rearrangement_smarts.md); this file should remain the decision
record for MASSter candidate generation.
