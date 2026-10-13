# MASSter SMARTS rules on Enveda-180: per-class gate

> **Status:** FINDING · **Updated:** 2026-09-29
> Model-free coverage gate, the NIST20 measure, run on every SMARTS rule in the code. Five of
> 41 rules add intensity beyond random on Enveda-180, all of them S=O rules; only one reaches
> 1%. None of the eight rules disabled on 2026-09-29 finds support. No model was trained.

## Answer

The question was whether the SMARTS rules, including the eight disabled on 2026-09-29 to
re-check here, are useful on Enveda-180 (Bruker timsTOF, `QTOF`/`CID`, Enamine molecules).

- **5 of 41 rules pass** (95% CI above 0 in an admitted polarity):
  - `primary_aryl_sulfonamide_so2`;
  - `aryl_sulfonamide_so2`;
  - `sulfoxide_so_loss`;
  - `primary_aryl_sulfonamide_o_transfer`;
  - `tertiary_aryl_sulfonamide_so2`.
- **Only one reaches 1% of intensity:** primary sulfonamide SO2 in `[M-H]-`, +1.48%. The
  others are +0.05% to +0.82%.
- **The rules are 2-10x weaker than on NIST20.** For example, `aryl_sulfonamide_so2` gives
  +0.82% in `[M-H]-` here against +3.75% on NIST20, and primary sulfonamide O transfer gives
  +0.23% in `[M+H]+` against +2.61%.
- **None of the eight disabled rules is useful here:**
  - five match no Enveda-180 molecule;
  - methyl benzoate (1,269 molecules) and diarylvinyl (189) score 0.00;
  - diaryl ketone cannot match: the library has no aryl ketone (0 of 183,476 molecules).
- **Sulfoxide is the one change of verdict.** On NIST20 it scored −0.05 / 0.00; here it gives
  +0.56% in `[M-H]-` (CI 0.23 to 1.01, 208 molecules). It is one of the three failing rules the
  catalog keeps pending an ablation.
- **25 rules match no molecule at all**, including eight registered rules: aryl sulfonate,
  nitramine, C2 haloalkyl, N-acyl sulfonamide, diacylhydrazine, phenylalanine, hydroxamic acid
  and PFAS sulfonate.

## Method

Probe: [`debug_masster_smarts_rule_gate_enveda.py`](../../../scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py).
It reuses the NIST20 gate's helpers from `debug_masster_smarts_rule_gate.py`: the product sets,
the connected-twin search and MASSter's channel function. So the measure is the
[per-class gate](../../reference/masster_smarts_rules.md#rules-added-by-the-per-class-gate-2026-09-27):
- the target intensity a rule's disconnected product explains through MASSter's channels
  (valence bound, ±4 H, no CMF);
- only on peaks that no connected atom set of the same heavy-atom composition reaches within
  4 H;
- against random disconnected atom sets of the same size;
- averaged per molecule, with molecule-bootstrap 95% CIs.

What differs from the NIST20 run:
- **Rules:** 41, every SMARTS rule in the code. That is:
  - the 14 registered MASSter rules;
  - the 8 disabled on 2026-09-29, from copies of their commented-out definitions;
  - the 12 dropped on 2026-09-27;
  - the 5 shared rules MASSter never used (`crf12_0`, `crf12_1`, `crf13_2`,
    `linear_tricyclic_bis_c4_loss`, `morphinan_ring_cleavage`);
  - the probe's 2 other candidates.
- **Polarity:** every rule is scored in both `[M+H]+` and `[M-H]-`. Verdicts use the polarities
  the rule is (or was) admitted for.
- **Sample:** the census covers all 177,341 molecules with an `[M+H]+` or `[M-H]-` spectrum.
  The gate scores up to 1,000 matched molecules per rule, drawn with a fixed seed.
- **Tolerance:** 10 ppm with a 200 m/z floor, the Enveda configs' QTOF loss tolerance. It is
  also the NIST20 run's OB/OB_F tolerance.

Verdict labels: **pass** means the CI's lower bound is above 0; **clear** means a pass at 1% or
more. The NIST20 catalog has no written threshold.

## Results

Rules with at least one matched Enveda-180 molecule. Values are % of intensity beyond random
(95% CI); `*` marks a polarity the rule is not admitted for.

| Rule | Group | Molecules matched | Scored `[M+H]+` / `[M-H]-` | `[M+H]+` | `[M-H]-` | Verdict |
|---|---|---:|---:|---|---|---|
| `primary_aryl_sulfonamide_so2` | registered, `[M-H]-` | 822 | 704 / 659 | +0.02 (−0.00 to 0.04)* | +1.48 (1.02 to 2.02) | pass, clear |
| `aryl_sulfonamide_so2` | registered, unguarded | 7,245 | 808 / 821 | +0.16 (0.10 to 0.23) | +0.82 (0.56 to 1.13) | pass |
| `sulfoxide_so_loss` | registered, unguarded | 729 (565 with product) | 517 / 208 | +0.05 (0.02 to 0.09) | +0.56 (0.23 to 1.01) | pass |
| `primary_aryl_sulfonamide_o_transfer` | registered, `[M+H]+` | 822 | 704 / 659 | +0.23 (0.16 to 0.31) | +0.06 (0.03 to 0.10)* | pass |
| `tertiary_aryl_sulfonamide_so2` | registered, unguarded | 7,135 | 931 / 324 | +0.05 (0.02 to 0.10) | +0.15 (0.03 to 0.34) | pass |
| `n_phenylbenzamide_co_smiles` | registered, `[M-H]-` | 8,295 | 944 / 690 | 0.00* | 0.00 | fails |
| `methyl_benzoate_co2_transfer` | disabled, `[M+H]+` | 1,269 | 903 / 449 | −0.01 (−0.02 to 0.00) | 0.00* | fails |
| `diarylvinyl_c2h2_loss` | disabled, unguarded | 189 | 179 / 124 | 0.00 | 0.00 | fails |
| `halogenated_quinoline_ring_cleavage` | dropped 09-27 | 883 | 855 / 317 | −0.00 (−0.01 to 0.00) | 0.00 | fails |
| `benzopyranone_ring_cleavage` | dropped 09-27 | 124 | 112 / 55 | −0.03 (−0.07 to 0.00) | +0.01 (0.00 to 0.02) | fails |
| `benzopyranone_bis_ch_loss` | dropped 09-27 | 124 | 113 / 55 | 0.00 | 0.00 | fails |
| `crf12_0` | shared, never MASSter | 176,076 (155,312 with product) | 943 / 390 | +0.01 (−0.00 to 0.02) | −0.01 (−0.04 to 0.00) | fails |
| `crf12_1` | shared, never MASSter | 2,352 (600) | 570 / 264 | −0.01 (−0.05 to 0.01) | 0.00 (−0.00 to 0.00) | fails |
| `crf13_2` | shared, never MASSter | 5,828 (5,446) | 949 / 302 | 0.00 (−0.00 to 0.01) | +0.01 (0.00 to 0.03) | fails |
| `linear_tricyclic_bis_c4_loss` | shared, never MASSter | 34 | 32 / 6 | 0.00 | 0.00 | fails |
| `n_methylanilinoethanol_o_smiles` | candidate, `[M-H]-` | 27 | 26 / 5 | 0.00* | −0.03 (−0.08 to 0.00) | fails |

**No matched molecule (25 rules):**
- **Registered:** `aryl_sulfonate_so2`, `n_nitramine_n2o`,
  `short_haloalkyl_ammonium_c2_transfer`, `n_acyl_aryl_sulfonamide_carbonyl_o_transfer`,
  `diacylhydrazine_carboxylate_o_transfer`, `phenylalanine_carboxyl_o_transfer`,
  `hydroxamic_acid_nh_excision`, `pfas_sulfonate_f_transfer`.
- **Disabled 2026-09-29:** `diaryl_ketone_co_loss`, `aminobenzothiazole_hcn_loss`,
  `pyridyl_carbamate_alkyl_transfer`, `anisoyl_fluoride_co_f_transfer`,
  `short_haloalkyl_ammonium_c3_transfer`, `n_fluoro_sulfonamide_f_transfer`.
- **Dropped 2026-09-27:** both anthraquinone rules, xanthone, phenoxazine, furanocoumarin, the
  morphinan arm and both desoxymorphinan rules, and `aryl_sulfonate_so3`.
- **Other:** `morphinan_ring_cleavage` and `pfas_sulfinate_f_transfer`.

**The zeros reflect the library, not the matcher.** Direct RDKit counts:
- no aryl ketone (`c-C(=O)-c`) among all 183,476 molecules;
- no ketone and no N-OH group in the first 20,000;
- in that same sample, 86% carry an amide, 11% a carboxylic acid and 7% an aryl sulfonamide.

**Reach:** products of the five passing rules exist for 15,652 molecules. That covers 7.8% of
the 590,683 `[M+H]+` spectra and 13.8% of the 185,030 `[M-H]-` spectra.

## Limits

- **Coverage, not model gain.** The gate bounds what a rule could add at the formula level.
  Pool reach (NIST20 [finding §3.1](masster_smarts_cuda_audit_2026_09_27.md#31-coverage-the-templates-add-beyond-the-pool))
  needs a trained Enveda proposal and was not measured.
- **Adducts:** only `[M+H]+` and `[M-H]-` are scored, 67% of Enveda-180 spectra. Dimers
  (28.7%), `[M+Na]+`, formate and the other adducts are not.
- **Peaks:** the proc tables come from the release's filtered export, which drops peaks below 1%
  of the base peak ([dataset finding §2](../enveda_180_dataset_2026_09_25.md#2-the--filtered-export-is-a-1-base-peak-floor)).
- **Sample:** rules with more than 1,000 matched molecules are sampled. At most 13% of a
  rule's scored spectra have a product that hit the twin search's cap
  (`benzopyranone_bis_ch_loss`); a capped product counts as covering nothing.
- **Channels:** as in the NIST20 run, a product's valence bound takes no slots back for its
  newly formed bonds, so its channels are an upper bound.

## Provenance

- **Command:**
  `python scripts/debug/smarts/debug_masster_smarts_rule_gate_enveda.py --workers 10 --max_molecules 1000`.
  It took 265 s on the workstation.
- **Outputs:** in `output/smarts_gate_enveda_2026_09_29/`: `census.pkl`, `rule_gate.pkl`,
  `summary.csv`, `run.log`.
- **Revision:** `git_rev.txt` records the HEAD revision and the dirty diff stat. The working
  tree held the 2026-09-29 rule disabling and another session's uncommitted SMARTS pathway work.
- **Data:** `data/proc/enveda_180/{mol_df,spec_df}.pkl`
  ([dataset finding](../enveda_180_dataset_2026_09_25.md)).
