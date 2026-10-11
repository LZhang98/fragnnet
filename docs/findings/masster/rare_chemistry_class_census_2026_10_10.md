# Rare chemistry census: A1 moves exposure, A3 does not

> **Status:** FINDING · **Updated:** 2026-10-10
> Training-only census for the A1/A3 sampling arms on NIST20 and MSGYM. A1 changes molecule exposure substantially, most on MSGYM. A3 with the functional-group panel barely differs from A1 and fails the plan's evaluation-support gate on both datasets.

## Decision

Run A0 and A1 on both datasets. Do not submit A3 as configured. Its molecule distribution is
almost identical to A1's, so a one-seed A3-versus-A1 screen would read seed noise. Its held-out
rare cohorts are also too small to resolve the plan's +0.01 rare-cohort gate.

Redesigning A3 is a user decision; see [options](#options-for-a3). This census reads only
training counts and held-out cohort sizes, never model outcomes, so a redesign stays
preregistered.

## Method

[`rare_chemistry_census.py`](../../../scripts/debug/rare_chemistry/rare_chemistry_census.py)
rebuilds each split with the dataset's own `_setup_dfs` from the control configs:
NIST20 `i1_inst_ob_obf.yml` and MSGYM `…_k1_ema999_fourier.yml`. Molecule identity is the
2D InChIKey block. The acquisition key is adduct, instrument, fragmentation mode and the raw
recorded CE fields, with missing values explicit. Classes come from the frozen
`functional_groups_v1` SMARTS panel in
[`rarity_sampling.py`](../../../src/fragnnet/dataset/rarity_sampling.py). Rarity bins and the
A3 weight formula are the plan's, frozen before any outcome. Outputs are in
`output/rare_chemistry_census/{nist20_i1,msgym}/`: `summary.json`, `class_census.csv`, and the
frozen per-molecule cohort table `held_out_cohorts.csv`.

The model dataset can drop a few more spectra for missing DAGs. The sampler logs its own counts
at training start.

## Results

| Training split | NIST20 | MSGYM |
| --- | ---: | ---: |
| Spectra | 157,678 | 99,306 |
| Molecules | 12,668 | 12,304 |
| Molecule-acquisition keys | 155,947 | 58,722 |
| Spectra per molecule, median / 90th pct / max | 13 / 16 / 86 | 3 / 19 / 280 |
| Held-out molecules sharing a 2D InChIKey with train | 0 | 0 |

NIST20 has nearly one acquisition key per spectrum, so its A1 is uniform molecule, then uniform
spectrum. MSGYM repeats acquisition keys, so its A1 also balances conditions within a molecule.

Exposure difference, as total variation distance between molecule distributions:

| Comparison | NIST20 | MSGYM |
| --- | ---: | ---: |
| A1 vs A0 | 0.109 | 0.479 |
| A3 vs A1 | 0.0015 | 0.0023 |
| Training molecules with A3 weight above 1 | 105 | 71 |
| Training molecules matching no class | 22 | 38 |

Held-out validation molecules by rarest matched class. The plan requires at least 20 per
reported cohort.

| Cohort (training molecules in rarest class) | NIST20 val | NIST20 test | MSGYM val | MSGYM test |
| --- | ---: | ---: | ---: | ---: |
| Very rare, 1-5 | 2 | 3 | 7 | 1 |
| Rare, 6-20 | 26 | 25 | 11 | 7 |
| Intermediate, 21-100 | 245 | 242 | 137 | 154 |
| Common, over 100 | 3,941 | 3,944 | 2,176 | 2,154 |
| No class matched | 8 | 9 | 0 | 2 |

The rarest classes are unusual elements and reactive groups: alkyl iodide, azide, selenium,
peroxide, silicon, nitroso and anhydride on NIST20; trivalent phosphorus, arsenic, silicon,
selenium and boron on MSGYM. Sulfoxide, acyl halide and "other element" initially matched zero
molecules; sulfoxide was a panel error, fixed before this census (below).

## Panel correction before freezing

The first census matched no sulfoxides on either dataset. Dataset SMILES write many sulfoxides
in charge-separated form, `[S+][O-]`, which the double-bond SMARTS missed. The v1 panel now
accepts both forms for sulfoxide, phosphoryl and thiophosphoryl. Sulfoxide then matches 31
NIST20 and 45 MSGYM training molecules. No run or cohort used the earlier panel; the reported
census is the corrected one.

## Options for A3

1. **Drop A3** and read only A1 versus A0. Simplest; the plan allows closing rarity weighting.
2. **ClassyFire classes on NIST20 only.** `data/classyfire/nist20_mol_classyfire.pkl` covers
   every NIST20 molecule with 341 classes, so many more classes fall under 20 molecules. It covers
   only 57% of MSGYM training molecules, so it cannot be applied identically to MSGYM.
3. **Stronger weights on the same panel**, for example a larger reference count. This raises the
   exposure change but not held-out support; the rare cohorts stay under 30 molecules.

Evaluation support is the binding constraint under every option for MSGYM.
