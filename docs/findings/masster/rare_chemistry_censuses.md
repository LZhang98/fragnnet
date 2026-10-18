# Rare chemistry censuses: class rarity and element swaps

> **Status:** FINDING · **Updated:** 2026-10-18
> Two training-only censuses behind the rare chemistry augmentation plan: class rarity for the A1/A3 sampling arms, and measured same-group element-swap pairs.

Merged 2026-10-18 from 2 findings on one topic. Each part below is the original
document unchanged except for heading levels; its own status line keeps its original date.
Former file names are listed so old links and `git log -- <former path>` can be traced.

| Part | Updated | Verdict | Former file |
| --- | --- | --- | --- |
| [Rare chemistry census: A1 moves exposure, A3 does not](#rare-chemistry-census-a1-moves-exposure-a3-does-not) | 2026-10-14 | A1 moves exposure (most on MSGYM); A3 fails the support gate; arms removed unrun | `rare_chemistry_class_census_2026_10_10.md` |
| [Rare chemistry element-swap census: halogens have measured pairs, Se/As/Si do not](#rare-chemistry-element-swap-census-halogens-have-measured-pairs-seassi-do-not) | 2026-10-14 | Br↔Cl has enough NIST20 pairs; Se/As/Si/Te none; MSGYM none usable | `rare_chemistry_element_swap_census_2026_10_14.md` |

## Rare chemistry census: A1 moves exposure, A3 does not

> **Status:** FINDING · **Updated:** 2026-10-14
> The A1/A3 sampler and this census script were removed at the user's request on 2026-10-14 and never ran; both are in Git at `ae0470d4`. Training-only census for the A1/A3 sampling arms on NIST20 and MSGYM. A1 changes molecule exposure substantially, most on MSGYM. A3 with the functional-group panel barely differs from A1 and fails the plan's evaluation-support gate on both datasets.

### Decision

Run A0 and A1 on both datasets. Do not submit A3 as configured. Its molecule distribution is
almost identical to A1's, so a one-seed A3-versus-A1 screen would read seed noise. Its held-out
rare cohorts are also too small to resolve the plan's +0.01 rare-cohort gate.

Redesigning A3 is a user decision; see [options](#options-for-a3). This census reads only
training counts and held-out cohort sizes, never model outcomes, so a redesign stays
preregistered.

### Method

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

### Results

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

### Panel correction before freezing

The first census matched no sulfoxides on either dataset. Dataset SMILES write many sulfoxides
in charge-separated form, `[S+][O-]`, which the double-bond SMARTS missed. The v1 panel now
accepts both forms for sulfoxide, phosphoryl and thiophosphoryl. Sulfoxide then matches 31
NIST20 and 45 MSGYM training molecules. No run or cohort used the earlier panel; the reported
census is the corrected one.

### Options for A3

1. **Drop A3** and read only A1 versus A0. Simplest; the plan allows closing rarity weighting.
2. **ClassyFire classes on NIST20 only.** `data/classyfire/nist20_mol_classyfire.pkl` covers
   every NIST20 molecule with 341 classes, so many more classes fall under 20 molecules. It covers
   only 57% of MSGYM training molecules, so it cannot be applied identically to MSGYM.
3. **Stronger weights on the same panel**, for example a larger reference count. This raises the
   exposure change but not held-out support; the rare cohorts stay under 30 molecules.

Evaluation support is the binding constraint under every option for MSGYM.

## Rare chemistry element-swap census: halogens have measured pairs, Se/As/Si do not

> **Status:** FINDING · **Updated:** 2026-10-14
> Measured same-group element-swap pairs on NIST20 (`i1` splits) and MSGYM. Br↔Cl gives enough pairs for a NIST20 correspondence screen; I has a few direct pairs; Se, As, Si and Te have essentially none; MSGYM has no held-out molecule with a training swap partner.

### Decision

Element-swap transfer, step 3 of the [rare-element handling](../../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14),
is a NIST20 halogen study only. Validate on Br↔Cl, then read I descriptively. Se, As, Si and
Te cannot be reached through measured pairs on either dataset; only a representation change
(step 4) or descriptive reporting (step 5) applies to them. MSGYM is out of scope for this step.

### Method

[`element_swap_census.py`](../../../scripts/debug/rare_chemistry/element_swap_census.py) rebuilds
each split with `_setup_dfs` from the control configs (NIST20 `i1_inst_ob_obf.yml`, MSGYM
`…_k1_ema999_fourier.yml`), as the [class census](#rare-chemistry-census-a1-moves-exposure-a3-does-not) does.
For each molecule carrying the heavier element, it replaces one atom (each site separately), and
separately all atoms, by the lighter cousin, sanitizes, and looks the product's 2D InChIKey block
up among measured molecules in all splits. Condition compatibility is read two ways: the pair
shares at least one (adduct, instrument, frag mode) combination ("coarse"), or one full
acquisition key including the raw CE fields ("exact"). Outputs:
`output/element_swap_census/{nist20_i1,msgym}.json`, and the NIST20 per-pair table
`nist20_i1_pairs.csv`.

Structure only; no spectrum or model output was read.

### Results

NIST20, `i1` splits. "Carriers" are molecules containing the source element.

| Swap | Train carriers | Val / test carriers | Train–train pairs | Coarse-compatible | Exact acquisition key | Held-out source with a train partner |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Br→Cl | 997 | 333 / 305 | 57 | 35 | 8 | 40 |
| I→Br | 203 | 65 / 62 | 12 | 10 | 6 | 14 |
| I→Cl | 203 | 65 / 62 | 11 | 9 | 1 | 16 |
| Se→S | 3 | 1 / 2 | 0 | 0 | 0 | 1 |
| As→P | 0 | 0 / 0 | – | – | – | – |
| Si→C | 9 | 5 / 4 | 0 | – | – | 0 |

MSGYM: Br→Cl 13 train–train pairs, I→Cl 6, Se→S 1, I→Br, As and Si none. No held-out MSGYM
molecule has a training swap partner, consistent with its structure-novelty split.

Of the 263 NIST20 pairs, 249 are single-site swaps and 14 are all-site swaps of polyhalides.
Examples:
`CN(C)CCC(c1ccc(Br)cc1)c1ccccn1` / its Cl analogue (brompheniramine/chlorphenamine), or
`COc1cc(CCN)c(OC)cc1I` / `…Br` (2C-I/2C-B).

### Reading

- **Exact CE matches are rare.** Only 8 of 57 Br→Cl training pairs share a full acquisition
  key, because NIST CE schedules differ between entries. A spectrum comparison must match CE
  within a tolerance or compare at the nearest recorded energy, not require identical schedules.
- **The validation family is small.** 35 coarse-compatible Br→Cl training pairs must cover
  fitting, calibration and confirmation. That supports a correspondence screen, not a fitted
  context table. Pooling the halogen family (Br→Cl, I→Br, I→Cl) gives 80 training pairs, 54 coarse-compatible.
- **Val pairs give a fair model comparison.** 21 val Br molecules have a training Cl partner
  (15 of those pairs coarse-compatible), and 14 val I molecules have a training Br or Cl
  partner. `i1` never trained on them, so its prediction can be compared with a transported
  training analogue. Test spectra stay unread.
- **Iodine is not rare overall.** 203 training molecules carry I; only alkyl iodide is rare in
  the class census. Whether held-out iodine molecules score worse is unmeasured (step 1).
