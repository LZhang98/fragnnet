# Rare chemistry element-swap census: halogens have measured pairs, Se/As/Si do not

> **Status:** FINDING · **Updated:** 2026-10-14
> Measured same-group element-swap pairs on NIST20 (`i1` splits) and MSGYM. Br↔Cl gives enough pairs for a NIST20 correspondence screen; I has a few direct pairs; Se, As, Si and Te have essentially none; MSGYM has no held-out molecule with a training swap partner.

## Decision

Element-swap transfer, step 3 of the [rare-element handling](../../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14),
is a NIST20 halogen study only. Validate on Br↔Cl, then read I descriptively. Se, As, Si and
Te cannot be reached through measured pairs on either dataset; only a representation change
(step 4) or descriptive reporting (step 5) applies to them. MSGYM is out of scope for this step.

## Method

[`element_swap_census.py`](../../../scripts/debug/rare_chemistry/element_swap_census.py) rebuilds
each split with `_setup_dfs` from the control configs (NIST20 `i1_inst_ob_obf.yml`, MSGYM
`…_k1_ema999_fourier.yml`), as the [class census](rare_chemistry_class_census_2026_10_10.md) does.
For each molecule carrying the heavier element, it replaces one atom (each site separately), and
separately all atoms, by the lighter cousin, sanitizes, and looks the product's 2D InChIKey block
up among measured molecules in all splits. Condition compatibility is read two ways: the pair
shares at least one (adduct, instrument, frag mode) combination ("coarse"), or one full
acquisition key including the raw CE fields ("exact"). Outputs:
`output/element_swap_census/{nist20_i1,msgym}.json`, and the NIST20 per-pair table
`nist20_i1_pairs.csv`.

Structure only; no spectrum or model output was read.

## Results

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

## Reading

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
