# CASMI 2026 train: Br/Cl/I carriers, quality and halogen-swap pairs

> **Status:** FINDING · **Updated:** 2026-10-16
> CASMI train holds 4.3x NIST20 HR's Br molecules and 7.6x its Cl molecules after a halogen-aware quality filter, almost all new to NIST20; Enveda-180 supplies most of them. MSnLib + Enveda-180 gives 354 high-quality Br→Cl pairs against NIST20's 57 training pairs. Structure and spectrum census only; no model was trained.

## Method

[`casmi26_halogen_census.py`](../../scripts/debug/casmi26/casmi26_halogen_census.py) scores every
CASMI train row whose `molecular_formula` carries Br, Cl or I (287,607 rows), at the tight
(ORB) tolerance of the [CASMI finding](casmi2026_dataset_2026_10_05.md#54-subformula-coverage).

- **Isotope-aware coverage.** The project's subformula check is monoisotopic, so ⁸¹Br/³⁷Cl peaks
  of halogenated fragments count as unexplained. Here a peak is also explained when it is a ¹³C,
  ⁸¹Br or ³⁷Cl (one or two heavy atoms) satellite of a matched peak. The ±0.05 Da shifted-peak chance
  control uses the same rule. (Matching against an isotope-expanded candidate set instead lets
  chance explain 0.77 of Enveda intensity, so it was rejected.)
- **Precursor.** Checked against the monoisotopic ion and the M+2 isotopologue; M+2 isolation is
  near zero everywhere except MoNA Br (15.5%).
- **High quality (HQ).** TOF/ORB, monomer, |precursor error| ≤ 10 ppm, ≥ 3 peaks, isotope-aware
  coverage ≥ 0.8, coverage − chance ≥ 0.3, base peak explained. Thresholds were set before reading
  the tables and were not tuned.
- **Pairs.** Single-site and all-site Br→Cl, I→Br, I→Cl swaps, matched on the 2D InChIKey block,
  as in the [element-swap census](masster/rare_chemistry_element_swap_census_2026_10_14.md). "Coarse"
  means the pair shares an (adduct, instrument class).
- **NIST20 HR** here is all splits of `nist20_hr*` in `data/proc/nist20`, not the `i1` train split.

Outputs are in `output/casmi26_halogen_2026_10_16/`: `casmi_halogen_rows.parquet` (one row per
halogen spectrum, with `rg`/`idx` into `train.parquet`; the `hq` flag is computed in the script),
`quality_by_lib.csv`, `carriers_by_lib.csv`, `swap_pairs_summary.csv`, `swap_pairs.csv`.

## Results

Carrier molecules (spectra). NIST20 HR, all splits: Br 1,837 (92,171), Cl 3,662 (134,565), I 387 (11,893).

| Pool | Tier | Br mols (spectra) | new vs NIST20 | Cl mols (spectra) | new vs NIST20 | I mols |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MSnLib (`pluskal_ms2`) | HQ | 1,588 (12,285) | 1,535 | 5,509 (47,752) | 5,157 | 237 |
| Enveda-180 | HQ | 6,026 (22,112) | 6,024 | 20,542 (76,321) | 20,539 | 0 |
| MSnLib + Enveda-180 | HQ | 7,558 (34,397) | 7,503 | 25,794 (124,073) | 25,439 | 237 |
| All CASMI | HQ | 7,994 (37,084) | 7,905 | 27,865 (155,270) | 27,153 | 324 |
| All CASMI | all rows | 8,522 (47,662) | 8,424 | 29,848 (236,661) | 29,074 | 369 |

HQ pass rate of halogen spectra, by library (Br / Cl): MSnLib 0.84 / 0.76, Enveda-180 0.80 / 0.61
(Cl loses mainly dimers; 0.78 of Cl HR monomers pass), MassBank 0.62 / 0.72, MoNA 0.60 / 0.80,
GNPS 0.67 / 0.64, Spectraverse 0.66 / 0.59, RIKEN 0.17 / 0.52 (Br median coverage 0.43), MS-DIAL
0.26 / 0.31. `drug_plus` has no HR monomer rows.

Br→Cl swap pairs (HQ):

| Pool | Pairs | Coarse-compatible | Not both in NIST20 |
| --- | ---: | ---: | ---: |
| MSnLib | 16 | 16 | 12 |
| Enveda-180 | 329 | 319 | 329 |
| MSnLib + Enveda-180 | 354 | 335 | 350 |
| All CASMI | 411 | 388 | 402 |
| NIST20 HR + MSnLib + Enveda-180 | 571 | 548 | 368 |

I→Br / I→Cl, HQ: MSnLib + Enveda-180 2 / 10, all CASMI 13 / 19, with NIST20 66 / 71 (nearly all from NIST20).
Enveda-180 contains no iodine.

## Reading

- **Br and Cl carriers are not scarce in CASMI.** NIST20's limit for the swap study was pairs
  (57 training Br→Cl), not carriers. Enveda-180 alone gives 329 HQ Br→Cl pairs, 319 under a shared
  adduct and instrument. These are combinatorial-library analogues (for example an aryl-Br amide and
  its aryl-Cl twin), which is what a correspondence screen needs.
- **MSnLib adds little to Br.** It brings 1,588 HQ Br molecules and 16 pairs, against Enveda's 6,026 and
  329. It is the only CASMI source of iodine.
- **Instrument and energy differ by pool.** Enveda-180 is timsTOF in eV, MSnLib is Orbitrap in NCE.
  Pairs are compatible within a library, not across one. Enveda-180 is standalone, ACE only, no
  EMA ([Enveda-180 finding](enveda_180_dataset_2026_09_25.md)), so combining it with MSnLib is a new
  dataset decision, not an extension of the Enveda baseline.
- **The HQ flag is a candidate label filter**, not proof of a correct label. RIKEN Br is the only
  large source whose spectra mostly fail it.

## Chemical-space UMAP of all datasets

[`collect_mols.py`](../../scripts/debug/dataset_umap/collect_mols.py) (`FRAGNNET-GPU`) and
[`plot_umap.py`](../../scripts/debug/dataset_umap/plot_umap.py) (separate `umap-viz` conda env:
umap-learn 0.5.7, scikit-learn 1.5; `FRAGNNET-GPU` has no UMAP) fit one joint UMAP (Morgan r=2,
2048 bits, Jaccard) on up to 25,000 molecules per dataset (103,322 unique). The sample is seeded and
the fit is not. Figures `umap_by_dataset.png` and `umap_halogens_by_dataset.png` are in
`output/dataset_umap_2026_10_16/`, with the coordinates.

Molecules, all rows (2D InChIKey block): NIST20 HR 26,800; NIST23 HR 44,831; MassSpecGym 16,957;
Spectraverse 44,234; MSnLib 46,821; Enveda-180 182,941; MSnLib + Enveda-180 227,487 (6,489 in NIST20 ∪ NIST23 HR).

- Enveda-180 and MSnLib fill the lower block of the map. NIST20/23 are densest in the upper right,
  and a large part of the lower block has little NIST coverage. MassSpecGym and Spectraverse span both.
- The NP proxy sits mostly in the upper arms and outlying clusters, where Enveda-180 is nearly absent.
- NIST20/23 Br and I carriers cluster in the upper-right region. Enveda-180 and MSnLib Br carriers
  spread through the lower block, so they add Br chemistry in a different region, not more of the same.

## Not checked

- Spectrum-level Br↔Cl correspondence (step 3a of the
  [rare-element plan](../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14)) on these pairs.
- CE overlap within pairs; molecule overlap with the hidden CASMI test (unknown).
