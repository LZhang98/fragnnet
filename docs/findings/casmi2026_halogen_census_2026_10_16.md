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

The pipeline's step order, environments and shared helpers are in
[`common.py`](../../scripts/debug/dataset_umap/common.py) (consolidated 2026-10-17).

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

## Without NIST, and where HMDB sits (2026-10-17)

[`open_vs_nist.py`](../../scripts/debug/dataset_umap/open_vs_nist.py): "open" = all CASMI train
libraries except `enveda-np-examples`, plus MassSpecGym and Spectraverse; 276,303 molecules (Br 8,541,
Cl 29,899, I 370) against NIST20 ∪ NIST23 HR 46,717 (Br 3,905, Cl 6,854, I 915). 34,220 NIST
molecules are absent from open data. Nearest open neighbour of each NIST molecule (Morgan r=2,
Tanimoto): identical 27%, ≥0.7 11%, 0.4–0.7 51%, <0.4 10%. For NIST Br molecules only 8% are
identical or ≥0.7. All 250 NP-proxy molecules are already in open data, so the proxy cannot measure
what NIST adds for the hidden test.

HMDB: the PubChem copy of the HMDB deposit (216,227 CIDs, 196,995 2D keys; hmdb.ca blocks scripted
downloads; no detected/expected status), in `data/raw/hmdb/`. Exact overlap: 87% of NP-proxy
molecules, 29% MassSpecGym, 26% NIST20 HR, 23% Spectraverse, 17% NIST23 HR, 10% MSnLib, and 6
Enveda-180 molecules. 7.0% of HMDB molecules have a spectrum in these datasets, 8,574 through NIST.
[`nn_fit.py`](../../scripts/debug/dataset_umap/nn_fit.py), 10,000 seeded HMDB molecules, fraction
with a neighbour ≥ 0.7:

| Reference | All | Long chain (72%) | Other (28%) |
| --- | ---: | ---: | ---: |
| NIST20 ∪ NIST23 HR | 0.81 | 0.90 | 0.57 |
| Spectraverse | 0.75 | 0.82 | 0.58 |
| MassSpecGym | 0.23 | 0.20 | 0.30 |
| MSnLib | 0.11 | 0.07 | 0.19 |
| Enveda-180 | 0.001 | 0.000 | 0.001 |
| All of the above | 0.86 | 0.93 | 0.67 |

"Long chain" = eight consecutive acyclic CH2, mostly enumerated lipids. The rule misses
polyunsaturated chains, so "other" still holds lipids. Lipid series share most fingerprint bits,
so ≥ 0.7 overstates their coverage. Enveda-180's median nearest neighbour to HMDB is 0.23
(spot-checked against an independent 40k-molecule recomputation; NIST20 to Enveda-180 is 0.31): it is
a separate chemical space from human metabolites. Outputs are in `output/dataset_umap_hmdb_2026_10_17/`.

**HMDB on the map.** A joint UMAP fit with HMDB (`output/dataset_umap_hmdb_2026_10_17/`) failed
spectral initialisation, because HMDB lipid series form disconnected components. It took 64 min and
scattered HMDB as islands, so it is not usable. Instead
[`project.py`](../../scripts/debug/dataset_umap/project.py) projects 24,990 HMDB molecules
(including the 10,000 similarity sample) onto the MS/MS-only map with `reducer.transform`
(`output/dataset_umap_proj_2026_10_17/umap_hmdb_projected.png`). 76% land on a few small lipid
islands, mostly Spectraverse with a few NIST molecules. One island of 88 training molecules
(cardiolipin and phospholipid-like) receives 10,761 HMDB molecules. 89% of off-cloud HMDB is
long-chain; of the 24% inside the main cloud, 18% is. HMDB's lipid bulk therefore sits on a
handful of library lipids, and Enveda-180 touches none of it.

**NP Atlas** (microbial natural products, `data/raw/npatlas/NPAtlas_download.tsv` from npatlas.org,
36,454 compounds, 33,497 2D keys; 63% fungal, 37% bacterial). It went through the same pipeline
(`collect_mols.py --extra`, `nn_fit.py --dataset "NP Atlas"`, `project.py --dataset "NP Atlas"`;
outputs `output/dataset_umap_npatlas_2026_10_17/`,
`output/dataset_umap_proj_2026_10_17/umap_npatlas_projected.png`). 1,922 molecules (5.7%) have a
spectrum in any dataset here; Spectraverse holds 1,471 of them. In a 10,000-molecule sample,
nearest MS/MS neighbour ≥ 0.7: all datasets 0.19, Spectraverse 0.16, NIST20 ∪ 23 0.12, MSnLib 0.08,
Enveda-180 0.000; < 0.4 for 32%. Median nearest neighbour 0.48. NP Atlas projects onto the
periphery of the main cloud, beside NIST and Spectraverse. Its largest landing spot (9,122
molecules) sits in a region with 3,522 training molecules, mostly Spectraverse. Unlike HMDB,
microbial natural products are largely uncovered by every library, including NIST.

## Candidate open training set: Spectraverse + MSnLib + Enveda-180 (2026-10-17)

[`combo_coverage.py`](../../scripts/debug/dataset_umap/combo_coverage.py); output
`output/dataset_umap_hmdb_2026_10_17/combo_coverage.json`. Spectraverse already holds 15,569 MSnLib
molecules (its `MSnLib` source). Only 1,330 Spectraverse molecules are MS-DIAL-only, so dropping
MS-DIAL changes every coverage number below by under 0.01.

| Set | Molecules | Br | HMDB ≥0.7: all / non-lipid | NIST sample ≥0.7 | NIST Br <0.4 |
| --- | ---: | ---: | --- | ---: | ---: |
| Spectraverse + MSnLib | 75,486 | 2,046 | 0.758 / 0.598 | 0.344 | 0.304 |
| + Enveda-180 | 256,141 | 8,281 | 0.758 / 0.598 | 0.350 | 0.269 |
| NIST20 ∪ NIST23 HR | 46,717 | 3,905 | 0.807 / 0.573 | — | — |
| All + NIST | — | — | 0.858 / 0.672 | — | — |

NIST sample: 5,000 seeded NIST20 ∪ NIST23 HR molecules (428 Br). Enveda-180 is 71% of the
combination's molecules but adds no HMDB coverage. It adds +0.006 NIST coverage. It lowers the
share of NIST Br molecules without a 0.4 analogue from 0.304 to 0.269. Its value is halogen
counts and Br↔Cl pairs, not HMDB or NIST chemistry. Spectraverse supplies nearly all the
HMDB-like coverage; for non-lipid HMDB it matches NIST (0.598 vs 0.573).

## Not checked

- Spectrum-level Br↔Cl correspondence (step 3a of the
  [rare-element plan](../plans/masster/rare_chemistry_augmentation_plan.md#rare-element-handling-revised-2026-10-14)) on these pairs.
- CE overlap within pairs; molecule overlap with the hidden CASMI test (unknown).
