# Spectraverse v1.0.3 split leak audit

> **Status:** FINDING · **Updated:** 2026-10-18
> No molecule is shared between train, val and test in any of the ten v1.0.3 folds under five identity keys. 2.3% of spectra belong to molecules (86% lipids or long chains) whose ECFP6 equals that of a molecule in another fold; for cv0/cv1 that is 3.9%/3.5% of test spectra.

## Question

Do the v1.0.3 benchmark splits (`data/split/spectraverse_1_0_3_benchmark/cv0-cv9`, built from the
MGF `FOLD=` field, [preproc README](../../preproc_scripts/README.md#spectraverse-v103)) leak a
molecule between train, val and test?

## Result

**No identity leak.** For every cv and every split pair (train-val, train-test, val-test), the
number of shared keys is 0 for all five keys:

| Key | What it merges |
| --- | --- |
| `mol_id` | one canonical SMILES in `mol_df.pkl` |
| `inchikey_s` | RDKit 2D InChIKey (first block): stereoisomers, isotopologues |
| `mgf_ik14` | first block of the MGF's own `INCHIKEY` |
| `parent_ik14` | 2D InChIKey after cleanup, largest fragment, uncharge, canonical tautomer: salts, charge forms, tautomers |
| `parent_smiles` | stereo-free canonical SMILES of the same parent |

All 44,300 molecules standardized (0 failures). All are single-component; 455 carry a +1 charge.
The split files cover every `spec_id` exactly once in each cv.

**Close analogs do cross folds.** 3,716 molecule pairs in different folds have ECFP6 (Morgan
radius 3, 2048 bits) Tanimoto 1.0. They involve 1,685 molecules and 10,909 spectra (2.3%):

| Class (first SMARTS match) | Molecules |
| --- | ---: |
| triacylglycerol | 526 |
| glycerophospholipid / diacylglycerol | 502 |
| long alkyl chain (C10+) | 228 |
| monoacyl / ether glycerolipid | 122 |
| perfluoro chain | 71 |
| other | 236 |

Most pairs differ only in chain length or double-bond position: 927 are pure CH2 homologs and
2,180 differ by another CxHy; 79 share a formula. Five sampled pairs of the 530 others were all
oligomer series (polypropylene glycols, repeated acetyl/benzoyl ethyleneamine chains, perfluoro
acids of different length). Fold 3 is in most pairs (fold 3-7 alone has 848). The spectra come mainly from
RIKEN (6,513) and GNPS (3,362).

Share of val/test spectra whose molecule has an ECFP6 Tanimoto-1.0 partner in that cv's train
split:

| cv | val | test |
| --- | ---: | ---: |
| 0 | 2.2% | 3.9% |
| 1 | 2.9% | 3.5% |
| 2 | 3.2% | 7.9% |
| 3 | 8.8% | 0.3% |
| 4 | 0.3% | 1.0% |
| 5 | 0.9% | 1.7% |
| 6 | 1.7% | 0.7% |
| 7 | 0.7% | 0.6% |
| 8 | 0.5% | 0.8% |
| 9 | 0.8% | 1.3% |

Median max ECFP6 Tanimoto of a val/test molecule to train is 0.44-0.76, highest for cv1/cv2
(folds 2 and 3).

With radius-2 Morgan bits (first run) the screen is coarser: 24,994 pairs, 5.6% of spectra,
9.7%/14.9% of cv0/cv1 test; outputs in `output/spv_1_0_3/leak_check/`.

## Reading

The release folds are clean at the molecule level, so the splits need no change for that. The
ECFP6-identical analogs are different molecules, but lipid homologs share head-group fragments and
neutral losses, so they make the cvs that test or validate on folds 2-3 (cv1-cv3) somewhat easier
than the others. That is a property of the release's MCES folds, not of our preprocessing. It is
not yet measured how much it raises scores.

## Method and limits

- Script: `scripts/debug/spectraverse/spectraverse_leak_check.py` (defaults are the v1.0.3
  paths); outputs in `output/spv_1_0_3/leak_check_ecfp6/` (`report.json`, `leaks.json`,
  `max_tanimoto_to_train.csv` with the per-cv spectrum counts, `cross_fold_tanimoto_1.csv`).
- Pair typing and classes were computed ad hoc from `cross_fold_tanimoto_1.csv`; results are in
  `cross_fold_tanimoto_1_typed.csv` and `tanimoto1_molecules_by_class.csv` in the same directory.
- ECFP6 Tanimoto 1.0 is a similarity screen, not an identity test, and it is not the MCES
  distance the folds were built on.
- Not checked: spectral near-duplicates across folds from different molecules.
