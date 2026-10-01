# Three EI SMARTS families: no rule admitted

> **Status:** COMPLETE · **Updated:** 2026-10-01
> Anisoyl fluoride and para-hydroxyazobenzene add no nominal coverage. Sulfonamide
> oxygen transfer has training-only support, insufficient for admission.

## Decision and measurements

Keep the three candidate definitions research-only. Do not enable templates or change EI
metadata on the strength of this screen. The
[probe](../../../scripts/debug/debug_masster_ei_smarts_screen.py) and its
[tests](../../../tests/test_debug_masster_ei_smarts_screen.py) are added; production code,
configs, datasets and existing rule indices are unchanged.

Counts are molecules; each matched molecule has one eligible spectrum. Coverage is the
mean fraction of observed L1 intensity added beyond the connected-cut baseline **including
natural isotopes**, expressed as percentage points. The null uses disconnected random
assemblies, matched to each rule product's heavy-atom count.

| Family | Train / val / test structural matches | Train added intensity | Val added intensity | Train rule minus null, 95% molecule-bootstrap CI |
| --- | ---: | ---: | ---: | ---: |
| N-H aromatic-acyl arylsulfonamide carbonyl-O transfer | 7 / 0 / 0 | 2.414% | No examples | +1.034 points [−0.010, +3.063] |
| Anisoyl fluoride CO loss / F transfer | 0 / 1 / 0 | No examples | 0.000% | Not estimable |
| Para-hydroxyazobenzene N2 loss | 3 / 3 / 0 | 0.000% | 0.000% | 0.000 points [0.000, 0.000] |

Every matched molecule produces one distinct disconnected product. All 32 null draws per
product succeeded. For anisoyl fluoride and azo products, **every valid nominal product
mass, including its retained isotope support, is already in the ordinary pool**; this is
stronger than merely observing no matching new peak.

The sulfonamide average is driven by training molecule `192781` (spectrum `105148`), which
adds 16.331% at m/z 126–129, versus a 9.254% null. Only that molecule exceeds 1% added
intensity. Molecules `21609` and `222767` add 0.167% and 0.402%; the other four add zero.
Without isotopes, mean training coverage gain would be 6.660%, overstating the new mass
support relative to the current natural-isotope EI configuration. No validation example
exists for replication, and the paired bootstrap interval includes zero.

## Why plausible chemistry can add no mass support

Direct atom-set witnesses are saved in `collision_witnesses.json` below:

- Anisoyl fluoride's proposed fluoroanisole composition C7H7FO has nominal m/z 126.
  An ordinary connected fragment with channel composition C6H3FO2 also has m/z 126.
- Para-hydroxyazobenzene's N2-loss composition C12H10O has nominal m/z 170.
  Ordinary channels C11H10N2 and C10H6N2O also have nominal m/z 170.
- For unsubstituted N-benzoylbenzenesulfonamide, the phenol radical cation at m/z 94
  overlaps the M+1 isotope support of an ordinary C7H9 channel at m/z 93.

These are explanations of **mass support**, not assignments of the real peak's chemistry.
In particular, a weak isotope row cannot necessarily reproduce a strong observed peak
without overpredicting its monoisotopic partner. This binary coverage screen does not
measure that abundance constraint, intensity ranking, or a trained-model benefit.

## Chemistry and scope

1. **Carbonyl-O transfer.** The [70-eV EI and oxygen-labeling study](https://doi.org/10.1002/rcm.6850)
   supports transfer of the *amide carbonyl* O to the sulfonyl-bearing aryl residue.
   The N-H branch yields a phenol radical cation; N-alkyl compounds favor nitrilium
   instead. The probe requires N-H and an aromatic group on the acyl carbon. Its
   historical local identifier contains `benzoyl`, but the actual query also matches
   aromatic-acyl extensions: one of the seven examples is a naphthoyl derivative
   (`222766`, zero added isotope-aware coverage). Such extensions remain exploratory.
2. **Anisoyl fluoride.** Reuse the exact disabled query for ortho/meta/para same-ring
   methoxy acyl fluorides, removing the mapped CO atoms while retaining F. Source:
   [Tajima et al., 1991](https://doi.org/10.1002/oms.1210261122); the full primary paper
   was not accessible in this session. The existing code definition supplies the map.
3. **Azo extrusion.** [Moanta 2012, Scheme 5](https://bch.ro/pdfRC/MOANTA%20A%201%2012.pdf)
   supplies an EI N2-loss example among substituted para-hydroxyazobenzenes. Remove
   the two azo N atoms and retain both rings. Require two nonfused, all-carbon phenyl
   rings and a para phenolic OH. Substituent-general matching is a screening hypothesis,
   not evidence that every derivative follows the reaction. The paper also reports
   absence of the biphenylene ion in this series; no additional cyclization is encoded.

The literal SMARTS and retained atom tuples are recorded in the probe and artifacts.
No ESI-only ion guards are broadened. The probe establishes EI eligibility from the explicit
`nist23_ei` source and verifies `[M]+`; it does not infer EI from missing `frag_mode` generally.
Production would still need a reliable EI metadata path if an EI-only rule were admitted.

## Method, provenance and limits

- Inputs: `data/proc/nist23_ei/{mol_df,spec_df}.pkl` and the train/val/test CSVs in
  `data/split/nist23_ei_inchikey`. Census covers all **250,644 molecules** in these three
  splits: 150,386 train, 50,130 val, 50,128 test; 258,366 spectra. Entries outside these
  splits, including the secondary set, are excluded. Test structures were counted;
  their peaks were not scored.
- Baseline: every connected component from zero, one or two precursor-bond cuts.
  Production `[M]+` CRF channels, ±4 H, valence bound, and formed-bond correction on
  real and random disconnected assemblies. This is a stand-in pool, not checkpoint
  inference and not an exhaustive connected-subgraph search.
- Natural isotope distributions use the production composition utility, order 4,
  cropped-distribution normalization, and the 0.001 pathway-share cutoff. Targets use
  positive intensity, m/z below 1500, and L1 normalization; nominal integer equality.
- Per molecule and rule: 32 seeded size-matched disconnected controls, excluding the
  real products. Means and paired 95% intervals use molecules; 2,000 bootstrap draws.
  These intervals do not add a separate Monte Carlo error term for the 32 null draws.
- Revision `e4c9496948b08085b124f91a1ceea0a6d7ce3429` plus recorded working-tree diff.
  Other concurrent edits are preserved. CPU, six workers, seed 20260930; configured
  `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`.
- Command: `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python
  scripts/debug/debug_masster_ei_smarts_screen.py --workers 6`.
- Final artifacts: [`output/masster_ei_smarts_2026_10_01/`](../../../output/masster_ei_smarts_2026_10_01/):
  `census.csv`, `rows.csv`, `summary.csv`, `provenance.json`, `starting.diff`, and the
  supplementary `collision_witnesses.json`. Provenance records input/source SHA256,
  versions and UTC times. The final run exited 0.
- The earlier `output/masster_ei_smarts_2026_09_30/` is a **superseded pilot**: its azo
  query admitted one heteroaryl and one fused-aryl example. Both had zero gain; they
  are excluded from the final results.

Validation: **63 distinct focused tests passed** (22 new probe tests, 13 reused G0/channel
checks, 28 nominal-isotope tests). The probe tests check known product formulas, conservation,
carbonyl-O identity, close negative controls, all anisoyl positions, random controls,
molecule weighting, nominal production parity and isotope overlap. Ruff and documentation
link/diff checks passed. No training, test-spectrum evaluation or full-suite run.

Next: retain these as non-admitted candidates. A future intensity/pathway experiment would
need its own gate; these coverage results cannot justify it. Other shortlist families
(TMS/ketene and steroid TMS migration) have not been screened here.
