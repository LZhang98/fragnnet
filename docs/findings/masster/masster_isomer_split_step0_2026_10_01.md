# MASSter within-peak isomer split: checkpoint and cut-class reads

> **Status:** FINDING · **Updated:** 2026-10-01
> Step 0 inference and CPU analysis complete. No training arm built.

## Question and gate

The [identity-supervision finding](masster_identity_supervision_2026_09_22.md) withdrew its
structural-teacher recommendation because moving intensity between same-mass isomers leaves the
predicted spectrum unchanged. The remaining hypothesis is that the underdetermined split harms
transfer to new molecules. The proposed cheap gate was: compare `i1` with an independent
checkpoint on the same train peaks, then ask whether cut-class preferences from unambiguous
train peaks predict the isomer that `i1` favors.

## Inputs and method

- `i1` epoch 59 (`i50ig7rf`) and `fa1` epoch 63 (`0vn8m2gr`), each with its own run config,
  were evaluated on the same seed-0 sample of 1,000 train molecules and 12,366 spectra. The
  extractor used `--all_target_peaks`; its export audit found all 353,185 `i1` and 353,646
  `fa1` target-matched candidates in the peak tables. Both spectrum ID sets are identical.
- Each output channel was identified by fragment atom set, H shift, and ion mode. Duplicate
  channels were summed. Comparison requires the same spectrum and m/z rounded to 6 decimals,
  at least two shared atom sets, each peak total at least 0.001, and at least 90% of each
  checkpoint's peak mass on shared structures. A second subset requires peak totals within
  10% relative difference. Jensen-Shannon divergence (bits), half-L1 distance, and top-atom-set
  agreement compare the normalized shares of the common structures.
- Cut-class features came from `MoleculeChemistry.annotate`. For a bounded chemistry read,
  three evenly spaced spectra per molecule were used (2,992 spectra). A ridge model fit
  log target intensity of single-structure peaks after within-spectrum centering, using either
  four structural controls (cut count, atom fraction, absolute H shift, precursor flag) or
  those controls plus cut-class counts. Models were fit on one parity of molecule IDs and
  checked on the other, then reversed. On ambiguous peaks, the held-out model's highest-scoring
  structure was compared with `i1`'s highest-share structure. The molecule-clustered interval
  resamples molecules 1,000 times. This is a prior-consistency check, not ground-truth
  identification of an isomer within an ambiguous peak.

## Results

| checkpoint comparison | peaks | median half-L1 | median JS bits | top atom set agrees |
| --- | ---: | ---: | ---: | ---: |
| All shared multi-structure peaks | 107,649 | 0.098 | 0.0223 | 69.3% |
| Totals within 10% | 15,629 | 0.0655 | 0.0133 | 72.4% |
| Totals within 10%, both top shares at least 0.5 | 10,934 | 0.0237 | 0.00462 | 85.9% |

The close-total subset is 14.5% of the shared multi-structure peaks. It has median total
difference 5.0%; its 90th-percentile JS divergence is 0.237 bits. Thus a minority of splits differ substantially, but the typical confident split is
similar across the two checkpoints. Because `fa1` changes the allocator architecture while
sharing `i1`'s nominal seed, disagreement cannot be assigned specifically to initialization or
noise.

The cut-class fit used 12,401 single-structure training peaks and checked 46,194 ambiguous
peaks from 996 molecules. A random top choice has 21.8% expected agreement with `i1`.
Structural controls reached 50.0%; adding cut classes reached 44.3%, a **-5.7 percentage-point**
change (molecule bootstrap 95% interval -6.46 to -4.98 points). On held-out single-structure peaks,
the within-spectrum R² was 0.216/0.220 for controls and 0.213/0.221 with cut classes across
the two folds. Cut classes add no dependable target-intensity signal to these controls. Their
failure to predict `i1`'s preferred isomer therefore does not establish that implicit EM chose
chemically wrong structures.

## Decision and limits

**Do not build the bond-split training arm from this gate.** The bulk of confident splits is
stable, and this cut-class prior has no validated incremental signal on the peaks where the
structure is known. The minority of changed splits could still matter, but these two different
architectures cannot identify their cause or a benefit from constraining them. The proposed
loss remains a hypothesis, not a measured correction. The analysis fix to `map_targets` now
assigns target intensity to every candidate in its tolerance window, as the loss does; this
changes future extractor tables relative to the older nearest-only base-peak artifacts.

## Reproduction and validation

The extractor is [`debug_masster_base_peak_errors.py`](../../../scripts/debug/debug_masster_base_peak_errors.py)
and the CPU read is [`analyze_masster_isomer_split.py`](../../../scripts/debug/analyze_masster_isomer_split.py).
The ignored local artifacts, including checkpoint/source hashes and full arguments in each
`.provenance.json`, are under `output/isomer_split_step0_2026_09_30/`; the summary is `read.json`
and the matched-peak table is `read.matched.pkl`. The checkpoint exports took about two minutes
each on the RTX 2070 SUPER. Focused tests: 30 passed; Ruff and `git diff --check` passed. No
training or val-set evaluation was performed.
