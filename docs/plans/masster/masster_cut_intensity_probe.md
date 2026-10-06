# Frozen cut-to-intensity probe

> **Status:** CLOSED NULL · **Updated:** 2026-10-01
> Full offline screen complete: tuning
> selects zero correction for both probes. No MASSter retraining or cluster submission.

## Objective and acceptance

Test whether train-only cut predictions improve `i1`'s intensity allocation on molecules the
correction never saw. This is the cheap check proposed in the
[worst-prediction finding](../../findings/masster/masster_i1_worst_predictions_2026_09_30.md#next-steps-training-arms-not-built).
Bond AUC alone does not establish fragment-ranking value, especially for multi-cut fragments.

The locked confirmation screen passes only if the selected probe improves instrument-tolerance
cosine by at least 0.005, its molecule-clustered 95% interval excludes zero, it beats the best
independently tuned control, and there is no significant loss on the independent floor or
unfamiliar molecules. The 0.005 is a practical screen threshold borrowed from the current
`i1` experiment plan, not a conversion between instrument cosine and the binned training metric.
This screen establishes an offline correction effect; it does not establish a training-arm gain.

## Frozen design

1. Reuse `i1` epoch-59 frozen training bond features, training annotated peak alternatives,
   and the existing train/validation membership. Audit molecular identity with `inchikey_s`.
   Fit labels only on training molecules. A base-peak label is the mean boundary-cut indicator
   over **all distinct stored compatible atom sets**, independent of their predicted shares.
   Exclude pure precursor/no-cut base groups; record omissions. These are candidate-derived
   weak labels, not experimentally identified bond breaks. The stored annotated peaks map each
   measured target to one rounded candidate mass; this is a limitation of the inherited labels.
2. Split eligible training identities into 80% probe fitting and 20% calibration, with seed
   397623. Cap fitting at three spectra per molecule, selected before looking at target values.
   Train fixed 100-iteration gradient-boosted regressors on fractional cut labels using either
   encoder PCA endpoint features + bond attributes, or the earlier handmade bond features;
   both receive NCE and instrument. Fit scalar logistic calibration on separate train molecules.
   The existing PCA was fitted on training-only atoms; it explains 87.8% of variance at 128
   components. Calibration molecules may contribute to that unsupervised basis.
3. Split validation identities 50/50 into tuning and locked confirmation. These validation
   spectra have been inspected in earlier research; confirmation is held out from **this
   correction selection**, not historically untouched. Do not read canonical test targets.
4. Run fresh annotated `i1` inference on every validation spectrum using CPU candidates
   (CUDA unavailable locally). For each emitted fragment, sum calibrated per-bond log odds
   over its own original-molecule boundary bonds. Add alpha times that sum to final fragment
   logits, preserving the ion/H conditional distribution and all emitted channels. No
   target-dependent candidate pruning or target-selected structure is used in corrections.
5. Independently tune existing fragment proposal-score mixing, signed cut-count bias,
   fragment-logit temperature and per-bond shuffled probe scores. Choose the probe family and
   alpha by tuning mean cosine only; ties prefer zero/smaller magnitude. Read confirmation
   after locking. Compare the selected probe with unchanged prediction and the strongest
   independently tuned control. Report spectrum-weighted and molecule-weighted means,
   base-peak hits, unfamiliar molecules (nearest-train Tanimoto <=0.4), and the floor selected
   by `fa1` epoch 63 (`cos_hun_inst_tol` <0.05), independent of `i1`.

All hyperparameters and paths are in
[the analysis config](../../../config/analysis/masster_cut_intensity_probe.yaml).
Numerical mass matching follows production: sum rounded 4-decimal peaks, measured-mass-relative
tolerance with a 200 Da floor, full unmatched-peak norms and one-to-one maximum-product matching.
Controls use the same fresh CPU candidate pool, so backend changes do not masquerade as effects.

## Validation and execution state

Initial repository status was clean; project-state entries describing earlier work as
uncommitted are stale. The user reports JEPA running without convincing gains so far; its older
local plan still says unsubmitted. This task does not refresh or change those jobs.

CPU smoke: strict epoch-59 weight load and annotated forward on 16 real validation spectra
passed, 0.251 seconds for the forward. No dataset/checkpoint originals were modified; retired
instrument labels in the archived validation subset are renamed in a diagnostic copy.

Completed: train-only fitting and calibration, all 52,020 fresh validation forwards, locked
tuning/confirmation read and the independent artifact audit. The tested additive cut correction
fails; no cut-supervision training arm is justified by this gate. See the
[canonical results, controls, provenance and limits](../../findings/masster/masster_cut_intensity_probe_2026_10_01.md).

Validation: 13 diagnostic and 10 existing fragment-comparison tests pass; Ruff, local links and
`git diff --check` pass. Artifacts and pytest temporary directories are under
`output/cut_intensity_probe_2026_10_01/`; no full model suite was needed for diagnostic-only work.
Re-run from the repository root with the configured interpreter, in order:

```bash
MPLCONFIGDIR=output/cut_intensity_probe_2026_10_01/mpl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage fit
MPLCONFIGDIR=output/cut_intensity_probe_2026_10_01/mpl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage score --device cpu
MPLCONFIGDIR=output/cut_intensity_probe_2026_10_01/mpl OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python \
  scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage report
```
