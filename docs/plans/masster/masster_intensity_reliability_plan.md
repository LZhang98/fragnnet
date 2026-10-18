# MASSter intensity reliability and transfer

> **Status:** DEFERRED · **Updated:** 2026-10-06
> Only the P1 census ran (G1 stops the observation-noise head pending replicate provenance); P2-P5 not run, no runs planned. Detail below (as of 2026-09-27).
> Selected `k1_ema` epoch 50 downloaded and checksum-verified. Detailed execution stages below
> now correct existing noise estimates with ladder-aware replicate matching, prioritize transfer, and
> use independent-seed ensembles for uncertainty. Replicate-trained head deferred; no new runs.

## Objective and boundary

Determine which intensity errors reflect variable measurements, missing conditions, or stable
chemistry that MASSter has not learned to transfer. Then choose a matching intervention.
Existing evidence already favors model/generalization error over measurement variability.
The immediate task is to correct the existing estimates, add dominant-pair stability, and
prioritize transfer. Uncertainty follows the existing ensemble finding, not a new dispersion head.

This extends the intensity-displacement plan (removed 2026-10-18; `git show 95341fce:docs/plans/masster/masster_intensity_displacement_plan.md`).
The [generalization literature review](../../notes/masster_generalization_methods_2026_09_27.md)
owns detailed R-Drop and representation-learning recipes; the
memorization review (removed 2026-10-18; `git show 95341fce:docs/findings/masster/masster_memorization_plan_review_2026_09_27.md`)
owns the critique of identity-removal experiments. This document owns the reliability diagnostic
and the choice among the following three directions.

| Observation | Interpretation to test | Appropriate response |
| --- | --- | --- |
| Replicates disagree under well-matched recorded conditions | Measurement variability or unrecorded conditions | Replicate targets, uncertainty, metadata audit |
| Disagreement falls after matching the acquisition setting | NCE-only grouping conflates conditions | Correct replicate attribution; keep model CE input unchanged |
| Replicates agree, but held-out molecule predictions fail | Stable chemistry not learned or transferred | Representation/data intervention |
| Several proposed structures explain one observed peak | Latent assignment is ambiguous | Preserve existing summed-spectrum supervision; not a new intervention |

These mechanisms can coexist. Variability is conditional on the information available: an
unrecorded acquisition setting is not an immutable physical noise floor.

## Evidence and limits

Read the existing analyses before proposing new noise or uncertainty work:

- Displacement Phase 0 (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_intensity_displacement_phase0_2026_09_24.md`)
  already measured the squared-error floor: 1,315 val pairs, floor 0.0132 against 0.1616
  displacement on the repeat subset, leaving 67% reducible shared error (63% on test).
- Per-peak uncertainty (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_per_peak_uncertainty_2026_09_24.md`)
  already measured 9.8% noise share (19% below NCE 20) on its repeat subset. Replicate cosine
  is 0.970 mean / 0.989 median in its 10-ppm clustered frame. Its ensemble uncertainty test
  found mostly spectrum-level signal; per-peak spread/votes added presence AUC 0.793 → 0.800
  and within-spectrum intensity-matched error concordance was about 58-60%.
- The 0.946 figure in the [attempts audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)
  uses a different, 0.01-Da binned frame. It is not interchangeable with clustered cosine and
  should not anchor a new noise estimate. Neither number is a universal prediction ceiling.
- Both historical replicate analyses matched NCE only. Their results remain historical
  measurements, but their attribution to observation noise needs strict eV matching.
- Ranking decomposition (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_ranking_decomposition_2026_09_23.md`):
  train/val ranking error 0.011/0.121; little train ranking error remains to optimize.
- Energy/parts (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_memorization_energy_vs_parts_2026_09_26.md`):
  27.7% unseen-environment share is of fragment-explained intensity, not total intensity.
  Bond-local learner (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_bond_local_learner_2026_09_24.md`): only
  about +0.0006 beyond NCE 60, so local supervision needs a complementary-signal gate.

### User-reported strict-energy census (2026-09-27)

The user supplied these new measurements on NIST20 `[M+H]+`/FT/HCD MS2. They have not been
independently rerun in this plan; preserve that provenance until the census script/output is linked.

| Quantity | Reported measurement |
| --- | --- |
| Same-molecule/same-NCE groups with multiple spectra | 6,973 |
| Those with differing recorded eV | 4,750 (68%) |
| Median eV gap / relative gap | 1 eV / about 7% (relative-gap 75th percentile 15%) |
| Strict molecule/NCE/eV groups / pairs, all splits | 2,362 / 3,200 |
| Strict training pairs | About 1,900 |
| Missing recorded eV, earlier census | 18.5%; latest report 21.3%, cohort denominator unresolved |
| Rows in same-NCE/different-eV groups, earlier census | 5.5%; latest report 3.0%, cohort denominator unresolved |
| Duplicate peak lists in strict pairs | None found |

### Acquisition ladders: matching is separate from model input

> **Correction (2026-09-28).** The ladder table, the "no instrument-model/lab/source columns"
> sentence and the 0.0003 rounding estimate below are superseded by the
> [Orbitrap instrument finding](../../findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md).
> - **The ladders are two instruments.** Ladder A is the Elite Orbitrap and ladder B the
>   Orbitrap Fusion Lumos. NCE is the set value on both.
> - **The 87% round-to-5 rule holds trivially.** Elite records `eV = floor(NCE × m/z / 500)`,
>   which always satisfies it. Only 3.5% of NIST20 `[M+H]+` FT/HCD spectra are eV-stepped
>   series whose NCE was rounded.
> - **The fields exist upstream.** Instrument model and data source are in NIST23's raw MSP
>   and in `data/df/nist23_hr_df.csv` (`inst`; `notes` holds `Data_source`, `Vial_ID`,
>   `Nreps`). They are dropped from `spec_df.pkl`.
> - **Match on instrument + NCE.** Only 81 of 11,654 NIST20 same-molecule same-NCE groups mix
>   instruments.
> - **Repeat floor.** For the same molecule and NCE, Elite spectra from two NIST campaigns agree
>   at 0.946 (0.01-bin) / 0.985 (tolerance-matched) cosine for `[M+H]+`. Lumos vs Elite agree
>   at 0.640 / 0.715.
> - **The rounding estimate no longer bounds the CE-input question.** The instrument gap is
>   partly a mass-dependent energy offset and partly not energy. `c0` under-fragments Lumos val
>   spectra (−0.054 cosine after adjustment).

The user's follow-up measurements distinguish two ladders. These results are user-reported,
not independently rerun here. Preserve both earlier and latest cohort counts until filters,
missing-value handling and denominators are reconciled; do not silently replace 18.5% by 21.3%
or 5.5% by 3.0%. The 2,362 groups / 3,200 pairs above used the earlier molecule/NCE/eV key;
recompute counts under the revised keys below rather than carrying them forward unchanged.

| Ladder | Evidence reported | Audit matching rule |
| --- | --- | --- |
| eV-set, 5-grid NCE | `NCE = round_to_5(eV * 500 / precursor_mz)` holds exactly for 87% of scoped rows with both values | Molecule + recorded eV + `res`, within fixed adduct/instrument/fragmentation scope and this ladder; **do not require equal NCE** |
| Other ladder, NCE 2, 5, 9, 14, 27, 44, … | Non-5-grid NCE; recorded eV, where present, is about 1.5 times the simple conversion | Molecule + recorded NCE + `res` is a **candidate** rule only; validate that NCE is the acquisition setting before admitting these as strict repeats |
| Unknown/ambiguous ladder or missing set value | Insufficient metadata | Report separately; do not pool into the strict noise estimate or infer missing eV from rounded NCE |

Keep ladder identity in the grouping context so identical numbers from different acquisition
scales do not create cross-ladder pairs. Classify from the full molecule/group energy sequence
and conversion agreement where both values exist; a single value divisible by five is insufficient.
For missing-eV rows, use independent sequence/provenance evidence if available, not assumed
conversion. Record the rounding/tie convention and exact denominator when checking the 87%.
`res` and `col_gas` reportedly do not vary within the earlier groups; check this rather than
adding new model features. There are no instrument-model/lab/source columns in this table.

Consecutive `dset_spec_id` values and shared `group_id` support the adjacent-energy interpretation.
Example: molecule 12 was stepped through whole eV values 6-65; 8/9/10 eV map to NCE 20.
The reported cosine for spectra 1 eV apart is 0.976 whether or not rounded NCE changes. NCE
bin boundaries therefore provide no observed discontinuity to justify requiring NCE equality
on this ladder. Strict energy matching still does not prove independent acquisition replicates.

**Model CE input remains unchanged.** The selected `k1_ema` uses recorded NCE and no ACE.
This plan does not switch it to eV, impute eV, or propose an ACE-training arm. In the latest
reported cohort, only 3.0% of spectra participate in same-NCE/different-eV groups, with mean
cosine 0.990 to their bin average: `0.030 * (1 - 0.990) ≈ 0.0003` overall discrepancy, an order
of magnitude below the +0.003 experiment gate. Treat this as a rounding-headroom estimate on
that cohort/metric, not a demonstrated model gain or a universal bound. Missing eV (21.3% in
the latest report) and inconsistent conversion on the second ladder further undermine treating
eV as one interchangeable input scale. This agrees with keeping the
feature-headroom audit (removed 2026-10-18; `git show 95341fce:docs/findings/feature_headroom_2026_09_22.md`) as the metadata baseline.

Energy mismatch can inflate apparent replicate noise, but the historical ~10% noise share is
not a proven population upper bound: changing the key also changes cohorts, pair weights and
model-error denominators. Report matched-cohort contrasts alongside full-cohort estimates.
The direction remains transfer-first; this correction concerns measurement attribution.

## 1. Uncertainty: extend the existing ensemble route

The per-peak finding (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_per_peak_uncertainty_2026_09_24.md`) is the
starting point. Ensemble uncertainty has been tested; what remains untested is whether independent
seeds materially improve per-peak information beyond spectrum-level confidence and intensity.

**Standing rule:** a residual-trained uncertainty/error model must use molecules held out from
its base predictor or out-of-fold predictions. Relabelling in-sample training residuals as
“predicted error” does not make them valid. Train fit near 0.92-0.96 would teach confidence in
memorized molecules. This restriction also applies to feature fitting and interval calibration.

Independent seed pairs now exist:

| Recipe | Seed 397623 | Seed 811237 |
| --- | --- | --- |
| `c0` | `vcpho5qm` | `1oxhvk0n` |
| `k1` | `wg0nxymi` | `e39ko0zr` |
| `u2` EMA | `c8hh9j1b` | `qwhsp2fg` |

Run completion is not checkpoint availability: the route plan says second-seed weights were not
uploaded; the live W&B file read confirmed no checkpoint files for `qwhsp2fg`. Locate preserved
cluster checkpoints or existing predictions before scheduling any export. Do not replace a
missing seed with an adjacent epoch and call it independent. The saved `k1_ema` is the main
single-model error reference; `k1_ema` plus `u2` is not a same-recipe seed pair.

Start at K=2 for one available same-recipe pair. Compare intensity-only, intensity plus
spectrum-level disagreement/CE/novelty, then additional per-peak spread/votes. Evaluate
within-spectrum intensity-matched concordance, incremental presence AUC and calibrated intervals
on disjoint molecules, reporting missed/unsupported peaks separately. Larger K is optional future
work, not permission to train additional backbones. Model spread is not measured observation noise
and agreeing members can share systematic errors.

**Replicate-trained dispersion head: DEFERRED.** About 1,900 strict training pairs is weak support
for a flexible head and existing measurements put most error elsewhere. Constant/CE-conditioned
noise summaries are sufficient for the immediate audit. Reopen a learned observation-variance
head only if strict repeats reveal substantial, reproducibly predictable variability beyond those
baselines. This is distinct from the held-out residual/ensemble error model.

## 2. Measure which intensity relationships are reproducible

A single spectrum-level replicate cosine hides whether the same dominant peaks consistently
win. [Place/NIST (2021)](https://www.nist.gov/publications/development-tool-determine-variability-consensus-mass-spectra)
measures m/z and intensity variability in replicate spectra, demonstrated using PFOA in simple
solution and house dust extract. This supports measuring variability directly; its magnitude
cannot be transferred to this dataset.

### Cohort and alignment

- Inventory unmerged spectra and available metadata first. Count molecules, condition groups,
  repeat counts, independent acquisitions and missing fields. Separate duplicate records from
  independent measurements; do not treat copied library entries as reproducibility evidence.
- Match molecular identity/stereochemistry where represented, adduct, charge, polarity,
  instrument/fragmentation method and ladder. On the eV-set ladder match recorded eV and `res`,
  without an NCE-equality condition. Keep the second-ladder NCE/res rule provisional. Check
  `col_gas`. The table
  lacks source/protocol identifiers. Report strict-energy groups separately from same-NCE-only
  groups and retain the provenance limitation. Never silently
  equate normalized and absolute collision energy or merge across energies.
- Construct a common observed peak space with an explicit instrument-appropriate tolerance.
  Flag ambiguous matches and test tolerance sensitivity. Retain peaks outside candidate
  coverage, and report both all-observed and candidate-covered results. Fine-bin boundary
  artifacts must not be mistaken for changing chemistry.
- Normalize each spectrum consistently before aggregation. Fit any matching/calibration choices
  on training data. Use validation for diagnosis and preserve untouched confirmation molecules;
  historical test results already used for development are not fresh confirmation.

### Statistics that distinguish the mechanisms

For each condition group, estimate peak occurrence frequency, mean intensity, dispersion, and
log-ratios for well-observed dominant pairs. State the small offset and its sensitivity when
using log-ratios. Measure how often the dominant pair swaps order, separating near ties from
large reversals. Select pairs from other replicates or a fixed rule, rather than repeatedly
selecting the evaluated target's winner. Report precursor-versus-fragment separately.

Two replicates give a pairwise discrepancy, not a reliable per-peak variance estimate. Show
repeat counts and uncertainty; pool or shrink sparse estimates using training-only strata.
Weight molecules equally for headline results so a few frequently measured compounds do not
dominate. Bootstrap molecules, keeping their spectra and condition groups together.

Compare a fixed held-out model against each target and against a consensus of the *other*
replicates. The evaluated target must not contribute to its own consensus. Report errors by CE,
environment frequency, molecular novelty, precursor fraction and metadata completeness.
This distinguishes stable model mistakes from orderings that measurements themselves reverse.

For a fixed common-coordinate vector `Y`, independent conditional replicates `Y` and `Y'`
with mean `mu` and covariance `Sigma` satisfy:

```text
0.5 * E[||Y - Y'||^2] = trace(Sigma)
E[||f - Y||^2] = ||f - mu||^2 + trace(Sigma)
```

These identities justify a squared-error diagnostic for a predictor fixed independently of
the evaluation measurements. They do not justify subtracting raw cosine disagreements. A
consensus of `r` independent replicates still has covariance `Sigma/r`; duplicated records,
condition mismatch and dependent acquisitions violate the simple interpretation. Use identical
normalization and coordinates for all terms, and report the assumptions with the estimates.

**Deliverables:** strict-energy corrections to the two existing noise estimates and a new
dominant-pair stability table. Reuse the historical frames first, then report alignment sensitivity
separately. No new corrected numbers have been produced in this plan.

## 3. Train intensity relationships that transfer across molecules

The missing supervision may be relationships shared by different training molecules. A
within-spectrum ranking loss does not necessarily supply them: MASSter already orders its
training spectra well. The intervention must change what generalizes, not just re-express
an already fitted label.

### A. Same-input consistency as a bounded first control

Use two stochastic views of the same molecule and conditions with fixed candidate support.
Apply R-Drop at aggregated peak distributions as specified in the
[method review](../../notes/masster_generalization_methods_2026_09_27.md). Compare against a
matched two-pass supervised control. This tests sensitivity to model stochasticity; it does
not teach measured intensity variance and must not equate dropout spread with replicate noise.
Do not impose intensity invariance across collision energies.

### B. Cross-molecule representation supervision

Before training, perform a training-only neighbor census under matched adduct/CE/acquisition.
Ask whether similar observed fragmentation patterns across different molecules are sufficiently
reproducible and frequent to define useful relationships. Check that similarity is not explained
only by precursor dominance, mass, peak count or chemical near-duplicates.

If that census passes, use an auxiliary projection of the learned representation and a
Rank-N-Contrast-inspired target-distance objective. Define the spectrum comparison explicitly:
absolute m/z, neutral-loss coordinates, or matched local outcomes answer different questions.
Neutral-loss alignment can expose shared transformations but also creates accidental matches;
compare against precursor/mass-only and shuffled-neighbor controls. Include true training
neighbors only, with no validation/test spectra in pair construction.

[Rank-N-Contrast (2023)](https://papers.nips.cc/paper_files/paper/2023/hash/39e9c5913c970e3e49c2df629daff636-Abstract-Conference.html)
is evidence for regression representation learning in other domains. The spectrum-distance
choice and the MASSter adaptation remain hypotheses. Keep the main peak-supervised objective;
do not force the entire representation to forget chemically necessary distinctions.

### C. Condition local relationships on competing pathways

The same fragment need not have the same intensity in different parent molecules. Its formation,
charge retention, survival, competing channels and acquisition conditions can differ. A local
auxiliary task should therefore predict a conditional relationship, such as precursor versus
contender given bond environment, parent context and CE, rather than assign a universal fragment
intensity. The current spectral loss already compares `pred_mzs`/`true_mzs` and aggregates
indistinguishable contributions. Preserve this behavior in any auxiliary objective; it is not
a new proposed fix.

Revisit this branch only after stable errors show complementary local signal beyond the existing
bond-local learner. Its small gain and high-CE failure are a real gate. Extra supervision through
a shared encoder can also harm the final head; use an explicit auxiliary readout and measure
that interference instead of assuming multitask learning helps.

**Acceptance:** improve held-out molecules on the production metric and ranking diagnostics,
including stable rare-environment and high-CE strata, against matched controls. Compare after
both arms receive the same calibration opportunity. Use molecule-level paired uncertainty,
confirm promising screens with additional seeds and reserved molecules, and retain coverage
checks. A lower representation loss or training ranking loss alone is not success.

**Data comparator:** the variance/novelty finding (removed 2026-10-18; `git show 731fe493:docs/findings/masster/masster_variance_novelty_2026_09_24.md`)
projects +0.018 single-model cosine from NIST23 coverage via the observed similarity curve.
Compare each transfer arm’s measured gain and cost with that projection and the measured ensemble
benefit. +0.018 is an opportunity-cost comparator, not a measured NIST23 training result or a
mandatory significance threshold.

## Detailed execution plan: selected `k1_ema` checkpoint

### P0. Acquire and freeze the reference (download complete)

User scope on 2026-09-27: download the selected checkpoint and prepare this detailed plan.
The following strict-energy reanalysis, any needed exports and ensemble calibration are
planned, not launched. Replicate-trained head fitting is deferred.

| Item | Pinned value |
| --- | --- |
| W&B run | [`k1_ema`, `j4u25qfd`](https://wandb.ai/frag-gnn/frag-gnn/runs/j4u25qfd) |
| Remote run file | `ckpt/model-epoch=050.ckpt` |
| Local file | `saved_ckpts/k1_ema_j4u25qfd/ckpt/model-epoch=050.ckpt` |
| Local metadata | `saved_ckpts/k1_ema_j4u25qfd/provenance.json` (W&B config, hashes, epoch, step) |
| Size | 147,770,232 bytes |
| SHA-256 | `eea40c0937e7c7e8307105845ea1866128c029e065cb02923d70e6e1c6e6b521` |
| W&B MD5, base64 | `/PZ36YP0kh5OD6LNfTu9Ew==` |
| Epoch / global step | 50 / 15,708 |
| Seed / EMA decay | 397623 / 0.999 |
| Validation metric | `val_cos_sim_0.01_epoch/mean`, 0.7427291870 at epoch 50 |
| Late validation mean | 0.7420841634 across epochs 54-63 (10 points) |
| Dataset / split | `data/proc/nist20/spec_df.pkl`; `data/split/nist20_inchikey` |
| Scope | Unmerged `[M+H]+`, FT, original unfiltered targets, no subsampling |

Selection was on validation, not test. This is the strongest recent comparable checkpoint
checked, not proof that the recipe's small increment over EMA alone is statistically established.
The cosine-schedule variant `f2oacs52` has best/tail 0.741983/0.741002; EMA-only `c8hh9j1b`
has 0.739387/0.739189. The historical plan still calls some now-finished runs unsubmitted;
use this live W&B read for this selection without rewriting unrelated experiment history.

Recover the config with `get_config_from_ckpt`, then use `init_model_from_ckpt` from
[`script_utils.py`](../../../src/fragnnet/utils/script_utils.py). Its strict state loading
must pass. The checkpoint's `state_dict` holds EMA weights; `ema_training_state_dict` holds
training weights and must not replace them for this analysis. Keep `eval()` and freeze all
parameters and BatchNorm state. Do not enable stochastic dropout to estimate measurement noise.
Record any runtime-only overrides separately; do not overlay the current template blindly.

**Load verification:** strict CPU loading passed with 9,212,861 parameters, after supplying
10 missing newer config keys. Unmodified checkpoint-only configuration first failed on
`masster_intensity_mz_positional_encoding`; current code does not automatically migrate all
these keys. The required overlay is saved in
`saved_ckpts/k1_ema_j4u25qfd/compatibility_defaults.json`:

```json
{
  "masster_encoder_atom_mask": 0.0,
  "masster_encoder_bond_dropout": 0.0,
  "masster_global_message_swap": 0.0,
  "masster_global_message_swap_unit": "atom",
  "masster_intensity_mass_window": 0.0,
  "masster_intensity_mz_positional_encoding": false,
  "masster_light_side_token": false,
  "masster_memorization_sink_dropout": 0.0,
  "masster_set_key_dropout": 0.0,
  "masster_set_message_substitution": 0.0
}
```

Apply only to absent keys, preserving all saved settings. These disable the newer mechanisms;
the swap-unit choice is inactive at zero swap rate. The original checkpoint was not rewritten.
Record this overlay with future exports. Strict loading was checked, not a forward pass or
full validation reproduction. No source-code compatibility fix was made in this task.


At execution, capture Git revision plus dirty-diff hash, package versions, checkpoint hash,
dataset/split fingerprints, resolved config and spectrum IDs in an output manifest. Acquisition
occurred at revision `394dbab4343e99ff61d26cb92c5dab3b0735a23e` with unrelated model/CUDA edits
present; successful loading alone does not demonstrate prediction parity with the training code.
Before GPU inference, check those edits are stable and confirm repeatable small-batch outputs.

### P1. Recover artifacts and confirm strict-energy groups (CPU)

First locate the historical exports and scripts. The findings reference
`output/cross_eval_mh_ft_clean_2026_09_23/analysis/{displacement,per_peak_uncertainty}/`, but
that parent directory is absent in this checkout at this revision. The user reports exports
are local; resolve their actual location or recover them before promising a quick rerun. Do not
claim the measurements were reproduced. Promote reused ignored scripts into a tracked analysis
entry point, `scripts/eval/masster_intensity_reliability.py`, with provenance. Reuse
mass-tolerance and molecule-bootstrap conventions from
[`debug_ce_replicate_noise_floors.py`](../../../scripts/debug/fragnnet/debug_ce_replicate_noise_floors.py)
and peer-comparison logic from
[`nist_bad_data_review.py`](../../../scripts/eval/nist_bad_data_review.py).
Use [`intensity_displacement.py`](../../../scripts/eval/intensity_displacement.py) to
reproduce its historical first-two-pairs/candidate-projected statistic, then change the grouping
explicitly. Do not treat that frame as interchangeable with the per-peak finding’s frame.

1. Read spectrum/molecule tables and the saved split. Verify unique `spec_id`, molecular-key
   membership and absence of molecular overlap across splits. Reproduce the saved preprocessing;
   distinguish the source table from targets after that preprocessing. Never modify either.
2. Confirm CE source/units, eV and `res` missingness, and the reported absence of instrument-model/
   lab/source columns. Start with `mol_id`, `prec_type`, `inst_type`, `frag_mode` and ladder identity.
   On the eV-set ladder require equal recorded eV and `res`, not equal NCE. Validate the
   second-ladder NCE/res rule separately; check `col_gas`.
   Keep inferred/model-filled CE
   in a separate tier. A numeric NCE match is not evidence of independent matched acquisition.
3. Label duplicate rows and identical peak-list copies; quantify them with and without exclusion.
   Use provenance to resolve independent acquisitions where possible. Do not invent independence
   when acquisition identifiers are absent: label those groups provenance-unknown.
4. Write group membership, excluded IDs/reasons, repeat-count distribution and counts of molecules
   in each split/tier. Report groups with 2, 3-4, and at least 5 replicates separately.

**G1:** report strict-energy-matched groups separately from the assumption of independent
acquisitions. Missing source identifiers remain a limitation, not a reason to block the correction.
Show old NCE-only and strict estimates on matched molecules/CE strata as well as their respective
full cohorts. Never substitute single-spectrum training residuals.

### P2. Correct existing floors, then add dominant-pair stability (CPU)

First reproduce each historical statistic with its original pairing, clustering, normalization
and denominator. Recompute with the ladder-specific strict keys above in the same frame. Then add an all-pairs,
molecule-balanced sensitivity read; do not confound grouping correction with an estimator change.
Keep the Phase 0 candidate-projected cosine frame separate from the per-peak clustered frame.
For the additional reproducibility tables, normalize each spectrum by its total
retained intensity before aggregation. Start FT alignment at the existing 10 ppm / 200 Da floor
(`max(10 ppm * m/z, 0.002 Da)`), with 0.5x and 2x tolerance sensitivity. This is an analysis
convention, not a claim of universal instrument accuracy.

Construct order-independent mass groups with a common-overlap constraint rather than chaining
arbitrarily distant peaks through neighbors. Flag ambiguous matches; never silently assign one
peak twice. Preserve unmatched peaks and mass totals. Test spectrum-order invariance. Report:

- Per group/peak: mean normalized intensity, unbiased sample variance, occurrence fraction,
  repeat count and ambiguity status. Absence means zero recorded intensity on this scale,
  not necessarily zero chemical production.
- Per group: replicate cosine, squared discrepancy, dominant-peak flip rate and precursor share.
- Per dominant pair: intensity margin and ordering frequency. For descriptive all-replicate
  summaries label pair selection as descriptive; for evaluation select pairs from other
  replicates or a fixed model-based rule. Separate near ties using thresholds frozen on train.
- Molecule-weighted aggregates and 95% molecule-bootstrap intervals (2,000 draws, seed 17),
  by CE, novelty and metadata tier. Also show spectrum-weighted results for metric reconciliation.

Data-dependent alignment can itself affect variance estimates. Repeat variance estimation on a
fixed coordinate system as a sensitivity check; label alignment-dependent results accordingly.
Do not mix tolerance-aligned diagnostics and 0.01-Da production cosine in one subtraction.

**G2:** proceed after synthetic checks recover zero variance for identical spectra, positive
variance for changing heights, and stable alignment for small mass shifts. Conclusions should
survive reasonable tolerance changes; otherwise investigate alignment before modeling noise.

### P3. Dominant-pair errors and transfer decision; export only as needed

Use existing f0/f1 exports first to correct their historical estimates. Use `k1_ema` for the
new current-model comparison; export only if its required predictions are unavailable. Use
[`run_save_inference_eval.py`](../../../scripts/eval/run_save_inference_eval.py) HDF5 path where
possible, verifying its resolved configuration against the checkpoint. First export a fixed
small smoke cohort, then the repeat cohorts. Full validation reproduction is a separate check,
not implied by a repeat-subset score. Match the original aggregation before comparing 0.742729.

The first pass needs only `spec_id`, molecular/condition keys, predicted m/z/intensity and
candidate coverage. Export full valid support with original probabilities; no target-informed
pruning or post hoc renormalization to observed peaks. Include predicted-only and observed-only
mass in comparisons. Join by identifiers, never row position. Cache repeated identical model
inputs only after verifying that all conditioning inputs and candidate generation agree.

For each held-out condition group, compare the model with individual replicates and their
consensus. Compare each measured replicate with the consensus of its peers. Quantify squared
error and replicate dispersion on the same coordinates, plus separate production cosine,
dominant-pair accuracy, precursor-versus-contender accuracy and uncovered intensity.
Do not force negative finite-sample estimates of excess squared error to zero before averaging.
The model/replicate independence assumption applies to held-out molecules, not memorized train
spectra. Training results are descriptive only.

For the dominant-pair comparison, freeze pair selection before scoring each held-out replicate.
Report replicate order-disagreement and model-versus-replicate ordering error on exactly those
same pairs, including margins/near ties. A deterministic NCE-only model has zero variation across
otherwise identical replicate inputs; its relevant “swap” statistic is disagreement with the
observed ordering, not stochastic changes between duplicate forward passes.

**G3 decision:** proceed to §3 transfer work as the primary direction unless corrected evidence
materially contradicts the existing result. P4 is the parallel ensemble-confidence extension,
not a prerequisite to transfer. Do not define noisy examples solely from model residuals.

### P4. Independent-seed uncertainty extension (after artifact check)

1. Recover one same-recipe seed pair in §1, with checkpoint hashes and exact validation-best
   epochs. Compare resolved settings and code revisions; seed labels alone do not ensure a
   controlled pair. Prefer a pair with both exports already available. No new seeds are launched.
2. Export identical raw held-out spectra for both members with complete candidate support.
   Aggregate member predictions in one mass frame; use target-independent predicted groups
   for inference features and account for target-only peaks separately at evaluation.
3. Reproduce the existing uncertainty probe at K=2. Fit/calibrate only on molecules unseen by
   both base models: split validation molecules deterministically into fit/calibration/evaluation
   partitions (50/25/25, seed 17; save IDs). Validation selected the base checkpoints, so these
   are development estimates. Historically inspected test data are not untouched confirmation.
   An out-of-fold training alternative is valid only if suitable OOF predictions already exist.
4. Compare B0 intensity-only, B1 plus spectrum-level confidence/CE/novelty, B2 plus per-peak
   spread/votes. Use the same simple presence/error model class and partitions for all three.
   Start with features already exported. Candidate descriptors are optional later additions;
   an encoder-feature export or a new neural head is not required for this first comparison.
5. Calibrate interval scale on the calibration partition only, using a positive scale floor
   fixed on fit data. Report empirical coverage and width with molecule clustering, conditional
   strata and a spectrum-level baseline. Do not claim simultaneous peak coverage or distribution-
   free peakwise coverage from molecule splitting alone; specify the sampling unit and assumptions.

### P5. Acceptance, tests and output contract

For the strict correction, report both historical noise shares with cohort/pair counts and
matched-cohort sensitivity. The novel statistic is dominant-pair stability against model ordering
error. Show CE/novelty/intensity strata and molecule-bootstrap intervals (2,000 draws, seed 17).
Do not call a changing subset ratio a universal upper bound on noise or conditioning gains.

For uncertainty, the gate is incremental value of B2 over B1: presence AUC, within-spectrum
intensity-matched error concordance, and calibrated interval width at matched empirical coverage.
Use paired molecule-bootstrap intervals. If incremental gains are negligible or inconclusive,
retain spectrum-level confidence plus intensity-conditioned uncertainty and stop per-peak model
complexity. Compare with the prior +0.007 AUC / 58-60% concordance without confusing recipes or
cohorts. Ensemble averaging gains and uncertainty calibration gains are separate outcomes.

Implementation tests: split integrity; ladder classification, eV/res matching without an NCE constraint on the eV-set ladder,
separate handling of missing eV and unverified second-ladder groups;
duplicate handling; historical estimator reproduction; mass/intensity conservation; alignment
ambiguity/order sensitivity; no self-inclusion in peer consensus; consistent pair selection;
feature/label joins; residuals only from held-out/OOF predictions; and unchanged frozen-model
predictions. Run focused pytest/Ruff and a small real-data smoke before any larger export.

Planned output root: `output/masster_intensity_reliability_j4u25qfd_e050/` (not yet created).
Store the provenance manifest, recovered artifact inventory, strict group IDs, historical/strict
floor comparisons, pair stability and model error tables, ensemble membership/config differences,
calibration split IDs and baseline/per-peak comparison tables. Keep large artifacts ignored;
link a findings document after new measurements exist. The main `k1_ema` file remains pinned in
P0; additional seed-pair weights have not been retrieved by this task.

## Execution order and handoff

1. Locate the existing artifacts and recompute Phase 0 and per-peak noise shares on strict
   ladder-specific groups (eV/res first; second-ladder NCE/res only after validation), preserving
   each original metric frame and adding matched-cohort
   sensitivity. The user-reported census supplies the starting counts, not a rerun by this plan.
2. Measure dominant-pair ordering stability in strict groups against model ordering error on val.
3. Prioritize §3 transfer, with the NIST23 +0.018 projection as a cost/benefit comparator.
4. Extend §1 through an existing independent-seed pair; retain a spectrum-level baseline.
   The replicate-trained dispersion head is deferred, not an immediate branch.

Completed: reviewed the previously omitted findings, revised priorities, and retained the
verified `k1_ema` checkpoint/provenance and compatibility overlay. No new analysis or training.
Next check: actual locations of historical exports and second-seed checkpoints. Some documented
local artifacts are absent here; report missing inputs rather than recreating results from prose.
Validation: local links and whitespace checks. The missing general roadmap is marked unavailable
in the index/state instead of being presented as a live canonical entry point. Other pre-existing
broken links remain outside this revision. No production code changed.

## P1 preliminary census (2026-09-27)

The staged audit in `scripts/eval/masster_intensity_reliability.py` read the source
NIST20 table and four saved split files without modifying them. Its current output is
`output/masster_intensity_reliability_j4u25qfd_e050/` (CSV membership because the
configured environment has no Parquet engine). Across the 268,188 saved split spectra,
strict matching on molecule, adduct, instrument/method, source, gas, resolution, CE type,
and both recorded CE fields yields 3,402 repeated-condition groups: 3,004 pairs, 370
groups of 3-4, and 28 groups of at least 5. Train has 1,492 such groups across 146
molecules; validation has 445 across 41 molecules. The secondary split shares 148
molecules with test; primary train/validation/test molecules are disjoint. There are 216
rows in identical-peak-list groups and no duplicate `(dset, dset_spec_id)` records.

The table has no acquisition identifier. Different library record IDs and different
peak lists cannot prove independent acquisitions. G1 therefore fails for observation-noise
claims and the P4 head must wait for independent acquisition provenance. This is a broad
source/split census, not yet the checkpoint-filtered `[M+H]+`/FT cohort or the
post-preprocessing target census. P2-P5 have not run. The next check is to locate
acquisition provenance in upstream NIST metadata or an independent repeat dataset; only
then assess condition-matched train and held-out cohorts.
