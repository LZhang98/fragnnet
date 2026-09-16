# MASShatter improvement plan

> **Status:** PROPOSED · **Updated:** 2026-09-08
> Prioritize intensity assignment and support gating while preserving candidate coverage.
> The next experiment targets unmerged NIST20 MA/MI spectra; training has not started under this plan.

## Objective and scope

Improve spectral similarity and practical output sparsity without materially reducing weighted
recall or formula-retrieval performance. Keep individual collision-energy spectra, full acquisition
conditioning, and the existing unmerged InChIKey split and target preprocessing.

This is the MASShatter-specific execution sequence under the
[model improvement roadmap](model_improvement_plan.md). Use the detailed
[intensity plan](masshatter_nonprobabilistic_intensity_plan.md) for output and loss contracts,
the [channel-selection findings](../findings/masshatter_channel_selection_findings.md) for support
semantics and pruning evidence, and the
[performance plan](masshatter_topology_assembly_performance_plan.md) for throughput gates.
For this task, the sequence below replaces the older intensity plan's initial cosine-first queue.
It does not declare a winning output parameterization or supersede compatibility requirements.

## Evidence and limits

- The recent merged Gaussian support-gated JSS run
  [6jwo00h6](https://wandb.ai/frag-gnn/frag-gnn/runs/6jwo00h6) reached validation
  `cos_sim_0.01` of 0.7743 at epoch 42, versus 0.7649 for the Gaussian-supervised
  [ungated run](https://wandb.ai/frag-gnn/frag-gnn/runs/771f78ge) at the same epoch.
  These histories were inspected during the preceding recommendation. They are preliminary
  single-seed evidence, not an unmerged result or a promotion-quality estimate.
- The [merged CE baseline](https://wandb.ai/frag-gnn/frag-gnn/runs/syjbloof) reached
  0.7750 at epoch 42. It used a separate proposal encoder and no proposal warmup;
  the JSS runs used shared encoder features and proposal-only training through epoch 15.
  Their comparison does not isolate the loss. The necessity of warmup is unresolved.
- The [channel-selection findings](../findings/masshatter_channel_selection_findings.md)
  distinguish output clutter, soft fragment attribution, and mass-compatible support.
  Their historical pruning numbers require chemistry revalidation. High retained proposal
  probability did not guarantee preserved spectral coverage.
- The [code audit](../findings/masshatter_code_audit.md) records chemistry, gradient-routing,
  and inference corrections. Re-evaluate affected checkpoints on corrected code before using
  their metrics as current references. Merged consensus targets also differ from historical
  exact-union targets and must not be compared as if the training data were identical.
- Earlier comparisons of shared versus separate encoders changed other training conditions.
  They do not establish that removing the second GNN caused better convergence.

## Ordered experiments

### 1. Establish a corrected unmerged control

Record one resolved unmerged anchor config, current code/package identity, data and split
fingerprints, seeds, tolerances, batch construction, and recoverable checkpoint locations using
the [session workflow](../workflow/session.md). Check real-checkpoint inference and chemistry
parity before interpreting validation metrics. Keep preprocessing fixed; do not transfer merged
consensus construction or introduce new peak filters into this comparison.

Use probability output with the existing Gaussian CE objective as the control. Keep U256
initially and freeze its enumeration budget across model comparisons. Preserve CE-specific
proposals, including adduct, instrument, and fragmentation-mode context.

**Check:** finite training losses, reproducible validation inputs, consistent prediction/export
behavior, and checkpoints that remain available after scheduler termination. Record excluded
or skipped samples and use common evaluation IDs when comparing models.

### 2. Locate the error and settle the proposal architecture

Measure candidate weighted recall, learned spectral similarity, and the best attainable
spectral similarity on the selected candidate support. Define the oracle under the same mass
aggregation, matching tolerance, and metric as the learned prediction; weighted recall alone
is not an oracle cosine. Report these by CE, adduct, instrument, and fragmentation mode.

- Strong support oracle with weak learned similarity: prioritize intensity assignment.
- Weak support oracle: investigate proposal ranking and coverage in the affected strata.

Follow the [P1-to-P2 gate](masshatter_nonprobabilistic_intensity_plan.md#p1-to-p2-remove-the-second-molecular-gnn):
compare one molecular GNN with a separate proposal GNN under the same CE-specific selection,
CE loss, batching, candidate budget, seed, and schedule. Sharing encoder features and sharing
one candidate set across collision energies are different treatments. Keep the latter disabled
for this comparison.

**Check:** adopt encoder sharing only if support recall and oracle quality remain within a
predeclared non-inferiority margin based on paired-seed variation, and report memory and full-step
throughput. Freeze the chosen architecture before comparing intensity objectives.

### 3. Compare CE with Gaussian support-gated JSS

On that fixed architecture, compare the corrected CE control with independent JSS intensity
and Gaussian support gating. The existing implementation combines a sigmoid support gate with
a Softplus magnitude and trains support with binary cross entropy. Reuse that implementation
and its explicit config fields; preserve dense chemical support during training with
`mass_shatter_apply_channel_selection_in_training: False`.

Use Gaussian attribution first because several structural channels may explain the same mass.
Retain Hungarian attribution as a later ablation, not unique chemical ground truth. Coalesce
equal predicted masses for spectral evaluation before interpreting precision or sparsity.

This first comparison tests the complete output/loss/support treatment. If it improves quality,
compare against independent JSS with identical Gaussian support supervision but the gate disabled
to isolate the contribution of gating. Do not credit the complete treatment's gain solely to JSS.

Use the same warmup policy in matched arms. A 16-epoch proposal-only phase is not established
as necessary for unmerged spectra. Resolve it from the current merged warmup controls or a
bounded matched pilot before making it part of the anchor; record the chosen policy explicitly.

**Check:** spectral quality improves beyond paired-seed variation while candidate coverage,
weighted recall, and stability are preserved. Inspect clipping and branch gradients using the
intensity plan's diagnostics if a change in loss also changes proposal behavior.

### 4. Tune practical sparsity after training

Evaluate each fixed checkpoint with a dense reference and conservative support or final-intensity
thresholds. Allow unsupported fragments to disappear from the sparse output; account for existing
nonempty-spectrum fallbacks. Avoid assigning a fixed number of channels to every fragment as
the default production rule.

Report weighted recall, count recall, cosine/JSS, precision, and emitted peak count after equal-mass
aggregation, including the standard exported-intensity cutoff and CE/adduct slices. Use the
[findings' proposed threshold gate](../findings/masshatter_channel_selection_findings.md#preserve-the-recall-advantage):
weighted-recall decrease at most 0.005 and absolute cosine decrease at most 0.001 versus dense
output. These are proposed acceptance limits, not demonstrated unmerged outcomes. Keep dense
output if no positive threshold passes; raw precision alone is insufficient.

### 5. Reduce training cost without sacrificing support

Profile complete optimizer steps with the actual DataLoader and hardware. Test E1024 versus
E4096 as a separate budget comparison after freezing the model treatment. Judge the tradeoff
using oracle quality, candidate weighted recall, learned similarity, memory, and optimizer
steps per wall-clock hour. Smaller enumeration budgets are not assumed to be faster.

Use the existing performance plan's measurement gates before adding candidate-kernel complexity.
Ensure jobs can resume and complete enough optimization to assess convergence. Compare equal
training progress as well as equal wall time; an unfinished run is not evidence of an
architecture ceiling. Extend matched arms equally or mark the comparison inconclusive when
their validation curves are still improving at the cutoff.

## Promotion and deferred work

Use paired seeds for screening, validation for selection, and the roadmap's multi-seed,
paired-bootstrap and retrieval checks before promotion. Report plateau behavior rather than
only best-epoch scores; record the selected checkpoint separately. Keep metric definitions
explicit: historical binned cosine and instrument-tolerance Hungarian cosine are not interchangeable.
Evaluate the locked winner on test only after validation selection.

Defer larger transformers, additional proposal mixtures, new chemistry heads, and generic candidate
expansion until the error/oracle analysis identifies their target. Keep cross-CE auxiliary losses
out of the initial comparisons. Revisit CRF/CMF mode supervision only on adducts with identifiable
mode signal, using the existing channel-selection plan's gates.

## Current state and next check

Execution update (2026-09-11): the [W&B screen](../findings/wandb_screen_2026_09_11.md)
records runs launched after the preparation handoff below, including loss, support,
subformula-filtered, and shorter-warmup arms. Its observed results and next checks
supersede the historical no-submission statements in this plan.

### Merged warmup loss/transform screen (2026-09-08)

Five configs are prepared as a precursor to the unmerged comparison:

| Config | Output and loss | Transform |
| --- | --- | --- |
| [CE control](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_probability_ce_warmup_e0_15_dense.yml) | Probability, Gaussian CE | none |
| [JSS](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_independent_jss_warmup_e0_15_dense.yml) | Independent, JSS | none |
| [JSS square root](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_independent_jss_sqrt_warmup_e0_15_dense.yml) | Independent, JSS | sqrt |
| [KL](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_independent_kl_warmup_e0_15_dense.yml) | Independent, KL | none |
| [KL square root](../../config/masshatter/merged/warmup_loss_transform/nist20_merged_independent_kl_sqrt_warmup_e0_15_dense.yml) | Independent, KL | sqrt |

All use merged tolerance-consensus targets, seed 397623, U256/E1024, one proposal mixture,
batch size 64 with accumulation 4, and proposal-only epochs 0-15 inclusive. They preserve
the existing warmup implementation: proposal gradients continue into the shared molecular
and context encoders after epoch 15. This screen compares losses under fixed routing;
it does not isolate warmup's causal effect. Proposal targets remain untransformed.
The four independent arms use Gaussian attribution without an intensity prior, support BCE
weight 0.1, and soft intensity gating. The CE control has neither gate nor support BCE.
Hard fragment/channel thresholds are zero during validation and checkpoint selection.
Existing experiment configs are preserved. No runs have been submitted.

Checkpoint uploading remains disabled; verify persistent checkpoint storage and adequate
wall time through the cluster submission workflow before launching these runs.


No model code changed. All five configs passed template resolution, model construction, and
epoch-15/16 phase-transition checks with the configured interpreter. A resolved-config comparison
confirmed that only the intended loss, transform, output/support settings, and run metadata vary.

Next: verify checkpoint persistence before launching the merged screen. Resolve the corrected
unmerged anchor and warmup policy before the matched CE encoder comparison. Run-specific
provenance, oracle measurements, paired-seed variance, and unmerged treatment results remain
pending. No training, scientific evaluation, or full test suite was run for this config handoff.

### Matched H-support ablation configs

Two additional JSS/no-transform controls are prepared:

- [No H support](../../config/masshatter/merged/support_ablation/nist20_merged_independent_jss_no_h_support_warmup_e0_15_dense.yml): support head, binary support logits, and gate disabled; support loss weight zero.
- [H supervision only](../../config/masshatter/merged/support_ablation/nist20_merged_independent_jss_h_supervision_only_warmup_e0_15_dense.yml): Gaussian support BCE weight 0.1, but no soft intensity gate.

Compare with the existing gated JSS/no-transform arm above. Other settings remain fixed,
including shared-encoder warmup and zero hard-pruning thresholds. The separate directory
preserves the original five-config glob. To generate these two jobs with the same Vulcan
procedure, use `config/masshatter/merged/support_ablation/*.yml` as the config glob.
No training jobs were submitted for these controls.

### Corrected unmerged CMF anchor prepared (2026-09-16)

Step 1's unmerged control now has a config:
[nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml](../../config/masshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_unique256_enum4096_gine_gsa384_bondcentric_a2_b3_transformer_full_context_dual_ce_proposal_selection_only_inst_tol.yml).
It is the lead unmerged arm
([`ef7fl5c6`](https://wandb.ai/frag-gnn/frag-gnn/runs/ef7fl5c6), validation `cos_sim_0.01`
0.7515) with exactly three deltas, confirmed by a resolved-config comparison in which
`spec_params` is the only differing key:

1. `spec_params.ace: True`. `MassShatterPL` passes `spec_params.nce`/`ace` straight to the
   model's `use_nce`/`use_ace`, so this turns on the dual-CE context of the selected
   full-metadata baseline. Parameter count 8,431,302 to 8,436,336.
2. `inst_types` drops `IT`. Measured on `data/split/nist20_ma_mi_inchikey`: 480,002 spectra,
   FT and QTOF only, zero IT rows. The IT slot only widened the embedding table.
3. `mass_shatter_channel_diagnostics: True`, so the valid/attributed/retained ion-H channel
   counts are logged without adding a support loss.

Enumeration budget stays at E4096 and the probability/cross-entropy output stays unchanged so
the run remains comparable with `ef7fl5c6`. `filter_subformula_peaks` stays `False`: the Na/Cl
charge-migration filter fix would change the target spectra at the same time as the CMF
candidate-universe change. The merged
[subformula sibling](../../config/masshatter/merged/subformula_filter/nist20_ma_mi_merged_independent_jss_warmup_e0_15_dense_subformula.yml)
remains the separate filter arm.

This scope is what makes the 2026-09-14 CMF alignment observable. Constructed from the resolved
config, `[M+Na]+` and `[M+Cl]-` carry a live CMF ladder (`cmf_precursor_valid` true,
`cmf_duplicate_crf_h_shift` `None`) while `[M+H]+` and `[M-H]-` stay suppressed at duplicate CRF
shifts `-1` and `+1`. At `max_h_transfer: 4` that is nine previously unenumerated channels per
fragment on the two carrier-switching adducts, which are 38,156 and 7,820 of the 480,002 split
spectra (9.6% combined).

Launch: [submit_masshatter_unmerged_cmf.sh](../../scripts/manual_slurm/vulcan/submit_masshatter_unmerged_cmf.sh),
generation-only, `vulcan:l40s`, job name `msh_unmrg_cmf_dualce`. It answers the wall-time half of
the convergence blocker with a twelve-hour allocation and `j=msh_unmrg_cmf_dualce`, which records
`job_id/msh_unmrg_cmf_dualce.id` so a re-submission resumes from `last.ckpt` in the same W&B run.
Checkpoints are written under `wandb/run-*/files/ckpt` in the cluster checkout rather than
node-local scratch, so a wall-time kill leaves them recoverable; the `.id` file is removed on clean
completion, so a later re-run starts fresh instead of silently resuming.

Validation at `c6a7b8b8` with torch 2.4.1+cu121 / lightning 2.4.0: config resolved through
`load_config`; both arms constructed; the generated job script passed `bash -n` and carries the
intended directives, staging, `wandb`/`job_id` symlinks, and final command. One full subsampled
epoch ran end-to-end on a local RTX 2070 SUPER (400 train / 400 validation spectra, batch 4,
`num_workers: 0`), exit 0, producing `val_cos_sim_0.01_epoch/mean` (so `checkpoint_metric`
resolves) plus `last.ckpt` and `model-epoch=000.ckpt`. Its `val_valid_ion_h_per_fragment_mean`
was 9.246 against 8.54 in the merged `[M+H]+`-only arms, consistent with the added CMF channels;
that is a 400-spectrum untrained-epoch number on a different scope, not a result.

No Vulcan job was generated or submitted from this workstation, which has no `sbatch`. No adduct-
stratified metric exists in the training loop, so the CMF comparison needs a post-hoc evaluation.
Next: run the launcher on the Vulcan login node, submit, and record the scheduler job ID and W&B
run ID here.
