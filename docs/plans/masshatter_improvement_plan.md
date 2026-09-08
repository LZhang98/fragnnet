# MASShatter improvement plan

> **Status:** PROPOSED · **Updated:** 2026-09-07
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

The support gate, intensity objectives, encoder-sharing option, and channel diagnostics already
exist. This task records the experiment sequence; no model/config changes or training runs were
made. The checkout was clean at entry, so older project-state references to uncommitted audit
fixes are historical rather than a description of current Git status.

Next: resolve the corrected unmerged anchor and warmup policy, verify checkpoint persistence,
and prepare the matched CE encoder comparison. Run-specific provenance, oracle measurements,
paired-seed variance, and unmerged treatment results remain pending. Validate documentation links
and whitespace for this handoff; model tests are unnecessary for this documentation-only change.
