# FraGNNet and MASShatter correctness audit

> **Status:** FINDING · **Updated:** 2026-09-12 · Reproduced defects; fixes pending.

## Scope and baseline

Audited revision `19a33c2869f058f11c5aeffda61da170410553bf` plus the existing
working-tree SMARTS warmup guard removal, regression tests, config comments, and
documentation changes. Those pre-existing edits were preserved. Reviewed FraGNNet
and MASShatter forward/loss contracts, candidate generation, shared metrics, dataset
and sampler handling, checkpoint loading, and inference serialization. This is a
correctness review, not an exhaustive security, chemistry, or performance certification.

Used `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python`. No production source,
configuration, tests, datasets, installed packages, or installed extension binaries
were changed during this audit. No training jobs or W&B queries were launched.
The prior [MASShatter audit](masshatter/masshatter_code_audit.md) records already fixed issues
and the user-deferred CPU bfloat16 mixture limitation; these are not new findings.

## 1. P1: Compiled checkpoints load random FraGNNet weights through inference setup

Location: [checkpoint loader](../../src/fragnnet/utils/script_utils.py), lines 311–332;
interaction with `apply_inference_config`, line 434, and the CLI's setup/load order.

Inference setup changes `config_d['compile']` to false. The loader subsequently checks
that mutated flag to decide whether to remove `._orig_mod` from checkpoint keys.
A compiled checkpoint therefore reaches the uncompiled model with mismatched keys.
After strict loading fails, the loader retries with `strict=False`, ignores missing
and unexpected keys, and returns the model as successfully loaded.

Reproduction: construct a real `FraGNNetPL`, wrap its model with
`torch.compile(..., backend='eager')`, save its actual state and compilation metadata,
then call `get_config_from_ckpt → apply_inference_config → init_model_from_ckpt`.
The returned state is **identical to a fresh model initialized with the loading seed**.
Of 77 parameter tensors, 55 differ from the checkpoint; constant-initialized tensors
explain the remaining equality. No compiled forward or GPU was needed.

Impact: affected CLI and evaluation consumers can write plausible predictions from
random weights. The loader logs a mismatch warning, but does not stop inference.
Detect compilation from checkpoint state/metadata before applying runtime overrides,
and require complete loading for inference rather than silently permitting missing weights.

## 2. P2: HDF5 batch cutoff silently loses buffered predictions

Location: [HDF5 writer](../../src/fragnnet/utils/script_utils.py), line 1240.

`run_inference_hdf5` flushes when its chunk is full or its processed batch count equals
`len(dataloader)`. A positive cutoff shorter than the loader stops the iterator without
meeting the second condition. There is no final flush after the loop.

Reproduction uses the existing HDF5 grouping test's column-major batches of sizes
2, 2, 1, with only the row iterator and column selection stubbed. A one-batch cutoff
and chunk size 10,000 processes two spectra but returns zero written; the file contains
only empty start/end datasets. The no-cutoff control writes all five spectra.
Flush the remaining buffer on iterator exhaustion, including cutoff termination.

## 3. P2: FraGNNet node/formula explanations disagree with its predicted spectrum

Location: [FraGNNet model](../../src/fragnnet/model/fragnnet_model.py), lines 2240–2242;
the resulting values feed the real-state normalization beginning at line 2405.

With `mlp_output_format='node_formula'`, spectral prediction correctly constructs
`log P(f,n) = log P(f|n) + log P(n)`. The separate explainability path instead globally
normalizes raw formula logits plus raw node logits. It omits the node-specific
conditional normalizers, producing a different joint distribution and marginals.

Reproduction uses the small model fixture and the molecule `CCCO`, one isotope per
formula. Reported formula probabilities differ from normalized spectral probabilities
by up to 0.00438. Adding a different constant to every node's formula-logit vector
leaves the spectral prediction exactly unchanged, as conditional softmax requires,
but changes reported formula probabilities by 0.07472. The `formula` head control
agrees to approximately 2.24e-8.

Impact: exported explanations and enabled marginal/joint entropy auxiliaries describe
a different distribution. This probe does not establish an error in the primary
spectral output. The inspected checked-in configs select the `formula` head, so this
concerns the supported alternate head. Derive real-state explanations from the same normalized joint used
for spectral prediction, preserving the intended null-state conditioning.

## 4. P2: FraGNNet edge losses have incompatible shapes and invalid log probabilities

Location: [FraGNNet model](../../src/fragnnet/model/fragnnet_model.py), lines 2357–2360
and 2377–2380; consumed by the edge-loss closure in
[FraGNNetPL](../../src/fragnnet/pl_model/fragnnet_pl.py), lines 707–718.

For `skip_edge_loss=False`, source and destination H-count arrays are both unsqueezed
on the same dimension, giving E×H differences. The accompanying pair probabilities
have E×H² entries. Moreover, `einsum` multiplies log probabilities, whereas independent
joint log probabilities require addition.

Reproduction: `CCCO`, five H channels, graph edges enabled, `h_counts` node features,
`h_range` edge features, and transfer-loss weight 0.1. The actual model returns
differences of shape `(20, 5)` and log probabilities of shape `(20, 25)`; the real
PL loss raises a size-25 versus size-5 `RuntimeError`. A purported log probability
is positive, approximately 7.6959. Construct the Cartesian H differences and combine
log probabilities by addition. Fixing dimensions alone leaves the mathematical error.

## 5. P2: Skipping explanations is accepted with a loss that needs them

Location: [FraGNNetPL](../../src/fragnnet/pl_model/fragnnet_pl.py), lines 509–527.

The compatibility guard for `skip_train_explainability=True` omits `joint_e`,
`joint_ne`, `nb_joint_e`, and `nb_joint_ne`. A nonzero joint entropy weight is accepted
although the training forward returns `pred_joint_logprobs=None`.

Reproduction: loss setup with `joint_entropy_weight=0.1` and explanation skipping
succeeds. Passing an actual training forward on `CCCO` to that loss produces
`TypeError` when the entropy function tries to clamp `None`. Reject the incompatible
combination during setup, or retain the required joint tensors. The ordinary joint
entropy case was reproduced; the neighbor-isomorphism variants share the omitted guard
but were not independently executed.

## 6. P2: Standard auxiliary metrics crash in test/evaluation mode

Location: [SpectrumPL metrics](../../src/fragnnet/pl_model/spectrum_pl.py), lines
573–579 and 966–980.

Training/validation preallocate the per-spectrum metric tensors. Test/predict instead
start with an empty dictionary, but NDCG, `true_oos_prob`, and `true_oos_e` assign to
`metric_d[name][batch_index]` without first creating their tensor.

Reproduction calls the real configured metric closure with observed peaks
`[(100, 0.8), (120, 0.2)]`, predictions `[(100, 0.6), (140, 0.4)]`, and precursor m/z
200. Each metric succeeds for `split='val'` and raises `KeyError` for `split='test'`.
All three metric names are in the template auxiliary-score list. `eval_test_split`
defaults false, so this is not a failure of every training run; it blocks evaluation
when these metrics are retained. MASShatter removes the two OOS metrics during setup,
but retains NDCG when requested. The current MASShatter warmup 0–4 screen overrides the
metric list and is unaffected; most production FraGNNet configs also override it.
Allocate the indexed metric outputs for all splits.

## 7. P2: MASShatter silently ignores precursor-pair auxiliary supervision

**Fixed (2026-09-13):** `MassShatterPL._setup_loss_fn` now rejects nonzero
`precursor_pair_loss_weight` for both probability and independent output, with an
explicit instruction to set it to zero. The precursor-pair objective remains
unsupported by MASShatter. Zero-weight configurations retain their existing behavior;
no model parameters or checkpoint schemas change. Regression coverage checks zero,
0.1, and 10.0 weights with probability cross-entropy, probability cosine distance,
and independent cosine distance. The MASShatter suite plus bond-fragment-head tests
passed **744 tests, with 7 skipped**, on CPU; GPU behavior was not tested.

Original finding:

Location: [MASShatterPL](../../src/fragnnet/pl_model/mass_shatter_pl.py), lines
1031–1039 and its final loss assembly near line 1731.

Probability-output MASShatter accepts a positive `precursor_pair_loss_weight`.
Shared `SpectrumPL` prepares the associated inputs, but MASShatter's loss never
consumes them. Only independent output explicitly rejects this option; the existing
precursor-pair implementation is in FraGNNetPL.

Reproduction builds unmerged probability-output configs with NCE, eligible `[M+H]+`
and FT, and FT relative tolerance 1e-5/minimum m/z 200. Both weight-zero and weight-ten
configs pass `validate_sampler_config` and real PL loss setup. For observed peaks
`[(50, 0.1), (100, 0.9)]`, precursor m/z 100, NCE 10, and predicted logits `[1, 0]`,
both return loss −4.464123249 and gradients `[0.631058574, −0.631058574]`, with no
precursor-pair metrics. The intended helper on the same example returns 1.213261724
with eligibility one, so identical losses are not explained by an ineligible target.

Impact: a valid experiment can claim an auxiliary objective that contributes nothing.
Reject unsupported positive weights or implement the term. No checked-in MASShatter
screen config was found using the option; this is not evidence of corruption in the
recent W&B screen. Probe: `/tmp/masshatter_audit_pair.py`.

## Validation and environment findings

- The unmodified local environment ran all 2,899 collected tests: **2,784 passed,
  92 failed, 21 skipped, 2 expected failures**, in 70.06 seconds.
- All 92 failures came from the local `multi_cut_bfs` extension rejecting source-level
  arguments: 85 for `max_ring_cut_steps`, seven for `ring_open_mask`. The source
  signature includes both; the existing `.so` signature does not.
- Copied that `.pyx` into `/tmp/fragnnet_audit_extension`, built it there with the
  configured interpreter, and loaded it only in the audit process. With the current
  extension source, the full suite yielded **2,876 passed, 21 skipped, 2 expected
  failures**, in 68.41 seconds. The installed `.so` remains unchanged.
- Targeted counterexamples above are outside the existing suite, so a passing suite
  does not resolve them. CUDA was unavailable; no real trained-checkpoint evaluation,
  real-data training, or held-out scientific comparison ran.
- Independent focused runs passed 624 MASShatter tests (seven skipped) and 43 FraGNNet
  tests. The MASShatter review covered candidate assembly, chemistry, proposal/intensity
  routing, SMARTS/warmup, support/pruning, and spectral losses. No additional defect in
  the current screen's active model path was established. Shared data inspection did
  not demonstrate cross-split leakage; final-loader sample identities were not audited.

Temporary reproduction scripts: `/tmp/audit_shared_inference_20260912.py`,
`/tmp/audit_shared_20260912.py`, `/tmp/fragnnet_model_audit_probe.py`, and
`/tmp/fragnnet_pl_audit_probe.py`. Full-suite logs are
`/tmp/fragnnet_audit_2026_09_12_pytest.log` and
`/tmp/fragnnet_audit_2026_09_12_rebuilt_pytest.log`. These are scratch artifacts;
the methods and observed values above are the portable record.

## Next check

Fix checkpoint loading and test-metric initialization first, with regressions through
their actual public call sequences. Correct the optional FraGNNet loss/explanation
contracts and HDF5 final flush, then rerun their targeted counterexamples and the
full suite using extensions built from the matching source. Reject or implement the
ignored MASShatter precursor-pair option. Preserve input schemas
and historical experiment provenance when evaluating corrected explanations or losses.
