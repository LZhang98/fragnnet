# Optional FastGNN-style MASShatter proposals

> **Status:** LANDED · **Updated:** 2026-09-17 · Experimental option implemented and locally
> validated; comparative speed and scientific accuracy remain unmeasured.

## Objective and scope

Add a selectable connected-fragment beam proposal inspired by the workspace
[FastGNN prototype](../../fast_gnn.py), retaining MASShatter's encoder options, chemistry,
intensity head, sparse prediction contract, and spectrum-based proposal supervision.
The existing `cut_masks` formulation and CPU/CUDA backends remain the default.
This is an alternative scientific model, not an exact acceleration of mask marginalization.

## Configuration and semantics

[template.yml](../../config/template.yml) declares:

```yaml
mass_shatter_proposal_method: connected_fragment_beam  # default: cut_masks
mass_shatter_unique_fragment_mode: true
mass_shatter_unique_fragment_target: 256
mass_shatter_fast_gnn_beam_width: 256
mass_shatter_fast_gnn_max_boundary_cuts: 3
```

The prepared [dual-CE arm](../../config/masshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml)
retains the existing fast control's encoder, intensity head, chemistry, loss weights,
optimizer and batch settings. It uses selection-only proposal supervision. No run was launched.

- Grow connected atom sets, ranking expansions by summed boundary logits. Keep a bounded
  beam after atom-set deduplication; consider at most four times the beam width before
  deduplication. This approximation can underfill the beam when many paths duplicate.
- Grow to `ceil(atom_count / 2)` per molecule, then add connected complements. Using each
  molecule's own limit avoids dependence on unrelated batch padding. This differs from
  the prototype's batch-maximum growth depth.
- Apply the boundary-cut limit to emitted proper fragments, not intermediate growth states.
  A temporarily oversized boundary can shrink as additional atoms join the fragment.
- Deduplicate final atom sets and enforce a strict final target including one reserved
  intact precursor. Unlike the prototype, mirrors cannot make the count exceed the cap.
  The intact precursor is retained even for disconnected inputs; all proper fragments
  must be connected. Large fragments with disconnected complements may be missed.
- Use multiword 63-bit atom masks, no fixed atom-count ceiling. Dense batch adjacency and
  beam expansion still impose practical memory limits. Discrete search runs in FP32 on
  the model device with autograd disabled; live score computation preserves gradients.
- A fragment score is `sum_e log(1-p_e) + sum_boundary logit(p_e)`. Normalize these scores
  over retained fragments for the existing support-coverage objective. No charge factor
  or latent-pathway marginal is used. The common molecular constant cancels under this
  normalization. Coupled intensity mode can also consume this alternative fragment prior.
- No extra encoder or FastGNN intensity/chemistry modules are introduced. The existing
  optional separate proposal encoder remains usable. Charge-head tensors remain in state
  dictionaries for compatibility, but the head is frozen and skipped for FastGNN proposals.
- Initial support: unique mode, one mixture, SMARTS off, shared-acquisition proposal reuse
  off, zero mask-probability floor. Unsupported combinations raise at construction.
  The fragment-probability floor and ion/H support options retain their normal meaning.
- `candidate_backend`, ranked-mask budget, CPU selection threads and forced-single-cut
  settings belong to `cut_masks`. FastGNN search follows the model device, independently
  of `candidate_backend`. PyTorch CPU and CUDA are supported; compilation remains disabled.
  Runner summaries record the resolved proposal method and `torch_device` backend for this option.
- Legacy templates and checkpoints explicitly migrate missing proposal keys to `cut_masks`.
  Fresh Lightning configurations require the declared keys. Explicit saved/overridden
  values retain precedence. Existing parameter tensor names and shapes are preserved.

## Integration and diagnostics

[Search implementation](../../src/fragnnet/frag/mass_shatter_fast_gnn.py) returns the existing
sparse membership adapter with one identity mask/mechanism row per unique fragment. These
adapter rows are not a latent cut-mask pool: complementary fragments may share a boundary
mask, so summing their exact-mask scores is not retained probability mass.

[Model integration](../../src/fragnnet/model/mass_shatter_model.py) bypasses charge/pathway
marginalization for this option and shares the remaining intensity and chemistry path.
The [Lightning wrapper](../../src/fragnnet/pl_model/mass_shatter_pl.py) omits meaningless
mask/pathway diagnostics and logs `fast_gnn_fragment_count`, `fast_gnn_mean_boundary_cut_count`,
`fast_gnn_boundary_cut_fraction_*`, and `fast_gnn_search_batch_ms` instead. Existing private
model-to-loss adapter names remain for interface compatibility. Timers measure host wall time,
not synchronized CUDA stage durations; use synchronized whole-step benchmarks for speed claims.

## Validation and next check

Acceptance checks: exhaustive small-graph membership and boundary-score references; normalized
score gradients; rings, disconnected and zero-bond inputs, multiword masks, budgets and batch
indexing; CPU/CUDA parity; real-model output normalization and proposal gradients; CE/JSS
loss and warmup routing; config rejection; strict new/legacy checkpoint loading; regression suites.

Final local validation:

- Full CPU-visible suite: **3,451 passed, 96 skipped, 2 xfailed**, 90.38 seconds.
  Log: `/tmp/masshatter_fast_gnn_full_suite.log` (local artifact).
- New-option suite: **29 passed, 3 skipped** inside the sandbox. Separate GPU execution:
  **3 passed**, covering candidate/gradient parity and real-model CE/JSS backward through
  the staged-training transition. Counts overlap the full suite; do not sum them.
- Two existing runner/resume logging regressions passed after adding resolved-proposal logging.
- Ruff and Git whitespace checks passed; documentation relative links checked.
- Real-data GPU smoke completed on RTX 2070 SUPER, PyTorch 2.4.1+cu121, seed 397623,
  using the prepared config, subsample 128, batch 16, one fixed batch, one warmup and two
  measured steps per monitoring variant. No optimizer updates or W&B logging. Finite losses;
  peak allocated GPU memory below 445 MB. This short, initially concurrent smoke is not
  comparative throughput evidence. Exact input IDs, resolved config and measurements are in
  `/tmp/masshatter_fast_gnn_smoke.json`; log: `/tmp/masshatter_fast_gnn_smoke.log`.

Reproduce the smoke from the repository root with the configured interpreter:

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/benchmark_mass_shatter_training.py \
  --config config/masshatter/ma_mi_unmerged/cmf/nist20_ma_mi_unmerged_cmf_dual_ce_fast_gnn_proposal.yml \
  --device cuda --subsample-size 128 --batch-size 16 --batches 1 --warmup 1 --steps 2 \
  --output /tmp/masshatter_fast_gnn_smoke.json
```

The CE integration checks use the vectorized path selected by the prepared arm. A preliminary
non-vectorized Gaussian-CE harness hit the existing `LOG_ZERO(torch.float64)` limitation with
model mass tensors; that shared loss path was not changed in this task.

No optimization/training run or validation-set quality evaluation was performed. Next scientific
check: compare the prepared arm to the fast control at matched candidate count and runtime,
reporting coverage, spectral accuracy and peak memory. The retained intensity head remains a
possible runtime bottleneck; this option does not claim full FastGNN-model speed.

## Proposal method naming (2026-09-17)

The canonical selector is now `mass_shatter_proposal_method`, with
`cut_masks` (default) and `connected_fragment_beam`. Config files, CLI overrides,
saved-run configuration and Lightning/repository checkpoint loading convert the original
`mass_shatter_proposal_type` key and its `cut_masks`/`fast_gnn` values. If both keys occur
in the same mapping, the canonical key wins; normal template/custom/CLI precedence is
preserved. Saved hyperparameters use the canonical name. Beam settings, metric names and
parameter tensors are unchanged.

Validation: 355 focused tests passed, 8 GPU tests skipped in the sandbox. These cover
old-name YAML/CLI loading, strict old/new checkpoint roundtrips and canonical override
precedence, plus model, loss, backend compatibility and runner-config regressions.
Ruff and whitespace checks passed. No training or scientific behavior changed.
