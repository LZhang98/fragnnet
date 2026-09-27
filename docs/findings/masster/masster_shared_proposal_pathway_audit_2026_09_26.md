# MASSter shared fragment proposal pathway audit

> **Status:** FINDING · **Updated:** 2026-09-26 (05:40 UTC on 09-27)
> Audit of `masster_share_proposal_across_ev` and the CE-context batch sampler it requires, made
> while arms `sp0`/`sp1` ran. **The sampler, not the sharing, was broken.** Every epoch ended with
> 28 one-spectrum batches of one molecule. These left the BatchNorm running statistics used at
> validation almost entirely from that molecule, and on `c0`'s checkpoint that alone costs −0.25
> val cosine. `sp0` and `sp1` collapsed (val cosine 0.03-0.24 against `c0`'s 0.58-0.66) and
> are marked crashed at epoch 2. The shared proposal path is correct. Its merged-target loss is
> exactly the plain per-spectrum loss, so sharing does not need this sampler.
> **Both bugs are fixed in the working tree (uncommitted; [Fixes](#fixes-2026-09-26)):** the
> sampler no longer emits the tail, and the shared ion-mode proposal head no longer reads CE.

## Question

Does `sp1` (= `sp0` + `masster_share_proposal_across_ev`) test what its header says: one CE-blind
candidate set per molecule, trained on the molecule's merged target, and nothing else? Does
`sp0` (= `c0` + `masster_batch_sampler`) price only "whole molecules per batch"?

## 1. The CE-context sampler's epoch tail poisons BatchNorm

`MassterCEContextDynamicBatchSampler` (and FraGNNet's `CEContextDynamicBatchSampler`, same
base class) packs each molecule into atoms. The first atom holds one spectrum per distinct CE,
up to `ce_context_spectra_per_context`. Every same-CE replicate becomes a one-spectrum atom
([`_context_atoms`](../../../src/fragnnet/dataset/data_sampler.py#L545)). Before the fix,
[`_pre_compute_batches`](../../../src/fragnnet/dataset/data_sampler.py#L600) (as of `b277b714`)
interleaved atoms round-robin by molecule and closed a batch as soon as the next atom's molecule
was already in it. The molecule with the most atoms was left alone at the end, one spectrum per
batch.

Measured on the `sp0`/`sp1` train split (`nist20_inchikey`, `[M+H]+`/FT, 157,678 spectra, seed
397623, script `sampler_profile.py`, `sampler_tail.py`):

- 376 batches per epoch against ~312 if all were full; median 505 spectra and 42 molecules.
- **The last 58 batches hold fewer than 64 spectra, and the last 28 hold one spectrum each, all of
  mol 32341** (inositol tetrakisphosphate, 28 heavy atoms, 86 train spectra, 6 replicates at
  several NCEs). Identical at epochs 0, 1, 5 and 30: rotation moves primaries, not the tail.
- Only 3.0% of spectra are same-CE replicates, so "whole molecules per batch" holds for 97%.
- Val has the same tail (130 batches, last 10 single-spectrum), harmless in eval mode.

The only BatchNorm layers are the 4 node-level `BatchNorm1d` in `precursor_encoder`
(`mol_normalization: batch`, momentum 0.1). After 28 updates from one molecule, 1 − 0.9²⁸ = 95%
of the running statistics validation uses come from mol 32341. Each epoch also ends with 28 full
Adam steps (lr 1e-3) on that one molecule.

**Probe on `c0`'s checkpoint** (`saved_ckpts/c0_control_vcpho5qm.ckpt`, 4,919 val spectra from 400
seeded molecules, weights never updated, only BatchNorm buffers change; `bn_tail_probe.py`):

| BatchNorm state | val cos_sim_0.01 |
| --- | ---: |
| checkpoint running statistics | 0.7236 |
| after 28 train-mode no-grad passes, one mol 32341 spectrum each | **0.4771** (−0.2465) |
| control: after 28 passes on random 128-spectrum train batches | 0.7233 (−0.0003) |

**W&B.** `sp0` (`85mfkk8i`) and `sp1` (`bu6dzjph`) logged three epochs, then both went `crashed`
at 03:55 UTC, 3 s apart on different nodes (fc11014, fc11005). No log was uploaded, so the cause
of the crash is unknown.

| epoch | `c0` val cos | `sp0` | `sp1` | val loss `c0` / `sp0` / `sp1` | train loss `c0` / `sp0` / `sp1` |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 0.583 | 0.094 | 0.116 | −2.11 / +0.97 / +1.65 | −1.54 / −1.17 / −1.11 |
| 1 | 0.629 | 0.236 | 0.059 | −2.35 / +0.90 / +1.04 | −2.40 / −1.54 / −1.35 |
| 2 | 0.661 | 0.125 | 0.026 | −2.48 / +1.54 / +2.84 | −2.64 / −1.69 / −1.60 |

- Train loss (train-mode BatchNorm) is worse than `c0`'s but still falls. Val loss is 3-4 nats
  above train, against 0.16 for `c0`. The damage is mostly at evaluation, as the probe predicts.
  Early checkpoints may be more sensitive than the trained `c0`. The 28 single-molecule gradient
  steps were not simulated.
- `sp0`'s val `candidate_wrecall` is 0.83-0.89 against `c0`'s 0.93, because the proposal reads the
  same encoder.
- **Neither run can be read**, and no config on this sampler should be submitted before it is
  fixed: `sp0`, `sp1`, and FraGNNet CE-pair Stage A `c1_sampler_ctrl`, `p1`, `p2`.
  - Stage A's split (`nist20_ma_mi_inchikey`, 5 spectra per context, batch 256) has the same
    tail: 52 batches under 32 spectra, the last 15 single-spectrum, mostly group 15553
    (`sampler_tail_mami.py`).
  - Its `mol_normalization` and `frag_normalization` are both `batch`.
- The three August unmerged runs on this sampler that "crashed by epoch 5" may share the cause.
  This was not checked.

## 2. The shared proposal path is correct

- **The merged-target loss is exactly the per-spectrum loss.** `_proposal_coverage_loss` is
  `−Σ_t w_t log(covered mass at t)`, linear in the target weights, and targets are L1-normalized per
  spectrum. The group target with unit mass per spectrum therefore equals the mean of its members'
  losses, and broadcasting it back and averaging over the batch gives the plain batch mean.
  - On 200 random batches (1-4 groups of 1-5 spectra, non-adjacent members, uncovered and
    duplicate peaks), the loss and gradient differ by at most 2.2e-16 (`equiv_check.py`).
  - The existing test's 0.55/0.45 target is this mean.
  - This does not hold under `masster_proposal_barcode_target`, where merging changes peak
    weights; it is off in `sp1`.
  - Consequences:
    - What sharing changes is only that the proposal is CE-blind. The grouping changes compute,
      not the objective.
    - The runner's rule that sharing requires `masster_batch_sampler`
      ([`runner.py:600`](../../../src/fragnnet/runner.py#L600)) is not needed for correctness.
    - The group loss's two Python loops over groups
      ([`masster_pl.py:622`](../../../src/fragnnet/pl_model/masster_pl.py#L622),
      [`:649`](../../../src/fragnnet/pl_model/masster_pl.py#L649)) add host syncs per group
      (~42 per batch) for no change in value. They bias read (3), seconds per epoch.
    - The logged per-spectrum `proposal_loss` in `sp1` is its group's mean. Do not break it down
      by NCE.
- **Candidates are identical across a group by construction.** The CUDA builder checks that grouped
  logits are bit-equal, ranks masks once per group and expands them to every member. The loss's
  compaction to the group's first spectrum is therefore exact.
- **The proposal context has no CE.** `include_collision_energy=False` zero-fills the CE slot
  before the shared projector. With one adduct and one instrument it is a constant, detached vector.
  The one head that read the CE-conditioned context, `ion_mode_proposal_head`, is off in every
  sharing config ([head-input asymmetry §4.7](masster_head_input_asymmetry_2026_09_22.md)); it
  now reads the shared context too ([Fixes](#fixes-2026-09-26)).
- **Groups are molecules.** In `nist20_inchikey`, every `group_id` is one molecule with one adduct
  and one instrument: 12,668 train groups, median 13 spectra. Evaluation does not depend on how
  groups fall into batches: BatchNorm uses running statistics and dropout is off.

## 3. `sp1` − `sp0` is not only the proposal

The shared branch runs `precursor_encoder` once per group and broadcasts it to every spectrum
([`masster_model.py:2570`](../../../src/fragnnet/model/masster_model.py#L2570)), and the
intensity path reads that encoder too. Two things change in the intensity path, both unmeasured:

- BatchNorm batch statistics weight each molecule once instead of by its spectrum count.
- Encoder dropout (`mol_dropout: 0.1`) draws one mask per molecule, shared by all its energies in
  the step, instead of one per spectrum.

## Fixes (2026-09-26)

Uncommitted, in the working tree. Checks run: a new regression test for each fix, which fails on
the old code and passes on the new; 625 tests across the sampler, MASSter model/PL, runner-config,
candidate and memorization-route files; ruff and black on the changed files.

**Sampler** ([`_pre_compute_batches`](../../../src/fragnnet/dataset/data_sampler.py#L600), shared
by the MASSter and FraGNNet CE-context samplers).

- Molecules' chunk sequences are merged in a uniformly random order, each keeping its own chunk
  order.
- A chunk whose molecule is already in the open batch waits, and is offered first to the next
  batch. It no longer closes the batch.
- What stays the same: at most one chunk per molecule per batch (the CE-pair loss rebuilds
  contexts from `group_id`), every spectrum once per epoch, chunk rotation, seeding.
- All existing CE-context tests pass unchanged. The new
  `test_spreads_a_many_chunk_context_over_the_epoch` gave 8 batches under 4 spectra on the old code.

After the fix (`sampler_after_fix.py`, epochs 0, 1, 5, 30):

| split | batches, old → new | small batches, old → new | smallest batch now |
| --- | ---: | ---: | --- |
| `sp0`/`sp1` train | 376 → 312 | 58 → 0 under 64 | 220-254, the final partial batch |
| `sp0`/`sp1` val | 130 → 103-104 | 10 single → at most 1 | 1, at 2 of 4 epochs (eval mode) |
| FraGNNet Stage A train | 1,188 → 1,131 | 52 → 0-1 under 32 | 88-97; one epoch ended at 4 |

- A lone small final batch is harmless. On `c0`'s val subset, one train-mode pass on a single
  mol 32341 spectrum costs 0.0003; one random 4-spectrum batch costs 0.0001
  (`bn_single_batch_probe.py`). The tail was 28 such passes in a row.
- `sp0`/`sp1` now take 312 steps per epoch, within 1.3% of `c0`'s 308.
- Stage A still sees every spectrum each epoch, so its 48-epoch spectra match stands. Its step
  counts (709 steps of ~406 spectra) are pre-fix, and batches are now fuller. Re-measure them with
  the real fragment-edge budget before submitting.

**Ion-mode proposal head under sharing**
([`masster_model.py:2686`](../../../src/fragnnet/model/masster_model.py#L2686)).

- It read each spectrum's CE-conditioned context. The flag's contract is to run the proposal heads
  once per group without collision-energy input, and this head's logits choose which ion modes
  each spectrum emits.
- It now reads the shared CE-free context, like the bond, charge, mixture and mechanism heads.
- Dormant: no config enables it with sharing, so no run changes.
- New `test_shared_proposal_ion_mode_head_is_collision_energy_blind`: the two energies' logits
  differed by 0.007-0.03 before the fix, and agree to 1.5e-8 after (matmul rounding).

**Not changed.**

- The runner guard that sharing requires the sampler. With the sampler fixed, `sp0`/`sp1` test
  what they were designed to test.
- The shared loss's per-group loops. They are an efficiency cost, and value and gradient are exact.
  The logged per-spectrum `proposal_loss` stays the group mean.
- Section 3's confound, which is a property of the design.

## What is left

1. Commit the fixes, then resubmit `sp0`/`sp1` under new run ids; their configs need no change.
   The first submissions (`85mfkk8i`, `bu6dzjph`) are void.
2. Stage A: re-measure steps per epoch on the fixed sampler before submitting `c1`/`p1`/`p2`.
3. Optional: `c0` + `masster_share_proposal_across_ev` on `c0`'s random batches measures a CE-blind
   proposal against `c0` directly, since the loss does not depend on grouping. The runner guard
   would have to be relaxed. Groups would be almost all singletons: about 0.04 same-molecule
   partners per spectrum in a random 512 batch.
4. Optional, for `sp1`'s timing read: route the shared loss to `_proposal_coverage_loss` when the
   barcode target is off.

## Limits

- The probe simulates only the BatchNorm half of the tail, on a trained checkpoint. The effect of
  the 28 gradient steps, and the size of the cost early in training, are not measured.
- The sampler profile replaces molecular edge counts with 2 × bonds. The edge budget
  (100,000) never binds at 512 spectra, so batch membership is unaffected.
- The crash cause of `sp0`/`sp1` is unknown.
- The fix is checked on the sampler's batches and by the BatchNorm probes, not by a training run.
  The resubmitted `sp0` is that check: it should track `c0` from epoch 0.

Scripts and log: `output/shared_proposal_audit_2026_09_26/` (local, git-ignored).
