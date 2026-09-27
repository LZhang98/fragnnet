# Which intensity-head input carries MASSter's memorization?

> **Status:** FINDING · **Updated:** 2026-09-24
> Inference-only corruption probe on two production checkpoints, no training. The train-only
> part of the intensity allocation sits in the **boundary block**, the pooled embedding of the
> severed bonds' endpoints. Corrupting it removes about 89% of the 0.19 eval-mode train/validation
> cosine gap while validation keeps about 0.39. Corruptions of the fragment/complement pools
> that cost validation as much leave 0.06-0.09 of the gap. The candidate descriptors behave as a
> generalizing feature. The candidate-set fingerprint carries under 11% of the gap. Both
> checkpoints give the same ranking, and their baseline and boundary gaps agree to 0.002.

## Scope and method

The question comes from the
[overfit read-out §7](masster_overfit_readout_2026_09_22.md#7-intensity-memorization-threat-model):
about 98% of the train/validation gap is intensity allocation, and five routes could carry it.
This measurement ranks the routes inside the intensity head. The toy-scale version is
[head-input asymmetry §7.1c](masster_head_input_asymmetry_2026_09_22.md#71c-the-probe-exists-and-its-toy-scale-validation-already-points-at-the-boundary-block).

|                     | f1                                                                         | f0                                |
| ------------------- | -------------------------------------------------------------------------- | --------------------------------- |
| run                 | `6h67zps9` `f1_mh_ft_all_filters`                                      | `3bej6fq7` `f0_mh_ft_control` |
| checkpoint          | `model-epoch=062` (best)                                                 | `model-epoch=061` (best)        |
| W&B best val cosine | 0.7372                                                                     | 0.7318                            |
| targets             | all filters on (`filter_subformula_peaks`, above-precursor, centroiding) | no filters                        |

Common to both: MASSter, unmerged, `[M+H]+`, FT, `data/split/nist20_inchikey`, 64 epochs,
seed 397623, and the same intensity head as `c4a_base`. That head uses induced attention, 256
unique fragments, `boundary_bond_encoder: true`, `proposal_anchored` and
`proposal_selection_only`, and the `c_cut_count`/`c_size`/`c_mass`/`c_charge` descriptors
(5 columns). The checkpoints came from W&B into `saved_ckpts/`. Each local config resolves to
its run's W&B config except three keys added later, all at inert defaults:
`masster_fragment_selection: mask_order`, `masster_selected_fragment_target: null` and
`spec_params.split_filters: {}`.

Probe: `scripts/debug/debug_pooled_embedding_memorization.py` at `a75eb35f`, which adds the
`features` and `set` scopes to the original `cc`, `boundary` and `all` scopes:

```bash
python scripts/debug/debug_pooled_embedding_memorization.py \
  --model_ckpt saved_ckpts/f1_mh_ft_all_filters_6h67zps9.ckpt \
  --custom_fp config/masster/mh_ft_clean_2026_09_22/f1_mh_ft_all_filters.yml \
  --eval_batch_size 512 --batch_cutoff 100 --output_fp f1_probe.csv
```

- **Model state:** both splits run in eval mode, with train auxiliary metrics forced on. This is
  the eval-mode train measurement the read-out's §7.1 gate asked for. It is group-weighted, not
  molecule-balanced.
- **Samples:** 100 batches of up to 512 spectra per split, about 51k spectra each.
  - Validation is 100 of its 102 batches, read in order (`SequentialSampler`).
  - Train is 100 of 308 batches from a seeded `RandomSampler` over spectra.
  - *Corrected 2026-09-27:* the CE-context batch sampler is off for these checkpoints (standard
    batching, batch 512). This finding originally said both splits used it.
    [Audit](masster_memorization_attempts_audit_2026_09_27.md).
- **Pairing:** every condition runs on the same batches, so its deltas are paired.
- **Hardware:** local RTX 2070 SUPER, about 45 min per checkpoint, peak 7.75 GB.

## Result

Cosine is `cos_sim_0.01`. "Kept above floor" is the share of a split's cosine above the
`all_zero` floor that survives the corruption, which is 0.135/0.133 (train/validation) for f1
and 0.102/0.100 for f0.

| condition                         | f1 train | f1 val |          f1 gap | f0 train | f0 val |          f0 gap | kept above floor, train / val (f1; f0) |
| --------------------------------- | -------: | -----: | --------------: | -------: | -----: | --------------: | -------------------------------------- |
| none                              |    0.928 |  0.739 | **0.189** |    0.923 |  0.733 | **0.190** | 1.00 / 1.00                            |
| `boundary_zero`                 |    0.412 |  0.390 | **0.022** |    0.397 |  0.375 | **0.022** | 0.35 / 0.42; 0.36 / 0.43               |
| `boundary_molmean`              |    0.412 |  0.392 | **0.021** |    0.408 |  0.386 | **0.022** | 0.35 / 0.43; 0.37 / 0.45               |
| `boundary_shuffle`              |    0.227 |  0.215 |           0.012 |    0.236 |  0.221 |           0.015 | 0.12 / 0.14; 0.16 / 0.19               |
| `cc_zero`                       |    0.470 |  0.410 |           0.059 |    0.469 |  0.407 |           0.061 | 0.42 / 0.46; 0.45 / 0.49               |
| `cc_shuffle`                    |    0.451 |  0.365 |           0.086 |    0.430 |  0.349 |           0.082 | 0.40 / 0.38; 0.40 / 0.39               |
| `cc_molmean`                    |    0.555 |  0.444 |           0.112 |    0.531 |  0.420 |           0.111 | 0.53 / 0.51; 0.52 / 0.50               |
| `features_molmean`              |    0.704 |  0.571 |           0.133 |    0.706 |  0.571 |           0.135 | 0.72 / 0.72; 0.74 / 0.74               |
| `features_shuffle`              |    0.564 |  0.459 |           0.105 |    0.564 |  0.464 |           0.100 | 0.54 / 0.54; 0.56 / 0.57               |
| `features_zero`                 |    0.059 |  0.055 |           0.003 |    0.073 |  0.067 |           0.006 | below floor                            |
| `set_drop` (50% of keys hidden) |    0.894 |  0.722 |           0.172 |    0.885 |  0.715 |           0.170 | 0.96 / 0.97; 0.95 / 0.97               |
| `all_zero`                      |    0.135 |  0.133 |           0.003 |    0.102 |  0.100 |           0.002 | floor                                  |

**The eval-mode gap is 0.19 on both checkpoints, and it is not coverage.** `wrecall` is
0.9715/0.9643 (f1) and 0.9471/0.9405 (f0), a 0.007 difference. It is identical under every
condition, because the corruption sits downstream of candidate generation and cannot change the
support. This task and sampler differ from the read-out's merged train-mode 0.151, so the two
gap sizes are not comparable.

### Why raw `gap_closed` is the wrong read

Every block corruption costs both splits heavily. Once both sit near a common floor, the gap
closes for no interesting reason: `all_zero` and `features_zero` "close" 97-99% of it. Two reads
survive that artifact, and both give the same answer.

1. **Matched validation damage.** Compare conditions that cost validation the same amount and
   ask how much of the train advantage survives. At a validation drop of 0.33-0.38:

   - the boundary corruptions leave a **0.021-0.022** gap on both checkpoints;
   - `cc_zero` (0.33 drop) leaves 0.059-0.061;
   - `cc_shuffle` (0.37-0.38 drop) leaves 0.082-0.086.

   Along the cc and features conditions, the gap shrinks smoothly with validation level: 0.11 at
   0.44, 0.13 at 0.57, 0.19 at 0.74. The boundary points fall well below that curve.
2. **Share kept above the floor.** Under `zero` and `molmean`, removing the boundary block
   leaves train with 35-37% of its above-floor cosine and validation with 42-45%, an asymmetry of
   7-8 points on both checkpoints. Under `shuffle` both splits fall close to the floor (12-19%
   kept), so that mode adds little. The descriptors are symmetric to 0.01 in every mode. The cc
   pools are symmetric under `molmean` and `shuffle` (within 0.02), and only `zero`, the
   off-manifold substitute, shows a 4-point asymmetry.

The primary loss agrees. Boundary corruptions close 74-78% (`zero`, `molmean`) and 88-90%
(`shuffle`) of the train/validation loss gap. The cc corruptions close 29-43%, and the
`molmean`/`shuffle` descriptor corruptions close 7-25%. `set_drop` closes 9-10% and costs train
loss 3.0-3.4x what it costs validation, so set-context fingerprinting is real but small.

## Reading

| route (from the read-out's threat model)                               | verdict on these checkpoints                            |
| ---------------------------------------------------------------------- | ------------------------------------------------------- |
| per-candidate identity in the **boundary block**                | **carries most of the train-only allocation**     |
| per-candidate identity in the fragment/complement (**cc**) pools | used heavily by both splits in proportion; generalizing |
| candidate**descriptors** (cut count, size, mass, charge)         | used heavily by both splits in proportion; generalizing |
| **candidate-set fingerprint** through the attention context      | real, under 11% of the gap                              |

This inverts the premise of the prepared `candidate_atom_dropout` arms. `e2`/`e3` perturb the cc
pools and deliberately leave the boundary embedding exact. They were screened null on the merged
base ([cut-set plan](../../plans/masster/masster_cut_set_features_plan.md)). On this
evidence, that null is what regularizing the generalizing columns while leaving the memorizing
one intact should produce.

`c4a_no_bbe` (`pkavjg8z`, merged, 128 epochs) is also null: tail mean 0.7246 against the 0.7270
control, inside the ±0.005 gate. It does not test this result. `boundary_bond_encoder: false`
changes how the boundary block is built but still feeds a boundary block to the head, and that
run logged no train cosine.

**Which module this is (checked 2026-09-24).** The probe corrupts the intensity head's
positional argument `boundary_embeddings`. With `boundary_bond_encoder: true`, which is on in
both checkpoints and in the toy check, that tensor is `CutBondEncoder([in, out, in*out])` applied
to each (fragment, cut bond) pair and mean-pooled per fragment
(`MassterModel._boundary_embeddings`). The same class, in a different instance, is
FraGNNet's `cut_bond` feature. The FraGNNet
[bond-set plan §7.2](../../plans/fragnnet/fragment_bond_set_representation.md#72-generalization-gap-by-representation)
reached the matching verdict offline in July: the cut-bond encoder arm had the largest
train→val gap of any arm, and raw static cut-bond features had a 3.5x smaller one. Commit
`0f99e7e4` now encodes each directed pair once instead of once per row (9.8x fewer rows on a
merged batch). That computes the same function, so it does not change this result.

## Limits

- **Different task from `c4a_base`.** Both checkpoints are unmerged `[M+H]+` FT. The ranking is
  evidence for the merged MA/MI lineage, not a measurement of it.
- **Not a clean seed replicate.** The two checkpoints share the seed and architecture and differ
  only in target filtering. The agreement is across target definitions.
- **Unequal train and validation samples.** Train is a random third of its split, and
  validation is 98% of its split, read in `spec_id` order.
  - *Corrected 2026-09-27:* no size-based skipping applies. `skip_too_big` belongs to the
    dynamic and CE-context samplers, which these checkpoints do not use.
  - The sample difference shifts the absolute gap, not the paired per-route contrasts.
- **Head input versus network part.** Corrupting at the head's input cannot say whether the
  identity is written by the GNN encoder or read by the head's own weights. For choosing a
  regularizer, the block is the actionable level.
- **Substitutes are off-manifold.** `zero` and `shuffle` inputs are unlike anything the head saw
  in training. The boundary result holds across all three modes, including `molmean`, the most
  conservative one.

## Next

The actionable arm is to regularize the boundary block itself, on a matched 128-epoch merged
screen with `compute_train_auxiliary_metrics: true`. Two options:

- extend `candidate_atom_dropout` so the dropped membership also reaches the boundary pool;
- add candidate-level dropout of the whole boundary block.

Read it on the gap *and* on validation cosine: three earlier gap-narrowing arms lost on
validation (`drop03`, `b3_wd_groups`, `e2_cc_atom_drop01`).
