# Reducing MASSter's per-molecule shortcut so its ranking transfers

> **Status:** ACTIVE · **Updated:** 2026-09-27
> **Switch-EMA arm `u3` finished and fails** ([section](#switch-ema-arm-u3-built-2026-09-25-not-submitted)):
> −0.0089 against `u2`, below it at all 64 epochs; it underfits. EMA helps only as an average
> kept off the training weights.
> **All four arms ran; W&B tail read done ([below](#tail-read-from-wb-2026-09-24)).** A weight EMA
> passes gate 1 on both lineages: `u2` +0.0094 over `3bej6fq7`, `t2` +0.0097 over `c4a_base_s1`,
> above the control at every epoch in both. `h1` (allocator dropout 0.2) is rejected at −0.0067,
> and `h2` (width 128) is null at −0.0015. Gates 2-3 (offline ranking decomposition and novelty
> bins) are not read yet. `t2`'s train-metrics control `c4a_base_s1_trainaux` was never
> submitted. The boundary-block dropout arms (`t1`, `u1`) were **dropped on 2026-09-24**: molecule
> identity reaches every head input, so blocking one input can reroute the shortcut. Their code
> stays, off by default. Jobs: `submit_masster_mh_ft_ranking_transfer.sh` (unmerged) and
> `submit_masster_regularization.sh transfer` (merged).

## Objective and acceptance

The [ranking decomposition](../../findings/masster/masster_ranking_decomposition_2026_09_23.md)
found that half of the validation cosine gap is candidate order, and that the order is nearly right
on training molecules: ranking gap 0.011 on train against 0.121 on val, base-peak hit 84.5% against
56.7%, train `cos_sim_0.01` 0.929 against 0.737. The
[shortcut audit](../../findings/masster/masster_overfit_readout_2026_09_22.md#7-intensity-memorization-threat-model)
(§7 and its 2026-09-23/24 updates) narrows the measured shortcut on this lineage to a
**per-molecule allocation lookup**:

| audit route | status on unmerged `[M+H]+`/FT |
| --- | --- |
| candidate-set fingerprint | under 11% of the gap ([memorization probe](../../findings/masster/masster_memorization_route_probe_2026_09_23.md)) |
| unequal exposure by molecule | not a driver: train cosine flat across 4 to 24+ spectra per molecule, Spearman 0.058 |
| QTOF NCE proxy | absent (FT only) |
| precursor/population prior | a fallback on unseen molecules (NCE ≤ 20 hit when the base peak is not the precursor: 0.90 train, 0.42 val) |
| cut-count and size proxies | behave as generalizing features (probe) |
| per-molecule allocation lookup | the shortcut. The boundary block carries most of it at the head input, but identity reaches every pooled input through the shared atom embeddings |

The interventions therefore act on the lookup's capacity or on optimizer noise, not on one input.
The audit's rule is to remove the opportunity to memorize allocation, not the chemistry.

An arm passes when, against its control at matched epoch:

1. the validation tail mean rises. Unmerged: `val_cos_sim_0.01_epoch/mean` over epochs 54-63
   against 0.7298. Merged: epochs 118-127 against the control, with a standing gate of ±0.005 and a
   tail-mean floor of 0.0004
   ([epoch-128 read-out §3.1](../../findings/masster/masster_architecture_arms_epoch128_2026_09_22.md));
2. the offline ranking decomposition shows the shortcut shrinking: a **lower val** ranking gap,
   usually with a **higher train** one;
3. validation cosine by nearest-train Tanimoto bin
   ([variance-novelty finding](../../findings/masster/masster_variance_novelty_2026_09_24.md))
   improves most in the least-similar bins.

An arm that only lowers train fit is underfitting, not a pass. `drop03`, `b3_wd_groups` and
`e2_cc_atom_drop01` all narrowed the gap and lost. Only `drop03` logged train cosine; `b3` and
`e2` narrowed only the train-mode loss gap, and `b3`'s change is inside that statistic's floor
(corrected 2026-09-27,
[audit](../../findings/masster/masster_memorization_attempts_audit_2026_09_27.md)).

## Arms

### Unmerged `[M+H]+`/FT (primary)

These live in `config/masster/mh_ft_ranking_transfer_2026_09_23/`. Each is
`f0_mh_ft_control.yml` plus one key and W&B naming, verified through `load_config`. The raw
target is used because cleaning is a
[closed null](../../findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md).
Train metrics stay off, as in the control, and the train side is read offline.

| arm | key | control → arm | what it limits |
| --- | --- | --- | --- |
| `h1_head_dropout02` | `masster_intensity_transformer_dropout` | 0.1 → 0.2 | Every route through the intensity allocator (§7.2 item 3). The encoder and proposal heads are unchanged. |
| `h2_head_dim128` | `masster_intensity_transformer_dim` | 256 → 128 | The allocator's capacity: head 2.46M → 0.97M parameters, model 9.21M → 7.72M. The input projection still sees the full token. |
| `u2_ema999` | `ema_decay` | null → 0.999 | Optimizer noise. At 308 steps/epoch the mean age is ~1,000 steps, about 3.2 of 64 epochs. |

Control `3bej6fq7`: val `cos_sim_0.01` tail mean over epochs 54-63 is **0.7298**, best 0.7318, and
the mean change between consecutive epochs is 0.0013. `6h67zps9` matches it on the raw target to
0.0001 (val) and 0.0007 (test), which bounds the run-to-run noise on the mean from above. No seed
replicate exists for this lineage. The arms run on code after `0f99e7e4` (CutBondEncoder
deduplication, equal up to float rounding). An exact same-code control would be a rerun of
`submit_masster_mh_ft_clean.sh f0`: `3bej6fq7` itself predates `0f99e7e4`. Profile `fir:h100-3g.40gb`, 12 h box, about 4-5 h of
training. `upload_checkpoints` is true.

### Merged MA/MI

`config/masster/ma_mi_merged/ranking_transfer_2026_09_23/t2_ema998.yml` is
`c4a_base_s1_trainaux` plus `ema_decay: 0.998`. The lineage runs 90 steps/epoch (`5qws8g9x`), so
the mean age is ~500 steps, about 5.5 epochs. The control is the same job as the `wd10` stage's, so
submit it once. Profile `fir:h100-80gb` for the train metrics. **Caution:** that lineage has never
uploaded checkpoints (`c4a_base_s1` has none on W&B), so the offline decomposition needs the
checkpoints copied off Fir, or `upload_checkpoints: true` added to both configs before submission.

### Dropped: boundary-block dropout (`t1`, `u1`)

`masster_candidate_boundary_dropout` drops each candidate's whole boundary block from the
intensity token. It was aimed at the probe's largest route. It was dropped before submission
because the probe measures where the trained model keeps the lookup, not where the lookup would go
if that block were blocked. The fragment and complement pools, the descriptors and the attended
candidate set all derive from the same atom embeddings. The `e2` and `c4a_no_bbe` nulls are
consistent with rerouting. The key and its tests remain, default 0.0. If ever run, read it with the
memorization probe re-run on its checkpoint, to see where the lookup moved.

## Implementation (working tree, uncommitted)

- **EMA.** `fragnnet.utils.pl_utils.ExponentialMovingAverage`, a Lightning callback that `runner.py`
  adds when `ema_decay` is not null (template default null). It keeps a per-step average of all
  floating-point state, parameters and BatchNorm statistics included, with the warm-up
  `d_t = min(decay, (1+t)/(10+t))`. Validation, `ModelCheckpoint`'s best-checkpoint choice, the
  post-fit `trainer.test(ckpt_path="best")` and every saved `state_dict` use the average, so
  inference loads it unchanged. The training weights are saved as
  `checkpoint["ema_training_state_dict"]` and restored on resume. Lightning 2.4's
  `StochasticWeightAveraging` was not used: it swaps the average in only at the end of training
  and imposes its own SWA learning rate.
- **Boundary-block dropout** (unused by any arm).
  `MassterModel._candidate_boundary_dropout` applies block-level inverted dropout at the
  intensity head's input. Config key `masster_candidate_boundary_dropout`, default 0.0, which
  draws no random numbers.
- `h1` and `h2` use existing keys only.

## Validation evidence

- `tests/test_ema_callback.py` (5 tests): the warmed-up recursion with once-per-step updates,
  validation and checkpoint holding the average, training weights and integer buffers stored
  correctly, exact resume, and refusal of a checkpoint written with another decay.
- `tests/test_masster_candidate_features.py`: 6 boundary-dropout tests.
- All 21 `tests/test_masster*.py` files plus the EMA tests pass (1,460). Config and runner tests
  pass (115). Ruff is clean on every changed file.
- Smokes on `debug_tiny_masster_cut_features.yml`, 2 epochs, local GPU:
  - with `ema_decay=0.9` and boundary dropout 0.3, `last.ckpt` held the average and the training
    weights, and loaded `strict=True` through `init_model_from_ckpt`;
  - with `masster_intensity_transformer_dim=128` and dropout 0.2, the run trained and
    validated.

  Logs are in `output/cross_eval_mh_ft_clean_2026_09_23/` (local).
- Generators: `bash -n` passes for both. The unmerged one checks the 64-epoch horizon, train
  metrics off and checkpoint upload, and all three configs pass. Job generation must run on Fir,
  where `wheels/` is present.

## Reading, after the runs

1. W&B: tail mean per arm against the control.
2. Export the best checkpoint on val and a train sample:
   `scripts/eval/run_save_inference_eval.py --eval_split {val,train} --batch_cutoff 100` (train
   only), with `--auxiliary_scores cos_sim cos_hun --eval_batch_size 256`. Then run
   `ranking_decomp.py`, which lives locally in `output/cross_eval_mh_ft_clean_2026_09_23/analysis/`
   (its variant table needs the new paths). Compare train and val ranking gaps, base-peak hit and
   the low-NCE precursor split against `3bej6fq7`'s exports.
3. Bin val cosine by nearest-train Tanimoto with the variance-novelty scripts.
4. For EMA arms, take train metrics from the offline eval: logged train metrics measure the raw
   weights, validation the average.

## Tail read from W&B (2026-09-24)

Read-only W&B query of `val_cos_sim_0.01_epoch/mean`, last value per epoch. All runs use seed
397623, except `c4a_base_s2` (811237). "Epochs above" counts matched epochs where the arm beats
its control.

| arm | run | tail mean | Δ vs control | epochs above | best (epoch) | min/epoch |
| --- | --- | --- | --- | --- | --- | --- |
| unmerged control `f0` | `3bej6fq7` | 0.7298 | — | — | 0.7318 (61) | 2.99 |
| `h1_head_dropout02` | `rmlyskqj` | 0.7231 | **−0.0067** | 1/64 | 0.7257 (55) | 2.86 |
| `h2_head_dim128` | `4jmgg2o0` | 0.7283 | −0.0015 | 6/64 | 0.7307 (61) | 2.73 |
| `u2_ema999` | `c8hh9j1b` | **0.7392** | **+0.0094** | 64/64 | 0.7394 (56) | 2.84 |
| merged control `c4a_base_s1` | `5qws8g9x` | 0.7266 | — | — | 0.7282 (122) | 0.86 |
| `c4a_base_s2` (seed replicate) | `6plumx6r` | 0.7273 | +0.0007 | 53/128 | 0.7289 (123) | 0.85 |
| `t2_ema998` | `49ub2p8y` | **0.7363** | **+0.0097** | 128/128 | 0.7373 (94) | 0.60 |

Unmerged tail is epochs 54-63; merged tail is 118-127.

- **EMA passes gate 1 on both lineages, by about 0.009-0.010.** That is twice the merged ±0.005
  gate and 14x the s1/s2 seed spread. On the unmerged lineage it is 7x the control's mean
  change between consecutive epochs (0.0013). The EMA scores come from the averaged weights:
  `ExponentialMovingAverage.on_validation_epoch_start` swaps the average in. The arm leads from
  epoch 0 (0.6066 vs 0.5844 unmerged) and holds at the plateau. Merged `t2` peaks at epoch 94 and
  drifts to 0.7363 by 127. Unmerged `u2` is flat from epoch 44 (slope +0.0002 per 10 epochs; best 0.7394 at 56). (Corrected 2026-09-24: this sentence said it was still rising.) At 128 epochs `t2`
  also beats the 630-epoch `long_control` (`tkkesus9`, best 0.7318 at epoch 555). For scale, on
  the cross-eval exports a two-checkpoint weight average gave +0.003 and the f0+f1 two-run
  ensemble +0.020 ([cross-eval](../../findings/masster/masster_target_cleaning_cross_eval_2026_09_23.md)).
  Those numbers are best-checkpoint exports, not W&B tail means.
- **`h1` is rejected.** It is below the control at 63 of 64 epochs. More dropout in the allocator
  costs validation without a gain, as `drop03` did.
- **`h2` is null.** At −0.0015 it is inside noise and below the control at 58 of 64 epochs. Cutting
  the allocator from 2.46M to 0.97M parameters barely moves it. That agrees with capacity not
  being the binding constraint. It ran 9% faster per epoch, but node-to-node spread is about 4-5%
  (`h1` and `u2` were also faster than the control), so the speed-up is not established.
- **Wall clock.** `t2` ran at 0.60 min/epoch against 0.86 for its control, but on
  `fir:h100-80gb` rather than the control's `3g.40gb` slice. This is a hardware difference, not an
  EMA effect.
- **Caveats.** Each arm is n=1, and the unmerged lineage has no seed replicate. The arms ran on
  code after `0f99e7e4` and the controls before it; that change is equal up to float rounding.
  `t2`'s matched control `c4a_base_s1_trainaux` was **never submitted**. So `t2` has no matched
  train cosine, and its logged train cosine (0.8767 tail) measures the raw weights anyway. `u2`
  and `h1`/`h2` uploaded checkpoints to W&B; `t2` did not (`upload_checkpoints: false`), so its
  checkpoint exists only on Fir.

Next, in order:

1. Offline gates 2-3 on `u2`, whose checkpoint is on W&B: the ranking decomposition on val and a
   train sample, and the Tanimoto bins, both against `3bej6fq7`. The question is whether EMA
   shrinks the val ranking gap and helps most on novel molecules. It should, if it removes the
   optimizer-noise part of the [epistemic variance](../../findings/masster/masster_variance_novelty_2026_09_24.md).
2. Whether EMA stacks with LR annealing. `d3`'s cosine decay also damps end-of-training noise,
   so the two may overlap. Test `t2` + `d3` against `d3` (`rcfzf67p`) on the merged screen.
3. Promotion is a user decision. The `ema_decay` template default stays null, so any config
   that wants the average must set it.

## Switch-EMA arm `u3` (built 2026-09-25, not submitted)

**Why.** `u2`'s EMA changes only the weights that are validated and saved. The callback never
writes the average into training, so `u2` trains exactly as its control and keeps the
per-molecule lookup. Its val loss still rises late
([memorization-route plan](masster_memorization_route_plan.md#batch-2-configs-2026-09-25-not-submitted)).
Switch EMA (SEMA, Li et al., 2024) copies the average into the training weights at the start of
every epoch, so the trajectory itself changes. The Hare & Tortoise reset is the same family. The
[literature note](../../notes/memorization_literature_2026_09_24.md) demoted resets, since they add
little once regularization is tuned (Zaidi et al., 2022). No reset had run here.

**Mechanism and risk (pure-drift model, the callback's warmed-up recursion, 308 steps/epoch).**
The average lags the training weights, so each switch pulls training back. The table gives how
far the switched run's average has moved relative to `u2`'s, and the share of each tail epoch's
movement kept. Directions with a restoring gradient undo the pull-back within the epoch, and
there the switch only removes noise.

| decay, switch interval | epoch 10 | epoch 63 | tail epoch kept |
| --- | ---: | ---: | ---: |
| 0.999, 1 epoch (`u3`) | 0.61 | 0.24 | 0.14 |
| 0.999, 4 epochs | 0.88 | 0.55 | 0.46 |
| 0.99, 1 epoch | 0.74 | 0.70 | 0.69 |

So `u3` damps slow, persistent movement. That is where a per-molecule lookup keeps forming after
val saturates (`u2` is flat from epoch 44). It can equally underfit, which shows as `u3` below
`u2` at every epoch with the gap not closing. The 4-epoch interval is the fallback.

**Implementation.**

- `ExponentialMovingAverage(decay, switch_every_n_epochs)` in `fragnnet.utils.pl_utils`, with the
  template key `ema_switch_every_n_epochs` (default null, requires `ema_decay`; the runner raises
  otherwise). At the start of every n-th epoch (never epoch 0), the average is copied into every
  floating-point state entry, BatchNorm statistics included. The optimizer state is kept. The
  average continues its own recursion.
- The switch runs at the epoch's start, not its end. Checkpoints are written during the
  preceding validation, so a run resumed from one switches as an uninterrupted run would. The
  callback state records the last switched epoch, so a mid-epoch resume does not switch twice.
- The callback state now stores the interval and the last switched epoch. A checkpoint written
  before this change (such as `u2`'s) loads as never switched, and resuming with a different
  interval is refused, as a different decay already was.

**Validation.**

- `tests/test_ema_callback.py`: 6 new tests, 11 total. They cover invalid intervals; epochs 2 and
  4 starting from the previous average when the interval is 2, and epochs 1 and 3 not switching;
  no switch at epoch 0 or twice in one epoch; resume after epoch 2 matching an uninterrupted
  4-epoch run bit for bit; a refused interval mismatch; and old callback state.
- Two broken callbacks were run against the new tests. A callback with no switch fails both. One
  that switches at epoch end passes the schedule test and fails the resume test.
- Runner: `ema_switch_every_n_epochs=1` without `ema_decay` raises before any data loads. A
  3-epoch CPU smoke on `debug_tiny_masster_cut_features.yml` with `ema_decay=0.9` and
  `ema_switch_every_n_epochs=1` exits 0.
- Config and runner tests pass (201 with the EMA tests). Ruff is clean on the changed Python.

**Config and job.** `config/masster/mh_ft_ranking_transfer_2026_09_23/u3_switch_ema999.yml` is
`u2_ema999.yml` plus one key and W&B naming, verified through `load_config`. Seed 397623, 64
epochs, `upload_checkpoints: true`, train metrics off. Generate on Fir with
`scripts/manual_slurm/fir/submit_masster_mh_ft_ranking_transfer.sh u3` (`fir:h100-3g.40gb`, 5 h).
Locally the submitter's config checks pass and it stops at the missing `pyteomics` wheel, as
before.

**Read.** The control is `u2` (`c8hh9j1b`, tail 0.7392 over epochs 54-63, best 0.7394 at 56). The
code has moved since `u2` only by default-off keys that draw nothing, as for `k1_ema`. Gate 1 is
+0.003 over `u2`, set against the 0.0022 `c0`/`f0` spread. Then run the offline gates on the
checkpoint, as for `u2`. Do not read `u3` against `u2_s2`, which uses another seed. A pass needs a
seed replicate.

**Result (W&B, 2026-09-26): `u3` (`71xhfgzi`) fails.** It was submitted on 2026-09-25 and
finished in 64 epochs.

| window | `u3` − `u2` cosine | `u3` − `c0` cosine | `u3` − `u2` train loss | `u3` − `u2` val loss |
| --- | ---: | ---: | ---: | ---: |
| 0-3 | −0.0049 | +0.0206 | +0.036 | +0.022 |
| 4-9 | −0.0100 | +0.0098 | +0.092 | +0.040 |
| 20-29 | −0.0112 | +0.0023 | +0.167 | +0.029 |
| 54-63 | −0.0089 | +0.0027 | +0.159 | −0.009 |

- Tail 0.7303 against `u2`'s 0.7392. `u3` is below `u2` at all 64 epochs, and the gap opens by
  epoch 4 and does not close: the underfitting outcome named above. Against `c0` it keeps
  +0.0027, a quarter of `u2`'s +0.0116 and inside the 0.0042 seed spread.
- The train loss (train-mode, training weights) sits 0.16 above `u2`'s for the whole second half:
  each switch pulls training back, and the epoch does not recover the loss.
- Val loss ends 0.009 below `u2`'s while cosine is 0.009 lower. `u3`'s predictions are spread
  more evenly: that suits cross-entropy, which pays for every target peak, and hurts cosine, which
  weighs the dominant ones.
- **Conclusion.** EMA helps here as an average kept off the training weights, not by changing the
  trajectory. The 4-epoch fallback keeps 46% of each tail epoch's movement, not 14%, but a gap
  that opens by epoch 4 and stays constant gives no reason to expect a gain. Not run.

## Open questions and next checks

- If `h1` or `h2` reduces the val ranking gap, try the pair together and the 1-layer allocator the
  audit also lists.
- Whether EMA stacks with `d3`'s LR decay.
- The structural lever against a lookup is more distinct molecules: NIST23-only `[M+H]+`/FT adds
  17,006, estimated at +0.018 (correlational). The alternative is distillation onto unlabeled
  structures.
- The merged lineage's train-vs-val ranking pattern is unmeasured.
