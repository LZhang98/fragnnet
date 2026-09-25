# MASShatter W&B campaign read-out, 2026-09-17 to 2026-09-20

> **Status:** FINDING · **Updated:** 2026-09-24 (America/Chicago)
> **§8 (added 2026-09-24) reads the bundle's leave-one-out arms: the bundle is worth 0.006 and no
> single component clears the gate. §6 carries a correction: the "2026-07-13 removal" of
> `boundary_bond_encoder` was a different FraGNNet idea.**
> Read-only W&B review of the 41 runs since 2026-09-15. The regularization sweep produced
> **no promotions and two missing arms**; the atom-decoder removal is **confirmed clock-matched
> on an isolated pair**; the merged track has been flat inside its own replicate floor for 71 runs.
> Two runs launched mid-review carry **three candidate features that have never been trained
> before**, are **+0.014 over the control** through epoch 12, and confound the horizon probe (§6).
> No code, config or run was changed for this review.

## Scope and method

Read-only queries against `frag-gnn/frag-gnn` on 2026-09-21, branch
`research/masshatter-ion-h-selection` at `f6d8bfc1`. Population: all 41 runs created since
2026-09-15, plus the 83 `mass_shatter`-tagged runs since 2026-09-03 for the config census, plus
the 71 runs on the `merge_method: tolerance_consensus` target for the leaderboard. Metric is
`val_cos_sim_0.01_epoch/mean` throughout; `_best/mean` is named explicitly where used.

Per-arm trajectories were re-aggregated by `epoch` from run history rather than read from run
summaries, because train and validation auxiliary metrics are logged on alternating steps and a
naive `dropna` on the validation key silently drops every train-side row.

**Campaign cost.** 155.3 GPU-hours over 41 runs: 7 finished, 31 crashed, 3 failed. Most
"crashed" states are one-hour wall-clock kills and are by design. Nothing has been running since
2026-09-20 01:36 UTC.

## Headline

1. The regularization sweep resolved nothing it was designed to resolve. Six of eight arms
   produced data, all six land inside ±0.0025 of the control mean at matched epoch against a
   ±0.007 screen resolution, and the one arm outside it (`drop03`) is rejected. **No promotions.**
2. **Removing the atom decoder is worth ~3x wall clock at equal quality** — now established on an
   isolated, seed-matched, replicated pair rather than the earlier n=4-vs-44 comparison.
3. The merged leaderboard's top seven runs span **0.0029**, and one config's own replicate pair
   spans **0.0049**. The loss-objective and proposal-warmup families measured nothing.
4. The real progress since 2026-09-17 was throughput: **5.8x epochs per hour**, worth +0.045 val
   cosine at a fixed one-hour budget with no modelling change.

## 1. Regularization sweep Stage A/B read-out

Executes [the sweep plan](../../plans/masshatter/masshatter_regularization_sweep_plan.md). Group
`nist20_mass_shatter_reg_sweep`, 10 jobs on `fir:h100-3g.40gb`, split `nist20_ma_mi_inchikey`.

### Acceptance criteria

| # | Criterion | Result |
| --- | --- | --- |
| A1 | `train_cos_sim_0.01_epoch/mean` logged on every arm | **PASS** — first time in this model family |
| A2 | Every arm reaches a common `E* >= 20` | **PASS** — all reached 64-68; `E* = 64` |
| A3 | Controls agree within 0.007 at `E*` | **PASS** — 0.7152 vs 0.7167, spread **0.0016** |
| A4 | Each arm classified promote / hold / reject | **done**, below: 0 promote, 5 hold, 1 reject |
| A5 | `mass_shatter_candidate_backend_resolved == cuda` | **PASS** on all 10 jobs |
| A6 | 2g-to-3g `min/epoch` ratio measured *before* the fan-out | **FAIL** — see protocol deviations |

### Arms, at matched epoch 64

Control mean 0.7159. Screen resolution ±0.007.

| Arm | Run | val@64 | train@64 | gap | Δ vs control | ep/h | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `ctrl_a` | `ri3zm215` | 0.7152 | 0.8435 | 0.128 | −0.0008 | 65.7 | reference |
| `ctrl_b` | `7cdsx970` | 0.7167 | 0.8463 | 0.130 | +0.0008 | 65.6 | reference |
| `layernorm` | `i1a4l2ya` | 0.7136 | 0.8467 | 0.133 | −0.0024 | 66.0 | hold (null) |
| `graphnorm` | `ucwtbubk` | 0.7135 | 0.8429 | 0.129 | −0.0025 | 65.0 | hold (null) |
| `wd10` | `kalc4mzo` | 0.7174 | 0.8390 | **0.122** | +0.0015 | 65.8 | hold (null, best mechanism) |
| `drop03` | `erfeoywq` | 0.7010 | 0.8139 | **0.113** | **−0.0149** | 66.7 | **reject at this exposure** |
| `hbound_complement` | `jks4gyhp` | 0.7151 | 0.8403 | 0.125 | −0.0009 | 68.0 | hold (free) |
| `hbound_valence` | `ppk9k8l9` | 0.7158 | 0.8445 | 0.129 | −0.0002 | 71.1 | hold (free) |

**Read best-of-epoch with suspicion.** Ranking the same arms by `val_cos_sim_0.01_best/mean`
promotes `hbound_valence` to the top (0.7204) and demotes `ctrl_a` to 0.7170. That ordering is an
artifact: a maximum over ~65 noisy evaluations is a biased estimator, and the plan's matched-epoch
protocol is the correct comparator. At `E*` the two readings disagree by more than the effects
being measured.

### What A1 bought: the gap is real, and only two arms touch it

Train-vs-validation cosine, now visible for the first time:

| Arm | ep 10 | ep 20 | ep 40 | ep 64 |
| --- | ---: | ---: | ---: | ---: |
| `ctrl_a` | +0.024 | +0.064 | +0.109 | +0.128 |
| `ctrl_b` | +0.025 | +0.067 | +0.110 | +0.130 |
| `layernorm` | +0.015 | +0.059 | +0.107 | +0.133 |
| `graphnorm` | +0.021 | +0.065 | +0.107 | +0.129 |
| `wd10` | +0.026 | +0.062 | +0.103 | **+0.122** |
| `drop03` | +0.005 | +0.039 | +0.086 | **+0.113** |
| `hbound_complement` | +0.027 | +0.067 | +0.108 | +0.125 |
| `hbound_valence` | +0.030 | +0.068 | +0.110 | +0.129 |

Only `wd10` and `drop03` move the gap at all, and `drop03` moves it by suppressing the train
cosine (0.814 vs 0.843) rather than by lifting validation. `layernorm` and `graphnorm` are exactly
null on both sides, confirming the
[regularization audit's](masshatter_regularization_audit_2026_09_19.md) prediction that the
train/eval candidate-set divergence and the generalization gap are different problems.

### Every null carries "at 10% exposure"

`max_epochs: 630`; every arm died at epoch 64-68 on the one-hour wall. **The best validation epoch
is the last epoch, or within six of it, for all eight arms** (58-65). Validation is still climbing
at the cutoff, +0.0006 to +0.0037 per 10 epochs over epochs 50-64, while train climbs +0.0077 to
+0.0119. Nothing converged, and the screen is structurally biased against arms that trade early
fitting speed for late generalization — which is exactly what `drop03` is doing, still gaining
+0.0027 per 10 epochs from a lower base at the moment it was judged.

### Two arms produced no data

- **`_drop02` (`6d3ajnpe`) failed on the out-of-range peak bug**, not on OOM:
  `AssertionError: (tensor(1501.7432), 1500)` at [spec_utils.py:433](../../../src/fragnnet/utils/spec_utils.py#L433),
  reached through the train-side auxiliary metrics. This is the incident the plan already
  root-caused; the guard landed in `ffe5d53a` at 20:02, about 70 minutes *after* this arm started.
  The dropout axis therefore has data at 0.1 and 0.3 and a hole at 0.2 — the midpoint that would
  say whether `drop03` is over-braked or dropout is simply wrong here.
- **`_ctxnorm` was never submitted.** The config exists and resolves
  (`config/massshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_ctxnorm.yml`)
  but no run in the project carries `mass_shatter_context_normalization` other than `none`.
  The plan rates this arm's mechanism as real; it remains untested.

### Protocol deviations worth recording

- **A6 was not met and the fan-out proceeded anyway.** The 2g pilot that was supposed to decide
  between sweep shapes A/B/C (`ybmf4gaq`) OOM'd, and `ctrl_a` launched three minutes later. Shape A
  was therefore taken by default rather than by measurement, and the 2g slice remains unmeasured.
- **The pilot used the shape the plan warned against.** It ran `batch512 × lb16`, the variant the
  plan flagged as "projected, not measured" at ~15.8 GiB, rather than the prepared
  `batch256×2 × lb32` at a measured 12.7 GiB. It OOM'd in `calculate_match_mzs` on the dense
  `|true| × |pred|` tensor while co-tenant with another process on a 19.62 GiB device.
- **The arms span three source revisions.** The six early arms ran at roughly `86cbf98d`; the two
  `hbound` arms ran after `ffe5d53a` and `904cefc6`. This was checked and is benign:
  `h_transfer_bound="none"` returns an all-ones mask (exact no-op), `atom_free_valence_slots` is a
  side attribute on the graph rather than an atom feature, and steps/epoch is 91.3-91.4 on all
  eight arms, so no spectra were filtered. Benign by inspection, not by design.

### The hydrogen-transfer bound is free, and slightly better than free

Candidate ion-H channels on validation, with the dataset unchanged (91.3-91.4 steps/epoch in all
three conditions):

| Mode | `val_valid_ion_h_count` | Δ | val@64 vs control |
| --- | ---: | ---: | ---: |
| `none` | 2030-2031 | — | reference |
| `complement` | 1898 | −6.5% | −0.0009 |
| `valence` | 1844 | **−9.2%** | −0.0002 |

Removing 9.2% of the candidate channels costs nothing measurable. This settles the plan's open
question — the unreachable channels are neither useful softmax negatives nor isotope crutches —
and supports `mass_shatter_h_transfer_bound: valence`, already the template default since
2026-09-20. The apparent +8% throughput on the `valence` arm is consistent with the smaller
candidate set but is confounded by the later launch and different revision; treat it as
suggestive.

## 2. The atom decoder: confirmed, and the campaign's only real gain

The [regularization audit §3.5](masshatter_regularization_audit_2026_09_19.md) removed the atom
decoder on the strength of a confounded n=4-vs-44 comparison. It has now been tested properly:
**two `_fast` against two `_fast_ad0`, seed 397623 throughout, resolved diff exactly
`mass_shatter_intensity_atom_decoder_num_layers: 4 -> 0`** (−34.8% params). Group
`nist20_mass_shatter_warmup_loss_transform`, split `nist20_inchikey`, 200 epochs.

| Run | decoder | min/ep | best val | @2h | @3h | @4h | @8.7h |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mgs5qts9` | 4 | 2.60 | 0.7990 | 0.7800 | 0.7910 | 0.7958 | 0.7990 |
| `lzhb3ozh` | 4 | 2.62 | 0.7941 | 0.7786 | 0.7844 | 0.7911 | 0.7941 |
| `zp06g7ml` | **0** | **1.44** | 0.7987 | **0.7968** | 0.7969 | 0.7987 | — |
| `6ru0lg6f` | **0** | **1.46** | 0.7981 | 0.7909 | 0.7959 | 0.7970 | — |

Epoch-matched the change is null: +0.0018 mean, inside the `_fast` pair's own 0.0049 spread.
**Clock-matched it wins outright**: the decoder-free runs average 0.7964 at three hours, which is
what the decoder versions reach at 8.7 hours, and they match the all-time leaderboard top
(`8v0i0fq7`, 0.8010 at 11.9 h) within noise at about four hours. Roughly **3x less compute for the
same answer**, consistent with the audit's prediction that the decoder's benefit decays to
+0.0008 by epoch 100 and loses clock-matched.

`mass_shatter_intensity_atom_decoder_num_layers: 0` is already the template default. This is the
isolated confirmation that default was missing.

**Secondary consequence, untested:** `qszc5huc` OOM'd at 44 GiB trying `batch512` on the merged
config, inside the atom decoder's cross-attention (`transformer.py` `_mha_block`). Removing the
decoder plausibly unblocks that batch size.

## 3. The merged track is flat inside its own noise floor

Top of the `tolerance_consensus` merged leaderboard, all runs since 2026-09-03 (`val_cos_sim_0.01_best/mean`):

| val cos | Run | hours | Distinguishing key |
| ---: | --- | ---: | --- |
| 0.8010 | `8v0i0fq7` | 11.9 | JSS, warmup e0=4 |
| 0.7996 | `38ugaxen` | 11.9 | JSS, warmup e0=15 |
| 0.7990 | `mgs5qts9` | 8.7 | JSS, e0=4, fast |
| 0.7987 | `zp06g7ml` | **4.8** | as above, no atom decoder |
| 0.7985 | `7abal2ru` | 9.7 | probability-CE |
| 0.7984 | `kj17tksj` | 12.0 | KL |
| 0.7981 | `6ru0lg6f` | **4.9** | no atom decoder |

**Seven runs, three loss objectives, span 0.0029.** Against that, the two nominally identical
`_fast` runs (same seed, same resolved config) differ by **0.0049**, and the two `_fast_ad0` runs
by 0.0006. The `_fast` pair sits two source commits apart, so 0.0049 is an upper bound on
reproducibility rather than pure nondeterminism — but it is consistent with the 0.003-0.005
same-seed floor recorded in the
[loss-objective comparison](masshatter_loss_audit_2026_09_19.md) and with the 0.007 four-replicate
spread in the sweep plan.

**The proposal-warmup sweep measured nothing.** It is a clean one-key isolation over
`mass_shatter_proposal_only_until_epoch`, same seed, and the result is non-monotone:

| e0 | 0 | 2 | 4 | 9 | 15 |
| --- | ---: | ---: | ---: | ---: | ---: |
| best val | 0.7939 | 0.7967 | **0.8010** | 0.7969 | 0.7996 |

Range 0.0071, no trend, replicate floor 0.0049. "e0=4 is best" is a coin flip that is now baked
into the `_fast` config lineage. Harmless, since any value works — but it should not be cited as a
tuned choice.

## 4. Throughput was the campaign's real product

Same one-hour budget, same `ma_mi` merged data, config and code changes accumulated between
2026-09-17 and 2026-09-20:

| Configuration | s/epoch | epochs in 1 h | best val |
| --- | ---: | ---: | ---: |
| `bs128`, CPU candidates, `lb8` (`r0tr3b9i`) | 294 | 11 | 0.6752 |
| + CUDA candidate backend (`qhs040sy`) | 164 | 21 | 0.7010 |
| + `bs512`, `lb64` (`o7zpxi78`) | 112 | 31 | 0.7043 |
| + batched losses, 09-18/19 (`bpbome2x`) | 72 | 48 | 0.7167 |
| + `eval_batch_size: 512` (`ri3zm215`) | 55 | 65 | 0.7170 |
| current (`ppk9k8l9`) | **51** | **68** | **0.7204** |

**5.8x throughput, +0.045 val cosine at fixed wall clock, no modelling change.** At matched epoch
20 every row agrees to 0.007, so the speedups are quality-neutral as intended. Component
attribution for each step is in the
[training-speed finding](masshatter_training_speed_2026_09_17.md); this table is the cumulative
end-to-end result on the sweep base.

## 5. Config census: what has still never been varied

Across the 83 `mass_shatter` runs since 2026-09-03:

| Key | Distribution |
| --- | --- |
| `lr_schedule` | `False` in **83/83** |
| `lr` | `0.001` in **83/83** |
| `gradient_clip_val` | `5` in **83/83** |
| `weight_decay` | `0.01` in 82/83 (the `wd10` arm is the exception) |
| `mass_shatter_head_dropout` | `0.1` in 81/83 (one at 0.2 that crashed, one at 0.3) |
| `seed` | `397623` in **81/83** |

Every MASShatter run ever trained has used a constant learning rate of 1e-3 with no warmup and no
decay: `lr_schedule_type: exponential` is inert because the master switch is off. The 200-epoch
merged runs peak at epochs 117-181 and then oscillate ±0.005 through epoch 199, which is what a
missing decay looks like, and that oscillation is the same order as the entire leaderboard spread
in §3.

**The documented blocker on an LR arm is gone.** The sweep plan states that
`configure_optimizers` drops `lr_schedule_type`, `lr_total_steps` and `lr_min_ratio`, so a
`cosine` config would silently run exponential. That forwarding was fixed in `031c370d`
(2026-09-20 01:20) with `tests/test_lr_schedule_forwarding.py`; the four tests pass on the current
checkout. The plan's statement is now stale, and the cosine branch is reachable — it requires
`lr_total_steps`, which `build_lr_scheduler` enforces by raising.

## 6. Live runs, 2026-09-21 06:42 UTC — the horizon probe is confounded, and looks good anyway

Two runs launched during this review and are still training.

| Run | Name | min/ep | State |
| --- | --- | ---: | --- |
| `tkkesus9` | `..._lb64_long_control` (the horizon probe) | 0.99 | running |
| `jvaj1lyw` | `..._lb64_warmup_e0_4` | 0.98 | running |

**The horizon probe is not the control plus one key.** Its config header states the only
deliberate difference from the sweep controls is `mass_shatter_h_transfer_bound: valence`, "so the
curves should overlay". The resolved config differs in **four** behavioral keys, because three
template defaults flipped in `99169f35` after the config was written and none of them is pinned in
the file:

| Key | `ctrl_a` | `tkkesus9` |
| --- | --- | --- |
| `mass_shatter_h_transfer_bound` | `none` | `valence` (intended; measured −0.0002) |
| `mass_shatter_candidate_mass_features` | absent | **`True`** |
| `mass_shatter_candidate_charge_feature` | absent | **`True`** |
| `mass_shatter_boundary_bond_encoder` | absent | **`True`** |
| `compute_train_auxiliary_metrics` | `true` | `false` (intended; no train cosine) |

Before these two runs, **no run in the project had ever trained with any of the three new
candidate features** (238 `mass_shatter` runs since 2026-08-01, all with the keys absent). This is
the shared-template trap again: a default change reaches every config that does not pin the key.

**Consequence for the probe's stated purpose.** It will measure where the *new* configuration
peaks, not where the configuration the sweep screened peaks, so its curve cannot be overlaid on
the §1 arms to retro-size that screen.

**Consequence that matters more.** Against the mean of the two sweep controls at matched epoch:

| epoch | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| control mean | 0.5823 | 0.6287 | 0.6527 | 0.6567 | 0.6744 | 0.6746 |
| `tkkesus9` | 0.6028 | 0.6381 | 0.6593 | 0.6753 | 0.6835 | 0.6895 |
| Δ | +0.0205 | +0.0094 | +0.0066 | +0.0187 | +0.0092 | **+0.0149** |

Mean +0.0137 over epochs 0-13, positive at every sampled epoch. **That is larger than any effect
the regularization sweep measured, and 2-3x the replicate floor.** Treat it as suggestive, not
established: epoch 10 is exactly where the plan's four-replicate spread was widest (0.0124), and
the lift is a three-key bundle, not an isolation. `hbound valence` is measured at −0.0002, so
whatever is happening is coming from the mass, charge or boundary-bond features.

`mass_shatter_boundary_bond_encoder` is on by default and is now inside this bundle.
**Correction 2026-09-24:** this paragraph used to say the key was "removed on 2026-07-13 as the
weaker bet against the competition head". That removal was a different, FraGNNet-side idea,
`BoundaryBondEncoder`, which was never committed. `CutBondEncoder` was added to FraGNNet the next
day as the `cut_bond` feature. MASShatter reused it as `boundary_bond_encoder` on 2026-09-20 (the
MASShatter model itself dates from 2026-08-14), and it became the template default on 2026-09-21.
It has no earlier MASShatter removal to reverse.

`jvaj1lyw` shows the expected proposal-only warmup signature — cosine near 0.06 through epoch 4,
then catching up — and is still 0.025 behind the control at epoch 12. It carries the same three
new features, so it is not a clean warmup isolation either.

## 7. Guards for anyone reading these numbers

- **`nist20_ma_mi_inchikey` and `nist20_inchikey` are different splits.** `ma_mi` is FT+QTOF ×
  HCD+CID × `[M+H]+/[M-H]-/[M+Na]+/[M+Cl]-`; the merged track is FT × `[M+H]+` only. The 0.717
  band and the 0.798 band are not comparable, and that gap is dataset scope, not a regression.
- Train and validation auxiliary metrics are logged on alternating steps; aggregate history by
  `epoch` before comparing them.
- The dense `|true| × |pred|` tensor in `calculate_match_mzs` is the mechanism behind the
  "train-aux metrics cost 10-26 GiB" note, and it is what OOM'd the 2g pilot.

## 8. The bundle's leave-one-out arms, read at the 128-epoch screen (added 2026-09-24)

Read-only W&B query. All five arms ran 128 epochs at seed 397623, paired with `c4a_base_s1`.
Each run's resolved config was checked against `c4a_base_s1`. The older
`candidate_mass_features`/`candidate_charge_feature` keys map onto the `c_mass`/`c_charge` entries
of the later `mass_shatter_candidate_features` list, and every other difference is an inert
default. Tail mean over epochs 118–127 against the 0.7270 control (gate ±0.005), as in the
[screen read-out](masshatter_architecture_arms_epoch128_2026_09_22.md):

| arm | run | removes | cosine | vs control |
| --- | --- | --- | ---: | ---: |
| `c4a_bundle_off` | `9uec7nwx` | all four below | 0.7210 | **−0.0060** |
| `c4a_no_acharge` | `fr4x7mgi` | `a_charge` atom input | 0.7245 | −0.0025 |
| `c4a_no_bbe` | `pkavjg8z` | `boundary_bond_encoder` | 0.7246 | −0.0024 |
| `c4a_no_charge_feat` | `gwvz8mdp` | `c_charge` candidate column | 0.7258 | −0.0012 |
| `c4a_no_mass_feats` | `xp66b5ft` | `c_mass` candidate columns | 0.7282 | +0.0012 |

- **The bundle is worth about 0.006 at the screen, and no single component carries it.** Each
  removal lands inside the gate, and the four single-arm deltas sum to −0.0049 against the
  bundle's −0.0060. The h-transfer bound was not part of `bundle_off`; §1 measured it at −0.0002.
- **The early +0.014 shrinks to 0.006 at the tail.** §6 read it at epoch 12, against a different
  control, near the epoch-10 point where the regularization plan's replicate spread was widest.
- **`c4a_no_bbe` is also 6% faster per epoch** (0.80 against 0.85–0.86 min/epoch). Its
  train-mode primary-loss gap is 0.6136 against a 0.6191 control mean, inside the 0.014 floor.
  The run logged no train cosine and uploaded no checkpoint, so the
  [memorization probe](masshatter_memorization_route_probe_2026_09_23.md) cannot be run on it.
- **The `c4a_no_bbe` config's pre-registered rule does not apply.** Its header says the
  "2026-07-13 removal stands" if the bundle's lift survives the arm. That removal never happened
  on MASShatter (see the §6 correction), so the rule has no premise. Whether the encoder stays on
  is an open decision that this screen does not settle.

## Next concrete checks

Ordered by expected value, with the prepared artifact where one exists.

1. **Isolate the three new candidate features** (§6). They are the largest live signal in the
   campaign, they are on by default in every unpinned config from `99169f35` onward, and they have
   never been trained in isolation. One job per key against a pinned control — `mass_features`,
   `charge_feature`, `boundary_bond_encoder` — settles which of the three is carrying +0.014.
   Until this runs, no later result on this base is attributable. **Ran; read in §8. No single
   component clears the gate, and the bundle as a whole is worth 0.006.**
2. **Let the horizon probe finish** (`tkkesus9`, ~10.4 h at 0.99 min/epoch). It no longer answers
   the question it was written for, but "where does val peak on the current default" is the more
   useful version of that question now. Budget a matched re-run on the pinned pre-`99169f35`
   config only if the §1 arms need retro-sizing.
3. **Add an LR-schedule arm**, now that forwarding works. This is the only knob in §5 that has
   never been varied and has a mechanism pointing at the §3 noise floor: the 200-epoch runs peak
   at epochs 117-181 then oscillate ±0.005, which is the same order as the entire leaderboard
   spread. Cosine needs `lr_total_steps` set from the run's planned steps, not from `max_epochs`,
   whenever the job is wall-clock capped. An exponential-decay arm is already prepared on the
   unmerged anchor (`nist20_ma_mi_unmerged_cmf_dual_ce_fast_batch256_cuda_lr_decay.yml`), but its
   header comment about the forwarding bug is stale.
4. **Fill the two holes in the sweep** — `_drop02` (rerun; the bug that killed it is fixed) and
   `_ctxnorm` (never submitted). Both are one job each and both configs exist. Pin the three new
   feature keys first, or they will not be comparable to the six arms that already ran.
5. **Re-run `drop03` and `wd10` at the horizon**, not at 64 epochs. They are the only two arms that
   moved the generalization gap, and both were judged at the exposure that flatters the control.
6. **Try `batch512` on the merged config now that the decoder is gone**, since the decoder's
   cross-attention is what OOM'd `qszc5huc`.

**Pin defaults in experiment configs.** Items 1 and 4 both exist because template defaults moved
under configs that were written days earlier. Any config that is part of a comparison should state
the keys its comparison depends on, rather than inheriting them.

Sizing note: at the measured ±0.007 screen resolution and a 0.003-0.005 replicate floor, a
one-hour n=1 arm cannot resolve the effects this family is producing. Either arms run long enough
to separate, or they run at n≥2. The campaign spent 155 GPU-hours to learn one thing that
replicated; the atom-decoder pair is the template for how to spend the next 155.
