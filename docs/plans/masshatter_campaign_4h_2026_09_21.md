# MASShatter 4 h exploration campaign (2026-09-21)

> **Status:** PREPARED, NOT SUBMITTED · **Created:** 2026-09-21
> 20 arms x 4 h on `fir:h100-3g.40gb`, all single-key diffs from one base.
> Read with the [campaign read-out](../findings/masshatter_campaign_readout_2026_09_21.md)
> and the [regularization sweep plan](masshatter_regularization_sweep_plan.md).

## Why this campaign exists

Three defaults flipped on **2026-09-20**, after the regularization sweep had already read out:

| Key | Was | Now | Ever A/B'd? |
| --- | --- | --- | --- |
| `mass_shatter_candidate_mass_features` | `False` | `True` | **No** |
| `mass_shatter_candidate_charge_feature` | `False` | `True` | **No** |
| `mass_shatter_h_transfer_bound` | `none` | `valence` | Yes (free, -0.0002) |

The first two are the fix for the mass-blindness the architecture review named as the largest gap
in the stack: without them "no mass, m/z or composition reaches a learned parameter at all". They
change the intensity head's input width. **There is currently no baseline on current defaults, and
every one of the 83 historical `mass_shatter` runs was trained on the old architecture.** Tier A
establishes that baseline and prices the change; without it nothing else in the campaign can be
placed against the existing leaderboard.

The second premise is exposure. Every arm of the previous sweep was still climbing at the one-hour
wall, at 10% of its schedule, and the best validation epoch was the last epoch for all eight. The
200-epoch merged runs peak at epochs 117-181, so a 4 h box (~180-270 epochs, see the throughput
risk below) is the cheapest length that plausibly contains the peak.

## Arms

Base: `config/massshatter/ma_mi_merged/nist20_ma_mi_merged_fast_batch512_cuda_lb64_long_control.yml`
(current defaults, `compute_train_auxiliary_metrics: false`, `lb64`, microbatch 512, seed 397623).
Every arm was verified by resolving through `load_config` and diffing against the resolved base:
each sets only its listed keys, and nothing else moves.

### Tier A — baseline and the unmeasured 2026-09-20 defaults (5)

| Arm | Diff from base |
| --- | --- |
| `c4a_base_s1` | none — this is the control |
| `c4a_base_s2` | `seed: 811237` (replicate floor at the new horizon) |
| `c4a_no_mass_feats` | `candidate_mass_features: false` |
| `c4a_no_charge_feat` | `candidate_charge_feature: false` |
| `c4a_pre0920_arch` | both false — the bridge to every historical number |

### Tier B — optimization schedule (6)

`lr_schedule: False` in 83/83 runs, `lr: 0.001` in 83/83, `gradient_clip_val: 5` in 83/83. The
stack is post-LN under a constant, unwarmed 1e-3, and the late-training ±0.005 oscillation is the
same order as the entire leaderboard spread. Forwarding was fixed in `031c370d`, so the cosine
branch is reachable for the first time.

| Arm | Diff from base |
| --- | --- |
| `c4b_lr_cos` | cosine + 1000-step warmup, `lr_total_steps: 18000` |
| `c4b_lr_exp` | exponential decay + 1000-step warmup |
| `c4b_lr_warmonly` | warmup only (`lr_decay_rate: 1.0`) |
| `c4b_lr_cos_nowarm` | cosine, `lr_warmup_steps: 0` |
| `c4b_lr_3e4` | constant `lr: 0.0003`, schedule still off |
| `c4b_clip1` | `gradient_clip_val: 1.0` |

The first four plus the control form a 2x2 over (warmup, decay); the control is the
(no warmup, no decay) corner.

**`lr_total_steps` is 18,000 = 90 steps/epoch x 200 epochs.** The 90 was measured by resolving the
base config against the real data and reading the train dataloader length (~46k merged rows at
microbatch 512), not derived from `max_epochs: 630`. It is deliberately sized for the *slow*
throughput case so the cosine completes inside the box; `build_lr_scheduler` clamps `progress` at
1.0, so an overrun simply trains the extra epochs at `lr_min_ratio x peak`.

Note the scale this sets: a 1000-step warmup is **11.1 epochs**, and `lr_decay_steps: 5000` is
55.6 epochs, so the exponential arm applies its 0.95 decay only 3-4 times across the box.

### Tier C — proposal warmup shape (4)

The merged `nist20_inchikey` sweep ran e0 = 0/2/4/9/15 -> 0.7939/0.7967/**0.8010**/0.7969/0.7996:
+0.0071 for "some warmup", e0=4 best, and the shape between 2 and 15 inside run noise. This is the
only knob with a measured positive effect anywhere in the record, and it is untested on `ma_mi`.
Arms set `mass_shatter_proposal_only_until_epoch` to 2, 4, 9, 15; e0=0 is `c4a_base_s1`.

### Tier D — the gap-movers, retested at the horizon (5)

Only `wd10` (gap 0.122) and `drop03` (0.113) moved the train/val gap off the control's 0.128, and
the sweep plan states *in advance* that a one-hour screen is biased against regularizers, which
"characteristically trail early and win late".

| Arm | Diff from base | Why |
| --- | --- | --- |
| `c4d_drop02` | all three dropouts 0.2 | never produced data; killed by the bug fixed in `ffe5d53a` |
| `c4d_drop03` | all three dropouts 0.3 | rejected at ep64 with the exact signature of a too-early judgement |
| `c4d_wd10` | `weight_decay: 0.1` | climbing fastest of any arm at the wall (+0.0047/10ep) |
| `c4d_drop_head03` | head dropout only 0.3 | decomposes drop03; 74% of the parameter budget is here |
| `c4d_drop_mol03` | mol dropout only 0.3 | the encoder half of the same decomposition |

## Read-out protocol — fixed before the numbers exist

1. **Matched epoch, never matched wall clock.** Throughput moved 2.5x between two same-config runs
   on one day. Pick `E*` = the minimum final epoch across arms and read every arm there.
2. **The floor comes from `c4a_base_s1` vs `c4a_base_s2`**, measured at `E*`, not inherited from
   the 0.003-0.005 figure, which was measured at 20-64 epochs.
3. **Every arm is n=1.** With a floor around 0.005 only effects above roughly 0.01 are credible.
   This campaign is a screen that nominates arms for replication; it cannot promote on its own.
4. **Tier A is read first.** If `c4a_pre0920_arch` differs materially from `c4a_base_s1`, all
   historical comparisons need restating before Tiers B-D mean anything.
5. Train and validation metrics log on alternating steps — aggregate by `epoch` before joining, or
   the comparison is silently NaN.
6. `compute_train_auxiliary_metrics` is **off** in every arm, which is what keeps peak GPU near
   26.8 GiB and the jobs on a 3g slice. The cost is that these runs report the validation curve
   only: no train cosine, so **no train/val gap**. Tier D is therefore judged on val cosine alone,
   even though the gap is what motivated it. Turning the flag on costs 10-26 GiB and forces a full
   80 GB card.

## Preconditions

- **Sync the Fir checkout** to current `HEAD`. Today's template refactor (`99169f35`) collapsed ten
  per-model templates into one `config/template.yml` and removed the sibling-template staging block
  from `setup_job_cc.py`, and these 20 configs did not exist before `97c41abc`. Note what this is
  *not*: `_drop02` was not lost to a stale checkout. Its guard landed in `ffe5d53a` at 20:02, about
  70 minutes **after** that arm started, so no sync would have saved it — the bug was simply live at
  submit time. It is fixed on the current checkout now.
- **Generate the job scripts on the Fir login node**, from the Fir checkout: `setup_job_cc.py`
  bakes `FRAGNNET_DP` into each script.
- No preprocessing regeneration is needed. `mol_df.pkl` stores RDKit `Mol` objects and Gasteiger
  charges are computed at graph-construction time (`feat_utils.py:276`), so the template's
  "caches written before 2026-09-20 must be regenerated" caveat does not apply to this path.

## Throughput risk

The 0.84-0.95 min/epoch figures behind "~240 epochs in 4 h" were measured on **full 80 GB H100s**,
because train-side auxiliary metrics forced the previous sweep off the MIG slice. The 2g-to-3g
ratio was never measured (the 2g pilot OOM'd, gate A6 failed). At 1.5x slower a 4 h box yields
~180 epochs rather than ~270. Consequences: `E*` lands lower than planned, and the cosine arms
finish their schedule early and hold at `lr_min_ratio` — both benign, but read `E*` from the runs
rather than assuming 240. Check `min/epoch` on the first arm before submitting the remainder.

## Validation performed

- All 20 configs resolve through `load_config`; each sets exactly its intended keys with no stray
  movement, verified against the resolved base.
- The three keys the campaign's comparisons depend on (`candidate_mass_features`,
  `candidate_charge_feature`, `h_transfer_bound`) are written literally into all 20 files rather
  than inherited, per the read-out's "pin defaults in experiment configs" rule; verified by text
  search, not just by resolution.
- `tests/test_config_templates.py` and `tests/test_lr_schedule_forwarding.py`: 26 passed.
- Cosine schedule smoke-tested end-to-end on `debug_tiny_mass_shatter` (the branch had never
  executed): `lr_schedule/type/total_steps/min_ratio/warmup_steps` all forwarded, 2 epochs clean.
- Current defaults smoke-tested end-to-end on `debug_tinyv4`: 2 epochs clean.
- All 20 job scripts generated and `bash -n` clean; resources resolve to
  `--gres=gpu:h100_3g.40gb:1`, 6 CPUs, 140G.
- Steps per epoch (90) measured from the real dataloader, not estimated.
- **Nothing was submitted.**
