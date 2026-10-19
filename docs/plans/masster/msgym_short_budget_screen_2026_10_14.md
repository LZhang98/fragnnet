# MSGYM short-schedule and candidate-budget screen

> **Status:** READ · **Updated:** 2026-10-18
> Six arms run twice at the same seed (2026-10-14/15), then formula-test retrieval (2026-10-17). 32 epochs is enough. E1024 costs validation wrecall but no resolved retrieval; U128 costs ~0.016 `loglik_cal` hit@1; floor 0.01 costs 0.041. See [Results](#results-2026-10-15) and [Retrieval](#formula-test-retrieval-2026-10-17).

## Questions

The user asked on 2026-10-14 whether the current MSGYM recipe needs (1) the 4096 ranked-mask
budget, (2) U256, (3) a per-molecule fragment probability floor ("dynamic filtering", confirmed
to mean `masster_min_fragment_relative_probability`), and (4) 128 epochs rather than 32 or 64.

## Evidence before the screen

- **Epochs.** In sweep `tiwhun9q`, best validation loss fell at epochs 11–66 (median 28); the
  leader re-run `f44tzxoc` selected epoch 19; `pow075` peaked on validation cosine at epoch 55
  ([sweep read](msgym_val_loss_sweep_48_2026_10_03.md#live-results-read-2026-10-11)). A shorter
  run compresses the cosine LR schedule, which is a different intervention from truncating a
  128-epoch curve, so `c64` and `c32` are new controls, not reads of old runs.
- **U.** On merged NIST20, mask-order U128 tied U256 on cosine at 1.65x the speed per epoch, at
  a cost of 0.022 `val_wrecall`; U64 lost 0.009
  ([tail read](masster_fragment_marginal_selection_plan.md#tail-read-2026-09-24-marginal-selection-fails-and-u128-is-a-free-speed-up)).
  Never tested on MSGYM.
- **E.** The cap binds only when a molecule exhausts it before reaching U unique fragments. The
  August NIST20 sweep processed about 1,550 ranked masks per molecule at U256 and 510 at U128.
  The MSGYM `msgym_u256_e1024_*` configs (2026-10-05, batch-512 recipe) were never run.
- **Floor.** Default 0 in every MSGYM run. It is applied in the model to fragment
  log-probabilities after candidate construction, so it is backend-independent (the CUDA backend
  rejects only the mask-level floor). Pruned fragments receive no proposal gradient. Several
  NIST20 configs used 0.01; no recorded read.

## Arms

All arms copy
[`msgym_sweep_leader_720jwmai.yml`](../../../config/masster/massspecgym/msgym_sweep_leader_720jwmai.yml)
(10.14M parameters, pow0.75, batch 256, LR 1.55e-3, weight decay 0.1, checkpoint on
`val_primary_loss` min, seed 397623). Schedule: epochs x 388 steps; warmup keeps the leader's
2000/49664 fraction. W&B group `msgym_masster_short_budget_20261014`.

| Arm | Config | Epochs | Steps / warmup | U / E | Floor | Read against |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `c64` | [msgym_sb_c64.yml](../../../config/masster/massspecgym/msgym_sb_c64.yml) | 64 | 24832 / 1000 | 256 / 4096 | 0 | `f44tzxoc` (128 epochs) |
| `c32` | [msgym_sb_c32.yml](../../../config/masster/massspecgym/msgym_sb_c32.yml) | 32 | 12416 / 500 | 256 / 4096 | 0 | `c64` |
| `u128` | [msgym_sb_u128.yml](../../../config/masster/massspecgym/msgym_sb_u128.yml) | 64 | 24832 / 1000 | 128 / 4096 | 0 | `c64` |
| `e1024` | [msgym_sb_e1024.yml](../../../config/masster/massspecgym/msgym_sb_e1024.yml) | 64 | 24832 / 1000 | 256 / 1024 | 0 | `c64` |
| `u128_e1024` | [msgym_sb_u128_e1024.yml](../../../config/masster/massspecgym/msgym_sb_u128_e1024.yml) | 64 | 24832 / 1000 | 128 / 1024 | 0 | `c64`, `u128` |
| `floor01` | [msgym_sb_floor01.yml](../../../config/masster/massspecgym/msgym_sb_floor01.yml) | 64 | 24832 / 1000 | 256 / 4096 | 0.01 | `c64` |

Launcher: [`submit_msgym_short_budget.sh`](../../../scripts/manual_slurm/fir/submit_msgym_short_budget.sh)
(`all` | `control` | `arms`; `fir:h100-3g.40gb`, 5 h, generate-only).

## How to read it

- One seed per arm; the earlier MSGYM screen's control spread on validation cosine was 0.0025, so
  smaller differences are unresolved.
- Wall-clock first (minutes per epoch from `_runtime`), then best `val_primary_loss` and its
  epoch, validation `cos_sim_0.01`, `val_wrecall`, and the always-logged candidate statistics:
  `selected_mask_count`, `unique_fragment_target_reached`, `fragment_count_after_cutoff`,
  `fragment_cutoff_pruned_fraction`. Coverage metrics stay off so timing is comparable.
- `e1024` matters only if `unique_fragment_target_reached` drops or `selected_mask_count` sits at
  the cap; if neither moves, E1024 is free.
- Validation cosine does not rank MSGYM checkpoints for retrieval
  ([arm screen finding](../../findings/masster/masster_msgym_formula_retrieval.md#msgym-arm-screen-val-and-test-retrieval-for-wd10-and-pow075)).
  Run formula-test retrieval, with `loglik_cal` re-selected on validation, for `c64` and any arm
  that holds it on validation before adopting a cheaper setting.

## Validation (2026-10-14)

All six resolve through `load_config`, pass `validate_run_config` (W&B online), and build on CPU
at 10.14M parameters. Resolved differences against the leader are exactly the epoch/schedule keys
plus each arm's U, E or floor key. The launcher passes `bash -n`. No GPU smoke: MSGYM
`spec_df.pkl` is not on the local workstation, so `e1024` and `floor01` (new to the CUDA
backend recipe) need a first-epoch check on Fir. Nothing submitted.

## Per-molecule U* sweep (prepared 2026-10-14)

The user asked whether U and E should be learned per molecule. First measurement: re-evaluate the
fixed `c64` checkpoint (`u80w840n`, epoch 21) on val at U = 64/128/192/256/384 (E stays 4096) and
write one row per spectrum per U, then derive each molecule's U* (smallest U past which recall
stops improving) and test whether size features predict it. A size rule is preferred over a
learned stop: `floor01` showed that pruning by the model's own probabilities starves the proposal
of gradient. Script: [`masster_mask_budget_sweep.py`](../../../scripts/eval/masster_mask_budget_sweep.py)
`--per_sample_output_fp`; launcher
[`submit_msgym_u_budget_sweep.sh`](../../../scripts/manual_slurm/fir/submit_msgym_u_budget_sweep.sh).
Local NIST20 smoke (3 batches, U64/U192) passed; MSGYM job not submitted.

## Results (2026-10-15)

Each arm ran twice with identical configs and seed 397623 (A: 2026-10-14 20:43, B: 2026-10-15
01:55). Values at the best-`val_primary_loss` epoch; Δ against the `c64` A/B mean.

| Arm | min/epoch | val cos A / B | Δ cos (mean) | val wrecall A / B | Δ wrecall | U reached |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| `c64` (`u80w840n`, `ixvr0qgk`) | 2.30 | 0.5928 / 0.5926 | — | 0.9013 / 0.9003 | — | 0.68 |
| `c32` (`gz4fa02d`, `3402ar4d`) | 2.29 | 0.5933 / 0.5958 | +0.0019 | 0.9001 / 0.9017 | +0.0001 | 0.70 |
| `u128` (`28dptwet`, `w9mjor6f`) | 1.51 | 0.5904 / 0.5866 | −0.0042 | 0.8611 / 0.8635 | −0.0385 | 0.99 |
| `e1024` (`5esngzcl`, `kt1xn7v5`) | 1.42 | 0.5919 / 0.6000 | +0.0032 | 0.8786 / 0.8786 | −0.0222 | 0.06 |
| `u128_e1024` (`2m4rb2c2`, `fddrhwcd`) | 1.20 | 0.5864 / 0.5864 | −0.0063 | 0.8590 / 0.8603 | −0.0412 | 0.73 |
| `floor01` (`c43o9rbn`, `sqj14xmm`) | 2.47 | 0.5861 / 0.5899 | −0.0047 | 0.7881 / 0.8010 | −0.1062 | 0.63 |

- **Same-seed noise.** Identical reruns differ by up to 0.0081 val cosine (`e1024`) and 0.0038
  (`u128`, `floor01`), on different MIG hosts. The 0.0025 control spread used to plan this screen
  understates it. Validation wrecall reproduces within 0.0016 except `floor01` (0.013).
- **Epochs:** `c32` equals `c64` on wrecall and cosine at half the time; the best epoch is 18 in both
  `c32` replicates.
- **U, E, floor:** cosine differences are within same-seed noise except `u128_e1024`; the wrecall
  losses (U128 −0.039, E1024 −0.022, floor −0.106) reproduce in both replicates.
- Retrieval was the deciding read: see [Formula-test retrieval](#formula-test-retrieval-2026-10-17).

## Formula-test retrieval (2026-10-17)

Ran with [`submit_msgym_sb_retrieval.sh`](../../../scripts/manual_slurm/fir/submit_msgym_sb_retrieval.sh)
on `fir:h100-2g.20gb`: replicate-A best-`val_primary_loss` checkpoints, 9,954 queries (ORB 8,374,
QTOF 1,580), 1,657,238 candidates, none excluded. `queries.csv` is md5-identical to the shards used
for `f44tzxoc` and `pow075`. `loglik_cal` keeps the `ll04i00m` rule (not re-selected). The rewire
control `rpa6uq8k` (config identical to `c32` apart from W&B identity, best epoch 19) was scored on the
same shards, so `c32` vs `rw_control` is a second-seed-path replicate of the same recipe.

Expected hit@1, all queries (Δ against `c64`):

| Arm (run @epoch) | `cos_sim_0.01` | `cos_sim_sqrt_0.01` | `mspred_ent` | `loglik` | `loglik_cal` (hit@5 / hit@20) | Δ `loglik_cal` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `c64` (`u80w840n` @21) | 0.371 | 0.482 | 0.532 | 0.513 | 0.552 (0.870 / 0.970) | — |
| `c32` (`gz4fa02d` @18) | 0.365 | 0.481 | 0.528 | 0.514 | 0.559 (0.870 / 0.973) | +0.007 |
| `rw_control` (`rpa6uq8k` @19) | 0.369 | 0.475 | 0.519 | 0.509 | 0.555 (0.876 / 0.970) | +0.003 |
| `e1024` (`5esngzcl` @26) | 0.375 | 0.478 | 0.529 | 0.508 | 0.545 (0.867 / 0.969) | −0.007 |
| `u128` (`28dptwet` @23) | 0.372 | 0.473 | 0.529 | 0.495 | 0.535 (0.856 / 0.965) | −0.016 |
| `u128_e1024` (`2m4rb2c2` @20) | 0.361 | 0.468 | 0.526 | 0.494 | 0.537 (0.862 / 0.964) | −0.015 |
| `floor01` (`c43o9rbn` @44) | 0.371 | 0.469 | 0.517 | 0.477 | 0.511 (0.837 / 0.955) | −0.041 |
| ref. `f44tzxoc` @19 (128 ep) | 0.365 | 0.470 | 0.523 | 0.503 | 0.545 | −0.007 |

`loglik_cal` hit@1 by instrument: ORB `c64` 0.605, `c32` 0.613, `rw_control` 0.609, `e1024` 0.600,
`u128` 0.594, `u128_e1024` 0.592, `floor01` 0.564; QTOF 0.270, 0.273, 0.266, 0.248, 0.225, 0.246, 0.233.

- **Noise.** Unpaired 95% molecule-bootstrap intervals are ±0.034 on every arm. The same recipe twice
  (`c32` vs `rw_control`) differs by 0.004 `loglik_cal`, 0.009 `mspred_ent`, 0.006 sqrt cosine, and by
  0.04 on the QTOF subset (`mspred_ent` 0.266 vs 0.228), so QTOF per-arm differences are unresolved.
- **Epochs.** `c32` and `c64` tie each other and the 128-epoch leader `f44tzxoc` on every score. 32
  epochs is enough for retrieval as well as validation.
- **E1024.** −0.007 `loglik_cal`, −0.003 `mspred_ent`, +0.004 cosine: inside the replicate spread. The
  −0.022 validation wrecall does not show up in retrieval at this resolution.
- **U128.** −0.016 `loglik_cal` and −0.017 `loglik` (4x the replicate spread, both arms with U128),
  while `mspred_ent` and cosine are flat. The loss is in likelihood scoring, consistent with fewer
  candidate fragments carrying probability mass for peaks the truth explains.
- **Floor 0.01.** −0.041 `loglik_cal`, −0.036 `loglik`, −0.015 `mspred_ent`. Rejected.
- **Not yet paired.** These are differences of marginal estimates; the per-candidate score pickles are on
  Fir (`/scratch/feiw/frag-gnn-data/msgym/formula_test/shard_*/<ckpt stem>/`). Before adopting E1024, run
  `scripts/eval/summarize_retrieval.py --compare sb_e1024:sb_c32 sb_c64:sb_c32` there.
- **Inference speed** is confounded by batch size (`c64`/`rw_control` at 512, the others at 1024; no
  `c32` GPU log), so it is not compared here.

Artifacts: `output/msgym_retrieval/formula_test/` `summary_<stem>.csv`, `summarize_<stem>.log`,
`gpu_usage_<stem>.csv` (copied from Fir via `summaries.zip`).
Tracked copies of the `summary_<stem>.csv` and `gpu_usage_<stem>.csv` files are in
[`docs/tables/msgym_short_budget_screen_2026_10_14/`](../../tables/msgym_short_budget_screen_2026_10_14/).

