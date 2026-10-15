# MSGYM short-schedule and candidate-budget screen

> **Status:** PREPARED · **Updated:** 2026-10-14
> Six arms on the sweep-leader recipe at 64 epochs (`c64`, `c32`, `u128`, `e1024`, `u128_e1024`, `floor01`); configs and Fir launcher ready, not submitted.

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
  ([arm screen finding](../../findings/masster/masster_msgym_arm_screen_formula_retrieval_2026_10_07.md)).
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
