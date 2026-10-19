# Spectraverse v1.0.3: first control and MSGYM-recipe arms

> **Status:** PREPARED · **Updated:** 2026-10-19
> Four arms × cv0/cv1 (8 runs) on v1.0.3, each one change from its parent; configs resolve and validate; nothing submitted.

## Objective and acceptance

Get the first v1.0.3 MASSter baseline, and test whether what MSGYM taught us transfers to
Spectraverse, where val peaks at epoch 15 of 40 and train cosine (0.590) sits far above val
(0.458) on v1.0.2 ([oracle ceiling](../../findings/masster/spectraverse_masster_findings.md#spectraverse-masster-oracle-cosine-ceiling)).
An arm is promoted only if it beats its parent on both folds.

## What is taken from MSGYM, and what is not

| MSGYM evidence | Used here |
| --- | --- |
| Sweep `tiwhun9q`: encoder dropout 0.2–0.3 and higher LR are the strongest signals; weight decay 0.1 and batch 256 also lead ([sweep read](msgym_val_loss_sweep_48_2026_10_03.md#what-the-random-search-suggests)) | `reg`: the leader `720jwmai`'s dropouts, LR, weight decay and batch |
| `mol_dropout` 0.5 on NIST20 `i4` (`md05`) runs about −0.01 behind ([generalization hub](../../workstreams/masster_generalization.md)) | `mol_dropout` capped at 0.2 (leader value) |
| Depth: the sweep resolved no winner | Depths stay at the SPV values (encoder 2, bond 3, intensity transformer 2) |
| `pow075` pair: val level, formula-test `mspred_ent` hit@1 +0.031 ([arm screen](../../findings/masster/masster_msgym_formula_retrieval.md#msgym-arm-screen-val-and-test-retrieval-for-wd10-and-pow075)) | `reg_pow075` |
| 32 epochs ties 64 and 128 on val and retrieval ([short-budget screen](msgym_short_budget_screen_2026_10_14.md#formula-test-retrieval-2026-10-17)) | `reg_e20` (v1.0.2 val already peaks at epoch 15) |
| U128 −0.016 and floor 0.01 −0.041 `loglik_cal`; E1024 unresolved (unpaired) | Kept U256/E4096, no floor |
| Loss-selected checkpoints for retrieval | Not used: checkpoint on `val_cos_hun_inst_tol` as the v1.0.2 control does, since val CE is not comparable across `ints_transform` |

## Arms

Configs: `config/masster/spectraverse_1_0_3/<arm>/cv{0,1}.yml`, W&B group
`spectraverse103_masster_msgym_20261019`. The base is the v1.0.2 control
(`config/masster/spectraverse/masster_c0rerun/`). Its header holds the recipe reasoning: CMF on,
unmerged, dual CE with a missing-CE token, fourier CE, EMA 0.999, `loss_batch_size` 4.

| Arm | Change from parent | Epochs | Batch | Steps cv0 / cv1 (warmup) | Read against |
| --- | --- | ---: | ---: | --- | --- |
| `c0` | v1.0.2 control on v1.0.3 data and paths | 40 | 512 | 28720 / 28720 (1000) | baseline |
| `reg` | `mol_dropout` 0.1→0.2, head dropout 0.1→0.3, boundary-encoder and intensity-input dropout 0→0.2, weight decay 0.01→0.1, LR 1e-3→1.55e-3, batch 512→256 | 40 | 256 | 57440 / 57400 (2000) | `c0` |
| `reg_pow075` | + `ints_transform: pow0.75`, `masster_merge_same_mz_rows: true` | 40 | 256 | 57440 / 57400 (2000) | `reg` |
| `reg_e20` | 20 epochs | 20 | 256 | 28720 / 28700 (1000) | `reg` |

Steps are epochs × ceil(train / batch), with train = 367,561 (cv0) and 367,274 (cv1). Resolved
configs were diffed against their parents: each differs in exactly the keys above, plus
`wandb_name`. cv0 and cv1 differ only in `split_dp`, `lr_total_steps` and the W&B name.

`reg` changes six knobs together. That matches the question "does the MSGYM recipe transfer?".
If it wins, attributing the gain needs follow-up arms.

## Data and provenance

- Source revision when prepared: `e70edd70` plus these uncommitted configs.
- On-disk paths are `data/proc/spectraverse_1_0_3/` and `data/split/spectraverse_benchmark_1_0_3/`.
  The [hub](../../workstreams/spectraverse.md) and the `preproc_scripts/spectraverse/` defaults
  still say `spectraverse_1_0_3_benchmark`. v1.0.2 now lives at `data/proc/spectraverse_1_0_2/`
  and `data/split/spectraverse_benchmark_1_0_2/`, so the v1.0.2 configs in `config/masster/spectraverse/`
  no longer resolve locally.

| File | md5 |
| --- | --- |
| `data/proc/spectraverse_1_0_3/spec_df.pkl` | `a2a0fb638640d26cf5f09d4f31e56c3d` |
| `data/proc/spectraverse_1_0_3/mol_df.pkl` | `f797f152709ea511c520812e01a3f60e` |
| `cv0/train_ids.csv` / `val_ids.csv` / `test_ids.csv` | `9a7efcfb...` / `1351d333...` / `a9733538...` |
| `cv1/train_ids.csv` / `val_ids.csv` / `test_ids.csv` | `8fa6fb84...` / `a9733538...` / `ba4cb870...` |

- **Ion trap.** The stored `spec_df` labels the 20,688 ion-trap spectra `Other`, which the configs
  list (20 ppm / 500 Da tolerance, as in v1.0.2). In a 2,000-spectrum sample, 84% of their peaks
  carry at most two decimals, so that tolerance is probably too tight for them. This is the
  same treatment v1.0.2's `Other` block had. If `spec_df` is rebuilt with the fixed parser, they
  become `IT` and every config needs `IT` in `inst_types` and in `inst_type_loss_tol`.
- ACE: 50.1% of rows; 2 values above `ace_max` 300 (max 355). Derived NCE p99 252, 0.55% above 300.

## Launch (Fir, needs MFA)

1. Copy `data/proc/spectraverse_1_0_3/` and `data/split/spectraverse_benchmark_1_0_3/cv{0,1}` to
   the Fir checkout under the same paths, and check the md5s above.
2. Per arm:
   `python scripts/setup_job_cc.py --frag_gnn_dp . --python_script_fp scripts/run_pl_model_fit.py --job_name spv103_<arm> --cv_config_dp config/masster/spectraverse_1_0_3/<arm> --cv_start 0 --cv_end 1 --cv_use_job_id --gpu_profile fir:h100-3g.40gb --time 12:00:00 --script_kwargs w=online --no_submit=True`,
   inspect, then `sbatch`. The v1.0.2 controls took 6.0 h for 40 epochs at batch 512
   (`ggifzj9m`, `u46v0k0g`). On MSGYM, batch-256 epochs ran 1.0–1.5× slower than batch-512
   epochs, so `reg` and `reg_pow075` should take about 6–9 h and `reg_e20` about 3–4.5 h. These
   are estimates, not measurements.
3. Record job IDs and W&B runs here.

## How to read it

- Wall clock first, then best-epoch `val_cos_sim_0.01_epoch/mean` and its epoch, then the
  last-5-epoch tail. Report `val_cos_hun_inst_tol` too.
- Noise: NIST20 seed spread is 0.0042. On MSGYM, same-seed reruns differed by up to 0.008. Both
  folds must move the same way.
- Peak epoch: does regularization move it past epoch 15? Does `reg_e20` match `reg` at half the
  time?
- `reg_pow075` targets retrieval-style small-peak weighting. A flat val cosine is the MSGYM
  outcome, not a failure. No Spectraverse retrieval benchmark is set up here.

## Validation

- All eight configs resolve through `load_config` and pass `validate_run_config` (W&B online).
- Real-data smoke 2026-10-19 (RTX 2070S, W&B disabled): `reg_pow075/cv0.yml` with a 2,000-spectrum
  subsample, 1 epoch, batch 32. It loads v1.0.3, builds (36.9 MB of parameters), trains 63 steps on
  the CUDA candidate backend and validates; exit 0. Log: `output/spv103_smoke/reg_pow075_cv0.log`.
  This was the only arm smoked; the others differ from it only in scalar keys checked by the diff.
- Assumed: no train spectra are filtered out, so steps per epoch = ceil(split size / batch). Every
  adduct, instrument label and fragmentation mode in v1.0.3 is in the config vocabularies.
- Not done: GPU memory at batch 256 on `3g.40gb`. Batch 256 halves the per-step peak relative to
  the batch-512 control, which ran on the same profile.

## Next

Copy the data to Fir, launch `c0` and `reg` first. Launch `reg_pow075` and `reg_e20` alongside if
the allocation allows.
