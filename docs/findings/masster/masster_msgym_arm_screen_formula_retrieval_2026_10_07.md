# MSGYM arm screen (val) and test retrieval for `wd10` and `pow075`

> **Status:** FINDING · **Updated:** 2026-10-08
> The k1/EMA/Fourier screen ran on 2026-10-02/03 (18 checkpoints in W&B). On formula test retrieval,
> `pow075` beats `ll04i00m` on likelihood and entropy scoring (`mspred_ent` +0.031 hit@1, CI excludes
> zero); `wd10` is +0.003 to +0.012 with every CI crossing zero. One seed each.

## Val screen (W&B, all MSGYM runs since `ll04i00m`)

All runs use `data/split/ms_gym_benchmark_1_5`, 128 epochs (`e64`: 64), seed 397623 (`s2`:
20261003). Checkpointing keeps the top-1 epoch by `val_cos_sim_0.01_epoch/mean` (max), so "val best"
is the val score of the uploaded `model-epoch=NNN.ckpt`. The saved `state_dict` is the EMA average.
Arm names are suffixes of `msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_`.

| run | arm | val best | ckpt epoch | val last |
| --- | --- | ---: | ---: | ---: |
| `8afoui7e` | k1_ema999_fourier_wd10 | 0.5867 | 50 | 0.5805 |
| `3ki6nyy6` | k1_ema999_fourier_pm1 | 0.5859 | 40 | 0.5804 |
| `7wh3jaqw` | k1_ema999_fourier_mol3 | 0.5849 | 39 | 0.5776 |
| `1vvyc70p` | k1_ema999_fourier_bond1 | 0.5838 | 49 | 0.5766 |
| `ocx2i6h5` | k1_ema999_fourier_enc256 | 0.5835 | 37 | 0.5779 |
| `kb5by4di` | k1_ema999_fourier (control rerun) | 0.5833 | 46 | 0.5766 |
| `apm3ok8v` | k1_ema999_fourier_enc512 | 0.5831 | 51 | 0.5784 |
| `8szjefc2` | k1_ema999_fourier_pow075_pm1 | 0.5829 | 51 | 0.5774 |
| `01pkzb92` | k1_ema999_fourier_e64 | 0.5821 | 38 | 0.5800 |
| `2y8rl6fs` | k1_ema999_fourier_s2 | 0.5819 | 47 | 0.5766 |
| `dz7u8i6w` | k1_ema999_fourier (control rerun) | 0.5815 | 39 | 0.5764 |
| `57bavwzy` | k1_ema999_fourier (control) | 0.5809 | 62 | 0.5777 |
| `q3oxbpgx` | k1_ema999_fourier_pow075 | 0.5790 | 55 | 0.5728 |
| `cgivkl1j` | k1_ema999_fourier_itf1 | 0.5785 | 38 | 0.5737 |
| `p3i0tudq` | glacier_feats_fourier | 0.5770 | 84 | 0.5733 |
| `3s4o3l02` | k1_ema999_fourier_lr5e4 | 0.5736 | 47 | 0.5705 |
| `ymaktpzu` | k1_ema999_fourier_sqrt_pm1 | 0.5721 | 29 | 0.5648 |
| `0hmzfttj` | k1_ema999_fourier_sqrt | 0.5712 | 43 | 0.5642 |
| `ll04i00m` | glacier_feats (reference) | 0.5705 | 60 | 0.5669 |

- No checkpoint: `1p4s2w61` b256 (crashed at epoch 110, val best 0.5875), `zb7lhmn4`
  glacier_feats_fourier (crashed, rerun as `p3i0tudq`), `1imawkx1`.
- Control spread: three control runs plus `s2` span 0.5809–0.5833 (0.0025). Only `wd10` (+0.0049)
  and `pm1` (+0.0041) clear the control mean 0.5819 by more than that.
- Sweep `tiwhun9q` has no runs.
- Val best is a max over 128 epochs, so it is selection-biased upward.

## Formula-track test retrieval

Protocol: [run_msgym_retrieval](../../guides/run_msgym_retrieval.md), formula candidates, test
fold, 9,954 simulation-challenge `[M+H]+` queries in 20 shards (ORB 8,374, QTOF 1,580), no
excluded candidates. Expected hit@k under random ties. `loglik_cal` is the rule
`ll_g0.75_t2_e1e-4_a0.5`, val-selected for `ll04i00m` and **not re-selected** for the new
checkpoints. `ll04i00m` was re-run on the same runner and reproduces the
[2026-10-03 test numbers](masster_msgym_formula_retrieval_full_2026_10_03.md) to three decimals.

hit@1 (hit@5 / hit@20):

| score | `ll04i00m` @60 | `wd10` @50 | `pow075` @55 |
| --- | ---: | ---: | ---: |
| `cos_sim_0.01` | 0.358 (0.711 / 0.894) | 0.366 (0.725 / 0.897) | 0.362 (0.722 / 0.898) |
| `cos_hun` | 0.365 | 0.372 | 0.368 |
| `cos_sim_sqrt_0.01` | 0.455 (0.796 / 0.939) | 0.467 (0.806 / 0.936) | 0.467 (0.808 / 0.940) |
| `mspred_ent` | 0.485 (0.818 / 0.949) | 0.495 (0.826 / 0.946) | 0.516 (0.842 / 0.955) |
| `loglik` | 0.485 (0.815 / 0.948) | 0.496 (0.829 / 0.951) | 0.504 (0.836 / 0.957) |
| `loglik_cal` | 0.528 (0.853 / 0.965) | 0.532 (0.857 / 0.963) | 0.542 (0.862 / 0.970) |

`loglik_cal` hit@1 by instrument: ORB 0.591 (`wd10`), 0.603 (`pow075`); QTOF 0.217, 0.218.

Paired hit@1 differences, 95% molecule bootstrap (2,000 resamples):

| score | `wd10` − `ll04i00m` | `pow075` − `ll04i00m` | `pow075` − `wd10` |
| --- | --- | --- | --- |
| `cos_sim_0.01` | +0.008 [−0.005, +0.021] | +0.004 [−0.009, +0.017] | −0.004 [−0.015, +0.007] |
| `cos_sim_sqrt_0.01` | +0.012 [−0.000, +0.024] | +0.012 [−0.002, +0.027] | +0.000 [−0.013, +0.014] |
| `mspred_ent` | +0.010 [−0.003, +0.023] | **+0.031 [+0.017, +0.046]** | **+0.021 [+0.007, +0.036]** |
| `loglik` | +0.011 [−0.003, +0.025] | **+0.018 [+0.003, +0.033]** | +0.008 [−0.006, +0.021] |
| `loglik_cal` | +0.003 [−0.011, +0.018] | +0.014 [−0.002, +0.029] | +0.011 [−0.003, +0.025] |

## Mass-track test retrieval, `pow075` (2026-10-08)

Same runner, mass candidates, test fold: 9,954 queries, 2,466,243 candidates (about 248 per query;
one unloadable non-truth candidate excluded; 19 queries with two truth rows). No comparison model
has been run on this track. Expected top-k in %, random ties; top-1 95% molecule bootstrap.

| score | top-1 [95% CI] | top-3 | top-5 | top-20 | ORB top-1 (8,374) | QTOF top-1 (1,580) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `loglik` | 71.35 [67.97, 74.66] | 86.78 | 91.23 | 96.59 | 75.89 | 47.28 |
| `loglik_cal` | 74.52 [70.92, 77.93] | 89.42 | 93.54 | 97.98 | 79.51 | 48.10 |
| `cos_sim_0.01` | 52.08 [49.15, 54.95] | 72.16 | 78.29 | 88.38 | 56.22 | 30.16 |
| `mspred_ent` | 69.18 [65.63, 72.48] | 83.80 | 87.89 | 94.91 | 74.06 | 43.29 |

- Paired top-1: `loglik` − cosine +19.26 [+17.48, +21.06]; `loglik` − entropy +2.17 [+0.79, +3.43].
- The likelihood margin over cosine is larger than on the formula track (+14.1), and largest on
  QTOF, where cosine top-20 is 75.4 against 92.2 for `loglik`.
- GLACIER's paper reports mass top-1 69.95 (w/ CF) and 69.69 (w/o CF) on ms-pred's candidate list.
  That is unpaired, and our mass list has not been checked against ms-pred's (the formula list was).
- Artifacts: `output/msgym_retrieval/mass_test/` (`run_pow075.{sh,log}`,
  `summary_q3oxbpgx_pow075_ep055.csv`, `topk_masster_mass.py`, `topk_1_3_5_20_q3oxbpgx_pow075_ep055.csv`).

## Interpretation

1. **Val cosine does not rank checkpoints for retrieval.** `wd10` leads `pow075` on val
   `cos_sim_0.01` by 0.008 and ties it on test cosine retrieval, but `pow075` wins on every
   likelihood/entropy score.
2. **`pow075` gains where small peaks count.** Entropy and log-likelihood weight low-intensity
   peaks; cosine and sqrt cosine show no `pow075` advantage. A flatter training target is a
   plausible cause, untested.
3. **Single seed.** These are one checkpoint per arm; the val seed spread (0.0025) has no test
   retrieval counterpart yet.
4. **`loglik_cal` is under-tuned for the new checkpoints.** Re-selecting it on val per checkpoint
   may move both arms.

## Artifacts

All in `output/msgym_retrieval/formula_test/`:

- `run_chain.sh` and `run_chain.log`: the three runs.
- `summary_<ckpt stem>.csv`: per-checkpoint hit@k with intervals.
- `compare_wd10_pow075_ll04.csv`: paired comparisons (`scripts/eval/summarize_retrieval.py --compare`).
- `shard_NNN/<ckpt stem>/`: predictions and scores. All deleted: `wd10` and `ll04i00m` by
  2026-10-13; `pow075` (formula and mass test) on 2026-10-17. Regenerate from the checkpoint.
- Checkpoints: `saved_ckpts/msgym/8afoui7e_wd10_ep050.ckpt`, `saved_ckpts/msgym/q3oxbpgx_pow075_ep055.ckpt`.
- A parallel session wrote `compare_vs_glacier.py` and `compare_own_best.py` (with CSVs) in the
  same directory, comparing these checkpoints with GLACIER's released checkpoint; not part of
  this finding.
