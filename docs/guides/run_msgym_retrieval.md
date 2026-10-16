# MassSpecGym retrieval runs, end to end

> **Status:** GUIDE · **Updated:** 2026-10-16
> One command scores a MASSter checkpoint on MSGYM 1.5 retrieval, locally or on Fir/Vulcan.
> Verified locally on the formula val fold. Prepare reproduces the
> [2026-10-03 run's](../findings/masster/masster_msgym_formula_retrieval_full_2026_10_03.md)
> 20 shards exactly. Scoring reproduces its per-candidate values from the same predictions.
> No cluster job has been submitted yet.

## What it runs

[`scripts/ms2c/run_msgym_retrieval.py`](../../scripts/ms2c/run_msgym_retrieval.py) runs four
resumable stages for one checkpoint, one candidate track (formula or mass) and one fold (val or
test). Each finished shard leaves a marker, so re-running the same command resumes the run.

1. **prepare.** Runs `01_prepare_msgym_retrieval.py export` (`--subset all`) and `build-proc`
   for the fold, then writes query shards of 500. These are the protocol of the
   [full formula run](../plans/annotation_aware_retrieval_plan.md#full-formula-track-run-frozen-protocol-2026-10-02-before-any-test-fold-read):
   - **Queries.** The checkpoint config's benchmark split rows for the fold. For
     `ms_gym_benchmark_1_5`, that means simulation-challenge `[M+H]+` spectra with a CE.
   - **Candidate rows.** Each is the query's benchmark row, with only the molecule and ids
     replaced, so the model sees the query's CE and instrument.
   - **Truth.** Matched by 2D InChIKey of MSGYM's source SMILES, never by list position.
   - **Unloadable candidates.** Candidates RDKit cannot fragment are dropped and listed in
     `excluded_candidates.csv` (val formula: one perchlorate salt, three rows). The run stops if
     one of them is a truth.
2. **infer.** `run_save_inference_eval.py` exports the full predicted distribution
   (`--min_peak_int -1`) with in-model `cos_sim_0.01`, `cos_sim_sqrt_0.01` and `cos_hun`.
3. **score.** `score_retrieval_candidates.py` computes per-candidate scores:
   - `loglik`: the frozen query log-likelihood.
   - `loglik_cal`: the calibrated rule `ll_g0.75_t2_e1e-4_a0.5`.
   - `mspred_ent`: ms-pred's binned entropy.
   - The three in-model scores are copied over.
4. **summarize.** `summarize_retrieval.py` reports expected hit@1/5/20 with molecule-bootstrap
   intervals, overall and per instrument.

`loglik_cal`'s parameters were val-selected for `ll04i00m`. For another checkpoint, re-select
them on val before reading test (`--cal_*` flags of the scoring CLI).

## Local run

```bash
FRAGNNET_PYTHON=/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python
"$FRAGNNET_PYTHON" scripts/ms2c/run_msgym_retrieval.py \
  --custom_fp config/masster/massspecgym/msgym_benchmark_1_5_unmerged_dual_ce_fast_batch512_cuda_glacier_feats.yml \
  --model_ckpt output/annotation_retrieval_pilot_2026_10_02/ckpt/ll04i00m_ep060_orb.ckpt \
  --tsv_fp data/raw/msgym_1_5/MassSpecGym1.5.tsv \
  --candidates_fp data/raw/msgym_1_5/MassSpecGym1.5_retrieval_candidates_formula.json \
  --fold val --run_dp output/msgym_retrieval/formula_val
```

- `--custom_fp` is the checkpoint's training config; its `spec_fp` and `split_dp` define the
  queries.
- `--stages` and `--shards` restrict a run, for example `--stages prepare infer score --shards 0`
  for a smoke test.
- `--keep_predictions false` deletes each shard's HDF5 once it is scored.

The run directory holds the checkpoint-independent shards. A second checkpoint pointed at the same
`--run_dp` reuses them and writes `shard_NNN/<ckpt stem>/` and `summary_<ckpt stem>.csv`.

## Fir and Vulcan

From the cluster checkout's login node:

```bash
bash scripts/manual_slurm/prepare_msgym_retrieval.sh fir \
  config/masster/massspecgym/<training_config>.yml saved_ckpts/<run>/<ckpt>.ckpt formula test
sbatch slurm_scripts/msgym_ret_formula_test_<ckpt stem>.sh
```

- **Clusters.** `vulcan` in place of `fir` uses the `vulcan:l40s` profile and `aip-rgreiner`;
  `fir` uses `fir:h100-3g.40gb` and `rrg-dwishart`.
- **Overrides.** `FRAGNNET_ACCOUNT`, `FRAGNNET_EMAIL`, `FRAGNNET_TIME` (default `8:00:00`) and
  `FRAGNNET_RETRIEVAL_DP`, the parent of the run directory.
- **Generation only.** The launcher wraps
  [`setup_job_cc.py`](submit_cluster_runs.md) and does not submit.
- **Staging.** The job stages the config, template, benchmark data and checkpoint onto the node.
  It reads the raw MSGYM TSV and JSON in place from the checkout.
- **Output.** It writes straight to the run directory: `${SCRATCH}/msgym_retrieval/<track>_<fold>`,
  or `output/msgym_retrieval/` if `SCRATCH` is unset.
- **Inputs.** The checkpoint path must be repo-relative. `data/raw/msgym_1_5/` must exist in the
  cluster checkout.
- **Resuming.** Re-submitting after a timeout resumes from the last finished shard.
- **Exit status.** The generated script ends with an `echo`, so the scheduler reports exit 0 even
  when the driver fails. Check the `.err` log and `summary_<ckpt stem>.csv`.

## GPU usage (2026-10-16)

Every inferred shard writes `shard_NNN/<label>/gpu_usage.json`: host, `SLURM_JOB_ID`, the
prediction loop's seconds and predictions per second, the device and its memory, the PyTorch
allocator peaks (`peak_allocated_gib`, `peak_reserved_gib`), and the driver's `infer_wall_s` /
`score_wall_s` (subprocess start-up included). After the infer/score loop the driver writes
`gpu_usage_<label>.csv` (one row per shard) and logs the maxima and totals. Shards inferred
before this change have no row.

- NVML reports no SM utilization on a MIG slice, so utilization is not recorded; size slices
  from `peak_allocated_gib` (what the model needs) and throughput. `peak_reserved_gib` includes
  allocator caching and grows to fill whatever the device has.
- Local check, `pow075` on formula test shard 0 (RTX 2070 SUPER, 8 GB): 65,406 predictions in
  166.5 s (393/s), `peak_allocated_gib` 1.22, `peak_reserved_gib` 7.23, `infer_wall_s` 223.3
  (57 s of start-up, checkpoint load and featurization outside the prediction loop).
- **Batch size (default 256 since 2026-10-16, was 64).** Same shard and GPU: batch 256 runs the
  loop in 108.8 s (601/s, 1.53x) at `peak_allocated_gib` 3.96; batch 512 ran out of memory on the
  8 GB card. Against batch 64, 0.77% of spectra differ (max 0.0094 cosine, mean 5e-6), the size of
  GPU run-to-run noise, so reads stay comparable with the batch-64 `pow075` / `f44tzxoc` runs.
  256 fits a 1g.10gb slice; set `FRAGNNET_EVAL_BATCH_SIZE=512` only on 2g.20gb or larger, and
  check the rewire arms' `peak_allocated_gib` first (not measured).
- Job-level CPU memory and elapsed time: `sacct -j <id> --format=JobID,JobName%50,Elapsed,State,MaxRSS,AllocTRES%60,TRESUsageInMax%80`.

## Cost (local, formula val)

| Step | Measured |
| --- | --- |
| export / build-proc / sharding | 4.5 / 6 / 1 min (val fold: 2.43M rows, 505k molecules) |
| inference | 3.9 min per 500-query shard (8 GB laptop GPU); 20 shards per fold |
| scoring | 15 s per shard |
| disk | `data/` 0.8 GB; predictions 0.4 GB per shard, about 8 GB per fold |

The mass track was not timed.

## Reproducibility

MASSter GPU inference is not bit-reproducible run to run. Two back-to-back runs on val shard 0
differ on 0.41% of candidates (max 0.010 cosine). Against the 2026-10-03 export, 0.39% differ
(max 0.021) and hit@1 is unchanged to four decimals (cosine 0.242, log-likelihood 0.422).
Compare runs by hit@k, not by per-candidate scores.

## Audit of the older ms2c code (2026-10-04)

- **Old Fir array scripts are stale.** `scripts/manual_slurm/fir/msgym_ms2c_{formula,mass}_pred_arr.sh`
  and `msgym_{formula,mass}_frag_gen_arr.sh` were written for FraGNNet. They call
  `scripts/ms2c/run_ms2c_predict_msgym.py` and
  `preproc_scripts/msgym_ms2c/02_prepare_mgsgym_prediction_frag.py`, neither of which exists, and
  read hard-coded `/scratch/feiw/...` tars. Do not use them; they were left in place.
- **The `scripts/ms2c/` predictor is FraGNNet-only.** `run_ms2c_predict_val.py` is for FraGNNet.
  The ICEBERG predictor was removed with the local ICEBERG fork on 2026-10-11; run ICEBERG through
  external ms-pred.
- **A stale README claim.** The ms2c inference README's `cos_sim_sqrt` auxiliary-score name is
  rejected by `run_save_inference_eval.py`, which raises `metric_name cos_sim_sqrt not
  recognized`. The sqrt cosine is enabled by `--override eval_bin_sqrt=True`.
- **The 2026-10-03 run lived in `output/`.** Its shard builder, runner, scoring and exclusion of
  unloadable molecules were ad hoc scripts under `output/msgym_formula_retrieval_2026_10_02/`.
  - The exclusion was a hand edit after an inference crash.
  - The calibrated rule lived in `output/msgym_retrieval_calibration_2026_10_03/`.
  - All of these are now in the driver and the scoring CLI.
- **Out of scope.** GLACIER inference (ms-pred environment) and the `[M+Na]+` / no-CE queries
  are not covered.

## Candidate-set files and sharded inference (from local memory, verified 2026-10-06)

`--fold`, `build-proc-large` and the `requires_dags` shard filter were re-checked in code.

- `data/ms2c/msgym/v1_5/retrieval_candidates_{formula,mass}_{test,val,train}` are built by
  `preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py` (`export`, then `build-proc`). The
  old `_all` suffix meant every test-fold spectrum (`--subset all`), never all folds.
- `build-proc` always writes the split as `test_ids.csv`, because `base_dataset.py` reads that
  name; the fold is in the directory name only.
- After the `(source_identifier, candidate_smiles)` dedup: formula test 17,556 queries /
  2,297,662 spectrum rows / 490,266 molecules; formula val 19,429 / 2,433,451 / 504,824; mass val
  19,429 / 4,640,577 / 796,564.
- Renaming a directory also means renaming its `.zip` and rewriting the zip's top-level path
  (e.g. `zipnote -w`).
- `*_train.zip` use `mol_df_shards/` plus `train_ids.csv` (20 shards formula, 28 mass;
  `build-proc-large`). Do not merge them: a live RDKit mol costs 10.9 KB of native heap, so merged
  train frames are 43.5 GB (formula, 3,987,596 molecules) and 59.2 GB (mass, 5,422,947),
  unwritable on the 31 GB workstation.
- `run_inference.py` predicts one shard directly: pass `--mol_fp .../part-000NN.pkl` with the
  whole `spec_df.pkl` (it filters to the shard's `mol_id`s in the `requires_dags` block), not
  `--start_mol_idx/--num_mols`. `--custom_fp` is required. Set `TMPDIR`, because DAGs go to a temp
  dir and local `/tmp` is small. Each shard re-reads the 2.2 GB (formula) or 2.8 GB (mass) `spec_df`.
- Export is single-threaded (about 6 min per fold) and `build-proc` is memory-hungry: run folds
  sequentially on a 31 GB box. `pandas.memory_usage(deep=True)` undercounts a live-mol column
  about 450x; measure RSS.
