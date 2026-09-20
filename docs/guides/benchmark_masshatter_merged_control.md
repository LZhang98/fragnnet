# Merged MASShatter arms on the CPU/CUDA candidate backend

> **Status:** GUIDE · **Updated:** 2026-09-17 · Configs written and resolved; merged dataset and
> CUDA model construction verified locally. No training run was launched and no cluster job was
> generated or submitted from this workstation.

The six configs under [`config/massshatter/ma_mi_merged/`](../../config/massshatter/ma_mi_merged/)
are merged arms on the multi-adduct split, at three batch sizes and both candidate backends.
They are real training arms, not benchmark-harness windows; for fixed-window throughput
measurement use [benchmark_masshatter_cpu.md](benchmark_masshatter_cpu.md) instead.

## Scope: this is not a controlled merged-vs-unmerged A/B

Each config is derived from an unmerged base in
[`config/massshatter/ma_mi_unmerged/cmf/`](../../config/massshatter/ma_mi_unmerged/cmf/), but
**merging is deliberately confounded with two other changes**, by decision of the user:

| Key | Unmerged base | Merged arm | Why it differs |
| --- | --- | --- | --- |
| `spec_params.merge` | `False` | `True` | the treatment under study |
| `spec_params.merge_keep_ces` | `False` | `True` | required once merging is on |
| `spec_params.merge_method` | `exact_sum` | `tolerance_consensus` | defines the merged target |
| `spec_params.merge_normalize_each_spectrum` | `False` | `True` | defines the merged target |
| `spec_params.merge_equal_ce_weight` | `False` | `True` | defines the merged target |
| **`spec_params.ace`** | **`True`** | **`False`** | **confound: NCE only** |
| **`mass_shatter_include_cmf`** | **`True`** | **`False`** | **confound: CMF channel off** |
| `min_epochs` / `max_epochs` | 60 / 100 | 378 / 630 | exposure matching, see below |

A difference between these arms and their unmerged bases therefore cannot be attributed to
merging alone. **To isolate merging, build unmerged siblings with `ace: false` and
`mass_shatter_include_cmf: false` and compare against those instead.** That pair does not exist
yet.

Split, adducts, instruments, fragmentation modes, candidate budgets (U256/E4096), encoder,
intensity head, losses, optimizer, precision, seed and batch settings are unchanged.
`ace_max: 300` is retained but inert while `ace: false`: `use_ace` comes straight from
`spec_params["ace"]`, so no ACE embedder is constructed. Turning ACE off and CMF off reduces the
model from 8,434,087 to 8,427,349 trainable parameters.

## Why CMF and dual CE are off

No merged MASShatter config has ever had a live CMF channel together with dual CE. Surveyed
across every config under `config/massshatter/`:

| Merged family | Split | CMF adducts present in data | dual CE |
| --- | --- | --- | --- |
| most of `merged/*` | `nist20_inchikey` | none (`[M+H]+`/FT only) | no |
| `context_ablation`, `smarts` | `nist20_inchikey` | none — Na/Cl declared in `prec_types` but absent from the split | yes |
| `unique_fragment/nist20_ma_mi_merged_fast_gnn_proposal*` | `nist20_ma_mi_inchikey` | present, but `mass_shatter_include_cmf: False` | no |
| `subformula_filter/…_dense_subformula` | `nist20_ma_mi_inchikey` | present | no (`ace: False`) |

Keeping both on would have made these the first merged arms to train a CMF channel *and* the
first to use dual CE, with no merged precedent for either. They instead stay inside the merged
family's established settings.

Turning CMF off removes the charged-molecular-formula channel, not the adducts: all four stay in
`prec_types`, and merging does not strip them. Measured on the merged train split, 5,993
`[M+Na]+` and 748 `[M+Cl]-` groups of 45,639 (14.8%), with no group mixing adducts, instruments
or fragmentation modes, so each merged row keeps a single well-defined adduct.

## CE conditioning under merging

A merged row carries the whole acquisition ladder rather than one energy: median 4 readings,
max 86. MASShatter's context encoder embeds each reading and **mean-pools per spectrum**
(`_embed_ce_values` in `src/fragnnet/model/mass_shatter_model.py`). So the merged arms condition
on a pooled NCE ladder. This is inherent to merging, not a config choice, and it is a further
reason the merged and unmerged arms are not a clean A/B on merging alone.

## Why 630 epochs and not 100

Merging groups spectra by acquisition `group_id`. Measured on `data/split/nist20_ma_mi_inchikey`
with these `spec_params` filters:

| Fold | Unmerged rows | Merged rows | Ratio | Molecules |
| --- | --- | --- | --- | --- |
| train | 287,721 | 45,639 | 6.304 | 15,514 |
| val | 96,153 | 15,257 | 6.302 | 5,172 |

An epoch means different amounts of data in the two families, so equal epochs would add training
exposure to the list of confounds. Scaling by 6.304 gives 378/630 against the base's 60/100:
630 x 45,639 = 28.75M training rows against 100 x 287,721 = 28.77M, a 0.07% difference. Per-epoch
validation cost scales down by the same factor, so total validation compute matches too.

The merged arms do pay 6.3x more *fixed* per-epoch overhead (dataloader worker respawn, epoch
bookkeeping, checkpoint writes) because they run 630 epochs instead of 100. That overhead is not
measured here. The merge itself runs once: `runner.py` builds both dataloaders before
`trainer.fit`, so `reload_dataloaders_every_n_epochs=1` reuses the same already-merged dataset.

To run an epoch-matched variant instead, override at launch rather than editing the config:

```bash
-o min_epochs=60 -o max_epochs=100
```

## Validation cosine is not comparable across families

`checkpoint_metric` is `val_cos_sim_0.01_epoch/mean` everywhere, but a merged arm scores against
consensus targets and an unmerged arm against individual spectra. Those are different targets, so
the W&B curves must not be read on one scale. Compare selected checkpoints on a single common
evaluation set at inference.

## The configs

| Config | Batch x accumulation | Candidate backend |
| --- | --- | --- |
| `nist20_ma_mi_merged_fast.yml` | 128 x 2 | cpu |
| `nist20_ma_mi_merged_fast_cuda.yml` | 128 x 2 | cuda |
| `nist20_ma_mi_merged_fast_batch256.yml` | 256 x 1 | cpu |
| `nist20_ma_mi_merged_fast_batch256_cuda.yml` | 256 x 1 | cuda |
| `nist20_ma_mi_merged_fast_batch512.yml` | 512 x 1 | cpu |
| `nist20_ma_mi_merged_fast_batch512_cuda.yml` | 512 x 1 | cuda |

All six keep one proposal mixture, unique-fragment mode, zero probability floors, SMARTS off and
FP32, so the CUDA siblings satisfy the backend's supported-mode envelope described in
[benchmark_masshatter_cpu.md](benchmark_masshatter_cpu.md#select-the-optional-cuda-backend).
Batch 512 exceeds the local 8 GB GPU with either backend; its memory headroom is an L40S check.

## What fits in a one-hour job

Merged epochs are cheap per epoch, but the configs cap at 630 epochs, so a one-hour allocation is
a partial run, not a finished one. The arithmetic, from the measured L40S epoch intervals in the
[training-speed finding](../findings/masshatter_training_speed_2026_09_17.md#live-fast-run-follow-up-earlier-candidate-code-still-running-2026-09-17)
(33.52 / 32.07 / 31.85 min at 128/256/512, validation included, on the unmerged rows):

| Quantity | Unmerged | Merged | Ratio |
| --- | ---: | ---: | ---: |
| train rows per epoch | 287,721 | 45,639 | 6.30x |
| peaks per row (mean) | 31.9 | 51.9 | 0.61x |
| total peak work per epoch | — | — | 3.83x |

Row count falls 6.30x but peak work only 3.83x, because a consensus row carries the whole
ladder's peaks. Candidate construction (about 46% of batch time at 128) is strictly per-row and
scales with the 6.30x; the loss and peak-matching scale with the 3.83x. So a merged epoch is
**roughly 5.3-6.5 min at batch 128**, bounded below by 33.52/6.30 = 5.32 and above by
33.52/3.83 = 8.75. This is an estimate from row and peak counts, **not a measured merged epoch
time**; measure one before planning a long campaign.

Allowing 5-10 min for the dataframe load and preprocessing, one hour yields roughly **8-10
epochs**:

| Goal | Epochs | One-hour jobs per arm | Six arms |
| --- | ---: | ---: | ---: |
| backend throughput/memory comparison | 8-10 | 1 | 6 |
| epoch-matched to the merged family's precedent | 100 | 10-13 | 60-78 |
| this config's `min_epochs` | 378 | 38-47 | 228-282 |
| this config's `max_epochs` (exposure-matched) | 630 | 63-79 | 378-474 |

**One hour per arm is the right size for comparing the CPU and CUDA candidate backends** — the
throughput, peak memory and candidate-count comparison needs steps, not convergence. It is not
enough to train a merged model to compare against anything scientifically. The epoch budget in
the config is a cap: a one-hour job simply stops at the wall clock and writes `last.ckpt`, and
`min_epochs` does not extend a run past its allocation.

## Generating the jobs

Follow the [cluster prerequisites](submit_cluster_runs.md) and generate from a checkout that
contains the CUDA Python modules and `.cu` sources.

Run this on the Vulcan login node, from the cluster checkout, with the cluster's `python`
(not `FRAGNNET_PYTHON`). It generates only; nothing is submitted.

```bash
FRAGNNET_ACCOUNT=aip-rgreiner
FRAGNNET_EMAIL=your_email@example.org
for backend in "" _cuda; do
  for batch in "" _batch256 _batch512; do
    name="nist20_ma_mi_merged_fast${batch}${backend}"
    python scripts/setup_job_cc.py \
      --frag_gnn_dp . \
      --python_script_fp scripts/run_pl_model_fit.py \
      --job_name "$name" \
      --gpu_profile vulcan:l40s \
      --cc_account "$FRAGNNET_ACCOUNT" \
      --cc_email "$FRAGNNET_EMAIL" \
      --time 1:00:00 \
      --script_kwargs w=online c="config/massshatter/ma_mi_merged/${name}.yml" j="$name" \
      --no_submit=True
    bash -n "slurm_scripts/${name}.sh"
  done
done
```

`w=online` is required for `j=` to work at all: the resume bookkeeping lives inside the W&B
branch, so with `w=disabled` there is no `job_id/<name>.id` and no `last.ckpt` resume.
Inspect the generated scripts, then submit explicitly. Record the resolved backend from the
`mass_shatter_candidate_backend_resolved` W&B summary field on every run and on every resume,
because saved run config can overwrite local settings. Compare `_cuda` against its CPU sibling at
the same batch size, and verify candidate counts and loss trajectories agree before reading any
throughput number.

To continue an arm past one hour, resubmit the same script: `j=<name>` resumes from `last.ckpt`.
Chain without babysitting using a scheduler dependency:

```bash
JOBID=$(sbatch --parsable slurm_scripts/nist20_ma_mi_merged_fast_cuda.sh)
for _ in $(seq 1 12); do
  JOBID=$(sbatch --parsable --dependency=afterany:"$JOBID" \
    slurm_scripts/nist20_ma_mi_merged_fast_cuda.sh)
done
```

`afterany` continues through wall-clock termination, which is the normal exit for a capped job.
Two things to confirm before trusting a chain: that each link logs `resuming from checkpoint`
rather than restarting at epoch 0, and that `ckpt_dp` (`wandb/run-*/files/ckpt` in the checkout)
actually holds `last.ckpt` between jobs. `runner.py` deletes `job_id/<name>.id` on clean
completion, so a chain that reaches `max_epochs` will start the next link from scratch.

## What was verified

- All six configs resolve through `load_config` against `config/*_template.yml`, and the resolved
  diff against each unmerged base is exactly the table above.
- The merged dataset constructs on real NIST20 data.
- `tolerance_consensus` merging of whole multi-spectrum groups produces consensus peak lists and
  collapses rows as measured.
- The CUDA config builds a MASShatter model whose resolved `candidate_backend` is `cuda`, which
  exercises the supported-mode validation.

Not verified: any training step, GPU memory at batch 512, epoch wall time, native CUDA
compilation on L40S, and every scientific claim about a merged-vs-unmerged difference.
