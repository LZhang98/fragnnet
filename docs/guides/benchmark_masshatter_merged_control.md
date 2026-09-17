# Merged MASShatter arms on the CPU/CUDA candidate backend

> **Status:** GUIDE · **Updated:** 2026-09-17 · Configs written and resolved; merged dataset and
> CUDA model construction verified locally. No training run was launched and no cluster job was
> generated or submitted from this workstation.

The six configs under [`config/masshatter/ma_mi_merged/`](../../config/masshatter/ma_mi_merged/)
are merged arms on the multi-adduct split, at three batch sizes and both candidate backends.
They are real training arms, not benchmark-harness windows; for fixed-window throughput
measurement use [benchmark_masshatter_cpu.md](benchmark_masshatter_cpu.md) instead.

## Scope: this is not a controlled merged-vs-unmerged A/B

Each config is derived from an unmerged base in
[`config/masshatter/ma_mi_unmerged/cmf/`](../../config/masshatter/ma_mi_unmerged/cmf/), but
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
across every config under `config/masshatter/`:

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

## Generating the jobs

Follow the [cluster prerequisites](submit_cluster_runs.md) and generate from a checkout that
contains the CUDA Python modules and `.cu` sources. Twelve hours matches the allocation used by
the unmerged CMF arms; confirm it against a measured epoch time before relying on it.

```bash
FRAGNNET_ACCOUNT=aip-rgreiner
FRAGNNET_EMAIL=your_email@example.org
for backend in "" _cuda; do
  for batch in "" _batch256 _batch512; do
    name="nist20_ma_mi_merged_fast${batch}${backend}"
    python scripts/setup_job_cc.py \
      --frag_gnn_dp "$PWD" --python_script_fp scripts/run_training.py \
      --job_name "$name" --gpu_profile vulcan:l40s \
      --cc_account "$FRAGNNET_ACCOUNT" --cc_email "$FRAGNNET_EMAIL" \
      --time 12:00:00 --no_submit=True \
      --script_kwargs c="config/masshatter/ma_mi_merged/${name}.yml" j="$name"
  done
done
```

Inspect the generated scripts, then submit explicitly. `j=` gives `last.ckpt` resume, which the
630-epoch budget needs across multiple twelve-hour allocations. Record the resolved backend from
the `mass_shatter_candidate_backend_resolved` W&B summary field on every run and on every resume,
because saved run config can overwrite local settings.

## What was verified

- All six configs resolve through `load_config` against `config/template.yml`, and the resolved
  diff against each unmerged base is exactly the table above.
- The merged dataset constructs on real NIST20 data.
- `tolerance_consensus` merging of whole multi-spectrum groups produces consensus peak lists and
  collapses rows as measured.
- The CUDA config builds a MASShatter model whose resolved `candidate_backend` is `cuda`, which
  exercises the supported-mode validation.

Not verified: any training step, GPU memory at batch 512, epoch wall time, native CUDA
compilation on L40S, and every scientific claim about a merged-vs-unmerged difference.
