# Controlled merged-vs-unmerged MASShatter arms on the GPU candidate backend

> **Status:** GUIDE · **Updated:** 2026-09-17 · Configs written and resolved; merged dataset and
> CUDA model construction verified locally. No training run was launched and no cluster job
> was generated or submitted from this workstation.

The six configs under [`config/masshatter/ma_mi_merged/cmf/`](../../config/masshatter/ma_mi_merged/cmf/)
are merged counterparts of the existing unmerged fast arms in
[`config/masshatter/ma_mi_unmerged/cmf/`](../../config/masshatter/ma_mi_unmerged/cmf/).
Together the two directories form a 2 (merge) x 3 (batch) x 2 (candidate backend) grid.
They are real training arms, not benchmark-harness windows; for fixed-window throughput
measurement use [benchmark_masshatter_cpu.md](benchmark_masshatter_cpu.md) instead.

## What is controlled

Each merged config differs from its unmerged base **only** by the merge settings, the epoch
budget, and the W&B name/tags. A resolved-config comparison through `load_config` shows exactly
these keys changing (plus `mass_shatter_candidate_backend` for the CUDA siblings):

| Key | Unmerged base | Merged arm |
| --- | --- | --- |
| `spec_params.merge` | `False` | `True` |
| `spec_params.merge_keep_ces` | `False` | `True` |
| `spec_params.merge_method` | `exact_sum` | `tolerance_consensus` |
| `spec_params.merge_normalize_each_spectrum` | `False` | `True` |
| `spec_params.merge_equal_ce_weight` | `False` | `True` |
| `min_epochs` / `max_epochs` | 60 / 100 | 378 / 630 |

`merge_tolerance_common_overlap` is already `True` in the template, and
`merge_tolerance_by_inst` is inherited from it as the FT/QTOF table that matches this config's
`inst_type_loss_tol`. The merge recipe is the one already used by the repository's other
multi-adduct merged arms, including
`config/masshatter/merged/unique_fragment/nist20_ma_mi_merged_fast_gnn_proposal.yml`.

Split, adducts, instruments, fragmentation modes, chemistry, candidate budgets (U256/E4096),
encoder, intensity head, losses, optimizer, precision, seed and batch settings are unchanged.
This is the like-for-like pair that the earlier merged-vs-unmerged numbers lacked: the
historical merged runs sat on `data/split/nist20_inchikey`, which is `[M+H]+`/FT only, so they
differed in task composition as well as in merging.

## Why 630 epochs and not 100

Merging groups spectra by acquisition `group_id`. Measured on `data/split/nist20_ma_mi_inchikey`
with this config's `spec_params` filters:

| Fold | Unmerged rows | Merged rows | Ratio | Molecules |
| --- | --- | --- | --- | --- |
| train | 287,721 | 45,639 | 6.304 | 15,514 |
| val | 96,153 | 15,257 | 6.302 | 5,172 |

An epoch therefore means different amounts of data in the two arms, so equal epochs would
confound merging with training exposure. Scaling by 6.304 gives 378/630 against the base's
60/100: 630 x 45,639 = 28.75M training rows against 100 x 287,721 = 28.77M, a 0.07% difference.
Per-epoch validation cost scales down by the same factor, so total validation compute is matched
as well.

The merged arm does pay 6.3x more *fixed* per-epoch overhead (dataloader worker respawn, epoch
bookkeeping, checkpoint writes) because it runs 630 epochs instead of 100. That overhead is not
measured here. The merge itself runs once: `runner.py` builds both dataloaders before
`trainer.fit`, so `reload_dataloaders_every_n_epochs=1` reuses the same already-merged dataset.

To run the epoch-matched variant instead, override at launch rather than editing the config:

```bash
-o min_epochs=60 -o max_epochs=100
```

That variant answers a different question (equal epochs, 6.3x less merged data) and is not the
controlled comparison.

## Validation cosine is not comparable across the pair

`checkpoint_metric` is `val_cos_sim_0.01_epoch/mean` in both arms, but the merged arm scores
against consensus targets and the unmerged arm against individual spectra. Those are different
targets, so the two W&B curves must not be read as one scale. Compare the two selected
checkpoints on a single common evaluation set at inference before claiming a merging effect.

## The configs

| Config | Batch x accumulation | Candidate backend |
| --- | --- | --- |
| `nist20_ma_mi_merged_cmf_dual_ce_fast.yml` | 128 x 2 | cpu |
| `nist20_ma_mi_merged_cmf_dual_ce_fast_cuda.yml` | 128 x 2 | cuda |
| `nist20_ma_mi_merged_cmf_dual_ce_fast_batch256.yml` | 256 x 1 | cpu |
| `nist20_ma_mi_merged_cmf_dual_ce_fast_batch256_cuda.yml` | 256 x 1 | cuda |
| `nist20_ma_mi_merged_cmf_dual_ce_fast_batch512.yml` | 512 x 1 | cpu |
| `nist20_ma_mi_merged_cmf_dual_ce_fast_batch512_cuda.yml` | 512 x 1 | cuda |

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
    name="nist20_ma_mi_merged_cmf_dual_ce_fast${batch}${backend}"
    python scripts/setup_job_cc.py \
      --frag_gnn_dp "$PWD" --python_script_fp scripts/run_training.py \
      --job_name "$name" --gpu_profile vulcan:l40s \
      --cc_account "$FRAGNNET_ACCOUNT" --cc_email "$FRAGNNET_EMAIL" \
      --time 12:00:00 --no_submit=True \
      --script_kwargs c="config/masshatter/ma_mi_merged/cmf/${name}.yml" j="$name"
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
- The merged dataset constructs on real NIST20 data with dual CE retained through the merge
  (`spec_ce`, `spec_nce` and `spec_ace` all populated).
- `tolerance_consensus` merging of whole multi-spectrum groups produces consensus peak lists
  and collapses rows as measured.
- The CUDA config builds a MASShatter model whose resolved `candidate_backend` is `cuda`,
  which exercises the supported-mode validation.

Not verified: any training step, GPU memory at batch 512, epoch wall time, native CUDA
compilation on L40S, and every scientific claim about the merged-vs-unmerged difference itself.
