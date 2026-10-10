# MASSter RT and CCS prediction heads

> **Status:** PREPARED · **Updated:** 2026-10-10
> Separate RTHead and CCSHead with independent labels; full Enveda-180 multitask arm remains prepared.

## Objective and acceptance

User decision: predict retention time (RT) and collision cross section (CCS) with additional
MASSter heads, jointly with MS/MS. Preserve the standalone Enveda baseline and its existing
molecule-disjoint split. Acceptance: labels join by source identity, missing labels produce
no regression gradient, both heads train the precursor encoder, RT/CCS predictions are
independent of collision energy at evaluation, and disabled heads preserve MS/MS weights
and predictions. A trained model's usefulness still needs held-out evaluation.

## Labels and evidence

The [Enveda dataset finding](../../findings/enveda_180_dataset_2026_09_25.md) describes the
source and split. The imported CSV retains `rtinseconds` and `ccs`, but step 2's column
selection drops both from the existing processed table. The optional label loader reads
only these scalars and `dset_spec_id` from the CSV and joins by that ID. It checks source
uniqueness and label-ID uniqueness. Each label set may cover different rows; unmatched
labels become NaN without dropping spectra. Neither the existing
processed artifacts nor split membership is rewritten. This CSV join supports one source;
combined-source datasets require source-qualified label identities first.

Read-only census on 2026-10-08: all 1,158,293 RT values are positive, while 42,401 CCS
values are invalid/nonpositive. Valid CCS labels number 1,115,892. RT spans 19.35–331.49 s;
valid CCS spans 120.76–301.13 in source units. All rows say HPLC/C18; that alone does not
establish identical gradients or calibration. RT is initially specific to this acquisition
setting. CCS unit/calibration provenance needs confirmation before cross-dataset comparison.

Among repeated source-SMILES/adduct groups, RT max-minus-min is 0.14 s at the median and
0.73 s at p99; CCS ranges are zero through p99 (maximum 0.24). These describe repeated
labels across source acquisitions, not an independent experimental error floor.

Canonical census: `output/enveda_rt_ccs_2026_10_08/labels.json`, generated with
[audit_rt_ccs_labels.py](../../../scripts/debug/enveda/audit_rt_ccs_labels.py):

```bash
/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/debug/enveda/audit_rt_ccs_labels.py \
  --csv data/df/enveda_180_df.csv --output output/enveda_rt_ccs_2026_10_08/labels.json
```

The script refuses to overwrite existing evidence; choose a new filename for a rerun.

## Implementation and constraints

User correction (2026-10-10): RT and CCS are **two separate heads and two label sets**.
This supersedes the earlier `RTandCCSHead` wrapper name.

- [RTHead and CCSHead](../../../src/fragnnet/model/rt_ccs.py) receive mean-pooled shared precursor atom
  embeddings. RT uses that vector; CCS concatenates a dedicated precursor-adduct embedding.
  Each two-layer MLP ends in softplus. `pred_rt` is seconds; `pred_ccs` retains source units.
  Both have one value per spectrum and are available through forward/inference and the
  `pred_rt` / `pred_ccs` pickle and tabular/HDF5 export tokens.
- Enable `masster_predict_rt` / `spec_params.rt` and `masster_predict_ccs` /
  `spec_params.ccs` independently. `spec_params.rt_labels_fp` reads only `dset_spec_id` and
  `rtinseconds`; `spec_params.ccs_labels_fp` reads only `dset_spec_id` and `ccs`. The files
  can contain disjoint sets of IDs. Both Enveda configs use separate columns from the same
  existing CSV, without duplicating or rewriting it. Without a sidecar, an enabled label
  set requires its own column in the processed table. Disable the label switches for
  prediction-only inputs; neither head needs labels at inference.
- Each enabled head has its own target tensor (`spec_rt` / `spec_ccs`), validity mask,
  loss weight, scale, and metrics. RT-only training does not instantiate or require CCS,
  and vice versa. The deprecated combined switches remain supported for old configs.
  Old wrapper checkpoint keys and optimizer parameter ordering migrate on load; existing
  checkpoint files are never rewritten.
- Context-in-GNN, encoder context tokens, merged spectra, and proposal-only warmup are
  rejected when heads are enabled. This first arm uses the existing context-free encoder
  and unmerged Enveda setup. Labels never enter the encoder or the spectrum context.
- Only finite positive labels are valid; missing values and the CCS `-1` sentinel are
  masked independently. The objective adds per-spectrum Smooth-L1 terms on
  `(prediction - target) / scale`, with beta 1, to the existing spectral/proposal loss.
  Starter scales are 60 seconds and 100 CCS source units; both weights are 0.1. These are
  fixed unit scales and provisional weights, not statistics fitted on validation/test data.
- The existing spectrum-weighted training reduction remains: invalid labels add zero;
  they do not cause per-batch renormalization. Repeated CE rows therefore still contribute
  repeated property supervision. No label deduplication or held-out target aggregation is
  performed. The InChIKey split keeps repeated molecules in the same partition.
- Epoch `rt_mae`, `ccs_mae`, `ccs_mape` (percent), and auxiliary losses exclude invalid
  labels from their weighted denominators. `rt_label_fraction` / `ccs_label_fraction`
  report coverage across all rows. An entirely unlabelled epoch has undefined/NaN property
  error, rather than a perfect score. Stored per-row errors use zero at missing labels;
  filter them using the corresponding label-fraction vector in per-datapoint analyses.
- Heads default off, add no disabled parameters, and are initialized after existing model
  branches with preserved RNG state. Legacy checkpoint migration explicitly defaults them
  off. Fresh configs must declare the new hparams through the shared template.

## Prepared experiment and next checks

[enveda_180_rt_ccs.yml](../../../config/masster/enveda/enveda_180_rt_ccs.yml) copies the
standalone ACE-only, no-EMA baseline and enables both heads and the label join. The MS/MS
baseline config is unchanged. The small local debug run below completed; the full arm has not run.

Before a full experiment, confirm CCS units/calibration and retain the same split, seeds, and spectrum
filters in the control. Compare RT/CCS errors to train-only constant baselines (RT globally,
CCS by adduct), and compare MS/MS cosine to the matched control. Read errors by adduct and
also aggregate by molecule/adduct to expose replication weighting. Tune weights on validation
only; reserve test for the chosen recipe. RT/CCS usefulness and any MS/MS benefit remain
unmeasured at full scale. GPU execution passed in the debug run; a full matched evaluation remains outstanding.

## Validation

Focused tests cover label identity and invalid values, dataset collation, analytical loss
values/gradients, shared-encoder gradients, CE independence, adduct conditioning, joint loss,
valid-label epoch denominators, unlabeled inference, prediction exports, strict checkpoint
restoration, independent RT-only/CCS-only training, disjoint label sets, legacy optimizer
slot alignment, and unchanged MS/MS initialization/outputs: **17 passed** on CPU in
`FRAGNNET-GPU`. Ruff and local documentation-link / `git diff --check` validation passed.
The real processed-table join also passed over all 1,158,293 rows: source IDs and order
were unchanged, with 1,158,293 valid RT and 1,115,892 valid CCS labels recovered.

Before the split-head correction, the broader MASSter model/Lightning/dataset, config, and inference regression selection gave
**474 passed, 5 skipped, 17 failed**. All 17 failures reproduced with identical exception
messages against unchanged revision `b9440051` in an isolated `/tmp` source copy: seven
existing tests enable sqrt intensity targets without `masster_merge_same_mz_rows`, and ten
runner tests omit `merge_train_only`. They are unrelated to RT/CCS and were left unchanged.
Logs are `output/enveda_rt_ccs_2026_10_08/{new_tests,regression_tests,baseline_failures}.log`.
The full repository suite was not run. CUDA was not visible inside the sandbox during the
initial unit-test pass; the subsequent local debug run used the GPU outside the sandbox.


## Local GPU debug training — completed 2026-10-10

The first run used the original combined wrapper (containing two regressors); its artifacts
remain unchanged. User requested a debug config and a trained model. The
[debug config](../../../config/masster/enveda/enveda_180_rt_ccs_debug.yml) uses the existing
InChIKey split, sampling 256 spectra per partition with seed 42 before adduct filtering:
**247 train and 247 validation spectra** remain. RT labels are valid on every row; CCS
labels are valid on 237 train and 235 validation rows. Test evaluation is disabled.

The debug model uses GINE_GSA width 64, two precursor layers, a 64-wide intensity transformer,
32 candidate fragments from a 128-mask budget, batch 16, and fixed AdamW learning rate 0.001.
RT and CCS retain the full arm's 0.1 loss weights and 60/100 scales. It trained jointly
with MS/MS for **10 epochs / 160 optimizer steps**, seed 397623, on the idle local
**RTX 2070 SUPER (8 GB)**: GPU neural network, CPU candidate construction, FP32,
OMP/MKL/OpenBLAS threads 4, no scheduler, W&B disabled. CUDA candidate kernels were not tested.
The run finished with exit code 0 and finite parameters; measured log interval was
20:27:08–20:28:02 UTC, including dataset setup. Deterministic algorithms were not enabled.

| Validation metric | Epoch 0 | Selected epoch 8 | Final epoch 9 |
| --- | ---: | ---: | ---: |
| RT MAE (s) | 80.2327 | 37.4279 | 41.3931 |
| CCS MAE (source units) | 41.0389 | 11.9031 | 9.8066 |
| CCS MAPE (%) | 22.6895 | 6.4688 | 5.3955 |
| MS/MS cosine, 0.01 Da | 0.1287 | 0.3780 | 0.3395 |

Epoch 8 has the lowest joint validation loss (-1.30294), the configured selection metric;
it is not selected independently for each property. Epoch 9 has better CCS error but worse
RT/MS/MS metrics. These are tiny-subset debug results, not full-dataset performance or evidence
of improvement over the MS/MS-only baseline. CCS is reported in source units without claiming
cross-dataset calibration. The full training/test protocol above remains pending.

- Job: `enveda_rt_ccs_debug_s397623_20261010a`, local run, no W&B ID.
- Selected checkpoint (epoch 8, step 144):
  `saved_ckpts/enveda_rt_ccs_debug_s397623_20261010a_local_2026_10_10/ckpt/model-epoch=008.ckpt`.
- Final checkpoint (epoch 9, step 160): `last.ckpt` in the same promoted directory.
- `provenance.json` beside `ckpt/` records checkpoint hashes, resolved hparams, revision
  `b9440051`, dirty patch identity, environment, source/split hashes, seeds and exact command.
- Evidence: `output/enveda_rt_ccs_2026_10_08/debug_2026_10_10a/` contains `launch.json`,
  `resolved_config.yml`, `working_tree.patch`, `train.log`, `completion.json`,
  `learning_curves.json`, and `selected_checkpoint.json`.
- The selected checkpoint originally reported class `RTandCCSHead`. It now restores with
  `strict=True` into `RTHead` and `CCSHead`; the new verification report is
  `output/enveda_rt_ccs_2026_10_08/debug_2026_10_10a/separate_heads_reload.json`.
  Curves and reload checks are regenerated by
  [summarize_rt_ccs_debug.py](../../../scripts/debug/enveda/summarize_rt_ccs_debug.py)
  (`--checkpoint PATH --output NEW_JSON_PATH`). Ruff and config resolution passed.

Training entry point (use a new job ID for another run to avoid automatic resume):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MPLCONFIGDIR=/tmp/fragnnet-mpl \
  /home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/run_pl_model_fit.py \
  -c config/masster/enveda/enveda_180_rt_ccs_debug.yml -w disabled \
  -j enveda_rt_ccs_debug_s397623_20261010a
```

## Separate-head debug rerun — completed 2026-10-10

The corrected architecture was trained from scratch with independent `RTHead` and `CCSHead`
modules and independent label switches/paths. The debug config enables both; each loader
selects its own target column from the existing Enveda CSV. The subset, seed, loss scales,
weights, GPU, and ten-epoch schedule match the first debug run above. Initial head weights
change because the two modules now initialize in their own registration order; this is not
a controlled accuracy comparison with the first run.

Run `enveda_rt_ccs_separate_debug_s397623_20261010b` completed with exit code 0 at 160
optimizer steps. The lowest joint validation loss selected epoch 8 (144 steps): RT MAE
**40.76 s**, CCS MAE **10.67 source units**, CCS MAPE **6.08%**, MS/MS cosine at 0.01 Da
**0.3536**. These are debug-subset measurements, not full-dataset benchmark results.

- Selected checkpoint:
  `saved_ckpts/enveda_rt_ccs_separate_debug_s397623_20261010b_local_2026_10_10/ckpt/model-epoch=008.ckpt`.
  The same directory retains `last.ckpt` and a parent `provenance.json` with hashes,
  launch command, input identities, dirty-diff identity, and resolved configuration.
- Evidence: `output/enveda_rt_ccs_2026_10_08/debug_separate_2026_10_10b/` contains
  `launch.json`, `resolved_config.yml`, `working_tree.patch`, `train.log`, `completion.json`,
  `selected_checkpoint.json`, `learning_curves.json`, and `tests.log`. The checkpoint
  reports come from [summarize_rt_ccs_debug.py](../../../scripts/debug/enveda/summarize_rt_ccs_debug.py)
  and verify strict restoration and finite weights for both heads.
- After the correction, the focused model, dataset, scalar-export, and property-head
  selection passed **183 tests, 5 skipped**. This includes 17 property-specific cases.
  Ruff and `git diff --check` passed. The full repository suite was not rerun.
