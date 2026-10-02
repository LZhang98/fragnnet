# Spectraverse MASSter fixed ±0.05 Da training arm

> **Status:** PREPARED · **Updated:** 2026-10-01 · Ten fold configs and real-data two-spectrum CUDA loss smoke validated; full training not launched.

## Objective and acceptance

Train a matched SPV MASSter arm using a fixed absolute ±0.05 Da peak-match window. Keep the completed instrument-tolerance configs as historical controls. The arm uses the same folds, seeds, model, data, optimizer, and 40-epoch schedule as each control. A changed match window changes the primary training objective and the interpretation of recall; compare held-out spectral quality and instrument-tolerance coverage separately.

## Prepared implementation

Configs: `config/spectraverse/masster_abs005/cv0.yml` through `cv9.yml`. Each resolves `loss_tolerance_rel: null`, `loss_tolerance_abs: 0.05`, and `inst_type_loss_tol: null`. This makes the loss and standard recall/weighted-recall/precision metrics use ±0.05 Da for every instrument. The historical `config/spectraverse/masster/cv*.yml` files are unchanged. W&B names and group are distinct.

The previous `cos_hun_inst_tol` metric requires `inst_type_loss_tol` at model setup, so the new arm logs `cos_hun` and selects `val_cos_hun_epoch/mean` at ±0.05 Da. It still logs `cos_sim_0.01` for a tolerance-independent binned comparison. Rescore checkpoints separately at the original instrument tolerance for direct historical comparison. Loss values and wide-window recall values are not directly comparable to the control because target matches differ.

The checkpoint rescore on 512 validation spectra per fold found candidate recall/wRecall of 0.759/0.844 (cv0) and 0.714/0.846 (cv1) at ±0.05 Da; these are properties of the **old checkpoints under a wider scoring window**, not outcomes from retraining. See the [tolerance finding](../../findings/masster/spectraverse_recall_tolerance_2026_10_01.md).

## Validation and next check

All ten configs resolved. Against their corresponding controls, the resolved differences are exactly the three tolerance keys, W&B name/group/tags, auxiliary metric list, and checkpoint metric. A cv0 CPU model construction confirmed `tolerance=0.05`, `relative=False`, and no per-instrument lookup; a two-spectrum metric smoke matched a 0.04 Da offset and rejected a 0.06 Da offset. A real-data two-spectrum CUDA forward/backward smoke produced finite loss 3.7193 and 158 finite gradient tensors with no optimizer step. The configured interpreter passed 58 focused tolerance/config tests; `git diff --check` passed. No full validation or training run has occurred.

Before a full run, record source revision and dirty diff, environment, input/split identity, resolved config, seed, hardware, job ID, and artifact paths per [session workflow](../../workflow/session.md). Record launch provenance, then train and compare cv0/cv1 against their matched historical controls before committing the remaining folds. The training command for one local fold is `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/run_pl_model_fit.py -c config/spectraverse/masster_abs005/cv0.yml -w online`; cluster submission should follow the [cluster guide](../../guides/submit_cluster_runs.md). No training run or cluster job has been started by this preparation.
