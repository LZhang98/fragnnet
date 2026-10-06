# Spectraverse MASSter fixed ±0.05 Da training arm

> **Status:** PREPARED · **Updated:** 2026-10-06
> Ten fold configs and a real-data two-spectrum CUDA loss smoke validated; full training not launched.

## Objective and acceptance

Train a matched SPV MASSter arm using a fixed absolute ±0.05 Da peak-match window. Keep the completed instrument-tolerance configs as historical controls. The arm uses the same folds, seeds, model, data, optimizer, and 40-epoch schedule as each control. A changed match window changes the primary training objective and the interpretation of recall; compare held-out spectral quality and instrument-tolerance coverage separately.

## Prepared implementation

Configs: `config/spectraverse/masster_abs005/cv0.yml` through `cv9.yml`. Each resolves `loss_tolerance_rel: null`, `loss_tolerance_abs: 0.05`, and `inst_type_loss_tol: null`. This makes the loss and standard recall/weighted-recall/precision metrics use ±0.05 Da for every instrument. The historical `config/spectraverse/masster/cv*.yml` files are unchanged. W&B names and group are distinct.

The previous `cos_hun_inst_tol` metric requires `inst_type_loss_tol` at model setup, so the new arm logs `cos_hun` and selects `val_cos_hun_epoch/mean` at ±0.05 Da. It still logs `cos_sim_0.01` for a tolerance-independent binned comparison. Rescore checkpoints separately at the original instrument tolerance for direct historical comparison. Loss values and wide-window recall values are not directly comparable to the control because target matches differ.

The checkpoint rescore on 512 validation spectra per fold found candidate recall/wRecall of 0.759/0.844 (cv0) and 0.714/0.846 (cv1) at ±0.05 Da; these are properties of the **old checkpoints under a wider scoring window**, not outcomes from retraining. See the [tolerance finding](../../findings/masster/spectraverse_recall_tolerance_2026_10_01.md).

## Validation and next check

All ten configs resolved. Against their corresponding controls, the resolved differences are exactly the three tolerance keys, W&B name/group/tags, auxiliary metric list, and checkpoint metric. A cv0 CPU model construction confirmed `tolerance=0.05`, `relative=False`, and no per-instrument lookup; a two-spectrum metric smoke matched a 0.04 Da offset and rejected a 0.06 Da offset. A real-data two-spectrum CUDA forward/backward smoke produced finite loss 3.7193 and 158 finite gradient tensors with no optimizer step. The configured interpreter passed 58 focused tolerance/config tests; `git diff --check` passed. No full validation or training run has occurred.

Before a full run, record source revision and dirty diff, environment, input/split identity, resolved config, seed, hardware, job ID, and artifact paths per [session workflow](../../workflow/session.md). Record launch provenance, then train and compare cv0/cv1 against their matched historical controls before committing the remaining folds. The training command for one local fold is `/home/feiw/miniconda3/envs/FRAGNNET-GPU/bin/python scripts/run_pl_model_fit.py -c config/spectraverse/masster_abs005/cv0.yml -w online`; cluster submission should follow the [cluster guide](../../guides/submit_cluster_runs.md). No training run or cluster job has been started by this preparation.

### Control configs `config/spectraverse/masster/cv0-cv9.yml` (2026-09-25, from project-state handoff)

Built in `4e29da7c`; CE input went to `qet` on 2026-09-26 (`66b18dd8`), then to `embed` + `int_embedder: fourier` on 2026-09-27. A Fir submission for cv0-cv3 was prepared and not submitted at that handoff (per-run status after that: see the arm plans and W&B).

- **Shape.** Unmerged, forked from the MassSpecGym `_ema999` recipe. The folds differ only in `wandb_name`, `split_dp` and `lr_total_steps` (40 x ceil(train / 512), 30,120-31,200). The cv0 header records the measurements behind each key.
- **Deltas.** `masster_include_cmf: true`, because `[M+NH4]+` and formate/acetate spectra fragment by losing the adduct, which the +/-4 H ladder cannot reach. All nine adducts; `inst_types` FT/QTOF/Other/NA (retired `FT` label, now `ORB`; see the [`[M]+` finding](../../findings/masster/spectraverse_mplus_handling_2026_10_01.md#old-spv-checkpoints-carry-the-retired-ft-label)), with FraGNNet's 20 ppm / 500 Da tolerance for Other and NA; 40 epochs. The 44% of rows with no CE train on the CE table's learned missing-CE token. CMF is live for `[M+Na]+`, `[M+NH4]+`, `[M+K]+`, `[M+HCOOH-H]-`, `[M+CH3COOH-H]-` and `[M+Cl]-`; `[M+H]+`/`[M-H]-` duplicate a CRF H shift and `[M]+` has no CMF shift, so those three emit no CMF channel. The comparator is `config/spectraverse/fragnnet`, not `fragnnet_merged`.
- **CE input `embed` + `fourier`.** `qet` was dropped because `n1`-`n3` all fail on NIST20 with train fit unchanged; `fourier` keeps the per-energy code (`n4` is its NIST20 test). At max 300 it gives each integer 0-300 its own row, where abs-sines aliases 44 pairs in 212-300. It does not fix rare or unseen derived NCE values (525 distinct).
- **`loss_batch_size: 4`, measured.** The recipe's 64 runs out of memory at batch 32 on the 8 GB card (each loss chunk pads to its widest target; SPV peaks are p99 783, max 4096). At batch 128, `lb4` peaks at 4.68 GiB and 0.346 s/step, against 4.25 GiB and 0.291 s/step for the MassSpecGym recipe at `lb64`. Batch 512 was not run.
- **Validation.** All ten resolve through `load_config`; cv0's resolved diff against the MassSpecGym `_ema999` config is exactly the intended keys. The real cv0 train and val datasets construct (387,399 / 46,057 rows, none dropped, 757 steps/epoch at batch 512). A 2-epoch smoke on a 512-spectrum subsample exits 0 (val cosine 0.094 -> 0.167; with `qet` 0.082 -> 0.169; with `fourier` 0.106 -> 0.162). Run local smokes with `TMPDIR` off `/tmp` (the post-fit checkpoint copy fails on the 3.7 GB volume).
- **Not measured at build time.** The 40-epoch horizon (sized from the MassSpecGym and NIST20 peaks in steps), batch-512 memory, and wall clock (estimated about 6 h per fold).
- **Launch recipe.** `setup_job_cc.py --cv_config_dp config/spectraverse/masster --cv_start 0 --cv_end 3 --cv_use_job_id`, job name `msh_spv`, `fir:h100-3g.40gb`, 12 h ([submit guide](../../guides/submit_cluster_runs.md#4-cross-validation-and-existing-batch-launchers)); read peak GPU memory and best epoch on cv0 before cv4-cv9.
