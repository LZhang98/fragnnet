# Does cleaning the `[M+H]+`/FT training target make a better MASShatter model?

> **Status:** CLOSED NULL · **Updated:** 2026-09-23
> Inference-only cross-eval, no training. It was the acceptance test the `mh_ft_clean` pair was
> built around. Cleaning the target taught the model nothing: on the raw target, the f1 checkpoint
> (`6h67zps9`, trained on cleaned targets) and the f0 control (`3bej6fq7`) differ by
> **+0.0001 (val) / +0.0007 (test)**, both inside ±0.003 CIs. The +0.0054 W&B shows between them
> is the easier target: scoring either checkpoint on the cleaned target adds **+0.0048 to
> +0.0052**. The same inference run found a much larger lever. The two runs disagree so much per
> spectrum that averaging their predicted spectra adds **+0.020** cosine on both splits. About
> 40% of that (+0.008) is already available from two adjacent checkpoints of one run. Averaging
> those two checkpoints' *weights* needs no training and no inference cost, and it beats each
> run's best checkpoint by **+0.0029 / +0.0036**.

## Scope and method

The pair and its reading rule are in the config headers of
`config/massshatter/mh_ft_clean_2026_09_22/` and in
`scripts/manual_slurm/fir/submit_masshatter_mh_ft_clean.sh`. `clean_spectra` runs at dataset load,
so each arm's W&B cosine is scored against its own target. The cross-eval scores both
checkpoints against both targets.

|                            | f1 (arm)                                                                  | f0 (control)                  |
| -------------------------- | ------------------------------------------------------------------------- | ----------------------------- |
| run                        | `6h67zps9` `f1_mh_ft_all_filters`                                       | `3bej6fq7` `f0_mh_ft_control` |
| best checkpoint            | `model-epoch=062`, W&B best val 0.73718 (filtered target)                 | `model-epoch=061`, 0.73175 (raw target) |
| last checkpoint            | epoch 63, `ckpt/last.ckpt` on the W&B run                                 | epoch 63                      |
| training target            | 20 ppm centroid merge, subformula filter at FT tolerance, above-precursor cut | none                      |

Both runs are unmerged MASShatter, `[M+H]+`/FT on `data/split/nist20_inchikey`, 64 epochs, seed
397623, with identical architecture (the checkpoints load `strict=True` under either config).

- **Targets.** "Raw" is `f0_mh_ft_control.yml`'s `spec_params`; "filtered" is
  `f1_mh_ft_all_filters.yml`'s. The model hyperparameters in the two files are identical.
- **Command**, per cell:

  ```bash
  python scripts/eval/run_save_inference_eval.py \
    --custom_fp config/massshatter/mh_ft_clean_2026_09_22/<f0_mh_ft_control|f1_mh_ft_all_filters>.yml \
    --model_ckpt saved_ckpts/<ckpt> --eval_split <val|test> \
    --auxiliary_scores cos_sim cos_hun --eval_batch_size 256 --output_format hdf5
  ```

  Batch 512 runs out of memory in the boundary-bond encoder on an 8 GB card. Eval-mode
  predictions do not depend on batch size.
- **Code.** A `git archive` snapshot of `b5cd5c6d` on `PYTHONPATH`. The working tree carried an
  uncommitted cut-bond dedup edit from another session that touches this checkpoint's boundary
  path, so it was kept out.
- **Pipeline check.** The cells reproduce W&B: f0 on raw val gives 0.73174 (W&B 0.73175), and f1
  on filtered val gives 0.737175 (W&B 0.737176).
- **Sets.** Val has 52,020 raw spectra and 52,010 filtered spectra (cleaning empties 10); test
  has 52,621 in both. That is 4,221 and 4,223 molecules. Contrasts are paired per spectrum on the
  common set.
- **Uncertainty.** 95% CIs come from 2,000 molecule-cluster bootstrap resamples. They cover
  evaluation-set sampling only. With one run per arm, training noise is not in them.
- **Hardware.** Local RTX 2070 SUPER, about 160-180 s per cell.

## Result: the gain is the denominator

`cos_sim_0.01`, spectrum-weighted means on the common set:

| split | checkpoint | raw target | filtered target |
| ----- | ---------- | ---------: | --------------: |
| val   | f0         |     0.7318 |          0.7369 |
| val   | f1         |     0.7319 |          0.7372 |
| test  | f0         |     0.7361 |          0.7409 |
| test  | f1         |     0.7369 |          0.7417 |

Paired contrasts, Δ with 95% CI:

| contrast                                     | val                            | test                           |
| -------------------------------------------- | ------------------------------ | ------------------------------ |
| **model effect on raw target** (f1 − f0)     | **+0.0001** [−0.0026, +0.0028] | **+0.0007** [−0.0019, +0.0034] |
| model effect on filtered target (f1 − f0)    | +0.0002 [−0.0023, +0.0029]     | +0.0008 [−0.0018, +0.0035]     |
| denominator, f0 (filtered − raw)             | +0.0051 [+0.0046, +0.0057]     | +0.0048 [+0.0042, +0.0053]     |
| denominator, f1 (filtered − raw)             | +0.0052 [+0.0047, +0.0058]     | +0.0048 [+0.0043, +0.0054]     |
| W&B-style read (f1 on filtered − f0 on raw)  | +0.0054 [+0.0028, +0.0081]     | +0.0056 [+0.0027, +0.0083]     |

`cos_hun` gives the same answer: model effect on raw +0.0002 (val) and +0.0006 (test), and
denominators of +0.0053/+0.0054 (val) and +0.0050 (test). f1 wins 50.4% of val spectra and
49.3% of test spectra on the raw target.

**What the filters change.** They keep 97.5% of target intensity on average (median 99.7%) and
cut peaks per spectrum from 35.6 to 32.3. In 60% of spectra the target does not change
(raw-vs-filtered cosine ≥ 0.9999). Removed intensity rises with energy: 1.5-2.3% up to NCE 60,
3.2% above it. So does the denominator: it is +0.0019 at NCE ≤ 20 and +0.0087 above NCE 60 on
val, and +0.0014 and +0.0088 on test.

**Why a null was the expected answer.** The MASShatter cross-entropy (`loss.py:1275-1300`) sums
`−p_true · log q` over target peaks that some candidate matches. Unmatched intensity feeds only
`oos_ce`, and `oos_loss` is false. Removing unreachable peaks therefore cannot move a
spectrum's optimum. What it does is re-weight spectra, because each spectrum's loss scales with
its matched share of intensity. The one pattern the data show is consistent with that
re-weighting, but not established. f1 is slightly worse at NCE ≤ 40 (val −0.0022
[−0.0057, +0.0014], test −0.0030 [−0.0063, +0.0007]) and slightly better above it (val +0.0017
[−0.0014, +0.0050], test +0.0033 [+0.0002, +0.0066]). High-CE spectra carry the most unmatched
intensity, so cleaning gives them back loss weight. A val-only gain in the 0.9-0.99
target-change stratum (+0.0089) did not replicate on test (+0.0022, CI spans zero).

## Result: the two runs disagree per spectrum, and averaging them pays

The means agree to 1e-4, but the individual spectra do not. The per-spectrum cosine of f0 and f1
correlates at 0.855 (val) and 0.857 (test). The median |Δ| is 0.04 and the 90th percentile 0.24.
Predicted-spectrum agreement between the runs, as cosine of f0's prediction with f1's, has mean
0.874, median 0.938 and 10th percentile 0.649 on val (0.876 / 0.939 / 0.653 on test).

Averaging the two unit-sum predicted spectra, scored the same way as the singles against the raw
target:

| split | mean single | f0 + f1 ensemble | gain                           |
| ----- | ----------: | ---------------: | ------------------------------ |
| val   |      0.7318 |           0.7518 | **+0.0200** [+0.0195, +0.0205] |
| test  |      0.7370 |           0.7568 | **+0.0198** [+0.0193, +0.0203] |

The singles here are recomputed from the saved peaks, which the export thresholds at 0.1% of
base peak. They match the pipeline's own cosine to within 1e-4.

The gain lives where the runs disagree. On val:

| f0-f1 agreement | spectra | mean single | ensemble |   gain |
| --------------- | ------: | ----------: | -------: | -----: |
| ≤ 0.5           |   2,133 |       0.414 |    0.499 | +0.085 |
| 0.5-0.7         |   4,781 |       0.524 |    0.581 | +0.057 |
| 0.7-0.85        |   8,673 |       0.625 |    0.661 | +0.036 |
| 0.85-0.95       |  12,512 |       0.728 |    0.744 | +0.017 |
| 0.95-0.99       |  11,532 |       0.807 |    0.812 | +0.005 |
| > 0.99          |  12,379 |       0.876 |    0.877 | +0.001 |

Disagreement grows with collision energy. Agreement falls from 0.939 (NCE ≤ 20) to 0.834
(NCE > 60), and the ensemble gain rises from +0.011 to +0.026. Test gives 0.944 → 0.832 and
+0.010 → +0.026. The regime is the same one that holds FraGNNet's GLACIER gap
([GLACIER CE decomposition](../fragnnet/glacier_gap_ce_decomposition_2026_09_21.md)).

**Within-run versus between-run.** The epoch-63 `last.ckpt` of each run was scored on raw val
too:

| ensemble                            | members' prediction agreement (mean) | gain over mean single          |
| ----------------------------------- | -----------------------------------: | ------------------------------ |
| f1 epoch 62 + epoch 63 (one run)    |                                0.950 | **+0.0076** [+0.0074, +0.0078] |
| f0 epoch 61 + epoch 63 (one run)    |                                0.947 | **+0.0081** [+0.0079, +0.0083] |
| f1 + f0, any of the four epoch pairs |                         0.873-0.876 | +0.0196 to +0.0202             |
| all four checkpoints                |                                      | 0.7555 against 0.7317 single   |

About 40% of the two-run gain is optimizer noise at the constant LR of 1e-3
(`lr_schedule: false`). One epoch moves the f1 single-model score from 0.7318 to 0.7303. The
other ~60% needs a genuinely different solution.

**Weight averaging.** The same two checkpoints per run were averaged tensor by tensor. That covers
171 floating-point tensors, BatchNorm running statistics included. The 9 integer counters were
taken from `last.ckpt`. The averaged checkpoint was then scored on raw val with the same command:

| run  | best checkpoint | `last.ckpt` | weights averaged | vs mean of the two             | vs best checkpoint             |
| ---- | --------------: | ----------: | ---------------: | ------------------------------ | ------------------------------ |
| f1   |          0.7318 |      0.7303 |       **0.7347** | +0.0037 [+0.0033, +0.0040]     | **+0.0029** [+0.0021, +0.0037] |
| f0   |          0.7317 |      0.7309 |       **0.7353** | +0.0040 [+0.0037, +0.0044]     | **+0.0036** [+0.0027, +0.0045] |

`cos_hun` agrees: 0.7602 against 0.7569 for f1, and 0.7605 against 0.7567 for f0. A two-checkpoint weight
average recovers about half of what averaging the same two checkpoints' predictions does (+0.0076
and +0.0081). It needs no training, costs nothing at inference, and beats a best checkpoint
that was selected on this split. BatchNorm statistics were averaged rather than recomputed on
training data, as SWA would do, so this is a lower bound on what a proper average gives.

## Side finding: `cos_sim_0.01` bin edges

On f0's raw-val predictions, the tolerance-matched `cos_hun` exceeds the 0.01 Da binned
`cos_sim_0.01` by 0.025 on average, and by more than 0.1 on 4.9% of spectra. Part of that is bin-edge
splitting. In spectrum 620548 (4-amino-5-propyl-1,2,4-triazole-3-thione, NCE 155), the 20 ppm
centroid merge folds 59.9884 into the 59.99 base peak and moves it about 0.1 mDa across a bin
edge. The prediction at 59.9902 then lands in the neighbouring bin, and `cos_sim_0.01` falls
from 0.982 to 0.001 while `cos_hun` stays at 0.987. The noise is symmetric across arms, so paired
verdicts survive it, but it inflates per-spectrum variance in every `cos_sim_0.01` read.

## Reading

- **Target cleaning is a null as a training change on `[M+H]+`/FT.** It is harmless, but it does
  not help, and scoring an arm on its own cleaned target inflates it by about 0.005. Any number
  scored on a cleaned target is not comparable with one scored on a raw target. That includes
  the ~0.006 the
  [overfit read-out](masshatter_overfit_readout_2026_09_22.md) cross-tab assigns to
  `filter_subformula_peaks` in the 0.73-vs-0.80 gap to `8v0i0fq7`, which this measurement says is
  denominator.
- **Run-to-run variance is the largest lever this line has measured.** +0.020 from two runs is
  more than three times the best architecture arm (`a1_prenorm` +0.0056,
  [epoch-128 read-out](masshatter_architecture_arms_epoch128_2026_09_22.md)). The within-run +0.008
  is the order of `d3`'s +0.0071 from LR decay (n=1), consistent with both acting on optimizer
  noise. A two-checkpoint weight average already captures +0.003 to +0.004 of it for free.

## Limits

- **One run per arm.** The CIs are evaluation-set CIs. The directional NCE trade in particular is
  at about the size training noise could produce.
- **The ensemble members are not seed replicates.** f0 and f1 share a seed and differ in target
  and in the non-deterministic training path. The between-run gain may differ for a pure seed
  pair, though the target change is small (60% of spectra unchanged).
- **The weight average is two checkpoints on val only.** Both best checkpoints were selected on
  val, which favours them over the average, so the sign is safe. The size of a many-epoch EMA or
  SWA average is not measured.
- **Unmerged `[M+H]+`/FT only.** The merged `c4a_base` lineage has not been measured this way.

## Next

1. **Weight averaging in training.** Nothing in `src/` implements EMA or SWA. A running average
   over the last several epochs, with BatchNorm statistics recomputed, is the direct extension of
   the free +0.003. Replicating `d3`'s LR decay tests the same noise from the schedule side.
   Retrofitting is also possible: runs that saved more than two checkpoints can be averaged
   without retraining.
2. **A seed pair of one config** to price the between-run share without the target confound.
3. **Report `cos_hun` beside `cos_sim_0.01`** in per-spectrum analyses, where the bin edges add
   noise.

## Artifacts

Local and Git-ignored, under `output/cross_eval_mh_ft_clean_2026_09_23/`: the eight cross-eval
HDF5 files plus four raw-val cells for the last and weight-averaged checkpoints
(`cross_eval/<target>_<split>/<ckpt>/`), logs, the averaged checkpoints (`ckpts/`), the
downloaded `last.ckpt` files (now `saved_ckpts/*_last.ckpt`), and `analysis/`. That folder holds
the runner scripts, the target dump, the paired-analysis and ensemble scripts, and the per-spectrum
tables and result JSONs.
