# False-Peak Penalty + Intensity Calibration — Loss Extension Plan

> **Status:** CLOSED NULL · **Updated:** 2026-10-06
> No headroom: `log1p_huber` is actively harmful above lambda=0.1. Kept for the negative result; `slurm_scripts/losses_false_peak_sweep.sh` and `losses_log1p_huber_sweep.sh` cite this doc. Recovered from branch `research/ce_perfrag_features`, where it was the only copy.

## Motivation

FraGNNet predicts over a **closed candidate set** (fragments enumerated by
the DAG), scored with a shared softmax that already includes an explicit
out-of-sample bucket (`pred_oos_logprobs` in
`src/fragnnet/model/loss.py:170-481`, `sparse_cross_entropy_seq`/`_vec`).
Because probability mass is conserved, a candidate peak with no matching
true peak already competes for mass with true peaks, inflating `ios_ce`
when it wins mass it shouldn't — so an explicit "false peak" loss term
looked redundant with what the existing objective already enforces.

Before touching the loss or model head, that assumption was checked
empirically against an existing prediction dump instead of a fresh
training run.

## Step 1 Results (2026-07-08, NIST23 test split, 85,660 spectra)

Built `scripts/debug/debug_false_peak_intensity_stats.py`: joins
`data/predicted/predictions_test.pkl` (existing test-set predictions) with
ground-truth peaks from `data/proc/nist23/spec_df.pkl` via `spec_id`, and
matches each predicted peak to the true spectrum at 10 ppm
(`calculate_match_mzs(tolerance=1e-5, relative=True, tolerance_min_mz=200.0)`,
matching `src/fragnnet/utils/spec_utils.py:587` defaults).

| Metric | Value |
|---|---|
| Unmatched predicted intensity (mean / median) | 17.9% / 9.0% |
| Unmatched predicted intensity, p90 | 49.6% |
| Spectra with zero unmatched intensity | 1.7% |
| Unmatched predicted peak *count* (unweighted) | 65.0% of emitted candidates |
| `pred_oos_prob` (mass off the candidate set entirely) | mean 15.1% / median 6.9% |
| Top-1 (base peak) match rate | 92.1% |
| corr(unmatched intensity fraction, cos_hun) | **−0.63** |

By `cos_hun` bucket:

| cos_hun bucket | mean unmatched intensity | median | n |
|---|---|---|---|
| 0.0–0.3 | 43.9% | 37.9% | 13,619 |
| 0.3–0.6 | 22.7% | 16.8% | 15,607 |
| 0.6–0.8 | 15.2% | 10.3% | 14,166 |
| 0.8–0.9 | 11.8% | 7.5% | 10,562 |
| 0.9–1.0 | 6.4% | 3.4% | 31,247 |

By `frag_mode` / `inst_type` (IT excluded from the "concentrated" reading —
n=35, cos_hun≈0, already known dead weight per prior eval work):

| Group | mean unmatched intensity | median | cos_hun mean | n |
|---|---|---|---|---|
| HCD | 14.6% | 7.7% | 0.706 | 74,281 |
| **CID** | **39.5%** | **31.0%** | 0.527 | 11,379 |
| FT | 16.4% | 8.5% | 0.691 | 80,801 |
| **QTOF** | **43.7%** | **35.4%** | 0.527 | 4,824 |
| IT | 99.8% | 100% | 0.001 | 35 |

CID and QTOF run ~2.7-3x the unmatched-intensity rate of HCD/FT, tracking
the same gap already known from the per-CE cos_hun numbers (HCD 0.781 vs
CID 0.547 vs QTOF ~0.616). `frag_mode x inst_type` shows CID×FT (36.0%)
and CID×QTOF (43.7%) both bad, so this is not purely an instrument
artifact — CID itself carries most of the signal.

**Conclusion: the redundancy hypothesis is wrong.** The existing
`ios_ce`/`oos_ce` softmax competition does not fully suppress false-peak
mass in practice — 9-18% of predicted intensity lands on non-matching
candidates on average, and it is not spread evenly as background noise:
it concentrates exactly where the model already struggles (44% unmatched
intensity in the worst cos_hun bucket vs. 6% in the best). This is
consistent with the CID/QTOF weak points already tracked in prior
evaluation work — low-confidence spectra have enough true-peak ambiguity
that wrong candidates aren't strongly penalized by the current objective.
`pred_oos_prob` (mass assigned to no candidate at all, a different
quantity) is a separate, already-modeled bucket and is not what `L_false`
would target.

**Decision so far**: an explicit false-peak penalty is empirically
justified, not just theoretically redundant busywork. Proceed to
implementation.

## Core idea (staged, loss-only first)

Do not redesign the model head yet. Extend the existing `loss_type`
mechanism in `src/fragnnet/pl_model/fragnnet_pl.py` (`_setup_loss_fn`,
`_loss_fn`) with two additive terms on top of the current primary loss
(`ios_ce + oos_ce`, or the cosine/JSD Hungarian alternatives):

1. **`L_false`** — penalize predicted intensity on candidates with no
   ppm-tolerant true-peak match:
   `L_false = sum_j(q_j * (1 - s_j)) / (sum_j(q_j) + eps)`, where `s_j` is
   the match indicator from `calculate_match_mzs` and `q_j` is predicted
   intensity/probability. Cheap to compute — the match mask is already
   computed inside `sparse_cross_entropy_seq`/`_vec`; the new term should
   reuse `b_match_mask`/`bl_both_mask` rather than recomputing it.
2. **`L_log1p`** — Huber loss on `log1p(beta * pred)` vs.
   `log1p(beta * true)` for ppm-matched peaks, to address intensity
   calibration, which none of the current `loss_type` options
   (`cross_entropy`, `cosine_distance[_hungarian]`, `jss_distance[_hungarian]`)
   directly supervise.

Both gated behind new hparams (`lambda_false`, `lambda_int`, `beta`) so
existing configs are unaffected by default (`lambda_false=0`,
`lambda_int=0` reproduces current behavior exactly).

## Remaining steps

```
2. [DONE 2026-07-08] Implemented as sparse_false_peak_penalty in
   src/fragnnet/model/loss.py (standalone function, own
   calculate_match_mzs call rather than reusing sparse_ce_fn's internal
   mask -- sparse_cross_entropy_seq/vec are shared by precursor_pl.py and
   graff/pl_model.py too, so threading a new return value through their
   signatures was judged higher-risk than the small extra match-mzs cost).
   Wired into fragnnet_pl.py as false_peak_loss_weight (default 0.0 in
   config/*_template.yml), following the existing *_weight/loss_names
   pattern (_setup_loss_names/_setup_loss_fn/_loss_fn).
   -> verified: 7 unit tests in tests/test_loss.py::TestSparseFalsePeakPenalty
   (all-matched=0, all-unmatched=1, mixed weighting, empty true/pred peaks,
   gradient flow, per-sample tolerance override) -- all pass. End-to-end
   smoke test on config/debug/debug_tiny_dual_ce.yml: with
   false_peak_loss_weight=0.05, false_peak_loss logs correctly
   (train/val ~0.95, expected for an undertrained 1-epoch model) and total
   loss includes it; with the override omitted (default weight=0),
   "false_peak_loss" does not appear in the logged metrics at all,
   confirming the term is fully skipped (via the loss_names guard), not
   just zero-weighted -- the default-config code path is unchanged.

3. [PARTIAL 2026-07-08] Ran lambda_false in {0.01, 0.05, 0.1, 0.25} plus a
   0.0 baseline on config/debug/debug_tinyv4_false_peak_sweep.yml (new
   config; full debug_tinyv4 split, 987 train / 424 val spectra, ~19% CID
   / ~11% QTOF, NodeMLP, 15 epochs, single seed). Results (val, final
   epoch):

   | lambda_false | val_cos_sim_0.01 | val_false_peak_loss | val_ios_ce |
   |---|---|---|---|
   | 0.0  | 0.281 | (term off) | -0.247 |
   | 0.01 | 0.275 | 0.743 | -0.345 |
   | 0.05 | 0.300 | 0.714 | -0.379 |
   | 0.10 | 0.299 | 0.710 | -0.381 |
   | 0.25 | 0.280 | 0.724 | -0.322 |

   **Inconclusive, not a confirmed win.** Mild bump at lambda=0.05-0.1
   (best cos_sim, lowest false_peak_loss), worse at both 0.01 and 0.25,
   but this is a single seed on debug_tinyv4's frag_dp (only 315 molecules
   total before train/val split, further reduced by spec_params.merge) --
   nowhere near enough to separate signal from noise. Attempted to scale
   up to a real nist20 subset (data/proc/nist20 + subsample_params) with a
   hand-assembled config and hit `mat1 and mat2 shapes cannot be
   multiplied` in frag_embedder. ROOT CAUSE (found after further digging,
   not a stale frag_dp as first guessed): the hand-assembled config copied
   debug_tinyv4's frag_params.pyg_node_feats = [cc, base_formula, depth,
   h_formulae_idx], but data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags
   was built for the fuller feature set used by the real
   config/sweeps/nist20_d3_ma_mi_*.yml configs: + cmf_h_formulae_idx +
   boundary_pair. The frag_dp itself is fine; the production sweep configs
   already use the correct feature list and should just work. See
   "Key files" below for the confirmed-working config list.
   -> verify: unmatched intensity drops without cos_hun/recall regressing
   beyond noise; check specifically whether the CID/QTOF cos_hun gap vs.
   HCD/FT narrows, not just that the global average improves (a global
   win driven only by HCD would not be the outcome this is meant to test).

   [RESULT 2026-07-09, real nist20 SLURM sweep] Ran the 5-point
   `config/losses/fraggnn_d3_ma_mi_nist20_amolc_all_false_peak_*.yml` sweep
   (100-epoch schedule, production `amolc_all` config) via
   `slurm_scripts/losses_false_peak_sweep.sh`. Pulled final
   `val_cos_sim_0.01_best/mean` from the wandb API
   (`frag-gnn/frag-gnn`) once runs reached epoch ~98-99:

   | lambda_false | epoch reached | val_cos_sim_0.01_best/mean | delta vs. own weight=0 control (milli) |
   |---|---|---|---|
   | 0.0 (control) | 99 (finished) | 0.71154 | -- |
   | 0.01 | 98 (crashed near end) | 0.71198 | +0.44 |
   | 0.05 | 98 (crashed near end) | 0.71218 | +0.64 |
   | 0.10 | 98 (crashed near end) | 0.71129 | -0.25 |
   | 0.25 | 99 (finished) | 0.71197 | +0.43 |

   Noise floor established by comparing the two weight=0 "no-op" control
   runs (`false_peak_000` and `log1p_huber_000`, which should be
   statistically identical to the true baseline) against the actual
   baseline run (0.71192): both landed within ±0.4 milli-cos_sim of it.
   Every lambda_false point above falls inside or barely outside that same
   ±0.4-0.6 milli band -- **no lambda tested clears the noise floor**.
   `val_false_peak_loss_epoch/mean` itself is also flat across lambda
   (0.183-0.192, no visible dose-response as lambda increases 0.01->0.25).
   **Verdict: null result.** false_peak_loss at these weights neither
   helps nor hurts val_cos_sim_0.01, and doesn't show a dose-response in
   its own target quantity either. See step 5 decision below -- CLOSED,
   do not keep sweeping this term further without a different mechanism
   or a much wider lambda range.

4. [DONE 2026-07-08] Implemented as sparse_log1p_huber_calibration in
   src/fragnnet/model/loss.py, alongside sparse_false_peak_penalty. For
   each true peak with >=1 ppm-tolerant predicted match, predicted
   intensity is the summed probability mass of all matching candidates
   (mirrors how ios_ce treats multiple candidates mapping to one true
   peak); Huber loss is computed on log1p(beta*intensity) for both sides
   and averaged over matched true peaks per spectrum (unmatched true
   peaks excluded from the average, not zero-filled -- that's L_false's
   job). Wired into fragnnet_pl.py as log1p_huber_loss_weight (default
   0.0) + log1p_huber_beta (default 10.0) in config/*_template.yml,
   following the same *_weight/loss_names pattern as false_peak_loss.
   -> verified: 10 unit tests in
   tests/test_loss.py::TestSparseLog1pHuberCalibration (perfect match=0,
   analytic Huber value on a mismatch, beta changes magnitude, unmatched
   true peak excluded from average, multi-candidate mass aggregation,
   empty true/pred peaks, gradient flow, per-sample tolerance override)
   -- all pass, plus full existing suite (57/57) still green. End-to-end
   smoke test on config/debug/debug_tiny_dual_ce.yml: with
   log1p_huber_loss_weight=0.1, log1p_huber_loss logs correctly
   (train/val ~0.20-0.23, expected for an undertrained 1-epoch model);
   with the override omitted (default weight=0), "log1p_huber_loss" does
   not appear in the logged metrics at all, confirming the term is fully
   skipped via the loss_names guard, not just zero-weighted.
   Debug config config/debug/debug_tinyv4_log1p_huber_sweep.yml (mirrors
   debug_tinyv4_false_peak_sweep.yml) confirmed the wiring end-to-end on
   the fuller debug_tinyv4 split: 1-epoch check (weight=0.1, beta=50)
   logs log1p_huber_loss and total loss includes it; full 5-epoch run
   trains cleanly with log1p_huber_loss decreasing (0.480 -> 0.452)
   alongside the primary loss.
   4 nist20 SLURM configs created in config/losses/ (log1p_huber_000/
   01/025/05, beta fixed at 10.0, lambda_int in {0, 0.1, 0.25, 0.5}) +
   slurm_scripts/losses_log1p_huber_sweep.sh (array 0-3), mirroring the
   false_peak sweep infrastructure -- ready to launch, not yet run.
   [RESULT 2026-07-09, real nist20 SLURM sweep] Ran the 4-point
   `config/losses/fraggnn_d3_ma_mi_nist20_amolc_all_log1p_huber_*.yml`
   sweep the same way (beta fixed at 10.0, lambda_int swept). 3 of the 4
   runs crashed 10-13 epochs before the 99-epoch schedule finished;
   checked how much the weight=0 control (`log1p_huber_000`) still gained
   in its own last 10-13 epochs (+0.9-1.0 milli) to judge whether the
   incomplete runs could still have closed their gap -- they could not,
   given the sizes below:

   | lambda_int (beta=10.0) | epoch reached | val_cos_sim_0.01_best/mean | delta vs. weight=0 control at matched epoch (milli) |
   |---|---|---|---|
   | 0.0 (control) | 99 (finished) | 0.71204 | -- |
   | 0.10 | 89 (crashed near end) | 0.71098 | -0.13 |
   | 0.25 | 86 (crashed near end) | 0.70932 | -1.79 |
   | 0.50 | 89 (crashed near end) | 0.70926 | -1.85 |

   `val_log1p_huber_loss_epoch/mean` itself is flat across lambda
   (0.065-0.066 for 0.1/0.25/0.5) -- same lack of dose-response seen in
   `false_peak_loss`.
   **Verdict: lambda_int=0.1 is a null result (within the ±0.4 milli noise
   floor established in step 3); lambda_int in {0.25, 0.5} is a real,
   consistent ~1.8 milli-cos_sim regression** -- clearly outside the noise
   floor even after accounting for the incomplete runs' remaining
   headroom. log1p_huber_loss as implemented (beta=10, this lambda range)
   trades away shape similarity without a compensating gain in its own
   calibration target. CLOSED -- do not increase lambda_int further; a
   lower beta or a much smaller lambda_int (<0.1) was not tested and would
   be the only reasonable next point if this line is revisited.

   Operational note: 6 of 9 sweep runs across both step 3 and step 4 show
   `state=crashed` in wandb despite ~11.9h runtime (well under the 24h
   SLURM time limit) and reaching epoch 86-98 of 99 -- likely a cluster-side
   preemption/node issue, not a code bug (loss values and epoch counts are
   coherent right up to the last logged step). Worth checking
   `jobs/*.err` on the cluster if this recurs on future sweeps.

5. DECISION GATE: combine the best lambda_false + lambda_int + beta into
   one config, run full NIST23 CV (or the per-CE/merged setup already
   used for reporting), compare against the Stage 0 baseline on: cos_hun,
   unmatched predicted intensity, true-peak recall, top-10 recall,
   explained intensity, retrieval rank. Only proceed to a presence/
   intensity dual-head model redesign if this shows a real, non-noise
   improvement — do not redesign the model head speculatively.

   [CLOSED 2026-07-09] Step 3 result: false_peak_loss is a null result
   across lambda in {0.01,0.05,0.1,0.25} on val_cos_sim_0.01. Step 4
   result: log1p_huber_loss (beta=10) is null at lambda=0.1 and actively
   harmful (~1.8 milli-cos_sim) at lambda in {0.25,0.5}. Neither term
   clears the "real, non-noise improvement" bar this gate requires, so
   there is no candidate config worth combining and running through full
   NIST23 CV. Per the gate's own rule: **do not proceed to a
   presence/intensity dual-head model redesign** on the basis of this
   work -- the loss-only approach did not earn it.
   Caveat: only one metric (val_cos_sim_0.01_best/mean) was checked, on
   the nist20 val split, single seed per point. The wider metric set the
   gate calls for (unmatched intensity, recall, retrieval rank, and the
   CID/QTOF-vs-HCD/FT gap specifically) was not pulled before closing,
   since the primary metric already showed null/negative results -- if
   this line is ever revisited, that fuller check should happen first,
   not be skipped a second time.
```

Optional, only after step 5 shows benefit and only if still needed: a weak
shape regularizer (cosine or JSD, not both) on top of the false+calibration
loss, per the original ablation table below.

## Ablation table

| Experiment | Loss | Purpose |
|---|---|---|
| A | current `ios_ce + oos_ce` (or cosine/JSD hungarian) | baseline |
| B | A + `L_false` | direct false-peak suppression |
| C | A + `L_log1p`-Huber | intensity calibration |
| D | A + `L_false` + `L_log1p`-Huber | main practical candidate |
| E | D + weak cosine | shape regularization (optional) |

Key comparison: A vs. D.

## Key files

- `src/fragnnet/model/loss.py` — `sparse_false_peak_penalty` (implemented
  step 2, standalone function; see "Remaining steps" step 2 note for why
  it wasn't threaded into `sparse_cross_entropy_seq`/`_vec`'s internal
  match mask); `sparse_log1p_huber_calibration` (step 4) sits right after
  it, same standalone-function pattern.
- `src/fragnnet/pl_model/fragnnet_pl.py` — `_setup_loss_names` (line 273),
  `_setup_loss_fn` (line 384), `_loss_fn` (line 561): `false_peak_loss`
  and `log1p_huber_loss` both follow the existing `*_weight`/`loss_names`
  pattern end to end.
- `tests/test_loss.py::TestSparseFalsePeakPenalty` — 7 unit tests for
  `sparse_false_peak_penalty` (all-matched, all-unmatched, mixed
  weighting, empty true/pred peaks, gradient flow, per-sample tolerance).
- `tests/test_loss.py::TestSparseLog1pHuberCalibration` — 10 unit tests
  for `sparse_log1p_huber_calibration` (perfect match, analytic Huber
  value, beta sensitivity, unmatched-peak exclusion, multi-candidate mass
  aggregation, empty true/pred peaks, gradient flow, per-sample
  tolerance).
- `config/*_template.yml` — `false_peak_loss_weight: 0.0`,
  `log1p_huber_loss_weight: 0.0`, `log1p_huber_beta: 10.0` defaults.
- `config/debug/debug_tinyv4_false_peak_sweep.yml` — quick local step-3
  sweep config (full `debug_tinyv4` split, NodeMLP, 15 epochs); useful for
  fast wiring checks, too small for a trustworthy result.
- `config/debug/debug_tinyv4_log1p_huber_sweep.yml` — same idea for step 4
  (5 epochs); `log1p_huber_loss_weight`/`log1p_huber_beta` left at
  template defaults, set via `--override` per sweep point. Confirmed
  working via a 1-epoch wiring check and a full 5-epoch run (2026-07-08).
- `config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc_all.yml` —
  production base the step-3 SLURM sweep configs below are cloned from
  (current production config, not the older `config/sweeps/` templates
  further down): NodeMLP, ACE+NCE dual CE, dynamic batch sampler, 100
  epochs, tuned dropout/LR. Already uses the correct `frag_dp` and full
  `pyg_node_feats` (`+cmf_h_formulae_idx +boundary_pair`). Sibling
  variants in the same directory (`_no_sgs`, `_ringmerge`, `_sqrt`,
  `_jss_hun_sqrt`, `_filtered_peak`) are other active experiment lines --
  clone one of those the same way if it's the more relevant baseline.
- `config/losses/fraggnn_d3_ma_mi_nist20_amolc_all_false_peak_{000,001,005,01,025}.yml`
  — [DONE 2026-07-08] the 5 step-3 sweep points as standalone config
  files (full copies of `amolc_all.yml` + `false_peak_loss_weight: 0.0 /
  0.01 / 0.05 / 0.1 / 0.25` + distinguishing `wandb_name`/tags), each
  validated to load via `load_config`. `000` is an explicit baseline
  copy (functionally identical to the unmodified `amolc_all.yml`, kept
  here so all 5 sweep points live together).
- `slurm_scripts/losses_false_peak_sweep.sh` — SLURM array job
  (`--array=0-4`, gitignored per repo convention like the rest of
  `slurm_scripts/`) that maps array task ID to one of the 5 configs
  above and launches `run_pl_model_fit.py -w online`. Stages
  `data/proc/nist20`, `data/split/nist20_ma_mi_inchikey`, and
  `data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse.tar` to node-local
  scratch, following the same pattern as `spectraverse_fragnnet.sh`.
  Does NOT hardcode `WANDB_API_KEY` (unlike some `spectraverse_*`
  scripts) -- must already be set in the submitting shell's environment.
- `config/losses/fraggnn_d3_ma_mi_nist20_amolc_all_log1p_huber_{000,01,025,05}.yml`
  — [DONE 2026-07-08] 4 step-4 sweep points (full copies of `amolc_all.yml`
  + `log1p_huber_loss_weight: 0.0 / 0.1 / 0.25 / 0.5` with `log1p_huber_beta`
  fixed at 10.0 + distinguishing `wandb_name`/tags), each validated to load
  via `load_config`. Beta itself is not swept in this first pass (kept at
  the template default) -- simplicity over the full 3x3 beta x lambda_int
  grid from the original plan; revisit beta only if a lambda_int point
  shows a real signal worth refining.
- `slurm_scripts/losses_log1p_huber_sweep.sh` — SLURM array job
  (`--array=0-3`, gitignored, same as `losses_false_peak_sweep.sh`) that
  maps array task ID to one of the 4 configs above; identical staging
  pattern (nist20 spec/mol/split/frag_dp to node-local scratch). Ready to
  launch, not yet run.
- Older `config/sweeps/nist20_d3_ma_mi_*.yml` templates (efa, focused,
  sqrt, jss_sqrt, frag_gine[_focused]) -- also confirmed compatible
  (same frag_dp/node_feats requirement), kept as a fallback/reference set
  but superseded by the config above as the actual production baseline.
  All use `train_batch_size: 256`, `max_epochs: 30`. The frag_gine
  variants OOM'd locally on an 8GB RTX 2070 at that batch size; likely
  fine on cluster GPUs with more VRAM. `nist20_d3_ma_mi.yml` (no `_smpFalse`
  suffix) was NOT confirmed -- that frag_dp doesn't exist in this local
  checkout.
- `scripts/debug/debug_false_peak_intensity_stats.py` — Step 1 diagnostic;
  reuse to evaluate checkpoints from steps 3-5 without a full eval run
  (needs a `predictions_*.pkl`-format inference dump, not yet generated
  for the step-3 sweep checkpoints -- step 3 so far only used the
  built-in per-epoch val metrics, not this script).
- `data/predicted/predictions_test.pkl` + `data/proc/nist23/spec_df.pkl` —
  existing prediction dump used for Step 1; re-run against fresh
  checkpoints' prediction dumps in the same format for later steps.

## Successor output-head lines, all null (from local memory, verified 2026-10-06)

None of the following code is on the current branch (`scatter_sparsemax`, `IntensityHead`, `PresenceHead`,
`FormulaBagCorrectionHead` are absent from `src/`). It lives on `research/it-takes-two` (two-head, sparsemax/entmax15,
`intensity_head`) and `research/ce_perfrag_features` / `research/frag_competition_v2` (formula bag, competition). Metric
`val_cos_sim_0.01_best` on NIST20, plain-softmax baseline band 0.710-0.712 (`9t63zils` 0.7119, `4ews8j6f`).

- **Sparse gate and `intensity_head` (2026-07-09 to 07-10): null.** Sparsemax and entmax15 gates (`output_normalization`)
  plus `IntensityHead` (a product-of-experts reweighting `g_j * mu_j`, renormalized; not a hurdle model) finished at
  `fq0f9vjb` sparsemax 0.702, `vyib86j1` entmax15 0.705, `f7mxbe4v` sparsemax + `output_scale` 0.685 (worse), and
  `9fjevxif` softmax + head 0.700 vs baseline 0.703 at epoch 46. Test inference (96,128 spectra): cos_hun_inst_tol
  0.743 / 0.748 vs baseline 0.751, bad-case rate (<0.3) 8.9% -> ~10%, predicted peaks 104 -> 78 / 68. The head does
  what it was built to do (fewer peaks) without improving quality: the trimming is indiscriminate. An earlier claim
  that the sparsemax gate "collapsed" (0.415 vs 0.703 at epoch 43) did not match the finished W&B history
  (smooth 0.431 -> 0.671 -> 0.702) and should not be cited. Real bug found on the way: `scatter_sparsemax` overflowed on
  `LOG_ZERO` (-3.4e38) masked slots (cumsum in raw logit space); any new scatter primitive that does arithmetic on raw
  logits needs its own `LOG_ZERO` safety pass.
- **Two-head presence/intensity auxiliary losses (2026-07-10 to 07-12): no evidence of benefit.** Independent
  `PresenceHead` (sigmoid) and `AuxIntensityHead` (softplus) never feed the primary spectrum. The first hard
  candidate-level BCE was replaced (user: "we don't have ground truth") by a noisy-OR observability loss over
  m/z-matched candidate bags plus responsibility-weighted Huber and a composed-spectrum CE. Sweep `4lgizh9g` (best
  0.7085, crashed step 149) and `6jneywv3` (0.7054, crashed step 93, `val_composed_spectrum_loss` 5.09, ~15x the
  other auxiliary terms): never beat baseline. A latent NaN (`exp` of an aux log-probability above ~88.7) was fixed.
  Presence is root-cause dead rather than under-tuned: `presence_bce`/`observability_loss` plateaus at 0.32-0.34 val in
  every design (BCE weight 1.0, noisy-OR at 0.1), while the aux-intensity Huber learns (~0.07). Three causes: noisy-OR
  saturates on large isomeric-tie bags, presence is CE-dependent so labels look XOR-like and collapse to the base
  rate, and an independent sigmoid has no competition. The composed presence x intensity spectrum was never scored.
  Do not sweep presence-head loss weights again.
- **Residual recalibration head (`residual_head`, KL-leashed `q' = softmax(logits + delta)`, 2026-07-09):** infrastructure and
  8 tests only; never run against a pretrained checkpoint. Given the nulls above, treat it as unvalidated.
- **Formula-bag correction (`FormulaBagCorrectionHead`, 2026-07-11): near-neutral.** Deep-Sets `delta_f` on the formula
  marginal with a KL leash: `kl000` (`mu3nb02n`) 0.7090, `kl005` (`t5fhykzo`) 0.7102, `kl01` (`o2fr25j1`) 0.7119 vs
  softmax ~0.709. More leash was better, so the extra capacity earned nothing. The CE-response variant (per-bag
  quadratic energy curve) learned a degenerate near-constant attractor (`mirlys0d` b_NCE +11.98 std 0.10; `ykh2hts0`
  +12.48 std 0.32), nearly uniform corrections that cancel under softmax; see also the Git-history report cited in
  `energy_based_spectrum_reranking_plan.md` (removed 2026-10-18; `git show 95341fce:docs/plans/energy_based_spectrum_reranking_plan.md`). Formula-level `delta_f` leaves P(r|f) exactly invariant; "smaller
  softmax denominator" is not a real advantage because logsumexp regroups the same Z.
- **Per-fragment CE features (`ce_perfrag_feats`, `zhjgd01n`, 2026-07-13): null.** `ce_insert_location: mol,mlp` plus
  `b_ring`, `b_conj`, `num_heavy_atoms`: best 0.7112 vs baseline `9t63zils` 0.7119 (-0.0007), final 0.7091. CE-binned
  metrics were not logged (`log_adduct_inst_ce_metrics: False`), so a high-CE-only effect is unexcluded. A silent
  drop bug (`"mol,mlp"` matching neither exact branch) was guarded.
- **Candidate multiplicity (2026-07-15, 349,882 matched peaks / 14,992 FT/HCD `[M+H]+` spectra, 10 ppm): null.**
  "Many-to-one candidate-to-peak matching stops identifying the formula at high CE" is unsupported: unique candidate
  m/z clusters per peak mean 1.30, >1 on 28.8% of peaks, flat-to-declining in CE at fixed m/z (the intensity-weighted
  rise 1.14 -> 1.46 is a mass mix effect). Do not build responsibility weighting or ambiguity penalties on that
  story. Systematic +/-H CMF/CRF aliasing needs the charge carrier to be a proton: `[M+H]+` (54.4% of NIST20) and
  `[M-H]-` (23.4%) alias, `[M+Na]+`/`[M+K]+` do not; live aliasing must be counted after `frag_joint_mask`
  (the earlier "63.3% duplicates" counted pre-mask rows and is wrong).
- **Fragment competition head (`FragmentCompetitionHead`, GLACIER Tier-3 analogue, 2026-07-13/14): never trained.**
  Top-K transformer over the highest-scoring formulas, zero-init `delta_f` on the spectrum path only, CE-conditioned.
  `scripts/eval/competition_topk_coverage.py`: K=96 covers ~95% of even high-NCE spectra (p95 94-100 significant
  peaks at the 1% threshold), K=128 is the cheap hedge. Four NaN crashes: FiLM `log_scale` drift (clamped to +/-10); a
  zero-init `0*inf` trap; gradient back-flow through the shared `ce_embed` into the frag encoder (proved by backward
  from the head alone; the fix is `detach` on every head input, and the regression test must assert that no non-head
  parameter receives gradient, not spot-check named inputs); and a `LOG_ZERO` (finite -3.4e38) score overflowing the
  token LayerNorm. The current-tree implementation is `src/fragnnet/model/fragment_competition.py`
  (`frag_competition*` keys in `config/template.yml`).
- **Whole-set readout (`frag_pool_combine`):** production already runs `subtract` (template default), so every node
  already gets `h_n - max_pool(h)`. The `concat` mode is built (6 tests, `fragnnet_model.py`) with config
  `..._context_embedding_bottleneck_pool_concat.yml`, never trained. If concat-max ties subtract, whole-set readout
  is exhausted and the next lever is m/z-local competition (fragments in the same predicted m/z bin, ~1.3/peak).
