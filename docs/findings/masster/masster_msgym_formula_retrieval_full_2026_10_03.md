# MSGYM 1.5 formula-track retrieval: full val and test, MASSter vs GLACIER

> **Status:** FINDING · **Updated:** 2026-10-03 · Complete. Simulation-challenge `[M+H]+` queries
> on the formula track; not leaderboard-comparable. On test, scoring each candidate by the query's
> log-likelihood under MASSter's full prediction gives hit@1 0.485 against 0.358 for the
> MassSpecGym cosine. GLACIER's released checkpoint is below MASSter on cosine (−0.023 hit@1 on
> `cos_sim_0.01`, −0.027 on `cos_sim_sqrt_0.01`) but level on ms-pred's entropy similarity
> (+0.012 [−0.005, +0.029]). Under ms-pred's own entropy scoring GLACIER reaches 49.65% hit@1,
> reproducing its paper's 48.49% (w/o CF), so the paper's retrieval numbers are entropy-ranked.

## Setup

The protocol was fixed before the test fold was read; see the
[retrieval plan](../../plans/annotation_aware_retrieval_plan.md#full-formula-track-run-frozen-protocol-2026-10-02-before-any-test-fold-read).
It extends the [400-query pilot](masster_annotation_retrieval_pilot_2026_10_02.md).

- **Queries.** All formula-track queries both checkpoints can serve: `[M+H]+` with a CE.
  - Val: 9,731 queries (the `ms_gym_benchmark_1_5` val split), 1,609,708 candidates.
  - Test: 9,954 queries (exactly the simulation-challenge test set), 1,657,238 candidates.
  - The full retrieval challenge also has `[M+Na]+` and no-CE queries (val 19,429, test 17,556).
- **Candidates.** MSGYM's formula-bounded candidates as given (unstandardized), each predicted
  under its query's CE, instrument and precursor.
  - One val decoy (a perchlorate salt with an impossible valence) is excluded from 3 queries;
    see `val/shard_09/excluded_candidates.csv`.
- **Truth.** Every candidate whose 2D key matches MSGYM's source SMILES: one per query on val;
  on test one for 9,935 queries and two (listed stereoisomers) for 19. A query is a hit if any
  truth is in the top k. Ties are broken at random, as an expected value.
- **MASSter.** `ll04i00m` at epoch 60 (`FT`→`ORB` hparams copy), `--min_peak_int -1`.
  - The full run reproduces the pilot's scores on their 26 shared queries; the truth's rank is
    identical in every query.
- **GLACIER.** The released MSG checkpoint, run on test only, at upstream's retrieval settings
  (ms-pred `run_scripts/glacier/03_run_retrieval.py` at `318bb7b`: `--batch-size 64 --sparse-k
  100`). Its output is squared to linear intensity and scored only with `cos_sim_0.01` and
  `cos_sim_sqrt_0.01` (user decisions).
- **Scores.** All are computed against the query's benchmark peaks.
  - `cos_sim_0.01`: in-model for MASSter; GLACIER uses the eval's CUDA binning.
  - `cos_sim_sqrt_0.01`: the eval's CUDA binning for both models.
  - `cos_hun_inst_tol` and its sqrt (test only): the in-model code path run offline. It
    reproduces in-model values to 1e-6 (sqrt: 8e-5).
  - Query log-likelihood: Σ y·log max(p̂, 1e-5), with p̂ the predicted intensity within the
    instrument tolerance. See the pilot.
  - Spectrum-blind consensus controls.
- **Intervals.** 95% molecule bootstrap, 2,000 resamples, paired.

## Results

### Test (9,954 queries)

| model / score | hit@1 | hit@5 | hit@20 | hit@1 ORB (8,374) | hit@1 QTOF (1,580) |
| --- | ---: | ---: | ---: | ---: | ---: |
| MASSter `cos_sim_0.01` | 0.358 | 0.711 | 0.894 | 0.398 | 0.150 |
| GLACIER `cos_sim_0.01` | 0.335 | 0.688 | 0.878 | 0.371 | 0.147 |
| MASSter `cos_sim_sqrt_0.01` | 0.455 | 0.796 | 0.939 | 0.505 | 0.192 |
| GLACIER `cos_sim_sqrt_0.01` | 0.428 | 0.770 | 0.919 | 0.474 | 0.184 |
| MASSter `cos_hun_inst_tol` | 0.364 | 0.716 | 0.894 | 0.406 | 0.144 |
| MASSter `cos_hun_inst_tol_sqrt` | 0.463 | 0.804 | 0.943 | 0.512 | 0.204 |
| MASSter entropy similarity | 0.444 | 0.788 | 0.935 | 0.493 | 0.188 |
| **MASSter query log-likelihood** | **0.485** | **0.815** | **0.948** | **0.538** | **0.208** |
| spectrum-blind consensus (cosine / ll) | 0.046 / 0.042 | 0.17 | 0.31 / 0.32 | — | — |

Paired hit@1 differences:

| comparison | Δhit@1 [95% CI] |
| --- | --- |
| GLACIER − MASSter, `cos_sim_0.01` | −0.023 [−0.037, −0.009] |
| GLACIER − MASSter, `cos_sim_sqrt_0.01` | −0.027 [−0.043, −0.012] |
| MASSter log-likelihood − GLACIER `cos_sim_sqrt_0.01` | +0.057 [+0.039, +0.075] |
| MASSter log-likelihood − MASSter `cos_sim_0.01` | +0.127 [+0.113, +0.142] |
| MASSter `cos_hun_inst_tol_sqrt` − MASSter `cos_sim_0.01` | +0.104 [+0.094, +0.116] |
| MASSter `cos_hun_inst_tol` − MASSter `cos_sim_0.01` | +0.006 [+0.002, +0.010] |

### Val (9,731 queries, MASSter only)

| score | hit@1 | hit@5 | hit@20 | Δhit@1 vs cosine [95% CI] |
| --- | ---: | ---: | ---: | --- |
| `cos_sim_0.01` | 0.383 | 0.718 | 0.894 | — |
| `cos_hun_inst_tol` | 0.391 | 0.721 | 0.895 | +0.008 [+0.003, +0.012] |
| numpy sqrt cosine (0.01 Da) | 0.479 | 0.804 | 0.936 | +0.095 [+0.083, +0.107] |
| entropy similarity | 0.466 | 0.790 | 0.929 | +0.083 [+0.072, +0.094] |
| query log-likelihood | 0.511 | 0.823 | 0.953 | +0.127 [+0.113, +0.142] |

Without the 400 pilot molecules (8,095 queries), the likelihood gain is +0.130 [+0.114, +0.147].

## Interpretation

1. **The scoring rule matters more than the model.** The log-likelihood gain over cosine is
   +12.7 hit@1 points on both val and test. MASSter's whole margin over GLACIER is 2–3 points.
2. **Most of the gain is intensity weighting.** Sqrt-cosine at 0.01 Da recovers about 10 of the
   12.7 points. Instrument-tolerance matching adds under 1 point, and its sqrt version adds about
   0.8 over 0.01 Da sqrt.
3. **MASSter is ahead of GLACIER under cosine, not under entropy.** The cosine margin is mostly
   on ORB; on QTOF the two are level within noise. The 2026-09-24 forward-similarity gap (0.5713
   vs 0.5455) carries into cosine retrieval. Under ms-pred's entropy similarity, GLACIER is level
   with MASSter's entropy and log-likelihood (see below). GLACIER gains more from the change of
   similarity than MASSter does.
4. **QTOF is harder, not bad data.** See the diagnostic below.

## Matching GLACIER's reported retrieval numbers (added 2026-10-03)

**The paper does not say which similarity ranks retrieval candidates.**
- §4.1 lists "top-k retrieval accuracy, cosine similarity, and Jensen-Shannon similarity
  (interchangeable with spectral entropy similarity)" without naming the ranking function.
- Everything around it points to entropy:
  - its contrastive fine-tuning optimizes "the entropy distance between the predicted spectrum
    … and the ground-truth spectrum" (Appendix E);
  - Table 1's "99.9% confidence intervals upon bootstrapping, 20,000 resamples" is ms-pred's
    `retrieval/bootstrap_metrics.py`, whose defaults are `--dist-fn entropy` and a
    `rerank_eval_entropy.yaml` input;
  - its retrieval script `run_scripts/glacier/03_run_retrieval.py` computes both entropy and
    cosine.

**ms-pred's similarity, from `retrieval_benchmark.py` at `318bb7b`:**
- **Target:** peaks above `parentmass + 1` are dropped, intensities max-normalized and
  square-rooted, then binned with **max** pooling.
- **Prediction:** GLACIER's stored sqrt intensities, binned with **sum** pooling. MASSter's
  linear intensities were square-rooted per peak to put them in the same space.
- **Bins:** `floor(mz · 149999/1500) + 1`, width 0.0100007 Da.
- **Similarity:** cosine (`_cosine_sparse`) or unweighted entropy similarity (`_entropy_sparse`)
  on those values. Ranking is a stable argsort, so a tie goes to the earlier-listed candidate.

All test candidates were rescored that way (`score_mspred_style.py`,
`compare_mspred_style.py`). Hit@1 in %:

| model / score | random ties | stable ties (ms-pred) | ours: 95% molecule bootstrap | ms-pred: 99.9% BCa, i.i.d. | ORB / QTOF |
| --- | ---: | ---: | --- | --- | --- |
| GLACIER, our `cos_sim_0.01` | 33.53 | 33.69 | [31.2, 36.0] | [32.2, 35.3] | 37.1 / 14.7 |
| GLACIER, our `cos_sim_sqrt_0.01` | 42.80 | 42.96 | [40.0, 45.6] | [41.4, 44.6] | 47.4 / 18.4 |
| GLACIER, ms-pred cosine | 40.89 | 41.13 | [38.2, 43.7] | [39.6, 42.7] | 45.6 / 16.0 |
| **GLACIER, ms-pred entropy** | **49.65** | **49.67** | [46.5, 52.8] | [48.1, 51.4] | 54.8 / 22.1 |
| MASSter, our `cos_sim_0.01` | 35.84 | 35.92 | [33.3, 38.5] | [34.3, 37.5] | 39.8 / 15.0 |
| MASSter, our `cos_sim_sqrt_0.01` | 45.50 | 45.58 | [42.4, 48.7] | [44.0, 47.2] | 50.5 / 19.2 |
| MASSter, ms-pred cosine | 45.55 | 45.66 | [42.6, 48.8] | [44.0, 47.3] | 50.5 / 19.6 |
| **MASSter, ms-pred entropy** | **48.50** | 48.51 | [45.5, 51.7] | [46.9, 50.3] | 53.5 / 22.3 |
| MASSter, query log-likelihood | 48.53 | 48.57 | [45.3, 51.8] | [47.0, 50.3] | 53.8 / 20.8 |

Paired GLACIER − MASSter hit@1 (95% molecule bootstrap):

| similarity | Δhit@1 [95% CI] |
| --- | --- |
| ms-pred cosine | −4.66 [−6.42, −3.11] |
| ms-pred entropy | +1.15 [−0.54, +2.86] |

- **The paper's GLACIER number reproduces under entropy only.** It reports formula-challenge
  Recall@1 48.49 (46.88–50.15) without contrastive fine-tuning, which is what the released
  checkpoint appears to be, and 49.88 with it. ms-pred's entropy gives 49.65 on our candidates
  and ms-pred's cosine 40.89. Our lower 33.5 is linear `cos_sim_0.01`, a different metric.
  Whether their test candidate table and query set exactly match ours is not verified.
- **Their query set is probably the same simulation-challenge set.** Their 99.9% i.i.d. interval
  is about ±1.6 points around 48.5%, which implies roughly 10,000 queries. That matches 9,954
  simulation-challenge queries, not the 17,556 full formula-challenge queries (about ±1.2). This
  is an inference from interval width.
- **The model comparison depends on the similarity.** Under cosine MASSter is ahead; under
  entropy the two are level. GLACIER's entropy (49.65) and MASSter's log-likelihood (48.53) are
  within about a point. Entropy is worth +16.1 points to GLACIER and +12.7 to MASSter.
- **ms-pred's cosine differs from our sqrt cosine.** It takes the square root per peak, then sums
  (prediction) or takes the max (target) within bins, and drops peaks above the precursor + 1.
  Ours sums linear intensities, then takes the square root. That costs GLACIER 1.9 points and
  leaves MASSter unchanged.
- **Their bootstrap is optimistic.**
  - `bootstrap_metrics.py` resamples individual queries as i.i.d. Bernoulli draws, rebuilt from
    the rounded hit rate.
  - Test has about 4 spectra per molecule, and those are correlated. So their 99.9% intervals
    (about ±1.6 points) are narrower than our 95% molecule-clustered ones (about ±2.6–3.1).
  - Separate per-model intervals cannot establish a model difference; that needs a paired
    interval like the ones above.
- **Tie rule.** ms-pred's stable list-order tie-break favours the first-listed candidate, which
  is the truth in MassSpecGym's ordering. Here it adds at most +0.24 points, because ties
  involving the truth are rare (7–161 of 9,954 queries depending on the score).

## QTOF diagnostic (val, MASSter)

| | ORB | QTOF |
| --- | ---: | ---: |
| intensity explained by precursor subformulas at instrument tolerance (chance control 0%) | 99.7% | 100% |
| median mass error of explained peaks | 1.2 ppm | 3.8 ppm |
| replicate agreement, same molecule/instrument/CE (inst-tol cosine) | 0.958 | 0.913 |
| train-molecule fit, inst-tol cosine / `cos_sim_0.01` | 0.940 / 0.916 | 0.921 / 0.819 |
| new-molecule (val) truth fit, inst-tol cosine | 0.601 | 0.546 |
| median candidates per query | 230 | 88 |
| mean truth cosine / mean best-decoy cosine | 0.587 / 0.578 | 0.477 / 0.580 |

- **The data is clean.** QTOF has slightly lower replicate agreement and a few-ppm mass error.
  That error costs about 0.1 binned cosine even on training molecules, but at instrument
  tolerance the fit is close to ORB's.
- **The deficit is generalization to new molecules.** QTOF's true-candidate similarity drops
  while its strongest decoy stays as good. Instrument-tolerance matching does not help QTOF
  retrieval (hit@1 0.150 → 0.144 on test).
- **Untested cause:** vendor-specific eV collision energies and mixed sources. MSGYM has no source
  column.
- **Artifacts:** `val/qtof_diagnostic.csv`, `replicate_agreement.py`, `train_fit/`.

## GLACIER batch dependence

GLACIER's predictions depend on batch composition. On 512 test candidates, batch 16 against
batch 64 (same flags otherwise) gives this:

- 81% of spectra are identical to float noise.
- 19% change, by a median 2% relative intensity shift among the same peaks (worst cosine 0.951).
- Peaks present in only one run carry at most 0.6% of intensity.
- Batch 16 rerun against itself is identical.

The likely mechanism is batch-size-dependent padding in its batched transformer, amplified by
its discrete fragment selection (argmax cut count, top-k cut atoms in `breakpoint_inference`).
This is not traced step by step.

On test shard 0 (500 queries), batch 16 / sparse-k 256 against batch 64 / sparse-k 100 changes
GLACIER `cos_sim_0.01` hit@1 by −0.004 [−0.013, +0.001]. The batch-16 shards 0–4 are kept in
`test/shard_0{0..4}/glacier_bs16_k256/` and are not used.

## Pipeline notes

- **`/tmp` filled up during the run**, from old Claude session folders. They were cleared with
  the user's approval. The run writes only under `output/`.
- **Another session paused and resumed the runner** at 23:31–23:37 for a mass-track pilot. Shard
  markers made this harmless.
- **Markers make the run resumable.** The runner, the inst-tol loops and the sqrt loop are
  idempotent per shard.

## Limits

- Simulation-challenge `[M+H]+` queries only. No `[M+Na]+` or no-CE policy, so these are not
  MassSpecGym leaderboard numbers.
- One MASSter checkpoint (`ll04i00m`), not the best MSGYM recipe; one GLACIER checkpoint, test
  only.
- GLACIER is scored with cosine, sqrt cosine and, post hoc, ms-pred's cosine and entropy. There
  is no GLACIER likelihood; its 100-peak output would understate one anyway. The ms-pred-style
  scores were added after the test read, to replicate the paper's protocol, not to select a
  score.
- The val log-likelihood was chosen on the pilot, so test is the clean read.

## Artifacts

All under `output/msgym_formula_retrieval_2026_10_02/`:

- **Inputs:** `build_shards.py`, `census.py`, `<fold>/queries.csv`, and per shard
  `spec_df.pkl`, `mol_df.pkl`, `candidates.csv` and `glacier_labels.tsv`.
- **Runners:** `runner.sh`, `runner.log`, `insttol_runner.sh`, `insttol_sqrt_runner.sh`,
  `sqrt01_runner.sh`, `skip_glacier_val.sh` and `restart_glacier_bs64.sh`.
- **Predictions:** `<fold>/shard_NN/masster/.../predictions_test.hdf5` (MASSter, about 21 GB in
  total) and `test/shard_NN/glacier/preds.hdf5`.
- **Scores:**
  - `score_shard.py` → `scores_<model>_<part>.pkl`;
  - `score_insttol.py` → `insttol[_sqrt]_masster_<part>.pkl`;
  - `score_sqrt01.py` → `sqrt01_<model>.pkl`.
- **Summaries:** `aggregate.py` → `<fold>/summary.csv`; `interim_compare.py`; `check_vs_pilot.py`.
- **ms-pred-style rescoring:** `score_mspred_style.py` → `test/shard_NN/mspred_style.pkl`;
  `compare_mspred_style.py`.
- **GLACIER results, consolidated:** `glacier_test_results/`.
  - `candidate_scores.pkl.gz`: every candidate, both models' scores.
  - `query_hits.csv`: per-query hit@k.
  - `summary.csv`, `prediction_files.csv` and `RUN_INFO.txt`.
- **GLACIER code:** `output/glacier_msg_eval_2026_09_24/ms-pred_318bb7b/`, exact upstream `318bb7b`
  source plus its GLACIER configs and run scripts (`PROVENANCE.txt`). `runner.sh` now uses it
  instead of the `/tmp` copy.
