# MSGYM 1.5 formula-track retrieval: full val and test, MASSter vs GLACIER

> **Status:** FINDING · **Updated:** 2026-10-03 · Complete. Simulation-challenge `[M+H]+` queries
> on the formula track; not leaderboard-comparable. On test, scoring each candidate by the query's
> log-likelihood under MASSter's full prediction gives hit@1 0.485 against 0.358 for the
> MassSpecGym cosine. GLACIER's released checkpoint is below MASSter on cosine (−0.023 hit@1 on
> `cos_sim_0.01`, −0.027 on `cos_sim_sqrt_0.01`) but level on ms-pred's binned entropy similarity
> (+0.012 [−0.005, +0.029]). GLACIER is ahead on entropy at instrument tolerance (+0.023
> [+0.006, +0.041]). The released checkpoint is GLACIER *with* contrastive fine-tuning: it was
> trained by the CF script (EarlyStopping patience 3). Ranked by ms-pred's entropy over ms-pred's
> own candidate table, which keeps decoy stereoisomers as separate rows, it gives 49.67 / 78.20 /
> 90.80 hit@1/5/20. The paper's w/ CF row is 49.88 / 78.13 / 90.75, so the paper ranks by entropy. GLACIER's
> entropy edge sits in predicted peaks below 1% of base: filtering those reverses the comparison.
> At instrument tolerance, linear JSS ranks worst for both models and favours MASSter (−0.032).
> Li-weighted entropy puts the two level (+0.000).

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

- **The released checkpoint is GLACIER *with* contrastive fine-tuning (CF), and under ms-pred's
  protocol it reproduces the paper's w/ CF row** (corrected 2026-10-03; earlier text here said w/o CF).

  | GLACIER formula challenge, hit@1 / @5 / @20 (%) | @1 | @5 | @20 |
  | --- | ---: | ---: | ---: |
  | paper, w/o CF | 48.49 | 76.56 | 90.60 |
  | paper, w/ CF | 49.88 | 78.13 | 90.75 |
  | released ckpt, ms-pred entropy, our candidate list | 49.67 | 82.65 | 94.22 |
  | released ckpt, ms-pred entropy, **ms-pred's full candidate list** | **49.67** | **78.20** | **90.80** |
  | MASSter, ms-pred entropy, ms-pred's full candidate list | 48.51 | 76.92 | 91.96 |
  | MASSter, query log-likelihood, ms-pred's full candidate list | 48.57 | 76.71 | 91.32 |

  Stable ties throughout. ms-pred's cosine gives GLACIER 41.13 on either list. Our lower 33.5 is
  linear `cos_sim_0.01`, a different metric.
  - **The candidate sets differ only in stereo duplicates.**
    - ms-pred's `--full-labels` table (`cands_df_test_formula_256.tsv`, from the README's Dropbox
      folder) has the same 2D-key set as ours for 99.65% of the 9,954 queries.
    - It has 187 rows per query against our 166. The extra rows are stereoisomers of decoys,
      each listed separately.
    - `retrieval_benchmark.py` ranks every row. Each duplicate gets its 2D key's score and takes
      its own rank slot.
    - That cannot change hit@1 but pushes the truth down at @5 and @20 (−4.5 and −3.4 points for
      GLACIER). Ranking over ms-pred's list with duplicates removed gives our numbers.
    - The deduplicated `..._no_stereo_clean_tauts.tsv` re-keys tautomers; 4.7% of our truths are
      absent from it under our keys. It is not what `03_run_retrieval.py` uses.
  - **The checkpoint came from the CF script.**
    - Its EarlyStopping state has patience 3. `train_contr_joint.py` hardcodes `patience=3`,
      `train_joint.py` hardcodes `patience=5`.
    - Its stage-1 hyperparameters (lr 4e-4, `contr_threshold` 0.5) prove nothing.
      `train_contr_joint.py` restores the stage-1 model with `load_from_checkpoint` and sets the
      CF values only as attributes, so a CF checkpoint stores the stage-1 hparams.
    - It ran 3,098 batches per epoch, i.e. batch 32, not the public CF config's 16. That is
      consistent with the run directory `joint_train_msg/split_rnd1/version_5` being an internal
      run.
  - **Consequence: every GLACIER comparison in this document is against the CF model.** On
    ms-pred's full protocol MASSter trails it by 1.2 hit@1 points (48.51 vs 49.67; paired
    interval from the molecule bootstrap above: +1.15 [−0.54, +2.86]). It trails by 1.3 at @5
    and leads by 1.2 at @20.
  - Artifacts: `mspred_cands/compare_cands.{py,log}`, `mspred_cands/rerank_on_mspred_full.{py,log}`.
- **Instrument-tolerance entropy on ms-pred's full candidate table** (stereo duplicates kept,
  stable ties). The scores are the ones above, re-ranked; hit@1 / @5 / @20 in %:

  | similarity | GLACIER (CF) | MASSter | GLACIER − MASSter @1 [95% mol. bootstrap] | @5 | @20 |
  | --- | --- | --- | --- | --- | --- |
  | binned 0.01 Da sqrt entropy (ms-pred) | 49.67 / 78.20 / 90.80 | 48.51 / 76.92 / 91.96 | +1.16 [−0.54, +2.87] | +1.28 [−0.50, +3.16] | −1.17 [−2.86, +0.49] |
  | **inst-tol sqrt entropy** | **51.40** / **79.04** / 91.91 | 49.07 / 77.41 / **92.06** | **+2.33 [+0.61, +4.15]** | +1.64 [−0.14, +3.58] | −0.15 [−1.55, +1.24] |
  | inst-tol Li-weighted entropy | 48.88 / 77.36 / 91.47 | 48.82 / 76.66 / 91.72 | +0.06 [−1.56, +1.62] | +0.69 [−0.94, +2.38] | −0.25 [−1.51, +1.06] |
  | inst-tol JSS (linear) | 41.99 / 72.25 / 88.45 | 45.11 / 74.30 / 90.57 | −3.11 [−4.73, −1.54] | −2.05 [−3.64, −0.37] | −2.12 [−3.60, −0.67] |
  | MASSter query log-likelihood | — | 48.57 / 76.71 / 91.32 | GLACIER inst-tol sqrt − this: +2.82 [+1.11, +4.65] | +2.33 [+0.42, +4.38] | +0.59 [−0.92, +2.01] |

  - **Instrument tolerance adds 1.7 points of hit@1 to GLACIER** (49.67 → 51.40) and 0.6 to
    MASSter. On this, its most favourable similarity, the CF model leads at @1, is borderline at
    @5 and is level at @20.
  - **The ordering of similarities matches the earlier analysis on our own candidate list.**
    GLACIER leads only under sqrt, the two are level under Li weighting, and MASSter leads under
    JSS. Stereo duplicates lower @5/@20 for both models alike.
  - **Stable ties are biased toward the truth here.** ms-pred lists the truth first in 100% of
    queries, so a tie always goes to it.
    - Unfiltered scores: the truth is tied for top in ≤ 8 queries, and stable and random-tie
      hit@1 differ by ≤ 0.08 points.
    - With the prediction 1% filter, the truth is tied for top in 186–196 queries for GLACIER and
      54–64 for MASSter. The stable rule then inflates GLACIER's hit@1 by about 1.6 points, so
      filtered rows are quoted with random ties only.
    - Random-tie hit@1 with the filter, GLACIER vs MASSter: sqrt 46.74 vs 49.44; Li 45.80 vs
      49.30; JSS 40.51 vs 44.32.
  - Artifacts: `mspred_cands/rerank_insttol_on_mspred_full.{py,log}`, `mspred_cands/tie_check.{py,log}`.
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

### Entropy variants: binning, instrument tolerance, 1% filter (added 2026-10-03)

- **"ppm"** is ms-pred's entropy formula with one-to-one peak matching at instrument tolerance
  (ORB 10 ppm / 200, QTOF 20 ppm / 750) instead of 0.01 Da bins. Peaks above precursor m/z +
  tolerance are removed from both spectra.
- **"1% filter"** drops predicted peaks below 1% of the base peak (linear intensity) before
  scoring.

Test, 9,954 queries:

| entropy variant | GLACIER hit@1 / @5 | MASSter hit@1 / @5 | GLACIER − MASSter hit@1 [95% mol. bootstrap] |
| --- | --- | --- | --- |
| binned 0.01 Da (ms-pred) | 49.65 / 82.62 | 48.50 / 81.83 | +1.15 [−0.54, +2.86] |
| binned, 1% filter | 45.04 / 78.77 | 48.64 / 82.00 | −3.60 [−5.10, −2.20] |
| ppm (instrument tolerance) | **51.32** / 83.49 | 49.03 / 82.11 | **+2.29 [+0.56, +4.11]** |
| ppm, 1% filter | 46.79 / 79.57 | **49.44** / 82.77 | −2.65 [−4.02, −1.33] |

MASSter's query log-likelihood is 48.53 / 81.50 for reference.

- **Instrument-tolerance matching helps both models:** GLACIER +1.67 [+1.01, +2.37], MASSter
  +0.53 [+0.03, +1.02]. GLACIER gains more, mostly on ORB.
- **The 1% filter separates them:** it costs GLACIER 4.53 points [3.50, 5.64] and does not change
  MASSter (+0.41 [−0.52, +1.42]).
  - Square-rooting inflates GLACIER's sub-1% predicted peaks, and entropy similarity rewards
    them; they carry real signal for GLACIER.
  - MASSter's sub-1% tail carries none.
  - MASSter trains a linear-intensity Gaussian-density cross-entropy, which down-weights small
    peaks; GLACIER trains a cosine on sqrt intensities.
- **Which number to quote.** On GLACIER's own metric (binned, unfiltered) the models are level. On
  the instrument-tolerance version GLACIER leads by about 2 points. A filtered metric would favour
  MASSter but is not a standard protocol.

### Intensity weighting at instrument tolerance: sqrt vs linear (JSS) vs Li-weighted

These use the same ppm matching, aggregation and above-precursor cut as the ppm row above. Only
the intensity transform before sum-normalization changes:
- **sqrt:** ms-pred's choice.
- **linear:** entropy similarity on linear intensities is exactly JSS, `1 − JSD/ln 2`.
- **Li-weighted:** Li et al. 2021. If a spectrum's natural-log entropy S is below 3, raise its
  intensities to `0.25 + 0.25 S`. Query and prediction are weighted separately.

Test, 9,954 queries, hit@1 / @5 in %, with ORB / QTOF hit@1 after `|`:

| weighting | GLACIER | MASSter | GLACIER − MASSter hit@1 [95% mol. bootstrap] |
| --- | --- | --- | --- |
| sqrt (ms-pred) | **51.32** / 83.49 \| 56.6 / 23.6 | 49.03 / 82.11 \| 53.7 / 24.2 | +2.29 [+0.56, +4.11] |
| sqrt, 1% filter | 46.79 / 79.57 | **49.44** / 82.77 | −2.65 [−4.02, −1.33] |
| linear (JSS) | 41.92 / 76.81 \| 46.4 / 18.2 | 45.07 / 79.29 \| 49.9 / 19.5 | −3.15 [−4.77, −1.57] |
| linear, 1% filter | 40.55 / 75.14 | 44.33 / 78.64 | −3.77 [−5.24, −2.31] |
| Li-weighted | 48.81 / 82.12 \| 54.3 / 19.8 | 48.79 / 81.69 \| 53.9 / 21.8 | +0.02 [−1.58, +1.58] |
| Li-weighted, 1% filter | 45.85 / 79.69 | 49.30 / 83.03 | −3.45 [−4.93, −1.99] |

Paired changes in hit@1, 95% molecule bootstrap:

| comparison | GLACIER | MASSter |
| --- | --- | --- |
| JSS − sqrt | −9.40 [−10.71, −8.16] | −3.96 [−5.05, −2.83] |
| Li − sqrt | −2.51 [−3.52, −1.60] | −0.24 [−1.45, +1.09] |
| Li − JSS | +6.89 [+5.78, +7.99] | +3.72 [+2.68, +4.78] |
| MASSter log-likelihood − this score | — | JSS +3.47 [+2.47, +4.49]; Li −0.25 [−1.31, +0.80]; sqrt −0.49 [−1.85, +0.88] |

- **Flattening the intensities is what helps.** Linear JSS is the worst score for both models.
  Any flattening (sqrt, or Li's exponent of 0.25–1) recovers 4–9 points. For MASSter, sqrt,
  Li-weighted and log-likelihood are tied within noise, at about 48.5–49.4.
- **GLACIER depends on flattening about twice as much as MASSter** (JSS − sqrt: −9.4 vs −4.0).
  This fits the 1% filter result: GLACIER's ranking signal is in small peaks that only count once
  intensities are flattened.
- **The model ranking depends on the weighting.** MASSter leads under JSS (+3.2) and under any
  1% filter. The two are level under Li weighting (+0.02). GLACIER leads only under ms-pred's
  unfiltered sqrt, at instrument tolerance.
- **Li weighting is gentler than sqrt for low-entropy spectra and does nothing for spectra with
  S ≥ 3.** It costs GLACIER 2.5 points but not MASSter. The cost is concentrated on QTOF (GLACIER
  23.6 → 19.8).
- **Artifacts:** `score_weighted_jss.py` (it reproduces `ppm_entropy.pkl` exactly in sqrt mode,
  checked on 2,463 candidates), `test/shard_*/weighted_jss.pkl`,
  `compare_insttol_weightings.{py,log}`.

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
- **ms-pred-style rescoring (exploratory):** `score_mspred_style.py`, `score_mspred_style_f1.py`,
  `score_ppm_entropy.py` → `test/shard_NN/{mspred_style,mspred_style_f1,ppm_entropy}.pkl`;
  `compare_mspred_style.py`, `compare_entropy_variants.py`.
- **In the codebase (2026-10-03):**
  - `src/fragnnet/utils/retrieval_utils.py`: ms-pred binned cosine/entropy replicas,
    instrument-tolerance entropy, query log-likelihood, tie-aware hit@k, molecule and ms-pred
    bootstraps.
  - `scripts/eval/score_retrieval_candidates.py`: per-candidate scores from a MASSter or ms-pred
    export.
  - `scripts/eval/summarize_retrieval.py`: hit@k, both interval styles, paired comparisons.
  - `tests/test_retrieval_utils.py`.
  - On test shard 0 the CLI reproduces the exploratory scripts' per-candidate values exactly
    (max difference 0, 65,406 candidates, both models).
- **GLACIER results, consolidated:** `glacier_test_results/`.
  - `candidate_scores.pkl.gz`: every candidate, both models' scores.
  - `query_hits.csv`: per-query hit@k.
  - `summary.csv`, `prediction_files.csv` and `RUN_INFO.txt`.
- **GLACIER code:** `output/glacier_msg_eval_2026_09_24/ms-pred_318bb7b/`, exact upstream `318bb7b`
  source plus its GLACIER configs and run scripts (`PROVENANCE.txt`). `runner.sh` now uses it
  instead of the `/tmp` copy.

## Calibrated likelihood: val-selected, one test read (added 2026-10-03)

The rule was chosen on val and frozen in the
[plan](../../plans/annotation_aware_retrieval_plan.md#scoring-rule-calibration-frozen-before-the-test-read-2026-10-03)
before test was read. Rule: `ll_g0.75_t2_e1e-4_a0.5`, i.e. Σ_k (√y_k / Σ√y) · log max(p̂_k, 1e-4),
with p̂_k the mass of p^0.75 (renormalized) within 2× the instrument tolerance. The same
`ll04i00m` predictions; no new inference. The frozen `ll` reproduces 0.4853.

| Test (9,954 queries) | hit@1 | hit@5 | hit@20 |
| --- | ---: | ---: | ---: |
| frozen `ll` | 0.4853 | 0.8150 | 0.9477 |
| **calibrated `ll`** | **0.5284** | **0.8530** | **0.9648** |
| Δ [95% CI] | +0.0431 [+0.0347, +0.0521] | +0.0380 [+0.0304, +0.0453] | +0.0171 [+0.0115, +0.0230] |
| Δ ORB (8,374) | +0.0485 [+0.0389, +0.0585] | +0.0396 | +0.0143 |
| Δ QTOF (1,580) | +0.0146 [−0.0064, +0.0361] | +0.0291 | +0.0316 |

The val held-out half predicted +0.0441; test gives +0.0431. Against GLACIER's best
spectrum score here (`cos_sim_sqrt_0.01`, 0.428) the gap is now about +0.10 hit@1, and against
its ms-pred entropy ranking (49.65%), about +3.2 points (not paired). Most of the gain comes from
softening both sides: the query weights (√y) and the predicted intensities (p^0.75), with a
higher floor (1e-4). Reverse-likelihood and explained-mass blends did not replicate on val.
Artifacts: `output/msgym_retrieval_calibration_2026_10_03/` (`val_selection.csv`, `test_read.csv`).

### GLACIER under the same likelihood rules (added 2026-10-03)

GLACIER's saved test predictions (released checkpoint, batch 64, sparse-k 100, squared), scored
with the same rules by `output/msgym_retrieval_calibration_2026_10_03/glacier_calib.py`; all
1,657,238 candidates have predictions. Paired by query, molecule bootstrap. The 1× tolerance row
was read on test at the user's request after the frozen 2× read; it is not a selected rule.

| Test, all 9,954 | top1 | top2 | top3 | top5 | top10 | top20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MASSter, calibrated ll (2× tol, frozen) | 0.528 | 0.686 | 0.767 | 0.853 | 0.928 | 0.965 |
| GLACIER, calibrated ll (2× tol, frozen) | 0.490 | 0.650 | 0.735 | 0.829 | 0.906 | 0.951 |
| Δ [95% CI] | +0.039 [+0.022, +0.055] | +0.035 | +0.032 | +0.024 [+0.009, +0.040] | +0.022 | +0.014 [+0.003, +0.027] |
| MASSter, calibrated ll (1× tol) | 0.531 | 0.691 | 0.771 | 0.854 | 0.928 | 0.965 |
| GLACIER, calibrated ll (1× tol) | 0.491 | 0.655 | 0.740 | 0.832 | 0.908 | 0.952 |
| MASSter, old ll | 0.485 | 0.642 | 0.728 | 0.815 | 0.897 | 0.948 |
| GLACIER, old ll | 0.444 | 0.611 | 0.700 | 0.793 | 0.885 | 0.936 |

The calibration lifts GLACIER by the same +0.045 hit@1 as MASSter, so the MASSter lead stays
about +0.04 at top1 and narrows to +0.014 at top20. ORB: MASSter 0.586 vs GLACIER 0.545 top1
(+0.041 [+0.025, +0.057]). QTOF: 0.222 vs 0.195 (+0.027 [−0.019, +0.074], not resolved). Full table:
`test_masster_vs_glacier.csv`.
