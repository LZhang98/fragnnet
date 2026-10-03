# Fragment and formula annotations for MSGYM retrieval: 400-query pilot

> **Status:** FINDING · **Updated:** 2026-10-02 · Pilot, not the plan's Gate 1. On formula
> val, `ll04i00m`'s fragment-cost scores add to linear cosine but add nothing once the query is
> scored by its likelihood under the full predicted spectrum. That likelihood beats cosine by
> +14.8 hit@1 points; most of that gain is small-peak weighting.
> GLACIER's released MSG checkpoint is level with `ll04i00m` on every spectrum-level score
> (cosine hit@1 0.465 against 0.477), and the likelihood lifts it by the same amount.

## Question

Can a candidate's predicted fragment annotations rank MassSpecGym retrieval candidates better
than cosine? The annotations are the per-peak ion formula plus the fragment's atom set over
the parent molecule. The design and gates are in the
[annotation-aware retrieval plan](../../plans/annotation_aware_retrieval_plan.md). This pilot
runs that plan's Step 1 scores and its Step 2 fragment scores together on a small sample, to
see whether fragment-level scoring is worth building at full scale.

## Setup

- **Checkpoint.** `ll04i00m` at epoch 60 (MASSter, MSGYM benchmark 1.5).
  - Its hparams name the retired `FT` label, so it was copied with `FT` renamed to `ORB`. MSGYM
    has one Orbitrap label, and the sorted embedding rows (FT, QTOF) → (ORB, QTOF) are
    unchanged. The weights are not modified.
  - **Re-check under current code:** MSGYM test mean `cos_sim_0.01` is **0.571272**, against
    0.571277 recorded ([GLACIER comparison](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md)).
    87 of 9,954 spectra moved by more than 1e-4 (max 0.0145).
- **Queries.** 400 molecules sampled with seed 20261002 from the 2,331 molecules of the
  `ms_gym_benchmark_1_5` val split (all `[M+H]+` with a CE), one spectrum each. That gives 375
  ORB and 25 QTOF queries, with median nearest-train Tanimoto 0.36.
- **Candidates.** MSGYM 1.5 formula val candidates: 72,946 spectra, median 235 per query
  (cap 256).
  - Each candidate row is a copy of its query's benchmark `spec_df` row with only the molecule
    swapped. CE, instrument and dual-CE extras are therefore exactly what the benchmark fed the
    model, and the in-model cosine is computed against the query.
  - The truth candidate's in-model cosine reproduces the benchmark val export (median
    |Δ| 1.2e-7 over 379 queries).
- **Export.** `--min_peak_int -1` with annotation: a mean 952 predicted peaks per candidate, 5.1 GB.
- **Scores.** All parameters were fixed in `score_pilot.py`'s docstring before the first read.
  - Tolerance is the checkpoint's instrument rule: ORB 10 ppm with a 200 m/z floor, QTOF 20 ppm
    with a 750 m/z floor.
  - (a) `cos`: in-model `cos_sim_0.01`.
  - (b) `supp`: the L2 share of non-precursor query peaks that the candidate's predicted support
    reaches.
  - (c) `ll`: Σ y·log max(p̂, 1e-5), with p̂ the predicted intensity within tolerance.
  - (F1) `cost`: Σ_k w_k · max over the candidate's fragments explaining peak k of
    2^−(cuts + |ΔH|). `cuts` counts parent bonds with one end in the atom set; ΔH = ion H −
    (atom-set H + 1). This is a MetFrag-style explanation cost on the model's own fragments.
  - (F3) `pars`: the same, but peaks are taken in decreasing intensity and each is charged only
    for cut bonds no earlier peak used (shared cleavage sites).
  - `_c` variants subtract the mean score of the query shifted by ±5, ±7 and ±11 Da.
  - Blends with cosine choose α on even-hash molecules (224) and report on odd-hash
    molecules (176).
  - Ties count as expected hits under random tie-breaking. Intervals are 2,000-resample query
    bootstraps; with one query per molecule, this is a molecule bootstrap.

On the formula track every candidate has the query's formula, so each query peak has the same
subformula candidates for all of them. Formula-level evidence therefore reduces to support
(b): the formula string adds nothing that the m/z at tolerance does not already carry.

## Results

### Single scores (400 queries; random hit@1 = 0.025)

| score | hit@1 [95% CI] | hit@5 | hit@20 | Δhit@1 vs `cos` |
| --- | --- | ---: | ---: | --- |
| (a) `cos` | 0.478 [0.428, 0.528] | 0.760 | 0.888 | — |
| `cos_hun` | 0.480 | 0.758 | 0.880 | +0.003 [−0.015, +0.018] |
| (b) `supp` | 0.239 | 0.540 | 0.808 | −0.239 |
| (b) `supp_c` | 0.187 | 0.446 | 0.713 | −0.291 |
| **(c) `ll`, ε 1e-5** | **0.625 [0.578, 0.668]** | **0.908** | **0.983** | **+0.148 [+0.100, +0.195]** |
| (c) `ll`, ε 1e-4 / 1e-6 | 0.618 / 0.630 | 0.888 / 0.908 | 0.978 / 0.983 | +0.140 / +0.153 |
| (F1) `cost` / `cost_c` | 0.342 / 0.340 | 0.705 / 0.688 | 0.932 / 0.915 | −0.136 / −0.138 |
| (F3) `pars` / `pars_c` | 0.330 / 0.314 | 0.703 / 0.692 | 0.935 / 0.940 | −0.147 / −0.164 |

### Blends, reported on the 176 odd molecules

| blend | α (even) | hit@1 | Δhit@1 [95% CI] | Δhit@5 | Δhit@20 |
| --- | ---: | ---: | --- | --- | --- |
| `cos` + `supp_c` | 0.9 | 0.432 | +0.000 [−0.023, +0.023] | +0.000 | +0.011 |
| `cos` + `cost_c` | 0.7 | 0.466 | +0.034 [−0.006, +0.080] | +0.074 [+0.028, +0.125] | +0.063 [+0.028, +0.102] |
| `cos` + `cost` | 0.7 | 0.483 | +0.051 [+0.006, +0.097] | +0.063 [+0.017, +0.108] | +0.063 [+0.023, +0.102] |
| `cos` + `pars_c` | 0.8 | 0.466 | +0.034 [−0.006, +0.074] | +0.051 [+0.011, +0.097] | +0.046 [+0.011, +0.085] |
| `cos` + `ll` | 0.2 | 0.585 | +0.153 [+0.091, +0.216] | +0.125 [+0.074, +0.176] | +0.080 [+0.046, +0.125] |
| `ll` + `cost_c` *(post hoc)* | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.006 | +0.000 |
| `ll` + `pars_c` *(post hoc)* | 0.9 | 0.608 | −0.011 [−0.034, +0.011] | +0.000 | +0.000 |
| `ll` + `supp_c` *(post hoc)* | 0.9 | 0.625 | +0.006 [−0.011, +0.028] | −0.006 | +0.000 |

The odd-half cosine baseline is 0.432 / 0.767 / 0.898. The `ll` base is 0.619 / 0.909 / 0.983.

### What the likelihood gain is (post hoc, 400 queries)

| score | hit@1 | hit@5 | hit@20 | Δhit@1 vs `ll` |
| --- | ---: | ---: | ---: | --- |
| `cos_sqrt` (0.01 Da bins, sqrt intensities) | 0.588 | 0.848 | 0.935 | −0.038 [−0.078, +0.000] |
| `ent` (unweighted spectral entropy similarity) | 0.573 | 0.845 | 0.930 | −0.053 [−0.090, −0.015] |
| `ll` without the precursor term | 0.628 | 0.895 | 0.981 | +0.003 [−0.020, +0.023] |

The precursor carries a median 0.4% of query intensity.

### Controls

- **Spectrum-blind consensus pseudo-query.** This is the top 30 bins of the mean predicted
  spectrum of the query's own candidates. Scored against it, every score stays between 0.018
  and 0.033 hit@1, close to random: `cos` 0.033, `ll` 0.028, `cost_c` 0.018, `pars_c` 0.031.
- **Same-formula partner query.** Only 49 queries have a partner (a val query with the same
  formula and another molecule). Under it the truth gets `cos` 0.102, `ll` 0.061 and
  `cost_c` 0.041 hit@1.
- **Support size and bond count** score 0.019–0.028 hit@1 in either direction.
- **Candidate list position is degenerate.** MassSpecGym's candidate JSON stores the truth at
  index 0 (3,000 of 3,000 sampled query SMILES), and the build keeps that order, so
  "first wins" gives hit@1 = 1.000. This build cannot measure the PubChem-order bias
  ([MassSpecGym audit](../../notes/massspecgym_in_the_wild_audit.md)); that needs PubChem CIDs.
  Any scorer that breaks ties by position leaks the answer.

### Subgroups (hit@1)

| group | n | `cos` | `ll` | `cost_c` | `pars_c` |
| --- | ---: | ---: | ---: | ---: | ---: |
| ORB | 375 | 0.493 | 0.635 | 0.348 | 0.324 |
| QTOF | 25 | 0.240 | 0.480 | 0.213 | 0.153 |
| nearest-train Tanimoto < 0.3 | 64 | 0.297 | 0.438 | 0.264 | 0.212 |
| 0.3–0.4 | 203 | 0.537 | 0.675 | 0.372 | 0.354 |
| ≥ 0.4 | 133 | 0.474 | 0.639 | 0.327 | 0.302 |

The likelihood gain over cosine is about +0.14 to +0.17 in every novelty bin. It is not
concentrated on novel molecules, as the plan hypothesized.

Mean within-query Spearman: `cos`–`ll` 0.75, `cost_c`–`pars_c` 0.92, `supp_c`–`cost_c` 0.73,
`cos`–`cost_c` 0.34.

### Query peaks no subformula explains (added 2026-10-02, post hoc)

Each query peak was put in one class: the precursor; feasible, meaning within tolerance of a
precursor-bounded, RDBE ≥ 0 subformula ion (production `_subformula_masses`, `[M+H]+`
budget); an isotopologue, meaning 1 or 2 × 1.003355 above a feasible peak or the precursor;
or unexplained, meaning everything else. Script: `unexplained_peaks.py`, log:
`analyze_unexplained.log`.

| class | mean share of query intensity | median | 95th pct | peaks |
| --- | ---: | ---: | ---: | ---: |
| precursor | 0.133 | 0.004 | 0.760 | 229 |
| feasible | 0.848 | 0.979 | 1.000 | 10,399 |
| isotopologue | 0.001 | 0.000 | 0.006 | 140 |
| unexplained | 0.017 | 0.005 | 0.056 | 1,550 |

- **How much is unexplained.** 5.2% of queries have more than 5% unexplained intensity, 1.5%
  more than 20%. QTOF's mean is higher than ORB's (0.044 against 0.016).
- **On the formula track, unexplained and isotopologue peaks do not affect the ranking.** No
  candidate puts any predicted intensity on them, so each adds the same `y·log ε` to every
  candidate of its query. Dropping them from `ll` changes hit@1 in 0 of 400 queries. Cosine is
  unaffected for the same reason: no candidate has intensity there, and the query norm is
  shared.
- **Feasibility is a weak filter for dirt.** A donor formula (another pilot query's formula in
  the same precursor-mass decile) explains 0.765 of the non-precursor intensity, against 0.969
  for the true formula. Most contaminant peaks would therefore be classed as feasible, and
  that class carries all of `ll`'s ranking.
- **Where dirt can hurt `ll`.** 8.6% of feasible intensity is on peaks the truth's prediction
  misses (p̂ < 1e-5) but some decoy reaches. On each of these, the truth takes the full ε
  penalty. In the 150 `ll` misses, the feasible class carries most of the truth-versus-top-decoy
  deficit in 85% of cases (mean −0.456 nats; precursor −0.024). Whether those peaks are
  contaminants or model misses is not separated here.
- **Capping the per-peak penalty hurts.** Raising ε lowers hit@1 from 0.625 (1e-5) to 0.618
  (1e-4), 0.595 (1e-3) and 0.525 (1e-2). In this data the full penalty helps the ranking more
  than any protection a cap gives against dirt.

### Mass track, same 400 queries (added 2026-10-02)

The same 400 queries were scored against their MSGYM 1.5 **mass** val candidates: 100,001
candidates, median 254 per query. Only 17.7% have the query's formula; the median query has 41
distinct formulas.
- 893 candidates (0.9%, in 270 queries) cannot be featurized (Na, K, metals, one valence error)
  and were not predicted. No truth is among them. "opt" ranks them last, which is itself a valid
  ranking rule because MassSpecGym leaves the score free. "pes" ranks every one of them above
  the truth, a vacuous bound.
- Scripts: `build_mass_pilot.py`, `filter_mass_pilot.py`, `mass/run_mass_pilot.sh`,
  `mass_unexplained.py`, `analyze_mass_pilot.py`; log `analyze_mass_pilot.log`.

| score | hit@1 | hit@5 | hit@20 | top-1 has true formula | Δhit@1 vs `cos` |
| --- | ---: | ---: | ---: | ---: | --- |
| `cos` | 0.623 | 0.800 | 0.885 | 0.782 | — |
| `cos_hun` | 0.640 | 0.812 | 0.885 | 0.800 | +0.018 [+0.000, +0.035] |
| `cos_sqrt` | 0.755 | 0.877 | 0.930 | 0.875 | +0.133 [+0.092, +0.172] |
| `ent` | 0.743 | 0.870 | 0.917 | 0.860 | +0.120 [+0.083, +0.160] |
| **`ll`, ε 1e-5** | **0.828** | **0.950** | **0.983** | **0.950** | **+0.205 [+0.160, +0.250]** |
| `ll`, ε 1e-3 / 1e-2 | 0.785 / 0.680 | 0.912 / 0.848 | 0.958 / 0.917 | 0.900 / 0.807 | +0.163 / +0.058 |
| `cfeas` (formula explainability alone) | 0.008 | 0.049 | 0.178 | 0.565 | −0.614 |
| consensus control `cos` / `ll` | 0.005 / 0.003 | | | | |

- **The mass track is easier than the formula track on these queries.** On the formula track,
  `cos` scores 0.477 and `ll` 0.625. Most mass decoys have the wrong formula, and the
  prediction rejects them.
- **`ll` picks the right formula 95% of the time.** 49 of its 69 misses lose to a same-formula
  isomer.
- **Unexplained peaks still do not decide the ranking.** Here "dirt" means a query peak no
  subformula of the TRUE formula explains: 1.7% of query intensity, as on the formula track.
  - In 101 of 400 queries some wrong-formula candidate does predict intensity on a dirt peak,
    gaining at most 0.52 nats.
  - Removing dirt peaks from `ll` (an oracle diagnostic, which needs the true formula) changes
    no hit@1, @5 or @20.
  - In the 20 misses to an other-formula decoy, the whole deficit sits on peaks the true
    formula can explain (−1.11 nats), none on dirt. The decoy formulas' subformula grids are
    only slightly denser (median ratio 1.14).
- **Formula feasibility alone carries no ranking signal.** Wrong formulas explain 0.835 of the
  non-precursor intensity, against 0.963 for the true one. It cannot separate isomers, and a
  blend with `ll` chose weight 0 on even molecules.
- **Capping the per-peak penalty hurts here too** (ε 1e-3: −0.043, 1e-2: −0.148 hit@1).

### GLACIER's released MSG checkpoint on the same 400 queries (added 2026-10-02)

The same 72,946 candidate spectra (each under its query's CE, instrument and precursor) were run
through ms-pred's `predict_smis_joint.py` with the checkpoint, ms-pred checkout and
`--sparse-k 256` of the [GLACIER comparison](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md).
Its output is squared to linear intensity. Every candidate got a prediction.

- `cos` uses that comparison's exact CUDA scorer.
- The other scores use the definitions above, applied to GLACIER's top-256 peaks.
- GLACIER exposes no fragment annotations in this output, so only spectrum-level scores apply.

| score | MASSter hit@1 / @5 / @20 | GLACIER hit@1 / @5 / @20 | GLACIER − MASSter hit@1 [95% CI] |
| --- | --- | --- | --- |
| `cos` | 0.477 / 0.760 / 0.887 | 0.465 / 0.748 / 0.875 | −0.013 [−0.068, +0.037] |
| `cos_sqrt` | 0.588 / 0.848 / 0.935 | 0.562 / 0.858 / 0.948 | −0.025 [−0.080, +0.028] |
| `ent` | 0.573 / 0.845 / 0.930 | 0.565 / 0.845 / 0.940 | −0.007 [−0.060, +0.045] |
| `ll` (ε 1e-5) | 0.625 / 0.907 / 0.983 | 0.615 / 0.887 / 0.958 | −0.010 [−0.060, +0.037] |
| `ll` without precursor | 0.627 / 0.895 / 0.980 | 0.600 / 0.877 / 0.958 | −0.027 [−0.077, +0.023] |

- **The two models are statistically level on every score.** The truth's mean `cos` is 0.615 for
  GLACIER and 0.634 for MASSter, the same direction as the test-set forward gap.
- **The likelihood gain is not model-specific.** It lifts GLACIER from 0.465 to 0.615 hit@1, as
  it lifts MASSter from 0.477 to 0.625.
- **GLACIER's `ll` is computed on its top 256 peaks** against MASSter's mean 952, so more of
  GLACIER's query peaks fall to the ε floor. The `ll` comparison partly measures that truncation.
- QTOF (n = 25): GLACIER `cos` hit@1 0.240, the same as MASSter.
- The first GLACIER pass was OOM-killed on its last 1,026 (largest) molecules: 8 workers each
  held a featurization cache of large molecules, pushing the machine into swap. Those 1,026
  were rerun with 3 workers and no cache in 24 s.

## Interpretation

1. **Fragment identity carries signal that linear cosine discards, and the full predicted
   distribution already holds it.** The explanation cost improves cosine at hit@5 and hit@20
   (+5 to +7 points, intervals above zero) but changes nothing on top of `ll`. A query peak that
   only a cheap fragment explains is one the model already gives non-negligible intensity, and
   cosine ignores it because it is small.
2. **Structural parsimony adds nothing beyond per-peak cost** (ρ 0.92, identical blend gains).
   The literature sweep found no ranking test of joint explanation coherence; this pilot is a
   negative one at this scale.
3. **Support alone is uninformative on the formula track.** MASSter's selected support averages
   952 peaks per candidate, so nearly every isomer reaches most query peaks.
   Chance-correction lowers it further. Formula-level matching adds nothing here (see Setup).
4. **The ranking lever is the similarity's intensity weighting.** Linear cosine is dominated by
   the few largest peaks, which are where the model's novel-molecule errors concentrate.
   Sqrt-cosine recovers +11.0 of the +14.8 points, entropy +9.5. The likelihood stays ahead at
   hit@5 and hit@20.

## Pipeline findings

- **Train/serve molecule skew.** Training molecules go through `02_prepare_proc.py` with
  tautomer canonicalization on by default ([02_prepare_proc.py:104](../../../preproc_scripts/02_prepare_proc.py#L104)).
  MSGYM retrieval candidates are built with `ml_standardize=False`
  ([01_prepare_msgym_retrieval.py:50](../../../preproc_scripts/msgym_ms2c/01_prepare_msgym_retrieval.py#L50)).
  - For 95 of 2,331 val molecules (660 of 9,731 queries), the benchmark's standardized molecule
    has a different 2D InChIKey from MSGYM's source molecule (anthracyclines as hydroquinone
    tautomers, for example).
  - Truth must be identified by the source SMILES' key, not the benchmark `mol_df` key.
  - In the 21 such pilot queries the truth's own prediction moved (median |Δcos| 0.034 against
    the benchmark).
  - **Standardizing the candidates does not matter for ranking** (tested 2026-10-02, below).
- **Candidate standardization test.** All 72,575 pilot candidate molecules were standardized
  exactly as `02_prepare_proc.py` does: `mol_from_smiles` → `mol_to_smiles` → re-parse. In current
  code that is RDKit Normalize, Reionize, a P–O normalization and tautomer canonicalization.
  Parent-fragment extraction and neutralization are commented out
  ([data_utils.py:194](../../../src/fragnnet/utils/data_utils.py#L194)), although the docstring
  still lists them.
  - **What changed.** 11.2% of candidate SMILES (4.6% of 2D keys; no formula changed), including
    sulfoxides to `[S+][O-]` and amide/imine tautomers. 40 of 400 truths changed.
  - **Ties.** In 10 queries a decoy standardizes to the truth's own molecule and then ties
    with it.
  - **Ranking.** `ll04i00m` was rerun on the standardized set with the same rows and conditions:

    | score | raw hit@1 / @5 | standardized hit@1 / @5 | Δhit@1 [95% CI] |
    | --- | --- | --- | --- |
    | `cos` | 0.477 / 0.760 | 0.482 / 0.752 | +0.005 [−0.007, +0.020] |
    | `ll` | 0.625 / 0.907 | 0.623 / 0.905 | −0.003 [−0.013, +0.007] |
    | `cos_sqrt` | 0.588 / 0.848 | 0.583 / 0.848 | −0.005 [−0.018, +0.007] |
    | `ent` | 0.573 / 0.845 | 0.568 / 0.840 | −0.005 [−0.015, +0.007] |

  - **Changed truths.** On the 40 queries whose truth changed, `ll` hit@1 is 0.625 either way;
    the truth's mean `cos` moves from 0.548 to 0.552.
  - **The only systematic effect is the forced ties,** which halve the truth's expected credit in
    those 10 queries (`ll` hit@1 0.20 → 0.10 there).
  - **What does need care:** identifying the truth by MSGYM's source key, which the scorer now
    does.
- **The retrieval `spec_df` files still say `FT`.** The current loader refuses them. The pilot
  relabelled its own subset; the files in `data/ms2c/msgym/v1_5/*.zip` are unchanged.

## Limits

- One checkpoint, which is not the best MSGYM recipe; formula track only; val only. 400 queries
  overall and 176 for blends. Only 25 QTOF and 49 partner queries.
- No minimum useful gain was fixed, so this is not the plan's Gate 1. The `ll`-based blends,
  `cos_sqrt`, `ent` and the precursor split were added after the first read.
- The fragment scores use the model-selected support, not a structural enumeration. The cost
  model `2^−(cuts+|ΔH|)` is untuned and has no bond energies.
- Val hit@1 is not comparable to published MSGYM test numbers. The pilot molecules must be
  excluded from any Gate 1 read.

## Artifacts

All under `output/annotation_retrieval_pilot_2026_10_02/`:

- `patch_ft_to_orb.py` and `ckpt/ll04i00m_ep060_orb.ckpt`; `test_recheck/` and its log.
- `build_pilot_inputs.py`, `data/pilot/` (spec/mol/split, `pilot_queries.csv`,
  `pilot_candidates.csv`) and `query_novelty.py`.
- `pred/ll04i00m_ep060_orb/predictions_test.hdf5` (5.1 GB, annotated, `--min_peak_int -1`) and
  `pred.log`.
- `score_pilot.py` (the fixed protocol), `analyze_pilot.py`, `analyze_pilot.log`,
  `scores/part_*.pkl` and `scores/summary_*.csv`.
- `score_extra.py`, `analyze_extra.py`, `analyze_extra.log` and `scores/extra_*.pkl`.
- `tautomer/`: `build_std_inputs.py`, `mol_df_std.pkl`, `std_changes.csv`, `pred_std/` (491 MB),
  `score_std.py`, `compare_std.py`, `compare_std.csv` and `.log`.
- `glacier/`: `build_labels.py`, `labels_pilot*.tsv`, `preds_k256*/preds.hdf5` and logs,
  `score_glacier_pilot.py`, `scores_k256.pkl`, `compare.py`, `compare_k256.csv` and `.log`.

## Retrieval versus forward similarity (2026-10-03, interim full-test read)

> Superseded by the complete 20-shard read: [full formula-track finding](masster_msgym_formula_retrieval_full_2026_10_03.md).

The full-test run now supersedes the 400-query pilot for the available matched population.
Read `output/msgym_formula_retrieval_2026_10_02/interim_compare.py` against the completed
GLACIER score/sqrt shards: **19 shards, 9,500 queries, 2,279 molecules, 1,586,527 candidates**;
8,025 ORB and 1,475 QTOF queries. This is an interim result, not the full 9,954-query comparison.
GLACIER uses batch 64 / sparse-k 100 here, unlike the pilot and original simulation audit;
see the [frozen protocol](../../plans/annotation_aware_retrieval_plan.md#full-formula-track-run-frozen-protocol-2026-10-02-before-any-test-fold-read).

| Model / score | Overall hit@1 | ORB hit@1 | QTOF hit@1 |
| --- | ---: | ---: | ---: |
| MASSter linear cosine | 0.362 | 0.402 | 0.143 |
| GLACIER linear cosine | 0.339 | 0.376 | 0.137 |
| MASSter sqrt cosine (eval CUDA bins) | 0.459 | 0.509 | 0.186 |
| GLACIER sqrt cosine (eval CUDA bins) | 0.429 | 0.476 | 0.171 |
| MASSter likelihood | 0.489 | 0.542 | 0.199 |

GLACIER minus MASSter paired overall hit@1: linear −0.023 [−0.037, −0.009], sqrt
−0.030 [−0.045, −0.013], 95% molecule-bootstrap intervals. MASSter's advantage in forward
similarity therefore also yields an overall retrieval advantage in this matched interim read.
There is no established instrument-specific difference from these overall intervals.

To inspect the QTOF deficit, group candidate score rows by query, take the maximum truth score
(the test can have two equivalent truths) and maximum nontruth score, then average over queries:

| Instrument / model | Mean true-candidate linear cosine | Mean strongest-decoy cosine | Mean truth minus decoy |
| --- | ---: | ---: | ---: |
| ORB / MASSter | 0.59306 | 0.58385 | +0.00950 |
| ORB / GLACIER | 0.55877 | 0.57060 | −0.01115 |
| QTOF / MASSter | 0.47408 | 0.58364 | −0.10956 |
| QTOF / GLACIER | 0.45744 | 0.58214 | −0.12470 |

Retrieval requires separating the truth from its strongest competitor, not merely a good
absolute prediction. QTOF has lower true-candidate similarity without correspondingly lower
strongest-decoy similarity. Both models therefore lose many more rankings there. Linear
cosine gives large peaks disproportionate influence; smaller diagnostic peaks matter for
same-formula isomers, consistent with the observed sqrt/likelihood gains. Broad predicted
support can match wrong isomers too, but these score means do not establish the causal
contribution of support size, CE, source composition or candidate chemistry. A query-level
strongest-decoy peak comparison is the next causal check.

Verification: reran the existing interim comparison with the configured FRAGNNET-GPU interpreter;
checked margins directly from its loaded MASSter/GLACIER candidate tables. No inference or
training launched. The simulation-only instrument audit is linked in the
[forward comparison](masster_msgym_vs_glacier_released_ckpt_2026_09_24.md#instrument-prediction-audit-2026-10-03).
